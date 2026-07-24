from __future__ import annotations

import base64
import hashlib
import os
import stat
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.immport_capture import (
    ImmportAuthenticatedArtifactSpec,
    ImmportAuthenticatedCollectionPlan,
)
from vaxreplay.operations.immport_producer import (
    ImmportProducerError,
    ImmportProducerRequest,
    ImmportProducerResponse,
    InheritedFdImmportProducerInvoker,
    IsolatedImmportProducerClient,
    PreparedImmportRequest,
    parse_immport_producer_response,
    produce_immport_response,
    producer_request_bytes,
    read_runtime_credential,
)
from vaxreplay.operations.immport_producer_cli import dispatch
from vaxreplay.operations.schema import AttemptLease, AttemptState
from vaxreplay.runner._process import BoundedProcessResult
from vaxreplay.sources.immport import ImmportTlsPeerBinding

_T0 = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
_SECRET = b'IMMPORT-API-KEY-CANARY-123456789'
_IMPLEMENTATION = 'a' * 64
_ENVIRONMENT = 'b' * 64
_DEFAULT_TLS = object()


def _urls() -> tuple[str, ...]:
    origin = 'https://www.immport.org'
    study = f'{origin}/data/query/api/study/SDY1234?format=json'
    manifest = f'{origin}/data/query/api/study/manifest/SDY1234?fileType=release_file&format=json'
    return (
        f'{origin}/data/query/v3/api-docs',
        study,
        manifest,
        f'{origin}/data/query/api/study/arm/SDY1234?format=json',
        f'{origin}/data/query/api/study/experiment/SDY1234?format=json',
        f'{origin}/data/query/api/study/link/SDY1234?format=json',
        manifest,
        study,
        f'{origin}/data/query/v3/api-docs',
    )


def _plan(*, plan_id: str = 'immport-production-test-v1') -> ImmportAuthenticatedCollectionPlan:
    return ImmportAuthenticatedCollectionPlan(
        plan_id=plan_id,
        source_id='immport:producer-test',
        study_accession='SDY1234',
        panel_deadline_seconds=60,
        artifacts=tuple(
            ImmportAuthenticatedArtifactSpec(
                artifact_id=f'a{ordinal:02d}-{name}',
                requested_url=url,
                authentication=('none' if url.endswith('/v3/api-docs') else 'immport_scoped_api_key_bearer_redacted'),
                max_body_bytes=1024,
                timeout_seconds=10,
            )
            for ordinal, (name, url) in enumerate(
                zip(
                    (
                        'openapi-before',
                        'study-before',
                        'manifest-before',
                        'arm',
                        'experiment',
                        'link',
                        'manifest-after',
                        'study-after',
                        'openapi-after',
                    ),
                    _urls(),
                    strict=True,
                ),
                start=1,
            )
        ),
    )


def _request(*, plan: ImmportAuthenticatedCollectionPlan | None = None) -> ImmportProducerRequest:
    return ImmportProducerRequest(
        plan=plan or _plan(),
        attempt=AttemptLease(
            attempt_id='attempt-' + '1' * 32,
            logical_run_id='run-' + '2' * 64,
            attempt_number=1,
            owner_id='immport-producer-test',
            state=AttemptState.STARTED,
            started_at=_T0,
            lease_expires_at=_T0 + timedelta(minutes=10),
        ),
        collector_implementation_sha256=_IMPLEMENTATION,
        collector_execution_environment_sha256=_ENVIRONMENT,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = _T0

    def __call__(self) -> datetime:
        self.value += timedelta(microseconds=100)
        return self.value


class _Monotonic:
    def __init__(self, *, step: float = 0.001) -> None:
        self.value = 100.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class _Response:
    def __init__(
        self,
        request: PreparedImmportRequest,
        body: bytes,
        *,
        headers: tuple[tuple[str, str], ...] | None = None,
        status_code: int = 200,
        final_url: str | None = None,
        tls_peer: ImmportTlsPeerBinding | None | object = _DEFAULT_TLS,
    ) -> None:
        self._body = body
        self._offset = 0
        self._headers = headers or (
            ('Content-Type', 'application/json;charset=UTF-8'),
            ('Content-Length', str(len(body))),
        )
        self._status_code = status_code
        self._final_url = final_url or request.url
        self._tls_peer = (
            ImmportTlsPeerBinding(
                tls_version='TLSv1.3',
                certificate_der_sha256='c' * 64,
            )
            if tls_peer is _DEFAULT_TLS
            else tls_peer
        )
        self.closed = False

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def final_url(self) -> str:
        return self._final_url

    @property
    def response_headers(self) -> tuple[tuple[str, str], ...]:
        return self._headers

    def read(self, size: int, *, timeout_seconds: float) -> bytes:
        assert size > 0
        assert timeout_seconds > 0
        if self._offset:
            return b''
        self._offset = len(self._body)
        return self._body

    def tls_peer_binding(self) -> ImmportTlsPeerBinding | None:
        assert self._tls_peer is None or isinstance(self._tls_peer, ImmportTlsPeerBinding)
        return self._tls_peer

    def close(self) -> None:
        self.closed = True


class _Transport:
    def __init__(self, mode: str = 'ok') -> None:
        self.mode = mode
        self.requests: list[PreparedImmportRequest] = []
        self.credential_lengths: list[int] = []
        self.responses: list[_Response] = []

    def open(self, request: PreparedImmportRequest, credential: memoryview) -> _Response:
        self.requests.append(request)
        self.credential_lengths.append(len(credential))
        protected_index = sum(item.authorization_applied for item in self.requests)
        if request.authorization_applied and protected_index == 1:
            if self.mode == 'exception-echo':
                raise RuntimeError(f'Authorization: Bearer {_SECRET.decode()}')
            if self.mode in {'body-echo', 'error-body-echo'}:
                body = b'{"credential":"' + _SECRET + b'"}'
            elif self.mode == 'base64-echo':
                body = b'{"credential_b64":"' + base64.b64encode(_SECRET) + b'"}'
            elif self.mode == 'urlsafe-base64-echo':
                body = b'{"credential_b64":"' + base64.urlsafe_b64encode(_SECRET).rstrip(b'=') + b'"}'
            elif self.mode == 'percent-echo':
                body = b'{"credential_url":"' + b''.join(f'%{value:02X}'.encode('ascii') for value in _SECRET) + b'"}'
            elif self.mode == 'hex-echo':
                body = b'{"credential_hex":"' + _SECRET.hex().encode('ascii') + b'"}'
            else:
                body = b'{"ok":true}'
            headers = None
            if self.mode == 'header-echo':
                headers = (
                    ('Content-Type', 'application/json'),
                    ('Content-Length', str(len(body))),
                    ('X-Canary', _SECRET.decode()),
                )
            elif self.mode == 'bad-length':
                headers = (
                    ('Content-Type', 'application/json'),
                    ('Content-Length', str(len(body) + 1)),
                )
            elif self.mode == 'ambiguous-framing':
                headers = (
                    ('Content-Type', 'application/json'),
                    ('Content-Length', str(len(body))),
                    ('Transfer-Encoding', 'chunked'),
                )
            elif self.mode == 'bad-content-type':
                headers = (
                    ('Content-Type', 'text/plain'),
                    ('Content-Length', str(len(body))),
                )
            response = _Response(
                request,
                body,
                headers=headers,
                status_code=(500 if self.mode == 'error-body-echo' else 302 if self.mode == 'redirect' else 200),
                final_url=('https://www.immport.org/redirected' if self.mode == 'changed-url' else None),
                tls_peer=(
                    None
                    if self.mode == 'no-tls'
                    else ImmportTlsPeerBinding(
                        tls_version='TLSv1.3',
                        certificate_der_sha256='c' * 64,
                    )
                ),
            )
        else:
            body = b'{"ok":true}'
            response = _Response(request, body)
        self.responses.append(response)
        return response


def _produce(
    transport: _Transport,
    *,
    request: ImmportProducerRequest | None = None,
    credential: bytearray | None = None,
    public_request_bytes: bytes | None = None,
    monotonic: _Monotonic | None = None,
) -> tuple[ImmportProducerResponse, bytearray]:
    active_request = request or _request()
    active_credential = credential or bytearray(_SECRET)
    response = produce_immport_response(
        active_request,
        active_credential,
        transport=transport,
        clock=_Clock(),
        monotonic=monotonic or _Monotonic(),
        public_request_bytes=public_request_bytes,
    )
    return response, active_credential


def test_one_shot_producer_exact_serial_panel_and_round_trip() -> None:
    transport = _Transport()
    request = _request()
    response, credential = _produce(transport, request=request)

    assert credential == bytearray(len(_SECRET))
    assert tuple(item.url for item in transport.requests) == _urls()
    assert tuple(item.authorization_applied for item in transport.requests) == (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    )
    assert transport.credential_lengths == [0, *([len(_SECRET)] * 7), 0]
    assert all(item.closed for item in transport.responses)

    payload = canonical_json_bytes(response)
    assert _SECRET not in payload
    exchanges = parse_immport_producer_response(payload, request)
    assert tuple(item.artifact_id for item in exchanges) == tuple(item.artifact_id for item in request.plan.artifacts)
    assert all(item.body == b'{"ok":true}' for item in exchanges)
    assert all(_SECRET not in item.receipt for item in exchanges)
    receipts = [item.receipt for item in response.exchanges]
    assert receipts[0].authorization_applied is False
    assert receipts[1].credential_source == 'runtime_secret_broker'
    assert receipts[-1].authorization_applied is False


@pytest.mark.parametrize(
    'mode',
    (
        'exception-echo',
        'body-echo',
        'error-body-echo',
        'header-echo',
        'base64-echo',
        'urlsafe-base64-echo',
        'percent-echo',
        'hex-echo',
    ),
)
def test_live_credential_echoes_fail_with_constant_diagnostic(mode: str) -> None:
    credential = bytearray(_SECRET)
    with pytest.raises(ImmportProducerError) as caught:
        _produce(_Transport(mode), credential=credential)
    assert credential == bytearray(len(_SECRET))
    assert _SECRET.decode() not in str(caught.value)
    assert _SECRET.decode() not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize('mode', ('redirect', 'changed-url'))
def test_redirects_and_changed_urls_fail_closed(mode: str) -> None:
    with pytest.raises(ImmportProducerError, match='authenticated ImmPort producer failed'):
        _produce(_Transport(mode))


@pytest.mark.parametrize(
    'mode',
    ('bad-length', 'ambiguous-framing', 'bad-content-type', 'no-tls'),
)
def test_malformed_response_contracts_fail_closed(mode: str) -> None:
    with pytest.raises(ImmportProducerError, match='authenticated ImmPort producer failed'):
        _produce(_Transport(mode))


def test_request_that_contains_live_credential_fails_before_network() -> None:
    plan = _plan(plan_id=_SECRET.decode())
    request = _request(plan=plan)
    payload = producer_request_bytes(request)
    credential = bytearray(_SECRET)
    transport = _Transport()
    with pytest.raises(ImmportProducerError):
        _produce(
            transport,
            request=request,
            credential=credential,
            public_request_bytes=payload,
        )
    assert transport.requests == []
    assert credential == bytearray(len(_SECRET))


def test_panel_deadline_is_enforced_and_secret_is_zeroized() -> None:
    credential = bytearray(_SECRET)
    with pytest.raises(ImmportProducerError):
        _produce(_Transport(), credential=credential, monotonic=_Monotonic(step=20.0))
    assert credential == bytearray(len(_SECRET))


def test_runtime_secret_fd_is_consumed_closed_and_trimmed() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, _SECRET + b'\n')
    os.close(write_fd)
    credential = read_runtime_credential(read_fd)
    assert credential == _SECRET
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_cli_dispatch_uses_fd_only_and_returns_canonical_output() -> None:
    request = _request()
    now = datetime.now(timezone.utc)
    request = request.model_copy(
        update={
            'attempt': request.attempt.model_copy(
                update={
                    'started_at': now - timedelta(seconds=1),
                    'lease_expires_at': now + timedelta(minutes=10),
                }
            )
        }
    )
    payload = producer_request_bytes(request)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, _SECRET)
    os.close(write_fd)
    transport = _Transport()
    with patch(
        'vaxreplay.operations.immport_producer.DirectImmportHttpsTransport',
        return_value=transport,
    ):
        output = dispatch(payload, credential_fd=read_fd)
    parsed = ImmportProducerResponse.model_validate_json(output)
    assert canonical_json_bytes(parsed) == output
    assert _SECRET not in output


def test_isolated_parent_client_emits_only_public_request_and_validates_response() -> None:
    request_payloads: list[bytes] = []

    def invoke(payload: bytes) -> bytes:
        request_payloads.append(payload)
        request = ImmportProducerRequest.model_validate_json(payload)
        response, _credential = _produce(_Transport(), request=request)
        return canonical_json_bytes(response)

    request = _request()
    client = IsolatedImmportProducerClient(
        collector_implementation_sha256=_IMPLEMENTATION,
        collector_execution_environment_sha256=_ENVIRONMENT,
        invoke=invoke,
    )
    exchanges = client(request.plan, request.attempt)
    assert len(exchanges) == 9
    assert len(request_payloads) == 1
    assert _SECRET not in request_payloads[0]


def test_tampered_isolated_response_identity_is_rejected() -> None:
    request = _request()
    response, _credential = _produce(_Transport(), request=request)
    first = response.exchanges[0]
    tampered_receipt = first.receipt.model_copy(update={'collector_implementation_sha256': 'f' * 64})
    tampered = response.model_copy(
        update={
            'exchanges': (
                first.model_copy(update={'receipt': tampered_receipt}),
                *response.exchanges[1:],
            )
        }
    )
    payload = canonical_json_bytes(tampered)
    with pytest.raises(ImmportProducerError, match='differs from its request'):
        parse_immport_producer_response(payload, request)


def test_body_digest_is_bound_in_wire_schema() -> None:
    response, _credential = _produce(_Transport())
    first = response.exchanges[0]
    wrong_receipt = first.receipt.model_copy(update={'body_sha256': hashlib.sha256(b'wrong').hexdigest()})
    with pytest.raises(ValueError, match='does not bind'):
        first.__class__(
            artifact_id=first.artifact_id,
            body_base64=first.body_base64,
            receipt=wrong_receipt,
        )


def test_inherited_fd_supervisor_passes_only_descriptor_three_and_bounds_process() -> None:
    request = _request()
    payload = producer_request_bytes(request)
    response, _credential = _produce(_Transport(), request=request)
    expected_output = canonical_json_bytes(response)
    process_result = BoundedProcessResult(
        exit_code=0,
        duration_ms=5,
        stdout=expected_output,
        stderr=b'',
        termination='exited',
        stdout_truncated=False,
        stderr_truncated=False,
    )
    invoker = InheritedFdImmportProducerInvoker(
        argv=(
            '/usr/local/bin/python',
            '-I',
            '-m',
            'vaxreplay.operations.immport_producer_cli',
        ),
        hard_deadline_margin_seconds=7,
    )
    with (
        patch(
            'vaxreplay.operations.immport_producer.os.fstat',
            return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
        ),
        patch('vaxreplay.operations.immport_producer.os.close') as close_fd,
        patch(
            'vaxreplay.operations.immport_producer.run_bounded_process',
            return_value=process_result,
        ) as run,
    ):
        assert invoker(payload) == expected_output
    run.assert_called_once()
    assert run.call_args.args == (invoker.argv,)
    assert run.call_args.kwargs == {
        'input_bytes': payload,
        'wall_seconds': request.plan.panel_deadline_seconds + 7,
        'max_stdout_bytes': 96 * 1024 * 1024,
        'max_stderr_bytes': 0,
        'on_abort': run.call_args.kwargs['on_abort'],
        'env': {},
        'pass_fds': (3,),
    }
    assert callable(run.call_args.kwargs['on_abort'])
    close_fd.assert_called_once_with(3)


def test_inherited_fd_supervisor_discards_secret_bearing_child_failure() -> None:
    result = BoundedProcessResult(
        exit_code=1,
        duration_ms=5,
        stdout=b'',
        stderr=b'Authorization: Bearer ' + _SECRET,
        termination='log_limit',
        stdout_truncated=False,
        stderr_truncated=True,
    )
    invoker = InheritedFdImmportProducerInvoker(
        argv=(
            '/usr/local/bin/python',
            '-I',
            '-m',
            'vaxreplay.operations.immport_producer_cli',
        )
    )
    with (
        patch(
            'vaxreplay.operations.immport_producer.os.fstat',
            return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
        ),
        patch('vaxreplay.operations.immport_producer.os.close'),
        patch(
            'vaxreplay.operations.immport_producer.run_bounded_process',
            return_value=result,
        ),
        pytest.raises(ImmportProducerError) as caught,
    ):
        invoker(producer_request_bytes(_request()))
    assert _SECRET.decode() not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_inherited_fd_supervisor_rejects_an_ordinary_secret_file() -> None:
    invoker = InheritedFdImmportProducerInvoker(
        argv=(
            '/usr/local/bin/python',
            '-I',
            '-m',
            'vaxreplay.operations.immport_producer_cli',
        )
    )
    with (
        patch(
            'vaxreplay.operations.immport_producer.os.fstat',
            return_value=SimpleNamespace(st_mode=stat.S_IFREG),
        ),
        patch('vaxreplay.operations.immport_producer.os.close'),
        patch('vaxreplay.operations.immport_producer.run_bounded_process') as run,
        pytest.raises(ImmportProducerError, match='isolated ImmPort producer process failed'),
    ):
        invoker(producer_request_bytes(_request()))
    run.assert_not_called()


@pytest.mark.parametrize(
    'argv',
    (
        ('python', '-I', '-m', 'vaxreplay.operations.immport_producer_cli'),
        ('/usr/bin/python', '-m', '-I', 'vaxreplay.operations.immport_producer_cli'),
        ('/usr/bin/python', '-I', '-m', 'different.module'),
    ),
)
def test_inherited_fd_supervisor_rejects_nonexact_command(argv: tuple[str, ...]) -> None:
    with pytest.raises(ImmportProducerError, match='command policy'):
        InheritedFdImmportProducerInvoker(argv=argv)  # type: ignore[arg-type]
