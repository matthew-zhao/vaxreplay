from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.gateway_auth import (
    GatewayFrameError,
    InMemoryGatewaySecretStore,
    maximum_gateway_frame_bytes,
)
from vaxreplay.agentic.protocol import AgenticRunLimits
from vaxreplay.agentic.provider_adapter import (
    ProviderFailureCode,
    ScriptedProviderAdapter,
    ScriptedProviderTurn,
)
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayError,
    AuthenticatedGatewayPolicy,
    AuthenticatedGatewaySession,
    AuthenticatedProviderGateway,
    GatewayCapabilityGrant,
    GatewayCapabilityRevocationReason,
    GatewayErrorCode,
    GatewayModelRoute,
    GatewayTerminalReason,
    GatewayWireResponse,
    SqliteGatewayLedger,
    build_gateway_request_frame,
    gateway_capability_grant_sha256,
    gateway_session_key_id,
    gateway_session_seal_hmac,
    issue_gateway_capability,
    parse_gateway_response_frame,
    verify_authenticated_gateway_session,
)
from vaxreplay.bundle import canonical_json_bytes

_SHA_A = 'a' * 64
_SHA_B = 'b' * 64
_SHA_C = 'c' * 64
_SHA_D = 'd' * 64
_RUN_ID = '1' * 32
_PEER_CID = 37
_SECRET = b'SECRET_DO_NOT_LEAK_0123456789ABC'
_RECEIPT_KEY = b'gateway-receipt-key-do-not-leak-0001'
_ISSUED_AT = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


@dataclass(frozen=True)
class _GatewayFixture:
    gateway: AuthenticatedProviderGateway
    ledger: SqliteGatewayLedger
    store: InMemoryGatewaySecretStore
    adapter: ScriptedProviderAdapter
    grant: GatewayCapabilityGrant
    route: GatewayModelRoute
    policy: AuthenticatedGatewayPolicy
    ledger_path: Path


def _make_fixture(
    tmp_path: Path,
    *,
    turns: tuple[ScriptedProviderTurn, ...] | None = None,
    limits: AgenticRunLimits | None = None,
    clock: Callable[[], datetime] | None = None,
    maximum_session_wire_bytes: int = 4 * 1024 * 1024,
    maximum_frame_body_bytes: int = 1024 * 1024,
    maximum_provider_call_seconds: int = 30,
) -> _GatewayFixture:
    policy = AuthenticatedGatewayPolicy(
        gateway_id='fixture-gateway',
        gateway_version='1.0.0',
        gateway_executable_sha256=_SHA_A,
        gateway_config_sha256=_SHA_B,
        model_registry_sha256=_SHA_C,
        receipt_key_id=gateway_session_key_id(_RECEIPT_KEY),
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=maximum_session_wire_bytes,
        maximum_provider_call_seconds=maximum_provider_call_seconds,
    )
    route = GatewayModelRoute(
        route_id='fixture-route',
        logical_model_id='fixture-logical-model',
        provider='fixture-provider',
        provider_model_id='fixture-model-v1',
        resolved_model_id='fixture-model-snapshot-2025-01-02',
        accepted_provider_model_ids=('fixture-model-snapshot-2025-01-02', 'fixture-model-v1'),
        adapter_id='scripted',
        adapter_version='1.0.0',
        adapter_executable_sha256=_SHA_A,
        adapter_config_sha256=_SHA_B,
        endpoint_origin='https://api.fixture.invalid',
        endpoint_path='/v1/responses',
        fixed_parameters_sha256=_SHA_D,
        max_context_tokens=128,
        max_output_tokens=32,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )
    adapter = ScriptedProviderAdapter(
        provider=route.provider,
        adapter_id=route.adapter_id,
        adapter_version=route.adapter_version,
        executable_sha256=route.adapter_executable_sha256,
        config_sha256=route.adapter_config_sha256,
        clock=clock or (lambda: _ISSUED_AT + timedelta(seconds=2)),
        turns=turns
        or (
            ScriptedProviderTurn(
                content='Candidate B has the strongest pre-cutoff support.',
                input_tokens=7,
                output_tokens=6,
                reasoning_tokens=3,
                provider_cost_usd=0.012,
            ),
        ),
    )
    actual_limits = limits or AgenticRunLimits(
        max_model_calls=3,
        max_input_tokens=100,
        max_output_tokens=40,
        max_reasoning_tokens=50,
    )
    store = InMemoryGatewaySecretStore()
    assert store.register(_SECRET)
    grant = issue_gateway_capability(
        secret=_SECRET,
        run_id=_RUN_ID,
        attempt_reservation_sha256=_SHA_A,
        execution_policy_sha256=_SHA_B,
        workspace_manifest_sha256=_SHA_C,
        policy=policy,
        route=route,
        issued_at=_ISSUED_AT,
        expires_at=_ISSUED_AT + timedelta(minutes=10),
        expected_peer_cid=_PEER_CID,
        limits=actual_limits,
    )
    ledger_path = tmp_path / 'gateway.sqlite3'
    ledger = SqliteGatewayLedger(ledger_path)
    gateway = AuthenticatedProviderGateway(
        policy=policy,
        ledger=ledger,
        secret_resolver=store,
        adapters=(adapter,),
        receipt_key=_RECEIPT_KEY,
    )
    gateway.register_session(grant=grant, route=route, secret=_SECRET)
    return _GatewayFixture(
        gateway=gateway,
        ledger=ledger,
        store=store,
        adapter=adapter,
        grant=grant,
        route=route,
        policy=policy,
        ledger_path=ledger_path,
    )


def _request(
    index: int = 0,
    *,
    run_id: str = _RUN_ID,
    max_output_tokens: int = 10,
    user_content: str = 'Rank the candidates using only the frozen workspace.',
) -> AgenticModelRequest:
    return AgenticModelRequest(
        run_id=run_id,
        call_index=index,
        messages=(
            AgenticModelMessage(role='system', content='Use only frozen evidence.'),
            AgenticModelMessage(role='user', content=user_content),
        ),
        max_output_tokens=max_output_tokens,
    )


def _send(
    fixture: _GatewayFixture,
    request: AgenticModelRequest,
    *,
    peer_cid: int = _PEER_CID,
    observed_at: datetime = _ISSUED_AT + timedelta(seconds=1),
) -> tuple[bytes, GatewayWireResponse]:
    request_frame = build_gateway_request_frame(fixture.grant, request, secret=_SECRET)
    response_frame = fixture.gateway.handle_frame(request_frame, peer_cid=peer_cid, observed_at=observed_at)
    return response_frame, parse_gateway_response_frame(response_frame, fixture.grant, secret=_SECRET)


def test_authenticated_gateway_success_replay_and_session_seal(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    assert fixture.route.schema_version == 'vaxreplay.gateway-model-route.v0.4'
    assert fixture.policy.schema_version == 'vaxreplay.authenticated-gateway-policy.v0.2'
    request = _request()

    first_frame, first = _send(fixture, request)
    first_wire_bytes = fixture.ledger.load(fixture.grant.capability_id).admitted_wire_bytes
    replay_frame, replay = _send(fixture, request)

    assert first.succeeded
    assert first.response is not None
    assert first.response.resolved_model_id == fixture.route.resolved_model_id
    assert first.response.usage.input_tokens == 7
    assert replay == first
    assert replay_frame == first_frame
    assert fixture.adapter.call_count == 1
    assert fixture.ledger.load(fixture.grant.capability_id).admitted_wire_bytes == 2 * first_wire_bytes
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_result is not None
    assert attempt.provider_result.resolved_model_id == fixture.route.resolved_model_id
    assert attempt.provider_result.provider_reported_model_id == fixture.route.resolved_model_id

    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
    )
    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert artifact.transcript.exchanges[0].request == request
    assert artifact.seal.attempt_count == 1
    assert artifact.schema_version == 'vaxreplay.authenticated-gateway-session.v0.3'
    assert artifact.seal.schema_version == 'vaxreplay.gateway-session-seal.v0.3'
    assert artifact.seal.successful_call_count == 1
    assert artifact.seal.provider_cost_usd == pytest.approx(0.012)
    assert artifact.seal.admitted_wire_bytes == 2 * first_wire_bytes
    assert artifact.seal.exact_replay_count == 1
    assert artifact.attempts[0].admitted_wire_bytes == 2 * first_wire_bytes
    assert artifact.attempts[0].exact_replay_count == 1
    assert not fixture.store.contains(fixture.grant.capability_id)


def test_authenticated_gateway_rejects_mutated_frames_before_dispatch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    frame = bytearray(build_gateway_request_frame(fixture.grant, _request(), secret=_SECRET))
    frame[-1] ^= 1

    with pytest.raises(GatewayFrameError, match='authentication failed'):
        fixture.gateway.handle_frame(bytes(frame), peer_cid=_PEER_CID, observed_at=_ISSUED_AT)

    assert fixture.adapter.call_count == 0
    assert fixture.ledger.attempts(fixture.grant.capability_id) == ()


@pytest.mark.parametrize(
    ('model_request', 'peer_cid', 'observed_at', 'error_code'),
    [
        (_request(run_id='2' * 32), _PEER_CID, _ISSUED_AT, GatewayErrorCode.WRONG_RUN),
        (_request(), _PEER_CID + 1, _ISSUED_AT, GatewayErrorCode.WRONG_PEER),
        (_request(), _PEER_CID, _ISSUED_AT - timedelta(microseconds=1), GatewayErrorCode.EXPIRED),
        (_request(), _PEER_CID, _ISSUED_AT + timedelta(minutes=10), GatewayErrorCode.EXPIRED),
    ],
)
def test_authenticated_gateway_rejects_wrong_run_peer_and_time_window_without_dispatch(
    tmp_path: Path,
    model_request: AgenticModelRequest,
    peer_cid: int,
    observed_at: datetime,
    error_code: GatewayErrorCode,
) -> None:
    fixture = _make_fixture(tmp_path)

    _, response = _send(fixture, model_request, peer_cid=peer_cid, observed_at=observed_at)

    assert not response.succeeded
    assert response.error_code == error_code
    assert response.error_message == 'gateway request rejected'
    assert fixture.adapter.call_count == 0
    attempts = fixture.ledger.attempts(fixture.grant.capability_id)
    assert len(attempts) == 1
    attempt = attempts[0][0]
    assert attempt.error_code == error_code
    assert attempt.provider_dispatched is False
    assert attempt.request_sha256 == hashlib.sha256(canonical_json_bytes(model_request)).hexdigest()
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == error_code
    assert artifact.attempts == (attempt,)
    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )


def test_authenticated_gateway_replay_conflict_fails_session_without_second_dispatch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _send(fixture, _request())

    _, conflict = _send(fixture, _request(user_content='A different request at the same call index.'))

    assert conflict.error_code == GatewayErrorCode.REPLAY_CONFLICT
    assert fixture.adapter.call_count == 1
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == GatewayErrorCode.REPLAY_CONFLICT
    assert artifact.seal.attempt_count == 1


def test_exact_cached_replay_cannot_bypass_terminal_state(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    original = _request()
    _send(fixture, original)
    _send(fixture, _request(user_content='Conflicting request that terminally fails the session.'))

    _, replay = _send(fixture, original)

    assert replay.error_code == GatewayErrorCode.UNAUTHORIZED
    assert fixture.adapter.call_count == 1


def test_exact_cached_replay_enforces_session_wire_budget_without_redispatch(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 2048
    wire_budget = 2 * maximum_gateway_frame_bytes(maximum_frame_body_bytes)
    fixture = _make_fixture(
        tmp_path / 'constrained',
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=wire_budget,
    )
    _, first = _send(fixture, _request())
    accepted_replays = 0
    while True:
        _, replay = _send(fixture, _request())
        if not replay.succeeded:
            break
        accepted_replays += 1

    assert first.succeeded
    assert replay.error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    assert fixture.adapter.call_count == 1
    session = fixture.ledger.load(fixture.grant.capability_id)
    assert session.admitted_wire_bytes <= wire_budget
    assert session.exact_replay_count == accepted_replays
    assert session.terminal_budget_rejection_wire_bytes > 0
    assert session.terminal_error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    assert artifact.seal.attempt_count == 1
    assert artifact.seal.admitted_wire_bytes <= wire_budget
    assert artifact.seal.terminal_budget_rejection_wire_bytes > 0
    assert artifact.seal.terminal_observed_overage_bytes == max(
        0,
        artifact.seal.admitted_wire_bytes + artifact.seal.terminal_budget_rejection_wire_bytes - wire_budget,
    )


def test_gateway_policy_rejects_budget_too_small_for_bounded_terminal_exchange(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 2048
    with pytest.raises(ValueError, match='maximum request and rejection frame'):
        _make_fixture(
            tmp_path,
            maximum_frame_body_bytes=maximum_frame_body_bytes,
            maximum_session_wire_bytes=2 * maximum_gateway_frame_bytes(maximum_frame_body_bytes) - 1,
        )
    with pytest.raises(ValueError, match='terminal error response'):
        _make_fixture(
            tmp_path / 'tiny-frame',
            maximum_frame_body_bytes=1,
            maximum_session_wire_bytes=1024,
        )


def test_near_boundary_reserves_largest_response_before_provider_dispatch(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 2048
    turns = tuple(
        ScriptedProviderTurn(
            content=f'candidate ranking {index}',
            input_tokens=2,
            output_tokens=2,
            reasoning_tokens=0,
        )
        for index in range(3)
    )
    fixture = _make_fixture(
        tmp_path,
        turns=turns,
        limits=AgenticRunLimits(
            max_model_calls=5,
            max_input_tokens=100,
            max_output_tokens=100,
            max_reasoning_tokens=100,
        ),
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=2 * maximum_gateway_frame_bytes(maximum_frame_body_bytes),
    )

    _, first = _send(fixture, _request(index=0))
    _, second = _send(fixture, _request(index=1))
    _, rejected = _send(fixture, _request(index=2))

    assert first.succeeded and second.succeeded
    assert rejected.error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    assert fixture.adapter.call_count == 2
    attempts = fixture.ledger.attempts(fixture.grant.capability_id)
    assert len(attempts) == 3
    assert attempts[-1][0].provider_dispatched is False
    assert attempts[-1][0].error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert artifact.seal.admitted_wire_bytes <= fixture.grant.maximum_session_wire_bytes


def test_oversized_provider_response_becomes_bounded_protocol_failure(tmp_path: Path) -> None:
    maximum_frame_body_bytes = 2048
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='x' * 5000,
                input_tokens=2,
                output_tokens=2,
                reasoning_tokens=0,
            ),
        ),
        maximum_frame_body_bytes=maximum_frame_body_bytes,
        maximum_session_wire_bytes=2 * maximum_gateway_frame_bytes(maximum_frame_body_bytes),
    )

    _, response = _send(fixture, _request())

    assert response.error_code == GatewayErrorCode.PROVIDER_PROTOCOL
    assert fixture.adapter.call_count == 1
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_dispatched is True
    assert attempt.admitted_wire_bytes <= fixture.grant.maximum_session_wire_bytes
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert artifact.seal.admitted_wire_bytes <= fixture.grant.maximum_session_wire_bytes


def test_authenticated_gateway_rejects_skipped_call_index(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    _, response = _send(fixture, _request(index=1))

    assert response.error_code == GatewayErrorCode.OUT_OF_ORDER
    assert fixture.adapter.call_count == 0


def test_authenticated_gateway_preflight_budget_failure_does_not_dispatch(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='never returned',
                input_tokens=7,
                output_tokens=1,
                reasoning_tokens=0,
                estimated_input_tokens=51,
            ),
        ),
        limits=AgenticRunLimits(max_model_calls=1, max_input_tokens=50, max_output_tokens=10),
    )

    _, response = _send(fixture, _request(max_output_tokens=10))

    assert response.error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    assert fixture.adapter.call_count == 0
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert not attempt.provider_dispatched
    assert attempt.provider_result is None


def test_authenticated_gateway_authoritative_usage_overrun_is_audited(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='provider exceeded budget',
                input_tokens=51,
                output_tokens=1,
                reasoning_tokens=0,
                estimated_input_tokens=5,
            ),
        ),
        limits=AgenticRunLimits(max_model_calls=1, max_input_tokens=50, max_output_tokens=10),
    )

    _, response = _send(fixture, _request(max_output_tokens=10))

    assert response.error_code == GatewayErrorCode.BUDGET_EXHAUSTED
    assert fixture.adapter.call_count == 1
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_dispatched
    assert attempt.provider_result is not None
    assert attempt.provider_result.usage.input_tokens == 51


def test_authenticated_gateway_rejects_provider_model_outside_pinned_route(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='response from the wrong model',
                input_tokens=2,
                output_tokens=2,
                reasoning_tokens=0,
                provider_reported_model_id='fixture-model-unpinned',
            ),
        ),
    )

    _, response = _send(fixture, _request())

    assert response.error_code == GatewayErrorCode.MODEL_FORBIDDEN
    assert fixture.adapter.call_count == 1
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_result is not None
    assert attempt.provider_result.resolved_model_id == 'fixture-model-unpinned'
    assert attempt.provider_result.provider_reported_model_id == 'fixture-model-unpinned'


def test_authenticated_gateway_rejects_requested_alias_as_resolved_identity(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='response reports only the requested moving alias',
                input_tokens=2,
                output_tokens=2,
                reasoning_tokens=0,
                provider_reported_model_id='fixture-model-v1',
            ),
        ),
    )

    _, response = _send(fixture, _request())

    assert response.error_code == GatewayErrorCode.MODEL_FORBIDDEN
    assert fixture.adapter.call_count == 1
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_result is not None
    assert attempt.provider_result.resolved_model_id == 'fixture-model-v1'
    assert attempt.provider_result.provider_reported_model_id == 'fixture-model-v1'


def test_gateway_route_requires_resolved_identity_and_external_nondefault_data_control_evidence(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    assert fixture.route.provider_data_control_attested is False
    assert fixture.route.provider_data_control_attestation_sha256 is None
    assert fixture.route.provider_data_control_evidence_verified_by_route_schema is False
    assert fixture.route.provider_data_control_evidence_requires_operator_artifact is True
    assert fixture.route.provider_model_snapshot_attested is False
    missing_resolved = fixture.route.model_dump(mode='json')
    missing_resolved['accepted_provider_model_ids'] = ('fixture-model-v1',)
    with pytest.raises(ValueError, match='pinned resolved model'):
        GatewayModelRoute.model_validate(missing_resolved)
    falsely_attested = fixture.route.model_dump(mode='python')
    falsely_attested['provider_data_control_attested'] = True
    with pytest.raises(ValueError, match='external attestation commitment'):
        GatewayModelRoute.model_validate(falsely_attested)
    unsupported_zdr_claim = fixture.route.model_dump(mode='python')
    unsupported_zdr_claim['provider_data_control'] = 'zero_data_retention'
    with pytest.raises(ValueError, match='external attestation commitment'):
        GatewayModelRoute.model_validate(unsupported_zdr_claim)
    attested_zdr = {
        **unsupported_zdr_claim,
        'provider_data_control_attested': True,
        'provider_data_control_attestation_sha256': _SHA_D,
    }
    committed_claim = GatewayModelRoute.model_validate(attested_zdr)
    assert committed_claim.provider_data_control == 'zero_data_retention'
    assert committed_claim.provider_data_control_attested
    assert committed_claim.provider_data_control_attestation_sha256 == _SHA_D
    assert committed_claim.provider_data_control_evidence_verified_by_route_schema is False
    provider_managed_storage = fixture.route.model_copy(update={'provider_storage_disabled': False})
    assert provider_managed_storage.provider_storage_disabled is False


def test_authenticated_gateway_maps_provider_failure_without_leaking_details(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        turns=(
            ScriptedProviderTurn(
                content='never returned',
                input_tokens=1,
                output_tokens=1,
                failure=ProviderFailureCode.RATE_LIMIT,
            ),
        ),
    )

    response_frame, response = _send(fixture, _request())

    assert response.error_code == GatewayErrorCode.PROVIDER_RATE_LIMIT
    assert response.error_message == 'gateway request rejected'
    assert fixture.adapter.call_count == 1
    assert _SECRET not in response_frame
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_dispatched
    assert attempt.provider_result is None


def test_gateway_secret_is_absent_from_wire_ledger_and_authenticated_artifact(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    request_frame = build_gateway_request_frame(fixture.grant, _request(), secret=_SECRET)
    response_frame = fixture.gateway.handle_frame(
        request_frame,
        peer_cid=_PEER_CID,
        observed_at=_ISSUED_AT + timedelta(seconds=1),
    )
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )

    assert _SECRET not in request_frame
    assert _SECRET not in response_frame
    assert _SECRET not in fixture.ledger_path.read_bytes()
    assert _SECRET not in canonical_json_bytes(artifact)


def test_gateway_session_seal_and_transcript_mutations_fail_verification(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _send(fixture, _request())
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    mutated_seal = artifact.model_copy(
        update={'seal': artifact.seal.model_copy(update={'transcript_sha256': '0' * 64})}
    )
    exchange = artifact.transcript.exchanges[0]
    mutated_response = exchange.response.model_copy(update={'content': 'tampered'})
    mutated_exchange = exchange.model_copy(update={'response': mutated_response})
    mutated_transcript = artifact.transcript.model_copy(update={'exchanges': (mutated_exchange,)})
    mutated_artifact = artifact.model_copy(update={'transcript': mutated_transcript})
    mutated_route = artifact.model_copy(
        update={'route': artifact.route.model_copy(update={'route_id': 'different-route'})}
    )

    with pytest.raises(ValueError, match='authentication failed'):
        verify_authenticated_gateway_session(
            mutated_seal,
            receipt_key=_RECEIPT_KEY,
            expected_receipt_key_id=fixture.policy.receipt_key_id,
        )
    with pytest.raises(ValueError, match='does not bind'):
        verify_authenticated_gateway_session(
            mutated_artifact,
            receipt_key=_RECEIPT_KEY,
            expected_receipt_key_id=fixture.policy.receipt_key_id,
        )
    with pytest.raises(ValueError, match='grant does not bind the authenticated model route'):
        verify_authenticated_gateway_session(
            mutated_route,
            receipt_key=_RECEIPT_KEY,
            expected_receipt_key_id=fixture.policy.receipt_key_id,
        )


def test_gateway_restart_never_redispatches_an_ambiguous_reserved_call(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    request = _request()
    fixture.ledger.reserve(
        fixture.grant.capability_id,
        request,
        request_frame_bytes=len(build_gateway_request_frame(fixture.grant, request, secret=_SECRET)),
    )
    restarted_ledger = SqliteGatewayLedger(fixture.ledger_path)
    restarted_gateway = AuthenticatedProviderGateway(
        policy=fixture.policy,
        ledger=restarted_ledger,
        secret_resolver=fixture.store,
        adapters=(fixture.adapter,),
        receipt_key=_RECEIPT_KEY,
    )
    request_frame = build_gateway_request_frame(fixture.grant, request, secret=_SECRET)

    response_frame = restarted_gateway.handle_frame(
        request_frame,
        peer_cid=_PEER_CID,
        observed_at=_ISSUED_AT + timedelta(seconds=1),
    )
    response = parse_gateway_response_frame(response_frame, fixture.grant, secret=_SECRET)

    assert response.error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert fixture.adapter.call_count == 0
    attempts = restarted_ledger.attempts(fixture.grant.capability_id)
    assert len(attempts) == 1
    assert attempts[0][0].error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert attempts[0][0].provider_dispatched is None
    artifact = restarted_gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert artifact.seal.attempt_count == 1


def test_sealing_materializes_reserved_call_zero_as_authenticated_ambiguous_failure(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    request = _request()
    fixture.ledger.reserve(
        fixture.grant.capability_id,
        request,
        request_frame_bytes=len(build_gateway_request_frame(fixture.grant, request, secret=_SECRET)),
    )

    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )

    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert fixture.adapter.call_count == 0
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert artifact.seal.attempt_count == 1
    assert artifact.attempts[0].call_index == 0
    assert artifact.attempts[0].error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert artifact.attempts[0].provider_dispatched is None
    forged_seal = artifact.seal.model_copy(
        update={
            'terminal_reason': GatewayTerminalReason.COMPLETED,
            'terminal_error_code': None,
        }
    )
    forged_completed = artifact.model_copy(
        update={
            'seal': forged_seal,
            'seal_hmac': gateway_session_seal_hmac(forged_seal, _RECEIPT_KEY),
        }
    )
    with pytest.raises(ValueError, match='completed gateway session contains a failed attempt'):
        verify_authenticated_gateway_session(
            forged_completed,
            receipt_key=_RECEIPT_KEY,
            expected_receipt_key_id=fixture.policy.receipt_key_id,
        )


def test_sealing_preserves_success_then_materializes_reserved_call_one(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _, first = _send(fixture, _request(index=0))
    assert first.succeeded
    second_request = _request(index=1)
    fixture.ledger.reserve(
        fixture.grant.capability_id,
        second_request,
        request_frame_bytes=len(build_gateway_request_frame(fixture.grant, second_request, secret=_SECRET)),
    )

    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )

    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED
    assert artifact.seal.terminal_error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert artifact.seal.attempt_count == 2
    assert artifact.seal.successful_call_count == 1
    assert len(artifact.transcript.exchanges) == 1
    assert artifact.attempts[0].succeeded
    assert artifact.attempts[1].call_index == 1
    assert artifact.attempts[1].error_code == GatewayErrorCode.AMBIGUOUS_IN_FLIGHT
    assert artifact.attempts[1].provider_dispatched is None


@pytest.mark.parametrize(
    ('started_at', 'finished_at'),
    [
        (_ISSUED_AT + timedelta(seconds=2), _ISSUED_AT + timedelta(minutes=10, microseconds=1)),
        (_ISSUED_AT + timedelta(seconds=2), _ISSUED_AT + timedelta(seconds=33)),
    ],
    ids=('finishes-after-capability-expiry', 'exceeds-provider-call-duration'),
)
def test_authenticated_gateway_rejects_provider_result_outside_committed_time_limits(
    tmp_path: Path,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    times = iter((started_at, finished_at))
    fixture = _make_fixture(tmp_path, clock=lambda: next(times))

    _, response = _send(fixture, _request())

    assert response.error_code == GatewayErrorCode.PROVIDER_PROTOCOL
    assert fixture.adapter.call_count == 1
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.provider_result is not None
    assert attempt.provider_result.started_at == started_at
    assert attempt.provider_result.finished_at == finished_at
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=11),
        revoke_secret=False,
    )
    verify_authenticated_gateway_session(
        artifact,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )
    assert artifact.seal.terminal_reason == GatewayTerminalReason.FAILED


def test_gateway_rejects_forged_grant_binding_without_provider_dispatch(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    forged_grant = fixture.grant.model_copy(update={'workspace_manifest_sha256': 'e' * 64})
    assert gateway_capability_grant_sha256(forged_grant) != gateway_capability_grant_sha256(fixture.grant)
    frame = build_gateway_request_frame(forged_grant, _request(), secret=_SECRET)

    response_frame = fixture.gateway.handle_frame(
        frame,
        peer_cid=_PEER_CID,
        observed_at=_ISSUED_AT + timedelta(seconds=1),
    )
    response = parse_gateway_response_frame(response_frame, fixture.grant, secret=_SECRET)

    assert response.error_code == GatewayErrorCode.UNAUTHORIZED
    assert fixture.adapter.call_count == 0
    attempt = fixture.ledger.attempts(fixture.grant.capability_id)[0][0]
    assert attempt.error_code == GatewayErrorCode.UNAUTHORIZED
    assert attempt.provider_dispatched is False


def test_gateway_ledger_requires_private_parent_and_single_link_file(tmp_path: Path) -> None:
    public = tmp_path / 'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='private, owned'):
        SqliteGatewayLedger(public / 'gateway.sqlite3')

    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    ledger = SqliteGatewayLedger(private / 'gateway.sqlite3')
    hardlink = private / 'second-name.sqlite3'
    hardlink.hardlink_to(ledger.path)
    with pytest.raises(ValueError, match='single-link'):
        SqliteGatewayLedger(ledger.path)


def test_gateway_ledger_and_runtime_reject_route_not_bound_by_stored_grant(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    tampered_route = fixture.route.model_copy(update={'route_id': 'different-route'})
    with sqlite3.connect(fixture.ledger_path) as connection:
        connection.execute(
            'UPDATE sessions SET route_json=? WHERE capability_id=?',
            (canonical_json_bytes(tampered_route), fixture.grant.capability_id),
        )

    with pytest.raises(ValueError, match='grant does not bind its stored model route'):
        fixture.ledger.load(fixture.grant.capability_id)
    frame = build_gateway_request_frame(fixture.grant, _request(), secret=_SECRET)
    with pytest.raises(ValueError, match='grant does not bind its stored model route'):
        fixture.gateway.handle_frame(frame, peer_cid=_PEER_CID, observed_at=_ISSUED_AT)
    assert fixture.adapter.call_count == 0


def test_gateway_ledger_rejects_legacy_route_row_instead_of_defaulting_new_claims(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    legacy_route = fixture.route.model_dump(mode='json')
    legacy_route['schema_version'] = 'vaxreplay.gateway-model-route.v0.3'
    legacy_route.pop('provider_data_control_evidence_verified_by_route_schema')
    legacy_route.pop('provider_data_control_evidence_requires_operator_artifact')
    with sqlite3.connect(fixture.ledger_path) as connection:
        connection.execute(
            'UPDATE sessions SET route_json=? WHERE capability_id=?',
            (canonical_json_bytes(legacy_route), fixture.grant.capability_id),
        )

    with pytest.raises(ValueError, match='gateway-model-route.v0.4'):
        fixture.ledger.load(fixture.grant.capability_id)


def test_authenticated_gateway_session_json_round_trip_preserves_verifiability(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _send(fixture, _request())
    artifact = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )

    reparsed = AuthenticatedGatewaySession.model_validate_json(canonical_json_bytes(artifact))

    verify_authenticated_gateway_session(
        reparsed,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=fixture.policy.receipt_key_id,
    )


def test_gateway_session_seal_is_durable_and_exactly_once(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _send(fixture, _request())
    first = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=1),
        revoke_secret=False,
    )

    repeated = fixture.gateway.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=2),
        revoke_secret=False,
    )
    assert canonical_json_bytes(repeated) == canonical_json_bytes(first)
    with pytest.raises(ValueError, match='different terminal reason'):
        fixture.gateway.seal_session(
            fixture.grant.capability_id,
            terminal_reason=GatewayTerminalReason.CANCELLED,
            sealed_at=_ISSUED_AT + timedelta(minutes=2),
            revoke_secret=False,
        )

    restarted = AuthenticatedProviderGateway(
        policy=fixture.policy,
        ledger=SqliteGatewayLedger(fixture.ledger_path),
        secret_resolver=fixture.store,
        adapters=(fixture.adapter,),
        receipt_key=_RECEIPT_KEY,
    )
    recovered = restarted.seal_session(
        fixture.grant.capability_id,
        terminal_reason=GatewayTerminalReason.COMPLETED,
        sealed_at=_ISSUED_AT + timedelta(minutes=3),
        revoke_secret=False,
    )
    assert canonical_json_bytes(recovered) == canonical_json_bytes(first)


def test_pre_registration_tombstone_survives_restart_and_blocks_resurrection(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path / 'fixture')
    path = tmp_path / 'pre-registration' / 'gateway.sqlite3'
    ledger = SqliteGatewayLedger(path)

    first = ledger.revoke_unregistered_capability(
        capability_id=fixture.grant.capability_id,
        expected_run_id=fixture.grant.run_id,
        expected_attempt_reservation_sha256=(fixture.grant.attempt_reservation_sha256),
        expected_model_route_sha256=fixture.grant.model_route_sha256,
        reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
        revoked_at=_ISSUED_AT + timedelta(seconds=1),
    )
    repeated = SqliteGatewayLedger(path).revoke_unregistered_capability(
        capability_id=fixture.grant.capability_id,
        expected_run_id=fixture.grant.run_id,
        expected_attempt_reservation_sha256=(fixture.grant.attempt_reservation_sha256),
        expected_model_route_sha256=fixture.grant.model_route_sha256,
        reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
        revoked_at=_ISSUED_AT + timedelta(seconds=2),
    )

    assert repeated == first
    assert repeated.registered_binding is None
    with pytest.raises(ValueError, match='durable revocation tombstone'):
        SqliteGatewayLedger(path).register(
            fixture.grant,
            fixture.route,
            fixture.policy,
        )


def test_registered_revocation_binds_route_and_denies_restarted_gateway(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    revocation = fixture.ledger.revoke_capability(
        capability_id=fixture.grant.capability_id,
        expected_run_id=fixture.grant.run_id,
        expected_attempt_reservation_sha256=(fixture.grant.attempt_reservation_sha256),
        expected_model_route_sha256=fixture.grant.model_route_sha256,
        reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
        revoked_at=_ISSUED_AT + timedelta(seconds=1),
    )

    assert revocation.registered_binding is not None
    assert revocation.model_route_sha256 == fixture.grant.model_route_sha256
    restarted = AuthenticatedProviderGateway(
        policy=fixture.policy,
        ledger=SqliteGatewayLedger(fixture.ledger_path),
        secret_resolver=fixture.store,
        adapters=(fixture.adapter,),
        receipt_key=_RECEIPT_KEY,
    )
    frame = build_gateway_request_frame(fixture.grant, _request(), secret=_SECRET)
    with pytest.raises(AuthenticatedGatewayError, match='unauthorized'):
        restarted.handle_frame(
            frame,
            peer_cid=_PEER_CID,
            observed_at=_ISSUED_AT + timedelta(seconds=2),
        )
    assert fixture.adapter.call_count == 0


def test_revocation_rejects_a_different_attempt_or_route(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    with pytest.raises(ValueError, match='exact binding'):
        fixture.ledger.revoke_capability(
            capability_id=fixture.grant.capability_id,
            expected_run_id=fixture.grant.run_id,
            expected_attempt_reservation_sha256='f' * 64,
            expected_model_route_sha256=fixture.grant.model_route_sha256,
            reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
            revoked_at=_ISSUED_AT + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match='exact binding'):
        fixture.ledger.revoke_capability(
            capability_id=fixture.grant.capability_id,
            expected_run_id=fixture.grant.run_id,
            expected_attempt_reservation_sha256=(fixture.grant.attempt_reservation_sha256),
            expected_model_route_sha256='f' * 64,
            reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
            revoked_at=_ISSUED_AT + timedelta(seconds=1),
        )
    assert fixture.ledger.capability_revocation(fixture.grant.capability_id) is None


def test_provider_call_and_revocation_linearize_on_the_same_durable_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    revoked = threading.Event()
    failures: list[BaseException] = []
    original_generate = fixture.adapter.generate

    def blocking_generate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        entered.set()
        if not release.wait(2):
            raise RuntimeError('test provider release timed out')
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(fixture.adapter, 'generate', blocking_generate)

    def send() -> None:
        try:
            _send(fixture, _request())
        except BaseException as error:
            failures.append(error)

    def revoke() -> None:
        try:
            fixture.ledger.revoke_capability(
                capability_id=fixture.grant.capability_id,
                expected_run_id=fixture.grant.run_id,
                expected_attempt_reservation_sha256=(fixture.grant.attempt_reservation_sha256),
                expected_model_route_sha256=fixture.grant.model_route_sha256,
                reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
                revoked_at=_ISSUED_AT + timedelta(seconds=3),
            )
            revoked.set()
        except BaseException as error:
            failures.append(error)

    call_thread = threading.Thread(target=send)
    revoke_thread = threading.Thread(target=revoke)
    call_thread.start()
    assert entered.wait(2)
    revoke_thread.start()
    assert not revoked.wait(0.05)
    release.set()
    call_thread.join(2)
    revoke_thread.join(2)

    assert not call_thread.is_alive()
    assert not revoke_thread.is_alive()
    assert failures == []
    assert revoked.is_set()
    assert fixture.adapter.call_count == 1
