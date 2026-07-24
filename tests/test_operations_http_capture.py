from __future__ import annotations

import hashlib
import io
import socket
import tempfile
import time
import unittest
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from vaxreplay.operations.http_capture import (
    BodyTooLargeError,
    CaptureDeadlineExceededError,
    ContentLengthMismatchError,
    DisallowedHostError,
    DnsAddressLimitError,
    DnsResolutionTimeoutError,
    HttpRequestHeader,
    HttpsCaptureReceipt,
    HttpsCaptureRequest,
    NormalizedResponseHeader,
    PreparedHttpsRequest,
    RedirectRejectedError,
    RequestPolicyError,
    ResponseProtocolError,
    SubprocessHttpsDnsResolver,
    TemporaryHttpsCapture,
    TlsPeerMetadata,
    TruncatedBodyError,
    UnexpectedStatusError,
    UrllibHttpsTransport,
    capture_https,
    capture_https_to_tempfile,
)

_START = datetime(2026, 7, 13, 8, 0, 0, tzinfo=timezone(timedelta(hours=-7)))
_END = _START + timedelta(seconds=2)


class _SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status_code: int = 200,
        final_url: str = 'https://archive.example.org/data?q=stable',
        headers: tuple[tuple[str, str], ...] | None = None,
        tls_peer: TlsPeerMetadata | None = None,
        read_chunk_size: int | None = None,
    ) -> None:
        self.status_code = status_code
        self.final_url = final_url
        self.response_headers = headers if headers is not None else (('Content-Length', str(len(body))),)
        self._body = body
        self._position = 0
        self._tls_peer = tls_peer
        self._read_chunk_size = read_chunk_size
        self.read_count = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        self.read_count += 1
        if self._position >= len(self._body):
            return b''
        amount = size if self._read_chunk_size is None else min(size, self._read_chunk_size)
        result = self._body[self._position : self._position + amount]
        self._position += len(result)
        return result

    def tls_peer_metadata(self) -> TlsPeerMetadata | None:
        return self._tls_peer

    def close(self) -> None:
        self.closed = True


class _FakeTransport:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[PreparedHttpsRequest] = []

    def open(self, request: PreparedHttpsRequest) -> _FakeResponse:
        self.requests.append(request)
        return self.response


class _FakeResolver:
    def __init__(
        self,
        answers: tuple[tuple[int, int, int, str, tuple[object, ...]], ...] = (),
        *,
        error: Exception | None = None,
        callback: Callable[[], None] | None = None,
    ) -> None:
        self.answers = answers
        self.error = error
        self.callback = callback
        self.calls: list[tuple[str, int, float, int]] = []

    def resolve(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float,
        max_addresses: int,
    ) -> tuple[tuple[int, int, int, str, tuple[object, ...]], ...]:
        self.calls.append((host, port, timeout_seconds, max_addresses))
        if self.callback is not None:
            self.callback()
        if self.error is not None:
            raise self.error
        return self.answers


class _ManualMonotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _slow_dns_worker(connection: Connection, host: str, port: int, max_addresses: int) -> None:
    del host, port, max_addresses
    time.sleep(0.5)
    connection.close()


def _request(**changes: object) -> HttpsCaptureRequest:
    values: dict[str, object] = {
        'url': 'https://archive.example.org/data?q=stable',
        'allowed_host': 'archive.example.org',
        'allowed_query_names': ('q',),
        'max_body_bytes': 1024,
    }
    values.update(changes)
    return HttpsCaptureRequest.model_validate(values)


class HttpsCaptureRequestTest(unittest.TestCase):
    def test_request_policy_is_closed_and_unauthenticated(self) -> None:
        invalid_values = (
            {'method': 'POST'},
            {'url': 'http://archive.example.org/data'},
            {'url': 'https://other.example.org/data'},
            {'url': 'https://user@archive.example.org/data'},
            {'url': 'https://archive.example.org:443/data'},
            {'url': 'https://archive.example.org/data#section'},
            {'url': 'https://archive.example.org/data\\suffix'},
            {'url': 'https://archive.example.org/data with space'},
            {'url': 'https://archive.example.org/data?'},
            {'url': 'https://archive.example.org/data?access_token=secret'},
            {'url': 'https://archive.example.org/data?credential=value'},
            {'url': 'https://archive.example.org/data?bearer=value'},
            {'url': 'https://archive.example.org/data?key=value'},
            {'url': 'https://archive.example.org/data?sig=value'},
            {'url': 'https://archive.example.org/data?x-amz-signature=value'},
            {'allowed_host': 'Archive.example.org'},
            {'allowed_host': '127.0.0.1', 'url': 'https://127.0.0.1/data'},
            {'allowed_host': 'localhost', 'url': 'https://localhost/data'},
            {'redirect_policy': 'follow'},
            {'allowed_status_codes': (200, 302)},
        )
        for changes in invalid_values:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                _request(**changes)

        for header_name in ('authorization', 'cookie', 'proxy-authorization', 'x-api-key'):
            with self.subTest(header_name=header_name), self.assertRaises(ValidationError):
                HttpRequestHeader(name=header_name, value='secret')

        identity = HttpRequestHeader(name='accept-encoding', value='identity')
        with self.assertRaisesRegex(ValidationError, 'callers cannot set accept-encoding'):
            _request(request_headers=(identity,))
        host = HttpRequestHeader(name='host', value='archive.example.org')
        with self.assertRaisesRegex(ValidationError, 'collector controls them'):
            _request(request_headers=(host,))
        with self.assertRaisesRegex(ValidationError, 'ASCII'):
            HttpRequestHeader(name='user-agent', value='collector-é')

    def test_requires_sorted_unique_safe_headers_and_statuses(self) -> None:
        accept = HttpRequestHeader(name='accept', value='application/json')
        range_header = HttpRequestHeader(name='range', value='bytes=0-9')
        request = _request(
            request_headers=(accept, range_header),
            allowed_status_codes=(200, 206),
        )
        self.assertEqual(request.allowed_status_codes, (200, 206))

        with self.assertRaisesRegex(ValidationError, 'sorted order'):
            _request(request_headers=(range_header, accept))
        with self.assertRaisesRegex(ValidationError, 'sorted unique'):
            _request(allowed_status_codes=(206, 200))
        with self.assertRaisesRegex(ValidationError, 'credential-like'):
            _request(allowed_query_names=('key',))
        with self.assertRaisesRegex(ValidationError, 'credential-like'):
            _request(allowed_query_names=('apiToken',))
        with self.assertRaisesRegex(ValidationError, 'explicitly allowlisted'):
            _request(allowed_query_names=())

        clinical_trials = HttpsCaptureRequest(
            url='https://archive.example.org/api/v2/studies?countTotal=true&markupFormat=markdown&pageSize=1000',
            allowed_host='archive.example.org',
            allowed_query_names=('countTotal', 'markupFormat', 'pageSize'),
            max_body_bytes=4096,
        )
        self.assertEqual(
            clinical_trials.allowed_query_names,
            ('countTotal', 'markupFormat', 'pageSize'),
        )


class ExactHttpsCaptureTest(unittest.TestCase):
    def test_direct_default_transport_derives_an_absolute_request_deadline(self) -> None:
        transport = _FakeTransport(_FakeResponse(b'body'))
        with patch(
            'vaxreplay.operations.http_capture.UrllibHttpsTransport',
            return_value=transport,
        ) as constructor:
            capture_https(
                _request(timeout_seconds=5.0),
                io.BytesIO(),
                clock=_SequenceClock(_START, _END),
                monotonic=lambda: 10.0,
            )

        constructor.assert_called_once()
        self.assertEqual(constructor.call_args.kwargs['deadline_monotonic'], 15.0)
        self.assertEqual(constructor.call_args.kwargs['dns_timeout_seconds'], 5.0)

    def test_streams_exact_bytes_and_builds_a_non_secret_receipt(self) -> None:
        body = b'first\x00second\n'
        tls_peer = TlsPeerMetadata(
            server_name='archive.example.org',
            peer_address='203.0.113.7',
            peer_port=443,
            tls_version='TLSv1.3',
            cipher_suite='TLS_AES_256_GCM_SHA384',
            cipher_bits=256,
            certificate_der_sha256='a' * 64,
        )
        response = _FakeResponse(
            body,
            headers=(
                ('Set-Cookie', 'session=must-not-be-recorded'),
                ('Location', 'https://cdn.example.org/object?token=must-not-be-recorded'),
                ('Content-Location', '/object?signature=must-not-be-recorded'),
                ('ETag', '"release-7"'),
                ('Content-Type', ' application/octet-stream '),
                ('Content-Length', str(len(body))),
            ),
            tls_peer=tls_peer,
            read_chunk_size=3,
        )
        transport = _FakeTransport(response)
        sink = io.BytesIO()

        receipt = capture_https(
            _request(request_headers=(HttpRequestHeader(name='accept', value='application/octet-stream'),)),
            sink,
            transport=transport,
            clock=_SequenceClock(_START, _END),
        )

        self.assertEqual(sink.getvalue(), body)
        self.assertEqual(receipt.body_sha256, hashlib.sha256(body).hexdigest())
        self.assertEqual(receipt.body_byte_count, len(body))
        self.assertEqual(receipt.started_at, _START.astimezone(timezone.utc))
        self.assertEqual(receipt.completed_at, _END.astimezone(timezone.utc))
        self.assertEqual(receipt.tls_peer, tls_peer)
        self.assertEqual(
            tuple(header.name for header in receipt.response_headers),
            ('content-length', 'content-type', 'etag'),
        )
        self.assertNotIn('session', receipt.model_dump_json())
        self.assertNotIn('must-not-be-recorded', receipt.model_dump_json())
        prepared = transport.requests[0]
        self.assertEqual(prepared.method, 'GET')
        self.assertEqual(
            tuple((header.name, header.value) for header in prepared.headers),
            (
                ('accept', 'application/octet-stream'),
                ('accept-encoding', 'identity'),
                ('host', 'archive.example.org'),
                ('user-agent', 'VaxReplay-Archival-Capture/0.1'),
            ),
        )
        self.assertTrue(response.closed)

    def test_rejects_redirects_changed_urls_disallowed_hosts_and_status(self) -> None:
        cases = (
            (_FakeResponse(b'', status_code=302), RedirectRejectedError),
            (
                _FakeResponse(b'', final_url='https://archive.example.org/elsewhere'),
                RedirectRejectedError,
            ),
            (
                _FakeResponse(b'', final_url='https://internal.example.org/data'),
                DisallowedHostError,
            ),
        )
        for response, error_type in cases:
            with self.subTest(error_type=error_type), self.assertRaises(error_type):
                capture_https(_request(), io.BytesIO(), transport=_FakeTransport(response))
            self.assertTrue(response.closed)
            self.assertEqual(response.read_count, 0)

        response = _FakeResponse(
            b'not found',
            status_code=404,
            headers=(
                ('Content-Type', 'text/plain'),
                ('Set-Cookie', 'secret=value'),
                ('Location', 'https://cdn.example.org/object?token=secret'),
                ('Content-Location', '/object?signature=secret'),
            ),
        )
        with self.assertRaises(UnexpectedStatusError) as caught:
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(response))
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.final_url, _request().url)
        self.assertEqual(tuple(header.name for header in caught.exception.response_headers), ('content-type',))
        self.assertNotIn('secret', repr(caught.exception.response_headers))
        self.assertEqual(response.read_count, 0)

    def test_rejects_declared_and_streamed_oversize_bodies(self) -> None:
        declared = _FakeResponse(b'', headers=(('Content-Length', '5'),))
        with self.assertRaises(BodyTooLargeError):
            capture_https(_request(max_body_bytes=4), io.BytesIO(), transport=_FakeTransport(declared))
        self.assertEqual(declared.read_count, 0)

        streamed = _FakeResponse(b'abcde', headers=(), read_chunk_size=3)
        sink = io.BytesIO()
        with self.assertRaises(BodyTooLargeError):
            capture_https(_request(max_body_bytes=4), sink, transport=_FakeTransport(streamed))
        self.assertEqual(sink.getvalue(), b'abc')
        self.assertTrue(streamed.closed)

    def test_rejects_truncated_overlong_and_ambiguous_content_length(self) -> None:
        truncated = _FakeResponse(b'abc', headers=(('Content-Length', '4'),))
        with self.assertRaises(TruncatedBodyError):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(truncated))

        overlong = _FakeResponse(b'abcd', headers=(('Content-Length', '3'),))
        with self.assertRaises(ContentLengthMismatchError):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(overlong))

        duplicate = _FakeResponse(
            b'abc',
            headers=(('Content-Length', '3'), ('Content-Length', '3')),
        )
        with self.assertRaisesRegex(ResponseProtocolError, 'multiple Content-Length'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(duplicate))
        self.assertEqual(duplicate.read_count, 0)

        unbounded_integer = _FakeResponse(b'', headers=(('Content-Length', '9' * 100),))
        with self.assertRaisesRegex(ResponseProtocolError, 'supported integer range'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(unbounded_integer))

    def test_rejects_excessive_raw_header_occurrences_before_body_read(self) -> None:
        response = _FakeResponse(
            b'body',
            headers=tuple(('X-Ignored', 'value') for _ in range(257)),
        )
        with self.assertRaisesRegex(ResponseProtocolError, 'field count'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(response))
        self.assertEqual(response.read_count, 0)

    def test_rejects_excessive_raw_header_characters_before_body_read(self) -> None:
        response = _FakeResponse(
            b'body',
            headers=(('X-Ignored', 'x' * (256 * 1024)),),
        )
        with self.assertRaisesRegex(ResponseProtocolError, 'characters'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(response))
        self.assertEqual(response.read_count, 0)

    def test_receipt_replay_rejects_impossible_selected_header_bounds(self) -> None:
        receipt = capture_https(
            _request(),
            io.BytesIO(),
            transport=_FakeTransport(_FakeResponse(b'body')),
            clock=_SequenceClock(_START, _END),
        )
        payload = receipt.model_dump(mode='python')
        payload['response_headers'] = (
            NormalizedResponseHeader(name='cache-control', values=tuple('x' for _ in range(257))),
        )
        with self.assertRaisesRegex(ValidationError, 'field count'):
            HttpsCaptureReceipt.model_validate(payload)

        payload['response_headers'] = (
            NormalizedResponseHeader(
                name='cache-control',
                values=tuple('x' * 16_384 for _ in range(17)),
            ),
        )
        with self.assertRaisesRegex(ValidationError, 'characters'):
            HttpsCaptureReceipt.model_validate(payload)

    def test_rejects_encoded_or_ambiguous_entity_framing(self) -> None:
        encoded = _FakeResponse(
            b'compressed',
            headers=(('Content-Encoding', 'gzip'), ('Content-Length', '10')),
        )
        with self.assertRaisesRegex(ResponseProtocolError, 'ignored Accept-Encoding'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(encoded))

        ambiguous = _FakeResponse(
            b'abc',
            headers=(('Content-Length', '3'), ('Transfer-Encoding', 'chunked')),
        )
        with self.assertRaisesRegex(ResponseProtocolError, 'both Content-Length'):
            capture_https(_request(), io.BytesIO(), transport=_FakeTransport(ambiguous))

        invalid_transfer_encodings = (
            (('Transfer-Encoding', 'gzip'),),
            (('Transfer-Encoding', 'chunked, gzip'),),
            (('Transfer-Encoding', 'chunked'), ('Transfer-Encoding', 'chunked')),
        )
        for headers in invalid_transfer_encodings:
            with (
                self.subTest(headers=headers),
                self.assertRaisesRegex(
                    ResponseProtocolError,
                    'exactly one chunked',
                ),
            ):
                capture_https(
                    _request(),
                    io.BytesIO(),
                    transport=_FakeTransport(_FakeResponse(b'abc', headers=headers)),
                )

        decoded_chunked = _FakeResponse(b'abc', headers=(('Transfer-Encoding', 'chunked'),))
        receipt = capture_https(
            _request(),
            io.BytesIO(),
            transport=_FakeTransport(decoded_chunked),
            clock=_SequenceClock(_START, _END),
        )
        self.assertEqual(receipt.body_byte_count, 3)

    def test_tempfile_helper_is_durable_on_success_and_cleans_up_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            successful = capture_https_to_tempfile(
                _request(max_body_bytes=10),
                directory=root,
                transport=_FakeTransport(_FakeResponse(b'payload')),
                clock=_SequenceClock(_START, _END),
            )
            self.assertIsInstance(successful, TemporaryHttpsCapture)
            self.assertEqual(successful.path.read_bytes(), b'payload')
            successful.delete()
            self.assertFalse(successful.path.exists())

            with self.assertRaises(BodyTooLargeError):
                capture_https_to_tempfile(
                    _request(max_body_bytes=4),
                    directory=root,
                    transport=_FakeTransport(_FakeResponse(b'abcdef', headers=(), read_chunk_size=3)),
                )
            self.assertEqual(tuple(root.iterdir()), ())


class DefaultTransportEgressTest(unittest.TestCase):
    def test_subprocess_resolver_returns_a_bounded_system_answer(self) -> None:
        answers = SubprocessHttpsDnsResolver().resolve(
            'localhost',
            443,
            timeout_seconds=5.0,
            max_addresses=16,
        )
        self.assertGreater(len(answers), 0)
        self.assertLessEqual(len(answers), 16)
        self.assertTrue(all(answer[4][1] == 443 for answer in answers))

    def test_subprocess_resolver_kills_a_resolution_past_its_deadline(self) -> None:
        with self.assertRaises(DnsResolutionTimeoutError):
            SubprocessHttpsDnsResolver(worker=_slow_dns_worker).resolve(
                'archive.example.org',
                443,
                timeout_seconds=0.05,
                max_addresses=16,
            )

    def test_default_transport_rejects_non_public_dns_before_connect(self) -> None:
        private_answer = ((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('169.254.169.254', 443)),)
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=1.0,
        )
        with (
            patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
            self.assertRaisesRegex(RequestPolicyError, 'non-public endpoint'),
        ):
            UrllibHttpsTransport(resolver=_FakeResolver(private_answer)).open(prepared)
        socket_constructor.assert_not_called()

    def test_default_transport_rejects_multicast_and_site_local_dns(self) -> None:
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=1.0,
        )
        cases = (
            (socket.AF_INET, ('224.0.0.1', 443)),
            (socket.AF_INET6, ('ff02::1', 443, 0, 0)),
            (socket.AF_INET6, ('fec0::1', 443, 0, 0)),
        )
        for family, address in cases:
            answer = ((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', address),)
            with (
                self.subTest(address=address[0]),
                patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
                self.assertRaisesRegex(RequestPolicyError, 'non-public endpoint'),
            ):
                UrllibHttpsTransport(resolver=_FakeResolver(answer)).open(prepared)
            socket_constructor.assert_not_called()

    def test_default_transport_rejects_mixed_public_private_dns(self) -> None:
        mixed_answers = (
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('8.8.8.8', 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', 443)),
        )
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=1.0,
        )
        with (
            patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
            self.assertRaises(RequestPolicyError),
        ):
            UrllibHttpsTransport(resolver=_FakeResolver(mixed_answers)).open(prepared)
        socket_constructor.assert_not_called()

    def test_default_transport_rejects_dns_answer_count_above_committed_cap(self) -> None:
        answers = (
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('8.8.8.8', 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('1.1.1.1', 443)),
        )
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=10.0,
        )
        with (
            patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
            self.assertRaisesRegex(DnsAddressLimitError, 'max_addresses=1'),
        ):
            UrllibHttpsTransport(
                resolver=_FakeResolver(answers),
                max_dns_addresses=1,
            ).open(prepared)
        socket_constructor.assert_not_called()

    def test_dns_is_debited_from_monotonic_request_deadline(self) -> None:
        monotonic = _ManualMonotonic()

        def exhaust_deadline() -> None:
            monotonic.value = 5.0

        resolver = _FakeResolver(
            ((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('8.8.8.8', 443)),),
            callback=exhaust_deadline,
        )
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=10.0,
        )
        with (
            patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
            self.assertRaises(CaptureDeadlineExceededError),
        ):
            UrllibHttpsTransport(
                resolver=resolver,
                monotonic=monotonic,
                deadline_monotonic=5.0,
                dns_timeout_seconds=4.0,
                max_dns_addresses=7,
            ).open(prepared)
        self.assertEqual(resolver.calls, [('archive.example.org', 443, 4.0, 7)])
        socket_constructor.assert_not_called()

    def test_dns_timeout_is_propagated_without_connection_attempt(self) -> None:
        resolver = _FakeResolver(error=DnsResolutionTimeoutError('simulated timeout'))
        prepared = PreparedHttpsRequest(
            method='GET',
            url='https://archive.example.org/data',
            headers=(HttpRequestHeader(name='accept-encoding', value='identity'),),
            timeout_seconds=10.0,
        )
        with (
            patch('vaxreplay.operations.http_capture.socket.socket') as socket_constructor,
            self.assertRaises(DnsResolutionTimeoutError),
        ):
            UrllibHttpsTransport(resolver=resolver).open(prepared)
        self.assertEqual(len(resolver.calls), 1)
        socket_constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
