"""One-shot CLI for the credential-bearing ImmPort producer."""

from __future__ import annotations

import sys

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.immport_producer import (
    IMMPORT_CREDENTIAL_FD,
    MAX_IMMPORT_PRODUCER_REQUEST_BYTES,
    parse_immport_producer_request,
    produce_immport_response,
    read_runtime_credential,
)

_ERROR_BYTES = canonical_json_bytes(
    {
        'error_code': 'authenticated_immport_producer_failed',
        'status': 'error',
    }
)


def dispatch(payload: bytes, *, credential_fd: int = IMMPORT_CREDENTIAL_FD) -> bytes:
    """Parse public stdin, consume fd 3, and return only a sanitized canonical result."""

    request = parse_immport_producer_request(payload)
    credential = read_runtime_credential(credential_fd)
    response = produce_immport_response(
        request,
        credential,
        public_request_bytes=payload,
    )
    return canonical_json_bytes(response)


def main() -> int:
    payload = sys.stdin.buffer.read(MAX_IMMPORT_PRODUCER_REQUEST_BYTES + 1)
    failed = False
    output = b''
    try:
        output = dispatch(payload)
    except BaseException:
        failed = True
    if failed:
        sys.stdout.buffer.write(_ERROR_BYTES)
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == '__main__':  # pragma: no cover - console-script path
    raise SystemExit(main())
