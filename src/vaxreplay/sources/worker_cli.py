"""Hermetic stdio entrypoints for reviewed production source workers.

OCI images can pin ``python -m vaxreplay.sources.worker_cli <worker-name>`` as
their entrypoint.  The generic operations layer validates the canonical callback
wire format; this module only selects one reviewed source implementation and never
opens files, reads ambient configuration, or accesses the network.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from vaxreplay.operations.hermetic_callback_worker import (
    run_adapter_worker,
    run_source_verifier_worker,
)
from vaxreplay.sources.clinicaltrials import (
    adapt_ctgov_study_candidates,
    verify_ctgov_source,
)
from vaxreplay.sources.iedb import (
    adapt_tier_a_iedb_antigen_targets,
    verify_tier_a_iedb_source,
)
from vaxreplay.sources.immport import (
    adapt_tier_a_immport_arms,
    verify_tier_a_immport_source,
)

_MAX_REQUEST_BYTES = 512 * 1024 * 1024

Worker = Callable[[bytes], bytes]


def run_iedb_source_verifier(request_bytes: bytes) -> bytes:
    return run_source_verifier_worker(request_bytes, verify_tier_a_iedb_source)


def run_iedb_antigen_adapter(request_bytes: bytes) -> bytes:
    return run_adapter_worker(request_bytes, adapt_tier_a_iedb_antigen_targets)


def run_ctgov_source_verifier(request_bytes: bytes) -> bytes:
    return run_source_verifier_worker(request_bytes, verify_ctgov_source)


def run_ctgov_study_adapter(request_bytes: bytes) -> bytes:
    return run_adapter_worker(request_bytes, adapt_ctgov_study_candidates)


def run_immport_source_verifier(request_bytes: bytes) -> bytes:
    return run_source_verifier_worker(request_bytes, verify_tier_a_immport_source)


def run_immport_arm_adapter(request_bytes: bytes) -> bytes:
    return run_adapter_worker(request_bytes, adapt_tier_a_immport_arms)


_WORKERS: dict[str, Worker] = {
    'ctgov-source-verifier': run_ctgov_source_verifier,
    'ctgov-study-adapter': run_ctgov_study_adapter,
    'iedb-antigen-adapter': run_iedb_antigen_adapter,
    'iedb-source-verifier': run_iedb_source_verifier,
    'immport-arm-adapter': run_immport_arm_adapter,
    'immport-source-verifier': run_immport_source_verifier,
}


def dispatch(worker_name: str, request_bytes: bytes) -> bytes:
    worker = _WORKERS.get(worker_name)
    if worker is None:
        raise ValueError(f'unknown production source worker: {worker_name!r}')
    if not isinstance(request_bytes, bytes) or not request_bytes:
        raise ValueError('hermetic source worker request must be nonempty exact bytes')
    if len(request_bytes) > _MAX_REQUEST_BYTES:
        raise ValueError('hermetic source worker request exceeds its fixed byte limit')
    return worker(request_bytes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one VaxReplay hermetic production source worker')
    parser.add_argument('worker', choices=tuple(sorted(_WORKERS)))
    arguments = parser.parse_args(argv)
    request_bytes = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    try:
        response_bytes = dispatch(arguments.worker, request_bytes)
    except (RuntimeError, TypeError, ValueError):
        # Source verifier/adapter errors can be derived from captured bytes.  Keep the process
        # boundary fail-closed without copying source-controlled values into runtime logs.
        print('production source worker rejected input', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(response_bytes)
    sys.stdout.buffer.flush()
    return 0


if __name__ == '__main__':  # pragma: no cover - exercised by the installed OCI entrypoint
    raise SystemExit(main())
