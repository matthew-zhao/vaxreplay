"""Worker-side dispatch for the canonical hermetic promotion callback protocol.

An OCI image supplies one reviewed source verifier or adapter implementation and
passes its stdin bytes to the matching function below.  These helpers do not create
an isolation claim; the host-side OCI executor creates and signs that evidence.
"""

from __future__ import annotations

from collections.abc import Callable

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.hermetic_callback_protocol import (
    HermeticAdapterOutput,
    HermeticNamedOutput,
    HermeticSourceVerifierOutput,
    decode_callback_bytes,
    encode_callback_bytes,
    parse_adapter_input,
    parse_source_input,
)
from vaxreplay.operations.hermetic_execution import (
    build_hermetic_worker_response,
    parse_hermetic_worker_request,
)
from vaxreplay.operations.promotion import (
    AdapterRunResult,
    AdapterSourceInput,
    ExactPromotedArtifact,
    ExactPromotedCapture,
    SourceVerificationInput,
    SourceVerifierRunResult,
    TrustedPromotionAdapter,
)
from vaxreplay.operations.promotion_schema import SourceRecordBinding

type HermeticSourceWorker = Callable[..., SourceVerifierRunResult]


class HermeticCallbackWorkerError(ValueError):
    """A worker request, callback result, or canonical output failed closed."""


def run_source_verifier_worker(request_bytes: bytes, worker: HermeticSourceWorker) -> bytes:
    """Execute a reviewed source worker and return canonical outer stdout bytes."""

    request = parse_hermetic_worker_request(request_bytes)
    if request.purpose != 'source_verifier':
        raise HermeticCallbackWorkerError('source-verifier image received a different callback purpose')
    inner = parse_source_input(decode_callback_bytes(request.input_base64))
    source_input = SourceVerificationInput(
        source_id=inner.source_id,
        captures=tuple(_capture(value) for value in inner.captures),
        capture_inventory_sha256=inner.capture_inventory_sha256,
    )
    try:
        result = worker(
            source_input,
            decode_callback_bytes(request.callback_policy_base64),
            implementation_sha256=request.implementation.sha256,
            execution_environment_sha256=request.execution_environment.sha256,
        )
    except Exception as error:
        raise HermeticCallbackWorkerError(f'source-verifier worker failed: {error}') from error
    if not isinstance(result, SourceVerifierRunResult) or not isinstance(result.verified_records, bytes):
        raise HermeticCallbackWorkerError('source-verifier worker returned the wrong result type')
    output = canonical_json_bytes(
        HermeticSourceVerifierOutput(
            result=result.result,
            verified_records_base64=encode_callback_bytes(result.verified_records),
        )
    )
    return build_hermetic_worker_response(request_bytes, output)


def run_adapter_worker(request_bytes: bytes, worker: TrustedPromotionAdapter) -> bytes:
    """Execute a reviewed normalization worker and return canonical outer stdout bytes."""

    request = parse_hermetic_worker_request(request_bytes)
    if request.purpose != 'adapter':
        raise HermeticCallbackWorkerError('adapter image received a different callback purpose')
    inner = parse_adapter_input(decode_callback_bytes(request.input_base64))
    inputs = tuple(
        AdapterSourceInput(
            source_id=value.source_id,
            captures=tuple(_capture(capture) for capture in value.captures),
            verification_result=value.verification_result,
            verified_records=_source_records(decode_callback_bytes(value.verified_records_base64)),
        )
        for value in inner.sources
    )
    try:
        result = worker(inputs, decode_callback_bytes(request.callback_policy_base64))
    except Exception as error:
        raise HermeticCallbackWorkerError(f'adapter worker failed: {error}') from error
    if not isinstance(result, AdapterRunResult):
        raise HermeticCallbackWorkerError('adapter worker returned the wrong result type')
    auxiliary = result.auxiliary_outputs or {}
    output = canonical_json_bytes(
        HermeticAdapterOutput(
            candidate_records_base64=encode_callback_bytes(result.candidate_records),
            evidence_records_base64=encode_callback_bytes(result.evidence_records),
            dispositions_base64=encode_callback_bytes(result.dispositions),
            auxiliary_outputs=tuple(
                HermeticNamedOutput(name=name, payload_base64=encode_callback_bytes(auxiliary[name]))
                for name in sorted(auxiliary)
            ),
        )
    )
    return build_hermetic_worker_response(request_bytes, output)


def _capture(value) -> ExactPromotedCapture:
    return ExactPromotedCapture(
        binding=value.binding,
        artifacts=tuple(
            ExactPromotedArtifact(
                binding=artifact.binding,
                payload=decode_callback_bytes(artifact.payload_base64),
            )
            for artifact in value.artifacts
        ),
    )


def _source_records(payload: bytes) -> tuple[SourceRecordBinding, ...]:
    if not payload or not payload.endswith(b'\n'):
        raise HermeticCallbackWorkerError('verified source records must be nonempty canonical JSONL')
    records: list[SourceRecordBinding] = []
    for ordinal, line in enumerate(payload[:-1].split(b'\n'), start=1):
        if not line:
            raise HermeticCallbackWorkerError('verified source records contain an empty JSONL row')
        try:
            record = SourceRecordBinding.model_validate_json(line)
        except ValueError as error:
            raise HermeticCallbackWorkerError(f'invalid source-record row {ordinal}') from error
        if canonical_json_bytes(record) != line:
            raise HermeticCallbackWorkerError(f'source-record row {ordinal} is not canonical JSON')
        records.append(record)
    if b''.join(canonical_json_bytes(record) + b'\n' for record in records) != payload:
        raise HermeticCallbackWorkerError('verified source-record JSONL has a noncanonical encoding')
    return tuple(records)
