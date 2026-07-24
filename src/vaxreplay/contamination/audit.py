"""Deterministic retrieval, run binding, and fixed contamination-audit aggregation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from datetime import datetime

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.contamination.schema import (
    ArtifactBinding,
    ArtifactContaminationAudit,
    AuditDisposition,
    AuditReasonCode,
    CalibrationPolicy,
    ContaminationAuditInput,
    ContaminationAuditManifest,
    ContaminationAuditPolicy,
    DeterministicRetrievalCandidate,
    ExactByteSpan,
    FindingSeverity,
    IdentifierNeedle,
    JudgeCalibrationResult,
    JudgeVerdict,
    LlmAuditRun,
    LlmJudgeOutput,
    PinnedLlmJudge,
    RetrievalKind,
    audit_manifest_sha256,
    model_sha256,
)

_TOKEN_PATTERN = re.compile(r'\w+(?:-\w+)*', flags=re.UNICODE)


class ContaminationAuditError(ValueError):
    """Raised when audit material or a pinned audit record fails closed."""


def artifact_binding(artifact_id: str, payload: bytes) -> ArtifactBinding:
    """Bind one non-empty exact byte artifact."""

    if not isinstance(payload, bytes) or not payload:
        raise ContaminationAuditError('audit artifacts must be non-empty bytes')
    return ArtifactBinding(
        artifact_id=artifact_id,
        sha256=_sha256(payload),
        byte_count=len(payload),
    )


def make_audit_input(
    *,
    case_id: str,
    episode_id: str,
    decision_package_sha256: str,
    episode_manifest_sha256: str,
    public_artifact_id: str,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
) -> ContaminationAuditInput:
    """Create canonical input commitments from the exact bytes that judges will receive."""

    if not comparison_payloads:
        raise ContaminationAuditError('at least one private comparison artifact is required')
    comparisons = tuple(
        artifact_binding(artifact_id, payload) for artifact_id, payload in sorted(comparison_payloads.items())
    )
    return ContaminationAuditInput(
        case_id=case_id,
        episode_id=episode_id,
        decision_package_sha256=decision_package_sha256,
        episode_manifest_sha256=episode_manifest_sha256,
        public_artifact=artifact_binding(public_artifact_id, public_payload),
        comparison_artifacts=comparisons,
    )


def validate_exact_byte_span(payload: bytes, binding: ArtifactBinding, span: ExactByteSpan) -> None:
    """Fail unless ``span`` is an exact UTF-8 byte slice of ``payload`` and its binding."""

    _require_artifact(payload, binding)
    if span.artifact_id != binding.artifact_id or span.artifact_sha256 != binding.sha256:
        raise ContaminationAuditError('exact byte span is bound to a different artifact')
    if span.end_byte > len(payload):
        raise ContaminationAuditError('exact byte span exceeds its artifact')
    if payload[span.start_byte : span.end_byte] != span.quote.encode('utf-8'):
        raise ContaminationAuditError('exact byte span quote does not match the bound artifact bytes')


def retrieve_exact_candidates(
    audit_input: ContaminationAuditInput,
    *,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
    identifiers: Iterable[IdentifierNeedle] = (),
    policy: ContaminationAuditPolicy,
) -> tuple[DeterministicRetrievalCandidate, ...]:
    """Find literal UTF-8 n-gram and identifier matches in canonical order.

    Matching is case-sensitive and byte-exact.  Hitting the configured candidate bound raises an
    error instead of truncating results and silently making the screen look cleaner.
    """

    _validate_materials(audit_input, public_payload, comparison_payloads)
    public_text = _decode_utf8(public_payload, audit_input.public_artifact.artifact_id)
    for artifact_id, payload in comparison_payloads.items():
        _decode_utf8(payload, artifact_id)

    by_id: dict[str, DeterministicRetrievalCandidate] = {}
    token_offsets = _token_byte_offsets(public_text)
    ngram_tokens = policy.retrieval.ngram_tokens
    if len(token_offsets) >= ngram_tokens:
        for index in range(len(token_offsets) - ngram_tokens + 1):
            start = token_offsets[index][0]
            end = token_offsets[index + ngram_tokens - 1][1]
            exact = public_payload[start:end]
            if len(exact) < policy.retrieval.minimum_ngram_bytes:
                continue
            quote = exact.decode('utf-8')
            for reference in audit_input.comparison_artifacts:
                reference_payload = comparison_payloads[reference.artifact_id]
                for reference_start in _find_all(reference_payload, exact):
                    candidate = _make_candidate(
                        kind=RetrievalKind.EXACT_NGRAM,
                        public_binding=audit_input.public_artifact,
                        public_start=start,
                        reference_binding=reference,
                        reference_start=reference_start,
                        quote=quote,
                        token_count=ngram_tokens,
                        identifier_id=None,
                    )
                    by_id[candidate.candidate_id] = candidate
                    _require_candidate_limit(by_id, policy)

    needles = tuple(sorted(identifiers, key=lambda value: (value.identifier_id, value.value)))
    needle_ids = tuple(needle.identifier_id for needle in needles)
    if len(needle_ids) != len(set(needle_ids)):
        raise ContaminationAuditError('identifier needles must have unique identifier IDs')
    comparisons_by_id = {artifact.artifact_id: artifact for artifact in audit_input.comparison_artifacts}
    for needle in needles:
        reference = comparisons_by_id.get(needle.reference_artifact_id)
        if reference is None:
            raise ContaminationAuditError(
                f'identifier {needle.identifier_id} references an unbound comparison artifact'
            )
        encoded = needle.value.encode('utf-8')
        public_offsets = tuple(_find_all(public_payload, encoded))
        reference_offsets = tuple(_find_all(comparison_payloads[reference.artifact_id], encoded))
        if not reference_offsets:
            raise ContaminationAuditError(
                f'identifier {needle.identifier_id} is absent from its declared reference artifact'
            )
        for public_start in public_offsets:
            for reference_start in reference_offsets:
                candidate = _make_candidate(
                    kind=RetrievalKind.IDENTIFIER,
                    public_binding=audit_input.public_artifact,
                    public_start=public_start,
                    reference_binding=reference,
                    reference_start=reference_start,
                    quote=needle.value,
                    token_count=None,
                    identifier_id=needle.identifier_id,
                )
                by_id[candidate.candidate_id] = candidate
                _require_candidate_limit(by_id, policy)
    return tuple(by_id[candidate_id] for candidate_id in sorted(by_id))


def judge_request_sha256(
    audit_input: ContaminationAuditInput,
    judge: PinnedLlmJudge,
) -> str:
    """Bind a judge request to exact input bytes through their commitments and the judge pin."""

    return _sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.contamination-judge-request.v0.1',
                'audit_input_sha256': model_sha256(audit_input),
                'judge_pin_sha256': model_sha256(judge),
            }
        )
    )


def make_llm_audit_run(
    *,
    run_id: str,
    judge: PinnedLlmJudge,
    audit_input: ContaminationAuditInput,
    output: LlmJudgeOutput,
    started_at: datetime,
    finished_at: datetime,
) -> LlmAuditRun:
    """Record externally executed provider output without depending on a provider SDK."""

    output_bytes = canonical_json_bytes(output)
    return LlmAuditRun(
        run_id=run_id,
        judge_id=judge.judge_id,
        judge_pin_sha256=model_sha256(judge),
        audit_input_sha256=model_sha256(audit_input),
        request_sha256=judge_request_sha256(audit_input, judge),
        output_sha256=_sha256(output_bytes),
        output_bytes=len(output_bytes),
        started_at=started_at,
        finished_at=finished_at,
        output=output,
    )


def build_contamination_audit(
    *,
    audit_id: str,
    audit_input: ContaminationAuditInput,
    policy: ContaminationAuditPolicy,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
    judge_runs: Iterable[LlmAuditRun],
    screened_at: datetime,
    identifiers: Iterable[IdentifierNeedle] = (),
) -> ArtifactContaminationAudit:
    """Validate every bound input and apply the fixed conservative aggregation rule."""

    candidates = retrieve_exact_candidates(
        audit_input,
        public_payload=public_payload,
        comparison_payloads=comparison_payloads,
        identifiers=identifiers,
        policy=policy,
    )
    runs = tuple(sorted(judge_runs, key=lambda run: run.judge_id))
    _validate_judge_runs(
        audit_input,
        policy,
        runs,
        public_payload=public_payload,
        comparison_payloads=comparison_payloads,
    )
    disposition, reasons = _aggregate(policy.calibration, candidates, runs)
    return ArtifactContaminationAudit(
        audit_id=audit_id,
        audit_input=audit_input,
        audit_input_sha256=model_sha256(audit_input),
        policy_sha256=model_sha256(policy),
        deterministic_candidates=candidates,
        judge_runs=runs,
        disposition=disposition,
        reason_codes=reasons,
        screened_at=screened_at,
    )


def verify_contamination_audit(
    audit: ArtifactContaminationAudit,
    *,
    policy: ContaminationAuditPolicy,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
    identifiers: Iterable[IdentifierNeedle] = (),
) -> None:
    """Rebuild an audit from its bound bytes and fail if any field was forged or reordered."""

    expected = build_contamination_audit(
        audit_id=audit.audit_id,
        audit_input=audit.audit_input,
        policy=policy,
        public_payload=public_payload,
        comparison_payloads=comparison_payloads,
        judge_runs=audit.judge_runs,
        screened_at=audit.screened_at,
        identifiers=identifiers,
    )
    if canonical_json_bytes(audit) != canonical_json_bytes(expected):
        raise ContaminationAuditError('contamination audit does not match its bound inputs and fixed aggregation')


def make_audit_manifest(
    *,
    manifest_id: str,
    case_universe_sha256: str,
    policy: ContaminationAuditPolicy,
    audits: Iterable[ArtifactContaminationAudit],
) -> ContaminationAuditManifest:
    """Create a sorted complete audit inventory under one pinned policy."""

    ordered = tuple(sorted(audits, key=lambda audit: audit.audit_input.case_id))
    manifest = ContaminationAuditManifest(
        manifest_id=manifest_id,
        case_universe_sha256=case_universe_sha256,
        policy_sha256=model_sha256(policy),
        audits=ordered,
    )
    # Compute once here so callers cannot accidentally publish an unhashable manifest shape.
    audit_manifest_sha256(manifest)
    return manifest


def _aggregate(
    calibration_policy: CalibrationPolicy,
    candidates: tuple[DeterministicRetrievalCandidate, ...],
    runs: tuple[LlmAuditRun, ...],
) -> tuple[AuditDisposition, tuple[AuditReasonCode, ...]]:
    reasons: set[AuditReasonCode] = set()
    if any(candidate.kind == RetrievalKind.IDENTIFIER for candidate in candidates):
        reasons.add(AuditReasonCode.IDENTIFIER_MATCH)
    if any(candidate.kind == RetrievalKind.EXACT_NGRAM for candidate in candidates):
        reasons.add(AuditReasonCode.EXACT_NGRAM_MATCH)

    verdicts = {run.output.verdict for run in runs}
    if len(verdicts) > 1:
        reasons.add(AuditReasonCode.JUDGE_DISAGREEMENT)
    if JudgeVerdict.SUSPICIOUS in verdicts:
        reasons.add(AuditReasonCode.JUDGE_SUSPICIOUS)
    if JudgeVerdict.CONTAMINATED in verdicts:
        reasons.add(AuditReasonCode.JUDGE_CONTAMINATED)
    findings = tuple(finding for run in runs for finding in run.output.findings)
    if findings:
        reasons.add(AuditReasonCode.JUDGE_FINDING)
    if any(finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL} for finding in findings):
        reasons.add(AuditReasonCode.HIGH_SEVERITY_FINDING)
    if any(not _calibration_passes(run.output.calibration, calibration_policy) for run in runs):
        reasons.add(AuditReasonCode.CALIBRATION_BELOW_THRESHOLD)

    quarantine_reasons = {
        AuditReasonCode.HIGH_SEVERITY_FINDING,
        AuditReasonCode.IDENTIFIER_MATCH,
        AuditReasonCode.JUDGE_CONTAMINATED,
    }
    if reasons & quarantine_reasons:
        disposition = AuditDisposition.QUARANTINE
    elif reasons:
        disposition = AuditDisposition.MANUAL_REVIEW
    else:
        disposition = AuditDisposition.PASS
        reasons.add(AuditReasonCode.NO_DETECTED_SIGNALS)
    return disposition, tuple(sorted(reasons, key=lambda reason: reason.value))


def _calibration_passes(result: JudgeCalibrationResult, policy: CalibrationPolicy) -> bool:
    return (
        result.canary_count >= policy.minimum_canary_count
        and result.negative_control_count >= policy.minimum_negative_control_count
        and result.canary_recall >= policy.minimum_canary_recall
        and result.false_positive_rate <= policy.maximum_false_positive_rate
    )


def _validate_judge_runs(
    audit_input: ContaminationAuditInput,
    policy: ContaminationAuditPolicy,
    runs: tuple[LlmAuditRun, ...],
    *,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
) -> None:
    expected = {judge.judge_id: judge for judge in policy.judges}
    actual = {run.judge_id: run for run in runs}
    if set(actual) != set(expected) or len(actual) != len(runs):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ContaminationAuditError(
            f'audit must contain exactly one run for every pinned judge; missing={missing}, extra={extra}'
        )
    input_sha256 = model_sha256(audit_input)
    comparisons_by_id = {artifact.artifact_id: artifact for artifact in audit_input.comparison_artifacts}
    for judge_id, judge in expected.items():
        run = actual[judge_id]
        if run.judge_pin_sha256 != model_sha256(judge):
            raise ContaminationAuditError(f'judge run {judge_id} does not match its pinned judge')
        if run.audit_input_sha256 != input_sha256:
            raise ContaminationAuditError(f'judge run {judge_id} is bound to a different audit input')
        if run.request_sha256 != judge_request_sha256(audit_input, judge):
            raise ContaminationAuditError(f'judge run {judge_id} request commitment is invalid')
        for finding in run.output.findings:
            validate_exact_byte_span(public_payload, audit_input.public_artifact, finding.public_span)
            if finding.reference_span is not None:
                binding = comparisons_by_id.get(finding.reference_span.artifact_id)
                if binding is None:
                    raise ContaminationAuditError(
                        f'judge finding {finding.finding_id} references an unbound comparison artifact'
                    )
                validate_exact_byte_span(
                    comparison_payloads[binding.artifact_id],
                    binding,
                    finding.reference_span,
                )


def _validate_materials(
    audit_input: ContaminationAuditInput,
    public_payload: bytes,
    comparison_payloads: Mapping[str, bytes],
) -> None:
    _require_artifact(public_payload, audit_input.public_artifact)
    expected = {artifact.artifact_id: artifact for artifact in audit_input.comparison_artifacts}
    if set(comparison_payloads) != set(expected):
        missing = sorted(set(expected) - set(comparison_payloads))
        extra = sorted(set(comparison_payloads) - set(expected))
        raise ContaminationAuditError(f'comparison artifact inventory mismatch; missing={missing}, extra={extra}')
    for artifact_id, binding in expected.items():
        _require_artifact(comparison_payloads[artifact_id], binding)


def _require_artifact(payload: bytes, binding: ArtifactBinding) -> None:
    if not isinstance(payload, bytes):
        raise ContaminationAuditError(f'artifact {binding.artifact_id} must be bytes')
    if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
        raise ContaminationAuditError(
            f'artifact {binding.artifact_id} does not match its hash and byte-count commitment'
        )


def _decode_utf8(payload: bytes, artifact_id: str) -> str:
    try:
        return payload.decode('utf-8', errors='strict')
    except UnicodeDecodeError as error:
        raise ContaminationAuditError(f'artifact {artifact_id} is not valid UTF-8') from error


def _token_byte_offsets(text: str) -> tuple[tuple[int, int], ...]:
    byte_offsets = [0]
    for character in text:
        byte_offsets.append(byte_offsets[-1] + len(character.encode('utf-8')))
    return tuple((byte_offsets[match.start()], byte_offsets[match.end()]) for match in _TOKEN_PATTERN.finditer(text))


def _find_all(haystack: bytes, needle: bytes) -> Iterable[int]:
    if not needle:
        return
    offset = 0
    while True:
        found = haystack.find(needle, offset)
        if found < 0:
            return
        yield found
        offset = found + 1


def _make_candidate(
    *,
    kind: RetrievalKind,
    public_binding: ArtifactBinding,
    public_start: int,
    reference_binding: ArtifactBinding,
    reference_start: int,
    quote: str,
    token_count: int | None,
    identifier_id: str | None,
) -> DeterministicRetrievalCandidate:
    byte_count = len(quote.encode('utf-8'))
    public_span = ExactByteSpan(
        artifact_id=public_binding.artifact_id,
        artifact_sha256=public_binding.sha256,
        start_byte=public_start,
        end_byte=public_start + byte_count,
        quote=quote,
    )
    reference_span = ExactByteSpan(
        artifact_id=reference_binding.artifact_id,
        artifact_sha256=reference_binding.sha256,
        start_byte=reference_start,
        end_byte=reference_start + byte_count,
        quote=quote,
    )
    identity = {
        'kind': kind.value,
        'public_span': public_span.model_dump(mode='json'),
        'reference_span': reference_span.model_dump(mode='json'),
        'token_count': token_count,
        'identifier_id': identifier_id,
    }
    return DeterministicRetrievalCandidate(
        candidate_id=_sha256(canonical_json_bytes(identity)),
        kind=kind,
        public_span=public_span,
        reference_span=reference_span,
        token_count=token_count,
        identifier_id=identifier_id,
    )


def _require_candidate_limit(
    candidates: Mapping[str, DeterministicRetrievalCandidate],
    policy: ContaminationAuditPolicy,
) -> None:
    if len(candidates) > policy.retrieval.maximum_candidates:
        raise ContaminationAuditError(
            'deterministic retrieval exceeded maximum_candidates; refusing to truncate audit signals'
        )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
