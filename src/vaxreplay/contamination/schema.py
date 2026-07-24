"""Strict, provider-neutral contracts for retrospective contamination screening.

These contracts describe a conservative screen, not a proof that model weights or benchmark
artifacts are uncontaminated.  Every ``pass`` result therefore retains an explicit residual-risk
flag and a literal prohibition on claiming absence of contamination.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

ARTIFACT_BINDING_SCHEMA_VERSION = 'vaxreplay.contamination-artifact-binding.v0.1'
AUDIT_INPUT_SCHEMA_VERSION = 'vaxreplay.contamination-audit-input.v0.1'
AUDIT_POLICY_SCHEMA_VERSION = 'vaxreplay.contamination-audit-policy.v0.1'
JUDGE_OUTPUT_SCHEMA_VERSION = 'vaxreplay.contamination-judge-output.v0.1'
JUDGE_RUN_SCHEMA_VERSION = 'vaxreplay.contamination-judge-run.v0.1'
ARTIFACT_AUDIT_SCHEMA_VERSION = 'vaxreplay.contamination-artifact-audit.v0.1'
AUDIT_MANIFEST_SCHEMA_VERSION = 'vaxreplay.contamination-audit-manifest.v0.1'
FIXED_AGGREGATION_VERSION = 'vaxreplay.contamination-fixed-aggregation.v0.1'
PASS_INTERPRETATION = 'no_signal_detected_under_pinned_screen; residual_contamination_remains_possible'


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{name} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RetrievalKind(str, enum.Enum):
    EXACT_NGRAM = 'exact_ngram'
    IDENTIFIER = 'identifier'


class FindingKind(str, enum.Enum):
    FUTURE_FACT_OVERLAP = 'future_fact_overlap'
    OUTCOME_DISCLOSURE = 'outcome_disclosure'
    SOURCE_IDENTIFIER = 'source_identifier'
    SOURCE_REIDENTIFICATION = 'source_reidentification'
    PROMPT_INJECTION = 'prompt_injection'
    OTHER = 'other'


class FindingSeverity(str, enum.Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class JudgeVerdict(str, enum.Enum):
    CLEAR = 'clear'
    SUSPICIOUS = 'suspicious'
    CONTAMINATED = 'contaminated'


class AuditDisposition(str, enum.Enum):
    PASS = 'pass'
    MANUAL_REVIEW = 'manual_review'
    QUARANTINE = 'quarantine'


class AuditReasonCode(str, enum.Enum):
    CALIBRATION_BELOW_THRESHOLD = 'calibration_below_threshold'
    EXACT_NGRAM_MATCH = 'exact_ngram_match'
    HIGH_SEVERITY_FINDING = 'high_severity_finding'
    IDENTIFIER_MATCH = 'identifier_match'
    JUDGE_CONTAMINATED = 'judge_contaminated'
    JUDGE_DISAGREEMENT = 'judge_disagreement'
    JUDGE_FINDING = 'judge_finding'
    JUDGE_SUSPICIOUS = 'judge_suspicious'
    NO_DETECTED_SIGNALS = 'no_detected_signals'


class ArtifactBinding(StrictModel):
    """Hash and byte-count commitment to one exact audit artifact."""

    schema_version: Literal['vaxreplay.contamination-artifact-binding.v0.1'] = ARTIFACT_BINDING_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)


class ExactByteSpan(StrictModel):
    """A UTF-8 quote whose offsets are byte offsets into an exact bound artifact."""

    artifact_id: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start_byte: int = Field(ge=0)
    end_byte: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=4_096)

    @model_validator(mode='after')
    def validate_byte_length(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError('exact byte span end must be greater than start')
        if self.end_byte - self.start_byte != len(self.quote.encode('utf-8')):
            raise ValueError('exact byte span offsets must equal the UTF-8 quote byte length')
        return self


class ContaminationAuditInput(StrictModel):
    """Exact public payload and private comparison namespace presented to every judge."""

    schema_version: Literal['vaxreplay.contamination-audit-input.v0.1'] = AUDIT_INPUT_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    decision_package_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    episode_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    public_artifact: ArtifactBinding
    comparison_artifacts: tuple[ArtifactBinding, ...] = Field(min_length=1)

    @field_validator('comparison_artifacts')
    @classmethod
    def validate_comparison_artifacts(cls, value: tuple[ArtifactBinding, ...]) -> tuple[ArtifactBinding, ...]:
        artifact_ids = tuple(artifact.artifact_id for artifact in value)
        if artifact_ids != tuple(sorted(artifact_ids)) or len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('comparison artifacts must have unique IDs in sorted order')
        return value

    @model_validator(mode='after')
    def validate_namespaces(self) -> Self:
        if self.public_artifact.artifact_id in {artifact.artifact_id for artifact in self.comparison_artifacts}:
            raise ValueError('public and comparison artifact IDs must be disjoint')
        return self


class ExactRetrievalConfig(StrictModel):
    """Pinned configuration for literal, case-sensitive retrieval."""

    ngram_tokens: int = Field(default=8, ge=2, le=64)
    minimum_ngram_bytes: int = Field(default=32, ge=4, le=4_096)
    maximum_candidates: int = Field(default=2_000, ge=1, le=100_000)
    case_sensitive: Literal[True] = True
    exact_bytes_only: Literal[True] = True


class IdentifierNeedle(StrictModel):
    """A literal private identifier to search for in the public payload."""

    identifier_id: str = Field(min_length=1)
    identifier_type: str = Field(min_length=1)
    value: str = Field(min_length=3, max_length=512)
    reference_artifact_id: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_value(self) -> Self:
        if self.value != self.value.strip() or '\x00' in self.value:
            raise ValueError('identifier values must be trimmed and cannot contain NUL')
        return self


class DeterministicRetrievalCandidate(StrictModel):
    """One exact public/reference byte match nominated for audit."""

    candidate_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    kind: RetrievalKind
    public_span: ExactByteSpan
    reference_span: ExactByteSpan
    token_count: int | None = Field(default=None, ge=2, le=64)
    identifier_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def validate_kind(self) -> Self:
        if self.kind == RetrievalKind.EXACT_NGRAM:
            if self.token_count is None or self.identifier_id is not None:
                raise ValueError('exact n-gram candidates require token_count and no identifier_id')
        elif self.identifier_id is None or self.token_count is not None:
            raise ValueError('identifier candidates require identifier_id and no token_count')
        if self.public_span.quote != self.reference_span.quote:
            raise ValueError('deterministic retrieval candidates must bind identical exact bytes')
        return self


class ContaminationFinding(StrictModel):
    """A structured judge finding grounded in an exact public byte span."""

    finding_id: str = Field(min_length=1)
    kind: FindingKind
    severity: FindingSeverity
    public_span: ExactByteSpan
    reference_span: ExactByteSpan | None = None
    detail: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def validate_reference(self) -> Self:
        if self.kind in {FindingKind.FUTURE_FACT_OVERLAP, FindingKind.OUTCOME_DISCLOSURE}:
            if self.reference_span is None:
                raise ValueError('future-fact and outcome findings require an exact reference span')
        return self


class JudgeCalibrationResult(StrictModel):
    canary_count: int = Field(ge=1)
    canary_detected_count: int = Field(ge=0)
    negative_control_count: int = Field(ge=1)
    false_positive_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.canary_detected_count > self.canary_count:
            raise ValueError('detected canaries cannot exceed the canary count')
        if self.false_positive_count > self.negative_control_count:
            raise ValueError('false positives cannot exceed the negative-control count')
        return self

    @property
    def canary_recall(self) -> float:
        return self.canary_detected_count / self.canary_count

    @property
    def false_positive_rate(self) -> float:
        return self.false_positive_count / self.negative_control_count


class LlmJudgeOutput(StrictModel):
    """Canonical structured output retained instead of free-form reasoning traces."""

    schema_version: Literal['vaxreplay.contamination-judge-output.v0.1'] = JUDGE_OUTPUT_SCHEMA_VERSION
    verdict: JudgeVerdict
    findings: tuple[ContaminationFinding, ...] = ()
    calibration: JudgeCalibrationResult

    @model_validator(mode='after')
    def validate_output(self) -> Self:
        finding_ids = tuple(finding.finding_id for finding in self.findings)
        if finding_ids != tuple(sorted(finding_ids)) or len(finding_ids) != len(set(finding_ids)):
            raise ValueError('judge findings must have unique IDs in sorted order')
        if self.verdict == JudgeVerdict.CLEAR and self.findings:
            raise ValueError('a clear judge verdict cannot contain findings')
        if self.verdict != JudgeVerdict.CLEAR and not self.findings:
            raise ValueError('a non-clear judge verdict requires at least one finding')
        if self.verdict == JudgeVerdict.CONTAMINATED and not any(
            finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL} for finding in self.findings
        ):
            raise ValueError('a contaminated verdict requires a high or critical finding')
        return self


class PinnedLlmJudge(StrictModel):
    """Organizer commitment to one judge identity, prompt, and configuration."""

    judge_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    system_fingerprint: str = Field(min_length=1)
    system_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    prompt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class CalibrationPolicy(StrictModel):
    minimum_canary_count: int = Field(default=5, ge=1, le=100_000)
    minimum_negative_control_count: int = Field(default=5, ge=1, le=100_000)
    minimum_canary_recall: float = Field(default=0.8, ge=0.0, le=1.0, allow_inf_nan=False)
    maximum_false_positive_rate: float = Field(default=0.2, ge=0.0, le=1.0, allow_inf_nan=False)


class ContaminationAuditPolicy(StrictModel):
    """Pinned inputs to the fixed, non-configurable disposition rule."""

    schema_version: Literal['vaxreplay.contamination-audit-policy.v0.1'] = AUDIT_POLICY_SCHEMA_VERSION
    policy_id: str = Field(min_length=1)
    aggregation_version: Literal['vaxreplay.contamination-fixed-aggregation.v0.1'] = FIXED_AGGREGATION_VERSION
    retrieval: ExactRetrievalConfig = Field(default_factory=ExactRetrievalConfig)
    calibration: CalibrationPolicy = Field(default_factory=CalibrationPolicy)
    judges: tuple[PinnedLlmJudge, ...] = Field(min_length=2)
    pass_interpretation: Literal['no_signal_detected_under_pinned_screen; residual_contamination_remains_possible'] = (
        PASS_INTERPRETATION
    )

    @field_validator('judges')
    @classmethod
    def validate_judges(cls, value: tuple[PinnedLlmJudge, ...]) -> tuple[PinnedLlmJudge, ...]:
        judge_ids = tuple(judge.judge_id for judge in value)
        if judge_ids != tuple(sorted(judge_ids)) or len(judge_ids) != len(set(judge_ids)):
            raise ValueError('at least two pinned judges must have unique IDs in sorted order')
        pin_hashes = tuple(model_sha256(judge) for judge in value)
        if len(pin_hashes) != len(set(pin_hashes)):
            raise ValueError('pinned judge commitments must be distinct')
        return value


class LlmAuditRun(StrictModel):
    """Canonical record of one structured output from one pinned judge."""

    schema_version: Literal['vaxreplay.contamination-judge-run.v0.1'] = JUDGE_RUN_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    judge_id: str = Field(min_length=1)
    judge_pin_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    audit_input_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_bytes: int = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    output: LlmJudgeOutput

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode='after')
    def validate_run(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('judge run cannot finish before it starts')
        output_bytes = canonical_json_bytes(self.output)
        if len(output_bytes) != self.output_bytes or _sha256(output_bytes) != self.output_sha256:
            raise ValueError('judge output does not match its canonical hash and byte count')
        return self


class ArtifactContaminationAudit(StrictModel):
    """One fully bound screen with a fixed disposition and explicit residual risk."""

    schema_version: Literal['vaxreplay.contamination-artifact-audit.v0.1'] = ARTIFACT_AUDIT_SCHEMA_VERSION
    audit_id: str = Field(min_length=1)
    audit_input: ContaminationAuditInput
    audit_input_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    deterministic_candidates: tuple[DeterministicRetrievalCandidate, ...] = ()
    judge_runs: tuple[LlmAuditRun, ...] = Field(min_length=2)
    disposition: AuditDisposition
    reason_codes: tuple[AuditReasonCode, ...] = Field(min_length=1)
    screened_at: datetime
    residual_contamination_possible: Literal[True] = True
    proves_absence_of_contamination: Literal[False] = False

    @field_validator('screened_at')
    @classmethod
    def validate_screened_at(cls, value: datetime) -> datetime:
        return _aware(value, 'screened_at')

    @model_validator(mode='after')
    def validate_audit(self) -> Self:
        if self.audit_input_sha256 != model_sha256(self.audit_input):
            raise ValueError('audit input hash does not bind the embedded input')
        candidate_ids = tuple(candidate.candidate_id for candidate in self.deterministic_candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('deterministic candidates must have unique IDs in sorted order')
        judge_ids = tuple(run.judge_id for run in self.judge_runs)
        if judge_ids != tuple(sorted(judge_ids)) or len(judge_ids) != len(set(judge_ids)):
            raise ValueError('judge runs must have unique judge IDs in sorted order')
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda reason: reason.value)):
            raise ValueError('audit reason codes must be unique and sorted')
        if self.disposition == AuditDisposition.PASS:
            if self.reason_codes != (AuditReasonCode.NO_DETECTED_SIGNALS,):
                raise ValueError('pass disposition requires only the no-detected-signals reason')
        elif AuditReasonCode.NO_DETECTED_SIGNALS in self.reason_codes:
            raise ValueError('non-pass dispositions cannot claim no detected signals')
        return self


class ContaminationAuditManifest(StrictModel):
    """Complete, canonical audit inventory for a sealed case universe."""

    schema_version: Literal['vaxreplay.contamination-audit-manifest.v0.1'] = AUDIT_MANIFEST_SCHEMA_VERSION
    manifest_id: str = Field(min_length=1)
    case_universe_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    inventory_complete: Literal[True] = True
    audits: tuple[ArtifactContaminationAudit, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        case_ids = tuple(audit.audit_input.case_id for audit in self.audits)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError('audit manifest cases must be unique and sorted by case_id')
        if any(audit.policy_sha256 != self.policy_sha256 for audit in self.audits):
            raise ValueError('every audit must use the manifest policy')
        return self


def model_sha256(value: StrictModel) -> str:
    return _sha256(canonical_json_bytes(value))


def audit_manifest_sha256(value: ContaminationAuditManifest) -> str:
    return model_sha256(value)
