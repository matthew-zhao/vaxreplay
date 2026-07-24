"""Canonical contracts for promoting witnessed operational captures.

These schemas describe a provenance-preserving bridge, not a complete Tier A
benchmark admission.  Promotion establishes that selected raw captures, source-
specific verification, and deterministic normalized outputs are bound together.
Case-universe sealing, decision-time sealing, and outcome isolation remain separate
release gates.
"""

from __future__ import annotations

import enum
import hashlib
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.plan_selection import PlanSelectionPolicyBinding
from vaxreplay.operations.schema import CaptureJobSpec, LedgerCheckpoint, aware_utc, job_spec_sha256
from vaxreplay.operations.witness import ExternalWitnessMethod, WitnessPolicyBinding

CAPTURE_PROMOTION_SCHEMA_VERSION = 'vaxreplay.capture-promotion.v0.6'
CAPTURE_INDEX_SCHEMA_VERSION = 'vaxreplay.capture-index.v0.6'
SOURCE_VERIFICATION_SCHEMA_VERSION = 'vaxreplay.source-verification.v0.3'
PROMOTION_SCOPE_POLICY_SCHEMA_VERSION = 'vaxreplay.promotion-scope-policy.v0.1'
PRE_CAPTURE_PLAN_SCHEMA_VERSION = 'vaxreplay.pre-capture-plan.v0.4'
PROMOTION_HANDOFF_SCHEMA_VERSION = 'vaxreplay.promotion-handoff.v0.6'
SCOPE_PRECOMMIT_PROMOTION_BINDING_SCHEMA_VERSION = 'vaxreplay.scope-precommit-promotion-binding.v0.2'
SOURCE_RECORD_BINDING_SCHEMA_VERSION = 'vaxreplay.source-record-binding.v0.1'
SOURCE_RECORD_DISPOSITION_SCHEMA_VERSION = 'vaxreplay.source-record-disposition.v0.1'
HERMETIC_EXECUTION_PROMOTION_BINDING_SCHEMA_VERSION = 'vaxreplay.hermetic-execution-promotion-binding.v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$'
_REASON_PATTERN = r'^[a-z][a-z0-9_]{0,99}$'
_PORTABLE_COMPONENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._@+-]{0,199}$')
_WINDOWS_RESERVED_COMPONENTS = {
    'aux',
    'con',
    'nul',
    'prn',
    *(f'com{ordinal}' for ordinal in range(1, 10)),
    *(f'lpt{ordinal}' for ordinal in range(1, 10)),
}


class PromotionIntegrityError(ValueError):
    """A capture cannot be promoted without weakening a committed invariant."""


class AuthoritativeReleaseBasis(str, enum.Enum):
    """Allowlisted publisher-side bases for ``source_release_at``.

    Generic HTTP ``Date`` and ``Last-Modified`` headers are intentionally absent.
    They are transport/cache metadata and cannot establish scientific release time.
    """

    SOURCE_SIGNED_METADATA = 'source_signed_metadata'
    SOURCE_VERSION_MANIFEST = 'source_version_manifest'
    SOURCE_API_PUBLICATION_FIELD = 'source_api_publication_field'
    PUBLISHER_RELEASE_RECORD = 'publisher_release_record'


class NormalizedOutputRole(str, enum.Enum):
    CANDIDATE_RECORDS = 'candidate_records'
    EVIDENCE_RECORDS = 'evidence_records'
    AUXILIARY = 'auxiliary'


def _portable_path(value: str, *, prefix: str | None = None) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or '\\' in value
        or ':' in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.is_absolute()
        or '..' in path.parts
        or path.as_posix() != value
        or any(part in {'', '.'} for part in path.parts)
        or any(
            not _PORTABLE_COMPONENT.fullmatch(part)
            or part.endswith('.')
            or part.casefold().split('.', 1)[0] in _WINDOWS_RESERVED_COMPONENTS
            for part in path.parts
        )
    ):
        raise ValueError('artifact paths must use safe normalized relative POSIX components')
    if prefix is not None and not value.startswith(f'{prefix}/'):
        raise ValueError(f'artifact path must remain below {prefix}/')
    return value


class PromotionFileBinding(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _portable_path(value)


class PromotionSourceScope(StrictModel):
    """Out-of-band source/job/time scope that prevents checkpoint cherry-picking."""

    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    job_spec_sha256s: tuple[str, ...] = Field(min_length=1)
    scheduled_from: datetime
    scheduled_through: datetime

    @field_validator('job_spec_sha256s')
    @classmethod
    def validate_job_specs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('scope job revisions must be sorted and unique')
        if any(len(item) != 64 or any(character not in '0123456789abcdef' for character in item) for item in value):
            raise ValueError('scope job revisions must be lowercase SHA-256 digests')
        return value

    @field_validator('scheduled_from', 'scheduled_through')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'scope schedule timestamp')

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.scheduled_through < self.scheduled_from:
            raise ValueError('scope scheduled_through cannot predate scheduled_from')
        return self


class PromotionScopePolicy(StrictModel):
    """Pinned V0 selection policy supplied independently at both build and load.

    V0 deliberately selects every successful static-HTTPS run for each listed job
    revision and schedule window.  It cannot express content-based exclusions, so an
    organizer cannot silently inspect a run and omit it from the portable promotion.
    """

    schema_version: Literal['vaxreplay.promotion-scope-policy.v0.1'] = PROMOTION_SCOPE_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    checkpoint_created_at_not_before: datetime
    checkpoint_created_at_not_after: datetime
    sources: tuple[PromotionSourceScope, ...] = Field(min_length=1)
    successful_run_strategy: Literal['all_scheduled_slots_must_succeed_and_select'] = (
        'all_scheduled_slots_must_succeed_and_select'
    )

    @field_validator('checkpoint_created_at_not_before', 'checkpoint_created_at_not_after')
    @classmethod
    def validate_checkpoint_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'scope checkpoint timestamp')

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, value: tuple[PromotionSourceScope, ...]) -> tuple[PromotionSourceScope, ...]:
        source_ids = tuple(item.source_id for item in value)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError('promotion source scopes must use sorted unique source IDs')
        all_job_specs = tuple(job for item in value for job in item.job_spec_sha256s)
        if len(all_job_specs) != len(set(all_job_specs)):
            raise ValueError('a job revision may belong to only one promotion source scope')
        return value

    @model_validator(mode='after')
    def validate_checkpoint_window(self) -> Self:
        if self.checkpoint_created_at_not_after < self.checkpoint_created_at_not_before:
            raise ValueError('scope checkpoint window is inverted')
        return self


class PrecommittedHermeticExecution(StrictModel):
    """Exact isolation and receipt authority frozen before any captured content."""

    sandbox_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    trusted_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    seccomp_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    authority_id: str = Field(pattern=_SAFE_ID_PATTERN)
    signing_key_id: str = Field(pattern=_SAFE_ID_PATTERN)
    runtime_id: Literal['docker-oci'] = 'docker-oci'
    network_disabled: Literal[True] = True
    read_only_root: Literal[True] = True
    no_host_mounts: Literal[True] = True


class PrecommittedSourceVerifier(StrictModel):
    """Exact source-verifier identity frozen before any scheduled capture."""

    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier_version: str = Field(pattern=_SAFE_ID_PATTERN)
    implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    hermetic_execution: PrecommittedHermeticExecution | None = None


class PrecommittedAdapter(StrictModel):
    """Exact normalization implementation and policy frozen before capture."""

    adapter_id: str = Field(pattern=_SAFE_ID_PATTERN)
    adapter_version: str = Field(pattern=_SAFE_ID_PATTERN)
    implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    hermetic_execution: PrecommittedHermeticExecution | None = None
    allowed_exclusion_reason_codes: tuple[str, ...] = ()

    @field_validator('allowed_exclusion_reason_codes')
    @classmethod
    def validate_allowed_exclusion_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('precommitted exclusion reason codes must be sorted and unique')
        if any(
            not code or len(code) > 100 or not code[0].islower() or not code.replace('_', '').isalnum()
            for code in value
        ):
            raise ValueError('precommitted exclusion reason codes must be lowercase identifiers')
        return value


class PreCapturePlan(StrictModel):
    """Selection and transformation protocol committed before the first slot.

    The plan deliberately commits the later capture-witness policy as well as the
    source verifiers and normalization adapter.  Otherwise an organizer could keep
    the source/job/time scope fixed while choosing transformations after seeing the
    captured content.
    """

    schema_version: Literal['vaxreplay.pre-capture-plan.v0.4'] = PRE_CAPTURE_PLAN_SCHEMA_VERSION
    scope_policy: PromotionScopePolicy
    selection_policy: PlanSelectionPolicyBinding
    capture_witness_policy: WitnessPolicyBinding
    source_verifiers: tuple[PrecommittedSourceVerifier, ...] = Field(min_length=1)
    adapter: PrecommittedAdapter

    @field_validator('source_verifiers')
    @classmethod
    def validate_source_verifiers(
        cls,
        value: tuple[PrecommittedSourceVerifier, ...],
    ) -> tuple[PrecommittedSourceVerifier, ...]:
        source_ids = tuple(item.source_id for item in value)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError('pre-capture source verifiers must use sorted unique source IDs')
        return value

    @model_validator(mode='after')
    def validate_source_coverage(self) -> Self:
        if {item.source_id for item in self.source_verifiers} != {item.source_id for item in self.scope_policy.sources}:
            raise ValueError('pre-capture source verifiers must exactly cover the scope sources')
        return self


class PromotedRawArtifactBinding(StrictModel):
    role: str = Field(pattern=r'^[a-z][a-z0-9._-]{0,127}$')
    file: PromotionFileBinding
    first_recorded_at: datetime
    stored_event_sequence: int = Field(ge=1)
    stored_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    attached_event_sequence: int = Field(ge=1)
    attached_event_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('first_recorded_at')
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'first_recorded_at')

    @model_validator(mode='after')
    def validate_event_order(self) -> Self:
        if self.stored_event_sequence >= self.attached_event_sequence:
            raise ValueError('raw artifact must be stored before it is attached')
        return self


class PromotedCaptureBinding(StrictModel):
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    job_id: str = Field(pattern=_SAFE_ID_PATTERN)
    collector_id: str = Field(pattern=_SAFE_ID_PATTERN)
    job_spec: CaptureJobSpec
    job_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    scheduled_for: datetime
    attempt_started_at: datetime
    captured_at: datetime
    job_registered_event_sequence: int = Field(ge=1)
    job_registered_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_registered_event_sequence: int = Field(ge=1)
    run_registered_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    started_event_sequence: int = Field(ge=1)
    started_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    succeeded_event_sequence: int = Field(ge=1)
    succeeded_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifacts: tuple[PromotedRawArtifactBinding, ...] = Field(min_length=1)

    @field_validator('scheduled_for', 'attempt_started_at', 'captured_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'capture timestamp')

    @field_validator('artifacts')
    @classmethod
    def validate_artifacts(
        cls,
        value: tuple[PromotedRawArtifactBinding, ...],
    ) -> tuple[PromotedRawArtifactBinding, ...]:
        roles = tuple(binding.role for binding in value)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError('promoted capture artifact roles must be sorted and unique')
        paths = tuple(binding.file.path for binding in value)
        if len(paths) != len(set(paths)):
            raise ValueError('promoted raw artifact paths must be unique')
        return value

    @model_validator(mode='after')
    def validate_lifecycle(self) -> Self:
        if (
            self.job_spec.job_id != self.job_id
            or self.job_spec.collector_id != self.collector_id
            or job_spec_sha256(self.job_spec) != self.job_spec_sha256
        ):
            raise ValueError('promoted capture does not bind its exact immutable job specification')
        if self.attempt_started_at > self.captured_at:
            raise ValueError('capture cannot complete before its attempt starts')
        if not (
            self.job_registered_event_sequence
            < self.run_registered_event_sequence
            < self.started_event_sequence
            < self.succeeded_event_sequence
        ):
            raise ValueError('job, run, attempt start, and success events must use lifecycle order')
        for artifact in self.artifacts:
            if not (
                artifact.stored_event_sequence < self.succeeded_event_sequence
                and self.started_event_sequence < artifact.attached_event_sequence < self.succeeded_event_sequence
            ):
                raise ValueError(
                    'selected artifacts must be stored before success and attached after start before success'
                )
        return self


class SuccessfulRunDisposition(StrictModel):
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    logical_run_id: str = Field(pattern=r'^run-[0-9a-f]{64}$')
    succeeded_event_sequence: int = Field(ge=1)
    succeeded_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition: Literal['selected', 'excluded']
    reason_code: str | None = Field(default=None, pattern=_REASON_PATTERN)

    @model_validator(mode='after')
    def validate_reason(self) -> Self:
        if (self.disposition == 'excluded') != (self.reason_code is not None):
            raise ValueError('exactly excluded successful runs require a reason_code')
        return self


class SourceVerifierIdentity(StrictModel):
    verifier_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier_version: str = Field(pattern=_SAFE_ID_PATTERN)
    implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)


class SourceRecordBinding(StrictModel):
    """One verifier-enumerated record backed by a selected body artifact.

    ``source_record_sha256`` is defined by the independently pinned verifier;
    ``source_artifact_sha256`` separately binds the exact captured body from
    which the record was verified.
    """

    schema_version: Literal['vaxreplay.source-record-binding.v0.1'] = SOURCE_RECORD_BINDING_SCHEMA_VERSION
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    source_record_id: str = Field(min_length=1, max_length=1024)
    source_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_locator: str = Field(min_length=1, max_length=4096)


class NormalizedRecordReference(StrictModel):
    """Exact identity and content commitment for one normalized JSONL row."""

    episode_id: str = Field(min_length=1, max_length=1024)
    record_id: str = Field(min_length=1, max_length=1024)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)


class SourceRecordDisposition(StrictModel):
    """Exhaustive normalization disposition for one verified source record."""

    schema_version: Literal['vaxreplay.source-record-disposition.v0.1'] = SOURCE_RECORD_DISPOSITION_SCHEMA_VERSION
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    source_record_id: str = Field(min_length=1, max_length=1024)
    source_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition: Literal['normalized', 'excluded']
    candidate_record_refs: tuple[NormalizedRecordReference, ...] = ()
    evidence_record_refs: tuple[NormalizedRecordReference, ...] = ()
    reason_code: str | None = Field(default=None, pattern=_REASON_PATTERN)

    @field_validator('candidate_record_refs', 'evidence_record_refs')
    @classmethod
    def validate_record_refs(
        cls,
        value: tuple[NormalizedRecordReference, ...],
    ) -> tuple[NormalizedRecordReference, ...]:
        keys = tuple((item.episode_id, item.record_id) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('normalized record references must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        has_refs = bool(self.candidate_record_refs or self.evidence_record_refs)
        if self.disposition == 'normalized':
            if self.reason_code is not None or not has_refs:
                raise ValueError('normalized source records require row references and cannot have a reason_code')
        elif self.reason_code is None or has_refs:
            raise ValueError('excluded source records require only a reason_code')
        return self


class AuthoritativeSourceRelease(StrictModel):
    source_release_at: datetime
    basis: AuthoritativeReleaseBasis
    authority_locator: str = Field(min_length=1, max_length=2048)
    authority_field: str = Field(min_length=1, max_length=256)
    evidence_attempt_id: str = Field(pattern=r'^attempt-[0-9a-f]{32}$')
    evidence_role: str = Field(pattern=r'^[a-z][a-z0-9._-]{0,127}$')
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_source_record_id: str = Field(min_length=1, max_length=1024)
    evidence_source_record_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('source_release_at')
    @classmethod
    def validate_source_release_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'source_release_at')

    @field_validator('authority_locator', 'authority_field')
    @classmethod
    def reject_transport_headers(cls, value: str) -> str:
        normalized = ''.join(character for character in value.lower() if character.isalnum())
        semantic_release_field = 'publicationdate' in normalized or 'releasedate' in normalized
        transport_date = normalized.endswith('date') and any(
            token in normalized
            for token in ('cache', 'header', 'http', 'origin', 'request', 'response', 'server', 'transport')
        )
        if 'lastmodified' in normalized or (
            not semantic_release_field
            and (
                normalized in {'date', 'httpdate', 'responsedate', 'transportdate'}
                or normalized.endswith('httpdate')
                or transport_date
            )
        ):
            raise ValueError('generic HTTP Date/Last-Modified metadata cannot establish source_release_at')
        return value


class SourceVerificationResult(StrictModel):
    """Trusted source-specific verifier output recorded verbatim in promotion."""

    schema_version: Literal['vaxreplay.source-verification.v0.3'] = SOURCE_VERIFICATION_SCHEMA_VERSION
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier: SourceVerifierIdentity
    verifier_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_attempt_ids: tuple[str, ...] = Field(min_length=1)
    source_release: AuthoritativeSourceRelease
    verified_capture_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_source_record_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_source_record_count: int = Field(gt=0)
    source_enumeration_complete: Literal[True] = True
    admissible: Literal[True] = True
    result_codes: tuple[str, ...] = Field(min_length=1)

    @field_validator('verified_attempt_ids')
    @classmethod
    def validate_attempt_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('verified attempt IDs must be sorted and unique')
        if any(not attempt_id.startswith('attempt-') or len(attempt_id) != 40 for attempt_id in value):
            raise ValueError('verified attempt IDs must use operational attempt identity syntax')
        return value

    @field_validator('result_codes')
    @classmethod
    def validate_result_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('source-verifier result codes must be sorted and unique')
        for code in value:
            if not code or len(code) > 100 or not code[0].islower() or not code.replace('_', '').isalnum():
                raise ValueError('source-verifier result codes must be lowercase identifiers')
        return value

    @model_validator(mode='after')
    def validate_release_evidence(self) -> Self:
        if self.source_release.evidence_attempt_id not in self.verified_attempt_ids:
            raise ValueError('source release evidence must come from a source-verified selected attempt')
        return self


class SourceVerificationBinding(StrictModel):
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier_policy: PromotionFileBinding
    verifier_implementation: PromotionFileBinding
    verifier_execution_environment: PromotionFileBinding
    verified_records: PromotionFileBinding
    result: SourceVerificationResult
    result_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_dispositions: tuple[SuccessfulRunDisposition, ...] = Field(min_length=1)

    @field_validator('run_dispositions')
    @classmethod
    def validate_dispositions(
        cls,
        value: tuple[SuccessfulRunDisposition, ...],
    ) -> tuple[SuccessfulRunDisposition, ...]:
        keys = tuple((item.succeeded_event_sequence, item.attempt_id) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('successful-run dispositions must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if self.source_id != self.result.source_id:
            raise ValueError('source verification result belongs to a different source')
        if self.verifier_policy.sha256 != self.result.verifier_policy_sha256:
            raise ValueError('source verification result does not bind its exact policy')
        if self.verifier_implementation.sha256 != self.result.verifier.implementation_sha256:
            raise ValueError('source verification result does not bind its copied verifier implementation')
        if self.verifier_execution_environment.sha256 != self.result.verifier.execution_environment_sha256:
            raise ValueError('source verification result does not bind its copied verifier execution environment')
        if (
            not self.verified_records.path.startswith('sources/')
            or not self.verified_records.path.endswith('/verified-records.jsonl')
            or self.verified_records.sha256 != self.result.verified_source_record_inventory_sha256
        ):
            raise ValueError('source verification result does not bind its canonical record inventory')
        if hashlib.sha256(canonical_json_bytes(self.result)).hexdigest() != self.result_sha256:
            raise ValueError('result_sha256 does not bind the canonical source verification result')
        if any(item.source_id != self.source_id for item in self.run_dispositions):
            raise ValueError('run disposition belongs to a different source')
        selected = tuple(sorted(item.attempt_id for item in self.run_dispositions if item.disposition == 'selected'))
        if selected != self.result.verified_attempt_ids:
            raise ValueError('selected run dispositions must exactly match source-verified attempts')
        return self


class AdapterInputInventoryBinding(StrictModel):
    source_id: str = Field(pattern=_SAFE_ID_PATTERN)
    capture_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_record_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_record_count: int = Field(gt=0)
    source_verification_result_sha256: str = Field(pattern=_SHA256_PATTERN)


class AdapterBinding(StrictModel):
    adapter_id: str = Field(pattern=_SAFE_ID_PATTERN)
    adapter_version: str = Field(pattern=_SAFE_ID_PATTERN)
    implementation: PromotionFileBinding
    policy: PromotionFileBinding
    execution_environment: PromotionFileBinding
    input_inventories: tuple[AdapterInputInventoryBinding, ...] = Field(min_length=1)
    allowed_exclusion_reason_codes: tuple[str, ...] = ()
    disposition_count: int = Field(gt=0)
    determinism_check_runs: Literal[2] = 2
    repeated_outputs_identical: Literal[True] = True

    @field_validator('input_inventories')
    @classmethod
    def validate_input_inventories(
        cls,
        value: tuple[AdapterInputInventoryBinding, ...],
    ) -> tuple[AdapterInputInventoryBinding, ...]:
        source_ids = tuple(item.source_id for item in value)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError('adapter input inventories must have sorted unique source IDs')
        return value

    @field_validator('allowed_exclusion_reason_codes')
    @classmethod
    def validate_allowed_exclusion_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('allowed exclusion reason codes must be sorted and unique')
        if any(
            not code or len(code) > 100 or not code[0].islower() or not code.replace('_', '').isalnum()
            for code in value
        ):
            raise ValueError('allowed exclusion reason codes must be lowercase identifiers')
        return value


class HermeticExecutionPromotionBinding(StrictModel):
    """Exact signed OCI execution artifacts retained by a portable promotion."""

    schema_version: Literal['vaxreplay.hermetic-execution-promotion-binding.v0.2'] = (
        HERMETIC_EXECUTION_PROMOTION_BINDING_SCHEMA_VERSION
    )
    subject_id: str = Field(pattern=_SAFE_ID_PATTERN)
    purpose: Literal['source_verifier', 'adapter']
    invocation_id: str = Field(pattern=_SAFE_ID_PATTERN)
    invocation_index: int = Field(ge=0)
    request: PromotionFileBinding
    response: PromotionFileBinding
    receipt: PromotionFileBinding
    image_inspection: PromotionFileBinding
    sandbox_policy: PromotionFileBinding
    seccomp_profile: PromotionFileBinding
    trusted_public_key: PromotionFileBinding
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_byte_count: int = Field(gt=0)
    authority_id: str = Field(pattern=_SAFE_ID_PATTERN)
    signing_key_id: str = Field(pattern=_SAFE_ID_PATTERN)
    issued_at: datetime

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'hermetic execution issued_at')

    @model_validator(mode='after')
    def validate_artifact_paths(self) -> Self:
        files = (
            self.request,
            self.response,
            self.receipt,
            self.image_inspection,
            self.sandbox_policy,
            self.seccomp_profile,
            self.trusted_public_key,
        )
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)) or any(not path.startswith('hermetic/') for path in paths):
            raise ValueError('hermetic execution artifacts require unique paths below hermetic/')
        return self


class NormalizedOutputBinding(StrictModel):
    role: NormalizedOutputRole
    source_ids: tuple[str, ...] = Field(min_length=1)
    file: PromotionFileBinding

    @field_validator('source_ids')
    @classmethod
    def validate_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('normalized output source IDs must be sorted and unique')
        return value

    @model_validator(mode='after')
    def validate_output_path(self) -> Self:
        _portable_path(self.file.path, prefix='normalized')
        return self


class ExternalWitnessPromotionBinding(StrictModel):
    witness_manifest: PromotionFileBinding
    checkpoint_file: PromotionFileBinding
    proof_file: PromotionFileBinding
    policy: PromotionFileBinding
    trust_policy: PromotionFileBinding
    verifier_implementation: PromotionFileBinding
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime
    authority_id: str = Field(pattern=_SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=_SAFE_ID_PATTERN)
    method: ExternalWitnessMethod
    policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    trust_policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    trust_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_id: str = Field(pattern=_SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_byte_count: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_pinned_materials(self) -> Self:
        if (
            self.policy.sha256 != self.policy_sha256
            or self.trust_policy.sha256 != self.trust_policy_sha256
            or self.verifier_implementation.sha256 != self.verifier_implementation_sha256
            or self.checkpoint_file.sha256 != self.checkpoint_sha256
            or self.proof_file.sha256 != self.proof_sha256
            or self.proof_file.byte_count != self.proof_byte_count
        ):
            raise ValueError('witness receipt metadata does not bind its copied policy and verifier materials')
        return self

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'witnessed_at')


class ScopePrecommitPromotionBinding(StrictModel):
    """Recursive binding to the complete copied pre-capture archive."""

    schema_version: Literal['vaxreplay.scope-precommit-promotion-binding.v0.2'] = (
        SCOPE_PRECOMMIT_PROMOTION_BINDING_SCHEMA_VERSION
    )
    archive_manifest: PromotionFileBinding
    archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    pre_capture_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_id: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=_SAFE_ID_PATTERN)
    # Canonical SHA-256 of the complete PlanSelectionPolicyBinding.  This is
    # distinct from the registry policy artifact digest carried inside it.
    selection_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_selection_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime
    store_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_through_sequence: int = Field(ge=1)
    checkpoint_through_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime

    @field_validator('witnessed_at', 'selected_at_upper_bound')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'scope precommit timestamp')

    @model_validator(mode='after')
    def validate_archive(self) -> Self:
        if self.archive_manifest.path != 'scope/precommit/scope-precommit.json':
            raise ValueError('scope precommit manifest must use its canonical promotion path')
        if self.archive_manifest.sha256 != self.archive_sha256:
            raise ValueError('scope precommit archive digest must bind its exact manifest bytes')
        return self


class _PlanSelectionIdentity(Protocol):
    @property
    def campaign_id(self) -> str: ...

    @property
    def selection_key(self) -> str: ...

    @property
    def selection_policy_sha256(self) -> str: ...

    @property
    def selection_policy_artifact_sha256(self) -> str: ...

    @property
    def plan_selection_commitment_sha256(self) -> str: ...

    @property
    def selection_manifest_sha256(self) -> str: ...

    @property
    def selected_at_upper_bound(self) -> datetime: ...


def _plan_selection_identity(
    value: _PlanSelectionIdentity,
) -> tuple[str, str, str, str, str, str, datetime]:
    """Return the exact selection identity repeated across promotion envelopes."""

    return (
        value.campaign_id,
        value.selection_key,
        value.selection_policy_sha256,
        value.selection_policy_artifact_sha256,
        value.plan_selection_commitment_sha256,
        value.selection_manifest_sha256,
        value.selected_at_upper_bound,
    )


class CaptureIndex(StrictModel):
    """Canonical downstream source-capture artifact consumed by release builders."""

    schema_version: Literal['vaxreplay.capture-index.v0.6'] = CAPTURE_INDEX_SCHEMA_VERSION
    promotion_id: str = Field(pattern=_SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_selection_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint: LedgerCheckpoint
    ledger_prefix: PromotionFileBinding
    scope_policy: PromotionFileBinding
    scope_policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    scope_precommit: ScopePrecommitPromotionBinding
    witness: ExternalWitnessPromotionBinding
    registered_jobs: PromotionFileBinding
    captures: tuple[PromotedCaptureBinding, ...] = Field(min_length=1)
    source_verifications: tuple[SourceVerificationBinding, ...] = Field(min_length=1)
    adapter: AdapterBinding
    hermetic_executions: tuple[HermeticExecutionPromotionBinding, ...] = ()
    normalization_dispositions: PromotionFileBinding
    normalized_outputs: tuple[NormalizedOutputBinding, ...] = Field(min_length=1)
    capture_provenance_promoted: Literal[True] = True
    tier_a_benchmark_release_established: Literal[False] = False

    @field_validator('selected_at_upper_bound')
    @classmethod
    def validate_selected_at_upper_bound(cls, value: datetime) -> datetime:
        return aware_utc(value, 'plan selection selected_at_upper_bound')

    @field_validator('captures')
    @classmethod
    def validate_captures(cls, value: tuple[PromotedCaptureBinding, ...]) -> tuple[PromotedCaptureBinding, ...]:
        keys = tuple((item.source_id, item.succeeded_event_sequence, item.attempt_id) for item in value)
        if (
            keys != tuple(sorted(keys))
            or len({item.attempt_id for item in value}) != len(value)
            or len({item.logical_run_id for item in value}) != len(value)
        ):
            raise ValueError('promoted captures must be sorted and use unique attempt IDs')
        return value

    @field_validator('source_verifications')
    @classmethod
    def validate_source_verifications(
        cls,
        value: tuple[SourceVerificationBinding, ...],
    ) -> tuple[SourceVerificationBinding, ...]:
        ids = tuple(item.source_id for item in value)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError('source verifications must have sorted unique source IDs')
        return value

    @field_validator('hermetic_executions')
    @classmethod
    def validate_hermetic_executions(
        cls,
        value: tuple[HermeticExecutionPromotionBinding, ...],
    ) -> tuple[HermeticExecutionPromotionBinding, ...]:
        keys = tuple((item.purpose, item.subject_id, item.invocation_index) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('hermetic execution bindings must use sorted unique invocation identities')
        all_paths = tuple(
            file.path
            for item in value
            for file in (
                item.request,
                item.response,
                item.receipt,
                item.image_inspection,
                item.sandbox_policy,
                item.seccomp_profile,
                item.trusted_public_key,
            )
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError('hermetic execution artifacts cannot share archive paths')
        return value

    @field_validator('normalized_outputs')
    @classmethod
    def validate_outputs(cls, value: tuple[NormalizedOutputBinding, ...]) -> tuple[NormalizedOutputBinding, ...]:
        keys = tuple((item.role.value, item.file.path) for item in value)
        if keys != tuple(sorted(keys)) or len({item.file.path for item in value}) != len(value):
            raise ValueError('normalized outputs must be sorted and use unique paths')
        reserved = [item.role for item in value if item.role is not NormalizedOutputRole.AUXILIARY]
        if len(reserved) != len(set(reserved)):
            raise ValueError('candidate and evidence output roles may each appear at most once')
        return value

    @model_validator(mode='after')
    def validate_cross_bindings(self) -> Self:
        if self.checkpoint_sha256 != hashlib.sha256(canonical_json_bytes(self.checkpoint)).hexdigest():
            raise ValueError('checkpoint_sha256 does not bind the canonical checkpoint')
        if self.witness.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError('external witness does not bind the promoted checkpoint')
        if self.ledger_prefix.path != 'ledger/events.jsonl':
            raise ValueError('portable ledger prefix must use ledger/events.jsonl')
        if self.scope_policy.path != 'scope/policy.json':
            raise ValueError('promotion scope policy must use scope/policy.json')
        if self.scope_precommit.scope_policy_sha256 != self.scope_policy.sha256:
            raise ValueError('scope precommit does not bind the promoted scope policy')
        if _plan_selection_identity(self) != _plan_selection_identity(self.scope_precommit):
            raise ValueError('capture index plan selection differs from its scope precommit')
        if self.scope_precommit.store_id != self.checkpoint.store_id:
            raise ValueError('scope precommit and capture checkpoint belong to different stores')
        if self.scope_precommit.checkpoint_through_sequence >= self.checkpoint.through_sequence:
            raise ValueError('scope precommit checkpoint must be a strict prefix of the capture checkpoint')
        if self.registered_jobs.path != 'ledger/jobs.jsonl':
            raise ValueError('portable scoped job inventory must use ledger/jobs.jsonl')
        if self.witness.witnessed_at < self.checkpoint.created_at:
            raise ValueError('external witnessed_at cannot predate checkpoint creation')
        source_ids = {item.source_id for item in self.source_verifications}
        if {item.source_id for item in self.captures} != source_ids:
            raise ValueError('promoted captures and source verifications must cover the same sources')
        if {item.source_id for item in self.adapter.input_inventories} != source_ids:
            raise ValueError('adapter inputs and source verifications must cover the same sources')
        if self.hermetic_executions:
            source_executions = tuple(item for item in self.hermetic_executions if item.purpose == 'source_verifier')
            adapter_executions = tuple(item for item in self.hermetic_executions if item.purpose == 'adapter')
            if {item.subject_id for item in source_executions} != source_ids or len(source_executions) != len(
                source_ids
            ):
                raise ValueError('hermetic source-verifier executions must exactly cover verified sources')
            if (
                len(adapter_executions) != 2
                or {item.subject_id for item in adapter_executions} != {self.adapter.adapter_id}
                or {item.invocation_index for item in adapter_executions} != {0, 1}
            ):
                raise ValueError('hermetic adapter evidence requires exactly two deterministic executions')
        if self.normalization_dispositions.path != 'normalized/dispositions.jsonl':
            raise ValueError('normalization dispositions must use normalized/dispositions.jsonl')
        if any(set(output.source_ids) - source_ids for output in self.normalized_outputs):
            raise ValueError('normalized output references an unverified source')
        if self.adapter.disposition_count != sum(item.source_record_count for item in self.adapter.input_inventories):
            raise ValueError('adapter disposition count must exhaustively cover verified source records')
        if any(item.succeeded_event_sequence > self.checkpoint.through_sequence for item in self.captures):
            raise ValueError('promoted capture success is outside the witnessed checkpoint prefix')
        for verification in self.source_verifications:
            source_captures = tuple(capture for capture in self.captures if capture.source_id == verification.source_id)
            capture_attempt_ids = tuple(sorted(capture.attempt_id for capture in source_captures))
            if capture_attempt_ids != verification.result.verified_attempt_ids:
                raise ValueError('source verification must exactly cover its promoted captures')
            capture_by_attempt = {capture.attempt_id: capture for capture in source_captures}
            for disposition in verification.run_dispositions:
                if disposition.succeeded_event_sequence > self.checkpoint.through_sequence:
                    raise ValueError('successful-run disposition lies outside the witnessed checkpoint prefix')
                if disposition.disposition == 'selected':
                    capture = capture_by_attempt.get(disposition.attempt_id)
                    if capture is None or (
                        disposition.logical_run_id,
                        disposition.succeeded_event_sequence,
                        disposition.succeeded_event_sha256,
                    ) != (
                        capture.logical_run_id,
                        capture.succeeded_event_sequence,
                        capture.succeeded_event_sha256,
                    ):
                        raise ValueError('selected run disposition does not bind its promoted capture event')
            adapter_input = next(
                item for item in self.adapter.input_inventories if item.source_id == verification.source_id
            )
            if (
                adapter_input.source_verification_result_sha256 != verification.result_sha256
                or adapter_input.capture_inventory_sha256 != verification.result.verified_capture_inventory_sha256
                or adapter_input.source_record_inventory_sha256
                != verification.result.verified_source_record_inventory_sha256
                or adapter_input.source_record_count != verification.result.verified_source_record_count
            ):
                raise ValueError('adapter input does not bind its source verification result')
            release = verification.result.source_release
            if not release.evidence_role.startswith('body.'):
                raise ValueError('authoritative source release evidence must come from a captured body artifact')
            evidence_capture = capture_by_attempt[release.evidence_attempt_id]
            evidence_artifact = next(
                (artifact for artifact in evidence_capture.artifacts if artifact.role == release.evidence_role),
                None,
            )
            if evidence_artifact is None or evidence_artifact.file.sha256 != release.evidence_sha256:
                raise ValueError('authoritative source release evidence is not a selected raw capture artifact')
            if release.source_release_at > self.witness.witnessed_at:
                raise ValueError('authoritative source release cannot postdate witnessed capture')
            if release.source_release_at > evidence_capture.captured_at:
                raise ValueError('authoritative source release cannot postdate its evidence capture')
        if any(capture.captured_at > self.witness.witnessed_at for capture in self.captures):
            raise ValueError('selected capture completion cannot postdate its external checkpoint witness')
        if self.scope_precommit.witnessed_at >= min(capture.scheduled_for for capture in self.captures):
            raise ValueError('scope precommit witness must predate the first selected schedule slot')
        if self.selected_at_upper_bound >= min(capture.scheduled_for for capture in self.captures):
            raise ValueError('plan selection authorization must be configured before the first covered slot')
        return self


class CapturePromotionManifest(StrictModel):
    schema_version: Literal['vaxreplay.capture-promotion.v0.6'] = CAPTURE_PROMOTION_SCHEMA_VERSION
    promotion_id: str = Field(pattern=_SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_selection_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime
    created_at: datetime
    capture_index: PromotionFileBinding
    scope_precommit: PromotionFileBinding
    files: tuple[PromotionFileBinding, ...] = Field(min_length=1)
    atomic_install: Literal[True] = True
    content_hash_bound: Literal[True] = True

    @field_validator('created_at', 'selected_at_upper_bound')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'created_at')

    @field_validator('files')
    @classmethod
    def validate_files(cls, value: tuple[PromotionFileBinding, ...]) -> tuple[PromotionFileBinding, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('promotion files must be sorted and unique')
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError('promotion file paths cannot collide under case folding')
        if any(path.casefold() == 'promotion.json' for path in paths):
            raise ValueError('promotion manifest cannot recursively bind itself')
        return value

    @model_validator(mode='after')
    def validate_capture_index(self) -> Self:
        matching = tuple(item for item in self.files if item.path == self.capture_index.path)
        if matching != (self.capture_index,):
            raise ValueError('capture index must be included exactly once in promotion files')
        matching_precommit = tuple(item for item in self.files if item.path == self.scope_precommit.path)
        if matching_precommit != (self.scope_precommit,):
            raise ValueError('scope precommit manifest must be included exactly once in promotion files')
        if self.scope_precommit.path != 'scope/precommit/scope-precommit.json':
            raise ValueError('scope precommit manifest uses the wrong promotion path')
        return self


class PromotionHandoffDescriptor(StrictModel):
    """Compact, canonical pointer from a decision package to a full promotion.

    The descriptor carries enough of the verified promotion identity to make a
    prospective package structurally self-checking.  It is deliberately *not* an
    admission capability: ``full_promotion_reverification_required`` records that
    an admission verifier must resolve the independently retained promotion root
    and rerun every witness, source-verifier, and adapter check.
    """

    schema_version: Literal['vaxreplay.promotion-handoff.v0.6'] = PROMOTION_HANDOFF_SCHEMA_VERSION
    promotion_id: str = Field(pattern=_SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_selection_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_at_upper_bound: datetime
    promotion_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    promotion_created_at: datetime
    capture_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    capture_index: CaptureIndex
    candidate_output: NormalizedOutputBinding
    evidence_output: NormalizedOutputBinding
    maximum_source_release_at: datetime
    maximum_captured_at: datetime
    witnessed_at: datetime
    full_promotion_reverification_required: Literal[True] = True

    @field_validator(
        'promotion_created_at',
        'maximum_source_release_at',
        'maximum_captured_at',
        'witnessed_at',
        'selected_at_upper_bound',
    )
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'promotion handoff timestamp')

    @model_validator(mode='after')
    def validate_handoff(self) -> Self:
        if self.promotion_id != self.capture_index.promotion_id:
            raise ValueError('promotion handoff belongs to a different capture index')
        if self.capture_index_sha256 != capture_index_sha256(self.capture_index):
            raise ValueError('promotion handoff does not bind its canonical capture index')
        if _plan_selection_identity(self) != _plan_selection_identity(self.capture_index):
            raise ValueError('promotion handoff plan selection differs from its capture index')
        outputs = {
            output.role: output
            for output in self.capture_index.normalized_outputs
            if output.role is not NormalizedOutputRole.AUXILIARY
        }
        if set(outputs) != {
            NormalizedOutputRole.CANDIDATE_RECORDS,
            NormalizedOutputRole.EVIDENCE_RECORDS,
        }:
            raise ValueError('promotion handoff requires exactly one candidate and evidence output')
        if (
            self.candidate_output.role is not NormalizedOutputRole.CANDIDATE_RECORDS
            or self.candidate_output != outputs[NormalizedOutputRole.CANDIDATE_RECORDS]
            or self.evidence_output.role is not NormalizedOutputRole.EVIDENCE_RECORDS
            or self.evidence_output != outputs[NormalizedOutputRole.EVIDENCE_RECORDS]
        ):
            raise ValueError('promotion handoff does not bind the exact normalized decision inputs')
        maximum_source_release_at = max(
            binding.result.source_release.source_release_at for binding in self.capture_index.source_verifications
        )
        maximum_captured_at = max(capture.captured_at for capture in self.capture_index.captures)
        if self.maximum_source_release_at != maximum_source_release_at:
            raise ValueError('promotion handoff source-release bound is not conservative and exact')
        if self.maximum_captured_at != maximum_captured_at:
            raise ValueError('promotion handoff capture-time bound is not conservative and exact')
        if self.witnessed_at != self.capture_index.witness.witnessed_at:
            raise ValueError('promotion handoff witness time differs from its capture index')
        if self.promotion_created_at < max(self.maximum_captured_at, self.witnessed_at):
            raise ValueError('promotion handoff creation predates its capture or witness')
        return self


def source_verification_result_sha256(result: SourceVerificationResult) -> str:
    return hashlib.sha256(canonical_json_bytes(result)).hexdigest()


def capture_index_sha256(index: CaptureIndex) -> str:
    return hashlib.sha256(canonical_json_bytes(index)).hexdigest()


def capture_promotion_sha256(manifest: CapturePromotionManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def promotion_handoff_descriptor_sha256(descriptor: PromotionHandoffDescriptor) -> str:
    return hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()
