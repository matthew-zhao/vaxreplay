"""Official, cohort-atomic post-outcome scoring for prospective releases.

This module is the only Tier A scoring entrypoint.  It never accepts a response
object from its caller: responses are recovered by ordinal from the globally
registered attempt completion and its authenticated run artifact.  It also
requires one outcome disposition for every case in the pre-outcome universe,
executes the frozen disposition policy through trusted verifier code, and uses
the complete preeligible universe as the leaderboard denominator.

The helpers in :mod:`vaxreplay.prospective_finalizer` are deliberately limited
to low-level verification and adaptation.  Their caller-constructible return
values are not verification capabilities, and that module exposes no scoring
entrypoint.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublication
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, jsonl_text
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    CaseUniverseDisposition,
    CaseUniverseManifest,
    case_selection_audit_sha256,
    case_universe_sha256,
    validate_case_selection_inventory,
)
from vaxreplay.case_schema import (
    RANKING_REWARD_VERSION,
    IssueCode,
    ScoreStatus,
    ScoreVector,
    StrictModel,
    ValidationIssue,
)
from vaxreplay.operations.prospective_release_approval_identity import (
    TierAProspectiveReleaseApprovalReplay,
)
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_finalizer import (
    ProspectiveFinalizationError,
    adapt_prospective_submission,
    finalize_prospective_episode,
)
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    load_prospective_cohort_release,
)
from vaxreplay.prospective_schema import (
    ProspectiveAttemptPolicy,
    ProspectiveEpisodeBinding,
    ProspectiveFinalizationBinding,
    ProspectiveSubmission,
    prospective_attempt_policy_sha256,
)
from vaxreplay.ranking_schema import ScoreVectorV1
from vaxreplay.runner.prospective_attempt_reservation import (
    LoadedProspectiveAttemptCompletion,
    LoadedProspectiveAttemptReservation,
    LoadedProspectiveAttemptStartAuthorization,
    ProspectiveAttemptCompletionStatus,
    ProspectiveAttemptRegistryVerifier,
    ProspectiveAttemptStartVerifier,
    load_prospective_attempt_completion,
    load_prospective_attempt_reservation,
    load_prospective_attempt_start_authorization,
    prospective_attempt_completion_manifest_sha256,
    prospective_attempt_completion_target_sha256,
    prospective_attempt_reservation_manifest_sha256,
    prospective_attempt_reservation_target_sha256,
    prospective_attempt_start_authorization_manifest_sha256,
    prospective_attempt_start_target_sha256,
)
from vaxreplay.runner.prospective_release_seal import (
    LoadedProspectiveReleaseSeal,
    ProspectiveReleaseTimestampVerifier,
    load_prospective_release_seal,
    prospective_release_seal_manifest_sha256,
    prospective_release_seal_target_sha256,
)
from vaxreplay.runner.schema import (
    EpisodeRunStatus,
    RunnerPolicy,
    SystemSubmissionManifest,
)
from vaxreplay.scoring import make_submission_evaluator
from vaxreplay.temporal_schema import (
    PROTOCOL_ARTIFACT_NAMES,
    TemporalAdmissionEnvelope,
    TemporalReceiptVerifier,
    model_sha256,
)

PROSPECTIVE_COHORT_FINALIZATION_SCHEMA_VERSION = 'vaxreplay.prospective-cohort-finalization.v0.4'
PROSPECTIVE_COHORT_SCORE_REPORT_SCHEMA_VERSION = 'vaxreplay.prospective-cohort-score-report.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_PATH_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$'
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_TREE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_FILES = 100_000
_MAX_DIRECTORIES = 20_000


class ProspectiveCohortFinalizationIntegrityError(ValueError):
    """Raised when a cohort finalization is incomplete, changed, or misclassified."""


type CaseSelectionPolicyVerifier = Callable[
    [bytes, CaseUniverseManifest, CaseSelectionAudit, Mapping[str, bytes]],
    bool,
]


@dataclass(frozen=True)
class ProspectiveCaseFinalizationInput:
    """Private outcome-side inputs for one admitted case."""

    final_bundle: EpisodeBundle
    temporal_admission: TemporalAdmissionEnvelope
    receipt_artifacts: Mapping[str, bytes]
    protocol_artifacts: Mapping[str, bytes]
    raw_outcome_source: bytes
    label_derivation_audit: bytes


class ProspectiveFinalizationFileBinding(StrictModel):
    path: str = Field(pattern=_PATH_PATTERN)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        _safe_relative_path(value)
        return value


class ProspectiveDispositionCount(StrictModel):
    disposition: CaseSelectionDisposition
    count: int = Field(ge=0)


class ProspectiveFinalizedCaseBinding(StrictModel):
    case_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    directory: str = Field(pattern=r'^cases/[0-9]{6}$')
    decision_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    final_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    labels_sha256: str = Field(pattern=_SHA256_PATTERN)
    temporal_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    finalization_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_status: EpisodeRunStatus | None = None
    response_record_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    response_record_bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def validate_response_binding(self) -> Self:
        values = (self.response_status, self.response_record_sha256, self.response_record_bytes)
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError('case response status, hash, and byte count must be declared together')
        return self


class ProspectiveCohortEpisodeScore(StrictModel):
    case_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    response_status: EpisodeRunStatus | None = None
    response_record_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    score: ScoreVector | ScoreVectorV1


class ProspectiveCohortScoreReport(StrictModel):
    schema_version: Literal['vaxreplay.prospective-cohort-score-report.v0.1'] = (
        PROSPECTIVE_COHORT_SCORE_REPORT_SCHEMA_VERSION
    )
    denominator_policy: Literal['all_preeligible_cases_invalid_or_unscored_equal_zero'] = (
        'all_preeligible_cases_invalid_or_unscored_equal_zero'
    )
    denominator_case_count: int = Field(gt=0)
    admitted_case_count: int = Field(ge=0)
    valid_score_count: int = Field(ge=0)
    invalid_response_count: int = Field(ge=0)
    unscored_case_count: int = Field(ge=0)
    reward_sum: float = Field(ge=0.0, allow_inf_nan=False)
    reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    episodes: tuple[ProspectiveCohortEpisodeScore, ...]

    @model_validator(mode='after')
    def validate_aggregate(self) -> Self:
        if self.admitted_case_count != len(self.episodes):
            raise ValueError('admitted case count must match the episode score inventory')
        if self.valid_score_count + self.invalid_response_count != self.admitted_case_count:
            raise ValueError('every admitted case must be valid or retain an invalid response')
        if self.admitted_case_count + self.unscored_case_count != self.denominator_case_count:
            raise ValueError('fixed denominator must include every preeligible case')
        expected_reward = self.reward_sum / self.denominator_case_count
        if abs(self.reward - expected_reward) > 1e-12:
            raise ValueError('cohort reward must use the fixed preeligible-case denominator')
        if tuple(item.ordinal for item in self.episodes) != tuple(sorted(item.ordinal for item in self.episodes)):
            raise ValueError('episode scores must be sorted by challenge ordinal')
        return self


class ProspectiveCohortFinalizationManifest(StrictModel):
    """Private atomic identity for one complete, policy-verified scoring event."""

    schema_version: Literal['vaxreplay.prospective-cohort-finalization.v0.4'] = (
        PROSPECTIVE_COHORT_FINALIZATION_SCHEMA_VERSION
    )
    release_id: str = Field(min_length=1)
    purpose: Literal['official_benchmark', 'prospective_research']
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_cohort_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    registered_entry_id: str = Field(min_length=1)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_status: ProspectiveAttemptCompletionStatus
    run_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    responses_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    runner_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_universe_sha256: str = Field(pattern=_SHA256_PATTERN)
    verifier_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_selection_audit_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition_policy_verified: Literal[True] = True
    universe_case_count: int = Field(gt=0)
    denominator_case_count: int = Field(gt=0)
    disposition_counts: tuple[ProspectiveDispositionCount, ...]
    cases: tuple[ProspectiveFinalizedCaseBinding, ...]
    score_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    cohort_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    finalized_at: datetime
    files: tuple[ProspectiveFinalizationFileBinding, ...] = Field(min_length=6)

    @field_validator('finalized_at')
    @classmethod
    def validate_finalized_at(cls, value: datetime) -> datetime:
        return _aware(value, 'finalized_at')

    @field_validator('files')
    @classmethod
    def validate_files(
        cls,
        value: tuple[ProspectiveFinalizationFileBinding, ...],
    ) -> tuple[ProspectiveFinalizationFileBinding, ...]:
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('finalization file bindings must be unique and sorted by path')
        return value

    @field_validator('cases')
    @classmethod
    def validate_cases(
        cls,
        value: tuple[ProspectiveFinalizedCaseBinding, ...],
    ) -> tuple[ProspectiveFinalizedCaseBinding, ...]:
        keys = tuple((item.case_id, item.episode_id) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('finalized cases must be unique and sorted by case and episode ID')
        ordinals = tuple(item.ordinal for item in value)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError('finalized cases cannot reuse a challenge ordinal')
        return value

    @field_validator('disposition_counts')
    @classmethod
    def validate_disposition_counts(
        cls,
        value: tuple[ProspectiveDispositionCount, ...],
    ) -> tuple[ProspectiveDispositionCount, ...]:
        names = tuple(item.disposition.value for item in value)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError('disposition counts must be unique and sorted by disposition')
        return value

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        run_values = (self.run_receipt_sha256, self.responses_sha256)
        if self.completion_status == ProspectiveAttemptCompletionStatus.SUCCESS:
            if any(value is None for value in run_values):
                raise ValueError('successful finalization must bind the completed run and responses')
            if any(case.response_status is None for case in self.cases):
                raise ValueError('successful finalization must bind every admitted response by ordinal')
        elif any(value is not None for value in run_values):
            raise ValueError('failed attempts cannot bind a successful run')
        elif any(case.response_status is not None for case in self.cases):
            raise ValueError('failed attempts cannot bind case response records')
        if sum(item.count for item in self.disposition_counts) != self.universe_case_count:
            raise ValueError('disposition counts must cover the complete case universe')
        if self.denominator_case_count < len(self.cases):
            raise ValueError('fixed denominator cannot be smaller than admitted cases')
        return self


@dataclass(frozen=True)
class LoadedProspectiveFinalizedCase:
    case_id: str
    ordinal: int
    prospective_episode: ProspectiveEpisodeBinding
    final_bundle: EpisodeBundle
    temporal_admission: TemporalAdmissionEnvelope
    receipt_artifacts: Mapping[str, bytes]
    protocol_artifacts: Mapping[str, bytes]
    raw_outcome_source: bytes
    label_derivation_audit: bytes
    finalization: ProspectiveFinalizationBinding


@dataclass(frozen=True)
class LoadedProspectiveCohortFinalization:
    root: Path
    manifest: ProspectiveCohortFinalizationManifest
    manifest_sha256: str
    release: LoadedProspectiveCohortRelease
    release_seal: LoadedProspectiveReleaseSeal
    reservation: LoadedProspectiveAttemptReservation
    start_authorization: LoadedProspectiveAttemptStartAuthorization
    completion: LoadedProspectiveAttemptCompletion
    system: SystemSubmissionManifest
    runner_policy: RunnerPolicy
    attempt_policy: ProspectiveAttemptPolicy
    case_selection_audit: CaseSelectionAudit
    disposition_evidence: Mapping[str, bytes]
    cases: tuple[LoadedProspectiveFinalizedCase, ...]
    score_report: ProspectiveCohortScoreReport


def prospective_cohort_finalization_sha256(
    manifest: ProspectiveCohortFinalizationManifest,
) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_cohort_finalization(
    output_dir: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    completion: LoadedProspectiveAttemptCompletion,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    case_selection_audit: CaseSelectionAudit,
    disposition_evidence: Mapping[str, bytes],
    case_inputs: Mapping[str, ProspectiveCaseFinalizationInput],
    finalized_at: datetime,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
    temporal_receipt_verifier: TemporalReceiptVerifier,
    case_selection_policy_verifier: CaseSelectionPolicyVerifier,
) -> LoadedProspectiveCohortFinalization:
    """Build one private atomic scoring artifact after every case is disposed."""

    finalized_at = _aware(finalized_at, 'finalized_at')
    release, release_seal, reservation, start_authorization, completion = _fresh_chain(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        completion=completion,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
        registry_verifier=registry_verifier,
        start_verifier=start_verifier,
        expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        error_type=ValueError,
    )
    audit = _canonical_model_value(case_selection_audit, CaseSelectionAudit, 'case-selection audit')
    evidence = _validate_selection_policy(
        release,
        audit,
        disposition_evidence,
        verifier=case_selection_policy_verifier,
        error_type=ValueError,
    )
    admitted_records = tuple(
        record for record in audit.records if record.disposition == CaseSelectionDisposition.ADMITTED
    )
    if set(case_inputs) != {record.case_id for record in admitted_records}:
        raise ValueError('case inputs must exactly match every admitted case and no others')
    inputs = {case_id: _normalize_case_input(value) for case_id, value in case_inputs.items()}
    try:
        validate_case_selection_inventory(
            release.verified_admission.case_universe,
            audit,
            (inputs[record.case_id].final_bundle for record in admitted_records),
        )
    except ValueError as error:
        raise ValueError(f'case-selection inventory failed: {error}') from error

    target = _publication_target(output_dir)
    with AtomicDirectoryPublication.create(target) as publication:
        writer = _FinalizationPublicationWriter(publication)
        writer.write('system.json', canonical_json_bytes(system))
        writer.write('runner-policy.json', canonical_json_bytes(runner_policy))
        writer.write('attempt-policy.json', canonical_json_bytes(attempt_policy))
        writer.write('case-selection-audit.json', canonical_json_bytes(audit))
        for index, record in enumerate(audit.records):
            writer.write(f'dispositions/{index:06d}.bin', evidence[record.case_id])

        loaded_cases = _materialize_cases(
            writer,
            eventual_root=target,
            release=release,
            completion=completion,
            audit=audit,
            inputs=inputs,
            finalized_at=finalized_at,
            temporal_receipt_verifier=temporal_receipt_verifier,
        )
        score_report = _compute_score_report(release, completion, audit, loaded_cases)
        score_bytes = canonical_json_bytes(score_report)
        writer.write('score-report.json', score_bytes)
        file_bindings = writer.file_bindings()
        manifest = _build_manifest(
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            start_authorization=start_authorization,
            completion=completion,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            audit=audit,
            cases=loaded_cases,
            score_report=score_report,
            finalized_at=finalized_at,
            file_bindings=file_bindings,
        )
        writer.write('finalization.json', canonical_json_bytes(manifest))
        installed_root = publication.publish(root_mode=0o700)
        loaded = load_prospective_cohort_finalization(
            installed_root,
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            start_authorization=start_authorization,
            completion=completion,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
            temporal_receipt_verifier=temporal_receipt_verifier,
            case_selection_policy_verifier=case_selection_policy_verifier,
        )
        publication.commit()
        return loaded


def load_prospective_cohort_finalization(
    root: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    completion: LoadedProspectiveAttemptCompletion,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
    temporal_receipt_verifier: TemporalReceiptVerifier,
    case_selection_policy_verifier: CaseSelectionPolicyVerifier,
    expected_finalization_sha256: str | None = None,
) -> LoadedProspectiveCohortFinalization:
    """Freshly reverify every pre- and post-outcome trust boundary."""

    release, release_seal, reservation, start_authorization, completion = _fresh_chain(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        completion=completion,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
        registry_verifier=registry_verifier,
        start_verifier=start_verifier,
        expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        error_type=ProspectiveCohortFinalizationIntegrityError,
    )
    resolved = _resolve_root(root)
    before = _inventory_tree(resolved)
    manifest_bytes = _read_file(resolved / 'finalization.json', _MAX_MODEL_BYTES)
    manifest = _canonical_model_bytes(
        manifest_bytes,
        ProspectiveCohortFinalizationManifest,
        'prospective cohort finalization manifest',
    )
    manifest_sha256 = _sha256(manifest_bytes)
    if expected_finalization_sha256 is not None and manifest_sha256 != expected_finalization_sha256:
        raise ProspectiveCohortFinalizationIntegrityError(
            'prospective cohort finalization does not match its expected identity'
        )

    expected_files = {'finalization.json', *(binding.path for binding in manifest.files)}
    expected_directories = _parent_directories(expected_files)
    if before.files != expected_files or before.directories != expected_directories:
        missing_files = sorted(expected_files - before.files)
        extra_files = sorted(before.files - expected_files)
        missing_directories = sorted(expected_directories - before.directories)
        extra_directories = sorted(before.directories - expected_directories)
        raise ProspectiveCohortFinalizationIntegrityError(
            'finalization exact tree allowlist mismatch; '
            f'missing_files={missing_files}, extra_files={extra_files}, '
            f'missing_directories={missing_directories}, extra_directories={extra_directories}'
        )
    file_bytes: dict[str, bytes] = {}
    for binding in manifest.files:
        payload = _read_file(resolved / binding.path, _MAX_FILE_BYTES)
        if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
            raise ProspectiveCohortFinalizationIntegrityError(f'finalization artifact changed: {binding.path}')
        file_bytes[binding.path] = payload

    stored_system = _canonical_model_bytes(file_bytes['system.json'], SystemSubmissionManifest, 'system manifest')
    stored_policy = _canonical_model_bytes(file_bytes['runner-policy.json'], RunnerPolicy, 'runner policy')
    stored_attempt_policy = _canonical_model_bytes(
        file_bytes['attempt-policy.json'], ProspectiveAttemptPolicy, 'attempt policy'
    )
    if (stored_system, stored_policy, stored_attempt_policy) != (system, runner_policy, attempt_policy):
        raise ProspectiveCohortFinalizationIntegrityError(
            'finalization system or execution policy differs from the registered attempt'
        )
    audit = _canonical_model_bytes(file_bytes['case-selection-audit.json'], CaseSelectionAudit, 'case-selection audit')
    disposition_evidence = {
        record.case_id: file_bytes[f'dispositions/{index:06d}.bin'] for index, record in enumerate(audit.records)
    }
    disposition_evidence = _validate_selection_policy(
        release,
        audit,
        disposition_evidence,
        verifier=case_selection_policy_verifier,
        error_type=ProspectiveCohortFinalizationIntegrityError,
    )

    cases = _load_cases(
        resolved,
        file_bytes=file_bytes,
        manifest=manifest,
        release=release,
        completion=completion,
        audit=audit,
        temporal_receipt_verifier=temporal_receipt_verifier,
    )
    try:
        validate_case_selection_inventory(
            release.verified_admission.case_universe,
            audit,
            (case.final_bundle for case in cases),
        )
    except ValueError as error:
        raise ProspectiveCohortFinalizationIntegrityError(f'case-selection inventory failed: {error}') from error

    score_report = _canonical_model_bytes(
        file_bytes['score-report.json'], ProspectiveCohortScoreReport, 'cohort score report'
    )
    expected_report = _compute_score_report(release, completion, audit, cases)
    if score_report != expected_report:
        raise ProspectiveCohortFinalizationIntegrityError(
            'stored cohort score report differs from exact sealed-response scoring'
        )
    expected_manifest = _build_manifest(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        completion=completion,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        audit=audit,
        cases=cases,
        score_report=score_report,
        finalized_at=manifest.finalized_at,
        file_bindings=manifest.files,
    )
    if manifest != expected_manifest:
        raise ProspectiveCohortFinalizationIntegrityError('finalization manifest is bound to different verified inputs')
    after = _inventory_tree(resolved)
    if after != before:
        raise ProspectiveCohortFinalizationIntegrityError('finalization tree changed while it was being verified')
    return LoadedProspectiveCohortFinalization(
        root=resolved,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        completion=completion,
        system=stored_system,
        runner_policy=stored_policy,
        attempt_policy=stored_attempt_policy,
        case_selection_audit=audit,
        disposition_evidence=disposition_evidence,
        cases=cases,
        score_report=score_report,
    )


def score_prospective_cohort_finalization(
    finalization: LoadedProspectiveCohortFinalization,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    completion: LoadedProspectiveAttemptCompletion,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
    temporal_receipt_verifier: TemporalReceiptVerifier,
    case_selection_policy_verifier: CaseSelectionPolicyVerifier,
) -> ProspectiveCohortScoreReport:
    """Return an official score only after a complete fresh proof-chain reload."""

    loaded = load_prospective_cohort_finalization(
        finalization.root,
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
        completion=completion,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
        registry_verifier=registry_verifier,
        start_verifier=start_verifier,
        expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        temporal_receipt_verifier=temporal_receipt_verifier,
        case_selection_policy_verifier=case_selection_policy_verifier,
        expected_finalization_sha256=finalization.manifest_sha256,
    )
    if loaded.manifest.purpose != 'official_benchmark':
        raise ProspectiveCohortFinalizationIntegrityError(
            'official scoring requires a hermetic promotion-backed benchmark admission'
        )
    return loaded.score_report


def _fresh_chain(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    completion: LoadedProspectiveAttemptCompletion,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
    error_type: type[ValueError],
) -> tuple[
    LoadedProspectiveCohortRelease,
    LoadedProspectiveReleaseSeal,
    LoadedProspectiveAttemptReservation,
    LoadedProspectiveAttemptStartAuthorization,
    LoadedProspectiveAttemptCompletion,
]:
    try:
        fresh_release = load_prospective_cohort_release(
            release.root,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_release_sha256=release.release_sha256,
        )
        fresh_seal = load_prospective_release_seal(
            release_seal.root,
            release=fresh_release,
            submissions_open_at=release_seal.target.submissions_open_at,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            timestamp_verifier=release_timestamp_verifier,
        )
        fresh_reservation = load_prospective_attempt_reservation(
            reservation.root,
            release=fresh_release,
            release_seal=fresh_seal,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
        )
        fresh_start_authorization = load_prospective_attempt_start_authorization(
            start_authorization.root,
            release=fresh_release,
            release_seal=fresh_seal,
            reservation=fresh_reservation,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        )
        fresh_completion = load_prospective_attempt_completion(
            completion.root,
            release=fresh_release,
            release_seal=fresh_seal,
            reservation=fresh_reservation,
            start_authorization=fresh_start_authorization,
            system=system,
            runner_policy=runner_policy,
            attempt_policy=attempt_policy,
            receipt_key=receipt_key,
            expected_receipt_key_id=expected_receipt_key_id,
            decision_receipt_verifier=decision_receipt_verifier,
            case_universe_seal_verifier=case_universe_seal_verifier,
            source_capture_verifier=source_capture_verifier,
            expected_approval_report_sha256=expected_approval_report_sha256,
            approval_replay=approval_replay,
            release_timestamp_verifier=release_timestamp_verifier,
            registry_verifier=registry_verifier,
            start_verifier=start_verifier,
            expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
            run=completion.run,
        )
    except ValueError as error:
        raise error_type(f'prospective proof-chain reverification failed: {error}') from error
    if fresh_seal.manifest_sha256 != release_seal.manifest_sha256:
        raise error_type('release seal changed during finalization')
    if fresh_reservation.manifest_sha256 != reservation.manifest_sha256:
        raise error_type('attempt reservation changed during finalization')
    if fresh_start_authorization.manifest_sha256 != start_authorization.manifest_sha256:
        raise error_type('attempt start authorization changed during finalization')
    if fresh_completion.manifest_sha256 != completion.manifest_sha256:
        raise error_type('attempt completion changed during finalization')
    return (
        fresh_release,
        fresh_seal,
        fresh_reservation,
        fresh_start_authorization,
        fresh_completion,
    )


def _validate_selection_policy(
    release: LoadedProspectiveCohortRelease,
    audit: CaseSelectionAudit,
    disposition_evidence: Mapping[str, bytes],
    *,
    verifier: CaseSelectionPolicyVerifier,
    error_type: type[ValueError],
) -> dict[str, bytes]:
    expected_policy_sha256 = _sha256(release.verifier_policy)
    if audit.selection_policy_sha256 != expected_policy_sha256:
        raise error_type('case-selection audit does not use the pre-outcome verifier policy')
    expected_case_ids = {record.case_id for record in audit.records}
    if set(disposition_evidence) != expected_case_ids:
        raise error_type('disposition evidence must cover every audited case exactly once')
    normalized: dict[str, bytes] = {}
    for case_id, payload in disposition_evidence.items():
        if not isinstance(payload, bytes) or not payload:
            raise error_type(f'disposition evidence for {case_id} must be non-empty bytes')
        if len(payload) > _MAX_FILE_BYTES:
            raise error_type(f'disposition evidence for {case_id} exceeds the size limit')
        normalized[case_id] = payload
    try:
        verified = verifier(
            release.verifier_policy,
            release.verified_admission.case_universe,
            audit,
            normalized,
        )
    except Exception as error:
        raise error_type(f'case-selection policy verifier failed: {error}') from error
    if not verified:
        raise error_type('case-selection policy verifier rejected the exhaustive disposition audit')
    return normalized


def _normalize_case_input(value: ProspectiveCaseFinalizationInput) -> ProspectiveCaseFinalizationInput:
    if not isinstance(value, ProspectiveCaseFinalizationInput):
        raise TypeError('case inputs must be ProspectiveCaseFinalizationInput values')
    value.final_bundle.validate_integrity()
    temporal = _canonical_model_value(
        value.temporal_admission,
        TemporalAdmissionEnvelope,
        'temporal admission',
    )
    receipts = _normalize_bytes_mapping(value.receipt_artifacts, 'receipt artifact')
    protocols = _normalize_bytes_mapping(value.protocol_artifacts, 'protocol artifact')
    if set(protocols) != set(PROTOCOL_ARTIFACT_NAMES):
        raise ValueError('case input requires exactly the three decision protocol artifacts')
    if not value.raw_outcome_source or not value.label_derivation_audit:
        raise ValueError('case input outcome source and derivation audit must be non-empty bytes')
    return ProspectiveCaseFinalizationInput(
        final_bundle=value.final_bundle,
        temporal_admission=temporal,
        receipt_artifacts=receipts,
        protocol_artifacts=protocols,
        raw_outcome_source=value.raw_outcome_source,
        label_derivation_audit=value.label_derivation_audit,
    )


def _normalize_bytes_mapping(values: Mapping[str, bytes], label: str) -> dict[str, bytes]:
    normalized: dict[str, bytes] = {}
    for key, value in values.items():
        if not key or not isinstance(value, bytes) or not value:
            raise ValueError(f'{label} names and bytes must be non-empty')
        if len(value) > _MAX_FILE_BYTES:
            raise ValueError(f'{label} {key} exceeds the size limit')
        normalized[key] = value
    return normalized


def _materialize_cases(
    writer: _FinalizationPublicationWriter,
    *,
    eventual_root: Path,
    release: LoadedProspectiveCohortRelease,
    completion: LoadedProspectiveAttemptCompletion,
    audit: CaseSelectionAudit,
    inputs: Mapping[str, ProspectiveCaseFinalizationInput],
    finalized_at: datetime,
    temporal_receipt_verifier: TemporalReceiptVerifier,
) -> tuple[LoadedProspectiveFinalizedCase, ...]:
    episode_by_id = {episode.episode_id: episode for episode in release.verified_admission.admission.episodes}
    ordinal_by_id = {envelope.binding.episode_id: envelope.ordinal for envelope in release.challenge.envelopes}
    admitted = tuple(record for record in audit.records if record.disposition == CaseSelectionDisposition.ADMITTED)
    loaded: list[LoadedProspectiveFinalizedCase] = []
    for index, record in enumerate(admitted):
        assert record.episode_id is not None
        try:
            prospective_episode = episode_by_id[record.episode_id]
            ordinal = ordinal_by_id[record.episode_id]
        except KeyError as error:
            raise ValueError(
                f'admitted case {record.case_id} is absent from the sealed prospective challenge'
            ) from error
        value = inputs[record.case_id]
        case_prefix = PurePosixPath('cases') / f'{index:06d}'
        case_root = eventual_root / Path(*case_prefix.parts)
        bundle_prefix = case_prefix / 'episode'
        _write_bundle(writer, bundle_prefix, value.final_bundle)
        bundle = replace(value.final_bundle, root=case_root / 'episode')
        bundle.validate_integrity()
        writer.write(case_prefix / 'temporal-admission.json', canonical_json_bytes(value.temporal_admission))
        for receipt_index, receipt in enumerate(value.temporal_admission.receipts):
            writer.write(
                case_prefix / 'receipt-proofs' / f'{receipt_index:06d}.bin',
                value.receipt_artifacts[receipt.receipt_id],
            )
        for name in sorted(PROTOCOL_ARTIFACT_NAMES):
            writer.write(case_prefix / 'protocols' / f'{name}.bin', value.protocol_artifacts[name])
        writer.write(case_prefix / 'raw-outcome-source.bin', value.raw_outcome_source)
        writer.write(case_prefix / 'label-derivation-audit.bin', value.label_derivation_audit)
        finalization = finalize_prospective_episode(
            release.verified_admission.admission,
            prospective_episode,
            bundle,
            value.temporal_admission,
            receipt_artifacts=value.receipt_artifacts,
            receipt_verifier=temporal_receipt_verifier,
            protocol_artifacts=value.protocol_artifacts,
            raw_outcome_source=value.raw_outcome_source,
            label_derivation_audit=value.label_derivation_audit,
            case_selection_audit=audit,
            case_selection_audit_commitment=case_selection_audit_sha256(audit),
            finalized_at=finalized_at,
        )
        writer.write(case_prefix / 'finalization-binding.json', canonical_json_bytes(finalization))
        loaded.append(
            LoadedProspectiveFinalizedCase(
                case_id=record.case_id,
                ordinal=ordinal,
                prospective_episode=prospective_episode,
                final_bundle=bundle,
                temporal_admission=value.temporal_admission,
                receipt_artifacts=dict(value.receipt_artifacts),
                protocol_artifacts=dict(value.protocol_artifacts),
                raw_outcome_source=value.raw_outcome_source,
                label_derivation_audit=value.label_derivation_audit,
                finalization=finalization,
            )
        )
    return tuple(loaded)


def _load_cases(
    root: Path,
    *,
    file_bytes: Mapping[str, bytes],
    manifest: ProspectiveCohortFinalizationManifest,
    release: LoadedProspectiveCohortRelease,
    completion: LoadedProspectiveAttemptCompletion,
    audit: CaseSelectionAudit,
    temporal_receipt_verifier: TemporalReceiptVerifier,
) -> tuple[LoadedProspectiveFinalizedCase, ...]:
    admitted = tuple(record for record in audit.records if record.disposition == CaseSelectionDisposition.ADMITTED)
    if tuple(item.case_id for item in manifest.cases) != tuple(record.case_id for record in admitted):
        raise ProspectiveCohortFinalizationIntegrityError(
            'finalized case inventory differs from the exhaustive admitted records'
        )
    episode_by_id = {episode.episode_id: episode for episode in release.verified_admission.admission.episodes}
    loaded: list[LoadedProspectiveFinalizedCase] = []
    for record, binding in zip(admitted, manifest.cases, strict=True):
        assert record.episode_id is not None
        if binding.episode_id != record.episode_id:
            raise ProspectiveCohortFinalizationIntegrityError(
                f'case {record.case_id} binds a different prospective episode'
            )
        prospective_episode = episode_by_id.get(record.episode_id)
        if prospective_episode is None:
            raise ProspectiveCohortFinalizationIntegrityError(
                f'case {record.case_id} is absent from the sealed challenge'
            )
        case_root = root / binding.directory
        bundle_root = case_root / 'episode'
        bundle = EpisodeBundle.load(bundle_root, include_private=True)
        for relative, expected in _bundle_file_payloads(bundle).items():
            path = f'{binding.directory}/episode/{relative}'
            if file_bytes.get(path) != expected:
                raise ProspectiveCohortFinalizationIntegrityError(
                    f'private scoring bundle file is not canonical: {path}'
                )
        temporal_path = f'{binding.directory}/temporal-admission.json'
        temporal = _canonical_model_bytes(file_bytes[temporal_path], TemporalAdmissionEnvelope, 'temporal admission')
        receipts = {
            receipt.receipt_id: file_bytes[f'{binding.directory}/receipt-proofs/{receipt_index:06d}.bin']
            for receipt_index, receipt in enumerate(temporal.receipts)
        }
        protocols = {name: file_bytes[f'{binding.directory}/protocols/{name}.bin'] for name in PROTOCOL_ARTIFACT_NAMES}
        raw_outcome = file_bytes[f'{binding.directory}/raw-outcome-source.bin']
        derivation = file_bytes[f'{binding.directory}/label-derivation-audit.bin']
        stored_finalization = _canonical_model_bytes(
            file_bytes[f'{binding.directory}/finalization-binding.json'],
            ProspectiveFinalizationBinding,
            'prospective finalization binding',
        )
        try:
            rebuilt_finalization = finalize_prospective_episode(
                release.verified_admission.admission,
                prospective_episode,
                bundle,
                temporal,
                receipt_artifacts=receipts,
                receipt_verifier=temporal_receipt_verifier,
                protocol_artifacts=protocols,
                raw_outcome_source=raw_outcome,
                label_derivation_audit=derivation,
                case_selection_audit=audit,
                case_selection_audit_commitment=case_selection_audit_sha256(audit),
                finalized_at=manifest.finalized_at,
            )
        except ProspectiveFinalizationError as error:
            raise ProspectiveCohortFinalizationIntegrityError(
                f'case {record.case_id} failed Tier A finalization: {error}'
            ) from error
        if rebuilt_finalization != stored_finalization:
            raise ProspectiveCohortFinalizationIntegrityError(
                f'case {record.case_id} finalization binding differs from verified inputs'
            )
        case = LoadedProspectiveFinalizedCase(
            case_id=record.case_id,
            ordinal=binding.ordinal,
            prospective_episode=prospective_episode,
            final_bundle=bundle,
            temporal_admission=temporal,
            receipt_artifacts=receipts,
            protocol_artifacts=protocols,
            raw_outcome_source=raw_outcome,
            label_derivation_audit=derivation,
            finalization=stored_finalization,
        )
        if binding != _case_binding(case, completion):
            raise ProspectiveCohortFinalizationIntegrityError(
                f'case {record.case_id} manifest binding differs from exact response and outcome inputs'
            )
        loaded.append(case)
    return tuple(loaded)


def _write_bundle(
    writer: _FinalizationPublicationWriter,
    prefix: PurePosixPath,
    bundle: EpisodeBundle,
) -> None:
    bundle.validate_integrity()
    if bundle.private_labels is None or bundle.label_commitment_key is None:
        raise ValueError('official finalization requires private labels and their HMAC key')
    for relative, payload in _bundle_file_payloads(bundle).items():
        writer.write(prefix / relative, payload)


def _bundle_file_payloads(bundle: EpisodeBundle) -> dict[str, bytes]:
    labels = bundle.private_labels
    key = bundle.label_commitment_key
    if labels is None or key is None:
        raise ValueError('official finalization requires private labels and their HMAC key')
    payloads = {
        'manifest.json': canonical_json_bytes(bundle.manifest),
        'candidates.jsonl': jsonl_text(bundle.candidates).encode(),
        'evidence.jsonl': jsonl_text(bundle.evidence).encode(),
        'private/outcomes.jsonl': jsonl_text(labels.outcomes).encode(),
        'private/assessments_gold.jsonl': jsonl_text(labels.assessments_gold).encode(),
        'private/evidence_gold.jsonl': jsonl_text(labels.evidence_gold).encode(),
        'private/label_commitment_key.hex': key.hex().encode('ascii') + b'\n',
    }
    if bundle.manifest.reward_version == RANKING_REWARD_VERSION:
        if bundle.ranking_labels is None:
            raise ValueError('V1 official finalization requires private ranking labels')
        payloads['private/ranking_labels.jsonl'] = jsonl_text(bundle.ranking_labels).encode()
    return payloads


def _case_binding(
    case: LoadedProspectiveFinalizedCase,
    completion: LoadedProspectiveAttemptCompletion,
) -> ProspectiveFinalizedCaseBinding:
    response_status = None
    response_sha256 = None
    response_bytes = None
    if completion.target.status == ProspectiveAttemptCompletionStatus.SUCCESS:
        run_binding = completion.target.run
        if run_binding is None or case.ordinal >= len(run_binding.episodes):
            raise ProspectiveCohortFinalizationIntegrityError(f'completed run is missing ordinal {case.ordinal}')
        episode = run_binding.episodes[case.ordinal]
        if episode.episode_id != case.prospective_episode.episode_id:
            raise ProspectiveCohortFinalizationIntegrityError(
                f'completed run ordinal {case.ordinal} binds a different episode'
            )
        response_status = episode.status
        response_sha256 = episode.response_record_sha256
        response_bytes = episode.response_record_bytes
    return ProspectiveFinalizedCaseBinding(
        case_id=case.case_id,
        ordinal=case.ordinal,
        episode_id=case.prospective_episode.episode_id,
        directory=f'cases/{_case_directory_index(case.case_id, case, completion):06d}',
        decision_snapshot_sha256=case.prospective_episode.decision_snapshot_sha256,
        decision_context_sha256=case.prospective_episode.decision_context_sha256,
        final_manifest_sha256=case.final_bundle.manifest_sha256,
        labels_sha256=case.final_bundle.manifest.labels_sha256,
        temporal_admission_sha256=model_sha256(case.temporal_admission),
        finalization_binding_sha256=model_sha256(case.finalization),
        response_status=response_status,
        response_record_sha256=response_sha256,
        response_record_bytes=response_bytes,
    )


def _case_directory_index(
    case_id: str,
    case: LoadedProspectiveFinalizedCase,
    completion: LoadedProspectiveAttemptCompletion,
) -> int:
    # The directory is assigned by sorted case order, not by challenge ordinal.
    # Recover it from the case root when available; callers loading an artifact
    # always use a normalized ``cases/NNNNNN/episode`` root.
    try:
        candidate = int(case.final_bundle.root.parent.name)
    except ValueError:
        candidate = -1
    if candidate >= 0 and case.final_bundle.root.parent.parent.name == 'cases':
        return candidate
    # During construction every normalized bundle has already been reloaded
    # from its staging directory, so reaching this branch signals misuse.
    raise ProspectiveCohortFinalizationIntegrityError(
        f'case {case_id} does not have a normalized finalization directory '
        f'for attempt {completion.target.attempt_key_sha256}'
    )


def _compute_score_report(
    release: LoadedProspectiveCohortRelease,
    completion: LoadedProspectiveAttemptCompletion,
    audit: CaseSelectionAudit,
    cases: tuple[LoadedProspectiveFinalizedCase, ...],
) -> ProspectiveCohortScoreReport:
    results = tuple(
        sorted(
            (_score_case(release, completion, case) for case in cases),
            key=lambda item: item.ordinal,
        )
    )
    denominator = sum(
        entry.disposition == CaseUniverseDisposition.PREELIGIBLE
        for entry in release.verified_admission.case_universe.entries
    )
    if denominator <= 0:
        raise ProspectiveCohortFinalizationIntegrityError(
            'official cohort requires at least one preeligible case denominator'
        )
    if len(cases) > denominator:
        raise ProspectiveCohortFinalizationIntegrityError(
            'admitted final cases exceed the frozen preeligible-case denominator'
        )
    valid = sum(item.score.status == ScoreStatus.VALID for item in results)
    invalid = len(results) - valid
    reward_sum = sum(item.score.reward for item in results if item.score.reward is not None)
    return ProspectiveCohortScoreReport(
        denominator_case_count=denominator,
        admitted_case_count=len(cases),
        valid_score_count=valid,
        invalid_response_count=invalid,
        unscored_case_count=denominator - len(cases),
        reward_sum=reward_sum,
        reward=reward_sum / denominator,
        episodes=results,
    )


def _score_case(
    release: LoadedProspectiveCohortRelease,
    completion: LoadedProspectiveAttemptCompletion,
    case: LoadedProspectiveFinalizedCase,
) -> ProspectiveCohortEpisodeScore:
    if completion.target.status == ProspectiveAttemptCompletionStatus.FAILURE:
        score = _invalid_score(
            case.final_bundle,
            IssueCode.RUNNER_FAILURE,
            'the globally registered first-and-only attempt terminated with an explicit failure',
        )
        return ProspectiveCohortEpisodeScore(
            case_id=case.case_id,
            ordinal=case.ordinal,
            episode_id=case.prospective_episode.episode_id,
            score=score,
        )
    run = completion.run
    if run is None:
        raise ProspectiveCohortFinalizationIntegrityError(
            'successful attempt completion is missing its authenticated run artifact'
        )
    if case.ordinal >= len(run.receipt.episodes) or case.ordinal >= len(run.response_records):
        raise ProspectiveCohortFinalizationIntegrityError(
            f'run artifact is missing finalized case ordinal {case.ordinal}'
        )
    receipt = run.receipt.episodes[case.ordinal]
    record = run.response_records[case.ordinal]
    if receipt.episode_id != case.prospective_episode.episode_id:
        raise ProspectiveCohortFinalizationIntegrityError(
            f'run response ordinal {case.ordinal} binds a different episode'
        )
    if _sha256(record) != receipt.response_record_sha256 or len(record) != receipt.response_record_bytes:
        raise ProspectiveCohortFinalizationIntegrityError(
            f'run response ordinal {case.ordinal} differs from its authenticated receipt'
        )
    if receipt.status != EpisodeRunStatus.ACCEPTED:
        score = _invalid_score(
            case.final_bundle,
            IssueCode.INVALID_RUN_RESPONSE,
            f'official runner retained terminal response status {receipt.status.value}',
        )
    else:
        try:
            submission = ProspectiveSubmission.model_validate_json(record)
            if record != canonical_json_bytes(submission) + b'\n':
                raise ValueError('accepted prospective response is not canonical JSONL')
            submission.require_episode(case.prospective_episode)
            adapted = adapt_prospective_submission(
                submission,
                prospective_admission=release.verified_admission.admission,
                prospective_episode=case.prospective_episode,
                final_bundle=case.final_bundle,
                finalization=case.finalization,
            )
            score = make_submission_evaluator(
                case.final_bundle,
                allow_sealed_test=True,
            ).score(adapted)
        except ValueError as error:
            score = _invalid_score(
                case.final_bundle,
                IssueCode.INVALID_RUN_RESPONSE,
                f'accepted runner record failed the sealed response contract: {error}',
            )
    return ProspectiveCohortEpisodeScore(
        case_id=case.case_id,
        ordinal=case.ordinal,
        episode_id=case.prospective_episode.episode_id,
        response_status=receipt.status,
        response_record_sha256=receipt.response_record_sha256,
        score=score,
    )


def _invalid_score(
    bundle: EpisodeBundle,
    code: IssueCode,
    detail: str,
) -> ScoreVector | ScoreVectorV1:
    issue = ValidationIssue(code=code, detail=detail)
    if bundle.manifest.reward_version == RANKING_REWARD_VERSION:
        return ScoreVectorV1(
            episode_id=bundle.manifest.episode_id,
            manifest_sha256=bundle.manifest_sha256,
            labels_sha256=bundle.manifest.labels_sha256,
            status=ScoreStatus.INVALID_SCHEMA,
            issues=[issue],
        )
    return ScoreVector(
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        labels_sha256=bundle.manifest.labels_sha256,
        status=ScoreStatus.INVALID_SCHEMA,
        issues=[issue],
    )


def _build_manifest(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    completion: LoadedProspectiveAttemptCompletion,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    audit: CaseSelectionAudit,
    cases: tuple[LoadedProspectiveFinalizedCase, ...],
    score_report: ProspectiveCohortScoreReport,
    finalized_at: datetime,
    file_bindings: tuple[ProspectiveFinalizationFileBinding, ...],
) -> ProspectiveCohortFinalizationManifest:
    finalized_at = _aware(finalized_at, 'finalized_at')
    if finalized_at < completion.manifest.registry_proof.witnessed_at:
        raise ProspectiveCohortFinalizationIntegrityError(
            'cohort finalization cannot precede the external completion registry proof'
        )
    if cases and finalized_at < max(case.temporal_admission.admitted_at for case in cases):
        raise ProspectiveCohortFinalizationIntegrityError(
            'cohort finalization cannot precede an admitted outcome snapshot'
        )
    disposition_values = sorted(
        {record.disposition for record in audit.records},
        key=lambda value: value.value,
    )
    disposition_counts = tuple(
        ProspectiveDispositionCount(
            disposition=disposition,
            count=sum(record.disposition == disposition for record in audit.records),
        )
        for disposition in disposition_values
    )
    run = completion.target.run
    return ProspectiveCohortFinalizationManifest(
        release_id=release.manifest.release_id,
        purpose=release.verified_admission.admission.purpose,
        prospective_release_sha256=release.release_sha256,
        release_tree_sha256=release_seal.target.release_tree_sha256,
        release_seal_manifest_sha256=prospective_release_seal_manifest_sha256(release_seal.manifest),
        release_seal_target_sha256=prospective_release_seal_target_sha256(release_seal.target),
        release_seal_proof_sha256=release_seal.manifest.timestamp_proof.proof_sha256,
        reservation_manifest_sha256=prospective_attempt_reservation_manifest_sha256(reservation.manifest),
        reservation_target_sha256=prospective_attempt_reservation_target_sha256(reservation.target),
        reservation_proof_sha256=reservation.manifest.registry_proof.proof_sha256,
        start_authorization_manifest_sha256=(
            prospective_attempt_start_authorization_manifest_sha256(start_authorization.manifest)
        ),
        start_authorization_target_sha256=prospective_attempt_start_target_sha256(start_authorization.target),
        start_authorization_proof_sha256=start_authorization.manifest.start_proof.proof_sha256,
        completion_manifest_sha256=prospective_attempt_completion_manifest_sha256(completion.manifest),
        completion_target_sha256=prospective_attempt_completion_target_sha256(completion.target),
        completion_proof_sha256=completion.manifest.registry_proof.proof_sha256,
        canonical_cohort_id=completion.target.canonical_cohort_id,
        track_id=completion.target.track_id,
        registered_entry_id=completion.target.registered_entry_id,
        attempt_key_sha256=completion.target.attempt_key_sha256,
        alias_key_sha256=completion.target.alias_key_sha256,
        completion_status=completion.target.status,
        run_receipt_sha256=run.run_receipt_sha256 if run is not None else None,
        responses_sha256=run.responses_sha256 if run is not None else None,
        system_manifest_sha256=_sha256(canonical_json_bytes(system)),
        runner_policy_sha256=_sha256(canonical_json_bytes(runner_policy)),
        attempt_policy_sha256=prospective_attempt_policy_sha256(attempt_policy),
        case_universe_sha256=case_universe_sha256(release.verified_admission.case_universe),
        verifier_policy_sha256=_sha256(release.verifier_policy),
        case_selection_audit_sha256=case_selection_audit_sha256(audit),
        universe_case_count=len(release.verified_admission.case_universe.entries),
        denominator_case_count=score_report.denominator_case_count,
        disposition_counts=disposition_counts,
        cases=tuple(
            sorted(
                (_case_binding(case, completion) for case in cases),
                key=lambda item: (item.case_id, item.episode_id),
            )
        ),
        score_report_sha256=_sha256(canonical_json_bytes(score_report)),
        cohort_reward=score_report.reward,
        finalized_at=finalized_at,
        files=file_bindings,
    )


@dataclass(frozen=True)
class _TreeInventory:
    files: frozenset[str]
    directories: frozenset[str]
    byte_count: int


class _FinalizationPublicationWriter:
    """Bounded exact-file writer over a descriptor-anchored publication."""

    def __init__(self, publication: AtomicDirectoryPublication) -> None:
        self._publication = publication
        self._bindings: dict[str, ProspectiveFinalizationFileBinding] = {}
        self._directories: set[str] = set()
        self._byte_count = 0

    def write(self, relative: str | PurePosixPath, payload: bytes) -> None:
        path = _safe_relative_path(relative)
        path_string = path.as_posix()
        if type(payload) is not bytes or not payload:
            raise ValueError(f'finalization artifact must contain non-empty bytes: {path_string}')
        if len(payload) > _MAX_FILE_BYTES:
            raise ValueError(f'finalization artifact exceeds the size limit: {path_string}')
        if path_string in self._bindings:
            raise ValueError(f'finalization artifact path was written more than once: {path_string}')
        if len(self._bindings) >= _MAX_FILES or self._byte_count + len(payload) > _MAX_TREE_BYTES:
            raise ValueError('finalization tree exceeds its file or byte limit')
        parent = path.parent
        new_directories: set[str] = set()
        while parent != PurePosixPath('.'):
            new_directories.add(parent.as_posix())
            parent = parent.parent
        if len(self._directories | new_directories) > _MAX_DIRECTORIES:
            raise ValueError('finalization tree exceeds its directory limit')
        if path.parent != PurePosixPath('.'):
            self._publication.make_directory(path.parent, mode=0o700)
        self._publication.write_bytes(path, payload, mode=0o600)
        self._directories.update(new_directories)
        self._byte_count += len(payload)
        self._bindings[path_string] = ProspectiveFinalizationFileBinding(
            path=path_string,
            sha256=_sha256(payload),
            byte_count=len(payload),
        )

    def file_bindings(self) -> tuple[ProspectiveFinalizationFileBinding, ...]:
        return tuple(self._bindings[path] for path in sorted(self._bindings))


def _publication_target(output_dir: Path) -> Path:
    return output_dir.expanduser().absolute()


def _resolve_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveCohortFinalizationIntegrityError('prospective cohort finalization root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ProspectiveCohortFinalizationIntegrityError(
            f'prospective cohort finalization root does not exist: {resolved}'
        )
    return resolved


def _inventory_tree(root: Path) -> _TreeInventory:
    files: set[str] = set()
    directories: set[str] = set()
    byte_count = 0
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        for name in directory_names:
            path = current_path / name
            relative = (relative_current / name).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ProspectiveCohortFinalizationIntegrityError(
                    f'finalization tree cannot contain symlink or special directory: {relative}'
                )
            directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = (relative_current / name).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ProspectiveCohortFinalizationIntegrityError(
                    f'finalization tree cannot contain symlink or special file: {relative}'
                )
            if metadata.st_size <= 0 or metadata.st_size > _MAX_FILE_BYTES:
                raise ProspectiveCohortFinalizationIntegrityError(f'finalization file has an invalid size: {relative}')
            files.add(relative)
            byte_count += metadata.st_size
            if len(files) > _MAX_FILES or byte_count > _MAX_TREE_BYTES:
                raise ProspectiveCohortFinalizationIntegrityError('finalization tree exceeds its file or byte limit')
        if len(directories) > _MAX_DIRECTORIES:
            raise ProspectiveCohortFinalizationIntegrityError('finalization tree exceeds its directory limit')
    return _TreeInventory(
        files=frozenset(files),
        directories=frozenset(directories),
        byte_count=byte_count,
    )


def _parent_directories(files: set[str]) -> frozenset[str]:
    directories: set[str] = set()
    for file in files:
        parent = PurePosixPath(file).parent
        while parent != PurePosixPath('.'):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


def _safe_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError('artifact paths must be normalized safe relative paths')
    return path


def _read_file(path: Path, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveCohortFinalizationIntegrityError(f'finalization artifact must be a regular file: {path}')
        if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
            raise ProspectiveCohortFinalizationIntegrityError(f'finalization artifact has an invalid size: {path}')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags)
        try:
            payload = b''
            while len(payload) <= maximum_bytes:
                chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload += chunk
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ProspectiveCohortFinalizationIntegrityError(
            f'cannot read finalization artifact {path}: {error}'
        ) from error
    if not payload or len(payload) > maximum_bytes:
        raise ProspectiveCohortFinalizationIntegrityError(f'finalization artifact has an invalid size: {path}')
    return payload


def _canonical_model_bytes[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveCohortFinalizationIntegrityError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(value):
        raise ProspectiveCohortFinalizationIntegrityError(f'{label} must use canonical JSON encoding')
    return value


def _canonical_model_value[ModelT: StrictModel](
    value: ModelT,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f'invalid {label}: {error}') from error


def _aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{label} must include a UTC offset')
    return value.astimezone(timezone.utc)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
