"""Pre-run registration and terminal retention for prospective benchmark attempts.

The post-run timestamp sidecar cannot prove that an organizer did not discard an
earlier execution.  This module therefore registers the executable and its one allowed
attempt *before* the challenge opens, then requires exactly one externally registered
terminal event: either the exact official run or an explicit retained failure.

The registry verifier is an injected production trust boundary.  It must enforce global
uniqueness, idempotent verification, and the organizer's canonical-cohort/alias mapping.
Local files and hashes deliberately do not claim to provide that global property.
"""

from __future__ import annotations

import enum
import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublication
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.prospective_release_approval_identity import (
    TierAProspectiveReleaseApprovalReplay,
)
from vaxreplay.prospective_admission import CaseUniverseSealVerifier, SourceCaptureVerifier
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    load_prospective_cohort_release,
)
from vaxreplay.prospective_schema import (
    PROSPECTIVE_RESPONSE_PROTOCOL,
    ProspectiveAttemptPolicy,
    prospective_attempt_policy_sha256,
)
from vaxreplay.runner.orchestrator import LoadedRunArtifact, load_run_artifact
from vaxreplay.runner.prospective_release_seal import (
    LoadedProspectiveReleaseSeal,
    ProspectiveReleaseTimestampVerifier,
    load_prospective_release_seal,
    prospective_release_seal_manifest_sha256,
    prospective_release_seal_target_sha256,
)
from vaxreplay.runner.schema import (
    EpisodeRunStatus,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)
from vaxreplay.temporal_schema import (
    TemporalReceiptAuthority,
    TemporalReceiptVerifier,
)

PROSPECTIVE_ATTEMPT_RESERVATION_TARGET_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-reservation-target.v0.1'
PROSPECTIVE_ATTEMPT_REGISTRY_PROOF_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-registry-proof.v0.1'
PROSPECTIVE_ATTEMPT_RESERVATION_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-reservation.v0.1'
PROSPECTIVE_ATTEMPT_START_TARGET_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-start-target.v0.1'
PROSPECTIVE_ATTEMPT_START_PROOF_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-start-proof.v0.1'
PROSPECTIVE_ATTEMPT_START_AUTHORIZATION_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-start-authorization.v0.1'
PROSPECTIVE_ATTEMPT_COMPLETION_TARGET_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-completion-target.v0.2'
PROSPECTIVE_ATTEMPT_COMPLETION_SCHEMA_VERSION = 'vaxreplay.prospective-attempt-completion.v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_PROOF_BYTES = 512 * 1024 * 1024
_MAX_FAILURE_BYTES = 64 * 1024 * 1024
_EXTERNAL_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}


class ProspectiveAttemptIntegrityError(ValueError):
    """Raised when an official reservation or terminal record fails closed."""


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class ProspectiveExecutableIdentity(StrictModel):
    """Alias-resistant executable identity, intentionally excluding submission ID."""

    image_ref: str = Field(min_length=1)
    entrypoint: tuple[str, ...] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harness_id: str = Field(min_length=1)
    response_protocol: Literal['vaxreplay.prospective-submission-json-stdout.v0.1'] = PROSPECTIVE_RESPONSE_PROTOCOL

    @classmethod
    def from_system(cls, system: SystemSubmissionManifest) -> ProspectiveExecutableIdentity:
        if system.response_protocol != PROSPECTIVE_RESPONSE_PROTOCOL:
            raise ValueError('prospective reservations require the prospective response protocol')
        return cls(
            image_ref=system.image_ref,
            entrypoint=system.entrypoint,
            model_id=system.model_id,
            harness_id=system.harness_id,
            response_protocol=system.response_protocol,
        )


def prospective_executable_identity_sha256(identity: ProspectiveExecutableIdentity) -> str:
    return _sha256(canonical_json_bytes(identity))


def prospective_executable_core_sha256(identity: ProspectiveExecutableIdentity) -> str:
    """Hash executable bytes/entrypoint while ignoring declared model/harness aliases."""

    return _sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.prospective-executable-core.v0.1',
                'image_ref': identity.image_ref,
                'entrypoint': identity.entrypoint,
                'response_protocol': identity.response_protocol,
            }
        )
    )


class ProspectiveAttemptReservationTarget(StrictModel):
    """Canonical system registration that must be witnessed before opening."""

    schema_version: Literal['vaxreplay.prospective-attempt-reservation-target.v0.1'] = (
        PROSPECTIVE_ATTEMPT_RESERVATION_TARGET_SCHEMA_VERSION
    )
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_cohort_id: str = Field(pattern=_ID_PATTERN)
    track_id: str = Field(pattern=_ID_PATTERN)
    registered_entry_id: str = Field(pattern=_ID_PATTERN)
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_manifest_bytes: int = Field(gt=0)
    executable: ProspectiveExecutableIdentity
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_core_sha256: str = Field(pattern=_SHA256_PATTERN)
    runner_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    runner_policy_bytes: int = Field(gt=0)
    attempt_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_policy_bytes: int = Field(gt=0)
    attempt_number: Literal[1] = 1
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    submissions_open_at: datetime
    run_deadline_at: datetime

    @field_validator('submissions_open_at', 'run_deadline_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'attempt reservation timestamp')

    @model_validator(mode='after')
    def validate_identity(self) -> Self:
        if self.submissions_open_at >= self.run_deadline_at:
            raise ValueError('submissions must open before the run deadline')
        if self.executable_sha256 != prospective_executable_identity_sha256(self.executable):
            raise ValueError('executable_sha256 does not bind the executable identity')
        if self.executable_core_sha256 != prospective_executable_core_sha256(self.executable):
            raise ValueError('executable_core_sha256 does not bind the executable image and entrypoint')
        expected_attempt = _attempt_key(
            release_sha256=self.prospective_release_sha256,
            canonical_cohort_id=self.canonical_cohort_id,
            track_id=self.track_id,
            registered_entry_id=self.registered_entry_id,
            executable_sha256=self.executable_sha256,
        )
        if self.attempt_key_sha256 != expected_attempt:
            raise ValueError('attempt_key_sha256 does not bind the registered attempt')
        expected_alias = _alias_key(
            canonical_cohort_id=self.canonical_cohort_id,
            track_id=self.track_id,
            executable_core_sha256=self.executable_core_sha256,
        )
        if self.alias_key_sha256 != expected_alias:
            raise ValueError('alias_key_sha256 does not bind the canonical cohort and executable')
        return self


class ProspectiveAttemptRegistryProof(StrictModel):
    """Metadata for a globally unique reservation or terminal registry record."""

    schema_version: Literal['vaxreplay.prospective-attempt-registry-proof.v0.1'] = (
        PROSPECTIVE_ATTEMPT_REGISTRY_PROOF_SCHEMA_VERSION
    )
    event_type: Literal['reservation', 'completion']
    receipt_id: str = Field(min_length=1)
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    target_schema_version: Literal[
        'vaxreplay.prospective-attempt-reservation-target.v0.1',
        'vaxreplay.prospective-attempt-completion-target.v0.2',
    ]
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    canonical_cohort_id: str = Field(pattern=_ID_PATTERN)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'registry witness timestamp')

    @field_validator('authority_type')
    @classmethod
    def validate_authority(cls, value: TemporalReceiptAuthority) -> TemporalReceiptAuthority:
        if value not in _EXTERNAL_AUTHORITIES:
            raise ValueError('attempt registry proofs require RFC 3161 or a public transparency log')
        return value

    @model_validator(mode='after')
    def validate_event_schema(self) -> Self:
        expected = {
            'reservation': PROSPECTIVE_ATTEMPT_RESERVATION_TARGET_SCHEMA_VERSION,
            'completion': PROSPECTIVE_ATTEMPT_COMPLETION_TARGET_SCHEMA_VERSION,
        }[self.event_type]
        if self.target_schema_version != expected:
            raise ValueError('registry event type does not match its target schema')
        return self


class ProspectiveAttemptReservationManifest(StrictModel):
    schema_version: Literal['vaxreplay.prospective-attempt-reservation.v0.1'] = (
        PROSPECTIVE_ATTEMPT_RESERVATION_SCHEMA_VERSION
    )
    target_path: Literal['target.json'] = 'target.json'
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    proof_path: Literal['registry-proof.bin'] = 'registry-proof.bin'
    registry_proof: ProspectiveAttemptRegistryProof

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if self.registry_proof.event_type != 'reservation':
            raise ValueError('attempt reservation requires a reservation registry proof')
        if (
            self.registry_proof.target_sha256 != self.target_sha256
            or self.registry_proof.target_bytes != self.target_bytes
        ):
            raise ValueError('registry proof does not bind the reservation target')
        return self


type ProspectiveAttemptRegistryVerifier = Callable[[ProspectiveAttemptRegistryProof, bytes], bool]


@dataclass(frozen=True)
class LoadedProspectiveAttemptReservation:
    root: Path
    manifest: ProspectiveAttemptReservationManifest
    target: ProspectiveAttemptReservationTarget
    proof_bytes: bytes
    manifest_sha256: str


def prospective_attempt_reservation_target_sha256(target: ProspectiveAttemptReservationTarget) -> str:
    return _sha256(canonical_json_bytes(target))


def prospective_attempt_reservation_manifest_sha256(
    manifest: ProspectiveAttemptReservationManifest,
) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_attempt_reservation_target(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    canonical_cohort_id: str,
    track_id: str,
    registered_entry_id: str,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
) -> ProspectiveAttemptReservationTarget:
    fresh_release, fresh_release_seal = _fresh_release_context(
        release=release,
        release_seal=release_seal,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
        error_type=ValueError,
    )
    _require_system_policy(
        fresh_release,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        error_type=ValueError,
    )
    executable = ProspectiveExecutableIdentity.from_system(system)
    executable_sha256 = prospective_executable_identity_sha256(executable)
    executable_core_sha256 = prospective_executable_core_sha256(executable)
    system_bytes = canonical_json_bytes(system)
    policy_bytes = canonical_json_bytes(runner_policy)
    attempt_bytes = canonical_json_bytes(attempt_policy)
    release_seal_sha256 = prospective_release_seal_manifest_sha256(fresh_release_seal.manifest)
    return ProspectiveAttemptReservationTarget(
        prospective_release_sha256=fresh_release.release_sha256,
        release_tree_sha256=fresh_release_seal.target.release_tree_sha256,
        release_seal_sha256=release_seal_sha256,
        release_seal_target_sha256=prospective_release_seal_target_sha256(fresh_release_seal.target),
        canonical_cohort_id=canonical_cohort_id,
        track_id=track_id,
        registered_entry_id=registered_entry_id,
        system_manifest_sha256=_sha256(system_bytes),
        system_manifest_bytes=len(system_bytes),
        executable=executable,
        executable_sha256=executable_sha256,
        executable_core_sha256=executable_core_sha256,
        runner_policy_sha256=_sha256(policy_bytes),
        runner_policy_bytes=len(policy_bytes),
        attempt_policy_sha256=_sha256(attempt_bytes),
        attempt_policy_bytes=len(attempt_bytes),
        attempt_key_sha256=_attempt_key(
            release_sha256=fresh_release.release_sha256,
            canonical_cohort_id=canonical_cohort_id,
            track_id=track_id,
            registered_entry_id=registered_entry_id,
            executable_sha256=executable_sha256,
        ),
        alias_key_sha256=_alias_key(
            canonical_cohort_id=canonical_cohort_id,
            track_id=track_id,
            executable_core_sha256=executable_core_sha256,
        ),
        submissions_open_at=fresh_release_seal.target.submissions_open_at,
        run_deadline_at=fresh_release_seal.target.run_deadline_at,
    )


def build_prospective_attempt_reservation(
    output_dir: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    canonical_cohort_id: str,
    track_id: str,
    registered_entry_id: str,
    registry_proof: ProspectiveAttemptRegistryProof,
    proof_bytes: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
) -> LoadedProspectiveAttemptReservation:
    target = build_prospective_attempt_reservation_target(
        release=release,
        release_seal=release_seal,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        canonical_cohort_id=canonical_cohort_id,
        track_id=track_id,
        registered_entry_id=registered_entry_id,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
    )
    target_bytes = canonical_json_bytes(target)
    _verify_registry_proof(
        target=target,
        target_bytes=target_bytes,
        registry_proof=registry_proof,
        proof_bytes=proof_bytes,
        earliest_witnessed_at=release_seal.manifest.timestamp_proof.witnessed_at,
        latest_witnessed_at=target.submissions_open_at,
        registry_verifier=registry_verifier,
        event_type='reservation',
        error_type=ValueError,
    )
    manifest = ProspectiveAttemptReservationManifest(
        target_sha256=_sha256(target_bytes),
        target_bytes=len(target_bytes),
        registry_proof=registry_proof,
    )
    target_root = _publication_target(output_dir)
    with AtomicDirectoryPublication.create(target_root) as publication:
        publication.write_bytes('target.json', target_bytes, mode=0o600)
        publication.write_bytes('registry-proof.bin', proof_bytes, mode=0o600)
        publication.write_bytes('reservation.json', canonical_json_bytes(manifest), mode=0o600)
        installed_root = publication.publish(root_mode=0o700)
        loaded = load_prospective_attempt_reservation(
            installed_root,
            release=release,
            release_seal=release_seal,
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
        publication.commit()
        return loaded


def load_prospective_attempt_reservation(
    root: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
) -> LoadedProspectiveAttemptReservation:
    resolved = _resolve_root(root, 'attempt reservation')
    _require_inventory(
        resolved,
        {'reservation.json', 'target.json', 'registry-proof.bin'},
        error_type=ProspectiveAttemptIntegrityError,
    )
    manifest_bytes = _read_file(resolved / 'reservation.json', _MAX_MODEL_BYTES)
    manifest = _canonical_model(
        manifest_bytes,
        ProspectiveAttemptReservationManifest,
        'attempt reservation manifest',
    )
    target_bytes = _read_file(resolved / manifest.target_path, _MAX_MODEL_BYTES)
    target = _canonical_model(target_bytes, ProspectiveAttemptReservationTarget, 'attempt reservation target')
    if _sha256(target_bytes) != manifest.target_sha256 or len(target_bytes) != manifest.target_bytes:
        raise ProspectiveAttemptIntegrityError('attempt reservation target does not match its manifest')
    expected = build_prospective_attempt_reservation_target(
        release=release,
        release_seal=release_seal,
        system=system,
        runner_policy=runner_policy,
        attempt_policy=attempt_policy,
        canonical_cohort_id=target.canonical_cohort_id,
        track_id=target.track_id,
        registered_entry_id=target.registered_entry_id,
        decision_receipt_verifier=decision_receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=source_capture_verifier,
        expected_approval_report_sha256=expected_approval_report_sha256,
        approval_replay=approval_replay,
        release_timestamp_verifier=release_timestamp_verifier,
    )
    if target != expected:
        raise ProspectiveAttemptIntegrityError('attempt reservation is bound to different official inputs')
    proof_bytes = _read_file(resolved / manifest.proof_path, _MAX_PROOF_BYTES)
    _verify_registry_proof(
        target=target,
        target_bytes=target_bytes,
        registry_proof=manifest.registry_proof,
        proof_bytes=proof_bytes,
        earliest_witnessed_at=release_seal.manifest.timestamp_proof.witnessed_at,
        latest_witnessed_at=target.submissions_open_at,
        registry_verifier=registry_verifier,
        event_type='reservation',
        error_type=ProspectiveAttemptIntegrityError,
    )
    return LoadedProspectiveAttemptReservation(
        root=resolved,
        manifest=manifest,
        target=target,
        proof_bytes=proof_bytes,
        manifest_sha256=_sha256(manifest_bytes),
    )


class ProspectiveAttemptStartTarget(StrictModel):
    """Exact reserved attempt identity presented to the external start authority."""

    schema_version: Literal['vaxreplay.prospective-attempt-start-target.v0.1'] = (
        PROSPECTIVE_ATTEMPT_START_TARGET_SCHEMA_VERSION
    )
    reservation_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_seal_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_cohort_id: str = Field(pattern=_ID_PATTERN)
    track_id: str = Field(pattern=_ID_PATTERN)
    registered_entry_id: str = Field(pattern=_ID_PATTERN)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_core_sha256: str = Field(pattern=_SHA256_PATTERN)
    submissions_open_at: datetime
    run_deadline_at: datetime

    @field_validator('submissions_open_at', 'run_deadline_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'attempt start target timestamp')

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.submissions_open_at >= self.run_deadline_at:
            raise ValueError('attempt start window must open before its run deadline')
        return self


class ProspectiveAttemptStartProof(StrictModel):
    """External proof that one exact reserved attempt was authorized to start."""

    schema_version: Literal['vaxreplay.prospective-attempt-start-proof.v0.1'] = (
        PROSPECTIVE_ATTEMPT_START_PROOF_SCHEMA_VERSION
    )
    receipt_id: str = Field(min_length=1)
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    target_schema_version: Literal['vaxreplay.prospective-attempt-start-target.v0.1'] = (
        PROSPECTIVE_ATTEMPT_START_TARGET_SCHEMA_VERSION
    )
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_cohort_id: str = Field(pattern=_ID_PATTERN)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'attempt start authorization timestamp')

    @field_validator('authority_type')
    @classmethod
    def validate_authority(cls, value: TemporalReceiptAuthority) -> TemporalReceiptAuthority:
        if value not in _EXTERNAL_AUTHORITIES:
            raise ValueError('attempt start proofs require RFC 3161 or a public transparency log')
        return value


class ProspectiveAttemptStartAuthorizationManifest(StrictModel):
    """Exact three-file artifact retaining a post-opening start authorization."""

    schema_version: Literal['vaxreplay.prospective-attempt-start-authorization.v0.1'] = (
        PROSPECTIVE_ATTEMPT_START_AUTHORIZATION_SCHEMA_VERSION
    )
    target_path: Literal['target.json'] = 'target.json'
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    proof_path: Literal['start-proof.bin'] = 'start-proof.bin'
    start_proof: ProspectiveAttemptStartProof

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if self.start_proof.target_sha256 != self.target_sha256 or self.start_proof.target_bytes != self.target_bytes:
            raise ValueError('start proof does not bind the exact attempt start target')
        return self


type ProspectiveAttemptStartVerifier = Callable[[ProspectiveAttemptStartProof, bytes], bool]


@dataclass(frozen=True)
class LoadedProspectiveAttemptStartAuthorization:
    root: Path
    manifest: ProspectiveAttemptStartAuthorizationManifest
    target: ProspectiveAttemptStartTarget
    proof_bytes: bytes
    manifest_sha256: str


def prospective_attempt_start_target_sha256(target: ProspectiveAttemptStartTarget) -> str:
    return _sha256(canonical_json_bytes(target))


def prospective_attempt_start_authorization_manifest_sha256(
    manifest: ProspectiveAttemptStartAuthorizationManifest,
) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_attempt_start_target(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
) -> ProspectiveAttemptStartTarget:
    """Reverify the reservation and build the only target a start authority may approve."""

    fresh = _fresh_reservation(
        reservation=reservation,
        release=release,
        release_seal=release_seal,
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
        error_type=ValueError,
    )
    reserved = fresh.target
    return ProspectiveAttemptStartTarget(
        reservation_manifest_sha256=fresh.manifest_sha256,
        reservation_target_sha256=prospective_attempt_reservation_target_sha256(reserved),
        prospective_release_sha256=reserved.prospective_release_sha256,
        release_tree_sha256=reserved.release_tree_sha256,
        release_seal_target_sha256=reserved.release_seal_target_sha256,
        canonical_cohort_id=reserved.canonical_cohort_id,
        track_id=reserved.track_id,
        registered_entry_id=reserved.registered_entry_id,
        attempt_key_sha256=reserved.attempt_key_sha256,
        alias_key_sha256=reserved.alias_key_sha256,
        executable_sha256=reserved.executable_sha256,
        executable_core_sha256=reserved.executable_core_sha256,
        submissions_open_at=reserved.submissions_open_at,
        run_deadline_at=reserved.run_deadline_at,
    )


def build_prospective_attempt_start_authorization(
    output_dir: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    start_proof: ProspectiveAttemptStartProof,
    proof_bytes: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
) -> LoadedProspectiveAttemptStartAuthorization:
    """Verify and retain one externally authorized, post-opening attempt start.

    The injected verifier authenticates the proof idempotently.  Local files prove
    exact identity and chronology but cannot establish global one-shot execution by
    themselves.  Immediately before backend work, the execution composition must
    invoke its separate durable ``ProspectiveAttemptStartConsumer`` to atomically
    consume this exact attempt/start identity once.
    """

    target = build_prospective_attempt_start_target(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
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
    target_bytes = canonical_json_bytes(target)
    _verify_start_proof(
        target=target,
        target_bytes=target_bytes,
        start_proof=start_proof,
        proof_bytes=proof_bytes,
        start_verifier=start_verifier,
        error_type=ValueError,
    )
    manifest = ProspectiveAttemptStartAuthorizationManifest(
        target_sha256=_sha256(target_bytes),
        target_bytes=len(target_bytes),
        start_proof=start_proof,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    target_root = _publication_target(output_dir)
    with AtomicDirectoryPublication.create(target_root) as publication:
        publication.write_bytes('target.json', target_bytes, mode=0o600)
        publication.write_bytes('start-proof.bin', proof_bytes, mode=0o600)
        publication.write_bytes('start-authorization.json', manifest_bytes, mode=0o600)
        installed_root = publication.publish(root_mode=0o700)
        loaded = load_prospective_attempt_start_authorization(
            installed_root,
            release=release,
            release_seal=release_seal,
            reservation=reservation,
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
            expected_start_authorization_manifest_sha256=_sha256(manifest_bytes),
        )
        publication.commit()
        return loaded


def load_prospective_attempt_start_authorization(
    root: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
) -> LoadedProspectiveAttemptStartAuthorization:
    """Freshly reverify an exact start authorization and its complete prerequisite chain."""

    _require_sha256(
        expected_start_authorization_manifest_sha256,
        'expected attempt start-authorization manifest',
        error_type=ProspectiveAttemptIntegrityError,
    )
    resolved = _resolve_root(root, 'attempt start authorization')
    _require_inventory(
        resolved,
        {'start-authorization.json', 'target.json', 'start-proof.bin'},
        error_type=ProspectiveAttemptIntegrityError,
    )
    manifest_bytes = _read_file(resolved / 'start-authorization.json', _MAX_MODEL_BYTES)
    manifest = _canonical_model(
        manifest_bytes,
        ProspectiveAttemptStartAuthorizationManifest,
        'attempt start-authorization manifest',
    )
    manifest_sha256 = _sha256(manifest_bytes)
    if manifest_sha256 != expected_start_authorization_manifest_sha256:
        raise ProspectiveAttemptIntegrityError(
            'attempt start authorization differs from its out-of-band manifest identity'
        )
    target_bytes = _read_file(resolved / manifest.target_path, _MAX_MODEL_BYTES)
    target = _canonical_model(target_bytes, ProspectiveAttemptStartTarget, 'attempt start target')
    if _sha256(target_bytes) != manifest.target_sha256 or len(target_bytes) != manifest.target_bytes:
        raise ProspectiveAttemptIntegrityError('attempt start target does not match its manifest')
    expected_target = build_prospective_attempt_start_target(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
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
    if target != expected_target:
        raise ProspectiveAttemptIntegrityError(
            'attempt start authorization is bound to different reservation or release inputs'
        )
    proof_bytes = _read_file(resolved / manifest.proof_path, _MAX_PROOF_BYTES)
    _verify_start_proof(
        target=target,
        target_bytes=target_bytes,
        start_proof=manifest.start_proof,
        proof_bytes=proof_bytes,
        start_verifier=start_verifier,
        error_type=ProspectiveAttemptIntegrityError,
    )
    return LoadedProspectiveAttemptStartAuthorization(
        root=resolved,
        manifest=manifest,
        target=target,
        proof_bytes=proof_bytes,
        manifest_sha256=manifest_sha256,
    )


class ProspectiveAttemptCompletionStatus(str, enum.Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'


class ProspectiveEpisodeCompletion(StrictModel):
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    status: EpisodeRunStatus
    response_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_record_bytes: int = Field(gt=0)


class ProspectiveRunCompletionBinding(StrictModel):
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    run_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_receipt_bytes: int = Field(gt=0)
    responses_sha256: str = Field(pattern=_SHA256_PATTERN)
    responses_bytes: int = Field(gt=0)
    resolved_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    started_at: datetime
    finished_at: datetime
    episodes: tuple[ProspectiveEpisodeCompletion, ...] = Field(min_length=1)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'run completion timestamp')

    @model_validator(mode='after')
    def validate_run(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('completed run cannot finish before it starts')
        if tuple(item.ordinal for item in self.episodes) != tuple(range(len(self.episodes))):
            raise ValueError('run completion episode ordinals must be contiguous')
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError('run completion episode IDs must be unique')
        return self

    @classmethod
    def from_run(cls, run: LoadedRunArtifact) -> ProspectiveRunCompletionBinding:
        receipt_bytes = canonical_json_bytes(run.receipt)
        return cls(
            run_id=run.receipt.run_id,
            run_receipt_sha256=run.receipt_sha256,
            run_receipt_bytes=len(receipt_bytes),
            responses_sha256=run.receipt.responses_sha256,
            responses_bytes=run.receipt.responses_bytes,
            resolved_image_id=run.receipt.resolved_image_id,
            started_at=run.receipt.started_at,
            finished_at=run.receipt.finished_at,
            episodes=tuple(
                ProspectiveEpisodeCompletion(
                    ordinal=episode.ordinal,
                    episode_id=episode.episode_id,
                    status=episode.status,
                    response_record_sha256=episode.response_record_sha256,
                    response_record_bytes=episode.response_record_bytes,
                )
                for episode in run.receipt.episodes
            ),
        )


class ProspectiveExplicitFailure(StrictModel):
    failure_code: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]{1,127}$')
    backend_id: str = Field(min_length=1)
    started_at: datetime
    failed_at: datetime
    failure_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    failure_record_bytes: int = Field(gt=0)

    @field_validator('started_at', 'failed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'explicit failure timestamp')

    @model_validator(mode='after')
    def validate_window(self) -> Self:
        if self.failed_at < self.started_at:
            raise ValueError('explicit failure cannot predate its attempt start')
        return self


class ProspectiveAttemptCompletionTarget(StrictModel):
    schema_version: Literal['vaxreplay.prospective-attempt-completion-target.v0.2'] = (
        PROSPECTIVE_ATTEMPT_COMPLETION_TARGET_SCHEMA_VERSION
    )
    reservation_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    reservation_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_target_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_authorization_proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_cohort_id: str = Field(pattern=_ID_PATTERN)
    track_id: str = Field(pattern=_ID_PATTERN)
    registered_entry_id: str = Field(pattern=_ID_PATTERN)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    alias_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: ProspectiveAttemptCompletionStatus
    run: ProspectiveRunCompletionBinding | None = None
    failure: ProspectiveExplicitFailure | None = None
    run_deadline_at: datetime

    @field_validator('run_deadline_at')
    @classmethod
    def validate_deadline(cls, value: datetime) -> datetime:
        return _aware(value, 'completion deadline')

    @model_validator(mode='after')
    def validate_terminal_state(self) -> Self:
        if self.status == ProspectiveAttemptCompletionStatus.SUCCESS:
            if self.run is None or self.failure is not None:
                raise ValueError('successful completion requires exactly one run binding')
            terminal_at = self.run.finished_at
        else:
            if self.failure is None or self.run is not None:
                raise ValueError('failed completion requires exactly one explicit failure binding')
            terminal_at = self.failure.failed_at
        if terminal_at > self.run_deadline_at:
            raise ValueError('attempt terminal event occurred after the run deadline')
        return self

    @property
    def terminal_at(self) -> datetime:
        if self.run is not None:
            return self.run.finished_at
        assert self.failure is not None
        return self.failure.failed_at


class ProspectiveAttemptCompletionManifest(StrictModel):
    schema_version: Literal['vaxreplay.prospective-attempt-completion.v0.2'] = (
        PROSPECTIVE_ATTEMPT_COMPLETION_SCHEMA_VERSION
    )
    target_path: Literal['target.json'] = 'target.json'
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    proof_path: Literal['registry-proof.bin'] = 'registry-proof.bin'
    registry_proof: ProspectiveAttemptRegistryProof
    failure_record_path: Literal['failure-record.bin'] | None = None

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if self.registry_proof.event_type != 'completion':
            raise ValueError('attempt completion requires a completion registry proof')
        if (
            self.registry_proof.target_sha256 != self.target_sha256
            or self.registry_proof.target_bytes != self.target_bytes
        ):
            raise ValueError('registry proof does not bind the completion target')
        return self


@dataclass(frozen=True)
class LoadedProspectiveAttemptCompletion:
    root: Path
    manifest: ProspectiveAttemptCompletionManifest
    target: ProspectiveAttemptCompletionTarget
    proof_bytes: bytes
    failure_record: bytes | None
    run: LoadedRunArtifact | None
    manifest_sha256: str


def prospective_attempt_completion_target_sha256(target: ProspectiveAttemptCompletionTarget) -> str:
    return _sha256(canonical_json_bytes(target))


def prospective_attempt_completion_manifest_sha256(
    manifest: ProspectiveAttemptCompletionManifest,
) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_attempt_completion_target(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
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
    run: LoadedRunArtifact | None = None,
    failure: ProspectiveExplicitFailure | None = None,
    failure_record: bytes | None = None,
) -> ProspectiveAttemptCompletionTarget:
    """Build the exact terminal target that an external registry must witness.

    This is deliberately separate from :func:`build_prospective_attempt_completion`:
    a registry must receive and sign the canonical target before its proof can be
    packaged into the completion artifact.  Construction freshly reverifies the
    reservation and either the official run artifact or retained failure bytes.  It
    does not register the target and does not replace the caller-supplied verifier.
    """

    fresh_reservation = _fresh_reservation(
        reservation=reservation,
        release=release,
        release_seal=release_seal,
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
        error_type=ValueError,
    )
    fresh_start_authorization = _fresh_start_authorization(
        start_authorization=start_authorization,
        expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        release=release,
        release_seal=release_seal,
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
        error_type=ValueError,
    )
    fresh_run, _normalized_failure_record = _terminal_material(
        release=release,
        reservation=fresh_reservation,
        start_authorization=fresh_start_authorization,
        system=system,
        runner_policy=runner_policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        run=run,
        failure=failure,
        failure_record=failure_record,
        error_type=ValueError,
    )
    return _completion_target(
        fresh_reservation,
        fresh_start_authorization,
        run=fresh_run,
        failure=failure,
    )


def build_prospective_attempt_completion(
    output_dir: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    registry_proof: ProspectiveAttemptRegistryProof,
    proof_bytes: bytes,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    expected_start_authorization_manifest_sha256: str,
    run: LoadedRunArtifact | None = None,
    failure: ProspectiveExplicitFailure | None = None,
    failure_record: bytes | None = None,
) -> LoadedProspectiveAttemptCompletion:
    target = build_prospective_attempt_completion_target(
        release=release,
        release_seal=release_seal,
        reservation=reservation,
        start_authorization=start_authorization,
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
        run=run,
        failure=failure,
        failure_record=failure_record,
    )
    normalized_failure_record = failure_record if target.status == ProspectiveAttemptCompletionStatus.FAILURE else None
    target_bytes = canonical_json_bytes(target)
    _verify_registry_proof(
        target=target,
        target_bytes=target_bytes,
        registry_proof=registry_proof,
        proof_bytes=proof_bytes,
        earliest_witnessed_at=target.terminal_at,
        latest_witnessed_at=target.run_deadline_at,
        registry_verifier=registry_verifier,
        event_type='completion',
        error_type=ValueError,
    )
    manifest = ProspectiveAttemptCompletionManifest(
        target_sha256=_sha256(target_bytes),
        target_bytes=len(target_bytes),
        registry_proof=registry_proof,
        failure_record_path='failure-record.bin' if normalized_failure_record is not None else None,
    )
    target_root = _publication_target(output_dir)
    with AtomicDirectoryPublication.create(target_root) as publication:
        publication.write_bytes('target.json', target_bytes, mode=0o600)
        publication.write_bytes('registry-proof.bin', proof_bytes, mode=0o600)
        if normalized_failure_record is not None:
            publication.write_bytes('failure-record.bin', normalized_failure_record, mode=0o600)
        publication.write_bytes('completion.json', canonical_json_bytes(manifest), mode=0o600)
        installed_root = publication.publish(root_mode=0o700)
        loaded = load_prospective_attempt_completion(
            installed_root,
            release=release,
            release_seal=release_seal,
            reservation=reservation,
            start_authorization=start_authorization,
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
            run=run,
        )
        publication.commit()
        return loaded


def load_prospective_attempt_completion(
    root: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
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
    run: LoadedRunArtifact | None = None,
) -> LoadedProspectiveAttemptCompletion:
    resolved = _resolve_root(root, 'attempt completion')
    actual = _flat_inventory(resolved, error_type=ProspectiveAttemptIntegrityError)
    manifest_bytes = _read_file(resolved / 'completion.json', _MAX_MODEL_BYTES)
    manifest = _canonical_model(
        manifest_bytes,
        ProspectiveAttemptCompletionManifest,
        'attempt completion manifest',
    )
    expected_files = {'completion.json', 'target.json', 'registry-proof.bin'}
    if manifest.failure_record_path is not None:
        expected_files.add(manifest.failure_record_path)
    if actual != expected_files:
        raise ProspectiveAttemptIntegrityError('attempt completion exact file allowlist mismatch')
    target_bytes = _read_file(resolved / manifest.target_path, _MAX_MODEL_BYTES)
    target = _canonical_model(target_bytes, ProspectiveAttemptCompletionTarget, 'attempt completion target')
    if _sha256(target_bytes) != manifest.target_sha256 or len(target_bytes) != manifest.target_bytes:
        raise ProspectiveAttemptIntegrityError('attempt completion target does not match its manifest')
    fresh_reservation = _fresh_reservation(
        reservation=reservation,
        release=release,
        release_seal=release_seal,
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
        error_type=ProspectiveAttemptIntegrityError,
    )
    fresh_start_authorization = _fresh_start_authorization(
        start_authorization=start_authorization,
        expected_start_authorization_manifest_sha256=(expected_start_authorization_manifest_sha256),
        release=release,
        release_seal=release_seal,
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
        error_type=ProspectiveAttemptIntegrityError,
    )
    failure_record = (
        _read_file(resolved / manifest.failure_record_path, _MAX_FAILURE_BYTES)
        if manifest.failure_record_path is not None
        else None
    )
    failure = target.failure
    fresh_run, normalized_failure_record = _terminal_material(
        release=release,
        reservation=fresh_reservation,
        start_authorization=fresh_start_authorization,
        system=system,
        runner_policy=runner_policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        run=run,
        failure=failure,
        failure_record=failure_record,
        error_type=ProspectiveAttemptIntegrityError,
    )
    expected_target = _completion_target(
        fresh_reservation,
        fresh_start_authorization,
        run=fresh_run,
        failure=failure,
    )
    if target != expected_target:
        raise ProspectiveAttemptIntegrityError('attempt completion is bound to different terminal inputs')
    proof_bytes = _read_file(resolved / manifest.proof_path, _MAX_PROOF_BYTES)
    _verify_registry_proof(
        target=target,
        target_bytes=target_bytes,
        registry_proof=manifest.registry_proof,
        proof_bytes=proof_bytes,
        earliest_witnessed_at=target.terminal_at,
        latest_witnessed_at=target.run_deadline_at,
        registry_verifier=registry_verifier,
        event_type='completion',
        error_type=ProspectiveAttemptIntegrityError,
    )
    return LoadedProspectiveAttemptCompletion(
        root=resolved,
        manifest=manifest,
        target=target,
        proof_bytes=proof_bytes,
        failure_record=normalized_failure_record,
        run=fresh_run,
        manifest_sha256=_sha256(manifest_bytes),
    )


def _fresh_release_context(
    *,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    error_type: type[ValueError],
) -> tuple[LoadedProspectiveCohortRelease, LoadedProspectiveReleaseSeal]:
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
    except ValueError as error:
        raise error_type(f'pre-run release context verification failed: {error}') from error
    if fresh_release != release or fresh_seal != release_seal:
        raise error_type('pre-run release context changed after trusted loading')
    return fresh_release, fresh_seal


def _require_system_policy(
    release: LoadedProspectiveCohortRelease,
    *,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    error_type: type[ValueError],
) -> None:
    if system.response_protocol != PROSPECTIVE_RESPONSE_PROTOCOL:
        raise error_type('prospective reservation requires the prospective response protocol')
    if runner_policy.required_isolation != IsolationTier.OFFICIAL:
        raise error_type('prospective reservation requires an official runner policy')
    attempt_bytes = canonical_json_bytes(attempt_policy)
    if release.attempt_policy != attempt_bytes:
        raise error_type('attempt policy does not match the exact policy packaged in the cohort release')
    if release.verified_admission.admission.attempt_policy_sha256 != prospective_attempt_policy_sha256(attempt_policy):
        raise error_type('attempt policy does not match the prospective admission commitment')


def _fresh_reservation(
    *,
    reservation: LoadedProspectiveAttemptReservation,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    error_type: type[ValueError],
) -> LoadedProspectiveAttemptReservation:
    try:
        fresh = load_prospective_attempt_reservation(
            reservation.root,
            release=release,
            release_seal=release_seal,
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
    except ValueError as error:
        raise error_type(f'attempt reservation reverification failed: {error}') from error
    if fresh != reservation:
        raise error_type('attempt reservation changed after trusted loading')
    return fresh


def _fresh_start_authorization(
    *,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    expected_start_authorization_manifest_sha256: str,
    release: LoadedProspectiveCohortRelease,
    release_seal: LoadedProspectiveReleaseSeal,
    reservation: LoadedProspectiveAttemptReservation,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    decision_receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
    expected_approval_report_sha256: str,
    approval_replay: TierAProspectiveReleaseApprovalReplay,
    release_timestamp_verifier: ProspectiveReleaseTimestampVerifier,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    start_verifier: ProspectiveAttemptStartVerifier,
    error_type: type[ValueError],
) -> LoadedProspectiveAttemptStartAuthorization:
    try:
        fresh = load_prospective_attempt_start_authorization(
            start_authorization.root,
            release=release,
            release_seal=release_seal,
            reservation=reservation,
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
    except ValueError as error:
        raise error_type(f'attempt start-authorization reverification failed: {error}') from error
    if fresh != start_authorization:
        raise error_type('attempt start authorization changed after trusted loading')
    return fresh


def _terminal_material(
    *,
    release: LoadedProspectiveCohortRelease,
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    system: SystemSubmissionManifest,
    runner_policy: RunnerPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    run: LoadedRunArtifact | None,
    failure: ProspectiveExplicitFailure | None,
    failure_record: bytes | None,
    error_type: type[ValueError],
) -> tuple[LoadedRunArtifact | None, bytes | None]:
    if (run is None) == (failure is None):
        raise error_type('attempt completion requires exactly one official run or explicit failure')
    reserved_at = reservation.manifest.registry_proof.witnessed_at
    submissions_open_at = reservation.target.submissions_open_at
    authorized_at = start_authorization.manifest.start_proof.witnessed_at
    deadline = reservation.target.run_deadline_at
    if authorized_at < submissions_open_at or authorized_at >= deadline:
        raise error_type('attempt start authorization is outside the official execution window')
    if run is not None:
        if failure_record is not None:
            raise error_type('successful attempt completion cannot include a failure record')
        try:
            fresh_run = load_run_artifact(
                run.root,
                challenge=release.challenge,
                system=system,
                policy=runner_policy,
                receipt_key=receipt_key,
                expected_receipt_key_id=expected_receipt_key_id,
                require_sealed=True,
            )
        except ValueError as error:
            raise error_type(f'official run reverification failed: {error}') from error
        if fresh_run != run:
            raise error_type('official run changed after trusted loading')
        if fresh_run.receipt.started_at < reserved_at:
            raise error_type('official run started before its externally witnessed reservation')
        if fresh_run.receipt.started_at < submissions_open_at:
            raise error_type('official run started before submissions opened')
        if fresh_run.receipt.started_at < authorized_at:
            raise error_type('official run started before its external start authorization')
        if fresh_run.receipt.finished_at > deadline:
            raise error_type('official run finished after the preregistered deadline')
        return fresh_run, None
    assert failure is not None
    if failure_record is None or not failure_record:
        raise error_type('explicit failure completion requires the retained failure record bytes')
    if len(failure_record) != failure.failure_record_bytes or _sha256(failure_record) != failure.failure_record_sha256:
        raise error_type('retained failure record does not match its explicit failure binding')
    if failure.started_at < reserved_at:
        raise error_type('failed attempt started before its externally witnessed reservation')
    if failure.started_at < submissions_open_at:
        raise error_type('failed attempt started before submissions opened')
    if failure.started_at < authorized_at:
        raise error_type('failed attempt started before its external start authorization')
    if failure.failed_at > deadline:
        raise error_type('failed attempt completed after the preregistered deadline')
    return None, failure_record


def _completion_target(
    reservation: LoadedProspectiveAttemptReservation,
    start_authorization: LoadedProspectiveAttemptStartAuthorization,
    *,
    run: LoadedRunArtifact | None,
    failure: ProspectiveExplicitFailure | None,
) -> ProspectiveAttemptCompletionTarget:
    reserved = reservation.target
    return ProspectiveAttemptCompletionTarget(
        reservation_manifest_sha256=reservation.manifest_sha256,
        reservation_target_sha256=prospective_attempt_reservation_target_sha256(reserved),
        start_authorization_manifest_sha256=start_authorization.manifest_sha256,
        start_authorization_target_sha256=prospective_attempt_start_target_sha256(start_authorization.target),
        start_authorization_proof_sha256=(start_authorization.manifest.start_proof.proof_sha256),
        prospective_release_sha256=reserved.prospective_release_sha256,
        canonical_cohort_id=reserved.canonical_cohort_id,
        track_id=reserved.track_id,
        registered_entry_id=reserved.registered_entry_id,
        attempt_key_sha256=reserved.attempt_key_sha256,
        alias_key_sha256=reserved.alias_key_sha256,
        status=(
            ProspectiveAttemptCompletionStatus.SUCCESS
            if run is not None
            else ProspectiveAttemptCompletionStatus.FAILURE
        ),
        run=ProspectiveRunCompletionBinding.from_run(run) if run is not None else None,
        failure=failure,
        run_deadline_at=reserved.run_deadline_at,
    )


def _verify_start_proof(
    *,
    target: ProspectiveAttemptStartTarget,
    target_bytes: bytes,
    start_proof: ProspectiveAttemptStartProof,
    proof_bytes: bytes,
    start_verifier: ProspectiveAttemptStartVerifier,
    error_type: type[ValueError],
) -> None:
    if start_verifier is None:  # type: ignore[comparison-overlap]
        raise error_type('a trusted external attempt-start verifier is required')
    if (
        start_proof.target_sha256 != _sha256(target_bytes)
        or start_proof.target_bytes != len(target_bytes)
        or start_proof.prospective_release_sha256 != target.prospective_release_sha256
        or start_proof.canonical_cohort_id != target.canonical_cohort_id
        or start_proof.attempt_key_sha256 != target.attempt_key_sha256
        or start_proof.alias_key_sha256 != target.alias_key_sha256
    ):
        raise error_type('attempt start proof does not bind the exact reserved attempt target')
    if start_proof.proof_sha256 != _sha256(proof_bytes) or start_proof.proof_bytes != len(proof_bytes):
        raise error_type('attempt start proof bytes do not match their binding')
    if start_proof.witnessed_at < target.submissions_open_at:
        raise error_type('attempt start authorization predates submissions opening')
    if start_proof.witnessed_at >= target.run_deadline_at:
        raise error_type('attempt start authorization is at or after the run deadline')
    try:
        accepted = start_verifier(start_proof, proof_bytes)
    except Exception as error:
        raise error_type(f'trusted attempt-start verifier failed: {error}') from error
    if accepted is not True:
        raise error_type('trusted attempt-start verifier rejected the authorization')


def _verify_registry_proof(
    *,
    target: ProspectiveAttemptReservationTarget | ProspectiveAttemptCompletionTarget,
    target_bytes: bytes,
    registry_proof: ProspectiveAttemptRegistryProof,
    proof_bytes: bytes,
    earliest_witnessed_at: datetime,
    latest_witnessed_at: datetime,
    registry_verifier: ProspectiveAttemptRegistryVerifier,
    event_type: Literal['reservation', 'completion'],
    error_type: type[ValueError],
) -> None:
    if registry_verifier is None:  # type: ignore[comparison-overlap]
        raise error_type('a trusted global attempt-registry verifier is required')
    if registry_proof.event_type != event_type:
        raise error_type('attempt registry proof has the wrong event type')
    if (
        registry_proof.target_sha256 != _sha256(target_bytes)
        or registry_proof.target_bytes != len(target_bytes)
        or registry_proof.canonical_cohort_id != target.canonical_cohort_id
        or registry_proof.attempt_key_sha256 != target.attempt_key_sha256
        or registry_proof.alias_key_sha256 != target.alias_key_sha256
    ):
        raise error_type('attempt registry proof does not bind the exact target and alias keys')
    if registry_proof.proof_sha256 != _sha256(proof_bytes) or registry_proof.proof_bytes != len(proof_bytes):
        raise error_type('attempt registry proof bytes do not match their binding')
    if registry_proof.witnessed_at < earliest_witnessed_at:
        raise error_type('attempt registry witness predates its prerequisite event')
    if (event_type == 'reservation' and registry_proof.witnessed_at >= latest_witnessed_at) or (
        event_type == 'completion' and registry_proof.witnessed_at > latest_witnessed_at
    ):
        raise error_type('attempt registry witness arrived at or after its allowed deadline')
    try:
        accepted = registry_verifier(registry_proof, proof_bytes)
    except Exception as error:
        raise error_type(f'trusted attempt-registry verifier failed: {error}') from error
    if not accepted:
        raise error_type('trusted attempt-registry verifier rejected the event')


def _attempt_key(
    *,
    release_sha256: str,
    canonical_cohort_id: str,
    track_id: str,
    registered_entry_id: str,
    executable_sha256: str,
) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.prospective-attempt-key.v0.2',
                'prospective_release_sha256': release_sha256,
                'canonical_cohort_id': canonical_cohort_id,
                'track_id': track_id,
                'registered_entry_id': registered_entry_id,
                'executable_sha256': executable_sha256,
            }
        )
    )


def _alias_key(*, canonical_cohort_id: str, track_id: str, executable_core_sha256: str) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.prospective-attempt-alias-key.v0.1',
                'canonical_cohort_id': canonical_cohort_id,
                'track_id': track_id,
                'executable_core_sha256': executable_core_sha256,
            }
        )
    )


def _canonical_model(payload: bytes, model: type[StrictModel], label: str):
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveAttemptIntegrityError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(value):
        raise ProspectiveAttemptIntegrityError(f'{label} must use canonical JSON encoding')
    return value


def _publication_target(output_dir: Path) -> Path:
    return output_dir.expanduser().absolute()


def _resolve_root(root: Path, label: str) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveAttemptIntegrityError(f'{label} root cannot be a symlink')
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise ProspectiveAttemptIntegrityError(f'cannot resolve {label}: {error}') from error
    if not resolved.is_dir():
        raise ProspectiveAttemptIntegrityError(f'{label} root must be a directory')
    return resolved


def _flat_inventory(root: Path, *, error_type: type[ValueError]) -> set[str]:
    files: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise error_type('attempt artifact cannot contain symlinks')
                if entry.is_file(follow_symlinks=False):
                    files.add(entry.name)
                else:
                    raise error_type('attempt artifact can contain only regular files')
    except ValueError:
        raise
    except OSError as error:
        raise error_type(f'cannot inventory attempt artifact: {error}') from error
    return files


def _require_inventory(root: Path, expected: set[str], *, error_type: type[ValueError]) -> None:
    if _flat_inventory(root, error_type=error_type) != expected:
        raise error_type('attempt artifact exact file allowlist mismatch')


def _read_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProspectiveAttemptIntegrityError(f'cannot open {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveAttemptIntegrityError(f'{path.name} is not a regular file')
        if metadata.st_size > maximum_bytes:
            raise ProspectiveAttemptIntegrityError(f'{path.name} exceeds its size limit')
        payload = bytearray()
        while True:
            remaining = maximum_bytes - len(payload)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(payload)
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise ProspectiveAttemptIntegrityError(f'{path.name} exceeds its size limit')
    except OSError as error:
        raise ProspectiveAttemptIntegrityError(f'cannot read {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _require_sha256(
    value: object,
    label: str,
    *,
    error_type: type[ValueError],
) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise error_type(f'{label} must be an exact lowercase SHA-256 digest')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
