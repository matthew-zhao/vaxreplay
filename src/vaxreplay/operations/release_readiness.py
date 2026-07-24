"""Machine-checked Tier A release-readiness evidence.

Applicability is derived from a fixed release scope, not selected by the release
author.  Every applicable gate requires one or more signed statements from public
keys pinned in the separately supplied policy, and every statement binds exact
immutable evidence bytes and the exact release subjects.

Cryptography cannot prove that an organization is genuinely independent, that an
HSM was configured as an audit says, or that an immutable locator remains available.
Those facts remain deployment and governance responsibilities; the verifier proves
that the policy-pinned external authority signed the retained claim and evidence.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

RELEASE_READINESS_POLICY_SCHEMA_VERSION = 'vaxreplay.tier-a-release-readiness-policy.v0.2'
RELEASE_READINESS_EVIDENCE_SCHEMA_VERSION = 'vaxreplay.tier-a-release-readiness-evidence.v0.1'
RELEASE_READINESS_MANIFEST_SCHEMA_VERSION = 'vaxreplay.tier-a-release-readiness-manifest.v0.1'
RELEASE_VERIFICATION_TIME_STATEMENT_SCHEMA_VERSION = 'vaxreplay.release-verification-time-statement.v0.1'
SIGNED_RELEASE_VERIFICATION_TIME_SCHEMA_VERSION = 'vaxreplay.signed-release-verification-time.v0.1'
RELEASE_READINESS_REPORT_SCHEMA_VERSION = 'vaxreplay.tier-a-release-readiness-report.v0.2'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SIGNATURE_DOMAIN = b'VaxReplay Tier A release readiness evidence v0.1\x00'
_VERIFICATION_TIME_SIGNATURE_DOMAIN = b'VaxReplay release verification time v0.1\x00'
_MAX_POLICY_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024

SourceId = Literal['clinicaltrials.gov', 'iedb', 'immport']
BenchmarkTask = Literal[
    'open_set_nomination',
    'antigen_target_prioritization',
    'preclinical_candidate_advancement',
    'early_clinical_arm_prioritization',
]

_BASE_GATES = {
    'backup_verified',
    'capture_witness_service_qualified',
    'checkpoint_gossip_quorum_current',
    'clock_synchronization_healthy',
    'dataset_card_governance_signoff',
    'frozen_case_universe_and_decision_package',
    'legal_rights_approved',
    'monitoring_and_incident_response_qualified',
    'outcome_label_isolation_verified',
    'outcome_label_protocol_frozen',
    'privacy_redistribution_review',
    'private_storage_controls',
    'promotion_and_official_admission_replay_verified',
    'real_capture_schedule_complete',
    'registry_witness_organizational_independence',
    'restore_drill_verified',
    'scientific_selection_protocol_frozen',
    'selection_registry_live',
    'selection_scope_precommitted',
    'signing_keys_hardware_backed',
    'source_worker_sandbox_qualified',
    'source_worker_supply_chain_verified',
    'timestamp_witnesses_independent',
    'trust_anchors_out_of_band',
}
_IEDB_GATES = {'iedb_source_profile_qualified'}
_CLINICALTRIALS_GATES = {
    'clinicaltrials_parent_query_complete',
    'clinicaltrials_source_profile_qualified',
}
_IMMPORT_GATES = {
    'immport_arm_construct_mapping_review',
    'immport_egress_tls_enforced',
    'immport_fd_broker_handoff_qualified',
    'immport_host_memory_controls_qualified',
    'immport_producer_runtime_image_enforced',
    'immport_publisher_time_semantics_accepted',
    'immport_secret_broker_qualified',
    'immport_secret_scanning_zeroization',
    'immport_source_profile_qualified',
}
_MODEL_GATES = {
    'model_harness_identity_policy_frozen',
    'provider_cancellation_qualified',
    'sealed_model_execution_qualified',
}
_OPEN_SET_GATES = {'open_set_nomination_protocol_frozen'}


class ReleaseReadinessError(ValueError):
    """A readiness policy, signature, artifact, or required-gate check failed."""


class _VerificationKey(Protocol):
    def verify(self, signature: bytes, data: bytes) -> None: ...


class ReadinessMaterial(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=1)


class NamedReadinessSubject(StrictModel):
    role: str = Field(pattern=r'^[a-z][a-z0-9._-]{0,127}$')
    material: ReadinessMaterial


class TierAReleaseScope(StrictModel):
    sources: tuple[SourceId, ...] = Field(min_length=1, max_length=3)
    tasks: tuple[BenchmarkTask, ...] = Field(min_length=1, max_length=4)
    includes_model_leaderboard: bool

    @field_validator('sources', 'tasks')
    @classmethod
    def validate_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('release scope entries must be sorted and unique')
        return value


def applicable_gate_ids(scope: TierAReleaseScope) -> tuple[str, ...]:
    gates = set(_BASE_GATES)
    if 'iedb' in scope.sources:
        gates.update(_IEDB_GATES)
    if 'clinicaltrials.gov' in scope.sources:
        gates.update(_CLINICALTRIALS_GATES)
    if 'immport' in scope.sources:
        gates.update(_IMMPORT_GATES)
    if scope.includes_model_leaderboard:
        gates.update(_MODEL_GATES)
    if 'open_set_nomination' in scope.tasks:
        gates.update(_OPEN_SET_GATES)
    return tuple(sorted(gates))


class ReadinessEvidenceAuthority(StrictModel):
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    organization_id: str = Field(pattern=SAFE_ID_PATTERN)
    failure_domain_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    role: Literal['external-evidence-authority'] = 'external-evidence-authority'
    declared_organizer_independent: Literal[True] = True


class ReadinessVerificationTimeAuthority(StrictModel):
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    organization_id: str = Field(pattern=SAFE_ID_PATTERN)
    failure_domain_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    role: Literal['external-verification-time-authority'] = 'external-verification-time-authority'
    declared_organizer_independent: Literal[True] = True


class ReadinessGateRequirement(StrictModel):
    gate_id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,127}$')
    allowed_authority_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    minimum_distinct_authorities: int = Field(ge=1, le=32)
    allowed_media_types: tuple[str, ...] = Field(min_length=1, max_length=32)
    maximum_evidence_age_seconds: int | None = Field(default=None, ge=1, le=10 * 366 * 24 * 60 * 60)

    @field_validator('allowed_authority_ids', 'allowed_media_types')
    @classmethod
    def validate_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('gate allowlists must be sorted and unique')
        if any(not item or len(item) > 200 or item.strip() != item or '\x00' in item for item in value):
            raise ValueError('gate allowlist entries must be bounded nonempty strings')
        return value

    @model_validator(mode='after')
    def validate_quorum(self) -> Self:
        if self.minimum_distinct_authorities > len(self.allowed_authority_ids):
            raise ValueError('gate authority quorum exceeds its allowlist')
        return self


class TierAReleaseReadinessPolicy(StrictModel):
    schema_version: Literal['vaxreplay.tier-a-release-readiness-policy.v0.2'] = RELEASE_READINESS_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    issued_at: datetime
    organizer_organization_id: str = Field(pattern=SAFE_ID_PATTERN)
    organizer_failure_domain_id: str = Field(pattern=SAFE_ID_PATTERN)
    scope: TierAReleaseScope
    authorities: tuple[ReadinessEvidenceAuthority, ...] = Field(min_length=1, max_length=128)
    verification_time_authority: ReadinessVerificationTimeAuthority
    gates: tuple[ReadinessGateRequirement, ...] = Field(min_length=1, max_length=128)

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'readiness policy issued_at')

    @model_validator(mode='after')
    def validate_policy(self) -> Self:
        authority_ids = tuple(item.authority_id for item in self.authorities)
        if authority_ids != tuple(sorted(authority_ids)) or len(authority_ids) != len(set(authority_ids)):
            raise ValueError('readiness authorities must use unique authority_id sort order')
        if len({item.signing_public_key_sha256 for item in self.authorities}) != len(self.authorities):
            raise ValueError('readiness authorities cannot share signing keys')
        if len({item.organization_id for item in self.authorities}) != len(self.authorities):
            raise ValueError('readiness authorities must represent distinct external organizations')
        if len({item.failure_domain_id for item in self.authorities}) != len(self.authorities):
            raise ValueError('readiness authorities must represent distinct external failure domains')
        if any(item.organization_id == self.organizer_organization_id for item in self.authorities):
            raise ValueError('evidence authorities must be policy-declared external organizations')
        if any(item.failure_domain_id == self.organizer_failure_domain_id for item in self.authorities):
            raise ValueError('evidence authorities must use external failure domains')
        time_authority = self.verification_time_authority
        if (
            time_authority.authority_id in set(authority_ids)
            or time_authority.organization_id
            in {self.organizer_organization_id, *(item.organization_id for item in self.authorities)}
            or time_authority.failure_domain_id
            in {
                self.organizer_failure_domain_id,
                *(item.failure_domain_id for item in self.authorities),
            }
            or time_authority.signing_public_key_sha256 in {item.signing_public_key_sha256 for item in self.authorities}
        ):
            raise ValueError('verification-time authority must have a distinct identity, key, and failure domain')
        gate_ids = tuple(item.gate_id for item in self.gates)
        if gate_ids != applicable_gate_ids(self.scope):
            raise ValueError('readiness policy gates must exactly equal scope-derived applicable gates')
        known_authorities = set(authority_ids)
        for gate in self.gates:
            if not set(gate.allowed_authority_ids).issubset(known_authorities):
                raise ValueError('gate references an unknown evidence authority')
        return self


class ReadinessEvidenceStatement(StrictModel):
    schema_version: Literal['vaxreplay.tier-a-release-readiness-evidence.v0.1'] = (
        RELEASE_READINESS_EVIDENCE_SCHEMA_VERSION
    )
    statement_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    issued_at: datetime
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    release_subjects: tuple[NamedReadinessSubject, ...] = Field(min_length=1, max_length=256)
    gate_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    evidence_artifact: ReadinessMaterial
    evidence_media_type: str = Field(min_length=1, max_length=200)
    immutable_locator: str = Field(min_length=1, max_length=512)
    status: Literal['satisfied'] = 'satisfied'
    externally_archived: Literal[True] = True

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'readiness evidence issued_at')

    @field_validator('release_subjects')
    @classmethod
    def validate_subjects(
        cls,
        value: tuple[NamedReadinessSubject, ...],
    ) -> tuple[NamedReadinessSubject, ...]:
        roles = tuple(item.role for item in value)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError('release evidence subjects must use unique role sort order')
        return value

    @field_validator('gate_ids')
    @classmethod
    def validate_gate_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('evidence gate_ids must be sorted and unique')
        if any(re.fullmatch(r'[a-z][a-z0-9_]{0,127}', item) is None for item in value):
            raise ValueError('evidence gate_ids use invalid syntax')
        return value

    @model_validator(mode='after')
    def validate_immutable_locator(self) -> Self:
        if self.immutable_locator != f'urn:sha256:{self.evidence_artifact.sha256}':
            raise ValueError('immutable_locator must be the evidence artifact SHA-256 URN')
        return self


class SignedReadinessEvidence(StrictModel):
    statement: ReadinessEvidenceStatement
    signature_base64: str = Field(min_length=88, max_length=88)

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError('readiness evidence signature is not canonical base64') from error
        if len(decoded) != 64 or base64.b64encode(decoded).decode('ascii') != value:
            raise ValueError('readiness evidence signature must contain exactly 64 bytes')
        return value


class ReleaseVerificationTimeStatement(StrictModel):
    schema_version: Literal['vaxreplay.release-verification-time-statement.v0.1'] = (
        RELEASE_VERIFICATION_TIME_STATEMENT_SCHEMA_VERSION
    )
    statement_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_at: datetime

    @field_validator('verified_at')
    @classmethod
    def validate_verified_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'release verification time')


class SignedReleaseVerificationTime(StrictModel):
    schema_version: Literal['vaxreplay.signed-release-verification-time.v0.1'] = (
        SIGNED_RELEASE_VERIFICATION_TIME_SCHEMA_VERSION
    )
    statement: ReleaseVerificationTimeStatement
    signature_base64: str = Field(min_length=88, max_length=88)

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError('verification-time signature is not canonical base64') from error
        if len(decoded) != 64 or base64.b64encode(decoded).decode('ascii') != value:
            raise ValueError('verification-time signature must contain exactly 64 bytes')
        return value


class TierAReleaseReadinessManifest(StrictModel):
    schema_version: Literal['vaxreplay.tier-a-release-readiness-manifest.v0.1'] = (
        RELEASE_READINESS_MANIFEST_SCHEMA_VERSION
    )
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    created_at: datetime
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope: TierAReleaseScope
    subjects: tuple[NamedReadinessSubject, ...] = Field(min_length=1, max_length=256)
    evidence: tuple[SignedReadinessEvidence, ...] = Field(min_length=1, max_length=4096)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'readiness manifest created_at')

    @field_validator('subjects')
    @classmethod
    def validate_subjects(
        cls,
        value: tuple[NamedReadinessSubject, ...],
    ) -> tuple[NamedReadinessSubject, ...]:
        roles = tuple(item.role for item in value)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError('readiness subjects must use unique role sort order')
        return value

    @model_validator(mode='after')
    def validate_evidence_inventory(self) -> Self:
        statement_ids = tuple(item.statement.statement_id for item in self.evidence)
        if statement_ids != tuple(sorted(statement_ids)) or len(statement_ids) != len(set(statement_ids)):
            raise ValueError('readiness evidence must use unique statement_id sort order')
        return self


class TierAReleaseReadinessReport(StrictModel):
    schema_version: Literal['vaxreplay.tier-a-release-readiness-report.v0.2'] = RELEASE_READINESS_REPORT_SCHEMA_VERSION
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_at: datetime
    subject_count: int = Field(ge=1)
    applicable_gate_count: int = Field(ge=1)
    evidence_statement_count: int = Field(ge=1)
    evidence_artifact_count: int = Field(ge=1)
    every_signature_verified: Literal[True] = True
    every_artifact_verified: Literal[True] = True
    every_release_subject_verified: Literal[True] = True
    every_applicable_gate_authority_claim_verified: Literal[True] = True
    authority_keys_policy_pinned: Literal[True] = True
    verification_time_attestation_verified: Literal[True] = True
    external_organizational_independence_cryptographically_proven: Literal[False] = False
    external_archive_availability_verified: Literal[False] = False
    machine_readiness_evidence_verified: Literal[True] = True
    tier_a_release_ready_claimed_by_policy_authorities: Literal[True] = True
    deployment_tier_a_status_independently_determined: Literal[False] = False

    @field_validator('verified_at')
    @classmethod
    def validate_verified_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'readiness verification time')


def readiness_evidence_signature_payload(statement: ReadinessEvidenceStatement) -> bytes:
    return _SIGNATURE_DOMAIN + canonical_json_bytes(statement)


def release_verification_time_signature_payload(
    statement: ReleaseVerificationTimeStatement,
) -> bytes:
    return _VERIFICATION_TIME_SIGNATURE_DOMAIN + canonical_json_bytes(statement)


def verify_tier_a_release_readiness(
    *,
    policy_bytes: bytes,
    expected_policy_sha256: str,
    manifest_bytes: bytes,
    release_subject_bytes: Mapping[str, bytes],
    evidence_artifact_bytes: Mapping[str, bytes],
    authority_public_key_bytes: Mapping[str, bytes],
    verification_time_evidence_bytes: bytes,
    verification_time_public_key_bytes: bytes,
    verified_at: datetime,
) -> TierAReleaseReadinessReport:
    """Verify all scope-derived gates, signed claims, retained keys, and artifacts."""

    release_subject_bytes = _snapshot_exact_bytes_mapping(
        release_subject_bytes,
        'retained release subjects',
    )
    evidence_artifact_bytes = _snapshot_exact_bytes_mapping(
        evidence_artifact_bytes,
        'readiness evidence artifacts',
    )
    authority_public_key_bytes = _snapshot_exact_bytes_mapping(
        authority_public_key_bytes,
        'readiness authority public keys',
    )
    policy = _parse_canonical_model(
        policy_bytes,
        TierAReleaseReadinessPolicy,
        'readiness policy',
        maximum=_MAX_POLICY_BYTES,
    )
    manifest = _parse_canonical_model(
        manifest_bytes,
        TierAReleaseReadinessManifest,
        'readiness manifest',
        maximum=_MAX_MANIFEST_BYTES,
    )
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    verification_time = aware_utc(verified_at, 'readiness verification time')
    if re.fullmatch(_SHA256_PATTERN, expected_policy_sha256) is None or policy_digest != expected_policy_sha256:
        raise ReleaseReadinessError('readiness policy differs from its out-of-band expected digest')
    if manifest.policy_sha256 != policy_digest or manifest.scope != policy.scope:
        raise ReleaseReadinessError('readiness manifest differs from the exact expected policy or scope')
    if manifest.created_at < policy.issued_at:
        raise ReleaseReadinessError('readiness manifest predates its policy')
    if manifest.created_at > verification_time:
        raise ReleaseReadinessError('readiness manifest claims a future assembly time')

    time_evidence = _parse_canonical_model(
        verification_time_evidence_bytes,
        SignedReleaseVerificationTime,
        'release verification-time evidence',
        maximum=1024 * 1024,
    )
    time_statement = time_evidence.statement
    time_authority = policy.verification_time_authority
    if (
        not isinstance(verification_time_public_key_bytes, bytes)
        or len(verification_time_public_key_bytes) != 32
        or hashlib.sha256(verification_time_public_key_bytes).hexdigest() != time_authority.signing_public_key_sha256
        or time_statement.authority_id != time_authority.authority_id
        or time_statement.release_id != manifest.release_id
        or time_statement.policy_sha256 != policy_digest
        or time_statement.readiness_manifest_sha256 != manifest_digest
        or time_statement.verified_at != verification_time
    ):
        raise ReleaseReadinessError(
            'verification-time evidence differs from its policy, release, manifest, or expected time'
        )
    try:
        _load_ed25519_public_key(verification_time_public_key_bytes).verify(
            base64.b64decode(time_evidence.signature_base64, validate=True),
            release_verification_time_signature_payload(time_statement),
        )
    except Exception as error:
        raise ReleaseReadinessError('verification-time evidence signature verification failed') from error
    scope_digest = hashlib.sha256(canonical_json_bytes(manifest.scope)).hexdigest()
    expected_subjects = {item.role: item.material for item in manifest.subjects}
    if set(release_subject_bytes) != set(expected_subjects):
        raise ReleaseReadinessError('retained release subject inventory differs from readiness manifest')
    for role, binding in expected_subjects.items():
        subject_bytes = release_subject_bytes[role]
        if (
            not isinstance(subject_bytes, bytes)
            or len(subject_bytes) != binding.byte_count
            or hashlib.sha256(subject_bytes).hexdigest() != binding.sha256
        ):
            raise ReleaseReadinessError(f'release subject differs from its readiness binding: {role}')

    authorities = {item.authority_id: item for item in policy.authorities}
    if set(authority_public_key_bytes) != set(authorities):
        raise ReleaseReadinessError('retained authority key inventory differs from policy')
    public_keys: dict[str, _VerificationKey] = {}
    for authority_id, authority in authorities.items():
        key_bytes = authority_public_key_bytes[authority_id]
        if not isinstance(key_bytes, bytes) or len(key_bytes) != 32:
            raise ReleaseReadinessError('readiness authority public keys must contain exactly 32 bytes')
        if hashlib.sha256(key_bytes).hexdigest() != authority.signing_public_key_sha256:
            raise ReleaseReadinessError('readiness authority public key differs from policy')
        public_keys[authority_id] = _load_ed25519_public_key(key_bytes)

    requirements = {item.gate_id: item for item in policy.gates}
    used_artifacts: set[str] = set()
    authorities_by_gate: dict[str, set[str]] = {gate_id: set() for gate_id in requirements}
    for envelope in manifest.evidence:
        statement = envelope.statement
        if (
            statement.release_id != manifest.release_id
            or statement.policy_sha256 != policy_digest
            or statement.scope_sha256 != scope_digest
            or statement.release_subjects != manifest.subjects
        ):
            raise ReleaseReadinessError('readiness evidence binds a different release, policy, scope, or subject')
        if (
            statement.issued_at < policy.issued_at
            or statement.issued_at > manifest.created_at
            or statement.issued_at > verification_time
        ):
            raise ReleaseReadinessError('readiness evidence falls outside the policy-to-release interval')
        authority = authorities.get(statement.authority_id)
        if authority is None:
            raise ReleaseReadinessError('readiness evidence uses an unknown authority')
        try:
            public_keys[statement.authority_id].verify(
                base64.b64decode(envelope.signature_base64, validate=True),
                readiness_evidence_signature_payload(statement),
            )
        except Exception as error:
            raise ReleaseReadinessError('readiness evidence signature verification failed') from error
        artifact_bytes = evidence_artifact_bytes.get(statement.evidence_artifact.sha256)
        if artifact_bytes is None:
            raise ReleaseReadinessError('readiness evidence artifact bytes are missing')
        if (
            not isinstance(artifact_bytes, bytes)
            or len(artifact_bytes) != statement.evidence_artifact.byte_count
            or hashlib.sha256(artifact_bytes).hexdigest() != statement.evidence_artifact.sha256
        ):
            raise ReleaseReadinessError('readiness evidence artifact differs from its signed binding')
        used_artifacts.add(statement.evidence_artifact.sha256)
        for gate_id in statement.gate_ids:
            requirement = requirements.get(gate_id)
            if requirement is None:
                raise ReleaseReadinessError('readiness evidence references a non-applicable gate')
            if statement.authority_id not in requirement.allowed_authority_ids:
                raise ReleaseReadinessError('readiness evidence authority is not allowed for a gate')
            if statement.evidence_media_type not in requirement.allowed_media_types:
                raise ReleaseReadinessError('readiness evidence media type is not allowed for a gate')
            if requirement.maximum_evidence_age_seconds is not None:
                age_seconds = (verification_time - statement.issued_at).total_seconds()
                if age_seconds > requirement.maximum_evidence_age_seconds:
                    raise ReleaseReadinessError('readiness evidence is older than its gate permits')
            authorities_by_gate[gate_id].add(statement.authority_id)

    if set(evidence_artifact_bytes) != used_artifacts:
        raise ReleaseReadinessError('retained evidence artifact inventory contains unreferenced bytes')
    for gate_id, requirement in requirements.items():
        if len(authorities_by_gate[gate_id]) < requirement.minimum_distinct_authorities:
            raise ReleaseReadinessError(f'applicable readiness gate lacks its authority quorum: {gate_id}')

    return TierAReleaseReadinessReport(
        release_id=manifest.release_id,
        policy_id=policy.policy_id,
        policy_sha256=policy_digest,
        manifest_sha256=manifest_digest,
        verified_at=verification_time,
        subject_count=len(manifest.subjects),
        applicable_gate_count=len(requirements),
        evidence_statement_count=len(manifest.evidence),
        evidence_artifact_count=len(used_artifacts),
    )


def _snapshot_exact_bytes_mapping(value: Mapping[str, bytes], label: str) -> dict[str, bytes]:
    """Freeze one potentially stateful mapping before any verification read."""

    if not isinstance(value, Mapping):
        raise ReleaseReadinessError(f'{label} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception as error:
        raise ReleaseReadinessError(f'{label} could not be snapshotted') from error
    result: dict[str, bytes] = {}
    for key, payload in items:
        if type(key) is not str or type(payload) is not bytes or key in result:
            raise ReleaseReadinessError(f'{label} must use unique exact-string keys and exact-bytes values')
        result[key] = payload
    return result


def _load_ed25519_public_key(key_bytes: bytes) -> _VerificationKey:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:  # pragma: no cover - deployment dependency failure
        raise ReleaseReadinessError('cryptography is required to verify readiness evidence') from error
    return Ed25519PublicKey.from_public_bytes(key_bytes)


def _parse_canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
    *,
    maximum: int,
) -> ModelT:
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise ReleaseReadinessError(f'{label} must be nonempty and at most {maximum} bytes')
    try:
        parsed = model.model_validate_json(payload)
    except ValueError as error:
        raise ReleaseReadinessError(f'{label} does not match its strict schema') from error
    if payload != canonical_json_bytes(parsed):
        raise ReleaseReadinessError(f'{label} must use exact canonical JSON bytes')
    return parsed
