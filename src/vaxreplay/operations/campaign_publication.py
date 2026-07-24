"""Canonical Tier-A campaign publication manifest and offline verifier.

The signed manifest is a bill of materials, never its own root of trust.  The
offline verifier requires a separately supplied trust policy containing the
release key and independent publication-authority keys.  Embedded or adjacent
keys that are absent from that policy are ignored and cannot authorize a release.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.checkpoint_gossip import (
    CheckpointGossipMonitorPolicy,
    GossipComparisonPolicy,
    WitnessGossipStreamPolicy,
    verify_gossip_bootstrap_head,
)
from vaxreplay.operations.clock_health import ClockHealthPolicy
from vaxreplay.operations.hermetic_execution import HermeticOciEnvironment, HermeticSandboxPolicy
from vaxreplay.operations.operator_trust import IsolatedProcessConfig
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc
from vaxreplay.operations.selection_registry import (
    SelectionRegistryPolicy,
    SelectionRegistryTrustPolicy,
    verify_signed_registry_checkpoint,
)
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.operations.witness_service import (
    verify_witness_service_artifact,
    verify_witness_service_signed_checkpoint,
)
from vaxreplay.operations.witness_service_schema import (
    WitnessedSignedRegistryCheckpoint,
    WitnessServicePolicy,
    WitnessServiceTrustPolicy,
)

CAMPAIGN_PUBLICATION_MANIFEST_SCHEMA_VERSION = 'vaxreplay.campaign-publication-manifest.v0.2'
SIGNED_CAMPAIGN_PUBLICATION_MANIFEST_SCHEMA_VERSION = 'vaxreplay.signed-campaign-publication-manifest.v0.2'
CAMPAIGN_PUBLICATION_TRUST_POLICY_SCHEMA_VERSION = 'vaxreplay.campaign-publication-trust-policy.v0.2'
PUBLICATION_RECEIPT_STATEMENT_SCHEMA_VERSION = 'vaxreplay.publication-receipt-statement.v0.1'
SIGNED_PUBLICATION_RECEIPT_SCHEMA_VERSION = 'vaxreplay.signed-publication-receipt.v0.1'
PUBLICATION_VERIFICATION_REPORT_SCHEMA_VERSION = 'vaxreplay.publication-verification-report.v0.2'

_MANIFEST_SIGNATURE_DOMAIN = b'VaxReplay campaign publication manifest v0.2\x00'
_RECEIPT_SIGNATURE_DOMAIN = b'VaxReplay independent publication receipt v0.1\x00'
_WORKER_PROVENANCE_SIGNATURE_DOMAIN = b'VaxReplay normalized worker image provenance v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_ARTIFACTS = 4096
MAX_PUBLICATION_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_PUBLICATION_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_ARTIFACT_BYTES = MAX_PUBLICATION_ARTIFACT_BYTES
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_POLICY_ARTIFACT_BYTES = 64 * 1024 * 1024


class CampaignPublicationError(ValueError):
    """Publication trust, manifest, receipt, or artifact verification failed closed."""


class PublicationArtifactBinding(StrictModel):
    artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    role: Literal[
        'registry_policy',
        'registry_trust_policy',
        'registry_bootstrap_head',
        'registry_bootstrap_witness',
        'witness_policy',
        'witness_trust_policy',
        'witness_bootstrap_head',
        'gossip_monitor_policy',
        'gossip_comparison_policy',
        'gossip_bootstrap_head',
        'clock_health_policy',
        'clock_health_process',
        'external_signer_process',
        'operator_executable',
        'worker_implementation',
        'worker_environment',
        'worker_sandbox_policy',
        'worker_image_sbom',
        'worker_image_provenance',
        'release_archive',
        'release_archive_index',
    ]
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=_MAX_ARTIFACT_BYTES)


class RuntimeProcessPublicationBinding(StrictModel):
    process_id: str = Field(pattern=SAFE_ID_PATTERN)
    purpose: Literal['clock_health', 'external_signer']
    process_config_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    executable_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)


class RegistryPublicationBinding(StrictModel):
    registry_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    bootstrap_head_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    bootstrap_witness_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_public_key_sha256s: tuple[str, ...] = Field(min_length=1, max_length=64)

    @field_validator('signing_public_key_sha256s')
    @classmethod
    def validate_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('registry key digests must be sorted and unique')
        if any(not _is_sha256(item) for item in value):
            raise ValueError('registry key digests must be lowercase SHA-256')
        return value


class WitnessPublicationBinding(StrictModel):
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    witness_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    bootstrap_head_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)


class GossipMonitorPublicationBinding(StrictModel):
    monitor_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    report_signing_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)


class GossipBootstrapHeadBinding(StrictModel):
    stream_id: str = Field(pattern=SAFE_ID_PATTERN)
    artifact_id: str = Field(pattern=SAFE_ID_PATTERN)


class GossipPublicationBinding(StrictModel):
    comparison_policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    monitors: tuple[GossipMonitorPublicationBinding, ...] = Field(min_length=2, max_length=32)
    bootstrap_heads: tuple[GossipBootstrapHeadBinding, ...] = Field(min_length=1, max_length=512)

    @field_validator('monitors')
    @classmethod
    def validate_monitors(
        cls,
        value: tuple[GossipMonitorPublicationBinding, ...],
    ) -> tuple[GossipMonitorPublicationBinding, ...]:
        ids = tuple(item.monitor_id for item in value)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError('gossip monitor bindings must be sorted and unique')
        key_digests = tuple(item.report_signing_public_key_sha256 for item in value)
        if len(key_digests) != len(set(key_digests)):
            raise ValueError('independent gossip monitors must not share report-signing keys')
        return value

    @field_validator('bootstrap_heads')
    @classmethod
    def validate_heads(
        cls,
        value: tuple[GossipBootstrapHeadBinding, ...],
    ) -> tuple[GossipBootstrapHeadBinding, ...]:
        stream_ids = tuple(item.stream_id for item in value)
        artifact_ids = tuple(item.artifact_id for item in value)
        if stream_ids != tuple(sorted(stream_ids)) or len(stream_ids) != len(set(stream_ids)):
            raise ValueError('gossip bootstrap heads must be sorted and unique by stream ID')
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('gossip bootstrap-head artifact IDs must be unique')
        return value


class WorkerPublicationBinding(StrictModel):
    worker_id: str = Field(pattern=SAFE_ID_PATTERN)
    purpose: Literal['source_verifier', 'adapter']
    implementation_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    environment_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    sandbox_policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    sbom_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    provenance_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    image_ref: str = Field(pattern=r'^.+@sha256:[0-9a-f]{64}$')
    resolved_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    provenance_builder_id: str = Field(pattern=SAFE_ID_PATTERN)
    provenance_build_type: str = Field(pattern=SAFE_ID_PATTERN)


class WorkerImageProvenance(StrictModel):
    """Normalized, release-bound facts extracted from signed OCI provenance."""

    schema_version: Literal['vaxreplay.worker-image-provenance.v0.1'] = 'vaxreplay.worker-image-provenance.v0.1'
    provenance_format: Literal['in_toto_slsa_v1'] = 'in_toto_slsa_v1'
    builder_id: str = Field(pattern=SAFE_ID_PATTERN)
    build_type: str = Field(pattern=SAFE_ID_PATTERN)
    created_at: datetime
    image_ref: str = Field(pattern=r'^.+@sha256:[0-9a-f]{64}$')
    resolved_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    image_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    sbom_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_repository_uri: str = Field(pattern=r'^https://[^\s?#]+(?:/[^\s?#]*)?$')
    source_commit_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'worker provenance creation time')


class SignedWorkerImageProvenance(StrictModel):
    schema_version: Literal['vaxreplay.signed-worker-image-provenance.v0.1'] = (
        'vaxreplay.signed-worker-image-provenance.v0.1'
    )
    statement: WorkerImageProvenance
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        _decode_base64(value, expected_bytes=64, label='worker provenance signature')
        return value


class CampaignPublicationManifest(StrictModel):
    """Complete, canonical release bill of materials signed by release authority."""

    schema_version: Literal['vaxreplay.campaign-publication-manifest.v0.2'] = (
        CAMPAIGN_PUBLICATION_MANIFEST_SCHEMA_VERSION
    )
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    created_at: datetime
    release_authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    artifacts: tuple[PublicationArtifactBinding, ...] = Field(min_length=1, max_length=_MAX_ARTIFACTS)
    registry: RegistryPublicationBinding
    witnesses: tuple[WitnessPublicationBinding, ...] = Field(min_length=1, max_length=32)
    gossip: GossipPublicationBinding
    clock_health_policy_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    runtime_processes: tuple[RuntimeProcessPublicationBinding, ...] = Field(
        min_length=2,
        max_length=256,
    )
    workers: tuple[WorkerPublicationBinding, ...] = Field(min_length=1, max_length=256)
    archive_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    archive_index_artifact_id: str = Field(pattern=SAFE_ID_PATTERN)

    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'publication manifest created_at')

    @field_validator('artifacts')
    @classmethod
    def validate_artifacts(
        cls,
        value: tuple[PublicationArtifactBinding, ...],
    ) -> tuple[PublicationArtifactBinding, ...]:
        ids = tuple(item.artifact_id for item in value)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError('publication artifacts must be sorted and unique by artifact_id')
        if sum(item.byte_count for item in value) > MAX_PUBLICATION_TOTAL_ARTIFACT_BYTES:
            raise ValueError('publication artifact inventory exceeds the total-byte limit')
        return value

    @field_validator('witnesses')
    @classmethod
    def validate_witnesses(
        cls,
        value: tuple[WitnessPublicationBinding, ...],
    ) -> tuple[WitnessPublicationBinding, ...]:
        identities = tuple((item.authority_id, item.witness_id) for item in value)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError('witness bindings must be sorted and unique')
        return value

    @field_validator('workers')
    @classmethod
    def validate_workers(
        cls,
        value: tuple[WorkerPublicationBinding, ...],
    ) -> tuple[WorkerPublicationBinding, ...]:
        ids = tuple(item.worker_id for item in value)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError('worker bindings must be sorted and unique')
        return value

    @field_validator('runtime_processes')
    @classmethod
    def validate_runtime_processes(
        cls,
        value: tuple[RuntimeProcessPublicationBinding, ...],
    ) -> tuple[RuntimeProcessPublicationBinding, ...]:
        identities = tuple((item.purpose, item.process_id) for item in value)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError('runtime process bindings must be sorted and unique')
        config_ids = tuple(item.process_config_artifact_id for item in value)
        if len(config_ids) != len(set(config_ids)):
            raise ValueError('runtime process configurations must be bound once')
        return value

    @model_validator(mode='after')
    def validate_references(self) -> Self:
        roles = {item.artifact_id: item.role for item in self.artifacts}

        def require(artifact_id: str, role: str) -> None:
            if roles.get(artifact_id) != role:
                raise ValueError(f'artifact {artifact_id!r} is absent or has the wrong role')

        registry = self.registry
        require(registry.policy_artifact_id, 'registry_policy')
        require(registry.trust_policy_artifact_id, 'registry_trust_policy')
        require(registry.bootstrap_head_artifact_id, 'registry_bootstrap_head')
        require(registry.bootstrap_witness_artifact_id, 'registry_bootstrap_witness')
        for witness in self.witnesses:
            require(witness.policy_artifact_id, 'witness_policy')
            require(witness.trust_policy_artifact_id, 'witness_trust_policy')
            require(witness.bootstrap_head_artifact_id, 'witness_bootstrap_head')
        require(self.gossip.comparison_policy_artifact_id, 'gossip_comparison_policy')
        for monitor in self.gossip.monitors:
            require(monitor.policy_artifact_id, 'gossip_monitor_policy')
        for head in self.gossip.bootstrap_heads:
            require(head.artifact_id, 'gossip_bootstrap_head')
        require(self.clock_health_policy_artifact_id, 'clock_health_policy')
        for runtime in self.runtime_processes:
            require(
                runtime.process_config_artifact_id,
                'clock_health_process' if runtime.purpose == 'clock_health' else 'external_signer_process',
            )
            require(runtime.executable_artifact_id, 'operator_executable')
        for worker in self.workers:
            require(worker.implementation_artifact_id, 'worker_implementation')
            require(worker.environment_artifact_id, 'worker_environment')
            require(worker.sandbox_policy_artifact_id, 'worker_sandbox_policy')
            require(worker.sbom_artifact_id, 'worker_image_sbom')
            require(worker.provenance_artifact_id, 'worker_image_provenance')
        require(self.archive_artifact_id, 'release_archive')
        require(self.archive_index_artifact_id, 'release_archive_index')
        referenced = {
            registry.policy_artifact_id,
            registry.trust_policy_artifact_id,
            registry.bootstrap_head_artifact_id,
            registry.bootstrap_witness_artifact_id,
            self.gossip.comparison_policy_artifact_id,
            *(item.artifact_id for item in self.gossip.bootstrap_heads),
            self.clock_health_policy_artifact_id,
            *(item.process_config_artifact_id for item in self.runtime_processes),
            *(item.executable_artifact_id for item in self.runtime_processes),
            self.archive_artifact_id,
            self.archive_index_artifact_id,
            *(item.policy_artifact_id for item in self.witnesses),
            *(item.trust_policy_artifact_id for item in self.witnesses),
            *(item.bootstrap_head_artifact_id for item in self.witnesses),
            *(item.policy_artifact_id for item in self.gossip.monitors),
            *(item.implementation_artifact_id for item in self.workers),
            *(item.environment_artifact_id for item in self.workers),
            *(item.sandbox_policy_artifact_id for item in self.workers),
            *(item.sbom_artifact_id for item in self.workers),
            *(item.provenance_artifact_id for item in self.workers),
        }
        if referenced != set(roles):
            raise ValueError('publication manifest contains unreferenced artifacts')
        return self


class SignedCampaignPublicationManifest(StrictModel):
    schema_version: Literal['vaxreplay.signed-campaign-publication-manifest.v0.2'] = (
        SIGNED_CAMPAIGN_PUBLICATION_MANIFEST_SCHEMA_VERSION
    )
    manifest: CampaignPublicationManifest
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        _decode_base64(value, expected_bytes=64, label='release manifest signature')
        return value


class PublicationTrustedKey(StrictModel):
    key_id: str = Field(pattern=SAFE_ID_PATTERN)
    public_key_base64: str
    valid_from: datetime
    valid_until: datetime | None = None

    @field_validator('public_key_base64')
    @classmethod
    def validate_public_key(cls, value: str) -> str:
        _decode_base64(value, expected_bytes=32, label='publication public key')
        return value

    @field_validator('valid_from', 'valid_until')
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else aware_utc(value, 'publication key validity')

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError('publication key validity interval is inverted')
        return self


class PublicationAuthorityTrust(StrictModel):
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    organization_id: str = Field(pattern=SAFE_ID_PATTERN)
    failure_domain_id: str = Field(pattern=SAFE_ID_PATTERN)
    keys: tuple[PublicationTrustedKey, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def validate_keys(self) -> Self:
        ids = tuple(item.key_id for item in self.keys)
        if len(ids) != len(set(ids)):
            raise ValueError('publication authority contains duplicate key IDs')
        return self


class CampaignPublicationTrustPolicy(StrictModel):
    """Out-of-band root; it must not be loaded from the release archive."""

    schema_version: Literal['vaxreplay.campaign-publication-trust-policy.v0.2'] = (
        CAMPAIGN_PUBLICATION_TRUST_POLICY_SCHEMA_VERSION
    )
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_organization_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_failure_domain_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_keys: tuple[PublicationTrustedKey, ...] = Field(min_length=1, max_length=16)
    worker_build_authorities: tuple[PublicationAuthorityTrust, ...] = Field(min_length=1, max_length=32)
    publication_authorities: tuple[PublicationAuthorityTrust, ...] = Field(min_length=2, max_length=32)
    precommitted_control_artifacts: tuple[PublicationArtifactBinding, ...] = Field(
        min_length=1,
        max_length=_MAX_ARTIFACTS,
    )
    required_workers: tuple[WorkerPublicationBinding, ...] = Field(min_length=1, max_length=256)
    minimum_independent_receipts: int = Field(ge=2, le=32)
    max_receipt_delay_seconds: int = Field(default=30 * 24 * 60 * 60, ge=1, le=365 * 24 * 60 * 60)

    @model_validator(mode='after')
    def validate_policy(self) -> Self:
        release_ids = tuple(item.key_id for item in self.release_keys)
        if len(release_ids) != len(set(release_ids)):
            raise ValueError('release trust contains duplicate key IDs')
        authority_ids = tuple(item.authority_id for item in self.publication_authorities)
        if len(authority_ids) != len(set(authority_ids)):
            raise ValueError('publication trust contains duplicate authority IDs')
        if self.release_authority_id in set(authority_ids):
            raise ValueError('publication receipt authorities must be independent of release authority')
        builder_ids = tuple(item.authority_id for item in self.worker_build_authorities)
        if len(builder_ids) != len(set(builder_ids)):
            raise ValueError('worker build trust contains duplicate authority IDs')
        if self.release_authority_id in set(builder_ids) or set(builder_ids) & set(authority_ids):
            raise ValueError('worker builders must be independent of release and publication authorities')
        if self.minimum_independent_receipts > len(authority_ids):
            raise ValueError('receipt quorum exceeds configured independent authorities')
        all_authorities = (*self.worker_build_authorities, *self.publication_authorities)
        all_ids = (self.release_authority_id, *(item.authority_id for item in all_authorities))
        all_organizations = (
            self.release_organization_id,
            *(item.organization_id for item in all_authorities),
        )
        all_failure_domains = (
            self.release_failure_domain_id,
            *(item.failure_domain_id for item in all_authorities),
        )
        if len(all_ids) != len(set(all_ids)):
            raise ValueError('release, builder, and publisher authority IDs must be distinct')
        if len(all_organizations) != len(set(all_organizations)):
            raise ValueError('release, builder, and publisher organizations must be distinct')
        if len(all_failure_domains) != len(set(all_failure_domains)):
            raise ValueError('release, builder, and publisher failure domains must be distinct')
        control_ids = tuple(item.artifact_id for item in self.precommitted_control_artifacts)
        if control_ids != tuple(sorted(control_ids)) or len(control_ids) != len(set(control_ids)):
            raise ValueError('precommitted control artifacts must be sorted and unique')
        if any(
            item.role in {'release_archive', 'release_archive_index'} for item in self.precommitted_control_artifacts
        ):
            raise ValueError('release archive bytes are not precommitted control artifacts')
        worker_ids = tuple(item.worker_id for item in self.required_workers)
        if worker_ids != tuple(sorted(worker_ids)) or len(worker_ids) != len(set(worker_ids)):
            raise ValueError('required workers must be sorted and unique')
        release_key_digests = {
            hashlib.sha256(_decode_base64(item.public_key_base64, expected_bytes=32, label='release key')).hexdigest()
            for item in self.release_keys
        }
        authority_key_owners: dict[str, str] = {}
        for authority in (*self.worker_build_authorities, *self.publication_authorities):
            for key in authority.keys:
                digest = hashlib.sha256(
                    _decode_base64(
                        key.public_key_base64,
                        expected_bytes=32,
                        label='publication authority key',
                    )
                ).hexdigest()
                if digest in release_key_digests or digest in authority_key_owners:
                    raise ValueError('independent publication authorities must use distinct keys')
                authority_key_owners[digest] = authority.authority_id
        return self


class PublicationReceiptStatement(StrictModel):
    schema_version: Literal['vaxreplay.publication-receipt-statement.v0.1'] = (
        PUBLICATION_RECEIPT_STATEMENT_SCHEMA_VERSION
    )
    receipt_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    signed_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    signed_manifest_bytes: int = Field(gt=0, le=_MAX_MANIFEST_BYTES)
    archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    archive_bytes: int = Field(gt=0, le=_MAX_ARTIFACT_BYTES)
    published_at: datetime
    publication_uri: str = Field(pattern=r'^https://[^\s?#]+(?:/[^\s?#]*)?$')

    @field_validator('published_at')
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'publication receipt time')


class SignedPublicationReceipt(StrictModel):
    schema_version: Literal['vaxreplay.signed-publication-receipt.v0.1'] = SIGNED_PUBLICATION_RECEIPT_SCHEMA_VERSION
    statement: PublicationReceiptStatement
    signature_base64: str

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        _decode_base64(value, expected_bytes=64, label='publication receipt signature')
        return value


class PublicationVerificationReport(StrictModel):
    schema_version: Literal['vaxreplay.publication-verification-report.v0.2'] = (
        PUBLICATION_VERIFICATION_REPORT_SCHEMA_VERSION
    )
    campaign_id: str = Field(pattern=SAFE_ID_PATTERN)
    release_id: str = Field(pattern=SAFE_ID_PATTERN)
    signed_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_artifact_count: int = Field(gt=0)
    verified_worker_count: int = Field(gt=0)
    independent_publication_authority_ids: tuple[str, ...] = Field(min_length=2)
    publication_organization_ids: tuple[str, ...] = Field(min_length=2)
    out_of_band_trust_policy_digest_verified: Literal[True] = True
    out_of_band_release_signature_verified: Literal[True] = True
    out_of_band_control_artifacts_verified: Literal[True] = True
    retained_runtime_process_configurations_verified: Literal[True] = True
    retained_runtime_executables_verified: Literal[True] = True
    every_control_timestamp_bounded_by_release: Literal[True] = True
    required_worker_inventory_verified: Literal[True] = True
    policy_distinct_publication_organization_quorum_verified: Literal[True] = True
    external_organizational_independence_cryptographically_proven: Literal[False] = False
    exact_artifact_bill_of_materials_verified: Literal[True] = True


def sign_campaign_publication_manifest(
    manifest: CampaignPublicationManifest,
    *,
    signer: Ed25519Signer,
) -> SignedCampaignPublicationManifest:
    canonical = CampaignPublicationManifest.model_validate_json(canonical_json_bytes(manifest))
    checked = checked_signer(signer)
    signature = checked.sign(_MANIFEST_SIGNATURE_DOMAIN + canonical_json_bytes(canonical))
    return SignedCampaignPublicationManifest(
        manifest=canonical,
        signature_base64=base64.b64encode(signature).decode('ascii'),
    )


def sign_publication_receipt(
    statement: PublicationReceiptStatement,
    *,
    signer: Ed25519Signer,
) -> SignedPublicationReceipt:
    canonical = PublicationReceiptStatement.model_validate_json(canonical_json_bytes(statement))
    checked = checked_signer(signer)
    signature = checked.sign(_RECEIPT_SIGNATURE_DOMAIN + canonical_json_bytes(canonical))
    return SignedPublicationReceipt(
        statement=canonical,
        signature_base64=base64.b64encode(signature).decode('ascii'),
    )


def sign_worker_image_provenance(
    statement: WorkerImageProvenance,
    *,
    signing_key_id: str,
    signer: Ed25519Signer,
) -> SignedWorkerImageProvenance:
    canonical = WorkerImageProvenance.model_validate_json(canonical_json_bytes(statement))
    checked = checked_signer(signer)
    signature = checked.sign(_WORKER_PROVENANCE_SIGNATURE_DOMAIN + canonical_json_bytes(canonical))
    return SignedWorkerImageProvenance(
        statement=canonical,
        signing_key_id=signing_key_id,
        signature_base64=base64.b64encode(signature).decode('ascii'),
    )


def verify_campaign_publication(
    signed_manifest_bytes: bytes,
    *,
    trust_policy_bytes: bytes,
    expected_trust_policy_sha256: str,
    artifacts: Mapping[str, bytes],
    publication_receipt_bytes: Sequence[bytes],
    verified_at: datetime,
) -> PublicationVerificationReport:
    """Offline-verify one release using only caller-supplied out-of-band trust."""

    artifacts = _snapshot_exact_bytes_mapping(artifacts, 'publication artifacts')
    publication_receipt_bytes = _snapshot_exact_bytes_sequence(
        publication_receipt_bytes,
        'publication receipts',
    )
    signed = _canonical_model(
        signed_manifest_bytes,
        SignedCampaignPublicationManifest,
        'signed campaign publication manifest',
        maximum=_MAX_MANIFEST_BYTES,
    )
    trust = _canonical_model(
        trust_policy_bytes,
        CampaignPublicationTrustPolicy,
        'out-of-band campaign publication trust policy',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    trust_policy_sha256 = hashlib.sha256(trust_policy_bytes).hexdigest()
    if not _is_sha256(expected_trust_policy_sha256) or not hmac.compare_digest(
        trust_policy_sha256,
        expected_trust_policy_sha256,
    ):
        raise CampaignPublicationError('campaign trust policy differs from its out-of-band expected digest')
    now = aware_utc(verified_at, 'publication verification time')
    manifest = signed.manifest
    if manifest.created_at > now:
        raise CampaignPublicationError('publication manifest claims a future creation time')
    if manifest.campaign_id != trust.campaign_id:
        raise CampaignPublicationError('manifest campaign differs from out-of-band trust')
    if manifest.release_authority_id != trust.release_authority_id:
        raise CampaignPublicationError('manifest release authority differs from out-of-band trust')
    manifest_controls = tuple(
        item for item in manifest.artifacts if item.role not in {'release_archive', 'release_archive_index'}
    )
    if manifest_controls != trust.precommitted_control_artifacts:
        raise CampaignPublicationError('manifest operational and worker control bytes differ from out-of-band trust')
    if manifest.workers != trust.required_workers:
        raise CampaignPublicationError('manifest worker inventory differs from out-of-band trust')
    release_key = _unique_key(trust.release_keys, manifest.release_signing_key_id, 'release')
    _require_key_valid(release_key, manifest.created_at)
    _verify_signature(
        release_key,
        _MANIFEST_SIGNATURE_DOMAIN + canonical_json_bytes(manifest),
        signed.signature_base64,
        'release manifest',
    )

    if not isinstance(artifacts, Mapping):
        raise CampaignPublicationError('publication artifacts must be a mapping')
    expected_ids = {item.artifact_id for item in manifest.artifacts}
    if set(artifacts) != expected_ids:
        raise CampaignPublicationError('supplied artifact IDs differ from the exact manifest bill of materials')
    artifact_bindings = {item.artifact_id: item for item in manifest.artifacts}
    if sum(item.byte_count for item in manifest.artifacts) > MAX_PUBLICATION_TOTAL_ARTIFACT_BYTES:
        raise CampaignPublicationError('publication artifact inventory exceeds the verifier total-byte limit')
    for artifact_id in sorted(expected_ids):
        payload = artifacts[artifact_id]
        if not isinstance(payload, bytes):
            raise CampaignPublicationError(f'artifact {artifact_id!r} is not bytes')
        binding = artifact_bindings[artifact_id]
        if len(payload) != binding.byte_count or not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(),
            binding.sha256,
        ):
            raise CampaignPublicationError(f'artifact {artifact_id!r} differs from its exact binding')

    clock_policy_bytes = artifacts[manifest.clock_health_policy_artifact_id]
    _canonical_model(
        clock_policy_bytes,
        ClockHealthPolicy,
        'clock-health policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    clock_policy_sha256 = hashlib.sha256(clock_policy_bytes).hexdigest()
    clock_process_sha256s, signer_process_sha256s = _verify_runtime_process_materials(
        manifest.runtime_processes,
        artifacts,
    )

    checkpoint_witness = _verify_registry_binding(
        manifest.registry,
        artifacts,
        clock_policy_sha256=clock_policy_sha256,
        clock_process_sha256s=clock_process_sha256s,
        signer_process_sha256s=signer_process_sha256s,
        release_created_at=manifest.created_at,
    )
    published_witnesses = frozenset(
        _verify_witness_binding(
            witness,
            artifacts,
            clock_policy_sha256=clock_policy_sha256,
            clock_process_sha256s=clock_process_sha256s,
            signer_process_sha256s=signer_process_sha256s,
            release_created_at=manifest.created_at,
        )
        for witness in manifest.witnesses
    )
    if checkpoint_witness not in published_witnesses:
        raise CampaignPublicationError('registry checkpoint witness is absent from the publication witness inventory')
    _verify_gossip_binding(
        manifest.gossip,
        artifacts,
        clock_policy_sha256=clock_policy_sha256,
        clock_process_sha256s=clock_process_sha256s,
        signer_process_sha256s=signer_process_sha256s,
        release_created_at=manifest.created_at,
        expected_registry=(
            manifest.registry.registry_id,
            manifest.registry.authority_id,
            manifest.registry.signing_public_key_sha256s,
        ),
        expected_witnesses=published_witnesses,
    )
    worker_sandbox_keys, worker_sandbox_authority_ids = _verify_workers(
        manifest.workers,
        artifacts,
        trust.worker_build_authorities,
        release_created_at=manifest.created_at,
    )
    release_key_sha256 = hashlib.sha256(
        _decode_base64(release_key.public_key_base64, expected_bytes=32, label='release key')
    ).hexdigest()
    trusted_nonoperational_keys = tuple(
        hashlib.sha256(
            _decode_base64(
                key.public_key_base64,
                expected_bytes=32,
                label='campaign authority key',
            )
        ).hexdigest()
        for authority in (*trust.worker_build_authorities, *trust.publication_authorities)
        for key in authority.keys
    )
    operational_keys = (
        *manifest.registry.signing_public_key_sha256s,
        *(item.signing_public_key_sha256 for item in manifest.witnesses),
        *(item.report_signing_public_key_sha256 for item in manifest.gossip.monitors),
        *worker_sandbox_keys,
        *trusted_nonoperational_keys,
        release_key_sha256,
    )
    if len(operational_keys) != len(set(operational_keys)):
        raise CampaignPublicationError('independent Tier-A authorities must not share signing keys')
    _verify_campaign_authority_id_separation(
        trust=trust,
        manifest=manifest,
        worker_sandbox_authority_ids=worker_sandbox_authority_ids,
    )

    archive_binding = artifact_bindings[manifest.archive_artifact_id]
    signed_sha256 = hashlib.sha256(signed_manifest_bytes).hexdigest()
    authorities = {item.authority_id: item for item in trust.publication_authorities}
    if not (trust.minimum_independent_receipts <= len(publication_receipt_bytes) <= len(authorities)):
        raise CampaignPublicationError('publication receipt count is outside the configured quorum bounds')
    verified_authorities: set[str] = set()
    for receipt_bytes in publication_receipt_bytes:
        receipt = _canonical_model(
            receipt_bytes,
            SignedPublicationReceipt,
            'independent publication receipt',
            maximum=1024 * 1024,
        )
        statement = receipt.statement
        authority = authorities.get(statement.authority_id)
        if authority is None:
            raise CampaignPublicationError('publication receipt authority is absent from out-of-band trust')
        if statement.authority_id in verified_authorities:
            raise CampaignPublicationError('duplicate publication authority cannot satisfy quorum')
        key = _unique_key(authority.keys, statement.signing_key_id, 'publication receipt')
        _require_key_valid(key, statement.published_at)
        if (
            statement.campaign_id != manifest.campaign_id
            or statement.release_id != manifest.release_id
            or statement.signed_manifest_sha256 != signed_sha256
            or statement.signed_manifest_bytes != len(signed_manifest_bytes)
            or statement.archive_sha256 != archive_binding.sha256
            or statement.archive_bytes != archive_binding.byte_count
        ):
            raise CampaignPublicationError('publication receipt differs from the release or archive')
        if statement.published_at < manifest.created_at or (statement.published_at - manifest.created_at) > timedelta(
            seconds=trust.max_receipt_delay_seconds
        ):
            raise CampaignPublicationError('publication receipt time is outside the allowed release window')
        if statement.published_at > now:
            raise CampaignPublicationError('publication receipt claims a future publication time')
        _verify_signature(
            key,
            _RECEIPT_SIGNATURE_DOMAIN + canonical_json_bytes(statement),
            receipt.signature_base64,
            'publication receipt',
        )
        verified_authorities.add(statement.authority_id)
    if len(verified_authorities) < trust.minimum_independent_receipts:
        raise CampaignPublicationError('independent publication receipt quorum was not met')
    return PublicationVerificationReport(
        campaign_id=manifest.campaign_id,
        release_id=manifest.release_id,
        signed_manifest_sha256=signed_sha256,
        archive_sha256=archive_binding.sha256,
        verified_artifact_count=len(manifest.artifacts),
        verified_worker_count=len(manifest.workers),
        independent_publication_authority_ids=tuple(sorted(verified_authorities)),
        publication_organization_ids=tuple(sorted(authorities[item].organization_id for item in verified_authorities)),
    )


def _snapshot_exact_bytes_mapping(value: Mapping[str, bytes], label: str) -> dict[str, bytes]:
    """Freeze one potentially stateful mapping before any verification read."""

    if not isinstance(value, Mapping):
        raise CampaignPublicationError(f'{label} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception as error:
        raise CampaignPublicationError(f'{label} could not be snapshotted') from error
    result: dict[str, bytes] = {}
    for key, payload in items:
        if type(key) is not str or type(payload) is not bytes or key in result:
            raise CampaignPublicationError(f'{label} must use unique exact-string keys and exact-bytes values')
        result[key] = payload
    return result


def _snapshot_exact_bytes_sequence(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    """Freeze one potentially stateful sequence before any verification read."""

    if isinstance(value, (bytes, bytearray, str)) or not isinstance(value, Sequence):
        raise CampaignPublicationError(f'{label} must be a sequence of exact bytes')
    try:
        items = tuple(value)
    except Exception as error:
        raise CampaignPublicationError(f'{label} could not be snapshotted') from error
    if any(type(payload) is not bytes for payload in items):
        raise CampaignPublicationError(f'{label} must contain exact bytes')
    return items


def _verify_registry_binding(
    binding: RegistryPublicationBinding,
    artifacts: Mapping[str, bytes],
    *,
    clock_policy_sha256: str,
    clock_process_sha256s: frozenset[str],
    signer_process_sha256s: frozenset[str],
    release_created_at: datetime,
) -> tuple[str, str, str]:
    policy = _canonical_model(
        artifacts[binding.policy_artifact_id],
        SelectionRegistryPolicy,
        'registry policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    trust = _canonical_model(
        artifacts[binding.trust_policy_artifact_id],
        SelectionRegistryTrustPolicy,
        'registry trust-policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    _require_tier_a_runtime_trust(
        clock_health_policy_sha256=policy.clock_health_policy_sha256,
        clock_health_process_sha256=policy.clock_health_process_sha256,
        external_signer_process_sha256=policy.external_signer_process_sha256,
        expected_clock_policy_sha256=clock_policy_sha256,
        retained_clock_process_sha256s=clock_process_sha256s,
        retained_signer_process_sha256s=signer_process_sha256s,
        label='registry policy',
    )
    _require_tier_a_runtime_trust(
        clock_health_policy_sha256=trust.checkpoint_witness_policy.clock_health_policy_sha256,
        clock_health_process_sha256=trust.checkpoint_witness_policy.clock_health_process_sha256,
        external_signer_process_sha256=(trust.checkpoint_witness_policy.external_signer_process_sha256),
        expected_clock_policy_sha256=clock_policy_sha256,
        retained_clock_process_sha256s=clock_process_sha256s,
        retained_signer_process_sha256s=signer_process_sha256s,
        label='registry checkpoint-witness policy',
    )
    if (
        policy.registry_id != binding.registry_id
        or policy.authority_id != binding.authority_id
        or trust.registry_id != binding.registry_id
        or trust.authority_id != binding.authority_id
    ):
        raise CampaignPublicationError('registry publication identity differs from exact policy artifacts')
    key_hashes = tuple(
        sorted(
            hashlib.sha256(_decode_base64(item.public_key_base64, expected_bytes=32, label='registry key')).hexdigest()
            for item in trust.signing_keys
        )
    )
    if key_hashes != binding.signing_public_key_sha256s:
        raise CampaignPublicationError('registry key ring differs from publication binding')
    if (
        base64.b64decode(trust.pinned_checkpoint.signed_checkpoint_base64, validate=True)
        != artifacts[binding.bootstrap_head_artifact_id]
        or base64.b64decode(trust.pinned_checkpoint.witness_proof_base64, validate=True)
        != artifacts[binding.bootstrap_witness_artifact_id]
    ):
        raise CampaignPublicationError('registry bootstrap artifacts differ from out-of-band trust policy pin')
    try:
        checkpoint = verify_signed_registry_checkpoint(
            artifacts[binding.bootstrap_head_artifact_id],
            artifacts[binding.trust_policy_artifact_id],
        )
        witness_facts = verify_witness_service_artifact(
            artifacts[binding.bootstrap_head_artifact_id],
            artifacts[binding.bootstrap_witness_artifact_id],
            policy_bytes=canonical_json_bytes(trust.checkpoint_witness_policy),
            trust_policy_bytes=canonical_json_bytes(trust.checkpoint_witness_trust_policy),
            checkpoint_schema_version=('vaxreplay.signed-plan-selection-registry-checkpoint.v0.1'),
        )
    except (TypeError, ValueError) as error:
        raise CampaignPublicationError(f'registry bootstrap authentication failed: {error}') from error
    if (
        checkpoint.tree_size != trust.pinned_checkpoint.tree_size
        or checkpoint.root_sha256 != trust.pinned_checkpoint.root_sha256
        or checkpoint.issued_at_upper_bound > witness_facts.witnessed_at
        or checkpoint.issued_at_upper_bound > release_created_at
        or witness_facts.witnessed_at > release_created_at
    ):
        raise CampaignPublicationError('registry bootstrap differs from its witnessed trust anchor')
    checkpoint_witness_trust = trust.checkpoint_witness_trust_policy
    checkpoint_witness_key = _decode_base64(
        checkpoint_witness_trust.public_key_base64,
        expected_bytes=32,
        label='registry checkpoint witness key',
    )
    return (
        checkpoint_witness_trust.authority_id,
        checkpoint_witness_trust.witness_id,
        hashlib.sha256(checkpoint_witness_key).hexdigest(),
    )


def _verify_witness_binding(
    binding: WitnessPublicationBinding,
    artifacts: Mapping[str, bytes],
    *,
    clock_policy_sha256: str,
    clock_process_sha256s: frozenset[str],
    signer_process_sha256s: frozenset[str],
    release_created_at: datetime,
) -> tuple[str, str, str]:
    policy_bytes = artifacts[binding.policy_artifact_id]
    policy = _canonical_model(
        policy_bytes,
        WitnessServicePolicy,
        'witness policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    trust = _canonical_model(
        artifacts[binding.trust_policy_artifact_id],
        WitnessServiceTrustPolicy,
        'witness trust-policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    _require_tier_a_runtime_trust(
        clock_health_policy_sha256=policy.clock_health_policy_sha256,
        clock_health_process_sha256=policy.clock_health_process_sha256,
        external_signer_process_sha256=policy.external_signer_process_sha256,
        expected_clock_policy_sha256=clock_policy_sha256,
        retained_clock_process_sha256s=clock_process_sha256s,
        retained_signer_process_sha256s=signer_process_sha256s,
        label='witness policy',
    )
    if (
        policy.authority_id != binding.authority_id
        or policy.witness_id != binding.witness_id
        or trust.authority_id != binding.authority_id
        or trust.witness_id != binding.witness_id
        or trust.service_policy_sha256 != hashlib.sha256(policy_bytes).hexdigest()
    ):
        raise CampaignPublicationError('witness publication identity or policy binding is inconsistent')
    public_key = _decode_base64(trust.public_key_base64, expected_bytes=32, label='witness key')
    if hashlib.sha256(public_key).hexdigest() != binding.signing_public_key_sha256:
        raise CampaignPublicationError('witness key differs from publication binding')
    try:
        signed_checkpoint = verify_witness_service_signed_checkpoint(
            artifacts[binding.bootstrap_head_artifact_id],
            policy_bytes=policy_bytes,
            trust_policy_bytes=artifacts[binding.trust_policy_artifact_id],
        )
    except (TypeError, ValueError) as error:
        raise CampaignPublicationError(f'witness bootstrap head is invalid: {error}') from error
    if signed_checkpoint.checkpoint.issued_at > release_created_at:
        raise CampaignPublicationError('witness bootstrap checkpoint postdates the release')
    return (binding.authority_id, binding.witness_id, binding.signing_public_key_sha256)


def _verify_gossip_binding(
    binding: GossipPublicationBinding,
    artifacts: Mapping[str, bytes],
    *,
    clock_policy_sha256: str,
    clock_process_sha256s: frozenset[str],
    signer_process_sha256s: frozenset[str],
    release_created_at: datetime,
    expected_registry: tuple[str, str, tuple[str, ...]],
    expected_witnesses: frozenset[tuple[str, str, str]],
) -> None:
    comparison = _canonical_model(
        artifacts[binding.comparison_policy_artifact_id],
        GossipComparisonPolicy,
        'gossip comparison policy artifact',
        maximum=_MAX_POLICY_ARTIFACT_BYTES,
    )
    comparison_pins = {item.monitor_id: item for item in comparison.monitors}
    if set(comparison_pins) != {item.monitor_id for item in binding.monitors}:
        raise CampaignPublicationError('gossip comparison monitor set differs from publication binding')
    for monitor in binding.monitors:
        policy_bytes = artifacts[monitor.policy_artifact_id]
        policy = _canonical_model(
            policy_bytes,
            CheckpointGossipMonitorPolicy,
            'gossip monitor policy artifact',
            maximum=_MAX_POLICY_ARTIFACT_BYTES,
        )
        _require_tier_a_runtime_trust(
            clock_health_policy_sha256=policy.clock_health_policy_sha256,
            clock_health_process_sha256=policy.clock_health_process_sha256,
            external_signer_process_sha256=policy.external_signer_process_sha256,
            expected_clock_policy_sha256=clock_policy_sha256,
            retained_clock_process_sha256s=clock_process_sha256s,
            retained_signer_process_sha256s=signer_process_sha256s,
            label=f'gossip monitor {policy.monitor_id!r} policy',
        )
        for stream in policy.streams:
            if isinstance(stream, WitnessGossipStreamPolicy):
                _require_tier_a_runtime_trust(
                    clock_health_policy_sha256=stream.service_policy.clock_health_policy_sha256,
                    clock_health_process_sha256=stream.service_policy.clock_health_process_sha256,
                    external_signer_process_sha256=(stream.service_policy.external_signer_process_sha256),
                    expected_clock_policy_sha256=clock_policy_sha256,
                    retained_clock_process_sha256s=clock_process_sha256s,
                    retained_signer_process_sha256s=signer_process_sha256s,
                    label=f'gossip monitor {policy.monitor_id!r} witness-source policy',
                )
        pin = comparison_pins[monitor.monitor_id]
        if (
            policy.monitor_id != monitor.monitor_id
            or pin.monitor_policy_sha256 != hashlib.sha256(policy_bytes).hexdigest()
            or canonical_json_bytes(pin.monitor_policy) != policy_bytes
        ):
            raise CampaignPublicationError('gossip comparison does not pin the exact monitor policy artifact')
        key = _decode_base64(
            policy.report_signing_public_key_base64,
            expected_bytes=32,
            label='gossip report key',
        )
        if hashlib.sha256(key).hexdigest() != monitor.report_signing_public_key_sha256:
            raise CampaignPublicationError('gossip report key differs from publication binding')
    heads = {item.stream_id: item.artifact_id for item in binding.bootstrap_heads}
    if set(heads) != set(comparison.required_stream_ids):
        raise CampaignPublicationError('gossip bootstrap heads do not exactly cover required streams')
    reference_policy = comparison.monitors[0].monitor_policy
    streams = {item.stream_id: item for item in reference_policy.streams}
    for stream in streams.values():
        if isinstance(stream, WitnessGossipStreamPolicy):
            source_key = _decode_base64(
                stream.service_trust_policy.public_key_base64,
                expected_bytes=32,
                label='gossip witness source key',
            )
            source_identity = (
                stream.service_policy.authority_id,
                stream.service_policy.witness_id,
                hashlib.sha256(source_key).hexdigest(),
            )
            if source_identity not in expected_witnesses:
                raise CampaignPublicationError('gossip witness source is absent from the publication witness inventory')
        else:
            source_keys = tuple(
                sorted(
                    hashlib.sha256(
                        _decode_base64(
                            key.public_key_base64,
                            expected_bytes=32,
                            label='gossip registry source key',
                        )
                    ).hexdigest()
                    for key in stream.registry_monitor.signing_keys
                )
            )
            source_identity = (
                stream.registry_monitor.registry_id,
                stream.registry_monitor.authority_id,
                source_keys,
            )
            if source_identity != expected_registry:
                raise CampaignPublicationError('gossip registry source differs from the publication registry inventory')
    for stream_id, artifact_id in heads.items():
        stream = streams[stream_id]
        expected_sha256 = stream.bootstrap_signed_checkpoint_sha256
        if hashlib.sha256(artifacts[artifact_id]).hexdigest() != expected_sha256:
            raise CampaignPublicationError('gossip bootstrap artifact differs from monitor policy pin')
        try:
            verified = verify_gossip_bootstrap_head(stream, artifacts[artifact_id])
        except (TypeError, ValueError) as error:
            raise CampaignPublicationError(f'gossip bootstrap authentication failed: {error}') from error
        if verified.stream_id != stream_id:
            raise CampaignPublicationError('gossip bootstrap verifier returned another stream identity')
        if isinstance(stream, WitnessGossipStreamPolicy):
            signed_source = verify_witness_service_signed_checkpoint(
                artifacts[artifact_id],
                policy_bytes=canonical_json_bytes(stream.service_policy),
                trust_policy_bytes=canonical_json_bytes(stream.service_trust_policy),
            )
            source_issued_at = signed_source.checkpoint.issued_at
        else:
            source_issued_at = _canonical_model(
                artifacts[artifact_id],
                WitnessedSignedRegistryCheckpoint,
                'gossip registry bootstrap checkpoint',
                maximum=_MAX_POLICY_ARTIFACT_BYTES,
            ).checkpoint.issued_at_upper_bound
        if source_issued_at > release_created_at:
            raise CampaignPublicationError('gossip bootstrap checkpoint postdates the release')


def _verify_runtime_process_materials(
    bindings: tuple[RuntimeProcessPublicationBinding, ...],
    artifacts: Mapping[str, bytes],
) -> tuple[frozenset[str], frozenset[str]]:
    by_purpose: dict[str, set[str]] = {'clock_health': set(), 'external_signer': set()}
    for binding in bindings:
        config_bytes = artifacts[binding.process_config_artifact_id]
        config = _canonical_model(
            config_bytes,
            IsolatedProcessConfig,
            f'{binding.purpose} process configuration',
            maximum=_MAX_POLICY_ARTIFACT_BYTES,
        )
        executable = artifacts[binding.executable_artifact_id]
        if (
            config.process_id != binding.process_id
            or len(executable) != config.executable_byte_count
            or not hmac.compare_digest(
                hashlib.sha256(executable).hexdigest(),
                config.executable_sha256,
            )
        ):
            raise CampaignPublicationError(
                f'{binding.purpose} process configuration differs from its retained executable'
            )
        by_purpose[binding.purpose].add(hashlib.sha256(config_bytes).hexdigest())
    if not by_purpose['clock_health'] or not by_purpose['external_signer']:
        raise CampaignPublicationError('publication must retain clock and external-signer process configurations')
    return (
        frozenset(by_purpose['clock_health']),
        frozenset(by_purpose['external_signer']),
    )


def _require_tier_a_runtime_trust(
    *,
    clock_health_policy_sha256: str | None,
    clock_health_process_sha256: str | None,
    external_signer_process_sha256: str | None,
    expected_clock_policy_sha256: str,
    retained_clock_process_sha256s: frozenset[str],
    retained_signer_process_sha256s: frozenset[str],
    label: str,
) -> None:
    if (
        clock_health_policy_sha256 is None
        or clock_health_process_sha256 is None
        or external_signer_process_sha256 is None
    ):
        raise CampaignPublicationError(f'{label} omits required Tier-A runtime-trust digests')
    if not hmac.compare_digest(clock_health_policy_sha256, expected_clock_policy_sha256):
        raise CampaignPublicationError(f'{label} binds a different clock-health policy')
    if clock_health_process_sha256 not in retained_clock_process_sha256s:
        raise CampaignPublicationError(f'{label} clock process configuration was not retained')
    if external_signer_process_sha256 not in retained_signer_process_sha256s:
        raise CampaignPublicationError(f'{label} signer process configuration was not retained')


def _verify_workers(
    workers: tuple[WorkerPublicationBinding, ...],
    artifacts: Mapping[str, bytes],
    builder_authorities: tuple[PublicationAuthorityTrust, ...],
    *,
    release_created_at: datetime,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    builders = {item.authority_id: item for item in builder_authorities}
    sandbox_keys: set[str] = set()
    sandbox_authority_ids: set[str] = set()
    for worker in workers:
        environment = _canonical_model(
            artifacts[worker.environment_artifact_id],
            HermeticOciEnvironment,
            'worker OCI environment artifact',
            maximum=_MAX_POLICY_ARTIFACT_BYTES,
        )
        sandbox = _canonical_model(
            artifacts[worker.sandbox_policy_artifact_id],
            HermeticSandboxPolicy,
            'worker sandbox-policy artifact',
            maximum=_MAX_POLICY_ARTIFACT_BYTES,
        )
        sandbox_keys.add(sandbox.signing_public_key_sha256)
        sandbox_authority_ids.add(sandbox.authority_id)
        if environment.image_ref != worker.image_ref or environment.expected_image_id != worker.resolved_image_id:
            raise CampaignPublicationError('worker image identity differs from its OCI environment')
        if not artifacts[worker.implementation_artifact_id]:
            raise CampaignPublicationError('worker implementation artifact is empty')
        if not artifacts[worker.sbom_artifact_id] or not artifacts[worker.provenance_artifact_id]:
            raise CampaignPublicationError('worker SBOM/provenance artifact is empty')
        signed_provenance = _canonical_model(
            artifacts[worker.provenance_artifact_id],
            SignedWorkerImageProvenance,
            'signed normalized worker image provenance artifact',
            maximum=_MAX_POLICY_ARTIFACT_BYTES,
        )
        provenance = signed_provenance.statement
        builder = builders.get(provenance.builder_id)
        if builder is None:
            raise CampaignPublicationError('worker provenance builder is absent from out-of-band trust')
        builder_key = _unique_key(
            builder.keys,
            signed_provenance.signing_key_id,
            'worker provenance',
        )
        _require_key_valid(builder_key, provenance.created_at)
        if provenance.created_at > release_created_at:
            raise CampaignPublicationError('worker provenance postdates the release')
        _verify_signature(
            builder_key,
            _WORKER_PROVENANCE_SIGNATURE_DOMAIN + canonical_json_bytes(provenance),
            signed_provenance.signature_base64,
            'worker provenance',
        )
        if (
            provenance.builder_id != worker.provenance_builder_id
            or provenance.build_type != worker.provenance_build_type
            or provenance.image_ref != worker.image_ref
            or provenance.resolved_image_id != worker.resolved_image_id
            or provenance.image_manifest_sha256 != worker.image_ref.rsplit('@sha256:', 1)[1]
            or provenance.implementation_sha256
            != hashlib.sha256(artifacts[worker.implementation_artifact_id]).hexdigest()
            or provenance.sbom_sha256 != hashlib.sha256(artifacts[worker.sbom_artifact_id]).hexdigest()
        ):
            raise CampaignPublicationError('worker provenance differs from image, implementation, or SBOM binding')
        if not sandbox.network_disabled or not sandbox.no_host_mounts or not sandbox.read_only_root:
            raise CampaignPublicationError('worker sandbox policy weakens required isolation')
    return tuple(sorted(sandbox_keys)), tuple(sorted(sandbox_authority_ids))


def _verify_campaign_authority_id_separation(
    *,
    trust: CampaignPublicationTrustPolicy,
    manifest: CampaignPublicationManifest,
    worker_sandbox_authority_ids: tuple[str, ...],
) -> None:
    identities_by_role = {
        'release': {trust.release_authority_id},
        'builder': {item.authority_id for item in trust.worker_build_authorities},
        'publisher': {item.authority_id for item in trust.publication_authorities},
        'registry': {manifest.registry.authority_id, manifest.registry.registry_id},
        'witness': {identity for item in manifest.witnesses for identity in (item.authority_id, item.witness_id)},
        'gossip-monitor': {item.monitor_id for item in manifest.gossip.monitors},
        'worker': {
            *worker_sandbox_authority_ids,
            *(item.worker_id for item in manifest.workers),
        },
    }
    owner_by_identity: dict[str, str] = {}
    for role, identities in identities_by_role.items():
        for identity in identities:
            previous = owner_by_identity.setdefault(identity, role)
            if previous != role:
                raise CampaignPublicationError(
                    'independent Tier-A campaign roles must use distinct authority identities'
                )


def _canonical_model(payload: bytes, model: type[StrictModel], label: str, *, maximum: int):
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise CampaignPublicationError(f'{label} has an invalid byte size')
    try:
        parsed = model.model_validate_json(payload)
    except (TypeError, ValueError):
        raise CampaignPublicationError(f'{label} is invalid') from None
    if canonical_json_bytes(parsed) != payload:
        raise CampaignPublicationError(f'{label} is not canonical JSON')
    return parsed


def _unique_key(
    keys: Sequence[PublicationTrustedKey],
    key_id: str,
    label: str,
) -> PublicationTrustedKey:
    matches = [item for item in keys if item.key_id == key_id]
    if len(matches) != 1:
        raise CampaignPublicationError(f'{label} signing key is not uniquely trusted out of band')
    return matches[0]


def _require_key_valid(key: PublicationTrustedKey, when: datetime) -> None:
    if when < key.valid_from or (key.valid_until is not None and when >= key.valid_until):
        raise CampaignPublicationError('publication signature time is outside key validity')


def _verify_signature(
    key: PublicationTrustedKey,
    payload: bytes,
    signature_base64: str,
    label: str,
) -> None:
    public_key = _decode_base64(key.public_key_base64, expected_bytes=32, label=f'{label} key')
    signature = _decode_base64(signature_base64, expected_bytes=64, label=f'{label} signature')
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError):
        raise CampaignPublicationError(f'{label} signature is invalid') from None


def _decode_base64(value: str, *, expected_bytes: int, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError):
        raise ValueError(f'{label} is invalid base64') from None
    if len(decoded) != expected_bytes or base64.b64encode(decoded).decode('ascii') != value:
        raise ValueError(f'{label} has an invalid canonical length')
    return decoded


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in '0123456789abcdef' for character in value)


__all__ = [
    'MAX_PUBLICATION_ARTIFACT_BYTES',
    'MAX_PUBLICATION_TOTAL_ARTIFACT_BYTES',
    'CampaignPublicationError',
    'CampaignPublicationManifest',
    'CampaignPublicationTrustPolicy',
    'GossipMonitorPublicationBinding',
    'GossipBootstrapHeadBinding',
    'GossipPublicationBinding',
    'PublicationArtifactBinding',
    'PublicationAuthorityTrust',
    'PublicationReceiptStatement',
    'PublicationTrustedKey',
    'PublicationVerificationReport',
    'RegistryPublicationBinding',
    'RuntimeProcessPublicationBinding',
    'SignedCampaignPublicationManifest',
    'SignedPublicationReceipt',
    'SignedWorkerImageProvenance',
    'WitnessPublicationBinding',
    'WorkerPublicationBinding',
    'WorkerImageProvenance',
    'sign_campaign_publication_manifest',
    'sign_publication_receipt',
    'sign_worker_image_provenance',
    'verify_campaign_publication',
]
