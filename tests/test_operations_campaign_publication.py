from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.test_operations_selection_registry import _registry
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations import campaign_publication as campaign_publication_module
from vaxreplay.operations.campaign_publication import (
    MAX_PUBLICATION_ARTIFACT_BYTES,
    CampaignPublicationError,
    CampaignPublicationManifest,
    CampaignPublicationTrustPolicy,
    GossipBootstrapHeadBinding,
    GossipMonitorPublicationBinding,
    GossipPublicationBinding,
    PublicationArtifactBinding,
    PublicationAuthorityTrust,
    PublicationReceiptStatement,
    PublicationTrustedKey,
    PublicationVerificationReport,
    RegistryPublicationBinding,
    RuntimeProcessPublicationBinding,
    SignedCampaignPublicationManifest,
    SignedWorkerImageProvenance,
    WitnessPublicationBinding,
    WorkerImageProvenance,
    WorkerPublicationBinding,
    sign_campaign_publication_manifest,
    sign_publication_receipt,
    sign_worker_image_provenance,
    verify_campaign_publication,
)
from vaxreplay.operations.checkpoint_gossip import (
    CheckpointGossipMonitorPolicy,
    GossipComparisonPolicy,
    GossipMonitorPolicyPin,
    RegistryGossipStreamPolicy,
    WitnessGossipStreamPolicy,
)
from vaxreplay.operations.clock_health import ClockHealthPolicy
from vaxreplay.operations.hermetic_execution import HermeticOciEnvironment, HermeticSandboxPolicy
from vaxreplay.operations.operator_trust import IsolatedProcessConfig
from vaxreplay.operations.selection_registry import SelectionRegistryPolicy, SelectionRegistryTrustPolicy
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.operations.witness_service_schema import (
    WitnessServiceProof,
    WitnessServiceSignedCheckpoint,
)

UTC = timezone.utc


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


@dataclass(frozen=True)
class PublicationFixture:
    signed_manifest_bytes: bytes
    trust_policy_bytes: bytes
    artifacts: dict[str, bytes]
    receipts: tuple[bytes, ...]
    verified_at: datetime
    manifest: CampaignPublicationManifest
    release_key: Ed25519PrivateKey
    builder_key: Ed25519PrivateKey


class _ChameleonBytesMapping(Mapping[str, bytes]):
    def __init__(self, snapshot: Mapping[str, bytes], later: Mapping[str, bytes]) -> None:
        self._snapshot = dict(snapshot)
        self._later = dict(later)
        self.indexed_reads = 0

    def __getitem__(self, key: str) -> bytes:
        self.indexed_reads += 1
        return self._later[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot)

    def __len__(self) -> int:
        return len(self._snapshot)

    def items(self):  # noqa: ANN201
        return self._snapshot.items()


def _publication(
    tmp_path: Path,
    *,
    manifest_time_offset_seconds: int = 1,
    worker_provenance_offset_seconds: int = 0,
    release_id: str = 'future-immport-release-001',
    release_archive_bytes: bytes = b'fictional immutable release archive bytes',
    release_archive_index_bytes: bytes = b'{"archive":"fictional"}\n',
) -> PublicationFixture:
    clock_policy = ClockHealthPolicy(
        policy_id='tier-a-clock-health-v1',
        provider_id='chrony-sidecar-a',
        max_observation_age_seconds=5,
        max_absolute_offset_milliseconds=5,
        max_root_distance_milliseconds=20,
        max_sample_age_milliseconds=1000,
        minimum_source_count=2,
    )
    clock_policy_bytes = canonical_json_bytes(clock_policy)
    clock_executable = b'#!/bin/sh\n# retained clock provider fixture\n'
    signer_executable = b'#!/bin/sh\n# retained signer broker fixture\n'
    clock_process = IsolatedProcessConfig(
        process_id='tier-a-clock-provider',
        argv=('/opt/vaxreplay/clock-provider',),
        executable_sha256=hashlib.sha256(clock_executable).hexdigest(),
        executable_byte_count=len(clock_executable),
    )
    signer_process = IsolatedProcessConfig(
        process_id='tier-a-external-signer',
        argv=('/opt/vaxreplay/external-signer',),
        executable_sha256=hashlib.sha256(signer_executable).hexdigest(),
        executable_byte_count=len(signer_executable),
    )
    clock_process_bytes = canonical_json_bytes(clock_process)
    signer_process_bytes = canonical_json_bytes(signer_process)
    runtime_trust_digests = (
        hashlib.sha256(clock_policy_bytes).hexdigest(),
        hashlib.sha256(clock_process_bytes).hexdigest(),
        hashlib.sha256(signer_process_bytes).hexdigest(),
    )
    registry, _registry_key, registry_policy_bytes, registry_trust_bytes = _registry(
        tmp_path / 'registry-fixture',
        runtime_trust_digests=runtime_trust_digests,
    )
    del registry
    registry_policy = SelectionRegistryPolicy.model_validate_json(registry_policy_bytes)
    registry_trust = SelectionRegistryTrustPolicy.model_validate_json(registry_trust_bytes)
    registry_head = base64.b64decode(
        registry_trust.pinned_checkpoint.signed_checkpoint_base64,
        validate=True,
    )
    registry_witness_proof = base64.b64decode(
        registry_trust.pinned_checkpoint.witness_proof_base64,
        validate=True,
    )
    witness_proof = WitnessServiceProof.model_validate_json(registry_witness_proof)
    witness_head = canonical_json_bytes(
        WitnessServiceSignedCheckpoint(
            checkpoint=witness_proof.statement.checkpoint,
            signature_base64=witness_proof.checkpoint_signature_base64,
        )
    )
    witness_policy = registry_trust.checkpoint_witness_policy
    witness_trust = registry_trust.checkpoint_witness_trust_policy
    witness_policy_bytes = canonical_json_bytes(witness_policy)
    witness_trust_bytes = canonical_json_bytes(witness_trust)

    registry_stream = RegistryGossipStreamPolicy(
        stream_id='registry-heads',
        registry_monitor=witness_policy.registry_monitors[0],
        bootstrap_tree_size=registry_trust.pinned_checkpoint.tree_size,
        bootstrap_signed_checkpoint_sha256=hashlib.sha256(registry_head).hexdigest(),
    )
    witness_stream = WitnessGossipStreamPolicy(
        stream_id='witness-heads',
        service_policy=witness_policy,
        service_trust_policy=witness_trust,
        bootstrap_tree_size=witness_proof.statement.checkpoint.tree_size,
        bootstrap_signed_checkpoint_sha256=hashlib.sha256(witness_head).hexdigest(),
    )
    created_at = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=manifest_time_offset_seconds)
    monitor_keys = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    monitor_policies = tuple(
        CheckpointGossipMonitorPolicy(
            monitor_id=f'independent-monitor-{index}',
            policy_id=f'independent-monitor-{index}-policy',
            streams=(registry_stream, witness_stream),
            max_observation_age_seconds=60,
            max_future_clock_skew_seconds=5,
            report_signing_key_id=f'independent-monitor-{index}-key',
            report_signing_public_key_base64=base64.b64encode(_public_bytes(key)).decode('ascii'),
            report_signing_key_valid_from=created_at - timedelta(days=1),
            clock_health_policy_sha256=runtime_trust_digests[0],
            clock_health_process_sha256=runtime_trust_digests[1],
            external_signer_process_sha256=runtime_trust_digests[2],
        )
        for index, key in enumerate(monitor_keys, start=1)
    )
    comparison = GossipComparisonPolicy(
        comparison_policy_id='tier-a-gossip-quorum-v1',
        monitors=tuple(
            GossipMonitorPolicyPin(
                monitor_id=policy.monitor_id,
                monitor_policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
                monitor_policy=policy,
            )
            for policy in monitor_policies
        ),
        required_stream_ids=('registry-heads', 'witness-heads'),
        max_report_age_seconds=60,
        max_observation_age_seconds=60,
        max_future_clock_skew_seconds=5,
    )

    environment = HermeticOciEnvironment(
        environment_id='immport-tier-a-worker-env',
        image_ref='registry.example/vaxreplay/immport@sha256:' + 'a' * 64,
        expected_image_id='sha256:' + 'b' * 64,
        platform='linux/amd64',
        entrypoint=('/opt/vaxreplay/worker', '--stdio'),
    )
    sandbox = HermeticSandboxPolicy(
        policy_id='tier-a-worker-sandbox',
        authority_id='independent-worker-authority',
        signing_key_id='worker-receipt-key',
        signing_public_key_sha256='c' * 64,
        seccomp_profile_sha256='d' * 64,
        wall_seconds=30,
        memory_mib=256,
        milli_cpus=500,
        pids=32,
        scratch_mib=32,
        open_files=64,
        max_input_bytes=1024 * 1024,
        max_callback_policy_bytes=1024 * 1024,
        max_output_bytes=1024 * 1024,
        max_worker_response_bytes=2 * 1024 * 1024,
        max_log_bytes=64 * 1024,
    )
    worker_implementation = b'immport source verifier and adapter wheel bytes'
    worker_sbom = b'{"bomFormat":"CycloneDX","specVersion":"1.5"}\n'
    builder_key = Ed25519PrivateKey.generate()
    worker_provenance = WorkerImageProvenance(
        builder_id='independent-oci-builder',
        build_type='reproducible-python-wheel-oci',
        created_at=created_at + timedelta(seconds=worker_provenance_offset_seconds),
        image_ref=environment.image_ref,
        resolved_image_id=environment.expected_image_id,
        image_manifest_sha256=environment.image_ref.rsplit('@sha256:', 1)[1],
        implementation_sha256=hashlib.sha256(worker_implementation).hexdigest(),
        sbom_sha256=hashlib.sha256(worker_sbom).hexdigest(),
        source_repository_uri='https://github.com/example/vaxreplay',
        source_commit_sha256='e' * 64,
    )
    signed_worker_provenance = sign_worker_image_provenance(
        worker_provenance,
        signing_key_id='independent-oci-builder-key',
        signer=LocalEd25519Signer(builder_key),
    )
    artifacts: dict[str, bytes] = {
        'archive': release_archive_bytes,
        'archive-index': release_archive_index_bytes,
        'clock-policy': clock_policy_bytes,
        'clock-process': clock_process_bytes,
        'clock-process-executable': clock_executable,
        'external-signer-process': signer_process_bytes,
        'external-signer-executable': signer_executable,
        'gossip-comparison': canonical_json_bytes(comparison),
        'gossip-registry-bootstrap': registry_head,
        'gossip-witness-bootstrap': witness_head,
        'monitor-1-policy': canonical_json_bytes(monitor_policies[0]),
        'monitor-2-policy': canonical_json_bytes(monitor_policies[1]),
        'registry-bootstrap': registry_head,
        'registry-bootstrap-witness': registry_witness_proof,
        'registry-policy': registry_policy_bytes,
        'registry-trust': registry_trust_bytes,
        'worker-environment': canonical_json_bytes(environment),
        'worker-implementation': worker_implementation,
        'worker-provenance': canonical_json_bytes(signed_worker_provenance),
        'worker-sandbox': canonical_json_bytes(sandbox),
        'worker-sbom': worker_sbom,
        'witness-bootstrap': witness_head,
        'witness-policy': witness_policy_bytes,
        'witness-trust': witness_trust_bytes,
    }
    roles = {
        'archive': 'release_archive',
        'archive-index': 'release_archive_index',
        'clock-policy': 'clock_health_policy',
        'clock-process': 'clock_health_process',
        'clock-process-executable': 'operator_executable',
        'external-signer-process': 'external_signer_process',
        'external-signer-executable': 'operator_executable',
        'gossip-comparison': 'gossip_comparison_policy',
        'gossip-registry-bootstrap': 'gossip_bootstrap_head',
        'gossip-witness-bootstrap': 'gossip_bootstrap_head',
        'monitor-1-policy': 'gossip_monitor_policy',
        'monitor-2-policy': 'gossip_monitor_policy',
        'registry-bootstrap': 'registry_bootstrap_head',
        'registry-bootstrap-witness': 'registry_bootstrap_witness',
        'registry-policy': 'registry_policy',
        'registry-trust': 'registry_trust_policy',
        'worker-environment': 'worker_environment',
        'worker-implementation': 'worker_implementation',
        'worker-provenance': 'worker_image_provenance',
        'worker-sandbox': 'worker_sandbox_policy',
        'worker-sbom': 'worker_image_sbom',
        'witness-bootstrap': 'witness_bootstrap_head',
        'witness-policy': 'witness_policy',
        'witness-trust': 'witness_trust_policy',
    }
    artifact_bindings = tuple(
        PublicationArtifactBinding(
            artifact_id=artifact_id,
            role=roles[artifact_id],  # type: ignore[arg-type]
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
        )
        for artifact_id, payload in sorted(artifacts.items())
    )
    manifest = CampaignPublicationManifest(
        campaign_id='future-immport-campaign',
        release_id=release_id,
        created_at=created_at,
        release_authority_id='vaxreplay-release-authority',
        release_signing_key_id='release-key-1',
        artifacts=artifact_bindings,
        registry=RegistryPublicationBinding(
            registry_id=registry_policy.registry_id,
            authority_id=registry_policy.authority_id,
            policy_artifact_id='registry-policy',
            trust_policy_artifact_id='registry-trust',
            bootstrap_head_artifact_id='registry-bootstrap',
            bootstrap_witness_artifact_id='registry-bootstrap-witness',
            signing_public_key_sha256s=tuple(
                sorted(
                    hashlib.sha256(base64.b64decode(item.public_key_base64, validate=True)).hexdigest()
                    for item in registry_trust.signing_keys
                )
            ),
        ),
        witnesses=(
            WitnessPublicationBinding(
                authority_id=witness_policy.authority_id,
                witness_id=witness_policy.witness_id,
                policy_artifact_id='witness-policy',
                trust_policy_artifact_id='witness-trust',
                bootstrap_head_artifact_id='witness-bootstrap',
                signing_public_key_sha256=witness_trust.public_key_sha256,
            ),
        ),
        gossip=GossipPublicationBinding(
            comparison_policy_artifact_id='gossip-comparison',
            monitors=tuple(
                GossipMonitorPublicationBinding(
                    monitor_id=policy.monitor_id,
                    policy_artifact_id=f'monitor-{index}-policy',
                    report_signing_public_key_sha256=hashlib.sha256(
                        base64.b64decode(policy.report_signing_public_key_base64, validate=True)
                    ).hexdigest(),
                )
                for index, policy in enumerate(monitor_policies, start=1)
            ),
            bootstrap_heads=(
                GossipBootstrapHeadBinding(
                    stream_id='registry-heads',
                    artifact_id='gossip-registry-bootstrap',
                ),
                GossipBootstrapHeadBinding(
                    stream_id='witness-heads',
                    artifact_id='gossip-witness-bootstrap',
                ),
            ),
        ),
        clock_health_policy_artifact_id='clock-policy',
        runtime_processes=(
            RuntimeProcessPublicationBinding(
                process_id=clock_process.process_id,
                purpose='clock_health',
                process_config_artifact_id='clock-process',
                executable_artifact_id='clock-process-executable',
            ),
            RuntimeProcessPublicationBinding(
                process_id=signer_process.process_id,
                purpose='external_signer',
                process_config_artifact_id='external-signer-process',
                executable_artifact_id='external-signer-executable',
            ),
        ),
        workers=(
            WorkerPublicationBinding(
                worker_id='immport-source-verifier-and-adapter',
                purpose='source_verifier',
                implementation_artifact_id='worker-implementation',
                environment_artifact_id='worker-environment',
                sandbox_policy_artifact_id='worker-sandbox',
                sbom_artifact_id='worker-sbom',
                provenance_artifact_id='worker-provenance',
                image_ref=environment.image_ref,
                resolved_image_id=environment.expected_image_id,
                provenance_builder_id=worker_provenance.builder_id,
                provenance_build_type=worker_provenance.build_type,
            ),
        ),
        archive_artifact_id='archive',
        archive_index_artifact_id='archive-index',
    )
    release_key = Ed25519PrivateKey.generate()
    signed_manifest = sign_campaign_publication_manifest(
        manifest,
        signer=LocalEd25519Signer(release_key),
    )
    signed_manifest_bytes = canonical_json_bytes(signed_manifest)
    publication_keys = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    trust = CampaignPublicationTrustPolicy(
        trust_policy_id='future-immport-publication-root',
        campaign_id=manifest.campaign_id,
        release_authority_id=manifest.release_authority_id,
        release_organization_id='vaxreplay-release-organization',
        release_failure_domain_id='release-control-plane',
        release_keys=(
            PublicationTrustedKey(
                key_id=manifest.release_signing_key_id,
                public_key_base64=base64.b64encode(_public_bytes(release_key)).decode('ascii'),
                valid_from=created_at - timedelta(days=1),
            ),
        ),
        worker_build_authorities=(
            PublicationAuthorityTrust(
                authority_id=worker_provenance.builder_id,
                organization_id='independent-builder-organization',
                failure_domain_id='independent-builder-control-plane',
                keys=(
                    PublicationTrustedKey(
                        key_id='independent-oci-builder-key',
                        public_key_base64=base64.b64encode(_public_bytes(builder_key)).decode('ascii'),
                        valid_from=created_at - timedelta(days=1),
                    ),
                ),
            ),
        ),
        publication_authorities=tuple(
            PublicationAuthorityTrust(
                authority_id=f'independent-publisher-{index}',
                organization_id=f'independent-publisher-{index}-organization',
                failure_domain_id=f'independent-publisher-{index}-control-plane',
                keys=(
                    PublicationTrustedKey(
                        key_id=f'independent-publisher-{index}-key',
                        public_key_base64=base64.b64encode(_public_bytes(key)).decode('ascii'),
                        valid_from=created_at - timedelta(days=1),
                    ),
                ),
            )
            for index, key in enumerate(publication_keys, start=1)
        ),
        precommitted_control_artifacts=tuple(
            item for item in artifact_bindings if item.role not in {'release_archive', 'release_archive_index'}
        ),
        required_workers=manifest.workers,
        minimum_independent_receipts=2,
    )
    archive_binding = next(item for item in artifact_bindings if item.artifact_id == 'archive')
    receipts = tuple(
        canonical_json_bytes(
            sign_publication_receipt(
                PublicationReceiptStatement(
                    receipt_id=f'independent-publication-receipt-{index}',
                    authority_id=f'independent-publisher-{index}',
                    signing_key_id=f'independent-publisher-{index}-key',
                    campaign_id=manifest.campaign_id,
                    release_id=manifest.release_id,
                    signed_manifest_sha256=hashlib.sha256(signed_manifest_bytes).hexdigest(),
                    signed_manifest_bytes=len(signed_manifest_bytes),
                    archive_sha256=archive_binding.sha256,
                    archive_bytes=archive_binding.byte_count,
                    published_at=created_at + timedelta(seconds=index),
                    publication_uri=f'https://publisher-{index}.example/releases/{manifest.release_id}',
                ),
                signer=LocalEd25519Signer(key),
            )
        )
        for index, key in enumerate(publication_keys, start=1)
    )
    return PublicationFixture(
        signed_manifest_bytes=signed_manifest_bytes,
        trust_policy_bytes=canonical_json_bytes(trust),
        artifacts=artifacts,
        receipts=receipts,
        verified_at=created_at + timedelta(seconds=3),
        manifest=manifest,
        release_key=release_key,
        builder_key=builder_key,
    )


def test_offline_publication_verifier_binds_all_tier_a_trust_materials(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    report = verify_campaign_publication(
        fixture.signed_manifest_bytes,
        trust_policy_bytes=fixture.trust_policy_bytes,
        expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
        artifacts=fixture.artifacts,
        publication_receipt_bytes=fixture.receipts,
        verified_at=fixture.verified_at,
    )
    assert report.campaign_id == fixture.manifest.campaign_id
    assert report.verified_artifact_count == len(fixture.artifacts)
    assert report.independent_publication_authority_ids == (
        'independent-publisher-1',
        'independent-publisher-2',
    )
    assert report.publication_organization_ids == (
        'independent-publisher-1-organization',
        'independent-publisher-2-organization',
    )
    assert report.out_of_band_trust_policy_digest_verified
    assert report.out_of_band_control_artifacts_verified
    assert report.retained_runtime_process_configurations_verified
    assert report.retained_runtime_executables_verified
    assert report.every_control_timestamp_bounded_by_release
    assert report.required_worker_inventory_verified
    assert not report.external_organizational_independence_cryptographically_proven


def test_publication_verifier_uses_one_immutable_artifact_snapshot(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    later = dict(fixture.artifacts)
    later[fixture.manifest.archive_artifact_id] = b'unsigned replacement archive'
    artifacts = _ChameleonBytesMapping(fixture.artifacts, later)

    report = verify_campaign_publication(
        fixture.signed_manifest_bytes,
        trust_policy_bytes=fixture.trust_policy_bytes,
        expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
        artifacts=artifacts,
        publication_receipt_bytes=fixture.receipts,
        verified_at=fixture.verified_at,
    )

    assert report.archive_sha256 == hashlib.sha256(fixture.artifacts[fixture.manifest.archive_artifact_id]).hexdigest()
    assert artifacts.indexed_reads == 0


def test_publication_library_requires_the_out_of_band_trust_digest(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    with pytest.raises(CampaignPublicationError, match='out-of-band expected digest'):
        verify_campaign_publication(
            fixture.signed_manifest_bytes,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256='f' * 64,
            artifacts=fixture.artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_registry_checkpoint_witness_must_be_in_the_published_witness_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _publication(tmp_path)
    real_verify = campaign_publication_module._verify_registry_binding

    def return_unpublished_witness(*args, **kwargs):
        real_verify(*args, **kwargs)
        return ('unpublished-authority', 'unpublished-witness', 'f' * 64)

    monkeypatch.setattr(
        campaign_publication_module,
        '_verify_registry_binding',
        return_unpublished_witness,
    )
    with pytest.raises(CampaignPublicationError, match='absent from the publication witness'):
        verify_campaign_publication(
            fixture.signed_manifest_bytes,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=fixture.artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_campaign_publication_v02_schemas_reject_v01_payloads(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    report = verify_campaign_publication(
        fixture.signed_manifest_bytes,
        trust_policy_bytes=fixture.trust_policy_bytes,
        expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
        artifacts=fixture.artifacts,
        publication_receipt_bytes=fixture.receipts,
        verified_at=fixture.verified_at,
    )
    inputs = (
        (
            CampaignPublicationManifest,
            fixture.manifest.model_dump(mode='json'),
            'vaxreplay.campaign-publication-manifest.v0.1',
        ),
        (
            SignedCampaignPublicationManifest,
            SignedCampaignPublicationManifest.model_validate_json(fixture.signed_manifest_bytes).model_dump(
                mode='json'
            ),
            'vaxreplay.signed-campaign-publication-manifest.v0.1',
        ),
        (
            CampaignPublicationTrustPolicy,
            CampaignPublicationTrustPolicy.model_validate_json(fixture.trust_policy_bytes).model_dump(mode='json'),
            'vaxreplay.campaign-publication-trust-policy.v0.1',
        ),
        (
            PublicationVerificationReport,
            report.model_dump(mode='json'),
            'vaxreplay.publication-verification-report.v0.1',
        ),
    )
    for model, payload, old_schema_version in inputs:
        with pytest.raises(ValueError, match='schema_version'):
            model.model_validate({**payload, 'schema_version': old_schema_version})


def test_campaign_manifest_rejects_aggregate_inventory_before_artifact_reads(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    payload = fixture.manifest.model_dump(mode='python')
    payload['artifacts'] = tuple(
        item.model_copy(update={'byte_count': MAX_PUBLICATION_ARTIFACT_BYTES}) for item in fixture.manifest.artifacts
    )
    with pytest.raises(ValueError, match='total-byte limit'):
        CampaignPublicationManifest.model_validate(payload)


def test_tier_a_verifier_rejects_precommitted_policy_without_runtime_trust(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    artifacts = dict(fixture.artifacts)
    policy = SelectionRegistryPolicy.model_validate_json(artifacts['registry-policy'])
    artifacts['registry-policy'] = canonical_json_bytes(
        policy.model_copy(
            update={
                'clock_health_policy_sha256': None,
                'clock_health_process_sha256': None,
                'external_signer_process_sha256': None,
            }
        )
    )
    bindings = tuple(
        item.model_copy(
            update={
                'sha256': hashlib.sha256(artifacts[item.artifact_id]).hexdigest(),
                'byte_count': len(artifacts[item.artifact_id]),
            }
        )
        if item.artifact_id == 'registry-policy'
        else item
        for item in fixture.manifest.artifacts
    )
    manifest = fixture.manifest.model_copy(update={'artifacts': bindings})
    signed = canonical_json_bytes(
        sign_campaign_publication_manifest(
            manifest,
            signer=LocalEd25519Signer(fixture.release_key),
        )
    )
    trust = CampaignPublicationTrustPolicy.model_validate_json(fixture.trust_policy_bytes)
    trust_bytes = canonical_json_bytes(
        trust.model_copy(
            update={
                'precommitted_control_artifacts': tuple(
                    item for item in bindings if item.role not in {'release_archive', 'release_archive_index'}
                )
            }
        )
    )
    with pytest.raises(CampaignPublicationError, match='omits required Tier-A runtime-trust'):
        verify_campaign_publication(
            signed,
            trust_policy_bytes=trust_bytes,
            expected_trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
            artifacts=artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_tier_a_verifier_requires_retained_runtime_process_configuration(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    artifacts = dict(fixture.artifacts)
    policy = SelectionRegistryPolicy.model_validate_json(artifacts['registry-policy'])
    artifacts['registry-policy'] = canonical_json_bytes(
        policy.model_copy(update={'clock_health_process_sha256': 'f' * 64})
    )
    bindings = tuple(
        item.model_copy(
            update={
                'sha256': hashlib.sha256(artifacts[item.artifact_id]).hexdigest(),
                'byte_count': len(artifacts[item.artifact_id]),
            }
        )
        if item.artifact_id == 'registry-policy'
        else item
        for item in fixture.manifest.artifacts
    )
    manifest = fixture.manifest.model_copy(update={'artifacts': bindings})
    signed = canonical_json_bytes(
        sign_campaign_publication_manifest(
            manifest,
            signer=LocalEd25519Signer(fixture.release_key),
        )
    )
    trust = CampaignPublicationTrustPolicy.model_validate_json(fixture.trust_policy_bytes)
    trust_bytes = canonical_json_bytes(
        trust.model_copy(
            update={
                'precommitted_control_artifacts': tuple(
                    item for item in bindings if item.role not in {'release_archive', 'release_archive_index'}
                )
            }
        )
    )
    with pytest.raises(CampaignPublicationError, match='configuration was not retained'):
        verify_campaign_publication(
            signed,
            trust_policy_bytes=trust_bytes,
            expected_trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
            artifacts=artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_campaign_control_timestamps_cannot_postdate_release_assembly(
    tmp_path: Path,
) -> None:
    future_provenance = _publication(
        tmp_path / 'future-provenance',
        worker_provenance_offset_seconds=1,
    )
    with pytest.raises(CampaignPublicationError, match='provenance postdates'):
        verify_campaign_publication(
            future_provenance.signed_manifest_bytes,
            trust_policy_bytes=future_provenance.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(future_provenance.trust_policy_bytes).hexdigest(),
            artifacts=future_provenance.artifacts,
            publication_receipt_bytes=future_provenance.receipts,
            verified_at=future_provenance.verified_at,
        )

    old_release = _publication(
        tmp_path / 'future-bootstrap',
        manifest_time_offset_seconds=-3600,
    )
    with pytest.raises(CampaignPublicationError, match='bootstrap'):
        verify_campaign_publication(
            old_release.signed_manifest_bytes,
            trust_policy_bytes=old_release.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(old_release.trust_policy_bytes).hexdigest(),
            artifacts=old_release.artifacts,
            publication_receipt_bytes=old_release.receipts,
            verified_at=old_release.verified_at,
        )


def test_self_carried_release_key_cannot_replace_out_of_band_anchor(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    attacker = Ed25519PrivateKey.generate()
    attacker_signed = canonical_json_bytes(
        sign_campaign_publication_manifest(
            fixture.manifest,
            signer=LocalEd25519Signer(attacker),
        )
    )
    with pytest.raises(CampaignPublicationError, match='signature is invalid'):
        verify_campaign_publication(
            attacker_signed,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=fixture.artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_publication_rejects_artifact_substitution_and_extra_files(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    substituted = dict(fixture.artifacts)
    substituted['worker-sbom'] += b' attacker mutation'
    with pytest.raises(CampaignPublicationError, match='differs from its exact binding'):
        verify_campaign_publication(
            fixture.signed_manifest_bytes,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=substituted,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )
    extra = dict(fixture.artifacts)
    extra['unmanifested-anchor'] = b'evil public key'
    with pytest.raises(CampaignPublicationError, match='artifact IDs differ'):
        verify_campaign_publication(
            fixture.signed_manifest_bytes,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=extra,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_one_publication_authority_cannot_clone_itself_to_satisfy_quorum(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    with pytest.raises(CampaignPublicationError, match='duplicate publication authority'):
        verify_campaign_publication(
            fixture.signed_manifest_bytes,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=fixture.artifacts,
            publication_receipt_bytes=(fixture.receipts[0], fixture.receipts[0]),
            verified_at=fixture.verified_at,
        )


def test_policy_rejects_distinct_labels_in_one_organization_or_failure_domain(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    trust = CampaignPublicationTrustPolicy.model_validate_json(fixture.trust_policy_bytes)
    first, second = trust.publication_authorities
    for replacement in (
        second.model_copy(update={'organization_id': first.organization_id}),
        second.model_copy(update={'failure_domain_id': first.failure_domain_id}),
    ):
        with pytest.raises(ValueError, match='organizations|failure domains'):
            CampaignPublicationTrustPolicy.model_validate(
                {
                    **trust.model_dump(mode='python'),
                    'publication_authorities': (first, replacement),
                }
            )


def test_release_signature_cannot_make_mismatched_worker_provenance_valid(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    artifacts = dict(fixture.artifacts)
    signed_provenance = SignedWorkerImageProvenance.model_validate_json(artifacts['worker-provenance'])
    provenance = signed_provenance.statement
    artifacts['worker-provenance'] = canonical_json_bytes(
        sign_worker_image_provenance(
            provenance.model_copy(update={'implementation_sha256': 'f' * 64}),
            signing_key_id=signed_provenance.signing_key_id,
            signer=LocalEd25519Signer(fixture.builder_key),
        )
    )
    bindings = tuple(
        item.model_copy(
            update={
                'sha256': hashlib.sha256(artifacts[item.artifact_id]).hexdigest(),
                'byte_count': len(artifacts[item.artifact_id]),
            }
        )
        if item.artifact_id == 'worker-provenance'
        else item
        for item in fixture.manifest.artifacts
    )
    manifest = fixture.manifest.model_copy(update={'artifacts': bindings})
    signed = canonical_json_bytes(
        sign_campaign_publication_manifest(
            manifest,
            signer=LocalEd25519Signer(fixture.release_key),
        )
    )
    with pytest.raises(CampaignPublicationError, match='control bytes differ'):
        verify_campaign_publication(
            signed,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )


def test_self_carried_builder_key_cannot_replace_out_of_band_builder_anchor(tmp_path: Path) -> None:
    fixture = _publication(tmp_path)
    artifacts = dict(fixture.artifacts)
    original = SignedWorkerImageProvenance.model_validate_json(artifacts['worker-provenance'])
    attacker = Ed25519PrivateKey.generate()
    artifacts['worker-provenance'] = canonical_json_bytes(
        sign_worker_image_provenance(
            original.statement,
            signing_key_id=original.signing_key_id,
            signer=LocalEd25519Signer(attacker),
        )
    )
    bindings = tuple(
        item.model_copy(
            update={
                'sha256': hashlib.sha256(artifacts[item.artifact_id]).hexdigest(),
                'byte_count': len(artifacts[item.artifact_id]),
            }
        )
        if item.artifact_id == 'worker-provenance'
        else item
        for item in fixture.manifest.artifacts
    )
    signed = canonical_json_bytes(
        sign_campaign_publication_manifest(
            fixture.manifest.model_copy(update={'artifacts': bindings}),
            signer=LocalEd25519Signer(fixture.release_key),
        )
    )
    with pytest.raises(CampaignPublicationError, match='control bytes differ'):
        verify_campaign_publication(
            signed,
            trust_policy_bytes=fixture.trust_policy_bytes,
            expected_trust_policy_sha256=hashlib.sha256(fixture.trust_policy_bytes).hexdigest(),
            artifacts=artifacts,
            publication_receipt_bytes=fixture.receipts,
            verified_at=fixture.verified_at,
        )
