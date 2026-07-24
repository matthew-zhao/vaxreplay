"""Atomic public/private packaging for provenance-bound evaluation releases."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

from vaxreplay.aggregation import make_suite_manifest
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, jsonl_text
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    CaseUniverseDisposition,
    CaseUniverseEntry,
    CaseUniverseManifest,
    case_selection_audit_sha256,
    case_universe_sha256,
    validate_case_selection_bindings,
    validate_case_selection_inventory,
)
from vaxreplay.case_schema import RANKING_REWARD_VERSION, LabelCommitmentScheme, Split
from vaxreplay.contamination import (
    AuditDisposition,
    ContaminationAuditManifest,
    ContaminationAuditPolicy,
    audit_manifest_sha256,
)
from vaxreplay.contamination import (
    model_sha256 as contamination_model_sha256,
)
from vaxreplay.dataset import (
    SplitAdmissionManifest,
    make_split_admission_manifest,
    split_admission_manifest_sha256,
    validate_split_admission_manifest,
    validate_split_admission_subset,
)
from vaxreplay.iedb.adapter import audit_episode
from vaxreplay.iedb.raw_schema import IEDB_ADAPTER_ID, IedbEpisodeSpec, IedbPrivateAudit
from vaxreplay.prompt import PromptVariant, model_facing_payload_bytes
from vaxreplay.release_schema import (
    ChallengeAdmissionCommitment,
    ChallengeTemporalAdmissionBinding,
    PrivateFileBinding,
    PrivateReleaseManifest,
    PublicReleaseManifest,
    ReleaseEpisodeBinding,
    ReleasePurpose,
    release_model_sha256,
)
from vaxreplay.runner.challenge import (
    LoadedChallengeBundle,
    build_challenge_bundle,
    challenge_admission_sha256,
    load_challenge_bundle,
)
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy
from vaxreplay.temporal_schema import (
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    PROTOCOL_ARTIFACT_NAMES,
    DecisionProtocolCommitments,
    DecisionTimeConfig,
    OutcomeSnapshotCommitment,
    OutcomeTargetAvailability,
    TemporalAdmissionEnvelope,
    TemporalAdmissionUse,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalProvenanceBasis,
    TemporalReceiptAuthority,
    TemporalReceiptVerifier,
    TemporalSourceTier,
    build_decision_snapshot_commitment,
    require_retrospective_temporal_admission,
)

_MAX_RELEASE_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_RELEASE_FILE_BYTES = 256 * 1024 * 1024
_MAX_RELEASE_FILES = 100_000
_MAX_RELEASE_DIRECTORIES = 20_000


class ReleaseIntegrityError(ValueError):
    """Raised when a release is incomplete, overclaims provenance, or was modified."""


@dataclass(frozen=True)
class TemporalAdmissionMaterial:
    admission: TemporalAdmissionEnvelope
    protocol_artifacts: dict[str, bytes]
    raw_outcome_source: bytes
    label_derivation_audit: bytes
    receipt_proofs: dict[str, bytes]


type TierBSourceMaterialVerifier = Callable[
    [str, EpisodeBundle, TemporalAdmissionMaterial, bytes, Mapping[str, bytes], CaseUniverseEntry],
    bool,
]
type CaseUniverseVerifier = Callable[[CaseUniverseManifest, bytes, CaseSelectionAudit, bytes], bool]
type ContaminationAuditVerifier = Callable[
    [ContaminationAuditManifest, ContaminationAuditPolicy, CaseUniverseManifest, CaseSelectionAudit],
    bool,
]


@dataclass(frozen=True)
class BuiltRelease:
    public_root: Path
    private_root: Path
    public_manifest: PublicReleaseManifest
    public_manifest_sha256: str
    private_manifest: PrivateReleaseManifest
    challenge: LoadedChallengeBundle
    episode_roots: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedRelease:
    public_root: Path
    private_root: Path
    public_manifest: PublicReleaseManifest
    public_manifest_sha256: str
    private_manifest: PrivateReleaseManifest
    challenge: LoadedChallengeBundle
    policy: RunnerPolicy
    split_admission: SplitAdmissionManifest
    case_universe: CaseUniverseManifest | None
    case_selection_audit: CaseSelectionAudit | None
    contamination_policy: ContaminationAuditPolicy | None
    contamination_audit_manifest: ContaminationAuditManifest | None
    temporal_admissions: tuple[TemporalAdmissionEnvelope, ...]
    bundles: tuple[EpisodeBundle, ...]


def public_release_sha256(manifest: PublicReleaseManifest) -> str:
    return release_model_sha256(manifest)


def build_synthetic_integration_release(
    *,
    release_id: str,
    challenge_id: str,
    suite_id: str,
    episode_dirs: tuple[Path, ...],
    policy: RunnerPolicy,
    receipt_key_id: str,
    public_output_dir: Path,
    private_output_dir: Path,
) -> BuiltRelease:
    """Package test-shaped synthetic episodes without claiming hidden or scientific labels.

    This path deliberately emits Tier C / ``synthetic_integration`` admissions and can never
    produce an officially sealed release. It validates the same HMAC, challenge, runner, and
    scorer mechanics that a future release will use.
    """

    if policy.required_isolation != IsolationTier.DEVELOPMENT:
        raise ValueError('synthetic integration releases require an explicit development policy')
    if len(receipt_key_id) != 64 or any(character not in '0123456789abcdef' for character in receipt_key_id):
        raise ValueError('receipt_key_id must be a lowercase SHA-256 digest')
    bundles = tuple(
        sorted(
            (EpisodeBundle.load(path.expanduser().resolve(), include_private=True) for path in episode_dirs),
            key=lambda bundle: bundle.manifest.episode_id,
        )
    )
    if not bundles:
        raise ValueError('cannot package a release without episodes')
    _validate_synthetic_release_bundles(bundles)

    split_admission = make_split_admission_manifest(f'{release_id}-selected-only', bundles)
    materials = tuple(
        _build_tier_c_material(bundle, release_id=release_id, source_audit=_source_audit(bundle)) for bundle in bundles
    )
    admission = ChallengeAdmissionCommitment(
        release_id=release_id,
        purpose=ReleasePurpose.SYNTHETIC_INTEGRATION,
        split_admission_sha256=split_admission_manifest_sha256(split_admission),
        split_inventory_complete=False,
        episodes=tuple(
            ChallengeTemporalAdmissionBinding(
                episode_id=bundle.manifest.episode_id,
                manifest_sha256=bundle.manifest_sha256,
                source_tier=material.admission.source_tier,
                temporal_admission_sha256=release_model_sha256(material.admission),
            )
            for bundle, material in zip(bundles, materials, strict=True)
        ),
    )

    public_requested = public_output_dir.expanduser().resolve(strict=False)
    private_requested = private_output_dir.expanduser().resolve(strict=False)
    if (
        public_requested == private_requested
        or public_requested in private_requested.parents
        or private_requested in public_requested.parents
    ):
        raise ValueError('public and private release outputs must be separate, non-overlapping directories')
    public_target, public_staging = _make_staging(public_requested, 'public release')
    try:
        private_target, private_staging = _make_staging(private_requested, 'private release')
    except BaseException:
        shutil.rmtree(public_staging, ignore_errors=True)
        raise
    private_installed = False
    public_installed = False
    try:
        challenge = build_challenge_bundle(
            public_staging / 'challenge',
            challenge_id=challenge_id,
            suite_id=suite_id,
            episode_dirs=(bundle.root for bundle in bundles),
            admission=admission,
        )
        policy_bytes = canonical_json_bytes(policy)
        (public_staging / 'policy.json').write_bytes(policy_bytes)
        policy_sha256 = _sha256(policy_bytes)

        file_bindings: list[PrivateFileBinding] = []
        _write_private_file(
            private_staging,
            'split-admission.json',
            canonical_json_bytes(split_admission),
            file_bindings,
            bind=False,
        )
        release_episode_bindings: list[ReleaseEpisodeBinding] = []
        episode_roots: list[Path] = []
        for ordinal, (bundle, material, admission_binding) in enumerate(
            zip(bundles, materials, admission.episodes, strict=True)
        ):
            episode_path = f'episodes/{ordinal:06d}'
            episode_root = private_staging / episode_path
            _write_scoring_episode(private_staging, episode_path, bundle, file_bindings)
            episode_roots.append(episode_root)
            temporal_path = f'temporal/{ordinal:06d}.json'
            _write_private_file(
                private_staging,
                temporal_path,
                canonical_json_bytes(material.admission),
                file_bindings,
            )
            for name, payload in sorted(material.protocol_artifacts.items()):
                _write_private_file(
                    private_staging,
                    f'protocols/{ordinal:06d}/{name}.json',
                    payload,
                    file_bindings,
                )
            _write_private_file(
                private_staging,
                f'protocols/{ordinal:06d}/raw-outcome-source.json',
                material.raw_outcome_source,
                file_bindings,
            )
            _write_private_file(
                private_staging,
                f'protocols/{ordinal:06d}/label-derivation-audit.json',
                material.label_derivation_audit,
                file_bindings,
            )
            for receipt_id, payload in sorted(material.receipt_proofs.items()):
                _write_private_file(
                    private_staging,
                    f'proofs/{ordinal:06d}/{receipt_id}.json',
                    payload,
                    file_bindings,
                )
            source_audit_bytes = material.label_derivation_audit
            _write_private_file(
                private_staging,
                f'source-audits/{ordinal:06d}.json',
                source_audit_bytes,
                file_bindings,
            )
            assert bundle.manifest.label_commitment_key_id is not None
            release_episode_bindings.append(
                ReleaseEpisodeBinding(
                    ordinal=ordinal,
                    episode_id=bundle.manifest.episode_id,
                    private_path=episode_path,
                    manifest_sha256=bundle.manifest_sha256,
                    labels_sha256=bundle.manifest.labels_sha256,
                    label_commitment_key_id=bundle.manifest.label_commitment_key_id,
                    temporal_admission_sha256=admission_binding.temporal_admission_sha256,
                    source_tier=admission_binding.source_tier,
                    source_audit_sha256=_sha256(source_audit_bytes),
                )
            )

        private_manifest = PrivateReleaseManifest(
            release_id=release_id,
            purpose=ReleasePurpose.SYNTHETIC_INTEGRATION,
            challenge_id=challenge_id,
            challenge_bundle_sha256=challenge.manifest_sha256,
            suite_manifest_sha256=challenge.manifest.suite_manifest_sha256,
            admission_sha256=challenge_admission_sha256(admission),
            policy_sha256=policy_sha256,
            receipt_key_id=receipt_key_id,
            split_admission_sha256=split_admission_manifest_sha256(split_admission),
            split_inventory_complete=False,
            episodes=tuple(release_episode_bindings),
            files=tuple(sorted(file_bindings, key=lambda binding: binding.path)),
        )
        private_manifest_bytes = canonical_json_bytes(private_manifest)
        (private_staging / 'package.json').write_bytes(private_manifest_bytes)

        public_manifest = PublicReleaseManifest(
            release_id=release_id,
            purpose=ReleasePurpose.SYNTHETIC_INTEGRATION,
            sealed_eligible=False,
            challenge_id=challenge_id,
            challenge_bundle_sha256=challenge.manifest_sha256,
            suite_manifest_sha256=challenge.manifest.suite_manifest_sha256,
            admission_sha256=challenge_admission_sha256(admission),
            policy_sha256=policy_sha256,
            receipt_key_id=receipt_key_id,
            private_package_sha256=_sha256(private_manifest_bytes),
            episode_count=len(bundles),
        )
        (public_staging / 'release.json').write_bytes(canonical_json_bytes(public_manifest))

        _set_tree_permissions(private_staging, directory_mode=0o700, file_mode=0o600)
        _set_tree_permissions(public_staging, directory_mode=0o755, file_mode=0o644)
        os.replace(private_staging, private_target)
        private_installed = True
        os.replace(public_staging, public_target)
        public_installed = True
        loaded = load_release(
            public_target,
            private_target,
            expected_public_release_sha256=public_release_sha256(public_manifest),
        )
    except BaseException:
        shutil.rmtree(public_staging, ignore_errors=True)
        shutil.rmtree(private_staging, ignore_errors=True)
        if public_installed:
            shutil.rmtree(public_target, ignore_errors=True)
        if private_installed:
            shutil.rmtree(private_target, ignore_errors=True)
        raise
    return BuiltRelease(
        public_root=loaded.public_root,
        private_root=loaded.private_root,
        public_manifest=loaded.public_manifest,
        public_manifest_sha256=loaded.public_manifest_sha256,
        private_manifest=loaded.private_manifest,
        challenge=loaded.challenge,
        episode_roots=tuple(bundle.root for bundle in loaded.bundles),
    )


def build_retrospective_research_release(
    *,
    release_id: str,
    challenge_id: str,
    suite_id: str,
    selected_episode_dirs: tuple[Path, ...],
    complete_inventory_episode_dirs: tuple[Path, ...],
    temporal_materials: Mapping[str, TemporalAdmissionMaterial],
    source_audits: Mapping[str, bytes],
    case_universe: CaseUniverseManifest,
    case_universe_proof: bytes,
    case_selection_audit: CaseSelectionAudit,
    verifier_policy: bytes,
    contamination_policy: ContaminationAuditPolicy,
    contamination_audit_manifest: ContaminationAuditManifest,
    temporal_receipt_verifier: TemporalReceiptVerifier,
    source_material_verifier: TierBSourceMaterialVerifier,
    case_universe_verifier: CaseUniverseVerifier,
    contamination_audit_verifier: ContaminationAuditVerifier,
    policy: RunnerPolicy,
    receipt_key_id: str,
    public_output_dir: Path,
    private_output_dir: Path,
    extra_private_files: Mapping[str, Mapping[str, bytes]] | None = None,
) -> BuiltRelease:
    """Package real, selected Tier B test episodes against a complete split inventory.

    The required verifier callbacks are organizer-controlled trusted code. The temporal
    verifier authenticates each independent receipt over the derived artifacts; the source
    verifier authenticates exact archived literature bytes, their availability proofs, panel
    completeness, and the decision-package seal represented by ``source_audit`` plus the private
    source files. The case-universe verifier authenticates the pre-outcome universe seal and the
    committed policy that governs both verifier configuration and post-outcome case selection.
    The contamination verifier authenticates the complete organizer-side multi-judge audit. Every
    admitted case must pass and bind the exact final model-facing bytes; quarantined cases remain in
    the case-selection inventory. The public package includes the audit policy while protected
    comparisons and findings stay in the private manifest.
    Optional extra files are stored below the selected episode's
    ``source-materials`` namespace.
    """

    if policy.required_isolation != IsolationTier.OFFICIAL:
        raise ValueError('retrospective research releases require an official isolation policy')
    _validate_receipt_key_id(receipt_key_id)
    bundles = _load_unique_bundles(selected_episode_dirs, label='selected')
    inventory_bundles = _load_unique_bundles(complete_inventory_episode_dirs, label='complete inventory')
    if not bundles:
        raise ValueError('cannot package a release without selected episodes')
    if not inventory_bundles:
        raise ValueError('retrospective releases require a non-empty complete inventory')
    if any(bundle.manifest.synthetic for bundle in inventory_bundles):
        raise ValueError('retrospective complete inventories cannot contain synthetic episodes')
    for bundle in bundles:
        _validate_retrospective_release_bundle(bundle)

    split_admission = make_split_admission_manifest(f'{release_id}-complete-inventory', inventory_bundles)
    validate_split_admission_subset(split_admission, bundles)
    case_universe = _canonical_case_universe(case_universe)
    case_selection_audit = _canonical_case_selection_audit(case_selection_audit)
    contamination_policy = _canonical_contamination_policy(contamination_policy)
    contamination_audit_manifest = _canonical_contamination_audit_manifest(contamination_audit_manifest)
    if not isinstance(case_universe_proof, bytes) or not case_universe_proof:
        raise ValueError('case-universe proof must be non-empty bytes')
    if (
        len(case_universe_proof) != case_universe.seal.proof_bytes
        or _sha256(case_universe_proof) != case_universe.seal.proof_sha256
    ):
        raise ValueError('case-universe proof does not match its seal')
    if not isinstance(verifier_policy, bytes) or not verifier_policy:
        raise ValueError('verifier policy must be non-empty bytes')
    if case_selection_audit.selection_policy_sha256 != _sha256(verifier_policy):
        raise ValueError('case-selection audit does not use the committed verifier policy')
    validate_case_selection_inventory(case_universe, case_selection_audit, inventory_bundles)
    universe_by_case_id = {entry.case_id: entry for entry in case_universe.entries}
    case_entry_by_episode = {
        record.episode_id: universe_by_case_id[record.case_id]
        for record in case_selection_audit.records
        if record.disposition == CaseSelectionDisposition.ADMITTED
    }
    inventory_outcome_times: list[datetime] = []
    for inventory_bundle in inventory_bundles:
        assert inventory_bundle.private_labels is not None
        inventory_outcome_times.extend(outcome.revealed_at for outcome in inventory_bundle.private_labels.outcomes)
    earliest_inventory_outcome = min(inventory_outcome_times)
    if case_universe.seal.witnessed_at >= earliest_inventory_outcome:
        raise ValueError('case universe must be independently sealed before inventory outcomes')
    try:
        case_universe_verified = case_universe_verifier(
            case_universe,
            case_universe_proof,
            case_selection_audit,
            verifier_policy,
        )
    except Exception as error:
        raise ValueError(f'case-universe verifier failed: {error}') from error
    if not case_universe_verified:
        raise ValueError('case-universe verifier rejected the inventory')
    _validate_contamination_inventory(
        case_universe=case_universe,
        case_selection_audit=case_selection_audit,
        contamination_policy=contamination_policy,
        contamination_audit_manifest=contamination_audit_manifest,
        inventory_bundles=inventory_bundles,
    )
    try:
        contamination_verified = contamination_audit_verifier(
            contamination_audit_manifest,
            contamination_policy,
            case_universe,
            case_selection_audit,
        )
    except Exception as error:
        raise ValueError(f'contamination-audit verifier failed: {error}') from error
    if not contamination_verified:
        raise ValueError('contamination-audit verifier rejected the inventory')
    episode_ids = tuple(bundle.manifest.episode_id for bundle in bundles)
    _require_exact_episode_keys('temporal_materials', temporal_materials, episode_ids)
    _require_exact_episode_keys('source_audits', source_audits, episode_ids)
    extras = extra_private_files or {}
    unknown_extra_ids = sorted(set(extras) - set(episode_ids))
    if unknown_extra_ids:
        raise ValueError(f'extra_private_files contains unknown selected episodes: {unknown_extra_ids}')

    normalized_materials: list[TemporalAdmissionMaterial] = []
    normalized_audits: list[bytes] = []
    normalized_extras: list[dict[str, bytes]] = []
    for bundle in bundles:
        episode_id = bundle.manifest.episode_id
        source_audit = source_audits[episode_id]
        if not isinstance(source_audit, bytes) or not source_audit:
            raise ValueError(f'source audit for {episode_id} must be non-empty bytes')
        material = _normalize_and_verify_tier_b_material(bundle, temporal_materials[episode_id])
        if material.label_derivation_audit != source_audit:
            raise ValueError(f'Tier B source audit for {episode_id} must be the temporal label-derivation audit')
        episode_extras = _normalize_extra_private_files(episode_id, extras.get(episode_id, {}))
        require_retrospective_temporal_admission(
            material.admission,
            bundle,
            receipt_artifacts=material.receipt_proofs,
            receipt_verifier=temporal_receipt_verifier,
            protocol_artifacts=material.protocol_artifacts,
            raw_outcome_source=material.raw_outcome_source,
            label_derivation_audit=material.label_derivation_audit,
        )
        try:
            verified_source = source_material_verifier(
                episode_id,
                bundle,
                material,
                source_audit,
                episode_extras,
                case_entry_by_episode[episode_id],
            )
        except Exception as error:
            raise ValueError(f'Tier B source-material verifier failed for {episode_id}: {error}') from error
        if not verified_source:
            raise ValueError(f'Tier B source-material verifier rejected {episode_id}')
        normalized_materials.append(material)
        normalized_audits.append(source_audit)
        normalized_extras.append(episode_extras)

    admission = ChallengeAdmissionCommitment(
        release_id=release_id,
        purpose=ReleasePurpose.RETROSPECTIVE_RESEARCH,
        split_admission_sha256=split_admission_manifest_sha256(split_admission),
        split_inventory_complete=True,
        case_universe_sha256=case_universe_sha256(case_universe),
        case_selection_audit_sha256=case_selection_audit_sha256(case_selection_audit),
        case_inventory_complete=True,
        verifier_policy_sha256=_sha256(verifier_policy),
        contamination_policy_sha256=contamination_model_sha256(contamination_policy),
        contamination_audit_manifest_sha256=audit_manifest_sha256(contamination_audit_manifest),
        contamination_inventory_complete=True,
        episodes=tuple(
            ChallengeTemporalAdmissionBinding(
                episode_id=bundle.manifest.episode_id,
                manifest_sha256=bundle.manifest_sha256,
                source_tier=material.admission.source_tier,
                temporal_admission_sha256=release_model_sha256(material.admission),
            )
            for bundle, material in zip(bundles, normalized_materials, strict=True)
        ),
    )

    public_requested = public_output_dir.expanduser().resolve(strict=False)
    private_requested = private_output_dir.expanduser().resolve(strict=False)
    if (
        public_requested == private_requested
        or public_requested in private_requested.parents
        or private_requested in public_requested.parents
    ):
        raise ValueError('public and private release outputs must be separate, non-overlapping directories')
    public_target, public_staging = _make_staging(public_requested, 'public release')
    try:
        private_target, private_staging = _make_staging(private_requested, 'private release')
    except BaseException:
        shutil.rmtree(public_staging, ignore_errors=True)
        raise
    private_installed = False
    public_installed = False
    try:
        challenge = build_challenge_bundle(
            public_staging / 'challenge',
            challenge_id=challenge_id,
            suite_id=suite_id,
            episode_dirs=(bundle.root for bundle in bundles),
            admission=admission,
        )
        policy_bytes = canonical_json_bytes(policy)
        (public_staging / 'policy.json').write_bytes(policy_bytes)
        policy_sha256 = _sha256(policy_bytes)
        contamination_policy_bytes = canonical_json_bytes(contamination_policy)
        (public_staging / 'contamination-policy.json').write_bytes(contamination_policy_bytes)

        file_bindings: list[PrivateFileBinding] = []
        _write_private_file(
            private_staging,
            'split-admission.json',
            canonical_json_bytes(split_admission),
            file_bindings,
            bind=False,
        )
        release_episode_bindings: list[ReleaseEpisodeBinding] = []
        _write_private_file(
            private_staging,
            'case-universe.json',
            canonical_json_bytes(case_universe),
            file_bindings,
            bind=False,
        )
        _write_private_file(
            private_staging,
            'case-universe-proof.bin',
            case_universe_proof,
            file_bindings,
            bind=False,
        )
        _write_private_file(
            private_staging,
            'case-selection-audit.json',
            canonical_json_bytes(case_selection_audit),
            file_bindings,
            bind=False,
        )
        _write_private_file(
            private_staging,
            'verifier-policy.json',
            verifier_policy,
            file_bindings,
            bind=False,
        )
        _write_private_file(
            private_staging,
            'contamination-audit.json',
            canonical_json_bytes(contamination_audit_manifest),
            file_bindings,
            bind=False,
        )
        for ordinal, (bundle, material, source_audit, episode_extras, admission_binding) in enumerate(
            zip(
                bundles,
                normalized_materials,
                normalized_audits,
                normalized_extras,
                admission.episodes,
                strict=True,
            )
        ):
            episode_path = f'episodes/{ordinal:06d}'
            _write_scoring_episode(private_staging, episode_path, bundle, file_bindings)
            _write_private_file(
                private_staging,
                f'temporal/{ordinal:06d}.json',
                canonical_json_bytes(material.admission),
                file_bindings,
            )
            for name, payload in sorted(material.protocol_artifacts.items()):
                _write_private_file(
                    private_staging,
                    f'protocols/{ordinal:06d}/{name}.json',
                    payload,
                    file_bindings,
                )
            _write_private_file(
                private_staging,
                f'protocols/{ordinal:06d}/raw-outcome-source.json',
                material.raw_outcome_source,
                file_bindings,
            )
            _write_private_file(
                private_staging,
                f'protocols/{ordinal:06d}/label-derivation-audit.json',
                material.label_derivation_audit,
                file_bindings,
            )
            for receipt_id, payload in sorted(material.receipt_proofs.items()):
                _write_private_file(
                    private_staging,
                    f'proofs/{ordinal:06d}/{receipt_id}.json',
                    payload,
                    file_bindings,
                )
            _write_private_file(
                private_staging,
                f'source-audits/{ordinal:06d}.json',
                source_audit,
                file_bindings,
            )
            for relative_path, payload in sorted(episode_extras.items()):
                _write_private_file(
                    private_staging,
                    f'source-materials/{ordinal:06d}/{relative_path}',
                    payload,
                    file_bindings,
                )
            assert bundle.manifest.label_commitment_key_id is not None
            release_episode_bindings.append(
                ReleaseEpisodeBinding(
                    ordinal=ordinal,
                    episode_id=bundle.manifest.episode_id,
                    private_path=episode_path,
                    manifest_sha256=bundle.manifest_sha256,
                    labels_sha256=bundle.manifest.labels_sha256,
                    label_commitment_key_id=bundle.manifest.label_commitment_key_id,
                    temporal_admission_sha256=admission_binding.temporal_admission_sha256,
                    source_tier=admission_binding.source_tier,
                    source_audit_sha256=_sha256(source_audit),
                )
            )

        private_manifest = PrivateReleaseManifest(
            release_id=release_id,
            purpose=ReleasePurpose.RETROSPECTIVE_RESEARCH,
            challenge_id=challenge_id,
            challenge_bundle_sha256=challenge.manifest_sha256,
            suite_manifest_sha256=challenge.manifest.suite_manifest_sha256,
            admission_sha256=challenge_admission_sha256(admission),
            policy_sha256=policy_sha256,
            receipt_key_id=receipt_key_id,
            split_admission_sha256=split_admission_manifest_sha256(split_admission),
            split_inventory_complete=True,
            case_universe_path='case-universe.json',
            case_universe_sha256=case_universe_sha256(case_universe),
            case_universe_proof_path='case-universe-proof.bin',
            case_selection_audit_path='case-selection-audit.json',
            case_selection_audit_sha256=case_selection_audit_sha256(case_selection_audit),
            verifier_policy_path='verifier-policy.json',
            verifier_policy_sha256=_sha256(verifier_policy),
            case_inventory_complete=True,
            contamination_audit_manifest_path='contamination-audit.json',
            contamination_audit_manifest_sha256=audit_manifest_sha256(contamination_audit_manifest),
            contamination_inventory_complete=True,
            episodes=tuple(release_episode_bindings),
            files=tuple(sorted(file_bindings, key=lambda binding: binding.path)),
        )
        private_manifest_bytes = canonical_json_bytes(private_manifest)
        (private_staging / 'package.json').write_bytes(private_manifest_bytes)
        public_manifest = PublicReleaseManifest(
            release_id=release_id,
            purpose=ReleasePurpose.RETROSPECTIVE_RESEARCH,
            sealed_eligible=False,
            challenge_id=challenge_id,
            challenge_bundle_sha256=challenge.manifest_sha256,
            suite_manifest_sha256=challenge.manifest.suite_manifest_sha256,
            admission_sha256=challenge_admission_sha256(admission),
            policy_sha256=policy_sha256,
            contamination_policy_path='contamination-policy.json',
            contamination_policy_sha256=contamination_model_sha256(contamination_policy),
            receipt_key_id=receipt_key_id,
            private_package_sha256=_sha256(private_manifest_bytes),
            episode_count=len(bundles),
        )
        (public_staging / 'release.json').write_bytes(canonical_json_bytes(public_manifest))

        _set_tree_permissions(private_staging, directory_mode=0o700, file_mode=0o600)
        _set_tree_permissions(public_staging, directory_mode=0o755, file_mode=0o644)
        os.replace(private_staging, private_target)
        private_installed = True
        os.replace(public_staging, public_target)
        public_installed = True
        loaded = load_release(
            public_target,
            private_target,
            expected_public_release_sha256=public_release_sha256(public_manifest),
        )
    except BaseException:
        shutil.rmtree(public_staging, ignore_errors=True)
        shutil.rmtree(private_staging, ignore_errors=True)
        if public_installed:
            shutil.rmtree(public_target, ignore_errors=True)
        if private_installed:
            shutil.rmtree(private_target, ignore_errors=True)
        raise
    return BuiltRelease(
        public_root=loaded.public_root,
        private_root=loaded.private_root,
        public_manifest=loaded.public_manifest,
        public_manifest_sha256=loaded.public_manifest_sha256,
        private_manifest=loaded.private_manifest,
        challenge=loaded.challenge,
        episode_roots=tuple(bundle.root for bundle in loaded.bundles),
    )


def load_release(
    public_root: Path,
    private_root: Path,
    *,
    expected_public_release_sha256: str,
) -> LoadedRelease:
    """Verify an exact release inventory before the private scorer loads labels."""

    public = _resolve_directory(public_root, 'public release')
    private = _resolve_directory(private_root, 'private release')
    public_manifest_bytes = _read_regular_file(public / 'release.json', _MAX_RELEASE_MANIFEST_BYTES)
    try:
        public_manifest = PublicReleaseManifest.model_validate_json(public_manifest_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid public release manifest: {error}') from error
    if public_manifest_bytes != canonical_json_bytes(public_manifest):
        raise ReleaseIntegrityError('public release manifest must use canonical JSON')
    actual_public_sha256 = public_release_sha256(public_manifest)
    if actual_public_sha256 != expected_public_release_sha256:
        raise ReleaseIntegrityError('public release does not match its preregistered hash')
    expected_public_files = {'release.json', 'policy.json'}
    if public_manifest.contamination_policy_path is not None:
        expected_public_files.add(public_manifest.contamination_policy_path)
    _require_exact_inventory(public, expected_public_files, {'challenge'})

    policy_bytes = _read_regular_file(public / public_manifest.policy_path, _MAX_RELEASE_MANIFEST_BYTES)
    try:
        policy = RunnerPolicy.model_validate_json(policy_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid release policy: {error}') from error
    if policy_bytes != canonical_json_bytes(policy) or _sha256(policy_bytes) != public_manifest.policy_sha256:
        raise ReleaseIntegrityError('release policy does not match the public manifest')
    if (
        public_manifest.purpose == ReleasePurpose.SYNTHETIC_INTEGRATION
        and policy.required_isolation != IsolationTier.DEVELOPMENT
    ):
        raise ReleaseIntegrityError('synthetic integration releases must remain development-tier')
    if (
        public_manifest.purpose in {ReleasePurpose.RETROSPECTIVE_RESEARCH, ReleasePurpose.OFFICIAL_BENCHMARK}
        and policy.required_isolation != IsolationTier.OFFICIAL
    ):
        raise ReleaseIntegrityError('research and official releases require an official isolation policy')
    contamination_policy = _load_public_contamination_policy(public, public_manifest)
    challenge = load_challenge_bundle(public / public_manifest.challenge_path)
    if (
        challenge.manifest.challenge_id != public_manifest.challenge_id
        or challenge.manifest_sha256 != public_manifest.challenge_bundle_sha256
        or challenge.manifest.suite_manifest_sha256 != public_manifest.suite_manifest_sha256
        or challenge.manifest.admission_sha256 != public_manifest.admission_sha256
    ):
        raise ReleaseIntegrityError('challenge does not match the public release manifest')
    if challenge.admission is None:
        raise ReleaseIntegrityError('release challenge is missing its admission commitment')

    private_manifest_bytes = _read_regular_file(private / 'package.json', _MAX_RELEASE_MANIFEST_BYTES)
    if _sha256(private_manifest_bytes) != public_manifest.private_package_sha256:
        raise ReleaseIntegrityError('private package does not match the public release commitment')
    try:
        private_manifest = PrivateReleaseManifest.model_validate_json(private_manifest_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid private release manifest: {error}') from error
    if private_manifest_bytes != canonical_json_bytes(private_manifest):
        raise ReleaseIntegrityError('private release manifest must use canonical JSON')
    _validate_release_cross_bindings(public_manifest, private_manifest, challenge)

    actual_private_files, actual_private_directories = _bounded_inventory(private)
    expected_private_files = {
        'package.json',
        'split-admission.json',
        *(binding.path for binding in private_manifest.files),
    }
    expected_private_files.update(
        path
        for path in (
            private_manifest.case_universe_path,
            private_manifest.case_universe_proof_path,
            private_manifest.case_selection_audit_path,
            private_manifest.verifier_policy_path,
            private_manifest.contamination_audit_manifest_path,
        )
        if path is not None
    )
    if actual_private_files != expected_private_files:
        raise ReleaseIntegrityError('private release file allowlist mismatch')
    expected_private_directories = _directory_prefixes(expected_private_files)
    if actual_private_directories != expected_private_directories:
        raise ReleaseIntegrityError('private release directory allowlist mismatch')
    file_binding_by_path = {binding.path: binding for binding in private_manifest.files}
    for path, binding in file_binding_by_path.items():
        payload = _read_regular_file(private / path, _MAX_RELEASE_FILE_BYTES)
        if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
            raise ReleaseIntegrityError(f'private release file does not match its binding: {path}')

    split_bytes = _read_regular_file(private / private_manifest.split_admission_path, _MAX_RELEASE_MANIFEST_BYTES)
    try:
        split_admission = SplitAdmissionManifest.model_validate_json(split_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid split admission: {error}') from error
    if split_bytes != canonical_json_bytes(split_admission):
        raise ReleaseIntegrityError('split admission must use canonical JSON')
    if split_admission_manifest_sha256(split_admission) != private_manifest.split_admission_sha256:
        raise ReleaseIntegrityError('split admission does not match the release commitment')

    case_universe, case_selection_audit = _load_case_inventory(private, private_manifest, split_admission)
    contamination_audit_manifest = _load_contamination_audit_manifest(
        private,
        private_manifest,
    )

    bundles: list[EpisodeBundle] = []
    temporal_admissions: list[TemporalAdmissionEnvelope] = []
    for binding in private_manifest.episodes:
        bundle = EpisodeBundle.load(private / binding.private_path, include_private=True)
        if (
            bundle.manifest.episode_id != binding.episode_id
            or bundle.manifest_sha256 != binding.manifest_sha256
            or bundle.manifest.labels_sha256 != binding.labels_sha256
            or bundle.manifest.label_commitment_key_id != binding.label_commitment_key_id
        ):
            raise ReleaseIntegrityError('private scoring episode does not match its release binding')
        temporal_bytes = _read_regular_file(
            private / f'temporal/{binding.ordinal:06d}.json',
            _MAX_RELEASE_MANIFEST_BYTES,
        )
        try:
            temporal = TemporalAdmissionEnvelope.model_validate_json(temporal_bytes)
        except ValueError as error:
            raise ReleaseIntegrityError(f'invalid temporal admission for {binding.episode_id}: {error}') from error
        if temporal_bytes != canonical_json_bytes(temporal):
            raise ReleaseIntegrityError('temporal admission must use canonical JSON')
        if (
            release_model_sha256(temporal) != binding.temporal_admission_sha256
            or temporal.episode_id != binding.episode_id
            or temporal.manifest_sha256 != binding.manifest_sha256
            or temporal.source_tier != binding.source_tier
        ):
            raise ReleaseIntegrityError('temporal admission does not match its release episode')
        if public_manifest.purpose == ReleasePurpose.SYNTHETIC_INTEGRATION:
            _verify_tier_c_material(private, binding, bundle, temporal)
        elif public_manifest.purpose == ReleasePurpose.RETROSPECTIVE_RESEARCH:
            _validate_retrospective_release_bundle(bundle)
            _verify_tier_b_material(private, binding, bundle, temporal)
        else:
            raise ReleaseIntegrityError('official Tier A release loading is not implemented')
        bundles.append(bundle)
        temporal_admissions.append(temporal)

    if make_suite_manifest(challenge.suite.suite_id, bundles) != challenge.suite:
        raise ReleaseIntegrityError('private scoring episodes do not reconstruct the public challenge suite')
    if private_manifest.split_inventory_complete:
        validate_split_admission_subset(split_admission, bundles)
    else:
        validate_split_admission_manifest(split_admission, bundles)
    if contamination_audit_manifest is not None:
        if contamination_policy is None or case_universe is None or case_selection_audit is None:
            raise ReleaseIntegrityError('contamination audit requires its public policy and complete case inventory')
        try:
            _validate_contamination_inventory(
                case_universe=case_universe,
                case_selection_audit=case_selection_audit,
                contamination_policy=contamination_policy,
                contamination_audit_manifest=contamination_audit_manifest,
                inventory_bundles=tuple(bundles),
                require_all_admitted_payloads=False,
            )
        except ValueError as error:
            raise ReleaseIntegrityError(str(error)) from error
    return LoadedRelease(
        public_root=public,
        private_root=private,
        public_manifest=public_manifest,
        public_manifest_sha256=actual_public_sha256,
        private_manifest=private_manifest,
        challenge=challenge,
        policy=policy,
        split_admission=split_admission,
        case_universe=case_universe,
        case_selection_audit=case_selection_audit,
        contamination_policy=contamination_policy,
        contamination_audit_manifest=contamination_audit_manifest,
        temporal_admissions=tuple(temporal_admissions),
        bundles=tuple(bundles),
    )


def _validate_receipt_key_id(value: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError('receipt_key_id must be a lowercase SHA-256 digest')


def _canonical_case_universe(value: CaseUniverseManifest) -> CaseUniverseManifest:
    try:
        return CaseUniverseManifest.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f'invalid case-universe manifest: {error}') from error


def _canonical_case_selection_audit(value: CaseSelectionAudit) -> CaseSelectionAudit:
    try:
        return CaseSelectionAudit.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f'invalid case-selection audit: {error}') from error


def _canonical_contamination_policy(
    value: ContaminationAuditPolicy,
) -> ContaminationAuditPolicy:
    try:
        return ContaminationAuditPolicy.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f'invalid contamination-audit policy: {error}') from error


def _canonical_contamination_audit_manifest(
    value: ContaminationAuditManifest,
) -> ContaminationAuditManifest:
    try:
        return ContaminationAuditManifest.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f'invalid contamination-audit manifest: {error}') from error


def _validate_contamination_inventory(
    *,
    case_universe: CaseUniverseManifest,
    case_selection_audit: CaseSelectionAudit,
    contamination_policy: ContaminationAuditPolicy,
    contamination_audit_manifest: ContaminationAuditManifest,
    inventory_bundles: tuple[EpisodeBundle, ...],
    require_all_admitted_payloads: bool = True,
) -> None:
    if contamination_audit_manifest.case_universe_sha256 != case_universe_sha256(case_universe):
        raise ValueError('contamination audit does not bind the sealed case universe')
    policy_sha256 = contamination_model_sha256(contamination_policy)
    if contamination_audit_manifest.policy_sha256 != policy_sha256:
        raise ValueError('contamination audit does not bind the supplied policy')

    preeligible = {
        entry.case_id: entry
        for entry in case_universe.entries
        if entry.disposition == CaseUniverseDisposition.PREELIGIBLE
    }
    audits_by_case = {audit.audit_input.case_id: audit for audit in contamination_audit_manifest.audits}
    if set(audits_by_case) != set(preeligible):
        missing = sorted(set(preeligible) - set(audits_by_case))
        extra = sorted(set(audits_by_case) - set(preeligible))
        raise ValueError(
            'contamination audit must cover every preeligible universe case exactly once; '
            f'missing={missing}, extra={extra}'
        )
    for case_id, entry in preeligible.items():
        audit_input = audits_by_case[case_id].audit_input
        if audit_input.decision_package_sha256 != entry.decision_package_sha256:
            raise ValueError(f'contamination audit decision binding mismatch for {case_id}')

    bundle_by_episode = {bundle.manifest.episode_id: bundle for bundle in inventory_bundles}
    selection_by_case = {record.case_id: record for record in case_selection_audit.records}
    for case_id, audit in audits_by_case.items():
        selection = selection_by_case[case_id]
        if selection.disposition == CaseSelectionDisposition.ADMITTED:
            if audit.disposition != AuditDisposition.PASS:
                raise ValueError(f'admitted case {case_id} does not have a passing contamination audit')
            assert selection.episode_id is not None
            assert selection.manifest_sha256 is not None
            if (
                audit.audit_input.episode_id != selection.episode_id
                or audit.audit_input.episode_manifest_sha256 != selection.manifest_sha256
            ):
                raise ValueError(f'contamination audit episode binding mismatch for {case_id}')
            bundle = bundle_by_episode.get(selection.episode_id)
            if bundle is None:
                if require_all_admitted_payloads:
                    raise ValueError(f'no scoring bundle is available for admitted contamination audit {case_id}')
                continue
            public_payload = model_facing_payload_bytes(bundle, variant=PromptVariant.FULL)
            public_binding = audit.audit_input.public_artifact
            if public_binding.sha256 != _sha256(public_payload) or public_binding.byte_count != len(public_payload):
                raise ValueError(f'contamination audit does not bind the final model-facing view for {case_id}')
        elif selection.disposition == CaseSelectionDisposition.QUARANTINED_CONTAMINATION:
            if audit.disposition == AuditDisposition.PASS:
                raise ValueError(f'contamination-quarantined case {case_id} cannot have a passing audit')


def _load_public_contamination_policy(
    public_root: Path,
    public_manifest: PublicReleaseManifest,
) -> ContaminationAuditPolicy | None:
    if public_manifest.contamination_policy_path is None:
        return None
    assert public_manifest.contamination_policy_sha256 is not None
    policy_bytes = _read_regular_file(
        public_root / public_manifest.contamination_policy_path,
        _MAX_RELEASE_MANIFEST_BYTES,
    )
    try:
        policy = ContaminationAuditPolicy.model_validate_json(policy_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid public contamination policy: {error}') from error
    if (
        policy_bytes != canonical_json_bytes(policy)
        or contamination_model_sha256(policy) != public_manifest.contamination_policy_sha256
    ):
        raise ReleaseIntegrityError('public contamination policy does not match its commitment')
    return policy


def _load_contamination_audit_manifest(
    private_root: Path,
    private_manifest: PrivateReleaseManifest,
) -> ContaminationAuditManifest | None:
    if private_manifest.contamination_audit_manifest_path is None:
        return None
    assert private_manifest.contamination_audit_manifest_sha256 is not None
    audit_bytes = _read_regular_file(
        private_root / private_manifest.contamination_audit_manifest_path,
        _MAX_RELEASE_MANIFEST_BYTES,
    )
    try:
        audit = ContaminationAuditManifest.model_validate_json(audit_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid private contamination audit: {error}') from error
    if (
        audit_bytes != canonical_json_bytes(audit)
        or audit_manifest_sha256(audit) != private_manifest.contamination_audit_manifest_sha256
    ):
        raise ReleaseIntegrityError('private contamination audit does not match its commitment')
    return audit


def _load_case_inventory(
    private_root: Path,
    private_manifest: PrivateReleaseManifest,
    split_admission: SplitAdmissionManifest,
) -> tuple[CaseUniverseManifest | None, CaseSelectionAudit | None]:
    paths = (
        private_manifest.case_universe_path,
        private_manifest.case_universe_proof_path,
        private_manifest.case_selection_audit_path,
        private_manifest.verifier_policy_path,
    )
    if all(path is None for path in paths):
        return None, None
    if any(path is None for path in paths):
        raise ReleaseIntegrityError('private case-inventory paths must be all present or all absent')
    assert private_manifest.case_universe_path is not None
    assert private_manifest.case_universe_proof_path is not None
    assert private_manifest.case_selection_audit_path is not None
    assert private_manifest.verifier_policy_path is not None
    universe_bytes = _read_regular_file(
        private_root / private_manifest.case_universe_path,
        _MAX_RELEASE_MANIFEST_BYTES,
    )
    selection_bytes = _read_regular_file(
        private_root / private_manifest.case_selection_audit_path,
        _MAX_RELEASE_MANIFEST_BYTES,
    )
    proof_bytes = _read_regular_file(
        private_root / private_manifest.case_universe_proof_path,
        _MAX_RELEASE_FILE_BYTES,
    )
    verifier_policy = _read_regular_file(
        private_root / private_manifest.verifier_policy_path,
        _MAX_RELEASE_FILE_BYTES,
    )
    try:
        universe = CaseUniverseManifest.model_validate_json(universe_bytes)
        selection = CaseSelectionAudit.model_validate_json(selection_bytes)
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid private case inventory: {error}') from error
    if universe_bytes != canonical_json_bytes(universe) or selection_bytes != canonical_json_bytes(selection):
        raise ReleaseIntegrityError('private case inventory must use canonical JSON')
    if (
        case_universe_sha256(universe) != private_manifest.case_universe_sha256
        or case_selection_audit_sha256(selection) != private_manifest.case_selection_audit_sha256
        or _sha256(verifier_policy) != private_manifest.verifier_policy_sha256
    ):
        raise ReleaseIntegrityError('private case inventory does not match its release commitments')
    if len(proof_bytes) != universe.seal.proof_bytes or _sha256(proof_bytes) != universe.seal.proof_sha256:
        raise ReleaseIntegrityError('private case-universe proof does not match its seal')
    if selection.selection_policy_sha256 != _sha256(verifier_policy):
        raise ReleaseIntegrityError('private case-selection audit does not use the verifier policy')
    try:
        validate_case_selection_bindings(
            universe,
            selection,
            (
                (binding.episode_id, binding.manifest_sha256, binding.lineage_group_id)
                for binding in split_admission.episodes
            ),
        )
    except ValueError as error:
        raise ReleaseIntegrityError(str(error)) from error
    return universe, selection


def _load_unique_bundles(paths: tuple[Path, ...], *, label: str) -> tuple[EpisodeBundle, ...]:
    bundles = tuple(
        sorted(
            (EpisodeBundle.load(path.expanduser().resolve(), include_private=True) for path in paths),
            key=lambda bundle: bundle.manifest.episode_id,
        )
    )
    episode_ids = tuple(bundle.manifest.episode_id for bundle in bundles)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f'{label} episode directories contain duplicate episode IDs')
    return bundles


def _require_exact_episode_keys(label: str, values: Mapping[str, object], episode_ids: tuple[str, ...]) -> None:
    expected = set(episode_ids)
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'{label} must exactly cover selected episodes; missing={missing}, extra={extra}')


def _normalize_extra_private_files(episode_id: str, files: Mapping[str, bytes]) -> dict[str, bytes]:
    normalized: dict[str, bytes] = {}
    for relative_path, payload in files.items():
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or path.is_absolute()
            or '..' in path.parts
            or path.as_posix() != relative_path
            or path == PurePosixPath('.')
        ):
            raise ValueError(f'extra private path for {episode_id} must be normalized and relative: {relative_path!r}')
        supported_parts = all(
            part and all(character.isalnum() or character in '._-' for character in part) for part in path.parts
        )
        if not supported_parts:
            raise ValueError(f'extra private path for {episode_id} contains unsupported characters: {relative_path!r}')
        if not isinstance(payload, bytes):
            raise ValueError(f'extra private payload for {episode_id}/{relative_path} must be bytes')
        normalized[relative_path] = payload
    return normalized


def _validate_retrospective_release_bundle(bundle: EpisodeBundle) -> None:
    if bundle.manifest.synthetic:
        raise ValueError('retrospective research releases cannot include synthetic episodes')
    if bundle.manifest.split != Split.TEST:
        raise ValueError('retrospective research release episodes must use the test split')
    if bundle.manifest.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256:
        raise ValueError('retrospective research release episodes require HMAC-SHA256 label commitments')
    if bundle.private_labels is None or bundle.label_commitment_key is None:
        raise ValueError('retrospective research releases require private labels and their HMAC key')
    if bundle.manifest.reward_version == RANKING_REWARD_VERSION and bundle.ranking_labels is None:
        raise ValueError('V1 retrospective research episodes require private ranking labels')
    bundle.validate_integrity()


def _normalize_and_verify_tier_b_material(
    bundle: EpisodeBundle,
    material: TemporalAdmissionMaterial,
) -> TemporalAdmissionMaterial:
    try:
        admission = TemporalAdmissionEnvelope.model_validate_json(canonical_json_bytes(material.admission))
    except ValueError as error:
        raise ReleaseIntegrityError(f'invalid Tier B temporal admission: {error}') from error
    normalized = TemporalAdmissionMaterial(
        admission=admission,
        protocol_artifacts=dict(material.protocol_artifacts),
        raw_outcome_source=material.raw_outcome_source,
        label_derivation_audit=material.label_derivation_audit,
        receipt_proofs=dict(material.receipt_proofs),
    )
    _verify_tier_b_components(
        bundle,
        normalized.admission,
        protocol_artifacts=normalized.protocol_artifacts,
        raw_outcome_source=normalized.raw_outcome_source,
        label_derivation_audit=normalized.label_derivation_audit,
        receipt_proofs=normalized.receipt_proofs,
    )
    return normalized


def _verify_tier_b_components(
    bundle: EpisodeBundle,
    temporal: TemporalAdmissionEnvelope,
    *,
    protocol_artifacts: Mapping[str, bytes],
    raw_outcome_source: bytes,
    label_derivation_audit: bytes,
    receipt_proofs: Mapping[str, bytes],
) -> None:
    if temporal.source_tier != TemporalSourceTier.TIER_B:
        raise ReleaseIntegrityError('retrospective research releases require Tier B temporal admission')
    if (
        temporal.admitted_use != TemporalAdmissionUse.RETROSPECTIVE_RESEARCH
        or temporal.provenance_basis != TemporalProvenanceBasis.INDEPENDENT_ARCHIVE
    ):
        raise ReleaseIntegrityError('Tier B admission has the wrong use or provenance basis')
    if temporal.episode_id != bundle.manifest.episode_id or temporal.manifest_sha256 != bundle.manifest_sha256:
        raise ReleaseIntegrityError('Tier B admission is not bound to the scoring episode manifest')
    if (
        temporal.outcome_snapshot.labels_sha256 != bundle.manifest.labels_sha256
        or temporal.outcome_snapshot.label_commitment_scheme != bundle.manifest.label_commitment_scheme
    ):
        raise ReleaseIntegrityError('Tier B outcome snapshot does not bind the private labels')
    if set(protocol_artifacts) != set(PROTOCOL_ARTIFACT_NAMES):
        raise ReleaseIntegrityError('Tier B material requires exactly the three protocol artifacts')
    if any(not isinstance(payload, bytes) for payload in protocol_artifacts.values()):
        raise ReleaseIntegrityError('Tier B protocol artifacts must be bytes')
    protocol = temporal.decision_snapshot.protocol_commitments
    actual_protocol_hashes = {
        'candidate_set_definition': _sha256(protocol_artifacts['candidate_set_definition']),
        'evidence_acquisition_spec': _sha256(protocol_artifacts['evidence_acquisition_spec']),
        'outcome_adjudication_spec': _sha256(protocol_artifacts['outcome_adjudication_spec']),
    }
    expected_protocol_hashes = {
        'candidate_set_definition': protocol.candidate_set_definition_sha256,
        'evidence_acquisition_spec': protocol.evidence_acquisition_spec_sha256,
        'outcome_adjudication_spec': protocol.outcome_adjudication_spec_sha256,
    }
    if actual_protocol_hashes != expected_protocol_hashes:
        raise ReleaseIntegrityError('Tier B protocol artifacts do not match their commitments')
    expected_decision = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        protocol,
    )
    if expected_decision != temporal.decision_snapshot:
        raise ReleaseIntegrityError('Tier B decision snapshot does not match the scoring episode')
    outcome = temporal.outcome_snapshot
    if (
        not isinstance(raw_outcome_source, bytes)
        or len(raw_outcome_source) != outcome.raw_outcome_source_bytes
        or _sha256(raw_outcome_source) != outcome.raw_outcome_source_sha256
    ):
        raise ReleaseIntegrityError('Tier B raw outcome source does not match its commitment')
    if (
        not isinstance(label_derivation_audit, bytes)
        or len(label_derivation_audit) != outcome.label_derivation_audit_bytes
        or _sha256(label_derivation_audit) != outcome.label_derivation_audit_sha256
    ):
        raise ReleaseIntegrityError('Tier B label derivation audit does not match its commitment')
    availability: dict[tuple[str, int], datetime] = {}
    assert bundle.private_labels is not None
    for outcome_record in bundle.private_labels.outcomes:
        key = (outcome_record.target_id, outcome_record.horizon_days)
        current = availability.get(key)
        if current is None or outcome_record.revealed_at < current:
            availability[key] = outcome_record.revealed_at
    expected_availability = tuple(
        OutcomeTargetAvailability(
            target_id=target_id,
            horizon_days=horizon_days,
            first_label_available_at=availability[(target_id, horizon_days)],
        )
        for target_id, horizon_days in sorted(availability)
    )
    if outcome.target_availability != expected_availability:
        raise ReleaseIntegrityError('Tier B outcome availability does not match the private outcomes')
    receipt_by_id = {receipt.receipt_id: receipt for receipt in temporal.receipts}
    if set(receipt_proofs) != set(receipt_by_id):
        raise ReleaseIntegrityError('Tier B material requires exactly one proof per temporal receipt')
    for receipt_id, receipt in receipt_by_id.items():
        _validate_file_component(receipt_id, label='Tier B receipt ID')
        payload = receipt_proofs[receipt_id]
        if (
            not isinstance(payload, bytes)
            or len(payload) != receipt.receipt_bytes
            or _sha256(payload) != receipt.receipt_sha256
        ):
            raise ReleaseIntegrityError(f'Tier B proof does not match receipt {receipt_id}')


def _validate_file_component(value: str, *, label: str) -> None:
    if not value or value in {'.', '..'} or any(not (character.isalnum() or character in '._-') for character in value):
        raise ReleaseIntegrityError(f'{label} is not a safe file component')


def _validate_synthetic_release_bundles(bundles: tuple[EpisodeBundle, ...]) -> None:
    episode_ids = tuple(bundle.manifest.episode_id for bundle in bundles)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError('release episode IDs must be unique')
    for bundle in bundles:
        if not bundle.manifest.synthetic:
            raise ValueError('synthetic integration releases cannot include real episodes')
        if bundle.manifest.split != Split.TEST:
            raise ValueError('synthetic integration release episodes must use the test split mechanically')
        if bundle.manifest.label_commitment_scheme != LabelCommitmentScheme.HMAC_SHA256:
            raise ValueError('synthetic integration release episodes require HMAC-SHA256 label commitments')
        if bundle.private_labels is None or bundle.label_commitment_key is None:
            raise ValueError('synthetic integration release requires private labels and their HMAC key')
        if bundle.manifest.reward_version == RANKING_REWARD_VERSION and bundle.ranking_labels is None:
            raise ValueError('V1 synthetic integration episodes require private ranking labels')
        bundle.validate_integrity()


def _source_audit(bundle: EpisodeBundle) -> dict[str, object]:
    provenance = bundle.manifest.source_provenance
    if provenance is not None and provenance.adapter_id == IEDB_ADAPTER_ID:
        audit_path = bundle.root / 'private' / 'iedb_audit.json'
        audit_bytes = _read_regular_file(audit_path, _MAX_RELEASE_FILE_BYTES)
        try:
            audit = IedbPrivateAudit.model_validate_json(audit_bytes)
        except ValueError as error:
            raise ReleaseIntegrityError(f'invalid private IEDB audit: {error}') from error
        canonical_audit = canonical_json_bytes(audit)
        return {
            'audit_type': 'iedb',
            'private_audit_sha256': _sha256(canonical_audit),
            'result': audit_episode(bundle.root),
        }
    return {
        'audit_type': 'core_bundle_integrity',
        'episode_id': bundle.manifest.episode_id,
        'manifest_sha256': bundle.manifest_sha256,
    }


def _build_tier_c_material(
    bundle: EpisodeBundle,
    *,
    release_id: str,
    source_audit: dict[str, object],
) -> TemporalAdmissionMaterial:
    assert bundle.private_labels is not None
    protocol_artifacts = {
        'candidate_set_definition': canonical_json_bytes(
            {
                'claim': 'synthetic integration fixture; candidate labels may be public',
                'episode_id': bundle.manifest.episode_id,
                'candidate_ids': bundle.manifest.candidate_ids,
            }
        ),
        'evidence_acquisition_spec': canonical_json_bytes(
            {
                'claim': 'retrospective reconstruction for plumbing tests only',
                'source_provenance': (
                    bundle.manifest.source_provenance.model_dump(mode='json')
                    if bundle.manifest.source_provenance is not None
                    else None
                ),
            }
        ),
        'outcome_adjudication_spec': canonical_json_bytes(
            {
                'adjudication_version': bundle.manifest.adjudication_version,
                'forecast_targets': [target.model_dump(mode='json') for target in bundle.manifest.forecast_targets],
                'reward_version': bundle.manifest.reward_version,
            }
        ),
    }
    protocol = DecisionProtocolCommitments(
        candidate_set_available_at=bundle.manifest.decision_at,
        candidate_set_definition_sha256=_sha256(protocol_artifacts['candidate_set_definition']),
        evidence_acquisition_spec_sha256=_sha256(protocol_artifacts['evidence_acquisition_spec']),
        outcome_adjudication_spec_sha256=_sha256(protocol_artifacts['outcome_adjudication_spec']),
    )
    decision = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        protocol,
    )
    raw_outcome_source = _raw_outcome_source(bundle)
    label_derivation_audit = canonical_json_bytes(source_audit)
    availability: dict[tuple[str, int], datetime] = {}
    for outcome in bundle.private_labels.outcomes:
        key = (outcome.target_id, outcome.horizon_days)
        current = availability.get(key)
        if current is None or outcome.revealed_at < current:
            availability[key] = outcome.revealed_at
    target_availability = tuple(
        OutcomeTargetAvailability(
            target_id=target_id,
            horizon_days=horizon_days,
            first_label_available_at=availability[(target_id, horizon_days)],
        )
        for target_id, horizon_days in sorted(availability)
    )
    outcome = OutcomeSnapshotCommitment(
        episode_id=bundle.manifest.episode_id,
        labels_sha256=bundle.manifest.labels_sha256,
        label_commitment_scheme=bundle.manifest.label_commitment_scheme,
        outcome_adjudication_spec_sha256=protocol.outcome_adjudication_spec_sha256,
        raw_outcome_source_sha256=_sha256(raw_outcome_source),
        raw_outcome_source_bytes=len(raw_outcome_source),
        label_derivation_audit_sha256=_sha256(label_derivation_audit),
        label_derivation_audit_bytes=len(label_derivation_audit),
        target_availability=target_availability,
    )
    proof = canonical_json_bytes(
        {
            'authority': 'organizer-self-attestation',
            'claim': 'Tier C reconstruction only; not an external timestamp proof',
            'episode_id': bundle.manifest.episode_id,
            'release_id': release_id,
        }
    )
    receipt = TemporalArtifactReceipt(
        receipt_id='tier-c-organizer-attestation',
        role=TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL,
        artifact_schema_version=CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        artifact_sha256=decision.candidate_universe_or_panel_sha256,
        artifact_bytes=decision.candidate_universe_or_panel_bytes,
        witnessed_at=bundle.manifest.decision_at,
        authority_type=TemporalReceiptAuthority.ORGANIZER_ATTESTATION,
        authority_id='vaxreplay-synthetic-integration-builder',
        receipt_sha256=_sha256(proof),
        receipt_bytes=len(proof),
        verification_uri=f'synthetic://vaxreplay/{release_id}/{bundle.manifest.episode_id}',
    )
    admitted_at = max(target.first_label_available_at for target in target_availability) + timedelta(microseconds=1)
    admission = TemporalAdmissionEnvelope(
        admission_id=f'{release_id}:{bundle.manifest.episode_id}',
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        source_tier=TemporalSourceTier.TIER_C,
        admitted_use=TemporalAdmissionUse.TRAIN_DEBUG,
        provenance_basis=TemporalProvenanceBasis.RETROSPECTIVE_RECONSTRUCTION,
        decision_snapshot=decision,
        outcome_snapshot=outcome,
        receipts=(receipt,),
        admitted_at=admitted_at,
    )
    return TemporalAdmissionMaterial(
        admission=admission,
        protocol_artifacts=protocol_artifacts,
        raw_outcome_source=raw_outcome_source,
        label_derivation_audit=label_derivation_audit,
        receipt_proofs={receipt.receipt_id: proof},
    )


def _write_scoring_episode(
    root: Path,
    relative_root: str,
    bundle: EpisodeBundle,
    bindings: list[PrivateFileBinding],
) -> None:
    assert bundle.private_labels is not None
    assert bundle.label_commitment_key is not None
    files = {
        'manifest.json': canonical_json_bytes(bundle.manifest),
        'candidates.jsonl': jsonl_text(bundle.candidates).encode('utf-8'),
        'evidence.jsonl': jsonl_text(bundle.evidence).encode('utf-8'),
        'private/outcomes.jsonl': jsonl_text(tuple(bundle.private_labels.outcomes)).encode('utf-8'),
        'private/assessments_gold.jsonl': jsonl_text(tuple(bundle.private_labels.assessments_gold)).encode('utf-8'),
        'private/evidence_gold.jsonl': jsonl_text(tuple(bundle.private_labels.evidence_gold)).encode('utf-8'),
        'private/label_commitment_key.hex': bundle.label_commitment_key.hex().encode('ascii') + b'\n',
    }
    if bundle.ranking_labels is not None:
        files['private/ranking_labels.jsonl'] = jsonl_text(bundle.ranking_labels).encode('utf-8')
    provenance = bundle.manifest.source_provenance
    if provenance is not None and provenance.adapter_id == IEDB_ADAPTER_ID:
        audit_bytes = _read_regular_file(bundle.root / 'private' / 'iedb_audit.json', _MAX_RELEASE_FILE_BYTES)
        spec_bytes = _read_regular_file(bundle.root / 'private' / 'iedb_episode_spec.json', _MAX_RELEASE_FILE_BYTES)
        try:
            audit = IedbPrivateAudit.model_validate_json(audit_bytes)
            spec = IedbEpisodeSpec.model_validate_json(spec_bytes)
        except ValueError as error:
            raise ReleaseIntegrityError(f'invalid private IEDB provenance record: {error}') from error
        files['private/iedb_audit.json'] = canonical_json_bytes(audit)
        files['private/iedb_episode_spec.json'] = canonical_json_bytes(spec)
    for name, payload in files.items():
        _write_private_file(root, f'{relative_root}/{name}', payload, bindings)


def _verify_tier_b_material(
    private_root: Path,
    binding: ReleaseEpisodeBinding,
    bundle: EpisodeBundle,
    temporal: TemporalAdmissionEnvelope,
) -> None:
    protocol_artifacts = {
        name: _read_regular_file(
            private_root / f'protocols/{binding.ordinal:06d}/{name}.json',
            _MAX_RELEASE_FILE_BYTES,
        )
        for name in PROTOCOL_ARTIFACT_NAMES
    }
    raw_outcome_source = _read_regular_file(
        private_root / f'protocols/{binding.ordinal:06d}/raw-outcome-source.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    label_derivation_audit = _read_regular_file(
        private_root / f'protocols/{binding.ordinal:06d}/label-derivation-audit.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    source_audit = _read_regular_file(
        private_root / f'source-audits/{binding.ordinal:06d}.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    if not source_audit or _sha256(source_audit) != binding.source_audit_sha256:
        raise ReleaseIntegrityError('Tier B source audit does not match its release binding')
    receipt_proofs: dict[str, bytes] = {}
    for receipt in temporal.receipts:
        _validate_file_component(receipt.receipt_id, label='Tier B receipt ID')
        receipt_proofs[receipt.receipt_id] = _read_regular_file(
            private_root / f'proofs/{binding.ordinal:06d}/{receipt.receipt_id}.json',
            _MAX_RELEASE_FILE_BYTES,
        )
    _verify_tier_b_components(
        bundle,
        temporal,
        protocol_artifacts=protocol_artifacts,
        raw_outcome_source=raw_outcome_source,
        label_derivation_audit=label_derivation_audit,
        receipt_proofs=receipt_proofs,
    )


def _verify_tier_c_material(
    private_root: Path,
    binding: ReleaseEpisodeBinding,
    bundle: EpisodeBundle,
    temporal: TemporalAdmissionEnvelope,
) -> None:
    if temporal.source_tier != TemporalSourceTier.TIER_C:
        raise ReleaseIntegrityError('synthetic release contains a non-Tier-C temporal admission')
    if (
        temporal.admitted_use != TemporalAdmissionUse.TRAIN_DEBUG
        or temporal.provenance_basis != TemporalProvenanceBasis.RETROSPECTIVE_RECONSTRUCTION
        or temporal.outcome_snapshot.labels_sha256 != bundle.manifest.labels_sha256
        or temporal.outcome_snapshot.label_commitment_scheme != bundle.manifest.label_commitment_scheme
    ):
        raise ReleaseIntegrityError('Tier C admission overclaims its use or does not bind the private labels')
    protocol_artifacts = {
        name: _read_regular_file(
            private_root / f'protocols/{binding.ordinal:06d}/{name}.json',
            _MAX_RELEASE_FILE_BYTES,
        )
        for name in ('candidate_set_definition', 'evidence_acquisition_spec', 'outcome_adjudication_spec')
    }
    expected_protocol = temporal.decision_snapshot.protocol_commitments
    actual_hashes = (
        _sha256(protocol_artifacts['candidate_set_definition']),
        _sha256(protocol_artifacts['evidence_acquisition_spec']),
        _sha256(protocol_artifacts['outcome_adjudication_spec']),
    )
    expected_hashes = (
        expected_protocol.candidate_set_definition_sha256,
        expected_protocol.evidence_acquisition_spec_sha256,
        expected_protocol.outcome_adjudication_spec_sha256,
    )
    if actual_hashes != expected_hashes:
        raise ReleaseIntegrityError('Tier C protocol artifacts do not match their commitments')
    expected_decision = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        expected_protocol,
    )
    if expected_decision != temporal.decision_snapshot:
        raise ReleaseIntegrityError('Tier C decision snapshot does not match the scoring episode')
    raw = _read_regular_file(
        private_root / f'protocols/{binding.ordinal:06d}/raw-outcome-source.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    audit = _read_regular_file(
        private_root / f'protocols/{binding.ordinal:06d}/label-derivation-audit.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    source_audit = _read_regular_file(
        private_root / f'source-audits/{binding.ordinal:06d}.json',
        _MAX_RELEASE_FILE_BYTES,
    )
    if (
        len(raw) != temporal.outcome_snapshot.raw_outcome_source_bytes
        or _sha256(raw) != temporal.outcome_snapshot.raw_outcome_source_sha256
        or raw != _raw_outcome_source(bundle)
        or len(audit) != temporal.outcome_snapshot.label_derivation_audit_bytes
        or _sha256(audit) != temporal.outcome_snapshot.label_derivation_audit_sha256
        or source_audit != audit
        or _sha256(source_audit) != binding.source_audit_sha256
        or source_audit != canonical_json_bytes(_source_audit(bundle))
    ):
        raise ReleaseIntegrityError('Tier C outcome artifacts do not match their commitments')
    expected_availability: dict[tuple[str, int], datetime] = {}
    assert bundle.private_labels is not None
    for outcome_record in bundle.private_labels.outcomes:
        key = (outcome_record.target_id, outcome_record.horizon_days)
        current = expected_availability.get(key)
        if current is None or outcome_record.revealed_at < current:
            expected_availability[key] = outcome_record.revealed_at
    observed_availability = {
        (target.target_id, target.horizon_days): target.first_label_available_at
        for target in temporal.outcome_snapshot.target_availability
    }
    if observed_availability != expected_availability:
        raise ReleaseIntegrityError('Tier C outcome availability does not match the private outcomes')
    for receipt in temporal.receipts:
        _validate_file_component(receipt.receipt_id, label='Tier C receipt ID')
        proof = _read_regular_file(
            private_root / f'proofs/{binding.ordinal:06d}/{receipt.receipt_id}.json',
            _MAX_RELEASE_FILE_BYTES,
        )
        if len(proof) != receipt.receipt_bytes or _sha256(proof) != receipt.receipt_sha256:
            raise ReleaseIntegrityError('Tier C organizer-attestation bytes do not match the receipt')


def _validate_release_cross_bindings(
    public: PublicReleaseManifest,
    private: PrivateReleaseManifest,
    challenge: LoadedChallengeBundle,
) -> None:
    if (
        private.release_id != public.release_id
        or private.purpose != public.purpose
        or private.challenge_id != public.challenge_id
        or private.challenge_bundle_sha256 != public.challenge_bundle_sha256
        or private.suite_manifest_sha256 != public.suite_manifest_sha256
        or private.admission_sha256 != public.admission_sha256
        or private.policy_sha256 != public.policy_sha256
        or private.receipt_key_id != public.receipt_key_id
        or len(private.episodes) != public.episode_count
    ):
        raise ReleaseIntegrityError('public and private release manifests disagree')
    assert challenge.admission is not None
    if challenge.admission.release_id != public.release_id or challenge.admission.purpose != public.purpose:
        raise ReleaseIntegrityError('challenge admission is bound to a different release')
    if (
        challenge.admission.split_admission_sha256 != private.split_admission_sha256
        or challenge.admission.split_inventory_complete != private.split_inventory_complete
    ):
        raise ReleaseIntegrityError('challenge admission does not match the private split admission')
    if (
        challenge.admission.case_universe_sha256 != private.case_universe_sha256
        or challenge.admission.case_selection_audit_sha256 != private.case_selection_audit_sha256
        or challenge.admission.case_inventory_complete != private.case_inventory_complete
        or challenge.admission.verifier_policy_sha256 != private.verifier_policy_sha256
    ):
        raise ReleaseIntegrityError('challenge admission does not match the private case inventory')
    if (
        challenge.admission.contamination_policy_sha256 != public.contamination_policy_sha256
        or challenge.admission.contamination_audit_manifest_sha256 != private.contamination_audit_manifest_sha256
        or challenge.admission.contamination_inventory_complete != private.contamination_inventory_complete
    ):
        raise ReleaseIntegrityError('challenge admission does not match the contamination audit inventory')
    challenge_bindings = tuple(
        (binding.episode_id, binding.temporal_admission_sha256) for binding in challenge.admission.episodes
    )
    private_bindings = tuple((binding.episode_id, binding.temporal_admission_sha256) for binding in private.episodes)
    if challenge_bindings != private_bindings:
        raise ReleaseIntegrityError('challenge and private temporal-admission bindings disagree')
    suite_bindings = tuple(
        (binding.episode_id, binding.manifest_sha256, binding.labels_sha256) for binding in challenge.suite.episodes
    )
    private_episode_bindings = tuple(
        (binding.episode_id, binding.manifest_sha256, binding.labels_sha256) for binding in private.episodes
    )
    if suite_bindings != private_episode_bindings:
        raise ReleaseIntegrityError('private scoring episodes do not match the challenge suite order')


def _raw_outcome_source(bundle: EpisodeBundle) -> bytes:
    assert bundle.private_labels is not None
    return canonical_json_bytes(
        {
            'claim': 'synthetic derived label material, not an independently witnessed raw source',
            'outcomes': [outcome.model_dump(mode='json') for outcome in bundle.private_labels.outcomes],
            'ranking_labels': (
                [label.model_dump(mode='json') for label in bundle.ranking_labels]
                if bundle.ranking_labels is not None
                else None
            ),
        }
    )


def _write_private_file(
    root: Path,
    relative_path: str,
    payload: bytes,
    bindings: list[PrivateFileBinding],
    *,
    bind: bool = True,
) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if bind:
        bindings.append(PrivateFileBinding(path=relative_path, sha256=_sha256(payload), byte_count=len(payload)))


def _make_staging(output_dir: Path, label: str) -> tuple[Path, Path]:
    target = output_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise ValueError(f'{label} output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    return target, staging


def _resolve_directory(root: Path, label: str) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ReleaseIntegrityError(f'{label} root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ReleaseIntegrityError(f'{label} root does not exist: {resolved}')
    return resolved


def _require_exact_inventory(root: Path, expected_files: set[str], expected_directories: set[str]) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ReleaseIntegrityError('release roots cannot contain symlinks')
                if entry.is_file(follow_symlinks=False):
                    files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    directories.add(entry.name)
                else:
                    raise ReleaseIntegrityError('release roots can contain only regular files and directories')
    except OSError as error:
        raise ReleaseIntegrityError(f'cannot inventory release root: {error}') from error
    if files != expected_files or directories != expected_directories:
        raise ReleaseIntegrityError('release root allowlist mismatch')


def _bounded_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > 5:
            raise ReleaseIntegrityError('private release directory nesting is too deep')
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise ReleaseIntegrityError('private release cannot contain symlinks')
                    path = Path(entry.path)
                    if entry.is_file(follow_symlinks=False):
                        files.add(path.relative_to(root).as_posix())
                        if len(files) > _MAX_RELEASE_FILES:
                            raise ReleaseIntegrityError('private release contains too many files')
                    elif entry.is_dir(follow_symlinks=False):
                        directories.add(path.relative_to(root).as_posix())
                        if len(directories) > _MAX_RELEASE_DIRECTORIES:
                            raise ReleaseIntegrityError('private release contains too many directories')
                        stack.append((path, depth + 1))
                    else:
                        raise ReleaseIntegrityError('private release can contain only regular files')
        except OSError as error:
            raise ReleaseIntegrityError(f'cannot inventory private release: {error}') from error
    return files, directories


def _directory_prefixes(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative_path in files:
        parent = PurePosixPath(relative_path).parent
        while parent != PurePosixPath('.'):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _set_tree_permissions(root: Path, *, directory_mode: int, file_mode: int) -> None:
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ReleaseIntegrityError('release staging tree cannot contain symlinks')
        if path.is_dir():
            path.chmod(directory_mode)
        elif path.is_file():
            path.chmod(file_mode)
        else:
            raise ReleaseIntegrityError('release staging tree can contain only files and directories')
    root.chmod(directory_mode)


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReleaseIntegrityError(f'cannot open release file {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseIntegrityError(f'release artifact is not a regular file: {path.name}')
        if metadata.st_size > maximum_bytes:
            raise ReleaseIntegrityError(f'release file exceeds its size limit: {path.name}')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ReleaseIntegrityError(f'release file exceeds its size limit: {path.name}')
    except OSError as error:
        raise ReleaseIntegrityError(f'cannot read release file {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
