"""Trusted pre-outcome gate for a complete prospective challenge cohort."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseUniverseDisposition,
    CaseUniverseManifest,
    CaseUniverseSeal,
    case_universe_sha256,
)
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.operations._immutable_tree import ImmutableTreeError, immutable_root_identity
from vaxreplay.operations.plan_selection import PlanSelectionMaterialSpecProtocol
from vaxreplay.operations.promotion import (
    AdapterSpec,
    SourceVerifierSpec,
    WitnessMaterialSpec,
    load_capture_promotion,
)
from vaxreplay.operations.promotion_schema import (
    PromotionHandoffDescriptor,
    PromotionIntegrityError,
    PromotionScopePolicy,
    capture_index_sha256,
)
from vaxreplay.prospective import (
    PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
    LoadedProspectiveDecisionPackage,
    LoadedProspectiveDecisionSeal,
    ProspectiveSourceCaptureBinding,
    is_promotion_bridge_source_schema_version,
    load_prospective_decision_package,
    load_prospective_decision_seal,
    prospective_decision_context_commitment,
)
from vaxreplay.prospective_schema import (
    ProspectiveChallengeAdmission,
    ProspectiveSplitInventory,
    ProspectiveSuiteManifest,
    prospective_split_inventory_sha256,
    prospective_suite_manifest_sha256,
)
from vaxreplay.temporal_schema import TemporalReceiptAuthority, TemporalReceiptVerifier

_PROSPECTIVE_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}

type CaseUniverseSealVerifier = Callable[[CaseUniverseSeal, bytes], bool]
type SourceCaptureVerifier = Callable[[ProspectiveSourceCaptureBinding, bytes, bytes], bool]

PROMOTION_ARCHIVE_ADMISSION_POLICY_SCHEMA_VERSION = 'vaxreplay.promotion-archive-admission-policy.v0.2'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$'


class ProspectiveAdmissionError(ValueError):
    """Raised when a cohort cannot make a complete prospective Tier A claim."""


class PromotionArchivePolicyEntry(StrictModel):
    """Governance allowlist entry for one exact promotion archive."""

    promotion_id: str = Field(pattern=_SAFE_ID_PATTERN)
    source_id: str = Field(pattern=r'^promotion:[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$')
    promotion_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    capture_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    handoff_descriptor_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_precommit_sha256: str = Field(pattern=_SHA256_PATTERN)
    campaign_id: str = Field(pattern=_SAFE_ID_PATTERN)
    selection_key: str = Field(pattern=_SAFE_ID_PATTERN)
    # SHA-256 of canonical PlanSelectionPolicyBinding bytes.
    selection_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    # SHA-256 of the external registry policy artifact named by that binding.
    selection_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    immport_scientific_contract: ImmportScientificContractBinding | None = None

    @model_validator(mode='after')
    def validate_source_identity(self) -> Self:
        if self.source_id != f'promotion:{self.promotion_id}':
            raise ValueError('promotion archive source_id must derive from promotion_id')
        return self


class ImmportScientificContractBinding(StrictModel):
    """Exact ImmPort task artifacts required in addition to generic promotion identity."""

    profile: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_set_definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_adjudication_spec_sha256: str = Field(pattern=_SHA256_PATTERN)


class PromotionArchiveAdmissionPolicy(StrictModel):
    """Canonical promotion-only source policy committed by official admission.

    The hash makes the chosen archive set auditable.  The independent first-write-
    wins selection authorization must be configured before each archive's first
    covered capture slot; merely approving an archive before outcomes is too late.
    """

    schema_version: Literal['vaxreplay.promotion-archive-admission-policy.v0.2'] = (
        PROMOTION_ARCHIVE_ADMISSION_POLICY_SCHEMA_VERSION
    )
    policy_id: str = Field(pattern=_SAFE_ID_PATTERN)
    mode: Literal['promotion_archive_only'] = 'promotion_archive_only'
    generic_source_captures_permitted: Literal[False] = False
    archives: tuple[PromotionArchivePolicyEntry, ...] = Field(min_length=1)

    @field_validator('archives')
    @classmethod
    def validate_archives(
        cls,
        value: tuple[PromotionArchivePolicyEntry, ...],
    ) -> tuple[PromotionArchivePolicyEntry, ...]:
        keys = tuple(entry.source_id for entry in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('promotion archive allowlist must use sorted unique source IDs')
        promotion_ids = tuple(entry.promotion_id for entry in value)
        if len(promotion_ids) != len(set(promotion_ids)):
            raise ValueError('promotion archive allowlist cannot duplicate promotion IDs')
        manifest_digests = tuple(entry.promotion_manifest_sha256 for entry in value)
        if len(manifest_digests) != len(set(manifest_digests)):
            raise ValueError('promotion archive allowlist cannot duplicate archive digests')
        return value


@dataclass(frozen=True)
class PromotionArchiveVerificationSpec:
    """Out-of-band archive location and every trusted replay dependency."""

    promotion_root: Path
    expected_promotion_sha256: str
    expected_scope_policy: PromotionScopePolicy
    scope_precommit_witness_materials: WitnessMaterialSpec
    witness_materials: WitnessMaterialSpec
    source_verifiers: Mapping[str, SourceVerifierSpec]
    adapter: AdapterSpec
    verified_at: datetime
    expected_scope_precommit_sha256: str
    expected_campaign_id: str
    expected_selection_key: str
    expected_selection_policy_sha256: str
    expected_selection_policy_artifact_sha256: str
    expected_selection_manifest_sha256: str
    selection_materials: PlanSelectionMaterialSpecProtocol
    immport_study_universe_registry_bytes: bytes | None = None
    immport_outcome_adjudication_spec_bytes: bytes | None = None


@dataclass(frozen=True)
class PromotionArchiveAdmissionVerifier:
    """Promotion verifier whose complete configuration is revalidated before trust."""

    policy: PromotionArchiveAdmissionPolicy
    policy_bytes: bytes
    archives: Mapping[str, PromotionArchiveVerificationSpec]
    require_hermetic_execution: bool = False

    def __call__(
        self,
        binding: ProspectiveSourceCaptureBinding,
        descriptor_bytes: bytes,
        source_capture_policy: bytes,
    ) -> bool:
        """Allow direct diagnostics while retaining full fail-closed replay semantics."""

        try:
            verifier = _revalidate_archive_verifier(
                self,
                require_hermetic_execution=self.require_hermetic_execution,
            )
            _verify_promotion_archive_binding(
                binding=binding,
                descriptor_bytes=descriptor_bytes,
                source_capture_policy=source_capture_policy,
                verifier=verifier,
                package=None,
            )
        except (ProspectiveAdmissionError, PromotionIntegrityError, ValueError):
            return False
        return True


@dataclass(frozen=True)
class VerifiedProspectiveAdmission:
    """Organizer-trusted result; callers should pass ``admission`` to the public builder."""

    admission: ProspectiveChallengeAdmission
    suite: ProspectiveSuiteManifest
    split_inventory: ProspectiveSplitInventory
    case_universe: CaseUniverseManifest
    packages: tuple[LoadedProspectiveDecisionPackage, ...]
    seals: tuple[LoadedProspectiveDecisionSeal, ...]


def promotion_archive_policy_bytes(policy: PromotionArchiveAdmissionPolicy) -> bytes:
    """Return the exact canonical source-capture policy committed by admission."""

    validated = PromotionArchiveAdmissionPolicy.model_validate_json(canonical_json_bytes(policy))
    return canonical_json_bytes(validated)


def promotion_archive_policy_sha256(policy: PromotionArchiveAdmissionPolicy) -> str:
    return hashlib.sha256(promotion_archive_policy_bytes(policy)).hexdigest()


def make_promotion_archive_admission_verifier(
    *,
    policy: PromotionArchiveAdmissionPolicy,
    archives: Mapping[str, PromotionArchiveVerificationSpec],
    require_hermetic_execution: bool = False,
) -> PromotionArchiveAdmissionVerifier:
    """Pin a promotion-only allowlist to exact roots, digests, and replay dependencies.

    ``require_hermetic_execution=False`` supports provenance diagnostics and older
    research archives.  The official Tier A factory below always enables it.
    """

    try:
        validated_policy = PromotionArchiveAdmissionPolicy.model_validate_json(canonical_json_bytes(policy))
    except (TypeError, ValueError) as error:
        raise ProspectiveAdmissionError(f'invalid promotion archive admission policy: {error}') from error
    entries = {entry.source_id: entry for entry in validated_policy.archives}
    supplied = dict(archives)
    if set(supplied) != set(entries):
        missing = sorted(set(entries) - set(supplied))
        extra = sorted(set(supplied) - set(entries))
        raise ProspectiveAdmissionError(
            f'promotion archive verification map must exactly match its allowlist; missing={missing}, extra={extra}'
        )
    normalized: dict[str, PromotionArchiveVerificationSpec] = {}
    root_identities: set[tuple[int, int]] = set()
    for source_id in sorted(entries):
        spec = supplied[source_id]
        if not isinstance(spec, PromotionArchiveVerificationSpec):
            raise ProspectiveAdmissionError(f'promotion archive verification spec has the wrong type: {source_id}')
        if require_hermetic_execution and (
            not spec.source_verifiers
            or any(item.hermetic_execution is None for item in spec.source_verifiers.values())
            or not isinstance(spec.adapter, AdapterSpec)
            or spec.adapter.hermetic_execution is None
        ):
            raise ProspectiveAdmissionError(
                f'Tier A promotion archive requires hermetic source-verifier and adapter specs: {source_id}'
            )
        entry = entries[source_id]
        if spec.expected_promotion_sha256 != entry.promotion_manifest_sha256:
            raise ProspectiveAdmissionError(f'promotion archive digest differs from its allowlist: {source_id}')
        if hashlib.sha256(canonical_json_bytes(spec.expected_scope_policy)).hexdigest() != entry.scope_policy_sha256:
            raise ProspectiveAdmissionError(f'promotion scope policy differs from its allowlist: {source_id}')
        if spec.expected_scope_precommit_sha256 != entry.scope_precommit_sha256:
            raise ProspectiveAdmissionError(f'promotion scope precommit differs from its allowlist: {source_id}')
        if (
            spec.expected_campaign_id != entry.campaign_id
            or spec.expected_selection_key != entry.selection_key
            or spec.expected_selection_policy_sha256 != entry.selection_policy_sha256
            or spec.expected_selection_policy_artifact_sha256 != entry.selection_policy_artifact_sha256
            or spec.expected_selection_manifest_sha256 != entry.selection_manifest_sha256
        ):
            raise ProspectiveAdmissionError(f'promotion plan selection differs from its allowlist: {source_id}')
        try:
            selection_policy = spec.selection_materials.policy
        except AttributeError as error:
            raise ProspectiveAdmissionError(
                f'promotion plan-selection materials have the wrong type: {source_id}'
            ) from error
        if (
            selection_policy.campaign_id != entry.campaign_id
            or selection_policy.selection_key != entry.selection_key
            or hashlib.sha256(canonical_json_bytes(selection_policy)).hexdigest() != entry.selection_policy_sha256
            or selection_policy.policy_sha256 != entry.selection_policy_artifact_sha256
        ):
            raise ProspectiveAdmissionError(
                f'promotion plan-selection material policy differs from its allowlist: {source_id}'
            )
        if (
            not _is_sha256(spec.expected_promotion_sha256)
            or not _is_sha256(spec.expected_scope_precommit_sha256)
            or not _is_sha256(spec.expected_selection_policy_sha256)
            or not _is_sha256(spec.expected_selection_policy_artifact_sha256)
            or not _is_sha256(spec.expected_selection_manifest_sha256)
        ):
            raise ProspectiveAdmissionError(f'promotion archive spec uses an invalid digest: {source_id}')
        if (spec.immport_study_universe_registry_bytes is None) != (
            spec.immport_outcome_adjudication_spec_bytes is None
        ):
            raise ProspectiveAdmissionError(
                f'ImmPort scientific verification materials must be supplied together: {source_id}'
            )
        if (entry.immport_scientific_contract is None) != (spec.immport_study_universe_registry_bytes is None):
            raise ProspectiveAdmissionError(
                f'ImmPort scientific materials differ from the archive allowlist: {source_id}'
            )
        try:
            validated_root, root_identity = immutable_root_identity(Path(spec.promotion_root))
        except ImmutableTreeError as error:
            raise ProspectiveAdmissionError(f'promotion archive root is unsafe: {source_id}: {error}') from error
        if root_identity in root_identities:
            raise ProspectiveAdmissionError('promotion archive allowlist cannot reuse one root for multiple sources')
        root_identities.add(root_identity)
        normalized[source_id] = PromotionArchiveVerificationSpec(
            promotion_root=validated_root,
            expected_promotion_sha256=spec.expected_promotion_sha256,
            expected_scope_policy=spec.expected_scope_policy,
            scope_precommit_witness_materials=spec.scope_precommit_witness_materials,
            witness_materials=spec.witness_materials,
            source_verifiers=MappingProxyType(dict(spec.source_verifiers)),
            adapter=spec.adapter,
            verified_at=spec.verified_at,
            expected_scope_precommit_sha256=spec.expected_scope_precommit_sha256,
            expected_campaign_id=spec.expected_campaign_id,
            expected_selection_key=spec.expected_selection_key,
            expected_selection_policy_sha256=spec.expected_selection_policy_sha256,
            expected_selection_policy_artifact_sha256=spec.expected_selection_policy_artifact_sha256,
            expected_selection_manifest_sha256=spec.expected_selection_manifest_sha256,
            selection_materials=spec.selection_materials,
            immport_study_universe_registry_bytes=spec.immport_study_universe_registry_bytes,
            immport_outcome_adjudication_spec_bytes=spec.immport_outcome_adjudication_spec_bytes,
        )
    _validate_immport_scientific_precommitments(validated_policy, normalized)
    exact_policy_bytes = canonical_json_bytes(validated_policy)
    return PromotionArchiveAdmissionVerifier(
        policy=validated_policy,
        policy_bytes=exact_policy_bytes,
        archives=MappingProxyType(normalized),
        require_hermetic_execution=require_hermetic_execution,
    )


def _validate_immport_scientific_precommitments(
    policy: PromotionArchiveAdmissionPolicy,
    archives: Mapping[str, PromotionArchiveVerificationSpec],
) -> None:
    """Validate exact SDY selection and future-label rules before archive replay."""

    from vaxreplay.sources.immport import (
        IMMPORT_ARM_ADAPTER_ID,
        IMMPORT_ARM_ADAPTER_VERSION,
        IMMPORT_SOURCE_VERIFIER_ID,
        IMMPORT_SOURCE_VERIFIER_VERSION,
        ImmportArmAdapterPolicy,
        ImmportSourceVerifierPolicy,
        parse_immport_study_universe,
        verify_immport_study_universe_precommit,
    )
    from vaxreplay.sources.immport_outcomes import (
        ImmportProspectiveOutcomeAdjudicationSpec,
        immport_outcome_adjudication_spec_bytes,
    )

    entries = {item.source_id: item for item in policy.archives}
    grouped: dict[
        str,
        list[
            tuple[
                PromotionArchivePolicyEntry,
                PromotionArchiveVerificationSpec,
                ImmportSourceVerifierPolicy,
                ImmportArmAdapterPolicy,
            ]
        ],
    ] = {}
    for source_id, spec in archives.items():
        entry = entries[source_id]
        is_immport = spec.adapter.adapter_id == IMMPORT_ARM_ADAPTER_ID or any(
            item.verifier_id == IMMPORT_SOURCE_VERIFIER_ID for item in spec.source_verifiers.values()
        )
        binding = entry.immport_scientific_contract
        if not is_immport:
            if binding is not None:
                raise ProspectiveAdmissionError(
                    f'non-ImmPort archive cannot claim an ImmPort scientific contract: {source_id}'
                )
            continue
        if (
            spec.adapter.adapter_id != IMMPORT_ARM_ADAPTER_ID
            or spec.adapter.adapter_version != IMMPORT_ARM_ADAPTER_VERSION
            or len(spec.source_verifiers) != 1
            or any(
                item.verifier_id != IMMPORT_SOURCE_VERIFIER_ID
                or item.verifier_version != IMMPORT_SOURCE_VERIFIER_VERSION
                for item in spec.source_verifiers.values()
            )
            or binding is None
            or spec.immport_study_universe_registry_bytes is None
            or spec.immport_outcome_adjudication_spec_bytes is None
        ):
            raise ProspectiveAdmissionError(
                f'ImmPort archive lacks its exact scientific admission materials: {source_id}'
            )
        source_spec = next(iter(spec.source_verifiers.values()))
        try:
            source_policy = ImmportSourceVerifierPolicy.model_validate_json(source_spec.policy_bytes)
            adapter_policy = ImmportArmAdapterPolicy.model_validate_json(spec.adapter.policy_bytes)
            outcome_spec = ImmportProspectiveOutcomeAdjudicationSpec.model_validate_json(
                spec.immport_outcome_adjudication_spec_bytes
            )
        except ValueError as error:
            raise ProspectiveAdmissionError(
                f'invalid ImmPort scientific precommitment: {source_id}: {error}'
            ) from error
        if (
            source_spec.policy_bytes != canonical_json_bytes(source_policy)
            or spec.adapter.policy_bytes != canonical_json_bytes(adapter_policy)
            or spec.immport_outcome_adjudication_spec_bytes != immport_outcome_adjudication_spec_bytes(outcome_spec)
        ):
            raise ProspectiveAdmissionError(f'ImmPort scientific precommitments must use canonical JSON: {source_id}')
        registry = parse_immport_study_universe(spec.immport_study_universe_registry_bytes)
        registry_sha256 = hashlib.sha256(spec.immport_study_universe_registry_bytes).hexdigest()
        outcome_sha256 = hashlib.sha256(spec.immport_outcome_adjudication_spec_bytes).hexdigest()
        if (
            binding.study_universe_registry_sha256 != registry_sha256
            or binding.outcome_adjudication_spec_sha256 != outcome_sha256
            or source_policy.study_universe_registry_sha256 != registry_sha256
            or adapter_policy.study_universe_registry_sha256 != registry_sha256
            or adapter_policy.outcome_adjudication_spec_sha256 != outcome_sha256
            or outcome_spec.study_universe_registry_sha256 != registry_sha256
            or outcome_spec.adapter_policy_id != adapter_policy.policy_id
            or outcome_spec.episode_id != adapter_policy.episode_id
            or outcome_spec.decision_at != adapter_policy.decision_at
        ):
            raise ProspectiveAdmissionError(
                f'ImmPort scientific policy hashes or decision identity disagree: {source_id}'
            )
        grouped.setdefault(registry_sha256, []).append((entry, spec, source_policy, adapter_policy))

    for registry_sha256, members in grouped.items():
        first = members[0]
        registry_bytes = first[1].immport_study_universe_registry_bytes
        assert registry_bytes is not None  # narrowed above
        if any(item[1].immport_study_universe_registry_bytes != registry_bytes for item in members):
            raise ProspectiveAdmissionError(f'ImmPort archives disagree on bytes for registry digest {registry_sha256}')
        registry = parse_immport_study_universe(registry_bytes)
        campaign_ids = {item[1].expected_campaign_id for item in members}
        if campaign_ids != {registry.campaign_id}:
            raise ProspectiveAdmissionError('ImmPort study registry campaign differs from archive campaign')
        first_scheduled_for = min(
            source.scheduled_from
            for _entry, spec, _source_policy, _adapter_policy in members
            for source in spec.expected_scope_policy.sources
        )
        try:
            verify_immport_study_universe_precommit(
                registry_bytes,
                source_policies=tuple(item[2] for item in members),
                adapter_policies=tuple(item[3] for item in members),
                campaign_id=registry.campaign_id,
                first_scheduled_for=first_scheduled_for,
            )
        except ValueError as error:
            raise ProspectiveAdmissionError(f'ImmPort study-universe precommitment failed: {error}') from error


def _revalidate_archive_verifier(
    verifier: PromotionArchiveAdmissionVerifier,
    *,
    require_hermetic_execution: bool,
) -> PromotionArchiveAdmissionVerifier:
    """Rebuild a verifier from its public fields instead of trusting its constructor.

    ``PromotionArchiveAdmissionVerifier`` is deliberately a plain data object rather
    than an unforgeable capability.  In particular, a caller must not be able to bind
    admission to arbitrary ``policy_bytes`` while replay enforces a different
    ``policy`` object.  Check that cross-binding first, then rerun every factory
    invariant and use only the normalized result downstream.
    """

    try:
        canonical_policy_bytes = promotion_archive_policy_bytes(verifier.policy)
    except (TypeError, ValueError) as error:
        raise ProspectiveAdmissionError(f'invalid promotion archive verifier policy: {error}') from error
    if verifier.policy_bytes != canonical_policy_bytes:
        raise ProspectiveAdmissionError('promotion archive verifier policy_bytes differ from its canonical policy')
    if require_hermetic_execution and not verifier.require_hermetic_execution:
        raise ProspectiveAdmissionError(
            'official promotion archive admission requires signed hermetic verifier and adapter execution'
        )
    return make_promotion_archive_admission_verifier(
        policy=verifier.policy,
        archives=verifier.archives,
        require_hermetic_execution=require_hermetic_execution,
    )


def build_verified_promotion_archive_admission(
    *,
    release_id: str,
    suite_id: str,
    packages: Sequence[LoadedProspectiveDecisionPackage],
    seals: Sequence[LoadedProspectiveDecisionSeal],
    split_inventory: ProspectiveSplitInventory,
    case_universe: CaseUniverseManifest,
    case_universe_proof: bytes,
    eligibility_protocol: bytes,
    verifier_policy: bytes,
    promotion_archive_policy: PromotionArchiveAdmissionPolicy,
    promotion_archives: Mapping[str, PromotionArchiveVerificationSpec],
    attempt_policy: bytes,
    run_deadline_at: datetime,
    receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
) -> VerifiedProspectiveAdmission:
    """Official promotion-only admission path with fresh full-archive replay.

    V0 requires every allowlisted archive exactly once.  It intentionally rejects
    mixed generic sources and shared/duplicate promotion bindings until a future
    canonical policy can commit those multiplicities without callback state.
    """

    archive_verifier = make_promotion_archive_admission_verifier(
        policy=promotion_archive_policy,
        archives=promotion_archives,
        require_hermetic_execution=True,
    )
    return build_verified_prospective_admission(
        release_id=release_id,
        suite_id=suite_id,
        packages=packages,
        seals=seals,
        split_inventory=split_inventory,
        case_universe=case_universe,
        case_universe_proof=case_universe_proof,
        eligibility_protocol=eligibility_protocol,
        verifier_policy=verifier_policy,
        source_capture_policy=archive_verifier.policy_bytes,
        attempt_policy=attempt_policy,
        run_deadline_at=run_deadline_at,
        receipt_verifier=receipt_verifier,
        case_universe_seal_verifier=case_universe_seal_verifier,
        source_capture_verifier=archive_verifier,
    )


def build_verified_prospective_admission(
    *,
    release_id: str,
    suite_id: str,
    packages: Sequence[LoadedProspectiveDecisionPackage],
    seals: Sequence[LoadedProspectiveDecisionSeal],
    split_inventory: ProspectiveSplitInventory,
    case_universe: CaseUniverseManifest,
    case_universe_proof: bytes,
    eligibility_protocol: bytes,
    verifier_policy: bytes,
    source_capture_policy: bytes,
    attempt_policy: bytes,
    run_deadline_at: datetime,
    receipt_verifier: TemporalReceiptVerifier,
    case_universe_seal_verifier: CaseUniverseSealVerifier,
    source_capture_verifier: SourceCaptureVerifier,
) -> VerifiedProspectiveAdmission:
    """Reverify pre-outcome proofs and bind complete case/split inventories.

    A generic caller-supplied ``source_capture_verifier`` produces an explicitly
    non-Tier-A ``prospective_research`` admission. A
    :class:`PromotionArchiveAdmissionVerifier` may produce ``official_benchmark``
    only after this boundary rebuilds it under every factory invariant and requires
    hermetic execution.
    """

    if not eligibility_protocol or not verifier_policy or not source_capture_policy or not attempt_policy:
        raise ProspectiveAdmissionError('eligibility, verifier, source-capture, and attempt policies cannot be empty')
    supplied_archive_verifier = (
        source_capture_verifier if isinstance(source_capture_verifier, PromotionArchiveAdmissionVerifier) else None
    )
    archive_verifier = (
        _revalidate_archive_verifier(
            supplied_archive_verifier,
            require_hermetic_execution=True,
        )
        if supplied_archive_verifier is not None
        else None
    )
    if archive_verifier is not None and source_capture_policy != archive_verifier.policy_bytes:
        raise ProspectiveAdmissionError(
            'promotion archive admission requires its exact canonical allowlist as source_capture_policy'
        )
    package_inputs = tuple(packages)
    seal_inputs = tuple(seals)
    if not package_inputs:
        raise ProspectiveAdmissionError('prospective admission requires at least one decision package')
    package_by_id: dict[str, LoadedProspectiveDecisionPackage] = {}
    verified_promotion_sources: set[str] = set()
    for supplied in package_inputs:
        package = load_prospective_decision_package(supplied.root)
        if package != supplied:
            raise ProspectiveAdmissionError('prospective package changed after it was loaded')
        episode_id = package.manifest.episode.episode_id
        if episode_id in package_by_id:
            raise ProspectiveAdmissionError('prospective package episode IDs must be unique')
        for binding in package.manifest.source_captures:
            manifest_bytes = package.source_capture_artifacts[binding.source_id]
            promoted = _is_promoted_source(binding, manifest_bytes)
            if archive_verifier is not None:
                if not promoted:
                    raise ProspectiveAdmissionError(
                        'promotion archive admission is promotion-only and cannot mix generic source captures'
                    )
                if binding.source_id in verified_promotion_sources:
                    raise ProspectiveAdmissionError(
                        f'duplicate promoted source binding in admission cohort: {binding.source_id!r}'
                    )
                _verify_promotion_archive_binding(
                    binding=binding,
                    descriptor_bytes=manifest_bytes,
                    source_capture_policy=source_capture_policy,
                    verifier=archive_verifier,
                    package=package,
                )
                verified_promotion_sources.add(binding.source_id)
                continue
            if promoted:
                raise ProspectiveAdmissionError(
                    'promotion-backed source captures require the official promotion archive admission factory'
                )
            try:
                source_verified = source_capture_verifier(binding, manifest_bytes, source_capture_policy)
            except Exception as error:
                raise ProspectiveAdmissionError(
                    f'source-capture verifier failed for {binding.source_id!r}: {error}'
                ) from error
            if source_verified is not True:
                raise ProspectiveAdmissionError(
                    f'source-capture verifier rejected prospective research eligibility for {binding.source_id!r}'
                )
        package_by_id[episode_id] = package
    if archive_verifier is not None:
        expected_promotion_sources = {entry.source_id for entry in archive_verifier.policy.archives}
        if verified_promotion_sources != expected_promotion_sources:
            missing = sorted(expected_promotion_sources - verified_promotion_sources)
            extra = sorted(verified_promotion_sources - expected_promotion_sources)
            raise ProspectiveAdmissionError(
                f'promotion archive admission must use every allowlisted source exactly once; '
                f'missing={missing}, extra={extra}'
            )
    seal_by_id = {seal.manifest.episode_id: seal for seal in seal_inputs}
    if len(seal_by_id) != len(seal_inputs) or seal_by_id.keys() != package_by_id.keys():
        raise ProspectiveAdmissionError('every prospective package requires exactly one decision seal')

    ordered_packages = tuple(package_by_id[episode_id] for episode_id in sorted(package_by_id))
    ordered_seals: list[LoadedProspectiveDecisionSeal] = []
    for package in ordered_packages:
        supplied = seal_by_id[package.manifest.episode.episode_id]
        try:
            verified = load_prospective_decision_seal(
                supplied.root,
                package=package,
                receipt_verifier=receipt_verifier,
            )
        except ValueError as error:
            raise ProspectiveAdmissionError(f'decision seal verification failed: {error}') from error
        if verified != supplied:
            raise ProspectiveAdmissionError('prospective decision seal changed after it was loaded')
        decision_context_bytes = canonical_json_bytes(prospective_decision_context_commitment(package.manifest))
        decision_context_sha256 = hashlib.sha256(decision_context_bytes).hexdigest()
        decision_receipt = verified.manifest.receipts[2]
        if (
            package.manifest.episode.decision_context_sha256 != decision_context_sha256
            or package.manifest.episode.decision_context_bytes != len(decision_context_bytes)
            or verified.manifest.decision_context_sha256 != decision_context_sha256
            or decision_receipt.artifact_schema_version != PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION
            or decision_receipt.artifact_sha256 != decision_context_sha256
            or decision_receipt.artifact_bytes != len(decision_context_bytes)
        ):
            raise ProspectiveAdmissionError(
                'prospective decision seal does not bind the package source-capture lineage'
            )
        ordered_seals.append(verified)

    first = ordered_packages[0].manifest.episode
    try:
        suite = ProspectiveSuiteManifest(
            suite_id=suite_id,
            task_type=first.task_type,
            reward_version=first.reward_version,
            split=Split.TEST,
            episodes=tuple(package.manifest.episode for package in ordered_packages),
        )
    except ValueError as error:
        raise ProspectiveAdmissionError(f'invalid prospective suite: {error}') from error

    split_by_id = {episode.episode_id: episode for episode in split_inventory.episodes}
    for episode in suite.episodes:
        if split_by_id.get(episode.episode_id) != episode:
            raise ProspectiveAdmissionError('split inventory omits or changes a prospective suite episode')

    expected_protocol_hash = hashlib.sha256(eligibility_protocol).hexdigest()
    if case_universe.eligibility_protocol_sha256 != expected_protocol_hash:
        raise ProspectiveAdmissionError('case universe does not bind the supplied eligibility protocol')
    proof = case_universe_proof
    seal = case_universe.seal
    if len(proof) != seal.proof_bytes or hashlib.sha256(proof).hexdigest() != seal.proof_sha256:
        raise ProspectiveAdmissionError('case-universe proof bytes do not match its seal')
    if seal.authority_type not in _PROSPECTIVE_AUTHORITIES:
        raise ProspectiveAdmissionError('Tier A case universes require a prospective timestamp authority')
    earliest_decision_at = min(episode.decision_at for episode in suite.episodes)
    latest_source_witnessed_at = max(
        source_capture.witnessed_at
        for package in ordered_packages
        for source_capture in package.manifest.source_captures
    )
    if seal.witnessed_at > earliest_decision_at:
        raise ProspectiveAdmissionError('case universe must be witnessed by the earliest decision cutoff')
    if seal.witnessed_at < latest_source_witnessed_at:
        raise ProspectiveAdmissionError('case-universe seal cannot predate its source-capture witnesses')
    try:
        verified_case_universe = case_universe_seal_verifier(seal, proof)
    except Exception as error:
        raise ProspectiveAdmissionError(f'case-universe verifier failed: {error}') from error
    if not verified_case_universe:
        raise ProspectiveAdmissionError('case-universe verifier rejected the external proof')

    preeligible = tuple(
        entry for entry in case_universe.entries if entry.disposition == CaseUniverseDisposition.PREELIGIBLE
    )
    expected_package_bindings = {
        (package.manifest_sha256, package.manifest.episode.lineage_group_id) for package in ordered_packages
    }
    actual_package_bindings = {(entry.decision_package_sha256, entry.lineage_group_id) for entry in preeligible}
    if actual_package_bindings != expected_package_bindings or len(preeligible) != len(ordered_packages):
        raise ProspectiveAdmissionError(
            'preeligible case universe must exactly cover every decision package and lineage'
        )

    try:
        admission = ProspectiveChallengeAdmission(
            release_id=release_id,
            purpose=(
                'official_benchmark'
                if archive_verifier is not None and archive_verifier.require_hermetic_execution
                else 'prospective_research'
            ),
            suite_sha256=prospective_suite_manifest_sha256(suite),
            split_inventory_sha256=prospective_split_inventory_sha256(split_inventory),
            case_universe_sha256=case_universe_sha256(case_universe),
            verifier_policy_sha256=hashlib.sha256(verifier_policy).hexdigest(),
            source_capture_policy_sha256=hashlib.sha256(source_capture_policy).hexdigest(),
            eligibility_protocol_sha256=expected_protocol_hash,
            attempt_policy_sha256=hashlib.sha256(attempt_policy).hexdigest(),
            run_deadline_at=run_deadline_at,
            episodes=suite.episodes,
        )
    except ValueError as error:
        raise ProspectiveAdmissionError(f'invalid prospective challenge admission: {error}') from error
    return VerifiedProspectiveAdmission(
        admission=admission,
        suite=suite,
        split_inventory=split_inventory,
        case_universe=case_universe,
        packages=ordered_packages,
        seals=tuple(ordered_seals),
    )


def _is_promoted_source(binding: ProspectiveSourceCaptureBinding, payload: bytes) -> bool:
    if binding.source_id.startswith('promotion:'):
        return True
    try:
        envelope = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(envelope, dict) and is_promotion_bridge_source_schema_version(envelope.get('schema_version'))


def _verify_promotion_archive_binding(
    *,
    binding: ProspectiveSourceCaptureBinding,
    descriptor_bytes: bytes,
    source_capture_policy: bytes,
    verifier: PromotionArchiveAdmissionVerifier,
    package: LoadedProspectiveDecisionPackage | None,
) -> None:
    if source_capture_policy != verifier.policy_bytes:
        raise ProspectiveAdmissionError('promotion source was checked under a different archive allowlist')
    spec = verifier.archives.get(binding.source_id)
    entry = next(
        (candidate for candidate in verifier.policy.archives if candidate.source_id == binding.source_id),
        None,
    )
    if spec is None or entry is None:
        raise ProspectiveAdmissionError(f'unregistered promoted source: {binding.source_id!r}')
    try:
        descriptor = PromotionHandoffDescriptor.model_validate_json(descriptor_bytes)
    except ValueError as error:
        raise ProspectiveAdmissionError(
            f'promoted source must use the canonical handoff descriptor: {binding.source_id!r}'
        ) from error
    if descriptor_bytes != canonical_json_bytes(descriptor):
        raise ProspectiveAdmissionError('promotion handoff descriptor is not canonical JSON')
    if binding.source_id != f'promotion:{descriptor.promotion_id}' or descriptor.promotion_id != entry.promotion_id:
        raise ProspectiveAdmissionError('promotion handoff identity differs from its allowlist')
    if descriptor.promotion_manifest_sha256 != entry.promotion_manifest_sha256:
        raise ProspectiveAdmissionError('promotion handoff archive digest differs from its allowlist')
    if hashlib.sha256(descriptor_bytes).hexdigest() != entry.handoff_descriptor_sha256:
        raise ProspectiveAdmissionError('promotion handoff descriptor digest differs from its allowlist')
    if capture_index_sha256(descriptor.capture_index) != entry.capture_index_sha256:
        raise ProspectiveAdmissionError('promotion handoff capture index differs from its allowlist')
    if descriptor.capture_index.scope_policy.sha256 != entry.scope_policy_sha256:
        raise ProspectiveAdmissionError('promotion handoff scope policy differs from its allowlist')
    if descriptor.capture_index.scope_precommit.archive_sha256 != entry.scope_precommit_sha256:
        raise ProspectiveAdmissionError('promotion handoff scope precommit differs from its allowlist')
    if (
        descriptor.campaign_id != entry.campaign_id
        or descriptor.selection_key != entry.selection_key
        or descriptor.selection_policy_sha256 != entry.selection_policy_sha256
        or descriptor.selection_policy_artifact_sha256 != entry.selection_policy_artifact_sha256
        or descriptor.selection_manifest_sha256 != entry.selection_manifest_sha256
    ):
        raise ProspectiveAdmissionError('promotion handoff plan selection differs from its allowlist')
    if spec.expected_promotion_sha256 != entry.promotion_manifest_sha256:
        raise ProspectiveAdmissionError('promotion verification spec archive digest differs from its allowlist')
    if hashlib.sha256(canonical_json_bytes(spec.expected_scope_policy)).hexdigest() != entry.scope_policy_sha256:
        raise ProspectiveAdmissionError('promotion verification scope policy differs from its allowlist')
    if spec.expected_scope_precommit_sha256 != entry.scope_precommit_sha256:
        raise ProspectiveAdmissionError('promotion verification scope precommit differs from its allowlist')
    if (
        spec.expected_campaign_id != entry.campaign_id
        or spec.expected_selection_key != entry.selection_key
        or spec.expected_selection_policy_sha256 != entry.selection_policy_sha256
        or spec.expected_selection_policy_artifact_sha256 != entry.selection_policy_artifact_sha256
        or spec.expected_selection_manifest_sha256 != entry.selection_manifest_sha256
    ):
        raise ProspectiveAdmissionError('promotion verification plan selection differs from its allowlist')
    try:
        loaded = load_capture_promotion(
            spec.promotion_root,
            expected_scope_policy=spec.expected_scope_policy,
            scope_precommit_witness_materials=spec.scope_precommit_witness_materials,
            witness_materials=spec.witness_materials,
            source_verifiers=spec.source_verifiers,
            adapter=spec.adapter,
            verified_at=spec.verified_at,
            expected_scope_precommit_sha256=spec.expected_scope_precommit_sha256,
            expected_promotion_sha256=spec.expected_promotion_sha256,
            selection_materials=spec.selection_materials,
            expected_selection_manifest_sha256=spec.expected_selection_manifest_sha256,
        )
    except Exception as error:
        raise ProspectiveAdmissionError(
            f'full promotion archive reverification failed for {binding.source_id!r}: {error}'
        ) from error
    if loaded.root != spec.promotion_root:
        raise ProspectiveAdmissionError('promotion loader resolved a different allowlisted archive root')
    if loaded.manifest_sha256 != entry.promotion_manifest_sha256:
        raise ProspectiveAdmissionError('reverified promotion manifest differs from its allowlist')
    if verifier.require_hermetic_execution and (
        not loaded.index.hermetic_executions or len(loaded.index.hermetic_executions) != len(spec.source_verifiers) + 2
    ):
        raise ProspectiveAdmissionError('reverified Tier A promotion omits complete hermetic execution evidence')
    if loaded.index_bytes != canonical_json_bytes(loaded.index):
        raise ProspectiveAdmissionError('reverified promotion capture index is not canonical JSON')
    if hashlib.sha256(loaded.index_bytes).hexdigest() != entry.capture_index_sha256:
        raise ProspectiveAdmissionError('reverified promotion capture index differs from its allowlist')
    if descriptor.capture_index != loaded.index or canonical_json_bytes(descriptor.capture_index) != loaded.index_bytes:
        raise ProspectiveAdmissionError('package handoff does not contain the reverified canonical capture index')
    if (
        descriptor != loaded.handoff_descriptor
        or descriptor_bytes != loaded.handoff_descriptor_bytes
        or hashlib.sha256(loaded.handoff_descriptor_bytes).hexdigest() != entry.handoff_descriptor_sha256
    ):
        raise ProspectiveAdmissionError('package handoff differs from the freshly reverified promotion archive')
    _verify_immport_scientific_decision(
        entry=entry,
        spec=spec,
        loaded=loaded,
        package=package,
    )
    if len(loaded.source_captures) != 1:
        raise ProspectiveAdmissionError('promotion archive must derive exactly one aggregate source handoff')
    loaded_source = loaded.source_captures[0]
    if (
        loaded_source.source_id != binding.source_id
        or loaded_source.manifest_bytes != descriptor_bytes
        or binding.source_release_at != descriptor.maximum_source_release_at
        or binding.captured_at != descriptor.maximum_captured_at
        or binding.witnessed_at != descriptor.witnessed_at
        or loaded_source.source_release_at != binding.source_release_at
        or loaded_source.captured_at != binding.captured_at
        or loaded_source.witnessed_at != binding.witnessed_at
        or binding.file.sha256 != hashlib.sha256(descriptor_bytes).hexdigest()
        or binding.file.byte_count != len(descriptor_bytes)
    ):
        raise ProspectiveAdmissionError('promotion handoff source identity, bytes, or derived times differ')
    if package is not None:
        candidate_output = descriptor.candidate_output.file
        evidence_output = descriptor.evidence_output.file
        if (
            package.manifest.candidates.sha256 != candidate_output.sha256
            or package.manifest.candidates.byte_count != candidate_output.byte_count
            or package.manifest.evidence.sha256 != evidence_output.sha256
            or package.manifest.evidence.byte_count != evidence_output.byte_count
            or package.candidates != loaded.candidates
            or package.evidence != loaded.evidence
            or package.source_capture_artifacts.get(binding.source_id) != descriptor_bytes
            or package.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
            != descriptor.promotion_created_at
        ):
            raise ProspectiveAdmissionError(
                'prospective decision inputs do not exactly reproduce the reverified promotion outputs'
            )


def _verify_immport_scientific_decision(
    *,
    entry: PromotionArchivePolicyEntry,
    spec: PromotionArchiveVerificationSpec,
    loaded: object,
    package: LoadedProspectiveDecisionPackage | None,
) -> None:
    """Bind ImmPort promotion output to the exact arm-level decision semantics."""

    from vaxreplay.operations.promotion import LoadedCapturePromotion
    from vaxreplay.sources.immport import (
        IMMPORT_ARM_ADAPTER_ID,
        IMMPORT_ARM_ADAPTER_VERSION,
        ImmportArmCandidateMap,
        ImmportArmCandidateSetDefinition,
    )
    from vaxreplay.sources.immport_outcomes import (
        ImmportProspectiveOutcomeAdjudicationSpec,
    )

    binding = entry.immport_scientific_contract
    is_immport = spec.adapter.adapter_id == IMMPORT_ARM_ADAPTER_ID
    if not is_immport:
        if binding is not None:
            raise ProspectiveAdmissionError('non-ImmPort promotion carries an ImmPort contract')
        return
    if not isinstance(loaded, LoadedCapturePromotion):
        raise ProspectiveAdmissionError('reverified ImmPort promotion has the wrong result type')
    if (
        binding is None
        or spec.adapter.adapter_version != IMMPORT_ARM_ADAPTER_VERSION
        or spec.immport_outcome_adjudication_spec_bytes is None
        or set(loaded.auxiliary_outputs) != {'immport-arm-candidate-map', 'immport-candidate-set-definition'}
    ):
        raise ProspectiveAdmissionError('ImmPort promotion lacks its exact scientific auxiliary inventory')
    map_bytes = loaded.auxiliary_outputs['immport-arm-candidate-map']
    definition_bytes = loaded.auxiliary_outputs['immport-candidate-set-definition']
    try:
        candidate_map = ImmportArmCandidateMap.model_validate_json(map_bytes)
        definition = ImmportArmCandidateSetDefinition.model_validate_json(definition_bytes)
        outcome_spec = ImmportProspectiveOutcomeAdjudicationSpec.model_validate_json(
            spec.immport_outcome_adjudication_spec_bytes
        )
    except ValueError as error:
        raise ProspectiveAdmissionError(f'invalid ImmPort scientific artifact: {error}') from error
    if (
        map_bytes != canonical_json_bytes(candidate_map)
        or definition_bytes != canonical_json_bytes(definition)
        or hashlib.sha256(definition_bytes).hexdigest() != binding.candidate_set_definition_sha256
        or definition.organizer_candidate_map_sha256 != hashlib.sha256(map_bytes).hexdigest()
        or definition.outcome_adjudication_spec_sha256
        != hashlib.sha256(spec.immport_outcome_adjudication_spec_bytes).hexdigest()
        or definition.study_universe_registry_sha256 != binding.study_universe_registry_sha256
        or candidate_map.study_universe_registry_sha256 != binding.study_universe_registry_sha256
    ):
        raise ProspectiveAdmissionError('ImmPort scientific artifacts differ from their allowlist')
    record_by_id = {item.candidate_id: item for item in loaded.candidates}
    map_by_id = {item.candidate_id: item for item in candidate_map.candidates}
    if len(record_by_id) != len(loaded.candidates) or set(record_by_id) != set(map_by_id):
        raise ProspectiveAdmissionError('ImmPort candidate map must exactly cover normalized arm records')
    expected_interventions = tuple(
        sorted(
            candidate_id
            for candidate_id, item in map_by_id.items()
            if item.decision_disposition == 'rankable_intervention_arm'
        )
    )
    expected_controls = tuple(
        sorted(
            candidate_id
            for candidate_id, item in map_by_id.items()
            if item.decision_disposition == 'contextual_control_not_ranked'
        )
    )
    expected_blocked = tuple(
        sorted(
            candidate_id
            for candidate_id, item in map_by_id.items()
            if item.decision_disposition == 'blocked_unclassified_arm_type'
        )
    )
    if (
        definition.intervention_candidate_ids != expected_interventions
        or definition.contextual_control_ids != expected_controls
        or definition.blocked_unclassified_ids != expected_blocked
        or expected_blocked
        or len(expected_interventions) < 2
        or any(record_by_id[item].eligible is not True for item in expected_interventions)
        or any(record_by_id[item].eligible is not False for item in (*expected_controls, *expected_blocked))
        or any(item.vaccine_construct_mapping != 'unverified_not_claimed' for item in candidate_map.candidates)
    ):
        raise ProspectiveAdmissionError('ImmPort arm roles are incomplete, inconsistent, or insufficient for ranking')
    if package is None:
        return
    config = package.manifest.episode.decision_snapshot.config
    expected_targets = tuple((item.target_id, item.horizon_days) for item in outcome_spec.targets)
    actual_targets = tuple((item.target_id, item.horizon_days) for item in config.forecast_targets)
    if (
        config.task_type != 'early_clinical_arm_prioritization'
        or config.episode_id != definition.episode_id
        or config.decision_at != outcome_spec.decision_at
        or tuple(config.candidate_ids) != expected_interventions
        or actual_targets != expected_targets
        or package.protocol_artifacts.get('candidate_set_definition') != definition_bytes
        or package.protocol_artifacts.get('outcome_adjudication_spec') != spec.immport_outcome_adjudication_spec_bytes
    ):
        raise ProspectiveAdmissionError(
            'prospective decision does not use the exact ImmPort arm task, candidates, or outcomes'
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def prospective_admission_material_sha256(value: object) -> str:
    """Convenience hash for canonical policy artifacts represented as strict models or JSON values."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
