"""Build and verify label-free public challenges from prospectively sealed decisions.

The ordinary runner challenge is created from a finalized :class:`EpisodeBundle`, whose identity
includes private label commitments.  A Tier A run has to happen earlier, so this module uses the
decision snapshot as the stable episode identity and accepts only ``ProspectiveSubmission``
responses.  The resulting artifact contains the exact public decision packages, independently
timestamped seal sidecars, and rendered messages; it never needs a finalized episode manifest.

``build_prospective_challenge_bundle`` accepts ``LoadedProspectiveDecisionSeal`` objects that have
already passed an organizer-supplied receipt verifier.  ``load_prospective_challenge_bundle``
always checks all canonical bytes, hashes, receipt-to-artifact bindings, and the exact file
allowlist.  It reports external authority proofs as reverified only when the caller supplies a
``receipt_verifier`` callback; merely loading receipt bytes is not cryptographic verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import (
    ANTIGEN_TARGET_PRIORITIZATION_TASK,
    EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
    PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK,
    RANKING_REWARD_VERSION,
    CandidateRecord,
    EvidenceRecord,
    Split,
    StrictModel,
)
from vaxreplay.prompt import PromptVariant
from vaxreplay.prospective import (
    LoadedProspectiveDecisionPackage,
    LoadedProspectiveDecisionSeal,
    ProspectiveDecisionPackageManifest,
    ProspectiveDecisionSealManifest,
    ProspectiveIntegrityError,
    load_prospective_decision_package,
    load_prospective_decision_seal,
    prospective_decision_seal_sha256,
)
from vaxreplay.prospective_schema import (
    PROSPECTIVE_RESPONSE_PROTOCOL,
    PROSPECTIVE_SUBMISSION_SCHEMA_VERSION,
    ProspectiveChallengeAdmission,
    ProspectiveEpisodeBinding,
    ProspectiveSuiteManifest,
    prospective_challenge_admission_sha256,
    prospective_suite_manifest_sha256,
)
from vaxreplay.runner.schema import ChatMessage
from vaxreplay.temporal_schema import TemporalReceiptVerifier

PROSPECTIVE_CHALLENGE_ENVELOPE_SCHEMA_VERSION = 'vaxreplay.prospective-challenge-envelope.v0.1'
PROSPECTIVE_CHALLENGE_BUNDLE_SCHEMA_VERSION = 'vaxreplay.prospective-challenge-bundle.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_ENVELOPE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_ENVELOPE_BYTES = 512 * 1024 * 1024
_MAX_PROOF_BYTES = 512 * 1024 * 1024

_SYSTEM_PROMPT = """You are participating in a prospective, closed-book VaxReplay benchmark.
Use only the decision-time candidates and evidence in the user message. Do not use external
knowledge, inferred future events, or sources that are not shown. Return exactly one JSON object
and no surrounding prose."""


class ProspectiveChallengeIntegrityError(ValueError):
    """Raised when a prospective public challenge is incomplete or has been altered."""


class ProspectiveChallengeEnvelope(StrictModel):
    """The complete label-free input delivered to one fresh worker over stdin."""

    schema_version: Literal['vaxreplay.prospective-challenge-envelope.v0.1'] = (
        PROSPECTIVE_CHALLENGE_ENVELOPE_SCHEMA_VERSION
    )
    challenge_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    prospective_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    ordinal: int = Field(ge=0)
    sample_index: int = Field(default=0, ge=0)
    prompt_variant: PromptVariant = PromptVariant.FULL
    binding: ProspectiveEpisodeBinding
    decision_package_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision_seal_sha256: str = Field(pattern=_SHA256_PATTERN)
    messages: tuple[ChatMessage, ChatMessage]
    response_protocol: Literal['vaxreplay.prospective-submission-json-stdout.v0.1'] = PROSPECTIVE_RESPONSE_PROTOCOL

    @field_validator('messages')
    @classmethod
    def validate_messages(
        cls,
        value: tuple[ChatMessage, ChatMessage],
    ) -> tuple[ChatMessage, ChatMessage]:
        if tuple(message.role for message in value) != ('system', 'user'):
            raise ValueError('prospective challenge messages require one system then one user message')
        return value


class ProspectiveChallengeEpisodeFile(StrictModel):
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    envelope_path: str = Field(pattern=r'^episodes/[0-9]{6}\.json$')
    envelope_sha256: str = Field(pattern=_SHA256_PATTERN)
    package_root: str = Field(pattern=r'^packages/[0-9]{6}$')
    decision_package_sha256: str = Field(pattern=_SHA256_PATTERN)
    seal_root: str = Field(pattern=r'^seals/[0-9]{6}$')
    decision_seal_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        ordinal = f'{self.ordinal:06d}'
        if self.envelope_path != f'episodes/{ordinal}.json':
            raise ValueError('prospective envelope path must match its ordinal')
        if self.package_root != f'packages/{ordinal}':
            raise ValueError('prospective package path must match its ordinal')
        if self.seal_root != f'seals/{ordinal}':
            raise ValueError('prospective seal path must match its ordinal')
        return self


class ProspectiveChallengeBundleManifest(StrictModel):
    """Canonical allowlist roots and hashes for a pre-outcome challenge."""

    schema_version: Literal['vaxreplay.prospective-challenge-bundle.v0.1'] = PROSPECTIVE_CHALLENGE_BUNDLE_SCHEMA_VERSION
    challenge_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    suite_path: Literal['suite.json'] = 'suite.json'
    prospective_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    admission_path: Literal['admission.json'] = 'admission.json'
    prospective_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_variant: PromptVariant = PromptVariant.FULL
    episodes: tuple[ProspectiveChallengeEpisodeFile, ...] = Field(min_length=1, max_length=4_096)

    @field_validator('episodes')
    @classmethod
    def validate_episodes(
        cls,
        value: tuple[ProspectiveChallengeEpisodeFile, ...],
    ) -> tuple[ProspectiveChallengeEpisodeFile, ...]:
        ordinals = tuple(binding.ordinal for binding in value)
        if ordinals != tuple(range(len(value))):
            raise ValueError('prospective challenge ordinals must be contiguous and start at zero')
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('prospective challenge episode IDs must be unique')
        return value


@dataclass(frozen=True)
class LoadedProspectiveChallengeBundle:
    root: Path
    manifest: ProspectiveChallengeBundleManifest
    suite: ProspectiveSuiteManifest
    admission: ProspectiveChallengeAdmission
    envelopes: tuple[ProspectiveChallengeEnvelope, ...]
    packages: tuple[LoadedProspectiveDecisionPackage, ...]
    seals: tuple[LoadedProspectiveDecisionSeal, ...]
    manifest_sha256: str
    authority_proofs_reverified: bool


def prospective_challenge_envelope_sha256(envelope: ProspectiveChallengeEnvelope) -> str:
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def prospective_challenge_bundle_sha256(manifest: ProspectiveChallengeBundleManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def build_prospective_episode_prompt(
    package: LoadedProspectiveDecisionPackage,
    *,
    variant: PromptVariant = PromptVariant.FULL,
) -> str:
    """Render one decision package without introducing a post-outcome identity."""

    binding = package.manifest.episode
    config = binding.decision_snapshot.config
    if variant == PromptVariant.NO_EVIDENCE:
        rendered_evidence: list[dict[str, object]] = []
    else:
        rendered_evidence = [
            {
                'evidence_id': evidence.evidence_id,
                'source_type': evidence.source_type,
                'available_at': evidence.available_at.isoformat(),
                'title': (
                    f'Historical source {index + 1}'
                    if variant == PromptVariant.BIBLIOGRAPHICALLY_SCRUBBED
                    else evidence.title
                ),
                'body': evidence.body,
                'related_candidate_ids': evidence.related_candidate_ids,
            }
            for index, evidence in enumerate(package.evidence)
        ]
    episode: dict[str, object] = {
        'episode_id': config.episode_id,
        'decision_at': config.decision_at.isoformat(),
        'decision_snapshot_sha256': binding.decision_snapshot_sha256,
        'task_type': config.task_type,
        'reward_version': config.reward_version,
        'portfolio_size': config.portfolio_size,
        'required_dimensions': config.required_dimensions,
        'forecast_targets': [target.model_dump(mode='json') for target in config.forecast_targets],
        'candidate_ids': [candidate.candidate_id for candidate in package.candidates if candidate.eligible],
        'evidence': rendered_evidence,
    }
    if variant != PromptVariant.FULL:
        episode['prompt_variant'] = variant.value
    if config.reward_version == RANKING_REWARD_VERSION:
        episode['ranking_objective'] = {
            'ndcg_at_portfolio_size': 0.50,
            'strict_pairwise_concordance': 0.25,
            'normalized_top_k_set_utility': 0.25,
        }
    output_contract = {
        'schema_version': PROSPECTIVE_SUBMISSION_SCHEMA_VERSION,
        'episode_id': config.episode_id,
        'decision_snapshot_sha256': binding.decision_snapshot_sha256,
        'ranking': ['every eligible candidate ID exactly once, best first'],
        'forecasts': [
            {
                'candidate_id': 'candidate ID',
                'target_id': 'forecast target ID',
                'horizon_days': 'forecast horizon from the episode',
                'probability': 'number from 0 to 1',
            }
        ],
        'assessments': [
            {
                'candidate_id': 'top-portfolio candidate ID',
                'dimension': 'one required dimension',
                'conclusion': 'favorable | concern | mixed | insufficient',
                'citations': [
                    {
                        'evidence_id': 'visible evidence ID',
                        'stance': 'support | concern',
                        'quote': 'an exact quote copied from that evidence body',
                    }
                ],
            }
        ],
    }
    if config.task_type == PRECLINICAL_CANDIDATE_ADVANCEMENT_TASK:
        task_opening = (
            'Rank only the already-defined candidates for preclinical advancement and forecast later '
            'validation. Do not invent or modify candidates, and do not propose experimental procedures. '
        )
    elif config.task_type == EARLY_CLINICAL_ARM_PRIORITIZATION_TASK:
        task_opening = (
            'Prioritize only the already-defined, blinded early-clinical vaccine regimens using only frozen '
            'pre-results protocol evidence. Rank the regimens by the benchmark-defined composite advancement '
            'objective, not by clinical efficacy. Forecast the probability that each regimen clears the '
            "benchmark-defined multi-endpoint advancement threshold. The objective divides each regimen's "
            'Day-91 point estimate by the concurrent control separately for the functional-antibody, '
            'binding-antibody, and polyfunctional-helper-T-cell endpoints, then takes their equal-weight '
            'geometric mean. A composite of at least 8 clears the binary threshold; ranking grades use the '
            'fixed bins below 1, 1 to below 2, 2 to below 4, 4 to below 8, and at least 8. Do not invent or '
            'modify regimens, infer unshown results, or propose experimental procedures. '
        )
    elif config.task_type == ANTIGEN_TARGET_PRIORITIZATION_TASK:
        task_opening = 'Rank the sealed antigen targets and forecast later functional validation. '
    else:  # pragma: no cover - TaskType is closed, but keep rendering fail-closed if it expands.
        raise ValueError(f'unsupported prospective task type: {config.task_type}')
    evidence_instruction = (
        'This diagnostic view intentionally contains no episode evidence. Do not recover or use external '
        'sources. Assessments may use empty citation lists. '
        if variant == PromptVariant.NO_EVIDENCE
        else ''
    )
    return (
        task_opening + 'The ranking must contain every candidate exactly once. Provide one forecast for every '
        f'candidate/target pair. For each of the top {config.portfolio_size} candidates, provide exactly '
        'one assessment for every required dimension. Citation quotes must be exact substrings of the '
        f'cited evidence body. {evidence_instruction}\n\n'
        f'EPISODE\n{json.dumps(episode, ensure_ascii=False, indent=2)}\n\n'
        f'OUTPUT CONTRACT\n{json.dumps(output_contract, ensure_ascii=False, indent=2)}'
    )


def build_prospective_challenge_bundle(
    output_dir: Path,
    *,
    challenge_id: str,
    suite_id: str,
    packages: Sequence[LoadedProspectiveDecisionPackage],
    seals: Sequence[LoadedProspectiveDecisionSeal],
    admission: ProspectiveChallengeAdmission,
    sample_index: int = 0,
    prompt_variant: PromptVariant = PromptVariant.FULL,
) -> LoadedProspectiveChallengeBundle:
    """Atomically create a self-contained public challenge from already verified seals.

    The seal objects must have been produced or loaded with a trusted receipt verifier.  This
    function rechecks their exact on-disk bytes and all non-cryptographic bindings, but it cannot
    independently know whether a callback used earlier was trustworthy.
    """

    package_inputs = tuple(packages)
    seal_inputs = tuple(seals)
    if not package_inputs:
        raise ValueError('cannot create a prospective challenge from zero packages')
    package_ids = tuple(package.manifest.episode.episode_id for package in package_inputs)
    seal_ids = tuple(seal.manifest.episode_id for seal in seal_inputs)
    if len(package_ids) != len(set(package_ids)):
        raise ValueError('prospective challenge package episode IDs must be unique')
    if len(seal_ids) != len(set(seal_ids)):
        raise ValueError('prospective challenge seal episode IDs must be unique')
    if set(package_ids) != set(seal_ids):
        raise ValueError('every prospective package requires exactly one verified decision seal')

    # Re-read package bytes and seal sidecars so a stale Loaded object cannot hide later tampering.
    reloaded_packages: dict[str, LoadedProspectiveDecisionPackage] = {}
    for supplied in package_inputs:
        reloaded = load_prospective_decision_package(supplied.root)
        if reloaded != supplied:
            raise ValueError(f'loaded prospective package changed on disk: {supplied.manifest.episode.episode_id}')
        reloaded_packages[reloaded.manifest.episode.episode_id] = reloaded
    supplied_seals = {seal.manifest.episode_id: seal for seal in seal_inputs}
    ordered_packages = tuple(reloaded_packages[episode_id] for episode_id in sorted(reloaded_packages))
    ordered_seals: list[LoadedProspectiveDecisionSeal] = []
    for package in ordered_packages:
        supplied_seal = supplied_seals[package.manifest.episode.episode_id]
        structural_seal = _load_structural_seal(supplied_seal.root, package)
        if structural_seal != supplied_seal:
            raise ValueError(f'loaded prospective seal changed on disk: {supplied_seal.manifest.episode_id}')
        ordered_seals.append(structural_seal)

    first = ordered_packages[0].manifest.episode
    suite = ProspectiveSuiteManifest(
        suite_id=suite_id,
        task_type=first.task_type,
        reward_version=first.reward_version,
        split=Split.TEST,
        episodes=tuple(package.manifest.episode for package in ordered_packages),
    )
    suite_sha256 = prospective_suite_manifest_sha256(suite)
    _require_admission(admission, suite, suite_sha256)

    envelopes = tuple(
        ProspectiveChallengeEnvelope(
            challenge_id=challenge_id,
            suite_id=suite.suite_id,
            prospective_suite_sha256=suite_sha256,
            ordinal=ordinal,
            sample_index=sample_index,
            prompt_variant=prompt_variant,
            binding=package.manifest.episode,
            decision_package_sha256=package.manifest_sha256,
            decision_seal_sha256=ordered_seals[ordinal].manifest_sha256,
            messages=(
                ChatMessage(role='system', content=_SYSTEM_PROMPT),
                ChatMessage(
                    role='user',
                    content=build_prospective_episode_prompt(package, variant=prompt_variant),
                ),
            ),
        )
        for ordinal, package in enumerate(ordered_packages)
    )
    if sum(len(canonical_json_bytes(envelope)) for envelope in envelopes) > _MAX_TOTAL_ENVELOPE_BYTES:
        raise ValueError('prospective challenge envelopes exceed the aggregate size limit')
    episode_files = tuple(
        ProspectiveChallengeEpisodeFile(
            ordinal=ordinal,
            episode_id=package.manifest.episode.episode_id,
            envelope_path=f'episodes/{ordinal:06d}.json',
            envelope_sha256=prospective_challenge_envelope_sha256(envelopes[ordinal]),
            package_root=f'packages/{ordinal:06d}',
            decision_package_sha256=package.manifest_sha256,
            seal_root=f'seals/{ordinal:06d}',
            decision_seal_sha256=ordered_seals[ordinal].manifest_sha256,
        )
        for ordinal, package in enumerate(ordered_packages)
    )
    manifest = ProspectiveChallengeBundleManifest(
        challenge_id=challenge_id,
        suite_id=suite.suite_id,
        prospective_suite_sha256=suite_sha256,
        prospective_admission_sha256=prospective_challenge_admission_sha256(admission),
        prompt_variant=prompt_variant,
        episodes=episode_files,
    )

    target = output_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise ValueError(f'prospective challenge output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        (staging / 'episodes').mkdir()
        (staging / 'packages').mkdir()
        (staging / 'seals').mkdir()
        (staging / 'suite.json').write_bytes(canonical_json_bytes(suite))
        (staging / 'admission.json').write_bytes(canonical_json_bytes(admission))
        for file_binding, envelope, package, seal in zip(
            episode_files,
            envelopes,
            ordered_packages,
            ordered_seals,
            strict=True,
        ):
            (staging / file_binding.envelope_path).write_bytes(canonical_json_bytes(envelope))
            _write_package(staging / file_binding.package_root, package)
            _write_seal(staging / file_binding.seal_root, seal)
        (staging / 'challenge.json').write_bytes(canonical_json_bytes(manifest))
        _normalize_permissions(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # The returned loader result deliberately does not claim to have rerun the external verifier.
    return load_prospective_challenge_bundle(target)


def load_prospective_challenge_bundle(
    root: Path,
    *,
    receipt_verifier: TemporalReceiptVerifier | None = None,
) -> LoadedProspectiveChallengeBundle:
    """Load a challenge, optionally re-verifying every external timestamp proof."""

    resolved = _resolve_root(root)
    actual_files, actual_directories = _scan_inventory(resolved)
    manifest_bytes = _read_regular_file(resolved / 'challenge.json', _MAX_MANIFEST_BYTES)
    try:
        manifest = ProspectiveChallengeBundleManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ProspectiveChallengeIntegrityError(f'invalid prospective challenge manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ProspectiveChallengeIntegrityError('prospective challenge manifest must use canonical JSON encoding')

    suite = _load_canonical_model(
        resolved / manifest.suite_path,
        ProspectiveSuiteManifest,
        'prospective suite',
    )
    suite_sha256 = prospective_suite_manifest_sha256(suite)
    if suite_sha256 != manifest.prospective_suite_sha256 or suite.suite_id != manifest.suite_id:
        raise ProspectiveChallengeIntegrityError('prospective suite does not match the challenge manifest')
    admission = _load_canonical_model(
        resolved / manifest.admission_path,
        ProspectiveChallengeAdmission,
        'prospective admission',
    )
    if prospective_challenge_admission_sha256(admission) != manifest.prospective_admission_sha256:
        raise ProspectiveChallengeIntegrityError('prospective admission hash does not match the challenge manifest')
    try:
        _require_admission(admission, suite, suite_sha256)
    except ValueError as error:
        raise ProspectiveChallengeIntegrityError(str(error)) from error
    if len(suite.episodes) != len(manifest.episodes):
        raise ProspectiveChallengeIntegrityError('prospective suite and challenge episode counts differ')

    packages: list[LoadedProspectiveDecisionPackage] = []
    seals: list[LoadedProspectiveDecisionSeal] = []
    envelopes: list[ProspectiveChallengeEnvelope] = []
    total_envelope_bytes = 0
    expected_files = {'challenge.json', manifest.suite_path, manifest.admission_path}
    expected_directories = {'episodes', 'packages', 'seals'}
    for file_binding, suite_binding in zip(manifest.episodes, suite.episodes, strict=True):
        package_root = resolved / file_binding.package_root
        try:
            package = load_prospective_decision_package(package_root)
        except ProspectiveIntegrityError as error:
            raise ProspectiveChallengeIntegrityError(
                f'invalid prospective package for {file_binding.episode_id}: {error}'
            ) from error
        if (
            package.manifest_sha256 != file_binding.decision_package_sha256
            or package.manifest.episode != suite_binding
            or package.manifest.episode.episode_id != file_binding.episode_id
        ):
            raise ProspectiveChallengeIntegrityError(
                f'prospective package binding mismatch for {file_binding.episode_id}'
            )
        seal_root = resolved / file_binding.seal_root
        try:
            seal = (
                load_prospective_decision_seal(
                    seal_root,
                    package=package,
                    receipt_verifier=receipt_verifier,
                )
                if receipt_verifier is not None
                else _load_structural_seal(seal_root, package)
            )
        except ProspectiveIntegrityError as error:
            raise ProspectiveChallengeIntegrityError(
                f'invalid prospective seal for {file_binding.episode_id}: {error}'
            ) from error
        if seal.manifest_sha256 != file_binding.decision_seal_sha256:
            raise ProspectiveChallengeIntegrityError(f'prospective seal binding mismatch for {file_binding.episode_id}')

        envelope_bytes = _read_regular_file(resolved / file_binding.envelope_path, _MAX_ENVELOPE_BYTES)
        total_envelope_bytes += len(envelope_bytes)
        if total_envelope_bytes > _MAX_TOTAL_ENVELOPE_BYTES:
            raise ProspectiveChallengeIntegrityError('prospective envelopes exceed the aggregate size limit')
        try:
            envelope = ProspectiveChallengeEnvelope.model_validate_json(envelope_bytes)
        except ValueError as error:
            raise ProspectiveChallengeIntegrityError(
                f'invalid prospective challenge envelope {file_binding.envelope_path}: {error}'
            ) from error
        if envelope_bytes != canonical_json_bytes(envelope):
            raise ProspectiveChallengeIntegrityError(
                f'prospective challenge envelope {file_binding.envelope_path} is not canonical JSON'
            )
        if prospective_challenge_envelope_sha256(envelope) != file_binding.envelope_sha256:
            raise ProspectiveChallengeIntegrityError(
                f'prospective challenge envelope hash mismatch for {file_binding.envelope_path}'
            )
        expected_messages = (
            ChatMessage(role='system', content=_SYSTEM_PROMPT),
            ChatMessage(
                role='user',
                content=build_prospective_episode_prompt(package, variant=manifest.prompt_variant),
            ),
        )
        if (
            envelope.ordinal != file_binding.ordinal
            or envelope.binding != suite_binding
            or envelope.binding.episode_id != file_binding.episode_id
            or envelope.decision_package_sha256 != package.manifest_sha256
            or envelope.decision_seal_sha256 != seal.manifest_sha256
            or envelope.challenge_id != manifest.challenge_id
            or envelope.suite_id != suite.suite_id
            or envelope.prospective_suite_sha256 != suite_sha256
            or envelope.prompt_variant != manifest.prompt_variant
            or envelope.messages != expected_messages
        ):
            raise ProspectiveChallengeIntegrityError(
                f'prospective envelope metadata or rendered payload mismatch for {file_binding.episode_id}'
            )
        packages.append(package)
        seals.append(seal)
        envelopes.append(envelope)
        package_files, package_directories = _expected_package_inventory(file_binding.package_root, package.manifest)
        seal_files, seal_directories = _expected_seal_inventory(file_binding.seal_root, seal.manifest)
        expected_files.update(package_files)
        expected_files.update(seal_files)
        expected_files.add(file_binding.envelope_path)
        expected_directories.update(package_directories)
        expected_directories.update(seal_directories)

    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ProspectiveChallengeIntegrityError(
            f'prospective challenge file allowlist mismatch; missing={missing}, extra={extra}'
        )
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        raise ProspectiveChallengeIntegrityError(
            f'prospective challenge directory allowlist mismatch; missing={missing}, extra={extra}'
        )
    return LoadedProspectiveChallengeBundle(
        root=resolved,
        manifest=manifest,
        suite=suite,
        admission=admission,
        envelopes=tuple(envelopes),
        packages=tuple(packages),
        seals=tuple(seals),
        manifest_sha256=prospective_challenge_bundle_sha256(manifest),
        authority_proofs_reverified=receipt_verifier is not None,
    )


def _require_admission(
    admission: ProspectiveChallengeAdmission,
    suite: ProspectiveSuiteManifest,
    suite_sha256: str,
) -> None:
    if admission.suite_sha256 != suite_sha256:
        raise ValueError('prospective admission suite hash does not match the selected suite')
    if admission.episodes != suite.episodes:
        raise ValueError('prospective admission episode bindings do not exactly match the selected suite')


def _load_structural_seal(
    root: Path,
    package: LoadedProspectiveDecisionPackage,
) -> LoadedProspectiveDecisionSeal:
    """Verify a seal's bytes and semantics without claiming authority-proof verification."""

    resolved = _resolve_nested_root(root, 'prospective decision seal')
    actual_files, actual_directories = _scan_inventory(resolved)
    manifest = _load_canonical_model(resolved / 'seal.json', ProspectiveDecisionSealManifest, 'decision seal')
    expected_files = {'seal.json', *(proof.path for proof in manifest.proofs)}
    if actual_files != expected_files or actual_directories != {'proofs'}:
        raise ProspectiveIntegrityError('prospective seal file allowlist mismatch')
    proof_artifacts: dict[str, bytes] = {}
    for receipt, proof in zip(manifest.receipts, manifest.proofs, strict=True):
        payload = _read_regular_file(resolved / proof.path, _MAX_PROOF_BYTES)
        if (
            len(payload) != proof.byte_count
            or _sha256(payload) != proof.sha256
            or len(payload) != receipt.receipt_bytes
            or _sha256(payload) != receipt.receipt_sha256
        ):
            raise ProspectiveIntegrityError(f'prospective proof does not match receipt {receipt.receipt_id}')
        proof_artifacts[receipt.receipt_id] = payload
    if (
        manifest.episode_id != package.manifest.episode.episode_id
        or manifest.decision_at != package.manifest.episode.decision_at
        or manifest.decision_package_sha256 != package.manifest_sha256
        or manifest.decision_snapshot_sha256 != package.manifest.episode.decision_snapshot_sha256
    ):
        raise ProspectiveIntegrityError('prospective seal does not bind the supplied decision package')
    for request, receipt in zip(package.receipt_requests, manifest.receipts, strict=True):
        if (
            request.role != receipt.role
            or request.artifact_schema_version != receipt.artifact_schema_version
            or request.artifact_sha256 != receipt.artifact_sha256
            or request.artifact_bytes != receipt.artifact_bytes
        ):
            raise ProspectiveIntegrityError(f'{receipt.role.value} receipt does not bind the requested artifact')
    snapshot = package.manifest.episode.decision_snapshot
    candidate_receipt, evidence_receipt, decision_receipt = manifest.receipts
    if candidate_receipt.witnessed_at < snapshot.protocol_commitments.candidate_set_available_at:
        raise ProspectiveIntegrityError('candidate receipt cannot predate candidate-set availability')
    if evidence_receipt.witnessed_at < snapshot.latest_visible_evidence_at:
        raise ProspectiveIntegrityError('evidence receipt cannot predate included evidence availability')
    if decision_receipt.witnessed_at < max(candidate_receipt.witnessed_at, evidence_receipt.witnessed_at):
        raise ProspectiveIntegrityError('decision receipt cannot predate its candidate or evidence components')
    return LoadedProspectiveDecisionSeal(
        root=resolved,
        manifest=manifest,
        manifest_sha256=prospective_decision_seal_sha256(manifest),
        proof_artifacts=proof_artifacts,
    )


def _write_package(root: Path, package: LoadedProspectiveDecisionPackage) -> None:
    root.mkdir()
    (root / 'protocols').mkdir()
    (root / 'source-captures').mkdir()
    (root / 'decision.json').write_bytes(canonical_json_bytes(package.manifest))
    (root / 'candidates.jsonl').write_bytes(_record_bytes(package.candidates))
    (root / 'evidence.jsonl').write_bytes(_record_bytes(package.evidence))
    for binding in package.manifest.protocols:
        (root / binding.file.path).write_bytes(package.protocol_artifacts[binding.name])
    for binding in package.manifest.source_captures:
        (root / binding.file.path).write_bytes(package.source_capture_artifacts[binding.source_id])


def _write_seal(root: Path, seal: LoadedProspectiveDecisionSeal) -> None:
    root.mkdir()
    (root / 'proofs').mkdir()
    (root / 'seal.json').write_bytes(canonical_json_bytes(seal.manifest))
    for proof in seal.manifest.proofs:
        (root / proof.path).write_bytes(seal.proof_artifacts[proof.receipt_id])


def _expected_package_inventory(
    root: str,
    manifest: ProspectiveDecisionPackageManifest,
) -> tuple[set[str], set[str]]:
    files = {
        f'{root}/decision.json',
        f'{root}/{manifest.candidates.path}',
        f'{root}/{manifest.evidence.path}',
        *(f'{root}/{binding.file.path}' for binding in manifest.protocols),
        *(f'{root}/{binding.file.path}' for binding in manifest.source_captures),
    }
    directories = {root, f'{root}/protocols', f'{root}/source-captures'}
    return files, directories


def _expected_seal_inventory(
    root: str,
    manifest: ProspectiveDecisionSealManifest,
) -> tuple[set[str], set[str]]:
    files = {f'{root}/seal.json', *(f'{root}/{proof.path}' for proof in manifest.proofs)}
    directories = {root, f'{root}/proofs'}
    return files, directories


def _record_bytes(records: Sequence[CandidateRecord] | Sequence[EvidenceRecord]) -> bytes:
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _load_canonical_model(path: Path, model: type[StrictModel], label: str):
    payload = _read_regular_file(path, _MAX_MANIFEST_BYTES)
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise ProspectiveChallengeIntegrityError(f'invalid {label}: {error}') from error
    if payload != canonical_json_bytes(value):
        raise ProspectiveChallengeIntegrityError(f'{label} must use canonical JSON encoding')
    return value


def _resolve_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveChallengeIntegrityError('prospective challenge root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ProspectiveChallengeIntegrityError(f'prospective challenge root does not exist: {resolved}')
    return resolved


def _resolve_nested_root(root: Path, label: str) -> Path:
    if root.is_symlink():
        raise ProspectiveIntegrityError(f'{label} root cannot be a symlink')
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ProspectiveIntegrityError(f'{label} root does not exist: {resolved}')
    return resolved


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProspectiveChallengeIntegrityError(f'cannot open prospective challenge file {path}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveChallengeIntegrityError(f'prospective challenge artifact is not a regular file: {path}')
        if metadata.st_size > maximum_bytes:
            raise ProspectiveChallengeIntegrityError(f'prospective challenge file exceeds its size limit: {path}')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ProspectiveChallengeIntegrityError(f'prospective challenge file exceeds its size limit: {path}')
    except OSError as error:
        raise ProspectiveChallengeIntegrityError(f'cannot read prospective challenge file {path}: {error}') from error
    finally:
        os.close(descriptor)


def _scan_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        for path in root.rglob('*'):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ProspectiveChallengeIntegrityError(
                    f'prospective challenge artifact cannot contain symlinks: {relative}'
                )
            if path.is_dir():
                directories.add(relative)
            elif path.is_file():
                files.add(relative)
            else:
                raise ProspectiveChallengeIntegrityError(
                    f'prospective challenge contains a non-regular artifact: {relative}'
                )
    except OSError as error:
        raise ProspectiveChallengeIntegrityError(f'cannot inventory prospective challenge: {error}') from error
    return files, directories


def _normalize_permissions(root: Path) -> None:
    for path in root.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
