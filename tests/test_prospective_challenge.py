from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseUniverseDisposition,
    CaseUniverseEntry,
    CaseUniverseManifest,
    CaseUniverseSeal,
    case_universe_content_sha256,
)
from vaxreplay.case_schema import (
    EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
    AssessmentConclusion,
    CandidateAssessment,
    CandidateForecast,
    Split,
)
from vaxreplay.operations.promotion_schema import (
    CAPTURE_INDEX_SCHEMA_VERSION,
    CAPTURE_PROMOTION_SCHEMA_VERSION,
    PROMOTION_HANDOFF_SCHEMA_VERSION,
)
from vaxreplay.operations.scope_precommit import SCOPE_PRECOMMIT_SCHEMA_VERSION
from vaxreplay.prompt import PromptVariant
from vaxreplay.prospective import (
    SourceCaptureArtifact,
    build_prospective_decision_package,
    build_prospective_decision_seal,
)
from vaxreplay.prospective_admission import (
    PromotionArchiveAdmissionPolicy,
    PromotionArchivePolicyEntry,
    ProspectiveAdmissionError,
    build_verified_prospective_admission,
    make_promotion_archive_admission_verifier,
)
from vaxreplay.prospective_schema import (
    PROSPECTIVE_RESPONSE_PROTOCOL,
    PROSPECTIVE_SUBMISSION_SCHEMA_VERSION,
    ProspectiveChallengeAdmission,
    ProspectiveSplitInventory,
    ProspectiveSubmission,
    ProspectiveSuiteManifest,
    prospective_suite_manifest_sha256,
)
from vaxreplay.runner.backend import PreparedBackend, RawExecutionResult, RawExecutionStatus
from vaxreplay.runner.orchestrator import receipt_key_id, run_challenge_bundle
from vaxreplay.runner.prospective_challenge import (
    ProspectiveChallengeIntegrityError,
    build_prospective_challenge_bundle,
    build_prospective_episode_prompt,
    load_prospective_challenge_bundle,
    prospective_challenge_envelope_sha256,
)
from vaxreplay.runner.schema import (
    BackendCapabilities,
    EpisodeRunStatus,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)
from vaxreplay.temporal_schema import (
    DecisionTimeConfig,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalReceiptAuthority,
)


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return receipt.authority_id == 'fixture-transparency-log' and proof.startswith(
        b'verified fictional transparency proof'
    )


def _source_verifier(binding, manifest_bytes: bytes, policy: bytes) -> bool:  # noqa: ANN001
    return (
        binding.source_id == 'fictional-source'
        and manifest_bytes.startswith(b'{')
        and policy == b'fixed source-capture replay and eligibility policy'
    )


def _build_inputs(root: Path, *, task_type: str | None = None):
    bundle = EpisodeBundle.load(_fixture())
    manifest_updates: dict[str, object] = {'synthetic': False, 'split': Split.TEST}
    if task_type is not None:
        manifest_updates['task_type'] = task_type
    config = DecisionTimeConfig.from_manifest(bundle.manifest.model_copy(update=manifest_updates))
    package = build_prospective_decision_package(
        root / 'package',
        config=config,
        candidates=bundle.candidates,
        evidence=bundle.visible_evidence,
        protocol_artifacts={
            'candidate_set_definition': b'fixed complete panel, inclusion, exclusion, and ordering rules',
            'evidence_acquisition_spec': b'fixed source releases, queries, mappings, and availability rules',
            'outcome_adjudication_spec': b'fixed endpoint, horizon, censoring, and derivation rules',
        },
        candidate_set_available_at=config.decision_at - timedelta(days=4),
        source_captures=(
            SourceCaptureArtifact(
                source_id='fictional-source',
                source_release_at=config.decision_at - timedelta(days=3),
                captured_at=config.decision_at - timedelta(days=2),
                witnessed_at=config.decision_at - timedelta(days=1),
                manifest_bytes=canonical_json_bytes(
                    {
                        'schema_version': 'fictional.capture.v1',
                        'source_id': 'fictional-source',
                        'release': '2024-02-27',
                        'records_sha256': 'a' * 64,
                    }
                ),
            ),
        ),
    )
    witnessed = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            package.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (package.manifest.episode.decision_snapshot.latest_visible_evidence_at),
        TemporalArtifactRole.DECISION_SNAPSHOT: package.manifest.episode.decision_at,
    }
    receipts = []
    proofs = {}
    for ordinal, request in enumerate(package.receipt_requests):
        receipt_id = f'receipt-{ordinal}'
        proof = f'verified fictional transparency proof {ordinal}'.encode()
        proofs[receipt_id] = proof
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=request.role,
                artifact_schema_version=request.artifact_schema_version,
                artifact_sha256=request.artifact_sha256,
                artifact_bytes=request.artifact_bytes,
                witnessed_at=witnessed[request.role],
                authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
                authority_id='fixture-transparency-log',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://transparency.invalid/{receipt_id}',
            )
        )
    seal = build_prospective_decision_seal(
        root / 'seal',
        package=package,
        receipts=tuple(receipts),
        proof_artifacts=proofs,
        receipt_verifier=_verifier,
        verified_at=config.decision_at + timedelta(seconds=1),
    )
    binding = package.manifest.episode
    suite = ProspectiveSuiteManifest(
        suite_id='prospective-suite-1',
        task_type=binding.task_type,
        reward_version=binding.reward_version,
        split=binding.split,
        episodes=(binding,),
    )
    admission = ProspectiveChallengeAdmission(
        release_id='prospective-release-1',
        suite_sha256=prospective_suite_manifest_sha256(suite),
        split_inventory_sha256='1' * 64,
        case_universe_sha256='2' * 64,
        verifier_policy_sha256='3' * 64,
        source_capture_policy_sha256='6' * 64,
        eligibility_protocol_sha256='4' * 64,
        attempt_policy_sha256='5' * 64,
        run_deadline_at=config.decision_at + timedelta(days=1),
        episodes=(binding,),
    )
    return package, seal, admission


def _build_challenge(root: Path):
    package, seal, admission = _build_inputs(root)
    challenge = build_prospective_challenge_bundle(
        root / 'challenge',
        challenge_id='prospective-challenge-1',
        suite_id='prospective-suite-1',
        packages=(package,),
        seals=(seal,),
        admission=admission,
    )
    return challenge, package, seal, admission


def _verified_admission(
    root: Path,
    *,
    source_verifier=_source_verifier,
    case_universe_predates_source_witness: bool = False,
):
    package, seal, _direct_admission = _build_inputs(root)
    eligibility = b'fixed exhaustive prospective case eligibility protocol'
    entry = CaseUniverseEntry(
        case_id='fictional-case-1',
        lineage_group_id=package.manifest.episode.lineage_group_id,
        disposition=CaseUniverseDisposition.PREELIGIBLE,
        decision_package_sha256=package.manifest_sha256,
    )
    universe_content_sha256 = case_universe_content_sha256(
        universe_id='prospective-universe-1',
        eligibility_protocol_sha256=hashlib.sha256(eligibility).hexdigest(),
        entries=(entry,),
    )
    proof = b'verified fictional case-universe transparency proof'
    case_universe = CaseUniverseManifest(
        universe_id='prospective-universe-1',
        eligibility_protocol_sha256=hashlib.sha256(eligibility).hexdigest(),
        entries=(entry,),
        universe_content_sha256=universe_content_sha256,
        seal=CaseUniverseSeal(
            universe_content_sha256=universe_content_sha256,
            witnessed_at=(
                package.manifest.source_captures[0].captured_at
                if case_universe_predates_source_witness
                else package.manifest.episode.decision_at
            ),
            authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
            authority_id='fixture-transparency-log',
            proof_sha256=hashlib.sha256(proof).hexdigest(),
            proof_bytes=len(proof),
            verification_uri='https://transparency.invalid/case-universe',
        ),
    )
    split_inventory = ProspectiveSplitInventory(
        inventory_id='prospective-split-1',
        episodes=(package.manifest.episode,),
    )
    verified = build_verified_prospective_admission(
        release_id='prospective-release-1',
        suite_id='prospective-suite-1',
        packages=(package,),
        seals=(seal,),
        split_inventory=split_inventory,
        case_universe=case_universe,
        case_universe_proof=proof,
        eligibility_protocol=eligibility,
        verifier_policy=b'fixed trusted verifier allowlist and proof semantics',
        source_capture_policy=b'fixed source-capture replay and eligibility policy',
        attempt_policy=b'one immutable attempt per system; failures remain invalid',
        run_deadline_at=package.manifest.episode.decision_at + timedelta(days=1),
        receipt_verifier=_verifier,
        case_universe_seal_verifier=lambda candidate, payload: (
            candidate.authority_id == 'fixture-transparency-log'
            and payload == b'verified fictional case-universe transparency proof'
        ),
        source_capture_verifier=source_verifier,
    )
    return verified, proof, eligibility


class _FakeBackend:
    def __init__(self, stdout: bytes):
        self.stdout = stdout
        self.prepare_calls = 0

    def prepare(self, _system, _policy) -> PreparedBackend:
        self.prepare_calls += 1
        return PreparedBackend(
            capabilities=BackendCapabilities(
                backend_id='prospective-test-backend',
                backend_version='1',
                isolation_tier=IsolationTier.DEVELOPMENT,
                network_isolation=True,
                host_filesystem_isolation=True,
                read_only_root=True,
                non_root_user=True,
                capability_drop=True,
                no_new_privileges=True,
                process_limit=True,
                memory_limit=True,
                cpu_limit=True,
                scratch_limit=True,
                fresh_worker_per_episode=True,
            ),
            resolved_image_id='sha256:' + 'b' * 64,
        )

    def run(self, **_kwargs) -> RawExecutionResult:
        return RawExecutionResult(
            status=RawExecutionStatus.EXITED,
            exit_code=0,
            duration_ms=2,
            stdout=self.stdout,
            stderr=b'',
            stdout_truncated=False,
            stderr_truncated=False,
        )


def _prospective_response(challenge) -> ProspectiveSubmission:
    binding = challenge.suite.episodes[0]
    config = binding.decision_snapshot.config
    ranking = tuple(config.candidate_ids)
    return ProspectiveSubmission(
        episode_id=binding.episode_id,
        decision_snapshot_sha256=binding.decision_snapshot_sha256,
        ranking=ranking,
        forecasts=tuple(
            CandidateForecast(
                candidate_id=candidate_id,
                target_id=target.target_id,
                horizon_days=target.horizon_days,
                probability=0.5,
            )
            for candidate_id in ranking
            for target in config.forecast_targets
        ),
        assessments=tuple(
            CandidateAssessment(
                candidate_id=candidate_id,
                dimension=dimension,
                conclusion=AssessmentConclusion.INSUFFICIENT,
            )
            for candidate_id in ranking[: config.portfolio_size]
            for dimension in config.required_dimensions
        ),
    )


class ProspectiveChallengeTest(unittest.TestCase):
    def test_promotion_archive_policy_cannot_omit_allowlist(self) -> None:
        with self.assertRaisesRegex(ValueError, 'at least 1'):
            PromotionArchiveAdmissionPolicy(
                policy_id='empty-policy',
                archives=(),
            )

        policy = PromotionArchiveAdmissionPolicy(
            policy_id='missing-runtime-map',
            archives=(
                PromotionArchivePolicyEntry(
                    promotion_id='promotion-1',
                    source_id='promotion:promotion-1',
                    promotion_manifest_sha256='1' * 64,
                    capture_index_sha256='2' * 64,
                    handoff_descriptor_sha256='3' * 64,
                    scope_policy_sha256='4' * 64,
                    scope_precommit_sha256='5' * 64,
                    campaign_id='pandemic-campaign-2027',
                    selection_key='antigen-prioritization-plan',
                    selection_policy_sha256='6' * 64,
                    selection_policy_artifact_sha256='7' * 64,
                    selection_manifest_sha256='8' * 64,
                ),
            ),
        )
        with self.assertRaisesRegex(ProspectiveAdmissionError, 'exactly match its allowlist'):
            make_promotion_archive_admission_verifier(policy=policy, archives={})

    def test_early_clinical_prompt_preserves_decision_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package, _seal, _admission = _build_inputs(
                Path(temporary_directory),
                task_type=EARLY_CLINICAL_ARM_PRIORITIZATION_TASK,
            )

            prompt = build_prospective_episode_prompt(package)

            self.assertIn('blinded early-clinical vaccine regimens', prompt)
            self.assertIn('frozen pre-results protocol evidence', prompt)
            self.assertIn('benchmark-defined composite advancement objective, not by clinical efficacy', prompt)
            self.assertIn('clears the benchmark-defined multi-endpoint advancement threshold', prompt)
            self.assertIn("divides each regimen's Day-91 point estimate by the concurrent control", prompt)
            self.assertIn('then takes their equal-weight geometric mean', prompt)
            self.assertIn('A composite of at least 8 clears the binary threshold', prompt)
            self.assertIn('infer unshown results', prompt)

    def test_trusted_admission_gate_binds_complete_case_and_split_inventories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            verified, _proof, _eligibility = _verified_admission(root)
            challenge = build_prospective_challenge_bundle(
                root / 'challenge',
                challenge_id='verified-prospective-challenge',
                suite_id=verified.suite.suite_id,
                packages=verified.packages,
                seals=verified.seals,
                admission=verified.admission,
            )

            self.assertEqual(challenge.admission.case_inventory_complete, True)
            self.assertEqual(challenge.admission.split_inventory_complete, True)
            self.assertEqual(challenge.admission.purpose, 'prospective_research')
            self.assertEqual(challenge.suite, verified.suite)

    def test_trusted_admission_gate_rejects_unverified_source_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ProspectiveAdmissionError,
                'source-capture verifier rejected prospective research eligibility',
            ):
                _verified_admission(
                    Path(temporary_directory),
                    source_verifier=lambda _binding, _manifest, _policy: False,
                )

    def test_opaque_source_verifier_cannot_admit_reserved_promotion_bridge_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package, seal, _admission = _build_inputs(root)
            original_binding = package.manifest.source_captures[0]
            cases = (
                ('reserved-namespace', 'promotion:forged', b'{}'),
                (
                    'promotion-handoff-current',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': PROMOTION_HANDOFF_SCHEMA_VERSION}),
                ),
                (
                    'promotion-handoff-future',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': 'vaxreplay.promotion-handoff.v9.9'}),
                ),
                (
                    'capture-index-current',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': CAPTURE_INDEX_SCHEMA_VERSION}),
                ),
                (
                    'capture-index-future',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': 'vaxreplay.capture-index.v9.9'}),
                ),
                (
                    'capture-promotion-current',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': CAPTURE_PROMOTION_SCHEMA_VERSION}),
                ),
                (
                    'capture-promotion-future',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': 'vaxreplay.capture-promotion.v9.9'}),
                ),
                (
                    'scope-precommit-current',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': SCOPE_PRECOMMIT_SCHEMA_VERSION}),
                ),
                (
                    'scope-precommit-future',
                    original_binding.source_id,
                    canonical_json_bytes({'schema_version': 'vaxreplay.scope-precommit.v9.9'}),
                ),
            )
            for case, source_id, source_bytes in cases:
                binding = original_binding.model_copy(update={'source_id': source_id})
                manifest = package.manifest.model_copy(update={'source_captures': (binding,)})
                forged = replace(
                    package,
                    manifest=manifest,
                    source_capture_artifacts={source_id: source_bytes},
                )
                with (
                    self.subTest(case=case, source_id=source_id),
                    patch(
                        'vaxreplay.prospective_admission.load_prospective_decision_package',
                        return_value=forged,
                    ),
                    self.assertRaisesRegex(ProspectiveAdmissionError, 'official promotion archive'),
                ):
                    build_verified_prospective_admission(
                        release_id='unused-release',
                        suite_id='unused-suite',
                        packages=(forged,),
                        seals=(seal,),
                        split_inventory=ProspectiveSplitInventory(
                            inventory_id='unused-inventory',
                            episodes=(forged.manifest.episode,),
                        ),
                        case_universe=None,  # type: ignore[arg-type]
                        case_universe_proof=b'unused-proof',
                        eligibility_protocol=b'unused-eligibility',
                        verifier_policy=b'unused-verifier-policy',
                        source_capture_policy=b'unused-source-policy',
                        attempt_policy=b'unused-attempt-policy',
                        run_deadline_at=forged.manifest.episode.decision_at + timedelta(days=1),
                        receipt_verifier=_verifier,
                        case_universe_seal_verifier=lambda _seal, _proof: True,
                        source_capture_verifier=lambda _binding, _manifest, _policy: True,
                    )

    def test_trusted_admission_gate_requires_case_universe_after_source_witness(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(
                ProspectiveAdmissionError,
                'cannot predate its source-capture witnesses',
            ),
        ):
            _verified_admission(
                Path(temporary_directory),
                case_universe_predates_source_witness=True,
            )

    def test_trusted_admission_gate_rejects_case_universe_not_bound_to_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package, seal, _admission = _build_inputs(root)
            eligibility = b'fixed exhaustive prospective case eligibility protocol'
            entry = CaseUniverseEntry(
                case_id='fictional-case-1',
                lineage_group_id=package.manifest.episode.lineage_group_id,
                disposition=CaseUniverseDisposition.PREELIGIBLE,
                decision_package_sha256='f' * 64,
            )
            content_hash = case_universe_content_sha256(
                universe_id='wrong-universe',
                eligibility_protocol_sha256=hashlib.sha256(eligibility).hexdigest(),
                entries=(entry,),
            )
            proof = b'verified fictional case-universe transparency proof'
            universe = CaseUniverseManifest(
                universe_id='wrong-universe',
                eligibility_protocol_sha256=hashlib.sha256(eligibility).hexdigest(),
                entries=(entry,),
                universe_content_sha256=content_hash,
                seal=CaseUniverseSeal(
                    universe_content_sha256=content_hash,
                    witnessed_at=package.manifest.episode.decision_at,
                    authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
                    authority_id='fixture-transparency-log',
                    proof_sha256=hashlib.sha256(proof).hexdigest(),
                    proof_bytes=len(proof),
                    verification_uri='https://transparency.invalid/wrong-universe',
                ),
            )
            with self.assertRaisesRegex(ProspectiveAdmissionError, 'exactly cover'):
                build_verified_prospective_admission(
                    release_id='prospective-release-1',
                    suite_id='prospective-suite-1',
                    packages=(package,),
                    seals=(seal,),
                    split_inventory=ProspectiveSplitInventory(
                        inventory_id='prospective-split-1',
                        episodes=(package.manifest.episode,),
                    ),
                    case_universe=universe,
                    case_universe_proof=proof,
                    eligibility_protocol=eligibility,
                    verifier_policy=b'fixed trusted verifier policy',
                    source_capture_policy=b'fixed source-capture replay and eligibility policy',
                    attempt_policy=b'first attempt only',
                    run_deadline_at=package.manifest.episode.decision_at + timedelta(days=1),
                    receipt_verifier=_verifier,
                    case_universe_seal_verifier=lambda _seal, _proof: True,
                    source_capture_verifier=_source_verifier,
                )

    def test_builds_self_contained_label_free_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first, package, seal, _admission = _build_challenge(root / 'first')
            second, _package, _seal, _admission = _build_challenge(root / 'second')

            self.assertEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertFalse(first.authority_proofs_reverified)
            envelope = first.envelopes[0]
            self.assertEqual(envelope.response_protocol, PROSPECTIVE_RESPONSE_PROTOCOL)
            self.assertEqual(envelope.decision_package_sha256, package.manifest_sha256)
            self.assertEqual(envelope.decision_seal_sha256, seal.manifest_sha256)
            self.assertIn(package.manifest.episode.decision_snapshot_sha256, envelope.messages[1].content)
            self.assertIn(PROSPECTIVE_SUBMISSION_SCHEMA_VERSION, envelope.messages[1].content)
            self.assertNotIn('manifest_sha256', envelope.messages[1].content)
            self.assertNotIn('labels_sha256', envelope.messages[1].content)
            self.assertNotIn('outcome_snapshot', envelope.messages[1].content)
            files = {path.relative_to(first.root).as_posix() for path in first.root.rglob('*') if path.is_file()}
            self.assertEqual(
                files,
                {
                    'admission.json',
                    'challenge.json',
                    'suite.json',
                    'episodes/000000.json',
                    'packages/000000/candidates.jsonl',
                    'packages/000000/decision.json',
                    'packages/000000/evidence.jsonl',
                    'packages/000000/protocols/candidate_set_definition.bin',
                    'packages/000000/protocols/evidence_acquisition_spec.bin',
                    'packages/000000/protocols/outcome_adjudication_spec.bin',
                    'packages/000000/source-captures/000000.json',
                    'seals/000000/seal.json',
                    'seals/000000/proofs/000000.bin',
                    'seals/000000/proofs/000001.bin',
                    'seals/000000/proofs/000002.bin',
                },
            )

    def test_requires_exact_admission_and_one_verified_seal_per_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package, seal, admission = _build_inputs(root)
            with self.assertRaisesRegex(ValueError, 'exactly one verified'):
                build_prospective_challenge_bundle(
                    root / 'missing-seal',
                    challenge_id='challenge',
                    suite_id='prospective-suite-1',
                    packages=(package,),
                    seals=(),
                    admission=admission,
                )
            wrong_admission = admission.model_copy(update={'suite_sha256': 'f' * 64})
            with self.assertRaisesRegex(ValueError, 'suite hash'):
                build_prospective_challenge_bundle(
                    root / 'wrong-admission',
                    challenge_id='challenge',
                    suite_id='prospective-suite-1',
                    packages=(package,),
                    seals=(seal,),
                    admission=wrong_admission,
                )

    def test_loader_rejects_tampering_extra_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original, _package, _seal, _admission = _build_challenge(root / 'source')
            for case in ('tamper', 'extra', 'symlink'):
                copied = root / case
                shutil.copytree(original.root, copied)
                if case == 'tamper':
                    envelope = json.loads((copied / 'episodes' / '000000.json').read_bytes())
                    envelope['sample_index'] = 17
                    (copied / 'episodes' / '000000.json').write_bytes(canonical_json_bytes(envelope))
                    message = 'hash mismatch'
                elif case == 'extra':
                    (copied / 'future-labels.json').write_text('{}', encoding='utf-8')
                    message = 'allowlist'
                else:
                    (copied / 'leak').symlink_to(_fixture() / 'private')
                    message = 'symlink'
                with (
                    self.subTest(case=case),
                    self.assertRaisesRegex(
                        ProspectiveChallengeIntegrityError,
                        message,
                    ),
                ):
                    load_prospective_challenge_bundle(copied)

    def test_strict_envelope_rejects_dummy_future_fields_even_with_rehashed_bytes(self) -> None:
        for future_field in ('labels_sha256', 'outcome_snapshot', 'final_manifest_sha256'):
            with self.subTest(future_field=future_field), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                challenge, _package, _seal, _admission = _build_challenge(root)
                envelope_path = challenge.root / 'episodes' / '000000.json'
                envelope = json.loads(envelope_path.read_bytes())
                envelope[future_field] = 'f' * 64
                envelope_path.write_bytes(canonical_json_bytes(envelope))
                manifest_path = challenge.root / 'challenge.json'
                manifest = json.loads(manifest_path.read_bytes())
                manifest['episodes'][0]['envelope_sha256'] = hashlib.sha256(envelope_path.read_bytes()).hexdigest()
                manifest_path.write_bytes(canonical_json_bytes(manifest))

                with self.assertRaisesRegex(
                    ProspectiveChallengeIntegrityError,
                    'invalid prospective challenge envelope',
                ):
                    load_prospective_challenge_bundle(challenge.root)

    def test_loader_only_claims_authority_reverification_with_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge, _package, _seal, _admission = _build_challenge(root)
            structural = load_prospective_challenge_bundle(challenge.root)
            reverified = load_prospective_challenge_bundle(challenge.root, receipt_verifier=_verifier)

            self.assertFalse(structural.authority_proofs_reverified)
            self.assertTrue(reverified.authority_proofs_reverified)

            with self.assertRaisesRegex(ProspectiveChallengeIntegrityError, 'verifier rejected'):
                load_prospective_challenge_bundle(
                    challenge.root,
                    receipt_verifier=lambda _receipt, _proof: False,
                )

    def test_fixed_prompt_variants_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package, seal, admission = _build_inputs(root)
            challenge = build_prospective_challenge_bundle(
                root / 'challenge',
                challenge_id='challenge-no-evidence',
                suite_id='prospective-suite-1',
                packages=(package,),
                seals=(seal,),
                admission=admission,
                prompt_variant=PromptVariant.NO_EVIDENCE,
            )
            self.assertIn('"evidence": []', challenge.envelopes[0].messages[1].content)
            self.assertEqual(challenge.envelopes[0].prompt_variant, PromptVariant.NO_EVIDENCE)
            self.assertEqual(
                challenge.manifest.episodes[0].envelope_sha256,
                prospective_challenge_envelope_sha256(challenge.envelopes[0]),
            )

    def test_existing_one_shot_runner_executes_prospective_response_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            challenge, _package, _seal, _admission = _build_challenge(root)
            response = _prospective_response(challenge)
            backend = _FakeBackend(response.model_dump_json(indent=2).encode())
            policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
            key = bytes(range(32))
            system = SystemSubmissionManifest(
                submission_id='prospective-system',
                image_ref='sha256:' + 'a' * 64,
                entrypoint=('/opt/vaxreplay/run',),
                model_id='fictional-model',
                harness_id='fictional-harness',
                response_protocol=PROSPECTIVE_RESPONSE_PROTOCOL,
            )
            run = run_challenge_bundle(
                challenge,
                expected_challenge_sha256=challenge.manifest_sha256,
                system=system,
                policy=policy,
                receipt_key=key,
                expected_receipt_key_id=receipt_key_id(key),
                output_dir=root / 'run',
                backend=backend,
            )

            self.assertEqual(run.receipt.episodes[0].status, EpisodeRunStatus.ACCEPTED)
            self.assertEqual(run.responses, canonical_json_bytes(response) + b'\n')
            self.assertEqual(
                run.receipt.admission_sha256,
                challenge.manifest.prospective_admission_sha256,
            )
            self.assertEqual(
                run.receipt.suite_manifest_sha256,
                challenge.manifest.prospective_suite_sha256,
            )

            wrong_system = system.model_copy(update={'response_protocol': 'vaxreplay.submission-json-stdout.v0.1'})
            with self.assertRaisesRegex(ValueError, 'response protocol'):
                run_challenge_bundle(
                    challenge,
                    expected_challenge_sha256=challenge.manifest_sha256,
                    system=wrong_system,
                    policy=policy,
                    receipt_key=key,
                    expected_receipt_key_id=receipt_key_id(key),
                    output_dir=root / 'wrong-run',
                    backend=_FakeBackend(b'{}'),
                )


if __name__ == '__main__':
    unittest.main()
