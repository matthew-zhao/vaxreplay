from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    CaseSelectionRecord,
    CaseUniverseDisposition,
    CaseUniverseEntry,
    CaseUniverseManifest,
    CaseUniverseSeal,
    case_universe_content_sha256,
    case_universe_sha256,
)
from vaxreplay.case_schema import Split
from vaxreplay.contamination import (
    AuditDisposition,
    AuditReasonCode,
    CalibrationPolicy,
    ContaminationAuditManifest,
    ContaminationAuditPolicy,
    ExactRetrievalConfig,
    IdentifierNeedle,
    JudgeCalibrationResult,
    JudgeVerdict,
    LlmJudgeOutput,
    PinnedLlmJudge,
    build_contamination_audit,
    make_audit_input,
    make_audit_manifest,
    make_llm_audit_run,
)
from vaxreplay.contamination import (
    model_sha256 as contamination_model_sha256,
)
from vaxreplay.iedb.adapter import build_episode
from vaxreplay.iedb.raw_schema import IedbEpisodeSpec
from vaxreplay.prompt import PromptVariant, model_facing_payload_bytes
from vaxreplay.release import (
    ReleaseIntegrityError,
    TemporalAdmissionMaterial,
    build_retrospective_research_release,
    build_synthetic_integration_release,
    load_release,
    public_release_sha256,
)
from vaxreplay.release_schema import PublicReleaseManifest, ReleasePurpose
from vaxreplay.runner.orchestrator import receipt_key_id
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy
from vaxreplay.temporal_schema import (
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    DECISION_SNAPSHOT_SCHEMA_VERSION,
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    OUTCOME_SNAPSHOT_SCHEMA_VERSION,
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
    TemporalSourceTier,
    build_decision_snapshot_commitment,
    model_sha256,
)

_LABEL_KEY = bytes(range(32))
_RECEIPT_KEY = bytes(range(32, 64))


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'iedb_fictional_history'


def _iedb_spec(
    *,
    split: Split,
    episode_id: str = 'iedb-fictional-cohort-001',
    lineage_group_id: str = 'iedb-fictional-lineage-alpha',
) -> IedbEpisodeSpec:
    spec = IedbEpisodeSpec.model_validate_json((_fixture_root() / 'spec.json').read_bytes())
    return spec.model_copy(
        update={
            'split': split,
            'episode_id': episode_id,
            'lineage_group_id': lineage_group_id,
        }
    )


def _build_iedb_episode(
    output_root: Path,
    *,
    split: Split = Split.TEST,
    episode_id: str = 'iedb-fictional-cohort-001',
    lineage_group_id: str = 'iedb-fictional-lineage-alpha',
) -> Path:
    build_episode(
        spec=_iedb_spec(
            split=split,
            episode_id=episode_id,
            lineage_group_id=lineage_group_id,
        ),
        snapshot_roots=(
            _fixture_root() / 'snapshot_decision',
            _fixture_root() / 'snapshot_outcome',
        ),
        output_root=output_root,
        label_commitment_key=_LABEL_KEY,
    )
    return output_root


def _make_non_synthetic_episode(
    output_root: Path,
    *,
    split: Split = Split.TEST,
    episode_id: str = 'iedb-fictional-cohort-001',
    lineage_group_id: str = 'iedb-fictional-lineage-alpha',
) -> Path:
    _build_iedb_episode(
        output_root,
        split=split,
        episode_id=episode_id,
        lineage_group_id=lineage_group_id,
    )
    bundle = EpisodeBundle.load(output_root, include_private=True)
    manifest = bundle.manifest.model_copy(update={'synthetic': False})
    (output_root / 'manifest.json').write_bytes(canonical_json_bytes(manifest))
    EpisodeBundle.load(output_root, include_private=True).validate_integrity()
    return output_root


def _tier_b_material(episode_root: Path) -> tuple[TemporalAdmissionMaterial, bytes]:
    bundle = EpisodeBundle.load(episode_root, include_private=True)
    assert bundle.private_labels is not None
    protocol_artifacts = {
        'candidate_set_definition': b'complete independently archived fictional panel',
        'evidence_acquisition_spec': b'exact pre-cutoff bytes and deterministic timestamp resolution',
        'outcome_adjudication_spec': b'later homogeneous IEDB qualitative labels for every panel member',
    }
    protocol = DecisionProtocolCommitments(
        candidate_set_available_at=bundle.manifest.decision_at - timedelta(days=2),
        candidate_set_definition_sha256=hashlib.sha256(protocol_artifacts['candidate_set_definition']).hexdigest(),
        evidence_acquisition_spec_sha256=hashlib.sha256(protocol_artifacts['evidence_acquisition_spec']).hexdigest(),
        outcome_adjudication_spec_sha256=hashlib.sha256(protocol_artifacts['outcome_adjudication_spec']).hexdigest(),
    )
    decision = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        protocol,
    )
    raw_outcome_source = b'fictional independently archived later IEDB outcome bytes'
    source_audit = canonical_json_bytes(
        {
            'fixture_only': True,
            'episode_id': bundle.manifest.episode_id,
            'source_archive': 'fictional exact-byte archive',
        }
    )
    label_derivation_audit = source_audit
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
    outcome_snapshot = OutcomeSnapshotCommitment(
        episode_id=bundle.manifest.episode_id,
        labels_sha256=bundle.manifest.labels_sha256,
        label_commitment_scheme=bundle.manifest.label_commitment_scheme,
        outcome_adjudication_spec_sha256=protocol.outcome_adjudication_spec_sha256,
        raw_outcome_source_sha256=hashlib.sha256(raw_outcome_source).hexdigest(),
        raw_outcome_source_bytes=len(raw_outcome_source),
        label_derivation_audit_sha256=hashlib.sha256(label_derivation_audit).hexdigest(),
        label_derivation_audit_bytes=len(label_derivation_audit),
        target_availability=target_availability,
    )
    role_artifacts = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            decision.candidate_universe_or_panel_sha256,
            decision.candidate_universe_or_panel_bytes,
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (
            EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            decision.visible_evidence_sha256,
            decision.visible_evidence_bytes,
        ),
        TemporalArtifactRole.DECISION_SNAPSHOT: (
            DECISION_SNAPSHOT_SCHEMA_VERSION,
            model_sha256(decision),
            len(canonical_json_bytes(decision)),
        ),
        TemporalArtifactRole.OUTCOME_SNAPSHOT: (
            OUTCOME_SNAPSHOT_SCHEMA_VERSION,
            model_sha256(outcome_snapshot),
            len(canonical_json_bytes(outcome_snapshot)),
        ),
    }
    witnessed_at = outcome_snapshot.first_label_available_at + timedelta(days=1)
    proofs: dict[str, bytes] = {}
    receipts: list[TemporalArtifactReceipt] = []
    for index, role in enumerate(TemporalArtifactRole):
        receipt_id = f'fixture-independent-receipt-{index}'
        proof = canonical_json_bytes(
            {
                'fixture_only': True,
                'receipt_id': receipt_id,
                'claim': 'structurally independent archive proof for tests only',
            }
        )
        proofs[receipt_id] = proof
        artifact_schema, artifact_sha256, artifact_bytes = role_artifacts[role]
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=role,
                artifact_schema_version=artifact_schema,
                artifact_sha256=artifact_sha256,
                artifact_bytes=artifact_bytes,
                witnessed_at=witnessed_at,
                authority_type=TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
                authority_id='fixture-independent-archive',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://fixture-archive.invalid/{receipt_id}',
            )
        )
    admission = TemporalAdmissionEnvelope(
        admission_id=f'fixture-tier-b:{bundle.manifest.episode_id}',
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        source_tier=TemporalSourceTier.TIER_B,
        admitted_use=TemporalAdmissionUse.RETROSPECTIVE_RESEARCH,
        provenance_basis=TemporalProvenanceBasis.INDEPENDENT_ARCHIVE,
        decision_snapshot=decision,
        outcome_snapshot=outcome_snapshot,
        receipts=tuple(receipts),
        admitted_at=witnessed_at + timedelta(seconds=1),
    )
    return (
        TemporalAdmissionMaterial(
            admission=admission,
            protocol_artifacts=protocol_artifacts,
            raw_outcome_source=raw_outcome_source,
            label_derivation_audit=label_derivation_audit,
            receipt_proofs=proofs,
        ),
        source_audit,
    )


def _fixture_temporal_receipt_verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    parsed = json.loads(proof)
    return (
        receipt.authority_id == 'fixture-independent-archive'
        and parsed.get('fixture_only') is True
        and parsed.get('receipt_id') == receipt.receipt_id
    )


def _fixture_source_material_verifier(
    episode_id: str,
    bundle: EpisodeBundle,
    material: TemporalAdmissionMaterial,
    source_audit: bytes,
    source_files: Mapping[str, bytes],
    case_universe_entry: CaseUniverseEntry,
) -> bool:
    audit = json.loads(source_audit)
    return (
        audit.get('fixture_only') is True
        and audit.get('episode_id') == episode_id
        and bundle.manifest.episode_id == episode_id
        and material.label_derivation_audit == source_audit
        and case_universe_entry.lineage_group_id == bundle.manifest.lineage_group_id
        and source_files.get('archive/exact-source.json') == b'fictional exact pre-cutoff archive bytes'
    )


def _case_inventory(
    bundles: tuple[EpisodeBundle, ...],
) -> tuple[CaseUniverseManifest, bytes, CaseSelectionAudit, bytes]:
    proof = canonical_json_bytes({'fixture_only': True, 'claim': 'sealed case universe for tests'})
    verifier_policy = canonical_json_bytes(
        {
            'fixture_only': True,
            'selection_rule': 'admit complete observed cases; report every omission',
            'source_verifier': 'fixture-source-verifier-v1',
            'temporal_verifier': 'fixture-temporal-verifier-v1',
        }
    )
    entries = tuple(
        sorted(
            (
                *(
                    CaseUniverseEntry(
                        case_id=f'case-{bundle.manifest.episode_id}',
                        lineage_group_id=bundle.manifest.lineage_group_id,
                        disposition=CaseUniverseDisposition.PREELIGIBLE,
                        decision_package_sha256=hashlib.sha256(
                            f'decision:{bundle.manifest.episode_id}'.encode()
                        ).hexdigest(),
                    )
                    for bundle in bundles
                ),
                CaseUniverseEntry(
                    case_id='case-excluded-predefined',
                    lineage_group_id='iedb-fictional-lineage-excluded',
                    disposition=CaseUniverseDisposition.EXCLUDED_PREDEFINED,
                    reason_code='failed-preoutcome-license-rule',
                ),
                CaseUniverseEntry(
                    case_id='case-quarantined-contamination',
                    lineage_group_id='iedb-fictional-lineage-contamination',
                    disposition=CaseUniverseDisposition.PREELIGIBLE,
                    decision_package_sha256='d' * 64,
                ),
                CaseUniverseEntry(
                    case_id='case-unscored-missing',
                    lineage_group_id='iedb-fictional-lineage-missing',
                    disposition=CaseUniverseDisposition.PREELIGIBLE,
                    decision_package_sha256='c' * 64,
                ),
            ),
            key=lambda entry: entry.case_id,
        )
    )
    universe_id = 'fixture-complete-case-universe-v1'
    eligibility_hash = hashlib.sha256(b'fixture pre-outcome eligibility protocol').hexdigest()
    content_hash = case_universe_content_sha256(
        universe_id=universe_id,
        eligibility_protocol_sha256=eligibility_hash,
        entries=entries,
    )
    universe = CaseUniverseManifest(
        universe_id=universe_id,
        eligibility_protocol_sha256=eligibility_hash,
        entries=entries,
        universe_content_sha256=content_hash,
        seal=CaseUniverseSeal(
            universe_content_sha256=content_hash,
            witnessed_at=min(bundle.manifest.decision_at for bundle in bundles) - timedelta(days=1),
            authority_type=TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
            authority_id='fixture-independent-archive',
            proof_sha256=hashlib.sha256(proof).hexdigest(),
            proof_bytes=len(proof),
            verification_uri='https://fixture-archive.invalid/case-universe-v1',
        ),
    )
    records: list[CaseSelectionRecord] = []
    for bundle in bundles:
        records.append(
            CaseSelectionRecord(
                case_id=f'case-{bundle.manifest.episode_id}',
                disposition=CaseSelectionDisposition.ADMITTED,
                episode_id=bundle.manifest.episode_id,
                manifest_sha256=bundle.manifest_sha256,
                panel_count=len(bundle.manifest.candidate_ids),
                observed_count=len(bundle.manifest.candidate_ids),
                missing_count=0,
                conflict_count=0,
            )
        )
    records.extend(
        (
            CaseSelectionRecord(
                case_id='case-excluded-predefined',
                disposition=CaseSelectionDisposition.EXCLUDED_PREDEFINED,
                panel_count=0,
                observed_count=0,
                missing_count=0,
                conflict_count=0,
                reason_code='failed-preoutcome-license-rule',
            ),
            CaseSelectionRecord(
                case_id='case-quarantined-contamination',
                disposition=CaseSelectionDisposition.QUARANTINED_CONTAMINATION,
                panel_count=4,
                observed_count=4,
                missing_count=0,
                conflict_count=0,
                reason_code='confirmed-answer-bearing-artifact',
            ),
            CaseSelectionRecord(
                case_id='case-unscored-missing',
                disposition=CaseSelectionDisposition.UNSCORED_MISSING,
                panel_count=4,
                observed_count=3,
                missing_count=1,
                conflict_count=0,
                reason_code='incomplete-later-outcome-panel',
            ),
        )
    )
    selection = CaseSelectionAudit(
        case_universe_sha256=case_universe_sha256(universe),
        selection_policy_sha256=hashlib.sha256(verifier_policy).hexdigest(),
        records=tuple(sorted(records, key=lambda record: record.case_id)),
    )
    return universe, proof, selection, verifier_policy


def _fixture_case_universe_verifier(
    universe: CaseUniverseManifest,
    proof: bytes,
    selection: CaseSelectionAudit,
    verifier_policy: bytes,
) -> bool:
    return (
        universe.seal.authority_id == 'fixture-independent-archive'
        and json.loads(proof).get('fixture_only') is True
        and json.loads(verifier_policy).get('fixture_only') is True
        and selection.inventory_complete
    )


def _contamination_judge(judge_id: str) -> PinnedLlmJudge:
    return PinnedLlmJudge(
        judge_id=judge_id,
        provider=f'fixture-provider-{judge_id}',
        model_id=f'fixture-model-{judge_id}',
        model_revision='fixture-revision-v1',
        system_fingerprint=f'fixture-fingerprint-{judge_id}',
        system_manifest_sha256=hashlib.sha256(f'system:{judge_id}'.encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(f'prompt:{judge_id}'.encode()).hexdigest(),
        config_sha256=hashlib.sha256(f'config:{judge_id}'.encode()).hexdigest(),
    )


def _contamination_policy() -> ContaminationAuditPolicy:
    return ContaminationAuditPolicy(
        policy_id='fixture-retrospective-contamination-screen-v1',
        retrieval=ExactRetrievalConfig(
            ngram_tokens=8,
            minimum_ngram_bytes=32,
            maximum_candidates=1_000,
        ),
        calibration=CalibrationPolicy(
            minimum_canary_count=5,
            minimum_negative_control_count=5,
            minimum_canary_recall=0.8,
            maximum_false_positive_rate=0.2,
        ),
        judges=(
            _contamination_judge('judge-alpha'),
            _contamination_judge('judge-beta'),
        ),
    )


def _contamination_manifest(
    universe: CaseUniverseManifest,
    selection: CaseSelectionAudit,
    bundles: tuple[EpisodeBundle, ...],
    policy: ContaminationAuditPolicy,
) -> ContaminationAuditManifest:
    selection_by_case = {record.case_id: record for record in selection.records}
    bundle_by_episode = {bundle.manifest.episode_id: bundle for bundle in bundles}
    audits = []
    screened_at = datetime(2026, 7, 13, tzinfo=timezone.utc)
    calibration = JudgeCalibrationResult(
        canary_count=5,
        canary_detected_count=5,
        negative_control_count=5,
        false_positive_count=0,
    )
    clear_output = LlmJudgeOutput(verdict=JudgeVerdict.CLEAR, calibration=calibration)
    for entry in universe.entries:
        if entry.disposition != CaseUniverseDisposition.PREELIGIBLE:
            continue
        record = selection_by_case.get(entry.case_id)
        if record is not None and record.disposition == CaseSelectionDisposition.ADMITTED:
            assert record.episode_id is not None
            assert record.manifest_sha256 is not None
            bundle = bundle_by_episode[record.episode_id]
            episode_id = record.episode_id
            manifest_sha256 = record.manifest_sha256
            public_payload = model_facing_payload_bytes(bundle, variant=PromptVariant.FULL)
        else:
            episode_id = f'unscored-{entry.case_id}'
            manifest_sha256 = hashlib.sha256(episode_id.encode()).hexdigest()
            public_payload = f'private audit view for {entry.case_id}'.encode()
        comparison_payloads = {'protected-outcome': f'ZXQ_PRIVATE_OUTCOME_{entry.case_id}_7Y9'.encode()}
        identifiers: tuple[IdentifierNeedle, ...] = ()
        if record is not None and record.disposition == CaseSelectionDisposition.QUARANTINED_CONTAMINATION:
            public_payload += b' LEAK-ID-123'
            comparison_payloads = {'protected-outcome': b'ZXQ_PRIVATE_OUTCOME LEAK-ID-123'}
            identifiers = (
                IdentifierNeedle(
                    identifier_id='fixture-leaked-identifier',
                    identifier_type='outcome-only-identifier',
                    value='LEAK-ID-123',
                    reference_artifact_id='protected-outcome',
                ),
            )
        assert entry.decision_package_sha256 is not None
        audit_input = make_audit_input(
            case_id=entry.case_id,
            episode_id=episode_id,
            decision_package_sha256=entry.decision_package_sha256,
            episode_manifest_sha256=manifest_sha256,
            public_artifact_id=f'model-facing:{episode_id}:full',
            public_payload=public_payload,
            comparison_payloads=comparison_payloads,
        )
        judge_runs = tuple(
            make_llm_audit_run(
                run_id=f'{entry.case_id}:{judge.judge_id}',
                judge=judge,
                audit_input=audit_input,
                output=clear_output,
                started_at=screened_at,
                finished_at=screened_at + timedelta(seconds=index + 1),
            )
            for index, judge in enumerate(policy.judges)
        )
        audits.append(
            build_contamination_audit(
                audit_id=f'audit:{entry.case_id}',
                audit_input=audit_input,
                policy=policy,
                public_payload=public_payload,
                comparison_payloads=comparison_payloads,
                judge_runs=judge_runs,
                screened_at=screened_at,
                identifiers=identifiers,
            )
        )
    return make_audit_manifest(
        manifest_id='fixture-complete-contamination-audit-v1',
        case_universe_sha256=case_universe_sha256(universe),
        policy=policy,
        audits=audits,
    )


def _fixture_contamination_audit_verifier(
    audit: ContaminationAuditManifest,
    policy: ContaminationAuditPolicy,
    universe: CaseUniverseManifest,
    selection: CaseSelectionAudit,
) -> bool:
    return (
        audit.inventory_complete
        and audit.case_universe_sha256 == case_universe_sha256(universe)
        and audit.policy_sha256 == contamination_model_sha256(policy)
        and len(audit.audits)
        == sum(entry.disposition == CaseUniverseDisposition.PREELIGIBLE for entry in universe.entries)
        and selection.inventory_complete
    )


class SyntheticIedbReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.episode = _build_iedb_episode(self.root / 'episode')
        self.policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)

    def _build_release(self, suffix: str = ''):
        return build_synthetic_integration_release(
            release_id='iedb-synthetic-pilot-v0',
            challenge_id='iedb-synthetic-pilot-challenge-v0',
            suite_id='iedb-synthetic-pilot-suite-v0',
            episode_dirs=(self.episode,),
            policy=self.policy,
            receipt_key_id=receipt_key_id(_RECEIPT_KEY),
            public_output_dir=self.root / f'public{suffix}',
            private_output_dir=self.root / f'private{suffix}',
        )

    def test_packages_a_tier_c_public_private_release_without_label_leakage(self) -> None:
        built = self._build_release()
        loaded = load_release(
            built.public_root,
            built.private_root,
            expected_public_release_sha256=built.public_manifest_sha256,
        )

        self.assertEqual(loaded.public_manifest.purpose, ReleasePurpose.SYNTHETIC_INTEGRATION)
        self.assertFalse(loaded.public_manifest.sealed_eligible)
        self.assertFalse(loaded.private_manifest.split_inventory_complete)
        self.assertEqual(loaded.split_admission.episodes[0].split, Split.TEST)
        self.assertIsNotNone(loaded.challenge.admission)
        assert loaded.challenge.admission is not None
        self.assertEqual(loaded.challenge.admission.purpose, ReleasePurpose.SYNTHETIC_INTEGRATION)
        self.assertFalse(loaded.challenge.admission.split_inventory_complete)
        self.assertEqual(loaded.temporal_admissions[0].source_tier, TemporalSourceTier.TIER_C)
        self.assertEqual(loaded.temporal_admissions[0].admitted_use, TemporalAdmissionUse.TRAIN_DEBUG)
        self.assertEqual(
            loaded.temporal_admissions[0].provenance_basis,
            TemporalProvenanceBasis.RETROSPECTIVE_RECONSTRUCTION,
        )

        public_files = {
            path.relative_to(built.public_root).as_posix() for path in built.public_root.rglob('*') if path.is_file()
        }
        self.assertEqual(
            public_files,
            {
                'release.json',
                'policy.json',
                'challenge/admission.json',
                'challenge/challenge.json',
                'challenge/suite.json',
                'challenge/episodes/000000.json',
            },
        )
        public_bytes = b''.join(path.read_bytes() for path in sorted(built.public_root.rglob('*')) if path.is_file())
        for forbidden in (
            _LABEL_KEY.hex().encode('ascii'),
            b'Fictional held-out cellular validation',
            b'Positive-High',
            b'"outcome":1',
            b'"relevance_grade":1',
        ):
            self.assertNotIn(forbidden, public_bytes)

        private_episode = built.private_root / 'episodes' / '000000'
        self.assertEqual(
            (private_episode / 'private' / 'label_commitment_key.hex').read_text(encoding='ascii').strip(),
            _LABEL_KEY.hex(),
        )
        self.assertTrue((private_episode / 'private' / 'outcomes.jsonl').is_file())
        self.assertTrue((private_episode / 'private' / 'ranking_labels.jsonl').is_file())

    def test_release_identity_is_deterministic_for_identical_inputs(self) -> None:
        first = self._build_release('-first')
        second = self._build_release('-second')

        self.assertEqual(first.public_manifest_sha256, second.public_manifest_sha256)
        self.assertEqual(first.public_manifest, second.public_manifest)
        self.assertEqual(first.private_manifest, second.private_manifest)
        self.assertEqual(first.challenge.manifest_sha256, second.challenge.manifest_sha256)

    def test_rejects_train_shaped_episode_instead_of_overclaiming_a_sealed_test(self) -> None:
        train_episode = _build_iedb_episode(self.root / 'train-episode', split=Split.TRAIN)

        with self.assertRaisesRegex(ValueError, 'test split'):
            build_synthetic_integration_release(
                release_id='invalid-train-release',
                challenge_id='invalid-train-challenge',
                suite_id='invalid-train-suite',
                episode_dirs=(train_episode,),
                policy=self.policy,
                receipt_key_id=receipt_key_id(_RECEIPT_KEY),
                public_output_dir=self.root / 'public',
                private_output_dir=self.root / 'private',
            )

        self.assertFalse((self.root / 'public').exists())
        self.assertFalse((self.root / 'private').exists())

    def test_rejects_official_isolation_claim_for_synthetic_integration_release(self) -> None:
        with self.assertRaisesRegex(ValueError, 'development policy'):
            build_synthetic_integration_release(
                release_id='overclaimed-release',
                challenge_id='overclaimed-challenge',
                suite_id='overclaimed-suite',
                episode_dirs=(self.episode,),
                policy=RunnerPolicy(),
                receipt_key_id=receipt_key_id(_RECEIPT_KEY),
                public_output_dir=self.root / 'public',
                private_output_dir=self.root / 'private',
            )

    def test_rejects_a_wrong_preregistered_public_release_hash(self) -> None:
        built = self._build_release()

        with self.assertRaisesRegex(ReleaseIntegrityError, 'preregistered hash'):
            load_release(
                built.public_root,
                built.private_root,
                expected_public_release_sha256='f' * 64,
            )

    def test_rejects_tampering_with_a_bound_private_label_file(self) -> None:
        built = self._build_release()
        outcomes = built.private_root / 'episodes' / '000000' / 'private' / 'outcomes.jsonl'
        outcomes.write_bytes(outcomes.read_bytes() + b'\n')

        with self.assertRaisesRegex(ReleaseIntegrityError, 'does not match its binding'):
            load_release(
                built.public_root,
                built.private_root,
                expected_public_release_sha256=built.public_manifest_sha256,
            )

    def test_rejects_an_extra_public_file(self) -> None:
        built = self._build_release()
        (built.public_root / 'unexpected.json').write_text('{}', encoding='utf-8')

        with self.assertRaisesRegex(ReleaseIntegrityError, 'allowlist'):
            load_release(
                built.public_root,
                built.private_root,
                expected_public_release_sha256=built.public_manifest_sha256,
            )

    def test_rejects_an_unexpected_empty_private_directory(self) -> None:
        built = self._build_release()
        (built.private_root / 'unexpected-empty').mkdir()

        with self.assertRaisesRegex(ReleaseIntegrityError, 'directory allowlist'):
            load_release(
                built.public_root,
                built.private_root,
                expected_public_release_sha256=built.public_manifest_sha256,
            )

    def test_installs_public_and_private_release_trees_with_explicit_permissions(self) -> None:
        built = self._build_release()

        for path in (built.public_root, *built.public_root.rglob('*')):
            expected_mode = 0o755 if path.is_dir() else 0o644
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode),
                expected_mode,
                f'unexpected public release mode for {path.relative_to(built.public_root)}',
            )
        for path in (built.private_root, *built.private_root.rglob('*')):
            expected_mode = 0o700 if path.is_dir() else 0o600
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode),
                expected_mode,
                f'unexpected private release mode for {path.relative_to(built.private_root)}',
            )

    def test_rejects_source_audit_that_disagrees_with_episode_binding(self) -> None:
        built = self._build_release()
        source_audit_path = built.private_root / 'source-audits' / '000000.json'
        forged_audit = canonical_json_bytes({'audit_type': 'forged-but-individually-bound'})
        source_audit_path.write_bytes(forged_audit)

        private_manifest_path = built.private_root / 'package.json'
        private_manifest = json.loads(private_manifest_path.read_bytes())
        source_file_binding = next(
            binding for binding in private_manifest['files'] if binding['path'] == 'source-audits/000000.json'
        )
        source_file_binding['sha256'] = hashlib.sha256(forged_audit).hexdigest()
        source_file_binding['byte_count'] = len(forged_audit)
        private_manifest_bytes = canonical_json_bytes(private_manifest)
        private_manifest_path.write_bytes(private_manifest_bytes)

        public_manifest_path = built.public_root / 'release.json'
        public_manifest_data = json.loads(public_manifest_path.read_bytes())
        public_manifest_data['private_package_sha256'] = hashlib.sha256(private_manifest_bytes).hexdigest()
        public_manifest = PublicReleaseManifest.model_validate_json(canonical_json_bytes(public_manifest_data))
        public_manifest_path.write_bytes(canonical_json_bytes(public_manifest))

        with self.assertRaisesRegex(ReleaseIntegrityError, 'outcome artifacts|source audit'):
            load_release(
                built.public_root,
                built.private_root,
                expected_public_release_sha256=public_release_sha256(public_manifest),
            )

    def test_failed_second_output_reservation_leaves_no_public_staging_directory(self) -> None:
        private_target = self.root / 'private'
        private_target.mkdir()
        before = {path.name for path in self.root.iterdir()}

        with self.assertRaisesRegex(ValueError, 'private release output already exists'):
            build_synthetic_integration_release(
                release_id='atomicity-test',
                challenge_id='atomicity-challenge',
                suite_id='atomicity-suite',
                episode_dirs=(self.episode,),
                policy=self.policy,
                receipt_key_id=receipt_key_id(_RECEIPT_KEY),
                public_output_dir=self.root / 'public',
                private_output_dir=private_target,
            )

        self.assertEqual({path.name for path in self.root.iterdir()}, before)

    def test_rejects_ancestor_or_descendant_public_private_outputs_without_creating_them(self) -> None:
        cases = (
            ('public-ancestor', Path('release'), Path('release/private')),
            ('private-ancestor', Path('release/public'), Path('release')),
        )
        for case_name, public_relative, private_relative in cases:
            with self.subTest(case_name=case_name):
                case_root = self.root / case_name
                public_output = case_root / public_relative
                private_output = case_root / private_relative

                with self.assertRaisesRegex(ValueError, 'separate, non-overlapping'):
                    build_synthetic_integration_release(
                        release_id=f'overlap-{case_name}',
                        challenge_id=f'overlap-{case_name}-challenge',
                        suite_id=f'overlap-{case_name}-suite',
                        episode_dirs=(self.episode,),
                        policy=self.policy,
                        receipt_key_id=receipt_key_id(_RECEIPT_KEY),
                        public_output_dir=public_output,
                        private_output_dir=private_output,
                    )

                self.assertFalse(case_root.exists())
                self.assertFalse(public_output.exists())
                self.assertFalse(private_output.exists())


class RetrospectiveResearchReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.selected = _make_non_synthetic_episode(self.root / 'selected')
        self.inventory_dev = _make_non_synthetic_episode(
            self.root / 'inventory-dev',
            split=Split.DEV,
            episode_id='iedb-fictional-cohort-inventory-dev',
            lineage_group_id='iedb-fictional-lineage-inventory-dev',
        )
        self.material, self.source_audit = _tier_b_material(self.selected)

    def _build_release(
        self,
        suffix: str = '',
        *,
        selected: tuple[Path, ...] | None = None,
        inventory: tuple[Path, ...] | None = None,
        temporal_materials: dict[str, TemporalAdmissionMaterial] | None = None,
        source_audits: dict[str, bytes] | None = None,
        policy: RunnerPolicy | None = None,
        extra_private_files: dict[str, dict[str, bytes]] | None = None,
        temporal_receipt_verifier=_fixture_temporal_receipt_verifier,
        source_material_verifier=_fixture_source_material_verifier,
        case_inventory_parts: tuple[CaseUniverseManifest, bytes, CaseSelectionAudit, bytes] | None = None,
        case_universe_verifier=_fixture_case_universe_verifier,
        contamination_parts: tuple[ContaminationAuditPolicy, ContaminationAuditManifest] | None = None,
        contamination_audit_verifier=_fixture_contamination_audit_verifier,
    ):
        selected_paths = selected or (self.selected,)
        inventory_paths = inventory or (self.selected, self.inventory_dev)
        inventory_bundles = tuple(EpisodeBundle.load(path, include_private=True) for path in inventory_paths)
        episode_id = EpisodeBundle.load(selected_paths[0], include_private=True).manifest.episode_id
        universe, universe_proof, selection_audit, verifier_policy = case_inventory_parts or _case_inventory(
            inventory_bundles
        )
        if contamination_parts is None:
            contamination_policy = _contamination_policy()
            contamination_manifest = _contamination_manifest(
                universe,
                selection_audit,
                inventory_bundles,
                contamination_policy,
            )
        else:
            contamination_policy, contamination_manifest = contamination_parts
        return build_retrospective_research_release(
            release_id='iedb-fictional-tier-b-v0',
            challenge_id='iedb-fictional-tier-b-challenge-v0',
            suite_id='iedb-fictional-tier-b-suite-v0',
            selected_episode_dirs=selected_paths,
            complete_inventory_episode_dirs=inventory_paths,
            temporal_materials=temporal_materials or {episode_id: self.material},
            source_audits=source_audits or {episode_id: self.source_audit},
            case_universe=universe,
            case_universe_proof=universe_proof,
            case_selection_audit=selection_audit,
            verifier_policy=verifier_policy,
            contamination_policy=contamination_policy,
            contamination_audit_manifest=contamination_manifest,
            temporal_receipt_verifier=temporal_receipt_verifier,
            source_material_verifier=source_material_verifier,
            case_universe_verifier=case_universe_verifier,
            contamination_audit_verifier=contamination_audit_verifier,
            policy=policy or RunnerPolicy(),
            receipt_key_id=receipt_key_id(_RECEIPT_KEY),
            public_output_dir=self.root / f'tier-b-public{suffix}',
            private_output_dir=self.root / f'tier-b-private{suffix}',
            extra_private_files=extra_private_files
            or {episode_id: {'archive/exact-source.json': b'fictional exact pre-cutoff archive bytes'}},
        )

    def test_packages_selected_tier_b_subset_against_complete_inventory(self) -> None:
        built = self._build_release()
        loaded = load_release(
            built.public_root,
            built.private_root,
            expected_public_release_sha256=built.public_manifest_sha256,
        )

        self.assertEqual(loaded.public_manifest.purpose, ReleasePurpose.RETROSPECTIVE_RESEARCH)
        self.assertFalse(loaded.public_manifest.sealed_eligible)
        self.assertEqual(loaded.policy.required_isolation, IsolationTier.OFFICIAL)
        self.assertTrue(loaded.private_manifest.split_inventory_complete)
        self.assertEqual(len(loaded.bundles), 1)
        self.assertEqual(len(loaded.split_admission.episodes), 2)
        self.assertIsNotNone(loaded.case_universe)
        self.assertIsNotNone(loaded.case_selection_audit)
        assert loaded.case_selection_audit is not None
        self.assertIn(
            CaseSelectionDisposition.UNSCORED_MISSING,
            {record.disposition for record in loaded.case_selection_audit.records},
        )
        self.assertIn(
            CaseSelectionDisposition.QUARANTINED_CONTAMINATION,
            {record.disposition for record in loaded.case_selection_audit.records},
        )
        self.assertEqual(loaded.temporal_admissions[0].source_tier, TemporalSourceTier.TIER_B)
        self.assertEqual(
            loaded.temporal_admissions[0].admitted_use,
            TemporalAdmissionUse.RETROSPECTIVE_RESEARCH,
        )
        self.assertGreater(
            loaded.temporal_admissions[0].receipts[2].witnessed_at,
            loaded.bundles[0].manifest.decision_at,
        )
        self.assertEqual(
            (built.private_root / 'source-materials/000000/archive/exact-source.json').read_bytes(),
            b'fictional exact pre-cutoff archive bytes',
        )
        self.assertIsNotNone(loaded.contamination_policy)
        self.assertIsNotNone(loaded.contamination_audit_manifest)
        assert loaded.contamination_audit_manifest is not None
        admitted_case_ids = {
            record.case_id
            for record in loaded.case_selection_audit.records
            if record.disposition == CaseSelectionDisposition.ADMITTED
        }
        self.assertEqual(
            {
                audit.audit_input.case_id
                for audit in loaded.contamination_audit_manifest.audits
                if audit.disposition == AuditDisposition.PASS
            }
            & admitted_case_ids,
            admitted_case_ids,
        )
        public_bytes = b''.join(path.read_bytes() for path in built.public_root.rglob('*') if path.is_file())
        self.assertIn(b'fixture-retrospective-contamination-screen-v1', public_bytes)
        self.assertNotIn(b'LEAK-ID-123', public_bytes)
        self.assertNotIn(b'ZXQ_PRIVATE_OUTCOME', public_bytes)

    def test_builder_requires_trusted_temporal_and_source_verifiers(self) -> None:
        with self.subTest(case='temporal'), self.assertRaisesRegex(ValueError, 'receipt verifier rejected'):
            self._build_release('-bad-temporal-verifier', temporal_receipt_verifier=lambda _receipt, _proof: False)
        with self.subTest(case='source'), self.assertRaisesRegex(ValueError, 'source-material verifier rejected'):
            self._build_release(
                '-bad-source-verifier',
                source_material_verifier=lambda _episode_id, _bundle, _material, _audit, _files, _case: False,
            )
        with self.subTest(case='case-universe'), self.assertRaisesRegex(ValueError, 'case-universe verifier rejected'):
            self._build_release(
                '-bad-case-universe-verifier',
                case_universe_verifier=lambda _universe, _proof, _selection, _policy: False,
            )
        with (
            self.subTest(case='contamination-audit'),
            self.assertRaisesRegex(ValueError, 'contamination-audit verifier rejected'),
        ):
            self._build_release(
                '-bad-contamination-verifier',
                contamination_audit_verifier=lambda _audit, _policy, _universe, _selection: False,
            )

    def test_rejects_incomplete_nonpassing_or_wrong_view_contamination_audit(self) -> None:
        inventory_bundles = (
            EpisodeBundle.load(self.selected, include_private=True),
            EpisodeBundle.load(self.inventory_dev, include_private=True),
        )
        universe, proof, selection, verifier_policy = _case_inventory(inventory_bundles)
        policy = _contamination_policy()
        manifest = _contamination_manifest(universe, selection, inventory_bundles, policy)

        incomplete = manifest.model_copy(update={'audits': manifest.audits[:-1]})
        with (
            self.subTest(case='incomplete'),
            self.assertRaisesRegex(ValueError, 'cover every preeligible universe case'),
        ):
            self._build_release(
                '-incomplete-contamination-audit',
                case_inventory_parts=(universe, proof, selection, verifier_policy),
                contamination_parts=(policy, incomplete),
            )

        selected_case_id = f'case-{self.material.admission.episode_id}'
        selected_audit = next(audit for audit in manifest.audits if audit.audit_input.case_id == selected_case_id)
        nonpassing = selected_audit.model_copy(
            update={
                'disposition': AuditDisposition.MANUAL_REVIEW,
                'reason_codes': (AuditReasonCode.JUDGE_SUSPICIOUS,),
            }
        )
        nonpassing_manifest = manifest.model_copy(
            update={
                'audits': tuple(
                    nonpassing if audit.audit_input.case_id == selected_case_id else audit for audit in manifest.audits
                )
            }
        )
        with (
            self.subTest(case='nonpassing'),
            self.assertRaisesRegex(ValueError, 'does not have a passing contamination audit'),
        ):
            self._build_release(
                '-nonpassing-contamination-audit',
                case_inventory_parts=(universe, proof, selection, verifier_policy),
                contamination_parts=(policy, nonpassing_manifest),
            )

        forged_public_binding = selected_audit.audit_input.public_artifact.model_copy(update={'sha256': 'f' * 64})
        forged_input = selected_audit.audit_input.model_copy(update={'public_artifact': forged_public_binding})
        forged_audit = selected_audit.model_copy(
            update={
                'audit_input': forged_input,
                'audit_input_sha256': contamination_model_sha256(forged_input),
            }
        )
        forged_manifest = manifest.model_copy(
            update={
                'audits': tuple(
                    forged_audit if audit.audit_input.case_id == selected_case_id else audit
                    for audit in manifest.audits
                )
            }
        )
        with self.subTest(case='wrong-final-view'), self.assertRaisesRegex(ValueError, 'final model-facing view'):
            self._build_release(
                '-wrong-contamination-view',
                case_inventory_parts=(universe, proof, selection, verifier_policy),
                contamination_parts=(policy, forged_manifest),
            )

    def test_rejects_silent_omission_from_case_selection_audit(self) -> None:
        inventory_bundles = (
            EpisodeBundle.load(self.selected, include_private=True),
            EpisodeBundle.load(self.inventory_dev, include_private=True),
        )
        universe, proof, selection, policy = _case_inventory(inventory_bundles)
        incomplete = selection.model_copy(update={'records': selection.records[:-1]})

        with self.assertRaisesRegex(ValueError, 'cover every universe case'):
            self._build_release(
                '-incomplete-case-audit',
                case_inventory_parts=(universe, proof, incomplete, policy),
            )

    def test_rejects_unbound_case_proof_or_verifier_policy(self) -> None:
        inventory_bundles = (
            EpisodeBundle.load(self.selected, include_private=True),
            EpisodeBundle.load(self.inventory_dev, include_private=True),
        )
        universe, proof, selection, policy = _case_inventory(inventory_bundles)

        with self.subTest(case='proof'), self.assertRaisesRegex(ValueError, 'proof does not match'):
            self._build_release(
                '-bad-case-proof',
                case_inventory_parts=(universe, proof + b'tamper', selection, policy),
            )
        with self.subTest(case='policy'), self.assertRaisesRegex(ValueError, 'committed verifier policy'):
            self._build_release(
                '-bad-verifier-policy',
                case_inventory_parts=(universe, proof, selection, policy + b'tamper'),
            )

    def test_rejects_selected_episode_absent_from_complete_inventory(self) -> None:
        with self.assertRaisesRegex(ValueError, 'absent from the complete split inventory'):
            self._build_release('-missing', inventory=(self.inventory_dev,))

    def test_rejects_synthetic_or_non_test_selected_episodes(self) -> None:
        synthetic = _build_iedb_episode(self.root / 'synthetic-selected')
        synthetic_material, synthetic_audit = _tier_b_material(synthetic)
        synthetic_id = EpisodeBundle.load(synthetic, include_private=True).manifest.episode_id
        with self.subTest(case='synthetic'), self.assertRaisesRegex(ValueError, 'synthetic'):
            self._build_release(
                '-synthetic',
                selected=(synthetic,),
                inventory=(synthetic,),
                temporal_materials={synthetic_id: synthetic_material},
                source_audits={synthetic_id: synthetic_audit},
                extra_private_files={},
            )

        dev_material, dev_audit = _tier_b_material(self.inventory_dev)
        dev_id = EpisodeBundle.load(self.inventory_dev, include_private=True).manifest.episode_id
        with self.subTest(case='dev'), self.assertRaisesRegex(ValueError, 'test split'):
            self._build_release(
                '-dev',
                selected=(self.inventory_dev,),
                inventory=(self.inventory_dev,),
                temporal_materials={dev_id: dev_material},
                source_audits={dev_id: dev_audit},
                extra_private_files={},
            )

    def test_rejects_development_policy_for_tier_b_release(self) -> None:
        with self.assertRaisesRegex(ValueError, 'official isolation policy'):
            self._build_release('-development', policy=RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT))

    def test_rejects_temporal_proof_bytes_that_do_not_match_receipt(self) -> None:
        proofs = dict(self.material.receipt_proofs)
        receipt_id = self.material.admission.receipts[0].receipt_id
        proofs[receipt_id] += b'tampered'
        bad_material = TemporalAdmissionMaterial(
            admission=self.material.admission,
            protocol_artifacts=self.material.protocol_artifacts,
            raw_outcome_source=self.material.raw_outcome_source,
            label_derivation_audit=self.material.label_derivation_audit,
            receipt_proofs=proofs,
        )
        episode_id = self.material.admission.episode_id

        with self.assertRaisesRegex(ReleaseIntegrityError, 'proof does not match receipt'):
            self._build_release('-bad-proof', temporal_materials={episode_id: bad_material})

    def test_rejects_extra_private_path_traversal_before_creating_outputs(self) -> None:
        episode_id = self.material.admission.episode_id

        with self.assertRaisesRegex(ValueError, 'normalized and relative'):
            self._build_release(
                '-path-traversal',
                extra_private_files={episode_id: {'../../escaped.json': b'forbidden'}},
            )
        self.assertFalse((self.root / 'tier-b-public-path-traversal').exists())
        self.assertFalse((self.root / 'tier-b-private-path-traversal').exists())


if __name__ == '__main__':
    unittest.main()
