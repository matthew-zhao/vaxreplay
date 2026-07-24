from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import vaxreplay.prospective_finalizer as prospective_finalizer
from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, ranking_labels_commitment
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    CaseSelectionRecord,
    case_selection_audit_sha256,
)
from vaxreplay.case_schema import LabelCommitmentScheme, Split
from vaxreplay.prospective_finalizer import (
    ProspectiveFinalizationError,
    adapt_prospective_submission,
    finalize_prospective_episode,
)
from vaxreplay.prospective_schema import (
    ProspectiveChallengeAdmission,
    ProspectiveEpisodeBinding,
    ProspectiveFinalizationBinding,
    ProspectiveSubmission,
    prospective_challenge_admission_sha256,
)
from vaxreplay.temporal_schema import (
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    OUTCOME_SNAPSHOT_SCHEMA_VERSION,
    PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
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


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _sealed_bundle() -> EpisodeBundle:
    bundle = EpisodeBundle.load(_fixture_root(), include_private=True)
    assert bundle.private_labels is not None
    assert bundle.ranking_labels is not None
    key = bytes(range(32))
    commitment = ranking_labels_commitment(
        bundle.private_labels,
        bundle.ranking_labels,
        LabelCommitmentScheme.HMAC_SHA256,
        key=key,
    )
    manifest = bundle.manifest.model_copy(
        update={
            'synthetic': False,
            'split': Split.TEST,
            'labels_sha256': commitment,
            'label_commitment_scheme': LabelCommitmentScheme.HMAC_SHA256,
            'label_commitment_key_id': hashlib.sha256(key).hexdigest(),
        }
    )
    sealed = replace(bundle, manifest=manifest, label_commitment_key=key)
    sealed.validate_integrity()
    return sealed


def _protocol_artifacts() -> dict[str, bytes]:
    return {
        'candidate_set_definition': b'fictional complete panel, eligibility, exclusion, and ordering rules',
        'evidence_acquisition_spec': b'fictional fixed sources, queries, pages, mappings, and cutoff rules',
        'outcome_adjudication_spec': b'fictional fixed endpoint, horizon, censoring, grades, and derivation rules',
    }


def _temporal_materials(
    bundle: EpisodeBundle,
) -> tuple[
    TemporalAdmissionEnvelope,
    dict[str, bytes],
    dict[str, bytes],
    bytes,
    bytes,
]:
    assert bundle.private_labels is not None
    protocols = _protocol_artifacts()
    commitments = DecisionProtocolCommitments(
        candidate_set_available_at=bundle.manifest.decision_at - timedelta(days=2),
        candidate_set_definition_sha256=hashlib.sha256(protocols['candidate_set_definition']).hexdigest(),
        evidence_acquisition_spec_sha256=hashlib.sha256(protocols['evidence_acquisition_spec']).hexdigest(),
        outcome_adjudication_spec_sha256=hashlib.sha256(protocols['outcome_adjudication_spec']).hexdigest(),
    )
    decision = build_decision_snapshot_commitment(
        DecisionTimeConfig.from_manifest(bundle.manifest),
        bundle.candidates,
        bundle.evidence,
        commitments,
    )
    raw_outcome_source = b'fictional immutable blinded outcome source'
    label_derivation_audit = b'fictional private label derivation audit'
    first_revealed: dict[tuple[str, int], object] = {}
    for outcome in bundle.private_labels.outcomes:
        key = (outcome.target_id, outcome.horizon_days)
        previous = first_revealed.get(key)
        if previous is None or outcome.revealed_at < previous:
            first_revealed[key] = outcome.revealed_at
    target_availability = tuple(
        OutcomeTargetAvailability(
            target_id=target_id,
            horizon_days=horizon_days,
            first_label_available_at=first_revealed[(target_id, horizon_days)],
        )
        for target_id, horizon_days in sorted(first_revealed)
    )
    outcome_snapshot = OutcomeSnapshotCommitment(
        episode_id=bundle.manifest.episode_id,
        labels_sha256=bundle.manifest.labels_sha256,
        label_commitment_scheme=LabelCommitmentScheme.HMAC_SHA256,
        outcome_adjudication_spec_sha256=commitments.outcome_adjudication_spec_sha256,
        raw_outcome_source_sha256=hashlib.sha256(raw_outcome_source).hexdigest(),
        raw_outcome_source_bytes=len(raw_outcome_source),
        label_derivation_audit_sha256=hashlib.sha256(label_derivation_audit).hexdigest(),
        label_derivation_audit_bytes=len(label_derivation_audit),
        target_availability=target_availability,
    )
    decision_context_bytes = canonical_json_bytes(
        {
            'decision_snapshot_bytes': len(canonical_json_bytes(decision)),
            'decision_snapshot_sha256': model_sha256(decision),
            'episode_id': bundle.manifest.episode_id,
            'schema_version': PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
            'source_captures': (
                {
                    'file_sha256': 'a' * 64,
                    'source_id': 'fictional-source',
                    'witnessed_at': (bundle.manifest.decision_at - timedelta(days=1))
                    .isoformat()
                    .replace('+00:00', 'Z'),
                },
            ),
        }
    )
    decision_context_sha256 = hashlib.sha256(decision_context_bytes).hexdigest()

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
            PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
            decision_context_sha256,
            len(decision_context_bytes),
        ),
        TemporalArtifactRole.OUTCOME_SNAPSHOT: (
            OUTCOME_SNAPSHOT_SCHEMA_VERSION,
            model_sha256(outcome_snapshot),
            len(canonical_json_bytes(outcome_snapshot)),
        ),
    }
    witnessed_at = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: commitments.candidate_set_available_at,
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: decision.latest_visible_evidence_at,
        TemporalArtifactRole.DECISION_SNAPSHOT: bundle.manifest.decision_at,
        TemporalArtifactRole.OUTCOME_SNAPSHOT: outcome_snapshot.first_label_available_at,
    }
    proofs: dict[str, bytes] = {}
    receipts: list[TemporalArtifactReceipt] = []
    for index, role in enumerate(TemporalArtifactRole):
        receipt_id = f'fictional-receipt-{index}'
        proof = f'fictional verified proof for {role.value}'.encode()
        proofs[receipt_id] = proof
        artifact_schema, artifact_sha256, artifact_bytes = role_artifacts[role]
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=role,
                artifact_schema_version=artifact_schema,
                artifact_sha256=artifact_sha256,
                artifact_bytes=artifact_bytes,
                witnessed_at=witnessed_at[role],
                authority_type=(
                    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG
                    if role != TemporalArtifactRole.OUTCOME_SNAPSHOT
                    else TemporalReceiptAuthority.SOURCE_SIGNED_VERSION
                ),
                authority_id='fictional-trusted-authority',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://transparency.invalid/{receipt_id}',
            )
        )
    admission = TemporalAdmissionEnvelope(
        admission_id='fictional-tier-a-admission',
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        source_tier=TemporalSourceTier.TIER_A,
        admitted_use=TemporalAdmissionUse.OFFICIAL_BENCHMARK,
        provenance_basis=TemporalProvenanceBasis.PROSPECTIVE_SEAL,
        decision_snapshot=decision,
        decision_context_sha256=decision_context_sha256,
        decision_context_bytes=len(decision_context_bytes),
        outcome_snapshot=outcome_snapshot,
        receipts=tuple(receipts),
        admitted_at=outcome_snapshot.first_label_available_at + timedelta(seconds=1),
    )
    return admission, proofs, protocols, raw_outcome_source, label_derivation_audit


def _verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return receipt.authority_id == 'fictional-trusted-authority' and proof.startswith(b'fictional verified proof')


def _prospective_materials(
    bundle: EpisodeBundle,
    temporal_admission: TemporalAdmissionEnvelope,
) -> tuple[ProspectiveEpisodeBinding, ProspectiveChallengeAdmission, CaseSelectionAudit]:
    assert temporal_admission.decision_context_sha256 is not None
    assert temporal_admission.decision_context_bytes is not None
    episode = ProspectiveEpisodeBinding.from_decision_snapshot(
        temporal_admission.decision_snapshot,
        decision_context_sha256=temporal_admission.decision_context_sha256,
        decision_context_bytes=temporal_admission.decision_context_bytes,
    )
    admission = ProspectiveChallengeAdmission(
        release_id='fictional-prospective-release',
        purpose='official_benchmark',
        suite_sha256='1' * 64,
        split_inventory_sha256='2' * 64,
        case_universe_sha256='3' * 64,
        verifier_policy_sha256='4' * 64,
        source_capture_policy_sha256='8' * 64,
        eligibility_protocol_sha256='5' * 64,
        attempt_policy_sha256='6' * 64,
        run_deadline_at=bundle.manifest.decision_at + timedelta(days=30),
        episodes=(episode,),
    )
    audit = CaseSelectionAudit(
        case_universe_sha256=admission.case_universe_sha256,
        selection_policy_sha256='7' * 64,
        records=(
            CaseSelectionRecord(
                case_id='fictional-case',
                disposition=CaseSelectionDisposition.ADMITTED,
                episode_id=bundle.manifest.episode_id,
                manifest_sha256=bundle.manifest_sha256,
                panel_count=len(bundle.manifest.candidate_ids),
                observed_count=len(bundle.manifest.candidate_ids),
                missing_count=0,
                conflict_count=0,
            ),
        ),
    )
    return episode, admission, audit


def _prospective_oracle(bundle: EpisodeBundle, episode: ProspectiveEpisodeBinding) -> ProspectiveSubmission:
    development_view = replace(
        bundle,
        manifest=bundle.manifest.model_copy(update={'split': Split.DEV}),
    )
    oracle = oracle_submission(development_view)
    return ProspectiveSubmission(
        episode_id=episode.episode_id,
        decision_snapshot_sha256=episode.decision_snapshot_sha256,
        ranking=tuple(oracle.ranking),
        forecasts=tuple(oracle.forecasts),
        assessments=tuple(oracle.assessments),
    )


class ProspectiveFinalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = _sealed_bundle()
        (
            self.temporal_admission,
            self.proofs,
            self.protocols,
            self.raw_outcome,
            self.derivation_audit,
        ) = _temporal_materials(self.bundle)
        (
            self.episode,
            self.prospective_admission,
            self.case_audit,
        ) = _prospective_materials(self.bundle, self.temporal_admission)
        self.finalized_at = self.temporal_admission.admitted_at + timedelta(seconds=1)

    def _finalize(
        self,
        *,
        episode: ProspectiveEpisodeBinding | None = None,
        prospective_admission: ProspectiveChallengeAdmission | None = None,
        temporal_admission: TemporalAdmissionEnvelope | None = None,
    ):
        return finalize_prospective_episode(
            prospective_admission or self.prospective_admission,
            episode or self.episode,
            self.bundle,
            temporal_admission or self.temporal_admission,
            receipt_artifacts=self.proofs,
            receipt_verifier=_verifier,
            protocol_artifacts=self.protocols,
            raw_outcome_source=self.raw_outcome,
            label_derivation_audit=self.derivation_audit,
            case_selection_audit=self.case_audit,
            case_selection_audit_commitment=case_selection_audit_sha256(self.case_audit),
            finalized_at=self.finalized_at,
        )

    def test_research_profile_finalization_preserves_non_official_purpose(self) -> None:
        research = self.prospective_admission.model_copy(update={'purpose': 'prospective_research'})
        finalization = self._finalize(prospective_admission=research)

        self.assertEqual(finalization.purpose, 'prospective_research')
        finalization.require_episode(research, self.episode)

    def test_rejects_decision_context_replay_from_different_source_lineage(self) -> None:
        receipts = list(self.temporal_admission.receipts)
        receipts[2] = receipts[2].model_copy(
            update={
                'artifact_sha256': 'f' * 64,
                'artifact_bytes': receipts[2].artifact_bytes + 1,
            }
        )
        replayed = TemporalAdmissionEnvelope.model_validate(
            {
                **self.temporal_admission.model_dump(),
                'decision_context_sha256': 'f' * 64,
                'decision_context_bytes': receipts[2].artifact_bytes,
                'receipts': tuple(receipts),
            }
        )
        with self.assertRaisesRegex(ProspectiveFinalizationError, 'prospectively admitted source lineage'):
            self._finalize(temporal_admission=replayed)

    def test_rejects_changed_decision_or_protocol(self) -> None:
        changed_config = self.episode.decision_snapshot.config.model_copy(
            update={'adjudication_version': 'post-outcome-rubric'}
        )
        changed_protocol = self.episode.decision_snapshot.protocol_commitments.model_copy(
            update={'outcome_adjudication_spec_sha256': 'f' * 64}
        )
        changed_snapshots = (
            self.episode.decision_snapshot.model_copy(update={'config': changed_config}),
            self.episode.decision_snapshot.model_copy(update={'protocol_commitments': changed_protocol}),
        )
        for changed_snapshot in changed_snapshots:
            changed_episode = ProspectiveEpisodeBinding.from_decision_snapshot(
                changed_snapshot,
                decision_context_sha256=self.episode.decision_context_sha256,
                decision_context_bytes=self.episode.decision_context_bytes,
            )
            changed_admission = self.prospective_admission.model_copy(update={'episodes': (changed_episode,)})
            with (
                self.subTest(snapshot=model_sha256(changed_snapshot)),
                self.assertRaisesRegex(
                    ProspectiveFinalizationError,
                    'changed the prospectively sealed decision or protocol',
                ),
            ):
                self._finalize(
                    episode=changed_episode,
                    prospective_admission=changed_admission,
                )

    def test_rejects_run_deadline_at_or_after_label_availability(self) -> None:
        tampered = self.prospective_admission.model_copy(
            update={
                'run_deadline_at': self.temporal_admission.outcome_snapshot.first_label_available_at,
            }
        )
        with self.assertRaisesRegex(
            ProspectiveFinalizationError,
            'run_deadline_at must precede',
        ):
            self._finalize(prospective_admission=tampered)

    def test_rejects_response_bound_to_wrong_decision(self) -> None:
        finalization = self._finalize()
        response = _prospective_oracle(self.bundle, self.episode).model_copy(
            update={'decision_snapshot_sha256': 'f' * 64}
        )
        with self.assertRaisesRegex(
            ProspectiveFinalizationError,
            'does not match the sealed decision snapshot',
        ):
            adapt_prospective_submission(
                response,
                prospective_admission=self.prospective_admission,
                prospective_episode=self.episode,
                final_bundle=self.bundle,
                finalization=finalization,
            )

    def test_fictional_oracle_join_preserves_response(self) -> None:
        finalization = self._finalize()
        response = _prospective_oracle(self.bundle, self.episode)
        adapted = adapt_prospective_submission(
            response,
            prospective_admission=self.prospective_admission,
            prospective_episode=self.episode,
            final_bundle=self.bundle,
            finalization=finalization,
        )

        self.assertEqual(adapted.manifest_sha256, self.bundle.manifest_sha256)
        self.assertEqual(tuple(adapted.ranking), response.ranking)
        self.assertEqual(tuple(adapted.forecasts), response.forecasts)
        self.assertEqual(tuple(adapted.assessments), response.assessments)

    def test_forged_self_consistent_binding_has_no_public_scoring_api(self) -> None:
        forged = ProspectiveFinalizationBinding(
            release_id=self.prospective_admission.release_id,
            purpose=self.prospective_admission.purpose,
            episode_id=self.episode.episode_id,
            prospective_admission_sha256=prospective_challenge_admission_sha256(self.prospective_admission),
            decision_snapshot_sha256=self.episode.decision_snapshot_sha256,
            decision_context_sha256=self.episode.decision_context_sha256,
            temporal_admission_sha256=model_sha256(self.temporal_admission),
            case_selection_audit_sha256=case_selection_audit_sha256(self.case_audit),
            finalized_at=self.finalized_at,
        )
        # The binding is internally consistent despite never passing through
        # ``finalize_prospective_episode`` or a receipt verifier.
        forged.require_episode(self.prospective_admission, self.episode)
        response = _prospective_oracle(self.bundle, self.episode)
        adapted = adapt_prospective_submission(
            response,
            prospective_admission=self.prospective_admission,
            prospective_episode=self.episode,
            final_bundle=self.bundle,
            finalization=forged,
        )
        self.assertEqual(adapted.manifest_sha256, self.bundle.manifest_sha256)
        self.assertNotIn('score_finalized_prospective_submission', prospective_finalizer.__all__)
        self.assertFalse(hasattr(prospective_finalizer, 'score_finalized_prospective_submission'))


if __name__ == '__main__':
    unittest.main()
