from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes, ranking_labels_commitment
from vaxreplay.case_schema import LabelCommitmentScheme, Split
from vaxreplay.temporal_schema import (
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    DECISION_SNAPSHOT_SCHEMA_VERSION,
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    OUTCOME_SNAPSHOT_SCHEMA_VERSION,
    PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
    DecisionProtocolCommitments,
    DecisionTimeConfig,
    OutcomeSnapshotCommitment,
    OutcomeTargetAvailability,
    TemporalAdmissionEnvelope,
    TemporalAdmissionError,
    TemporalAdmissionUse,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalProvenanceBasis,
    TemporalReceiptAuthority,
    TemporalSourceTier,
    build_decision_snapshot_commitment,
    model_sha256,
    require_official_temporal_admission,
    require_retrospective_temporal_admission,
)


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _sealed_bundle(*, synthetic: bool = False) -> EpisodeBundle:
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
            'synthetic': synthetic,
            'split': Split.TEST,
            'labels_sha256': commitment,
            'label_commitment_scheme': LabelCommitmentScheme.HMAC_SHA256,
            'label_commitment_key_id': hashlib.sha256(key).hexdigest(),
        }
    )
    sealed = replace(bundle, manifest=manifest, label_commitment_key=key)
    sealed.validate_integrity()
    return sealed


def _artifact_bytes() -> dict[str, bytes]:
    return {
        'candidate_set_definition': b'complete source, query, inclusion, exclusion, and ordering rules',
        'evidence_acquisition_spec': b'source versions, queries, pages, and candidate mappings',
        'outcome_adjudication_spec': b'endpoint, horizon, censoring, grades, and label derivation rules',
    }


def _build_admission(
    bundle: EpisodeBundle,
    *,
    source_tier: TemporalSourceTier = TemporalSourceTier.TIER_A,
) -> tuple[
    TemporalAdmissionEnvelope,
    dict[str, bytes],
    dict[str, bytes],
    bytes,
    bytes,
]:
    assert bundle.private_labels is not None
    protocol_artifacts = _artifact_bytes()
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
    raw_outcome_source = b'fictional raw blinded outcome source bytes'
    label_derivation_audit = b'fictional private adjudication and label derivation audit'
    availability: dict[tuple[str, int], object] = {}
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
        label_commitment_scheme=LabelCommitmentScheme.HMAC_SHA256,
        outcome_adjudication_spec_sha256=protocol.outcome_adjudication_spec_sha256,
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
    decision_artifact = (
        (
            PROSPECTIVE_DECISION_CONTEXT_SCHEMA_VERSION,
            hashlib.sha256(decision_context_bytes).hexdigest(),
            len(decision_context_bytes),
        )
        if source_tier == TemporalSourceTier.TIER_A
        else (
            DECISION_SNAPSHOT_SCHEMA_VERSION,
            model_sha256(decision),
            len(canonical_json_bytes(decision)),
        )
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
        TemporalArtifactRole.DECISION_SNAPSHOT: decision_artifact,
        TemporalArtifactRole.OUTCOME_SNAPSHOT: (
            OUTCOME_SNAPSHOT_SCHEMA_VERSION,
            model_sha256(outcome_snapshot),
            len(canonical_json_bytes(outcome_snapshot)),
        ),
    }
    witnessed_at = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: protocol.candidate_set_available_at,
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: decision.latest_visible_evidence_at,
        TemporalArtifactRole.DECISION_SNAPSHOT: bundle.manifest.decision_at,
        TemporalArtifactRole.OUTCOME_SNAPSHOT: outcome_snapshot.first_label_available_at,
    }
    proof_artifacts: dict[str, bytes] = {}
    receipts: list[TemporalArtifactReceipt] = []
    for index, role in enumerate(TemporalArtifactRole):
        proof = f'externally verifiable proof for {role.value}'.encode()
        receipt_id = f'receipt-{index}'
        proof_artifacts[receipt_id] = proof
        schema_version, artifact_sha256, artifact_size = role_artifacts[role]
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=role,
                artifact_schema_version=schema_version,
                artifact_sha256=artifact_sha256,
                artifact_bytes=artifact_size,
                witnessed_at=witnessed_at[role],
                authority_type=(
                    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG
                    if role != TemporalArtifactRole.OUTCOME_SNAPSHOT
                    else TemporalReceiptAuthority.SOURCE_SIGNED_VERSION
                ),
                authority_id='fixture-trusted-authority',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://transparency.invalid/{receipt_id}',
            )
        )

    profile = {
        TemporalSourceTier.TIER_A: (
            TemporalAdmissionUse.OFFICIAL_BENCHMARK,
            TemporalProvenanceBasis.PROSPECTIVE_SEAL,
        ),
        TemporalSourceTier.TIER_B: (
            TemporalAdmissionUse.RETROSPECTIVE_RESEARCH,
            TemporalProvenanceBasis.INDEPENDENT_ARCHIVE,
        ),
        TemporalSourceTier.TIER_C: (
            TemporalAdmissionUse.TRAIN_DEBUG,
            TemporalProvenanceBasis.RETROSPECTIVE_RECONSTRUCTION,
        ),
    }[source_tier]
    admission = TemporalAdmissionEnvelope(
        admission_id='fictional-temporal-admission',
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        source_tier=source_tier,
        admitted_use=profile[0],
        provenance_basis=profile[1],
        decision_snapshot=decision,
        decision_context_sha256=(
            hashlib.sha256(decision_context_bytes).hexdigest() if source_tier == TemporalSourceTier.TIER_A else None
        ),
        decision_context_bytes=(len(decision_context_bytes) if source_tier == TemporalSourceTier.TIER_A else None),
        outcome_snapshot=outcome_snapshot,
        receipts=tuple(receipts),
        admitted_at=outcome_snapshot.first_label_available_at + timedelta(seconds=1),
    )
    return admission, proof_artifacts, protocol_artifacts, raw_outcome_source, label_derivation_audit


def _receipt_verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return (
        receipt.authority_id == 'fixture-trusted-authority'
        and receipt.authority_type != TemporalReceiptAuthority.ORGANIZER_ATTESTATION
        and proof.startswith(b'externally verifiable proof')
    )


def _decision_context_args(admission: TemporalAdmissionEnvelope) -> dict[str, str | int]:
    assert admission.decision_context_sha256 is not None
    assert admission.decision_context_bytes is not None
    return {
        'expected_decision_context_sha256': admission.decision_context_sha256,
        'expected_decision_context_bytes': admission.decision_context_bytes,
    }


class TemporalSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = _sealed_bundle()
        (
            self.admission,
            self.proofs,
            self.protocol_artifacts,
            self.raw_outcome_source,
            self.label_derivation_audit,
        ) = _build_admission(self.bundle)

    def _require_official(self, admission: TemporalAdmissionEnvelope | None = None) -> None:
        require_official_temporal_admission(
            admission or self.admission,
            self.bundle,
            receipt_artifacts=self.proofs,
            receipt_verifier=_receipt_verifier,
            protocol_artifacts=self.protocol_artifacts,
            raw_outcome_source=self.raw_outcome_source,
            label_derivation_audit=self.label_derivation_audit,
            **_decision_context_args(self.admission),
        )

    def test_verified_tier_a_sidecar_is_officially_eligible(self) -> None:
        result = require_official_temporal_admission(
            self.admission,
            self.bundle,
            receipt_artifacts=self.proofs,
            receipt_verifier=_receipt_verifier,
            protocol_artifacts=self.protocol_artifacts,
            raw_outcome_source=self.raw_outcome_source,
            label_derivation_audit=self.label_derivation_audit,
            **_decision_context_args(self.admission),
        )

        self.assertEqual(result.source_tier, TemporalSourceTier.TIER_A)
        self.assertEqual(result.decision_snapshot.config.split, Split.TEST)
        self.assertEqual(result.schema_version, 'vaxreplay.temporal-admission.v0.2')

    def test_verified_tier_b_sidecar_is_eligible_only_for_retrospective_research(self) -> None:
        tier_b, proofs, protocols, raw, audit = _build_admission(
            self.bundle,
            source_tier=TemporalSourceTier.TIER_B,
        )

        result = require_retrospective_temporal_admission(
            tier_b,
            self.bundle,
            receipt_artifacts=proofs,
            receipt_verifier=_receipt_verifier,
            protocol_artifacts=protocols,
            raw_outcome_source=raw,
            label_derivation_audit=audit,
        )
        self.assertEqual(result.source_tier, TemporalSourceTier.TIER_B)
        with self.assertRaisesRegex(TemporalAdmissionError, 'Tier A'):
            require_official_temporal_admission(
                tier_b,
                self.bundle,
                receipt_artifacts=proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=protocols,
                raw_outcome_source=raw,
                label_derivation_audit=audit,
                expected_decision_context_sha256='0' * 64,
                expected_decision_context_bytes=1,
            )

    def test_decision_snapshot_excludes_future_labels_and_evidence(self) -> None:
        revised = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'labels_sha256': 'f' * 64}),
        )
        future_evidence = tuple(
            record for record in revised.evidence if record.available_at <= revised.manifest.decision_at
        )
        revised = replace(
            revised,
            evidence=(
                *future_evidence,
                *reversed(
                    tuple(record for record in revised.evidence if record.available_at > revised.manifest.decision_at)
                ),
            ),
        )
        snapshot = build_decision_snapshot_commitment(
            DecisionTimeConfig.from_manifest(revised.manifest),
            revised.candidates,
            revised.evidence,
            self.admission.decision_snapshot.protocol_commitments,
        )

        self.assertEqual(snapshot, self.admission.decision_snapshot)

    def test_split_rubric_and_protocol_hashes_are_frozen(self) -> None:
        changed_split = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'split': Split.DEV}),
        )
        changed_rubric = replace(
            self.bundle,
            manifest=self.bundle.manifest.model_copy(update={'adjudication_version': 'post-hoc'}),
        )

        self.assertNotEqual(
            DecisionTimeConfig.from_manifest(changed_split.manifest),
            self.admission.decision_snapshot.config,
        )
        self.assertNotEqual(
            DecisionTimeConfig.from_manifest(changed_rubric.manifest),
            self.admission.decision_snapshot.config,
        )
        bad_protocol = dict(self.protocol_artifacts)
        bad_protocol['outcome_adjudication_spec'] += b' changed after outcomes'
        with self.assertRaisesRegex(TemporalAdmissionError, 'outcome_adjudication_spec'):
            require_official_temporal_admission(
                self.admission,
                self.bundle,
                receipt_artifacts=self.proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=bad_protocol,
                raw_outcome_source=self.raw_outcome_source,
                label_derivation_audit=self.label_derivation_audit,
                **_decision_context_args(self.admission),
            )

    def test_tier_a_rejects_late_or_self_attested_decision_receipts(self) -> None:
        receipts = list(self.admission.receipts)
        receipts[2] = receipts[2].model_copy(
            update={'witnessed_at': self.bundle.manifest.decision_at + timedelta(seconds=1)}
        )
        with self.subTest(case='late'), self.assertRaisesRegex(ValidationError, 'at or before decision_at'):
            TemporalAdmissionEnvelope.model_validate({**self.admission.model_dump(), 'receipts': tuple(receipts)})

    def test_tier_a_rejects_legacy_bare_decision_snapshot_receipt(self) -> None:
        receipts = list(self.admission.receipts)
        receipts[2] = receipts[2].model_copy(
            update={
                'artifact_schema_version': DECISION_SNAPSHOT_SCHEMA_VERSION,
                'artifact_sha256': model_sha256(self.admission.decision_snapshot),
                'artifact_bytes': len(canonical_json_bytes(self.admission.decision_snapshot)),
            }
        )
        with self.assertRaisesRegex(ValidationError, 'legacy bare decision-snapshot'):
            TemporalAdmissionEnvelope.model_validate(
                {
                    **self.admission.model_dump(),
                    'decision_context_sha256': None,
                    'decision_context_bytes': None,
                    'receipts': tuple(receipts),
                }
            )
        with self.assertRaises(ValidationError):
            TemporalAdmissionEnvelope.model_validate(
                {**self.admission.model_dump(), 'schema_version': 'vaxreplay.temporal-admission.v0.1'}
            )

    def test_official_gate_rejects_self_consistent_context_replay_from_other_lineage(self) -> None:
        receipts = list(self.admission.receipts)
        receipts[2] = receipts[2].model_copy(
            update={
                'artifact_sha256': 'f' * 64,
                'artifact_bytes': receipts[2].artifact_bytes + 1,
            }
        )
        replayed = TemporalAdmissionEnvelope.model_validate(
            {
                **self.admission.model_dump(),
                'decision_context_sha256': 'f' * 64,
                'decision_context_bytes': receipts[2].artifact_bytes,
                'receipts': tuple(receipts),
            }
        )
        with self.assertRaisesRegex(TemporalAdmissionError, 'prospectively admitted source lineage'):
            self._require_official(replayed)

        receipts = list(self.admission.receipts)
        receipts[0] = receipts[0].model_copy(update={'authority_type': TemporalReceiptAuthority.ORGANIZER_ATTESTATION})
        with (
            self.subTest(case='self-attested'),
            self.assertRaisesRegex(ValidationError, 'prospective timestamp authority'),
        ):
            TemporalAdmissionEnvelope.model_validate({**self.admission.model_dump(), 'receipts': tuple(receipts)})

    def test_official_gate_revalidates_model_copy_and_verifies_receipt_bytes(self) -> None:
        promoted = self.admission.model_copy(
            update={
                'source_tier': TemporalSourceTier.TIER_B,
                'admitted_use': TemporalAdmissionUse.OFFICIAL_BENCHMARK,
            }
        )
        with (
            self.subTest(case='model-copy'),
            self.assertRaisesRegex(TemporalAdmissionError, 'invalid temporal admission'),
        ):
            self._require_official(promoted)

        bad_proofs = dict(self.proofs)
        bad_proofs['receipt-0'] += b'tampered'
        with self.subTest(case='proof-bytes'), self.assertRaisesRegex(TemporalAdmissionError, 'proof bytes'):
            require_official_temporal_admission(
                self.admission,
                self.bundle,
                receipt_artifacts=bad_proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=self.protocol_artifacts,
                raw_outcome_source=self.raw_outcome_source,
                label_derivation_audit=self.label_derivation_audit,
                **_decision_context_args(self.admission),
            )
        with self.subTest(case='verifier'), self.assertRaisesRegex(TemporalAdmissionError, 'rejected'):
            require_official_temporal_admission(
                self.admission,
                self.bundle,
                receipt_artifacts=self.proofs,
                receipt_verifier=lambda _receipt, _proof: False,
                protocol_artifacts=self.protocol_artifacts,
                raw_outcome_source=self.raw_outcome_source,
                label_derivation_audit=self.label_derivation_audit,
                **_decision_context_args(self.admission),
            )

    def test_official_gate_rejects_tier_b_synthetic_or_public_only_episodes(self) -> None:
        tier_b, proofs, protocols, raw, audit = _build_admission(
            self.bundle,
            source_tier=TemporalSourceTier.TIER_B,
        )
        with self.subTest(case='tier-b'), self.assertRaisesRegex(TemporalAdmissionError, 'Tier A'):
            require_official_temporal_admission(
                tier_b,
                self.bundle,
                receipt_artifacts=proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=protocols,
                raw_outcome_source=raw,
                label_derivation_audit=audit,
                expected_decision_context_sha256='0' * 64,
                expected_decision_context_bytes=1,
            )

        synthetic_bundle = _sealed_bundle(synthetic=True)
        synthetic_admission, proofs, protocols, raw, audit = _build_admission(synthetic_bundle)
        with self.subTest(case='synthetic'), self.assertRaisesRegex(TemporalAdmissionError, 'synthetic'):
            require_official_temporal_admission(
                synthetic_admission,
                synthetic_bundle,
                receipt_artifacts=proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=protocols,
                raw_outcome_source=raw,
                label_derivation_audit=audit,
                **_decision_context_args(synthetic_admission),
            )

        with self.subTest(case='public-only'), self.assertRaisesRegex(TemporalAdmissionError, 'private evaluator'):
            require_official_temporal_admission(
                self.admission,
                replace(self.bundle, private_labels=None, ranking_labels=None),
                receipt_artifacts=self.proofs,
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=self.protocol_artifacts,
                raw_outcome_source=self.raw_outcome_source,
                label_derivation_audit=self.label_derivation_audit,
                **_decision_context_args(self.admission),
            )

    def test_outcome_time_must_match_private_labels_and_horizon(self) -> None:
        target = self.admission.outcome_snapshot.target_availability[0]
        too_early = target.model_copy(
            update={'first_label_available_at': self.bundle.manifest.decision_at + timedelta(days=1)}
        )
        bad_outcome = self.admission.outcome_snapshot.model_copy(update={'target_availability': (too_early,)})

        with self.assertRaisesRegex(ValidationError, 'forecast horizon'):
            TemporalAdmissionEnvelope.model_validate(
                {
                    **self.admission.model_dump(),
                    'outcome_snapshot': bad_outcome,
                    'receipts': self.admission.receipts,
                }
            )

    def test_tier_c_can_record_an_incomplete_receipt_chain_but_is_not_official(self) -> None:
        tier_c, proofs, protocols, raw, audit = _build_admission(
            self.bundle,
            source_tier=TemporalSourceTier.TIER_C,
        )
        partial = TemporalAdmissionEnvelope.model_validate({**tier_c.model_dump(), 'receipts': tier_c.receipts[:1]})

        self.assertEqual(len(partial.receipts), 1)
        with self.assertRaisesRegex(TemporalAdmissionError, 'Tier A'):
            require_official_temporal_admission(
                partial,
                self.bundle,
                receipt_artifacts={'receipt-0': proofs['receipt-0']},
                receipt_verifier=_receipt_verifier,
                protocol_artifacts=protocols,
                raw_outcome_source=raw,
                label_derivation_audit=audit,
                expected_decision_context_sha256='0' * 64,
                expected_decision_context_bytes=1,
            )


if __name__ == '__main__':
    unittest.main()
