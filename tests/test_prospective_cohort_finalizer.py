from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from tests.test_prospective_attempt_reservation import (
    _authorized_attempt,
    _completion_kwargs,
    _proof,
    _Registry,
)
from tests.test_prospective_finalizer import _prospective_oracle, _sealed_bundle
from tests.test_prospective_release import _case_verifier, _decision_verifier, _source_capture_verifier
from tests.test_prospective_release_seal import (
    _approval_kwargs,
    _release_decision_verifier,
    _timestamp_verifier,
)
from tests.test_prospective_run_seal import _RECEIPT_KEY, _Backend
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    CaseSelectionRecord,
)
from vaxreplay.case_schema import LabelCommitmentScheme, ScoreStatus
from vaxreplay.prospective_cohort_finalizer import (
    ProspectiveCaseFinalizationInput,
    ProspectiveCohortFinalizationIntegrityError,
    build_prospective_cohort_finalization,
    load_prospective_cohort_finalization,
    score_prospective_cohort_finalization,
)
from vaxreplay.runner.backend import IsolationBackend
from vaxreplay.runner.orchestrator import (
    load_run_artifact,
    receipt_hmac_sha256,
    receipt_key_id,
    run_challenge_bundle,
)
from vaxreplay.runner.prospective_attempt_reservation import (
    ProspectiveAttemptCompletionStatus,
    ProspectiveExplicitFailure,
    build_prospective_attempt_completion,
    build_prospective_attempt_completion_target,
)
from vaxreplay.temporal_schema import (
    OUTCOME_SNAPSHOT_SCHEMA_VERSION,
    OutcomeSnapshotCommitment,
    OutcomeTargetAvailability,
    TemporalAdmissionEnvelope,
    TemporalAdmissionUse,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalProvenanceBasis,
    TemporalReceiptAuthority,
    TemporalSourceTier,
    model_sha256,
)

pytestmark = pytest.mark.usefixtures('synthetic_official_replay_patch')

_OUTCOME_PROOF = b'fictional source-signed outcome proof'


def _temporal_verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    if receipt.role == TemporalArtifactRole.OUTCOME_SNAPSHOT:
        return receipt.authority_id == 'fixture-outcome-source' and proof == _OUTCOME_PROOF
    return _decision_verifier(receipt, proof)


def _selection_verifier(policy, universe, audit, evidence) -> bool:
    return (
        hashlib.sha256(policy).hexdigest() == audit.selection_policy_sha256
        and set(evidence) == {entry.case_id for entry in universe.entries}
        and all(value.startswith(b'complete-disposition-evidence:') for value in evidence.values())
    )


def _temporal_materials(release, bundle):
    package = release.verified_admission.packages[0]
    decision = package.manifest.episode.decision_snapshot
    raw_outcome = b'fictional immutable blinded outcome source for cohort finalization'
    derivation = b'fictional complete private label derivation audit for cohort finalization'
    availability: dict[tuple[str, int], datetime] = {}
    assert bundle.private_labels is not None
    for outcome in bundle.private_labels.outcomes:
        key = (outcome.target_id, outcome.horizon_days)
        previous = availability.get(key)
        if previous is None or outcome.revealed_at < previous:
            availability[key] = outcome.revealed_at
    targets = tuple(
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
        outcome_adjudication_spec_sha256=(decision.protocol_commitments.outcome_adjudication_spec_sha256),
        raw_outcome_source_sha256=hashlib.sha256(raw_outcome).hexdigest(),
        raw_outcome_source_bytes=len(raw_outcome),
        label_derivation_audit_sha256=hashlib.sha256(derivation).hexdigest(),
        label_derivation_audit_bytes=len(derivation),
        target_availability=targets,
    )
    decision_seal = release.verified_admission.seals[0]
    receipts = list(decision_seal.manifest.receipts)
    outcome_receipt = TemporalArtifactReceipt(
        receipt_id='fixture-outcome-receipt',
        role=TemporalArtifactRole.OUTCOME_SNAPSHOT,
        artifact_schema_version=OUTCOME_SNAPSHOT_SCHEMA_VERSION,
        artifact_sha256=model_sha256(outcome_snapshot),
        artifact_bytes=len(canonical_json_bytes(outcome_snapshot)),
        witnessed_at=outcome_snapshot.first_label_available_at,
        authority_type=TemporalReceiptAuthority.SOURCE_SIGNED_VERSION,
        authority_id='fixture-outcome-source',
        receipt_sha256=hashlib.sha256(_OUTCOME_PROOF).hexdigest(),
        receipt_bytes=len(_OUTCOME_PROOF),
        verification_uri='https://outcomes.invalid/fixture-outcome-receipt',
    )
    receipts.append(outcome_receipt)
    proofs = dict(decision_seal.proof_artifacts)
    proofs[outcome_receipt.receipt_id] = _OUTCOME_PROOF
    admission = TemporalAdmissionEnvelope(
        admission_id='fixture-official-cohort-outcome',
        episode_id=bundle.manifest.episode_id,
        manifest_sha256=bundle.manifest_sha256,
        source_tier=TemporalSourceTier.TIER_A,
        admitted_use=TemporalAdmissionUse.OFFICIAL_BENCHMARK,
        provenance_basis=TemporalProvenanceBasis.PROSPECTIVE_SEAL,
        decision_snapshot=decision,
        decision_context_sha256=decision_seal.manifest.decision_context_sha256,
        decision_context_bytes=decision_seal.manifest.receipts[2].artifact_bytes,
        outcome_snapshot=outcome_snapshot,
        receipts=tuple(receipts),
        admitted_at=outcome_snapshot.first_label_available_at + timedelta(seconds=1),
    )
    return admission, proofs, package.protocol_artifacts, raw_outcome, derivation


def _successful_context(root: Path):
    registry = _Registry()
    materials = _authorized_attempt(root, registry)
    release, _opening, _release_seal, system, policy, _attempt_policy, _reservation, start = materials
    bundle = _sealed_bundle()
    response = _prospective_oracle(
        bundle,
        release.challenge.suite.episodes[0],
    )
    raw_run = run_challenge_bundle(
        release.challenge,
        expected_challenge_sha256=release.challenge.manifest_sha256,
        system=system,
        policy=policy,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
        output_dir=root / 'run',
        backend=cast(
            IsolationBackend,
            _Backend(response.model_dump_json().encode()),
        ),
    )
    started_at = start.manifest.start_proof.witnessed_at + timedelta(seconds=1)
    receipt = raw_run.receipt.model_copy(
        update={'started_at': started_at, 'finished_at': started_at + timedelta(seconds=1)}
    )
    (raw_run.root / 'run.json').write_bytes(canonical_json_bytes(receipt))
    (raw_run.root / 'run.hmac').write_text(
        receipt_hmac_sha256(receipt, _RECEIPT_KEY) + '\n',
        encoding='ascii',
    )
    run = load_run_artifact(
        raw_run.root,
        challenge=release.challenge,
        system=system,
        policy=policy,
        receipt_key=_RECEIPT_KEY,
        expected_receipt_key_id=receipt_key_id(_RECEIPT_KEY),
        require_sealed=True,
    )
    target = build_prospective_attempt_completion_target(
        **_completion_kwargs(materials, registry),
        run=run,
    )
    proof, raw = _proof(
        target,
        event_type='completion',
        witnessed_at=run.receipt.finished_at + timedelta(seconds=1),
        receipt_id='cohort-finalizer-success',
    )
    completion = build_prospective_attempt_completion(
        root / 'completion',
        **_completion_kwargs(materials, registry),
        registry_proof=proof,
        proof_bytes=raw,
        run=run,
    )
    return registry, materials, completion


def _failure_context(root: Path):
    registry = _Registry()
    materials = _authorized_attempt(root, registry)
    start = materials[-1]
    record = b'fictional official backend failed before producing any response'
    failure = ProspectiveExplicitFailure(
        failure_code='backend_start_failure',
        backend_id='fixture-official-backend',
        started_at=start.manifest.start_proof.witnessed_at + timedelta(seconds=1),
        failed_at=start.manifest.start_proof.witnessed_at + timedelta(seconds=2),
        failure_record_sha256=hashlib.sha256(record).hexdigest(),
        failure_record_bytes=len(record),
    )
    target = build_prospective_attempt_completion_target(
        **_completion_kwargs(materials, registry),
        failure=failure,
        failure_record=record,
    )
    proof, raw = _proof(
        target,
        event_type='completion',
        witnessed_at=failure.failed_at,
        receipt_id='cohort-finalizer-failure',
    )
    completion = build_prospective_attempt_completion(
        root / 'completion',
        **_completion_kwargs(materials, registry),
        registry_proof=proof,
        proof_bytes=raw,
        failure=failure,
        failure_record=record,
    )
    return registry, materials, completion


def _finalization_inputs(release):
    bundle = _sealed_bundle()
    admission, proofs, protocols, raw_outcome, derivation = _temporal_materials(release, bundle)
    audit = CaseSelectionAudit(
        case_universe_sha256=release.verified_admission.admission.case_universe_sha256,
        selection_policy_sha256=hashlib.sha256(release.verifier_policy).hexdigest(),
        records=(
            CaseSelectionRecord(
                case_id='fictional-case-1',
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
    case_input = ProspectiveCaseFinalizationInput(
        final_bundle=bundle,
        temporal_admission=admission,
        receipt_artifacts=proofs,
        protocol_artifacts=protocols,
        raw_outcome_source=raw_outcome,
        label_derivation_audit=derivation,
    )
    evidence = {'fictional-case-1': b'complete-disposition-evidence:admitted'}
    return audit, evidence, {'fictional-case-1': case_input}, admission.admitted_at + timedelta(seconds=1)


def _build_kwargs(materials, completion, registry):
    (
        release,
        _opening,
        release_seal,
        system,
        policy,
        attempt_policy,
        reservation,
        start_authorization,
    ) = materials
    return {
        'release': release,
        'release_seal': release_seal,
        'reservation': reservation,
        'start_authorization': start_authorization,
        'completion': completion,
        'system': system,
        'runner_policy': policy,
        'attempt_policy': attempt_policy,
        'receipt_key': _RECEIPT_KEY,
        'expected_receipt_key_id': receipt_key_id(_RECEIPT_KEY),
        'decision_receipt_verifier': _release_decision_verifier(release),
        'source_capture_verifier': _source_capture_verifier,
        'case_universe_seal_verifier': _case_verifier,
        **_approval_kwargs(release),
        'release_timestamp_verifier': _timestamp_verifier,
        'registry_verifier': registry,
        'start_verifier': registry.verify_start,
        'expected_start_authorization_manifest_sha256': start_authorization.manifest_sha256,
        'temporal_receipt_verifier': _temporal_verifier,
        'case_selection_policy_verifier': _selection_verifier,
    }


class ProspectiveCohortFinalizerTest(unittest.TestCase):
    def test_post_publish_replacement_is_left_untouched_and_cleanup_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / 'finalization'
            displaced = root / 'displaced-owned-finalization'
            marker = b'unrelated replacement content'
            registry, materials, completion = _successful_context(root)
            audit, evidence, case_inputs, finalized_at = _finalization_inputs(materials[0])

            def replacing_selection_verifier(policy, universe, selection_audit, selection_evidence):
                if target.exists():
                    os.rename(target, displaced)
                    target.mkdir()
                    (target / 'do-not-delete.txt').write_bytes(marker)
                    return False
                return _selection_verifier(policy, universe, selection_audit, selection_evidence)

            kwargs = _build_kwargs(materials, completion, registry)
            kwargs['case_selection_policy_verifier'] = replacing_selection_verifier
            with self.assertRaisesRegex(ValueError, 'cleanup failed closed'):
                build_prospective_cohort_finalization(
                    target,
                    **kwargs,
                    case_selection_audit=audit,
                    disposition_evidence=evidence,
                    case_inputs=case_inputs,
                    finalized_at=finalized_at,
                )

            self.assertEqual((target / 'do-not-delete.txt').read_bytes(), marker)
            self.assertTrue((displaced / 'finalization.json').is_file())

    def test_scores_only_registered_response_and_reverifies_complete_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry, materials, completion = _successful_context(root)
            audit, evidence, case_inputs, finalized_at = _finalization_inputs(materials[0])
            kwargs = _build_kwargs(materials, completion, registry)
            built = build_prospective_cohort_finalization(
                root / 'finalization',
                **kwargs,
                case_selection_audit=audit,
                disposition_evidence=evidence,
                case_inputs=case_inputs,
                finalized_at=finalized_at,
            )

            self.assertEqual(built.manifest.completion_manifest_sha256, completion.manifest_sha256)
            self.assertEqual(
                built.manifest.start_authorization_manifest_sha256,
                materials[-1].manifest_sha256,
            )
            self.assertEqual(built.start_authorization, materials[-1])
            assert completion.target.run is not None
            self.assertEqual(
                built.manifest.cases[0].response_record_sha256,
                completion.target.run.episodes[0].response_record_sha256,
            )
            self.assertEqual(built.score_report.denominator_case_count, 1)
            self.assertEqual(built.score_report.valid_score_count, 1)
            self.assertEqual(
                built.score_report.episodes[0].score.status,
                ScoreStatus.VALID,
            )
            self.assertEqual(built.score_report.reward, 1.0)
            self.assertEqual(
                built.manifest.schema_version,
                'vaxreplay.prospective-cohort-finalization.v0.4',
            )
            self.assertEqual(built.manifest.purpose, 'official_benchmark')
            self.assertEqual(
                built.cases[0].finalization.schema_version,
                'vaxreplay.prospective-finalization-binding.v0.3',
            )
            self.assertEqual(built.cases[0].finalization.purpose, 'official_benchmark')
            self.assertTrue((built.root / 'cases/000000/episode/private/label_commitment_key.hex').is_file())

            loaded = load_prospective_cohort_finalization(built.root, **kwargs)
            self.assertEqual(loaded.manifest_sha256, built.manifest_sha256)
            self.assertEqual(
                score_prospective_cohort_finalization(built, **kwargs),
                built.score_report,
            )

    def test_rejects_score_or_private_artifact_substitution(self) -> None:
        for tamper in ('score', 'outcome'):
            with (
                self.subTest(tamper=tamper),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                registry, materials, completion = _successful_context(root)
                audit, evidence, case_inputs, finalized_at = _finalization_inputs(materials[0])
                kwargs = _build_kwargs(materials, completion, registry)
                built = build_prospective_cohort_finalization(
                    root / 'finalization',
                    **kwargs,
                    case_selection_audit=audit,
                    disposition_evidence=evidence,
                    case_inputs=case_inputs,
                    finalized_at=finalized_at,
                )
                path = (
                    built.root / 'score-report.json'
                    if tamper == 'score'
                    else built.root / 'cases/000000/raw-outcome-source.bin'
                )
                path.write_bytes(path.read_bytes() + b'tampered')
                with self.assertRaisesRegex(
                    ProspectiveCohortFinalizationIntegrityError,
                    'artifact changed',
                ):
                    load_prospective_cohort_finalization(built.root, **kwargs)

    def test_frozen_policy_is_executed_not_merely_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry, materials, completion = _successful_context(root)
            audit, evidence, case_inputs, finalized_at = _finalization_inputs(materials[0])
            kwargs = _build_kwargs(materials, completion, registry)
            kwargs['case_selection_policy_verifier'] = lambda *_args: False
            with self.assertRaisesRegex(ValueError, 'policy verifier rejected'):
                build_prospective_cohort_finalization(
                    root / 'finalization',
                    **kwargs,
                    case_selection_audit=audit,
                    disposition_evidence=evidence,
                    case_inputs=case_inputs,
                    finalized_at=finalized_at,
                )

    def test_missing_outcome_stays_in_fixed_denominator_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry, materials, completion = _successful_context(root)
            release = materials[0]
            audit = CaseSelectionAudit(
                case_universe_sha256=release.verified_admission.admission.case_universe_sha256,
                selection_policy_sha256=hashlib.sha256(release.verifier_policy).hexdigest(),
                records=(
                    CaseSelectionRecord(
                        case_id='fictional-case-1',
                        disposition=CaseSelectionDisposition.UNSCORED_MISSING,
                        panel_count=4,
                        observed_count=3,
                        missing_count=1,
                        conflict_count=0,
                        reason_code='one_prespecified_endpoint_absent',
                    ),
                ),
            )
            kwargs = _build_kwargs(materials, completion, registry)
            built = build_prospective_cohort_finalization(
                root / 'finalization',
                **kwargs,
                case_selection_audit=audit,
                disposition_evidence={'fictional-case-1': b'complete-disposition-evidence:verified-missing'},
                case_inputs={},
                finalized_at=completion.target.terminal_at + timedelta(days=400),
            )
            self.assertEqual(built.score_report.denominator_case_count, 1)
            self.assertEqual(built.score_report.unscored_case_count, 1)
            self.assertEqual(built.score_report.reward, 0.0)

    def test_finalization_cannot_precede_external_completion_registry_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry, materials, completion = _successful_context(root)
            release = materials[0]
            audit = CaseSelectionAudit(
                case_universe_sha256=release.verified_admission.admission.case_universe_sha256,
                selection_policy_sha256=hashlib.sha256(release.verifier_policy).hexdigest(),
                records=(
                    CaseSelectionRecord(
                        case_id='fictional-case-1',
                        disposition=CaseSelectionDisposition.UNSCORED_MISSING,
                        panel_count=4,
                        observed_count=3,
                        missing_count=1,
                        conflict_count=0,
                        reason_code='one_prespecified_endpoint_absent',
                    ),
                ),
            )
            with self.assertRaisesRegex(
                ProspectiveCohortFinalizationIntegrityError,
                'cannot precede the external completion registry proof',
            ):
                build_prospective_cohort_finalization(
                    root / 'early-finalization',
                    **_build_kwargs(materials, completion, registry),
                    case_selection_audit=audit,
                    disposition_evidence={'fictional-case-1': b'complete-disposition-evidence:verified-missing'},
                    case_inputs={},
                    finalized_at=(completion.manifest.registry_proof.witnessed_at - timedelta(microseconds=1)),
                )

    def test_registered_attempt_failure_is_retained_as_invalid_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry, materials, completion = _failure_context(root)
            audit, evidence, case_inputs, finalized_at = _finalization_inputs(materials[0])
            kwargs = _build_kwargs(materials, completion, registry)
            built = build_prospective_cohort_finalization(
                root / 'finalization',
                **kwargs,
                case_selection_audit=audit,
                disposition_evidence=evidence,
                case_inputs=case_inputs,
                finalized_at=finalized_at,
            )

            self.assertEqual(
                built.manifest.completion_status,
                ProspectiveAttemptCompletionStatus.FAILURE,
            )
            self.assertIsNone(built.manifest.cases[0].response_record_sha256)
            self.assertEqual(built.score_report.invalid_response_count, 1)
            self.assertEqual(built.score_report.reward, 0.0)
            self.assertEqual(
                built.score_report.episodes[0].score.issues[0].code.value,
                'RUNNER_FAILURE',
            )


if __name__ == '__main__':
    unittest.main()
