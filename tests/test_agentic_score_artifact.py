from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

import pytest

from tests.test_agentic_run_artifact import KEY as RUN_RECEIPT_KEY
from tests.test_agentic_run_artifact import STARTED, _finalize
from vaxreplay.agentic.admission import AgenticWorkspaceAdmission
from vaxreplay.agentic.protocol import agentic_receipt_key_id
from vaxreplay.agentic.schema import agentic_model_sha256
from vaxreplay.agentic.score_artifact import (
    AgenticScoreArtifactError,
    agentic_score_receipt_hmac,
    agentic_score_receipt_key_id,
    finalize_agentic_score,
    load_agentic_score_artifact,
)
from vaxreplay.agentic.scoring import (
    AgenticPrivateGoldV1,
    AgenticScoringContract,
    score_agentic_submission,
)
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace
from vaxreplay.bundle import canonical_json_bytes

SCORE_RECEIPT_KEY = bytes.fromhex('cd' * 32)
FINALIZED_AT = STARTED + timedelta(seconds=2)


class _TrustedInputs(TypedDict):
    workspace: LoadedAgenticWorkspace
    admission: AgenticWorkspaceAdmission
    expected_admission_sha256: str
    scoring_contract: AgenticScoringContract
    expected_scoring_contract_sha256: str
    gold: AgenticPrivateGoldV1
    gold_commitment_key: bytes
    expected_private_gold_commitment_sha256: str
    expected_private_gold_commitment_key_id: str
    run_receipt_key: bytes
    expected_run_receipt_key_id: str
    score_receipt_key: bytes
    expected_score_receipt_key_id: str


def _score(case):
    return score_agentic_submission(
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        gold=case.gold,
        gold_commitment_key=case.gold_commitment_key,
        expected_gold_commitment_sha256=case.gold_commitment_sha256,
        submission=case.oracle,
    )


def _trusted_inputs(case) -> _TrustedInputs:
    contract = AgenticScoringContract.from_workspace(case.workspace)
    return {
        'workspace': case.workspace,
        'admission': case.admission,
        'expected_admission_sha256': case.admission_sha256,
        'scoring_contract': contract,
        'expected_scoring_contract_sha256': agentic_model_sha256(contract),
        'gold': case.gold,
        'gold_commitment_key': case.gold_commitment_key,
        'expected_private_gold_commitment_sha256': case.gold_commitment_sha256,
        'expected_private_gold_commitment_key_id': hashlib.sha256(case.gold_commitment_key).hexdigest(),
        'run_receipt_key': RUN_RECEIPT_KEY,
        'expected_run_receipt_key_id': agentic_receipt_key_id(RUN_RECEIPT_KEY),
        'score_receipt_key': SCORE_RECEIPT_KEY,
        'expected_score_receipt_key_id': agentic_score_receipt_key_id(SCORE_RECEIPT_KEY),
    }


def _finalized(tmp_path: Path):
    case, _, run = _finalize(tmp_path)
    score = _score(case)
    inputs = _trusted_inputs(case)
    artifact = finalize_agentic_score(
        output_root=tmp_path / 'score',
        run_artifact=run,
        finalized_at=FINALIZED_AT,
        **inputs,
    )
    return case, run, score, inputs, artifact


def test_finalizer_binds_exact_run_score_gold_and_leaderboard_identity(tmp_path: Path) -> None:
    case, run, score, _, artifact = _finalized(tmp_path)
    receipt = artifact.receipt

    assert {entry.name for entry in artifact.root.iterdir()} == {
        'score-receipt.json',
        'score-receipt.hmac',
        'score-vector.json',
    }
    assert artifact.score_vector == score
    assert receipt.run_id == run.receipt.run_id
    assert receipt.run_receipt_sha256 == run.receipt_sha256
    assert receipt.run_receipt_key_id == run.receipt.receipt_key_id
    assert receipt.receipt_key_id == agentic_score_receipt_key_id(SCORE_RECEIPT_KEY)
    assert receipt.receipt_key_id != receipt.run_receipt_key_id
    assert receipt.final_submission_sha256 == run.receipt.final_submission_sha256
    assert receipt.attempt_reservation_sha256 == run.receipt.attempt_reservation_sha256
    assert receipt.workspace_admission_sha256 == case.admission_sha256
    assert receipt.scoring_contract_sha256 == score.scoring_contract_sha256
    assert receipt.private_gold_commitment_sha256 == case.gold_commitment_sha256
    assert receipt.private_gold_commitment_key_id == hashlib.sha256(case.gold_commitment_key).hexdigest()
    assert receipt.harness_id == run.receipt.harness_id
    assert receipt.harness_manifest_sha256 == run.receipt.harness_manifest_sha256
    assert receipt.harness_behavior_sha256 == run.receipt.harness_behavior_sha256
    assert receipt.harness_execution_mode == run.receipt.harness_execution_mode
    assert receipt.requested_model_id == run.receipt.requested_model_id
    assert receipt.resolved_model_id == run.receipt.resolved_model_id
    assert receipt.status == score.status
    assert receipt.reward == score.reward
    assert receipt.finalized_at == FINALIZED_AT


def test_loader_rejects_tampered_score_vector_hmac_and_extra_inventory(tmp_path: Path) -> None:
    _, run, _, inputs, artifact = _finalized(tmp_path)
    score_path = artifact.root / 'score-vector.json'
    original_score = score_path.read_bytes()
    score_path.write_bytes(original_score + b' ')
    with pytest.raises(AgenticScoreArtifactError, match='canonical encoding'):
        load_agentic_score_artifact(artifact.root, run_artifact=run, **inputs)

    score_path.write_bytes(original_score)
    hmac_path = artifact.root / 'score-receipt.hmac'
    original_hmac = hmac_path.read_bytes()
    hmac_path.write_bytes(b'0' * 64 + b'\n')
    with pytest.raises(AgenticScoreArtifactError, match='HMAC authentication failed'):
        load_agentic_score_artifact(artifact.root, run_artifact=run, **inputs)

    hmac_path.write_bytes(original_hmac)
    extra = artifact.root / 'uncommitted-metadata.json'
    extra.write_bytes(b'{}')
    extra.chmod(0o600)
    with pytest.raises(AgenticScoreArtifactError, match='exact file inventory mismatch'):
        load_agentic_score_artifact(artifact.root, run_artifact=run, **inputs)


def test_finalizer_rejects_gold_and_contract_splicing(tmp_path: Path) -> None:
    case, _, run = _finalize(tmp_path)
    inputs = _trusted_inputs(case)

    wrong_gold = inputs.copy()
    wrong_gold['expected_private_gold_commitment_sha256'] = '0' * 64
    with pytest.raises(AgenticScoreArtifactError, match='private.gold.*commitment'):
        finalize_agentic_score(
            output_root=tmp_path / 'spliced-gold',
            run_artifact=run,
            finalized_at=FINALIZED_AT,
            **wrong_gold,
        )

    wrong_contract = inputs.copy()
    wrong_contract['expected_scoring_contract_sha256'] = '1' * 64
    with pytest.raises(AgenticScoreArtifactError, match='scoring contract'):
        finalize_agentic_score(
            output_root=tmp_path / 'spliced-contract',
            run_artifact=run,
            finalized_at=FINALIZED_AT,
            **wrong_contract,
        )


def test_finalizer_reauthenticates_run_and_rejects_post_load_mutation(tmp_path: Path) -> None:
    case, _, run = _finalize(tmp_path)
    submission_path = run.root / 'submission.json'
    submission_path.write_bytes(submission_path.read_bytes() + b' ')

    with pytest.raises(AgenticScoreArtifactError, match='canonical JSON'):
        finalize_agentic_score(
            output_root=tmp_path / 'score',
            run_artifact=run,
            finalized_at=FINALIZED_AT,
            **_trusted_inputs(case),
        )


def test_finalizer_rehashes_non_submission_run_components(tmp_path: Path) -> None:
    case, _, run = _finalize(tmp_path)
    transcript_path = run.root / 'transcript.json'
    transcript_path.write_bytes(transcript_path.read_bytes() + b' ')

    with pytest.raises(AgenticScoreArtifactError, match='component hashes'):
        finalize_agentic_score(
            output_root=tmp_path / 'score',
            run_artifact=run,
            finalized_at=FINALIZED_AT,
            **_trusted_inputs(case),
        )


def test_score_receipt_requires_distinct_key_and_post_run_timestamp(tmp_path: Path) -> None:
    case, _, run = _finalize(tmp_path)
    inputs = _trusted_inputs(case)
    same_key_inputs = inputs.copy()
    same_key_inputs['score_receipt_key'] = RUN_RECEIPT_KEY
    same_key_inputs['expected_score_receipt_key_id'] = agentic_score_receipt_key_id(RUN_RECEIPT_KEY)
    with pytest.raises(AgenticScoreArtifactError, match='distinct HMAC keys'):
        finalize_agentic_score(
            output_root=tmp_path / 'same-key',
            run_artifact=run,
            finalized_at=FINALIZED_AT,
            **same_key_inputs,
        )

    with pytest.raises(AgenticScoreArtifactError, match='before its run finished'):
        finalize_agentic_score(
            output_root=tmp_path / 'early',
            run_artifact=run,
            finalized_at=STARTED,
            **inputs,
        )


def test_loader_rejects_authenticated_formula_consistent_forged_score(tmp_path: Path) -> None:
    _, run, score, inputs, artifact = _finalized(tmp_path)
    assert score.reward == pytest.approx(1.0)
    forged_score = score.model_copy(
        update={
            'reward': 0.25,
            'retrieval_precision': 0.25,
            'retrieval_recall': 0.25,
            'retrieval_f1': 0.25,
            'extraction_score': 0.25,
            'extraction_signed_utility': -0.5,
            'analysis_score': 0.25,
            'analysis_signed_utility': -0.5,
            'citation_precision': 0.25,
            'citation_recall': 0.25,
            'citation_f1': 0.25,
            'ndcg_at_k': 0.25,
            'top_k_utility': 0.25,
            'decision_score': 0.25,
            'process_score': 0.25,
        }
    )
    forged_bytes = canonical_json_bytes(forged_score)
    # Prove this is a well-formed score vector, not merely an invalid-model mutation.
    assert type(score).model_validate_json(forged_bytes).reward == pytest.approx(0.25)
    forged_receipt = artifact.receipt.model_copy(
        update={
            'score_vector_sha256': hashlib.sha256(forged_bytes).hexdigest(),
            'reward': 0.25,
        }
    )
    (artifact.root / 'score-vector.json').write_bytes(forged_bytes)
    (artifact.root / 'score-receipt.json').write_bytes(canonical_json_bytes(forged_receipt))
    (artifact.root / 'score-receipt.hmac').write_bytes(
        (agentic_score_receipt_hmac(forged_receipt, SCORE_RECEIPT_KEY) + '\n').encode('ascii')
    )

    with pytest.raises(AgenticScoreArtifactError, match='deterministic private-gold recomputation'):
        load_agentic_score_artifact(artifact.root, run_artifact=run, **inputs)
