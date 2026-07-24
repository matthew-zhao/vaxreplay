from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.baselines import oracle_submission, uniform_submission
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.qa.admission import (
    InMemoryAdmissionTokenConsumer,
    gradient_admission_signing_key_id,
    issue_gradient_admission_token,
)
from vaxreplay.qa.attack_catalog import attack_catalog_sha256, default_attack_catalog
from vaxreplay.qa.firewall import (
    QuarantinedBatch,
    ReplayScorerPair,
    RewardFirewallQuarantine,
    RewardFirewallReason,
    TrainingRewardFirewall,
    build_quarantined_batch,
    release_training_batch,
)
from vaxreplay.qa.schema import (
    ComponentFloorSpec,
    ComponentScore,
    QACategory,
    QADisposition,
    QASeverity,
    RewardContract,
    RewardQAFinding,
    RewardQAReport,
    TrainingRunAdmission,
    reward_contract_sha256,
    reward_qa_report_sha256,
)
from vaxreplay.scoring import LocalSubmissionEvaluator

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
_DIGESTS = tuple(f'{value:064x}' for value in range(1, 20))
_AUDIT_TRACE = b'{"events":[],"recorder":"test"}'


def _bundle() -> EpisodeBundle:
    return EpisodeBundle.load(
        Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0',
        include_private=True,
    )


def _contract(bundle: EpisodeBundle) -> RewardContract:
    return RewardContract(
        contract_id='firewall-contract',
        reward_version=bundle.manifest.reward_version,
        scientific_objective='Grounded vaccine-target ranking from decision-time evidence.',
        scorer_sha256=_DIGESTS[0],
        reference_scorer_sha256=_DIGESTS[1],
        prompt_compiler_sha256=_DIGESTS[2],
        qa_policy_sha256=_DIGESTS[3],
        attack_catalog_sha256=attack_catalog_sha256(default_attack_catalog()),
        required_dimensions=tuple(sorted(bundle.manifest.required_dimensions)),
        required_attack_ids=('parser.duplicate-key',),
        component_floors=(
            ComponentFloorSpec(metric='assessment_accuracy', minimum=0.01),
            ComponentFloorSpec(metric='grounding_f1', minimum=0.01),
        ),
        prohibited_strategies=('outcome lookup', 'reward tampering'),
    )


def _firewall(bundle: EpisodeBundle, contract: RewardContract) -> TrainingRewardFirewall:
    return TrainingRewardFirewall(
        run_id='run-1',
        bundle=bundle,
        primary_evaluator=LocalSubmissionEvaluator(bundle),
        reference_evaluator=LocalSubmissionEvaluator(bundle),
        primary_scorer_sha256=contract.scorer_sha256,
        reference_scorer_sha256=contract.reference_scorer_sha256,
        reward_contract=contract,
    )


def _report(batch, contract: RewardContract) -> RewardQAReport:
    return RewardQAReport(
        report_id='report-1',
        run_id=batch.envelope.run_id,
        created_at=_NOW,
        trajectory_batch_sha256=batch.envelope.trajectory_batch_sha256,
        reward_artifact_sha256=batch.envelope.reward_artifact_sha256,
        episode_manifest_sha256s=batch.envelope.episode_manifest_sha256s,
        reward_contract=contract,
        reward_contract_sha256=reward_contract_sha256(contract),
        component_scores=(
            ComponentScore(metric='assessment_accuracy', value=1.0),
            ComponentScore(metric='grounding_f1', value=1.0),
        ),
        findings=(
            RewardQAFinding(
                finding_id='finding-duplicate-key',
                check_id='parser.duplicate-key',
                category=QACategory.PARSER_INTEGRITY,
                severity=QASeverity.INFO,
                passed=True,
                disposition_on_failure=QADisposition.REJECT,
                detail='strict parser rejected duplicate object members',
                artifact_sha256=_DIGESTS[4],
            ),
        ),
        disposition=QADisposition.ADMIT,
        all_required_attacks_executed=True,
        independent_scorer_agreement=True,
        future_taint_reachability_zero=True,
        exact_replay=True,
        tamper_success_zero=True,
        item_level_private_feedback_withheld=True,
    )


def _admission(batch, contract: RewardContract, signer: LocalEd25519Signer) -> TrainingRunAdmission:
    report = _report(batch, contract)
    return TrainingRunAdmission(
        admission_id='admission-1',
        run_id=batch.envelope.run_id,
        issued_at=_NOW,
        not_before=_NOW,
        expires_at=_NOW + timedelta(hours=1),
        trajectory_batch_sha256=batch.envelope.trajectory_batch_sha256,
        reward_artifact_sha256=batch.envelope.reward_artifact_sha256,
        model_sha256=_DIGESTS[5],
        harness_sha256=_DIGESTS[6],
        tool_policy_sha256=_DIGESTS[7],
        environment_sha256=_DIGESTS[8],
        dataset_sha256=_DIGESTS[9],
        optimizer_config_sha256=_DIGESTS[10],
        episode_manifest_sha256s=batch.envelope.episode_manifest_sha256s,
        reward_qa_report=report,
        reward_qa_report_sha256=reward_qa_report_sha256(report),
        reward_qa_attestation_sha256=_DIGESTS[11],
        qa_signing_key_id=_DIGESTS[12],
        reward_contract_sha256=reward_contract_sha256(contract),
        attack_catalog_sha256=contract.attack_catalog_sha256,
        signing_key_id=gradient_admission_signing_key_id(signer.public_key_bytes()),
    )


def _release(batch, contract, admission, token, signer, consumer, **overrides):
    expected = {
        'expected_model_sha256': admission.model_sha256,
        'expected_harness_sha256': admission.harness_sha256,
        'expected_tool_policy_sha256': admission.tool_policy_sha256,
        'expected_environment_sha256': admission.environment_sha256,
        'expected_dataset_sha256': admission.dataset_sha256,
        'expected_optimizer_config_sha256': admission.optimizer_config_sha256,
        'replay_scorers': {
            batch.envelope.episode_manifest_sha256s[0]: ReplayScorerPair(
                bundle=_bundle(),
                primary_evaluator=LocalSubmissionEvaluator(_bundle()),
                reference_evaluator=LocalSubmissionEvaluator(_bundle()),
                primary_scorer_sha256=contract.scorer_sha256,
                reference_scorer_sha256=contract.reference_scorer_sha256,
            )
        },
    }
    expected.update(overrides)
    return release_training_batch(
        batch,
        reward_contract=contract,
        admission=admission,
        token=token,
        trusted_public_key_bytes=signer.public_key_bytes(),
        now=_NOW + timedelta(minutes=1),
        consume_token=consumer,
        **expected,
    )


def test_reward_stays_quarantined_until_signed_batch_release() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    pending = _firewall(bundle, contract).quarantine(
        trajectory_id='trajectory-1',
        response=oracle_submission(bundle).model_dump_json(),
        audit_trace=_AUDIT_TRACE,
    )
    batch = build_quarantined_batch((pending,))
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(batch, contract, signer)
    token = issue_gradient_admission_token(admission, signer)

    [released] = _release(
        batch,
        contract,
        admission,
        token,
        signer,
        InMemoryAdmissionTokenConsumer(),
    )

    assert released.reward == 1.0
    assert released.trajectory_id == 'trajectory-1'
    assert released.gradient_admission_token_id == token.token_id


def test_uniform_positive_reward_is_quarantined_by_component_veto() -> None:
    bundle = _bundle()
    contract = _contract(bundle)

    with pytest.raises(RewardFirewallQuarantine) as caught:
        _firewall(bundle, contract).quarantine(
            trajectory_id='uniform',
            response=uniform_submission(bundle).model_dump_json(),
            audit_trace=_AUDIT_TRACE,
        )

    assert caught.value.reason is RewardFirewallReason.COMPONENT_FLOOR


def test_duplicate_key_is_quarantined_before_scoring() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    response = oracle_submission(bundle).model_dump_json()
    response = response.replace(
        f'"episode_id":"{bundle.manifest.episode_id}"',
        f'"episode_id":"wrong","episode_id":"{bundle.manifest.episode_id}"',
        1,
    )

    with pytest.raises(RewardFirewallQuarantine) as caught:
        _firewall(bundle, contract).quarantine(
            trajectory_id='duplicate',
            response=response,
            audit_trace=_AUDIT_TRACE,
        )

    assert caught.value.reason is RewardFirewallReason.STRICT_PARSE


def test_primary_reference_disagreement_is_quarantined() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    primary = LocalSubmissionEvaluator(bundle)

    class DisagreeingReference:
        def score(self, _submission):
            return primary.score(uniform_submission(bundle))

    firewall = TrainingRewardFirewall(
        run_id='run-1',
        bundle=bundle,
        primary_evaluator=primary,
        reference_evaluator=DisagreeingReference(),
        primary_scorer_sha256=contract.scorer_sha256,
        reference_scorer_sha256=contract.reference_scorer_sha256,
        reward_contract=contract,
    )

    with pytest.raises(RewardFirewallQuarantine) as caught:
        firewall.quarantine(
            trajectory_id='disagreement',
            response=oracle_submission(bundle).model_dump_json(),
            audit_trace=_AUDIT_TRACE,
        )

    assert caught.value.reason is RewardFirewallReason.SCORE_INTEGRITY


def test_runtime_binding_mismatch_does_not_consume_valid_token() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    pending = _firewall(bundle, contract).quarantine(
        trajectory_id='trajectory-1',
        response=oracle_submission(bundle).model_dump_json(),
        audit_trace=_AUDIT_TRACE,
    )
    batch = build_quarantined_batch((pending,))
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(batch, contract, signer)
    token = issue_gradient_admission_token(admission, signer)
    consumer = InMemoryAdmissionTokenConsumer()

    with pytest.raises(RewardFirewallQuarantine) as caught:
        _release(
            batch,
            contract,
            admission,
            token,
            signer,
            consumer,
            expected_model_sha256='f' * 64,
        )
    assert caught.value.reason is RewardFirewallReason.ADMISSION
    assert _release(batch, contract, admission, token, signer, consumer)[0].reward == 1.0


def test_released_token_is_single_use() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    batch = build_quarantined_batch(
        (
            _firewall(bundle, contract).quarantine(
                trajectory_id='trajectory-1',
                response=oracle_submission(bundle).model_dump_json(),
                audit_trace=_AUDIT_TRACE,
            ),
        )
    )
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(batch, contract, signer)
    token = issue_gradient_admission_token(admission, signer)
    consumer = InMemoryAdmissionTokenConsumer()
    _release(batch, contract, admission, token, signer, consumer)

    with pytest.raises(RewardFirewallQuarantine, match='already consumed'):
        _release(batch, contract, admission, token, signer, consumer)


def test_score_mutation_is_detected_before_token_consumption() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    pending = _firewall(bundle, contract).quarantine(
        trajectory_id='trajectory-1',
        response=oracle_submission(bundle).model_dump_json(),
        audit_trace=_AUDIT_TRACE,
    )
    batch = build_quarantined_batch((pending,))
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(batch, contract, signer)
    token = issue_gradient_admission_token(admission, signer)
    consumer = InMemoryAdmissionTokenConsumer()
    mutated = replace(
        batch,
        trajectories=(
            replace(
                pending,
                score=pending.score.model_copy(update={'reward': 0.9}),
            ),
        ),
    )
    assert isinstance(mutated, QuarantinedBatch)

    with pytest.raises(RewardFirewallQuarantine) as caught:
        _release(mutated, contract, admission, token, signer, consumer)
    assert caught.value.reason is RewardFirewallReason.BATCH_BINDING
    assert _release(batch, contract, admission, token, signer, consumer)[0].reward == 1.0


def test_release_replays_raw_response_instead_of_trusting_a_bound_oracle_score() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    pending = _firewall(bundle, contract).quarantine(
        trajectory_id='trajectory-1',
        response=oracle_submission(bundle).model_dump_json(),
        audit_trace=_AUDIT_TRACE,
    )
    substituted = uniform_submission(bundle)
    substituted_bytes = substituted.model_dump_json().encode()
    forged = replace(
        pending,
        envelope=pending.envelope.model_copy(
            update={
                'response_sha256': hashlib.sha256(substituted_bytes).hexdigest(),
                'submission_sha256': hashlib.sha256(canonical_json_bytes(substituted)).hexdigest(),
            }
        ),
        raw_response=substituted_bytes,
        submission=substituted,
    )
    batch = build_quarantined_batch((forged,))
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(batch, contract, signer)
    token = issue_gradient_admission_token(admission, signer)

    with pytest.raises(RewardFirewallQuarantine, match='score replay'):
        _release(
            batch,
            contract,
            admission,
            token,
            signer,
            InMemoryAdmissionTokenConsumer(),
        )


def test_batch_builder_revalidates_copied_score_schema() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    pending = _firewall(bundle, contract).quarantine(
        trajectory_id='trajectory-1',
        response=oracle_submission(bundle).model_dump_json(),
        audit_trace=_AUDIT_TRACE,
    )
    forged_score = pending.score.model_copy(update={'forecast_brier': -1.0, 'forecast_reward': 2.0, 'reward': 1.0})
    forged = replace(
        pending,
        envelope=pending.envelope.model_copy(
            update={'score_sha256': hashlib.sha256(canonical_json_bytes(forged_score)).hexdigest()}
        ),
        score=forged_score,
    )

    with pytest.raises(RewardFirewallQuarantine, match='schema replay'):
        build_quarantined_batch((forged,))
