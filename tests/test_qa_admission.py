from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.qa.admission import (
    GradientAdmissionError,
    InMemoryAdmissionTokenConsumer,
    gradient_admission_signing_key_id,
    issue_gradient_admission_token,
    verify_gradient_admission_token,
)
from vaxreplay.qa.attack_catalog import attack_catalog_sha256, default_attack_catalog
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

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
_DIGESTS = tuple(f'{value:064x}' for value in range(1, 20))


def _contract() -> RewardContract:
    catalog = default_attack_catalog()
    return RewardContract(
        contract_id='qa-contract-v0',
        reward_version='v0.1',
        scientific_objective='Rank vaccine targets using only decision-time evidence.',
        scorer_sha256=_DIGESTS[0],
        reference_scorer_sha256=_DIGESTS[1],
        prompt_compiler_sha256=_DIGESTS[2],
        qa_policy_sha256=_DIGESTS[3],
        attack_catalog_sha256=attack_catalog_sha256(catalog),
        required_dimensions=('feasibility', 'immunogenicity'),
        required_attack_ids=tuple(attack.attack_id for attack in catalog.attacks),
        component_floors=(
            ComponentFloorSpec(metric='assessment_accuracy', minimum=0.5),
            ComponentFloorSpec(metric='grounding_f1', minimum=0.5),
        ),
        prohibited_strategies=(
            'candidate identity lookup',
            'post-cutoff evidence access',
            'reward or evaluator tampering',
        ),
    )


def _findings(contract: RewardContract, *, failed_check: str | None = None) -> tuple[RewardQAFinding, ...]:
    values = [
        RewardQAFinding(
            finding_id=f'finding-{index:02d}',
            check_id=check_id,
            category=(
                QACategory.TEMPORAL_INTEGRITY
                if check_id == 'temporal.post-cutoff-lure'
                else QACategory.COUNTERFACTUAL_CONSISTENCY
            ),
            severity=QASeverity.FATAL if check_id == failed_check else QASeverity.INFO,
            passed=check_id != failed_check,
            disposition_on_failure=(QADisposition.REJECT if check_id == failed_check else QADisposition.QUARANTINE),
            detail='deterministic seeded QA result',
            artifact_sha256=f'{index + 100:064x}',
        )
        for index, check_id in enumerate(contract.required_attack_ids)
    ]
    return tuple(sorted(values, key=lambda finding: finding.finding_id))


def _report(
    *,
    grounding_f1: float = 1.0,
    failed_check: str | None = None,
    disposition: QADisposition = QADisposition.ADMIT,
) -> RewardQAReport:
    contract = _contract()
    return RewardQAReport(
        report_id='report-1',
        run_id='run-1',
        created_at=_NOW,
        trajectory_batch_sha256=_DIGESTS[4],
        reward_artifact_sha256=_DIGESTS[5],
        episode_manifest_sha256s=(_DIGESTS[6], _DIGESTS[7]),
        reward_contract=contract,
        reward_contract_sha256=reward_contract_sha256(contract),
        component_scores=(
            ComponentScore(metric='assessment_accuracy', value=1.0),
            ComponentScore(metric='grounding_f1', value=grounding_f1),
        ),
        findings=_findings(contract, failed_check=failed_check),
        disposition=disposition,
        all_required_attacks_executed=True,
        independent_scorer_agreement=True,
        future_taint_reachability_zero=True,
        exact_replay=True,
        tamper_success_zero=True,
        item_level_private_feedback_withheld=True,
    )


def _admission(signer: LocalEd25519Signer) -> TrainingRunAdmission:
    report = _report()
    return TrainingRunAdmission(
        admission_id='admission-1',
        run_id=report.run_id,
        issued_at=_NOW,
        not_before=_NOW,
        expires_at=_NOW + timedelta(hours=1),
        trajectory_batch_sha256=report.trajectory_batch_sha256,
        reward_artifact_sha256=report.reward_artifact_sha256,
        model_sha256=_DIGESTS[8],
        harness_sha256=_DIGESTS[9],
        tool_policy_sha256=_DIGESTS[10],
        environment_sha256=_DIGESTS[11],
        dataset_sha256=_DIGESTS[12],
        optimizer_config_sha256=_DIGESTS[13],
        episode_manifest_sha256s=report.episode_manifest_sha256s,
        reward_qa_report=report,
        reward_qa_report_sha256=reward_qa_report_sha256(report),
        reward_qa_attestation_sha256=_DIGESTS[14],
        qa_signing_key_id=_DIGESTS[15],
        reward_contract_sha256=report.reward_contract_sha256,
        attack_catalog_sha256=report.reward_contract.attack_catalog_sha256,
        signing_key_id=gradient_admission_signing_key_id(signer.public_key_bytes()),
    )


def _verify(
    token,
    admission,
    public_key: bytes,
    consumer,
    **overrides,
):
    expected = {
        'now': _NOW + timedelta(minutes=1),
        'consume_token': consumer,
        'expected_run_id': admission.run_id,
        'expected_trajectory_batch_sha256': admission.trajectory_batch_sha256,
        'expected_reward_artifact_sha256': admission.reward_artifact_sha256,
        'expected_model_sha256': admission.model_sha256,
        'expected_harness_sha256': admission.harness_sha256,
        'expected_tool_policy_sha256': admission.tool_policy_sha256,
        'expected_environment_sha256': admission.environment_sha256,
        'expected_dataset_sha256': admission.dataset_sha256,
        'expected_optimizer_config_sha256': admission.optimizer_config_sha256,
        'expected_reward_contract_sha256': admission.reward_contract_sha256,
        'expected_episode_manifest_sha256s': admission.episode_manifest_sha256s,
    }
    expected.update(overrides)
    return verify_gradient_admission_token(token, admission, public_key, **expected)


def test_admitted_report_can_issue_and_consume_one_gradient_token() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)
    consumer = InMemoryAdmissionTokenConsumer()

    assert _verify(token, admission, signer.public_key_bytes(), consumer) == admission
    with pytest.raises(GradientAdmissionError, match='already consumed'):
        _verify(token, admission, signer.public_key_bytes(), consumer)


def test_low_level_issuer_revalidates_model_copy_mutated_admission() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    failed_finding = admission.reward_qa_report.findings[0].model_copy(update={'passed': False})
    invalid_report = admission.reward_qa_report.model_copy(update={'findings': (failed_finding,)})
    invalid_admission = admission.model_copy(
        update={
            'reward_qa_report': invalid_report,
            'reward_qa_report_sha256': reward_qa_report_sha256(invalid_report),
        }
    )

    with pytest.raises(GradientAdmissionError, match='schema revalidation during issuance'):
        issue_gradient_admission_token(invalid_admission, signer)


def test_verifier_revalidates_model_copy_mutated_inputs() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)
    failed_finding = admission.reward_qa_report.findings[0].model_copy(update={'passed': False})
    invalid_report = admission.reward_qa_report.model_copy(update={'findings': (failed_finding,)})
    invalid_admission = admission.model_copy(
        update={
            'reward_qa_report': invalid_report,
            'reward_qa_report_sha256': reward_qa_report_sha256(invalid_report),
        }
    )

    with pytest.raises(GradientAdmissionError, match='schema revalidation during verification'):
        _verify(
            token,
            invalid_admission,
            signer.public_key_bytes(),
            InMemoryAdmissionTokenConsumer(),
        )

    invalid_token = token.model_copy(update={'signature_algorithm': 'not-ed25519'})
    with pytest.raises(GradientAdmissionError, match='token failed canonical schema revalidation'):
        _verify(
            invalid_token,
            admission,
            signer.public_key_bytes(),
            InMemoryAdmissionTokenConsumer(),
        )


def test_in_memory_consumer_is_globally_single_use_by_token_id_and_thread_safe() -> None:
    consumer = InMemoryAdmissionTokenConsumer()
    token_id = 'a' * 32

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = tuple(
            executor.map(
                lambda admission_sha256: consumer(token_id, admission_sha256),
                (f'{value:064x}' for value in range(64)),
            )
        )

    assert results.count(True) == 1
    assert results.count(False) == 63


def test_admission_and_token_have_canonical_json_round_trips() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)

    assert TrainingRunAdmission.model_validate_json(canonical_json_bytes(admission)) == admission
    assert type(token).model_validate_json(canonical_json_bytes(token)) == token


def test_tampered_token_or_untrusted_key_fails_before_consumption() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)
    consumer = InMemoryAdmissionTokenConsumer()

    tampered = token.model_copy(update={'signature_base64': 'A' * 88})
    with pytest.raises(GradientAdmissionError, match='signature'):
        _verify(tampered, admission, signer.public_key_bytes(), consumer)
    other = LocalEd25519Signer(Ed25519PrivateKey.generate())
    with pytest.raises(GradientAdmissionError, match='untrusted key'):
        _verify(token, admission, other.public_key_bytes(), consumer)
    assert _verify(token, admission, signer.public_key_bytes(), consumer) == admission


def test_issuer_rejects_signature_from_a_different_key() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    other = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)

    class SubstitutingSigner:
        def public_key_bytes(self) -> bytes:
            return signer.public_key_bytes()

        def sign(self, message: bytes) -> bytes:
            return other.sign(message)

    with pytest.raises(GradientAdmissionError, match='another key'):
        issue_gradient_admission_token(admission, SubstitutingSigner())


@pytest.mark.parametrize(
    ('field', 'argument'),
    [
        ('run_id', 'expected_run_id'),
        ('trajectory_batch_sha256', 'expected_trajectory_batch_sha256'),
        ('reward_artifact_sha256', 'expected_reward_artifact_sha256'),
        ('model_sha256', 'expected_model_sha256'),
        ('harness_sha256', 'expected_harness_sha256'),
        ('tool_policy_sha256', 'expected_tool_policy_sha256'),
        ('environment_sha256', 'expected_environment_sha256'),
        ('dataset_sha256', 'expected_dataset_sha256'),
        ('optimizer_config_sha256', 'expected_optimizer_config_sha256'),
        ('reward_contract_sha256', 'expected_reward_contract_sha256'),
        ('episode_manifest_sha256s', 'expected_episode_manifest_sha256s'),
    ],
)
def test_every_runtime_binding_fails_closed(field: str, argument: str) -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)
    wrong = ('f' * 64,) if field == 'episode_manifest_sha256s' else 'f' * 64
    if field == 'run_id':
        wrong = 'wrong-run'

    with pytest.raises(GradientAdmissionError, match=field):
        _verify(
            token,
            admission,
            signer.public_key_bytes(),
            InMemoryAdmissionTokenConsumer(),
            **{argument: wrong},
        )


def test_expired_or_not_yet_valid_admission_fails_closed() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    admission = _admission(signer)
    token = issue_gradient_admission_token(admission, signer)

    for when in (_NOW - timedelta(seconds=1), admission.expires_at):
        with pytest.raises(GradientAdmissionError, match='not currently valid'):
            _verify(
                token,
                admission,
                signer.public_key_bytes(),
                InMemoryAdmissionTokenConsumer(),
                now=when,
            )


def test_component_floor_and_fatal_attack_derive_non_admit_dispositions() -> None:
    quarantined = _report(grounding_f1=0.0, disposition=QADisposition.QUARANTINE)
    assert quarantined.disposition == QADisposition.QUARANTINE
    rejected = _report(
        failed_check='temporal.post-cutoff-lure',
        disposition=QADisposition.REJECT,
    )
    assert rejected.disposition == QADisposition.REJECT

    with pytest.raises(ValueError, match='derived as quarantine'):
        _report(grounding_f1=0.0, disposition=QADisposition.ADMIT)
    with pytest.raises(ValueError, match='derived as reject'):
        _report(
            failed_check='temporal.post-cutoff-lure',
            disposition=QADisposition.ADMIT,
        )


def test_non_admitted_report_cannot_become_training_admission() -> None:
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    report = _report(grounding_f1=0.0, disposition=QADisposition.QUARANTINE)
    admitted = _admission(signer)

    with pytest.raises(ValueError, match='only an admitted QA report'):
        TrainingRunAdmission(
            **admitted.model_dump(
                exclude={'reward_qa_report', 'reward_qa_report_sha256'},
            ),
            reward_qa_report=report,
            reward_qa_report_sha256=reward_qa_report_sha256(report),
        )
