from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import EpisodeBundle
from vaxreplay.operations.signing import LocalEd25519Signer
from vaxreplay.qa.attack_catalog import attack_catalog_sha256, default_attack_catalog
from vaxreplay.qa.attestation import attest_reward_qa_report
from vaxreplay.qa.authority import TrainingRuntimeBindings, issue_training_batch_admission
from vaxreplay.qa.firewall import TrainingRewardFirewall, build_quarantined_batch
from vaxreplay.qa.reporting import make_reward_qa_report
from vaxreplay.qa.schema import (
    ComponentFloorSpec,
    QACategory,
    QADisposition,
    QASeverity,
    RewardContract,
    RewardQAFinding,
)
from vaxreplay.scoring import LocalSubmissionEvaluator

_NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _materials():
    bundle = EpisodeBundle.load(
        Path(__file__).parent / 'fixtures' / 'synthetic_antigen_v0',
        include_private=True,
    )
    catalog = default_attack_catalog()
    contract = RewardContract(
        contract_id='authority-contract',
        reward_version='v0.1',
        scientific_objective='Grounded prioritization.',
        scorer_sha256='1' * 64,
        reference_scorer_sha256='2' * 64,
        prompt_compiler_sha256='3' * 64,
        qa_policy_sha256='4' * 64,
        attack_catalog_sha256=attack_catalog_sha256(catalog),
        required_dimensions=tuple(sorted(bundle.manifest.required_dimensions)),
        required_attack_ids=('parser.duplicate-key',),
        component_floors=(ComponentFloorSpec(metric='grounding_f1', minimum=0.01),),
        prohibited_strategies=('outcome lookup',),
    )
    firewall = TrainingRewardFirewall(
        run_id='run',
        bundle=bundle,
        primary_evaluator=LocalSubmissionEvaluator(bundle),
        reference_evaluator=LocalSubmissionEvaluator(bundle),
        primary_scorer_sha256=contract.scorer_sha256,
        reference_scorer_sha256=contract.reference_scorer_sha256,
        reward_contract=contract,
    )
    batch = build_quarantined_batch(
        (
            firewall.quarantine(
                trajectory_id='trajectory',
                response=oracle_submission(bundle).model_dump_json(),
                audit_trace=b'{"events":[],"recorder":"test"}',
            ),
        )
    )
    finding = RewardQAFinding(
        finding_id='parser-pass',
        check_id='parser.duplicate-key',
        category=QACategory.PARSER_INTEGRITY,
        severity=QASeverity.INFO,
        passed=True,
        disposition_on_failure=QADisposition.REJECT,
        detail='duplicate members rejected',
        artifact_sha256='5' * 64,
    )
    report = make_reward_qa_report(
        report_id='report',
        run_id=batch.envelope.run_id,
        created_at=_NOW,
        trajectory_batch_sha256=batch.envelope.trajectory_batch_sha256,
        reward_artifact_sha256=batch.envelope.reward_artifact_sha256,
        episode_manifest_sha256s=batch.envelope.episode_manifest_sha256s,
        reward_contract=contract,
        component_scores={'grounding_f1': 1.0},
        findings=(finding,),
        all_required_attacks_executed=True,
        independent_scorer_agreement=True,
        future_taint_reachability_zero=True,
        exact_replay=True,
        tamper_success_zero=True,
        item_level_private_feedback_withheld=True,
    )
    runtime = TrainingRuntimeBindings(
        model_sha256='6' * 64,
        harness_sha256='7' * 64,
        tool_policy_sha256='8' * 64,
        environment_sha256='9' * 64,
        dataset_sha256='a' * 64,
        optimizer_config_sha256='b' * 64,
    )
    return batch, report, runtime, catalog


def test_authority_signs_only_an_exact_admitted_batch() -> None:
    batch, report, runtime, catalog = _materials()
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    qa_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())

    grant = issue_training_batch_admission(
        batch,
        report,
        runtime,
        signer,
        report_attestation=attest_reward_qa_report(report, qa_signer),
        trusted_qa_public_key_bytes=qa_signer.public_key_bytes(),
        attack_catalog=catalog,
        admission_id='admission',
        issued_at=_NOW,
        not_before=_NOW,
        expires_at=_NOW + timedelta(hours=1),
    )

    assert grant.admission.trajectory_batch_sha256 == batch.envelope.trajectory_batch_sha256
    assert grant.token.training_run_admission_sha256


def test_authority_rejects_report_for_different_batch() -> None:
    batch, report, runtime, catalog = _materials()
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    qa_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    mismatched = report.model_copy(update={'trajectory_batch_sha256': 'f' * 64})

    with pytest.raises(ValueError, match='trajectory_batch_sha256'):
        issue_training_batch_admission(
            batch,
            mismatched,
            runtime,
            signer,
            report_attestation=attest_reward_qa_report(mismatched, qa_signer),
            trusted_qa_public_key_bytes=qa_signer.public_key_bytes(),
            attack_catalog=catalog,
            admission_id='admission',
            issued_at=_NOW,
            not_before=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        )


def test_authority_rejects_untrusted_qa_attestation() -> None:
    batch, report, runtime, catalog = _materials()
    admission_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    qa_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    untrusted = LocalEd25519Signer(Ed25519PrivateKey.generate())

    with pytest.raises(ValueError, match='untrusted key'):
        issue_training_batch_admission(
            batch,
            report,
            runtime,
            admission_signer,
            report_attestation=attest_reward_qa_report(report, qa_signer),
            trusted_qa_public_key_bytes=untrusted.public_key_bytes(),
            attack_catalog=catalog,
            admission_id='admission',
            issued_at=_NOW,
            not_before=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        )


def test_authority_requires_separate_qa_and_gradient_signing_keys() -> None:
    batch, report, runtime, catalog = _materials()
    signer = LocalEd25519Signer(Ed25519PrivateKey.generate())

    with pytest.raises(ValueError, match='separate signing keys'):
        issue_training_batch_admission(
            batch,
            report,
            runtime,
            signer,
            report_attestation=attest_reward_qa_report(report, signer),
            trusted_qa_public_key_bytes=signer.public_key_bytes(),
            attack_catalog=catalog,
            admission_id='admission',
            issued_at=_NOW,
            not_before=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        )


def test_authority_rejects_catalog_not_bound_by_contract() -> None:
    batch, report, runtime, catalog = _materials()
    admission_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    qa_signer = LocalEd25519Signer(Ed25519PrivateKey.generate())
    first = catalog.attacks[0]
    changed_catalog = catalog.model_copy(
        update={
            'attacks': (
                first.model_copy(update={'description': 'tampered description'}),
                *catalog.attacks[1:],
            )
        }
    )

    with pytest.raises(ValueError, match='does not bind'):
        issue_training_batch_admission(
            batch,
            report,
            runtime,
            admission_signer,
            report_attestation=attest_reward_qa_report(report, qa_signer),
            trusted_qa_public_key_bytes=qa_signer.public_key_bytes(),
            attack_catalog=changed_catalog,
            admission_id='admission',
            issued_at=_NOW,
            not_before=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        )
