"""Admission-authority composition for an independently completed QA report."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.qa.admission import (
    gradient_admission_signing_key_id,
    issue_gradient_admission_token,
)
from vaxreplay.qa.attack_catalog import (
    AttackCatalog,
    ExpectedDisposition,
    attack_catalog_sha256,
)
from vaxreplay.qa.attestation import (
    SignedRewardQAReport,
    reward_qa_report_attestation_sha256,
    verify_reward_qa_report_attestation,
)
from vaxreplay.qa.firewall import QuarantinedBatch
from vaxreplay.qa.schema import (
    GradientAdmissionToken,
    QADisposition,
    RewardQAFinding,
    RewardQAReport,
    TrainingRunAdmission,
    reward_qa_report_sha256,
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class TrainingRuntimeBindings(StrictModel):
    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_sha256: str = Field(pattern=_SHA256_PATTERN)
    tool_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    optimizer_config_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class SignedGradientAdmission:
    admission: TrainingRunAdmission
    token: GradientAdmissionToken


def issue_training_batch_admission(
    batch: QuarantinedBatch,
    report: RewardQAReport,
    runtime: TrainingRuntimeBindings,
    signer: Ed25519Signer,
    *,
    report_attestation: SignedRewardQAReport,
    trusted_qa_public_key_bytes: bytes,
    attack_catalog: AttackCatalog,
    admission_id: str,
    issued_at: datetime,
    not_before: datetime,
    expires_at: datetime,
    token_id: str | None = None,
) -> SignedGradientAdmission:
    """Bind and sign one exact admitted batch; never construct or soften QA."""

    report_attestation = SignedRewardQAReport.model_validate_json(canonical_json_bytes(report_attestation))
    report = verify_reward_qa_report_attestation(
        report,
        report_attestation,
        trusted_qa_public_key_bytes,
    )
    try:
        attack_catalog = AttackCatalog.model_validate_json(attack_catalog.model_dump_json())
    except (TypeError, ValueError) as error:
        raise ValueError('attack catalog failed canonical schema revalidation') from error
    if report.disposition != QADisposition.ADMIT:
        raise ValueError('admission authority cannot sign a quarantined or rejected report')
    if attack_catalog_sha256(attack_catalog) != report.reward_contract.attack_catalog_sha256:
        raise ValueError('supplied attack catalog does not bind the reward contract')
    _require_catalog_coverage(report, attack_catalog)
    envelope = batch.envelope
    checks = (
        ('run_id', report.run_id, envelope.run_id),
        (
            'trajectory_batch_sha256',
            report.trajectory_batch_sha256,
            envelope.trajectory_batch_sha256,
        ),
        (
            'reward_artifact_sha256',
            report.reward_artifact_sha256,
            envelope.reward_artifact_sha256,
        ),
        (
            'episode_manifest_sha256s',
            report.episode_manifest_sha256s,
            envelope.episode_manifest_sha256s,
        ),
        (
            'reward_contract_sha256',
            report.reward_contract_sha256,
            envelope.reward_contract_sha256,
        ),
    )
    for field_name, observed, expected in checks:
        if observed != expected:
            raise ValueError(f'QA report {field_name} does not bind the quarantined batch')

    signer = checked_signer(signer)
    admission_public_key = signer.public_key_bytes()
    signer = checked_signer(signer, expected_public_key=admission_public_key)
    signing_key_id = gradient_admission_signing_key_id(admission_public_key)
    if hmac.compare_digest(
        admission_public_key,
        trusted_qa_public_key_bytes,
    ):
        raise ValueError('QA report and gradient admission must use separate signing keys')
    admission = TrainingRunAdmission(
        admission_id=admission_id,
        run_id=envelope.run_id,
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        trajectory_batch_sha256=envelope.trajectory_batch_sha256,
        reward_artifact_sha256=envelope.reward_artifact_sha256,
        model_sha256=runtime.model_sha256,
        harness_sha256=runtime.harness_sha256,
        tool_policy_sha256=runtime.tool_policy_sha256,
        environment_sha256=runtime.environment_sha256,
        dataset_sha256=runtime.dataset_sha256,
        optimizer_config_sha256=runtime.optimizer_config_sha256,
        episode_manifest_sha256s=envelope.episode_manifest_sha256s,
        reward_qa_report=report,
        reward_qa_report_sha256=reward_qa_report_sha256(report),
        reward_qa_attestation_sha256=reward_qa_report_attestation_sha256(report_attestation),
        qa_signing_key_id=report_attestation.qa_signing_key_id,
        reward_contract_sha256=report.reward_contract_sha256,
        attack_catalog_sha256=report.reward_contract.attack_catalog_sha256,
        signing_key_id=signing_key_id,
    )
    return SignedGradientAdmission(
        admission=admission,
        token=issue_gradient_admission_token(
            admission,
            signer,
            token_id=token_id,
            expected_signer_public_key_bytes=admission_public_key,
        ),
    )


def _require_catalog_coverage(
    report: RewardQAReport,
    attack_catalog: AttackCatalog,
) -> None:
    catalog_by_id = {attack.attack_id: attack for attack in attack_catalog.attacks}
    missing_catalog = set(report.reward_contract.required_attack_ids) - set(catalog_by_id)
    if missing_catalog:
        raise ValueError(f'reward contract requires attacks absent from the catalog {sorted(missing_catalog)}')
    findings_by_check: dict[str, list[RewardQAFinding]] = {}
    for finding in report.findings:
        findings_by_check.setdefault(finding.check_id, []).append(finding)
    for check_id in report.reward_contract.required_attack_ids:
        findings = findings_by_check.get(check_id, [])
        if len(findings) != 1:
            raise ValueError(f'QA report must contain exactly one finding for required check {check_id}')
        expected = catalog_by_id[check_id].expected_disposition
        expected_failure = QADisposition.REJECT if expected == ExpectedDisposition.REJECT else QADisposition.QUARANTINE
        if findings[0].disposition_on_failure != expected_failure:
            raise ValueError(f'QA finding {check_id} failure disposition differs from the bound catalog')
