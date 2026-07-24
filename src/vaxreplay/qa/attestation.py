"""Independent Ed25519 attestation of a complete reward-QA report."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.signing import Ed25519Signer, checked_signer
from vaxreplay.qa.schema import RewardQAReport, reward_qa_report_sha256

QA_REPORT_ATTESTATION_SCHEMA_VERSION = 'vaxreplay.reward-qa-report-attestation.v0.1'

_SIGNING_DOMAIN = b'vaxreplay.reward-qa-report-attestation.v0.1\x00'
_KEY_ID_DOMAIN = b'vaxreplay.reward-qa-key-id.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class QAReportAttestationError(ValueError):
    """A QA-report schema, identity, signature, or binding failed closed."""


class SignedRewardQAReport(StrictModel):
    schema_version: Literal['vaxreplay.reward-qa-report-attestation.v0.1'] = QA_REPORT_ATTESTATION_SCHEMA_VERSION
    reward_qa_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    qa_signing_key_id: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signature_base64: str = Field(min_length=88, max_length=88)


def qa_report_signing_key_id(public_key_bytes: bytes) -> str:
    if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
        raise QAReportAttestationError('trusted QA Ed25519 public key must contain 32 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes)
    except ValueError as error:
        raise QAReportAttestationError('trusted QA Ed25519 public key is invalid') from error
    return hashlib.sha256(_KEY_ID_DOMAIN + public_key_bytes).hexdigest()


def _revalidate_report(report: RewardQAReport) -> RewardQAReport:
    try:
        return RewardQAReport.model_validate_json(canonical_json_bytes(report))
    except (TypeError, ValueError) as error:
        raise QAReportAttestationError('reward QA report failed canonical schema revalidation') from error


def _payload(report_sha256: str) -> bytes:
    return _SIGNING_DOMAIN + bytes.fromhex(report_sha256)


def attest_reward_qa_report(
    report: RewardQAReport,
    signer: Ed25519Signer,
) -> SignedRewardQAReport:
    """Sign a canonically revalidated report with the independent QA identity."""

    report = _revalidate_report(report)
    signer = checked_signer(signer)
    public_key = signer.public_key_bytes()
    signer = checked_signer(signer, expected_public_key=public_key)
    key_id = qa_report_signing_key_id(public_key)
    report_sha256 = reward_qa_report_sha256(report)
    try:
        signature = signer.sign(_payload(report_sha256))
    except BaseException:
        raise QAReportAttestationError('QA report signer operation failed') from None
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise QAReportAttestationError('QA report signer returned an invalid signature length')
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            _payload(report_sha256),
        )
    except InvalidSignature:
        raise QAReportAttestationError('QA report signer returned a signature from another key') from None
    return SignedRewardQAReport(
        reward_qa_report_sha256=report_sha256,
        qa_signing_key_id=key_id,
        signature_base64=base64.b64encode(signature).decode('ascii'),
    )


def verify_reward_qa_report_attestation(
    report: RewardQAReport,
    attestation: SignedRewardQAReport,
    trusted_public_key_bytes: bytes,
) -> RewardQAReport:
    """Verify an out-of-band trust root and return the revalidated report."""

    report = _revalidate_report(report)
    try:
        attestation = SignedRewardQAReport.model_validate_json(canonical_json_bytes(attestation))
    except (TypeError, ValueError) as error:
        raise QAReportAttestationError('QA report attestation failed canonical schema revalidation') from error
    expected_key_id = qa_report_signing_key_id(trusted_public_key_bytes)
    if not hmac.compare_digest(attestation.qa_signing_key_id, expected_key_id):
        raise QAReportAttestationError('QA report attestation uses an untrusted key')
    report_sha256 = reward_qa_report_sha256(report)
    if not hmac.compare_digest(attestation.reward_qa_report_sha256, report_sha256):
        raise QAReportAttestationError('QA report attestation does not bind the supplied report')
    try:
        signature = base64.b64decode(attestation.signature_base64, validate=True)
    except ValueError as error:
        raise QAReportAttestationError('QA report signature is not canonical base64') from error
    if len(signature) != 64 or base64.b64encode(signature).decode() != attestation.signature_base64:
        raise QAReportAttestationError('QA report signature is not canonical Ed25519 bytes')
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key_bytes).verify(
            signature,
            _payload(report_sha256),
        )
    except (InvalidSignature, ValueError) as error:
        raise QAReportAttestationError('QA report signature verification failed') from error
    return report


def reward_qa_report_attestation_sha256(
    attestation: SignedRewardQAReport,
) -> str:
    try:
        attestation = SignedRewardQAReport.model_validate_json(canonical_json_bytes(attestation))
    except (TypeError, ValueError) as error:
        raise QAReportAttestationError('QA report attestation failed canonical schema revalidation') from error
    return hashlib.sha256(canonical_json_bytes(attestation)).hexdigest()


__all__ = [
    'QAReportAttestationError',
    'SignedRewardQAReport',
    'attest_reward_qa_report',
    'qa_report_signing_key_id',
    'reward_qa_report_attestation_sha256',
    'verify_reward_qa_report_attestation',
]
