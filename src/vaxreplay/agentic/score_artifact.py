"""Authenticated leaderboard handoff for deterministic Agentic Replay scores.

The scorer remains a pure function.  This module is the trusted organizer boundary that
re-authenticates an accepted run, verifies that its deterministic score is bound to the exact
submission, admission, task, workspace, scoring contract, and private-gold commitment, and then
emits an immutable exact-inventory score artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.admission import AgenticWorkspaceAdmission
from vaxreplay.agentic.protocol import (
    AGENTIC_RECEIPT_AUTHENTICATION,
    AgenticRunReceipt,
    agentic_receipt_key_id,
    agentic_run_receipt_hmac,
)
from vaxreplay.agentic.run_artifact import LoadedAgenticRunArtifact
from vaxreplay.agentic.schema import AgenticAssuranceProfile, agentic_model_sha256
from vaxreplay.agentic.scoring import (
    AGENTIC_REWARD_VERSION,
    AGENTIC_SCORE_SCHEMA_VERSION,
    AgenticPrivateGoldV1,
    AgenticScoreVectorV1,
    AgenticScoringContract,
    AgenticSubmissionV1,
    score_agentic_submission,
)
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import ScoreStatus, StrictModel
from vaxreplay.runner.schema import IsolationTier

AGENTIC_SCORE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.agentic-score-receipt.v0.2'
AGENTIC_SCORE_RECEIPT_AUTHENTICATION = 'hmac-sha256-domain-separated-agentic-score'
_SCORE_RECEIPT_HMAC_DOMAIN = b'vaxreplay.agentic-score-receipt.v0.2\x00'
_SCORE_RECEIPT_KEY_ID_DOMAIN = b'vaxreplay.agentic-score-receipt-key-id.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SCORE_FILES = {'score-receipt.json', 'score-receipt.hmac', 'score-vector.json'}
_RUN_FILES = {
    'run.json',
    'run.hmac',
    'transcript.json',
    'tool-events.json',
    'scratch-manifest.json',
    'submission.json',
    'workspace-broker-attestation.json',
}
_MAX_JSON_BYTES = 64 * 1024 * 1024


class AgenticScoreArtifactError(ValueError):
    """Raised when a leaderboard score handoff is unauthenticated or can be spliced."""


class AgenticLeaderboardScoreReceipt(StrictModel):
    """Organizer-authenticated binding of one deterministic score to one terminal run."""

    schema_version: Literal['vaxreplay.agentic-score-receipt.v0.2'] = AGENTIC_SCORE_RECEIPT_SCHEMA_VERSION
    receipt_authentication: Literal['hmac-sha256-domain-separated-agentic-score'] = AGENTIC_SCORE_RECEIPT_AUTHENTICATION
    receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    finalized_at: datetime

    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    run_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_receipt_authentication: Literal['hmac-sha256-domain-separated'] = AGENTIC_RECEIPT_AUTHENTICATION
    run_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    attempt_reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    final_submission_sha256: str = Field(pattern=_SHA256_PATTERN)

    score_schema_version: Literal['vaxreplay.agentic-score.v1'] = AGENTIC_SCORE_SCHEMA_VERSION
    score_vector_sha256: str = Field(pattern=_SHA256_PATTERN)
    scoring_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_gold_commitment_key_id: str = Field(pattern=_SHA256_PATTERN)
    reward_version: Literal['vaxreplay.agentic-reward.v1.0'] = AGENTIC_REWARD_VERSION

    task_id: str = Field(min_length=1)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    assurance_profile: AgenticAssuranceProfile
    admitted_use: Literal['prospective_research', 'retrospective_research', 'best_effort_research', 'fixture']

    harness_id: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    harness_image_or_commitment: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    harness_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_behavior_sha256: str = Field(pattern=_SHA256_PATTERN)
    harness_execution_mode: Literal['fixed_model_loop', 'submitted_guest_agent']
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    adapter_id: str = Field(min_length=1)
    isolation_tier: IsolationTier
    sealed: bool
    development_only: bool

    status: ScoreStatus
    reward: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator('finalized_at')
    @classmethod
    def validate_finalized_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('finalized_at must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.receipt_key_id == self.run_receipt_key_id:
            raise ValueError('score and run receipts must use distinct authentication key IDs')
        expected_sealed = self.isolation_tier == IsolationTier.OFFICIAL
        if self.sealed != expected_sealed or self.development_only == self.sealed:
            raise ValueError('score receipt isolation flags are inconsistent')
        if (self.status == ScoreStatus.VALID) != (self.reward is not None):
            raise ValueError('only valid scores can contain a leaderboard reward')
        expected_use = {
            AgenticAssuranceProfile.PROSPECTIVE_EXACT: 'prospective_research',
            AgenticAssuranceProfile.INDEPENDENT_EXACT_BYTE: 'retrospective_research',
            AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT: 'best_effort_research',
            AgenticAssuranceProfile.FIXTURE: 'fixture',
        }[self.assurance_profile]
        if self.admitted_use != expected_use:
            raise ValueError('score receipt admitted_use must reflect its assurance profile')
        return self


@dataclass(frozen=True)
class LoadedAgenticScoreArtifact:
    root: Path
    receipt: AgenticLeaderboardScoreReceipt
    receipt_sha256: str
    score_vector: AgenticScoreVectorV1


def agentic_score_receipt_key_id(key: bytes) -> str:
    """Return the domain-separated public identifier for a leaderboard receipt key."""

    _require_hmac_key(key, 'Agentic score receipt')
    return hashlib.sha256(_SCORE_RECEIPT_KEY_ID_DOMAIN + key).hexdigest()


def agentic_score_receipt_hmac(receipt: AgenticLeaderboardScoreReceipt, key: bytes) -> str:
    """Authenticate a canonical score receipt in a domain distinct from run receipts."""

    _require_hmac_key(key, 'Agentic score receipt')
    return hmac.new(key, _SCORE_RECEIPT_HMAC_DOMAIN + canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def finalize_agentic_score(
    *,
    output_root: Path,
    run_artifact: LoadedAgenticRunArtifact,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    scoring_contract: AgenticScoringContract,
    expected_scoring_contract_sha256: str,
    gold: AgenticPrivateGoldV1,
    gold_commitment_key: bytes,
    expected_private_gold_commitment_sha256: str,
    expected_private_gold_commitment_key_id: str,
    run_receipt_key: bytes,
    expected_run_receipt_key_id: str,
    score_receipt_key: bytes,
    expected_score_receipt_key_id: str,
    finalized_at: datetime,
) -> LoadedAgenticScoreArtifact:
    """Authenticate one deterministic score after rechecking every leaderboard binding."""

    _validate_run_authentication(
        run_artifact,
        run_receipt_key=run_receipt_key,
        expected_run_receipt_key_id=expected_run_receipt_key_id,
    )
    if hmac.compare_digest(run_receipt_key, score_receipt_key):
        raise AgenticScoreArtifactError('run and score receipts must use distinct HMAC keys')
    score_key_id = agentic_score_receipt_key_id(score_receipt_key)
    if score_key_id != expected_score_receipt_key_id:
        raise AgenticScoreArtifactError('score receipt key does not match the release-pinned key ID')
    score_vector = _recompute_score(
        run_artifact=run_artifact,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        gold=gold,
        gold_commitment_key=gold_commitment_key,
        expected_private_gold_commitment_sha256=expected_private_gold_commitment_sha256,
    )
    _validate_score_bindings(
        run_artifact=run_artifact,
        score_vector=score_vector,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        scoring_contract=scoring_contract,
        expected_scoring_contract_sha256=expected_scoring_contract_sha256,
        expected_private_gold_commitment_sha256=expected_private_gold_commitment_sha256,
        expected_private_gold_commitment_key_id=expected_private_gold_commitment_key_id,
    )
    finalized_at = _aware_utc(finalized_at, 'finalized_at')
    if finalized_at < run_artifact.receipt.finished_at:
        raise AgenticScoreArtifactError('a score cannot be finalized before its run finished')
    receipt = _make_receipt(
        run_artifact=run_artifact,
        score_vector=score_vector,
        score_receipt_key_id=score_key_id,
        finalized_at=finalized_at,
    )

    target = output_root.expanduser().resolve()
    if target.is_relative_to(run_artifact.root.resolve()):
        raise AgenticScoreArtifactError('score artifact output cannot be nested inside its immutable run artifact')
    if target.exists():
        raise AgenticScoreArtifactError(f'score artifact output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        files = {
            'score-receipt.json': canonical_json_bytes(receipt),
            'score-receipt.hmac': (agentic_score_receipt_hmac(receipt, score_receipt_key) + '\n').encode('ascii'),
            'score-vector.json': canonical_json_bytes(score_vector),
        }
        for name, content in files.items():
            destination = staging / name
            destination.write_bytes(content)
            destination.chmod(0o600)
        staging.chmod(0o700)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_agentic_score_artifact(
        target,
        run_artifact=run_artifact,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        scoring_contract=scoring_contract,
        expected_scoring_contract_sha256=expected_scoring_contract_sha256,
        gold=gold,
        gold_commitment_key=gold_commitment_key,
        expected_private_gold_commitment_sha256=expected_private_gold_commitment_sha256,
        expected_private_gold_commitment_key_id=expected_private_gold_commitment_key_id,
        run_receipt_key=run_receipt_key,
        expected_run_receipt_key_id=expected_run_receipt_key_id,
        score_receipt_key=score_receipt_key,
        expected_score_receipt_key_id=expected_score_receipt_key_id,
    )


def load_agentic_score_artifact(
    root: Path,
    *,
    run_artifact: LoadedAgenticRunArtifact,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    scoring_contract: AgenticScoringContract,
    expected_scoring_contract_sha256: str,
    gold: AgenticPrivateGoldV1,
    gold_commitment_key: bytes,
    expected_private_gold_commitment_sha256: str,
    expected_private_gold_commitment_key_id: str,
    run_receipt_key: bytes,
    expected_run_receipt_key_id: str,
    score_receipt_key: bytes,
    expected_score_receipt_key_id: str,
) -> LoadedAgenticScoreArtifact:
    """Verify a private exact-inventory leaderboard score artifact and its parent run."""

    _validate_run_authentication(
        run_artifact,
        run_receipt_key=run_receipt_key,
        expected_run_receipt_key_id=expected_run_receipt_key_id,
    )
    if hmac.compare_digest(run_receipt_key, score_receipt_key):
        raise AgenticScoreArtifactError('run and score receipts must use distinct HMAC keys')
    resolved = _validate_private_root(root, expected_files=_SCORE_FILES, artifact_name='score artifact')
    receipt_bytes = _read_private_file(resolved / 'score-receipt.json', _MAX_JSON_BYTES)
    score_bytes = _read_private_file(resolved / 'score-vector.json', _MAX_JSON_BYTES)
    hmac_bytes = _read_private_file(resolved / 'score-receipt.hmac', 65)
    try:
        receipt = AgenticLeaderboardScoreReceipt.model_validate_json(receipt_bytes)
        score_vector = AgenticScoreVectorV1.model_validate_json(score_bytes)
    except ValueError as error:
        raise AgenticScoreArtifactError(f'invalid Agentic score artifact: {error}') from error
    if receipt_bytes != canonical_json_bytes(receipt) or score_bytes != canonical_json_bytes(score_vector):
        raise AgenticScoreArtifactError('score artifact JSON must use canonical encoding')

    score_key_id = agentic_score_receipt_key_id(score_receipt_key)
    if score_key_id != expected_score_receipt_key_id or receipt.receipt_key_id != score_key_id:
        raise AgenticScoreArtifactError('score receipt uses a different authentication key')
    expected_hmac = (agentic_score_receipt_hmac(receipt, score_receipt_key) + '\n').encode('ascii')
    if not hmac.compare_digest(hmac_bytes, expected_hmac):
        raise AgenticScoreArtifactError('score receipt HMAC authentication failed')
    if receipt.score_vector_sha256 != _sha256(score_bytes):
        raise AgenticScoreArtifactError('score receipt does not bind the exact score vector')
    recomputed_score_vector = _recompute_score(
        run_artifact=run_artifact,
        workspace=workspace,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        gold=gold,
        gold_commitment_key=gold_commitment_key,
        expected_private_gold_commitment_sha256=expected_private_gold_commitment_sha256,
    )
    if not hmac.compare_digest(score_bytes, canonical_json_bytes(recomputed_score_vector)):
        raise AgenticScoreArtifactError(
            'authenticated score vector does not equal deterministic private-gold recomputation'
        )
    _validate_score_bindings(
        run_artifact=run_artifact,
        score_vector=score_vector,
        admission=admission,
        expected_admission_sha256=expected_admission_sha256,
        scoring_contract=scoring_contract,
        expected_scoring_contract_sha256=expected_scoring_contract_sha256,
        expected_private_gold_commitment_sha256=expected_private_gold_commitment_sha256,
        expected_private_gold_commitment_key_id=expected_private_gold_commitment_key_id,
    )
    expected_receipt = _make_receipt(
        run_artifact=run_artifact,
        score_vector=score_vector,
        score_receipt_key_id=score_key_id,
        finalized_at=receipt.finalized_at,
    )
    if receipt != expected_receipt:
        raise AgenticScoreArtifactError('score receipt bindings do not match the authenticated run and score')
    if receipt.finalized_at < run_artifact.receipt.finished_at:
        raise AgenticScoreArtifactError('a score cannot be finalized before its run finished')
    return LoadedAgenticScoreArtifact(
        root=resolved,
        receipt=receipt,
        receipt_sha256=_sha256(receipt_bytes),
        score_vector=score_vector,
    )


def _recompute_score(
    *,
    run_artifact: LoadedAgenticRunArtifact,
    workspace: LoadedAgenticWorkspace,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    gold: AgenticPrivateGoldV1,
    gold_commitment_key: bytes,
    expected_private_gold_commitment_sha256: str,
) -> AgenticScoreVectorV1:
    """Derive the only score eligible for certification from authenticated inputs."""

    submission = run_artifact.submission
    if submission is None:
        raise AgenticScoreArtifactError('a score cannot be computed for a run without a final submission')
    try:
        return score_agentic_submission(
            workspace=workspace,
            admission=admission,
            expected_admission_sha256=expected_admission_sha256,
            gold=gold,
            gold_commitment_key=gold_commitment_key,
            expected_gold_commitment_sha256=expected_private_gold_commitment_sha256,
            submission=submission,
        )
    except ValueError as error:
        raise AgenticScoreArtifactError(f'deterministic private-gold score recomputation failed: {error}') from error


def _make_receipt(
    *,
    run_artifact: LoadedAgenticRunArtifact,
    score_vector: AgenticScoreVectorV1,
    score_receipt_key_id: str,
    finalized_at: datetime,
) -> AgenticLeaderboardScoreReceipt:
    run = run_artifact.receipt
    return AgenticLeaderboardScoreReceipt(
        receipt_key_id=score_receipt_key_id,
        finalized_at=finalized_at,
        run_id=run.run_id,
        run_receipt_sha256=run_artifact.receipt_sha256,
        run_receipt_key_id=run.receipt_key_id,
        attempt_reservation_sha256=run.attempt_reservation_sha256,
        final_submission_sha256=run.final_submission_sha256,
        score_vector_sha256=_sha256(canonical_json_bytes(score_vector)),
        scoring_contract_sha256=score_vector.scoring_contract_sha256,
        private_gold_commitment_sha256=score_vector.private_gold_commitment_sha256,
        private_gold_commitment_key_id=score_vector.private_gold_commitment_key_id,
        reward_version=score_vector.reward_version,
        task_id=run.task_id,
        workspace_manifest_sha256=run.workspace_manifest_sha256,
        workspace_admission_sha256=run.workspace_admission_sha256,
        assurance_profile=score_vector.assurance_profile,
        admitted_use=score_vector.admitted_use,
        harness_id=run.harness_id,
        harness_version=run.harness_version,
        harness_image_or_commitment=run.harness_image_or_commitment,
        harness_manifest_sha256=run.harness_manifest_sha256,
        harness_behavior_sha256=run.harness_behavior_sha256,
        harness_execution_mode=run.harness_execution_mode,
        requested_model_id=run.requested_model_id,
        resolved_model_id=run.resolved_model_id,
        adapter_id=run.adapter_id,
        isolation_tier=run.isolation_tier,
        sealed=run.sealed,
        development_only=run.development_only,
        status=score_vector.status,
        reward=score_vector.reward,
    )


def _validate_run_authentication(
    run_artifact: LoadedAgenticRunArtifact,
    *,
    run_receipt_key: bytes,
    expected_run_receipt_key_id: str,
) -> None:
    """Re-authenticate the run at score-finalization time to close post-load mutation gaps."""

    resolved = _validate_private_root(run_artifact.root, expected_files=_RUN_FILES, artifact_name='run artifact')
    receipt_bytes = _read_private_file(resolved / 'run.json', _MAX_JSON_BYTES)
    hmac_bytes = _read_private_file(resolved / 'run.hmac', 65)
    submission_bytes = _read_private_file(resolved / 'submission.json', _MAX_JSON_BYTES)
    transcript_bytes = _read_private_file(resolved / 'transcript.json', _MAX_JSON_BYTES)
    tool_event_bytes = _read_private_file(resolved / 'tool-events.json', _MAX_JSON_BYTES)
    scratch_bytes = _read_private_file(resolved / 'scratch-manifest.json', _MAX_JSON_BYTES)
    broker_attestation_bytes = _read_private_file(
        resolved / 'workspace-broker-attestation.json',
        _MAX_JSON_BYTES,
    )
    try:
        receipt = AgenticRunReceipt.model_validate_json(receipt_bytes)
        submission = AgenticSubmissionV1.model_validate_json(submission_bytes)
    except ValueError as error:
        raise AgenticScoreArtifactError(f'invalid authenticated Agentic run: {error}') from error
    if receipt_bytes != canonical_json_bytes(receipt) or submission_bytes != canonical_json_bytes(submission):
        raise AgenticScoreArtifactError('authenticated run receipt and submission must use canonical JSON')
    if receipt != run_artifact.receipt or submission != run_artifact.submission:
        raise AgenticScoreArtifactError('loaded run object does not match its on-disk authenticated artifact')
    if _sha256(receipt_bytes) != run_artifact.receipt_sha256:
        raise AgenticScoreArtifactError('loaded run receipt SHA does not match its exact receipt bytes')
    run_key_id = agentic_receipt_key_id(run_receipt_key)
    if run_key_id != expected_run_receipt_key_id or receipt.receipt_key_id != run_key_id:
        raise AgenticScoreArtifactError('run receipt uses a different authentication key')
    expected_hmac = (agentic_run_receipt_hmac(receipt, run_receipt_key) + '\n').encode('ascii')
    if not hmac.compare_digest(hmac_bytes, expected_hmac):
        raise AgenticScoreArtifactError('run receipt HMAC authentication failed')
    if not receipt.accepted or receipt.failure_code is not None or run_artifact.submission is None:
        raise AgenticScoreArtifactError('only an accepted run with a final submission can be scored')
    if receipt.final_submission_sha256 != _sha256(submission_bytes):
        raise AgenticScoreArtifactError('authenticated run receipt does not bind its exact final submission')
    if (
        receipt.transcript_sha256,
        receipt.tool_events_sha256,
        receipt.scratch_tree_sha256,
        receipt.workspace_broker_attestation_sha256,
    ) != (
        _sha256(transcript_bytes),
        _sha256(tool_event_bytes),
        _sha256(scratch_bytes),
        _sha256(broker_attestation_bytes),
    ):
        raise AgenticScoreArtifactError('authenticated run component hashes do not match the exact handoff files')
    if (
        transcript_bytes != canonical_json_bytes(run_artifact.transcript)
        or tool_event_bytes
        != canonical_json_bytes([event.model_dump(mode='json') for event in run_artifact.tool_events])
        or scratch_bytes
        != canonical_json_bytes([entry.model_dump(mode='json') for entry in run_artifact.scratch_manifest])
        or broker_attestation_bytes != canonical_json_bytes(run_artifact.workspace_broker_attestation)
    ):
        raise AgenticScoreArtifactError('loaded run object does not match all authenticated handoff components')


def _validate_score_bindings(
    *,
    run_artifact: LoadedAgenticRunArtifact,
    score_vector: AgenticScoreVectorV1,
    admission: AgenticWorkspaceAdmission,
    expected_admission_sha256: str,
    scoring_contract: AgenticScoringContract,
    expected_scoring_contract_sha256: str,
    expected_private_gold_commitment_sha256: str,
    expected_private_gold_commitment_key_id: str,
) -> None:
    _require_sha256(expected_admission_sha256, 'expected admission')
    _require_sha256(expected_scoring_contract_sha256, 'expected scoring contract')
    _require_sha256(expected_private_gold_commitment_sha256, 'expected private-gold commitment')
    _require_sha256(expected_private_gold_commitment_key_id, 'expected private-gold commitment key ID')
    run = run_artifact.receipt
    admission_sha256 = agentic_model_sha256(admission)
    contract_sha256 = agentic_model_sha256(scoring_contract)
    submission = run_artifact.submission
    if submission is None:
        raise AgenticScoreArtifactError('a score cannot be bound to a run without a final submission')
    submission_sha256 = _sha256(canonical_json_bytes(submission))
    if (
        admission_sha256 != expected_admission_sha256
        or run.workspace_admission_sha256 != expected_admission_sha256
        or score_vector.workspace_admission_sha256 != expected_admission_sha256
    ):
        raise AgenticScoreArtifactError('score, run, and admission commitments do not match')
    if (
        score_vector.assurance_profile != admission.assurance_profile
        or score_vector.admitted_use != admission.admitted_use
    ):
        raise AgenticScoreArtifactError('score assurance and admitted use do not match the authenticated admission')
    if (
        score_vector.task_id != run.task_id
        or scoring_contract.task_id != run.task_id
        or submission.task_id != run.task_id
    ):
        raise AgenticScoreArtifactError('score, contract, submission, and run task IDs do not match')
    if (
        score_vector.workspace_manifest_sha256 != run.workspace_manifest_sha256
        or scoring_contract.workspace_manifest_sha256 != run.workspace_manifest_sha256
        or submission.workspace_manifest_sha256 != run.workspace_manifest_sha256
    ):
        raise AgenticScoreArtifactError('score, contract, submission, and run workspace commitments do not match')
    if score_vector.submission_sha256 != submission_sha256 or run.final_submission_sha256 != submission_sha256:
        raise AgenticScoreArtifactError('score and run are bound to different final submissions')
    if (
        contract_sha256 != expected_scoring_contract_sha256
        or score_vector.scoring_contract_sha256 != expected_scoring_contract_sha256
    ):
        raise AgenticScoreArtifactError('score does not match the release-pinned scoring contract')
    if (
        score_vector.private_gold_commitment_sha256 != expected_private_gold_commitment_sha256
        or score_vector.private_gold_commitment_key_id != expected_private_gold_commitment_key_id
    ):
        raise AgenticScoreArtifactError('score does not match the release-pinned private-gold commitment')
    if score_vector.reward_version != scoring_contract.reward_version:
        raise AgenticScoreArtifactError('score and scoring contract use different reward versions')


def _validate_private_root(root: Path, *, expected_files: set[str], artifact_name: str) -> Path:
    unresolved = root.expanduser()
    if unresolved.is_symlink():
        raise AgenticScoreArtifactError(f'{artifact_name} root cannot be a symlink')
    resolved = unresolved.resolve()
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise AgenticScoreArtifactError(f'cannot inspect {artifact_name} root: {error}') from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise AgenticScoreArtifactError(f'{artifact_name} root must be a private mode-0700 directory')
    if {entry.name for entry in os.scandir(resolved)} != expected_files:
        raise AgenticScoreArtifactError(f'{artifact_name} exact file inventory mismatch')
    return resolved


def _read_private_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AgenticScoreArtifactError(f'cannot open private artifact file {path}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
        ):
            raise AgenticScoreArtifactError('score handoff must contain private bounded regular files')
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes - len(content) + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise AgenticScoreArtifactError('private artifact file exceeds its byte limit')
    finally:
        os.close(descriptor)


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AgenticScoreArtifactError(f'{field_name} must include a UTC offset')
    return value.astimezone(UTC)


def _require_hmac_key(key: bytes, label: str) -> None:
    if len(key) < 32:
        raise ValueError(f'{label} HMAC key must contain at least 32 bytes')


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise AgenticScoreArtifactError(f'{field_name} must be a lowercase SHA-256 digest')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
