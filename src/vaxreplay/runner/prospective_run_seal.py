"""Independently timestamp the first and only run of a prospective challenge.

The ordinary runner HMAC proves only that an artifact passed through an organizer-controlled
harness.  It does not prove *when* the run existed and, by itself, lets an organizer retry a
system and publish the preferred result.  This module adds a separate, externally witnessed seal
over the exact challenge, system, runner policy, run receipt, and response bytes.

The fixed attempt key is intentionally independent of run output.  A transparency-log verifier
can therefore reject a second target for the same ``attempt_key_sha256``.  RFC 3161 establishes
time and byte identity but does not itself provide a globally enumerable uniqueness check, so an
official RFC 3161 deployment must make the first proof public and have its trusted verifier reject
conflicting proofs.  The local contract also rejects every attempt number other than one and
retains failed first runs as invalid results rather than permitting retries.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.prospective_schema import (
    PROSPECTIVE_RESPONSE_PROTOCOL,
    ProspectiveAttemptPolicy,
    ProspectiveChallengeAdmission,
    prospective_attempt_policy_sha256,
    prospective_challenge_admission_sha256,
    prospective_suite_manifest_sha256,
)
from vaxreplay.runner.orchestrator import LoadedRunArtifact
from vaxreplay.runner.prospective_challenge import (
    LoadedProspectiveChallengeBundle,
    load_prospective_challenge_bundle,
)
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy, SystemSubmissionManifest
from vaxreplay.temporal_schema import TemporalReceiptAuthority

PROSPECTIVE_RUN_SEAL_TARGET_SCHEMA_VERSION = 'vaxreplay.prospective-run-seal-target.v0.1'
PROSPECTIVE_RUN_TIMESTAMP_PROOF_SCHEMA_VERSION = 'vaxreplay.prospective-run-timestamp-proof.v0.1'
PROSPECTIVE_RUN_SEAL_SCHEMA_VERSION = 'vaxreplay.prospective-run-seal.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_PROOF_BYTES = 512 * 1024 * 1024
_MAX_RUN_FILE_BYTES = 512 * 1024 * 1024
_EXTERNAL_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
}


class ProspectiveRunSealIntegrityError(ValueError):
    """Raised when a prospective run seal or one of its bound inputs changed."""


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field_name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class ProspectiveRunSealTarget(StrictModel):
    """Canonical bytes sent to an independent timestamp authority after a run finishes."""

    schema_version: Literal['vaxreplay.prospective-run-seal-target.v0.1'] = PROSPECTIVE_RUN_SEAL_TARGET_SCHEMA_VERSION
    release_id: str = Field(min_length=1)
    prospective_admission_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_admission_bytes: int = Field(gt=0)
    challenge_id: str = Field(min_length=1)
    challenge_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    challenge_manifest_bytes: int = Field(gt=0)
    suite_id: str = Field(min_length=1)
    prospective_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    prospective_suite_bytes: int = Field(gt=0)
    submission_id: str = Field(min_length=1)
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_manifest_bytes: int = Field(gt=0)
    runner_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    runner_policy_bytes: int = Field(gt=0)
    attempt_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_policy_bytes: int = Field(gt=0)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    attempt_number: Literal[1] = 1
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    run_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    run_receipt_bytes: int = Field(gt=0)
    responses_sha256: str = Field(pattern=_SHA256_PATTERN)
    responses_bytes: int = Field(gt=0)
    run_started_at: datetime
    run_finished_at: datetime
    run_deadline_at: datetime

    @field_validator('run_started_at', 'run_finished_at', 'run_deadline_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _aware(value, 'prospective run target timestamp')

    @model_validator(mode='after')
    def validate_run_window(self) -> Self:
        if self.run_finished_at < self.run_started_at:
            raise ValueError('prospective run cannot finish before it starts')
        if self.run_finished_at > self.run_deadline_at:
            raise ValueError('prospective run finished after the preregistered run deadline')
        return self


class ProspectiveRunTimestampProof(StrictModel):
    """Metadata for exact external proof bytes over one canonical run-seal target."""

    schema_version: Literal['vaxreplay.prospective-run-timestamp-proof.v0.1'] = (
        PROSPECTIVE_RUN_TIMESTAMP_PROOF_SCHEMA_VERSION
    )
    receipt_id: str = Field(min_length=1)
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    target_schema_version: Literal['vaxreplay.prospective-run-seal-target.v0.1'] = (
        PROSPECTIVE_RUN_SEAL_TARGET_SCHEMA_VERSION
    )
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    attempt_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    witnessed_at: datetime
    proof_sha256: str = Field(pattern=_SHA256_PATTERN)
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'prospective run witness timestamp')

    @field_validator('authority_type')
    @classmethod
    def validate_authority(cls, value: TemporalReceiptAuthority) -> TemporalReceiptAuthority:
        if value not in _EXTERNAL_AUTHORITIES:
            raise ValueError('run seals require RFC 3161 or a public transparency log')
        return value


class ProspectiveRunSealManifest(StrictModel):
    """Exact three-file allowlist for a verified run-seal artifact."""

    schema_version: Literal['vaxreplay.prospective-run-seal.v0.1'] = PROSPECTIVE_RUN_SEAL_SCHEMA_VERSION
    target_path: Literal['target.json'] = 'target.json'
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_bytes: int = Field(gt=0)
    proof_path: Literal['timestamp-proof.bin'] = 'timestamp-proof.bin'
    timestamp_proof: ProspectiveRunTimestampProof

    @model_validator(mode='after')
    def validate_proof_target(self) -> Self:
        if (
            self.timestamp_proof.target_sha256 != self.target_sha256
            or self.timestamp_proof.target_bytes != self.target_bytes
        ):
            raise ValueError('external timestamp proof does not bind the declared run target')
        return self


type ProspectiveRunTimestampVerifier = Callable[[ProspectiveRunTimestampProof, bytes], bool]


@dataclass(frozen=True)
class LoadedProspectiveRunSeal:
    root: Path
    manifest: ProspectiveRunSealManifest
    target: ProspectiveRunSealTarget
    proof_bytes: bytes
    manifest_sha256: str


def prospective_run_seal_target_sha256(target: ProspectiveRunSealTarget) -> str:
    return _sha256(canonical_json_bytes(target))


def prospective_run_seal_manifest_sha256(manifest: ProspectiveRunSealManifest) -> str:
    return _sha256(canonical_json_bytes(manifest))


def build_prospective_run_seal_target(
    *,
    challenge: LoadedProspectiveChallengeBundle,
    run: LoadedRunArtifact,
    admission: ProspectiveChallengeAdmission,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    attempt_number: int = 1,
) -> ProspectiveRunSealTarget:
    """Validate current source bytes and construct the one object an authority timestamps."""

    if attempt_number != 1:
        raise ValueError('prospective attempt policy permits only the first and only attempt')
    _require_bound_official_inputs(
        challenge=challenge,
        run=run,
        admission=admission,
        system=system,
        policy=policy,
        attempt_policy=attempt_policy,
    )
    admission_bytes = canonical_json_bytes(admission)
    challenge_bytes = canonical_json_bytes(challenge.manifest)
    suite_bytes = canonical_json_bytes(challenge.suite)
    system_bytes = canonical_json_bytes(system)
    policy_bytes = canonical_json_bytes(policy)
    attempt_policy_bytes = canonical_json_bytes(attempt_policy)
    receipt_bytes = canonical_json_bytes(run.receipt)
    attempt_key = _attempt_key_sha256(
        release_id=admission.release_id,
        admission_sha256=_sha256(admission_bytes),
        system_sha256=_sha256(system_bytes),
        submission_id=system.submission_id,
    )
    return ProspectiveRunSealTarget(
        release_id=admission.release_id,
        prospective_admission_sha256=_sha256(admission_bytes),
        prospective_admission_bytes=len(admission_bytes),
        challenge_id=challenge.manifest.challenge_id,
        challenge_bundle_sha256=_sha256(challenge_bytes),
        challenge_manifest_bytes=len(challenge_bytes),
        suite_id=challenge.suite.suite_id,
        prospective_suite_sha256=_sha256(suite_bytes),
        prospective_suite_bytes=len(suite_bytes),
        submission_id=system.submission_id,
        system_manifest_sha256=_sha256(system_bytes),
        system_manifest_bytes=len(system_bytes),
        runner_policy_sha256=_sha256(policy_bytes),
        runner_policy_bytes=len(policy_bytes),
        attempt_policy_sha256=_sha256(attempt_policy_bytes),
        attempt_policy_bytes=len(attempt_policy_bytes),
        attempt_key_sha256=attempt_key,
        attempt_number=1,
        run_id=run.receipt.run_id,
        run_receipt_sha256=_sha256(receipt_bytes),
        run_receipt_bytes=len(receipt_bytes),
        responses_sha256=_sha256(run.responses),
        responses_bytes=len(run.responses),
        run_started_at=run.receipt.started_at,
        run_finished_at=run.receipt.finished_at,
        run_deadline_at=admission.run_deadline_at,
    )


def build_prospective_run_seal(
    output_dir: Path,
    *,
    challenge: LoadedProspectiveChallengeBundle,
    run: LoadedRunArtifact,
    admission: ProspectiveChallengeAdmission,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    timestamp_proof: ProspectiveRunTimestampProof,
    proof_bytes: bytes,
    timestamp_verifier: ProspectiveRunTimestampVerifier,
    attempt_number: int = 1,
) -> LoadedProspectiveRunSeal:
    """Verify an external witness and atomically write a prospective run seal."""

    if timestamp_verifier is None:  # type: ignore[comparison-overlap]
        raise ValueError('an independent trusted timestamp verifier is required')
    target = build_prospective_run_seal_target(
        challenge=challenge,
        run=run,
        admission=admission,
        system=system,
        policy=policy,
        attempt_policy=attempt_policy,
        attempt_number=attempt_number,
    )
    target_bytes = canonical_json_bytes(target)
    _require_valid_timestamp_proof(
        target=target,
        target_bytes=target_bytes,
        timestamp_proof=timestamp_proof,
        proof_bytes=proof_bytes,
        timestamp_verifier=timestamp_verifier,
        error_type=ValueError,
    )
    manifest = ProspectiveRunSealManifest(
        target_sha256=_sha256(target_bytes),
        target_bytes=len(target_bytes),
        timestamp_proof=timestamp_proof,
    )

    target_root = output_dir.expanduser().absolute()
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target_root):
        raise ValueError(f'prospective run-seal output already exists: {target_root}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target_root.name}.', dir=target_root.parent))
    try:
        (staging / manifest.target_path).write_bytes(target_bytes)
        (staging / manifest.proof_path).write_bytes(proof_bytes)
        (staging / 'seal.json').write_bytes(canonical_json_bytes(manifest))
        for path in staging.iterdir():
            path.chmod(0o644)
        staging.chmod(0o755)
        os.replace(staging, target_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_prospective_run_seal(
        target_root,
        challenge=challenge,
        run=run,
        admission=admission,
        system=system,
        policy=policy,
        attempt_policy=attempt_policy,
        timestamp_verifier=timestamp_verifier,
    )


def load_prospective_run_seal(
    root: Path,
    *,
    challenge: LoadedProspectiveChallengeBundle,
    run: LoadedRunArtifact,
    admission: ProspectiveChallengeAdmission,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
    timestamp_verifier: ProspectiveRunTimestampVerifier,
) -> LoadedProspectiveRunSeal:
    """Fail closed on altered bytes or context and rerun the trusted external verifier."""

    if timestamp_verifier is None:  # type: ignore[comparison-overlap]
        raise ProspectiveRunSealIntegrityError('an independent trusted timestamp verifier is required')
    resolved = _resolve_root(root)
    _require_exact_seal_inventory(resolved)
    manifest_bytes = _read_regular_file(resolved / 'seal.json', _MAX_MODEL_BYTES)
    try:
        manifest = ProspectiveRunSealManifest.model_validate_json(manifest_bytes)
    except ValueError as error:
        raise ProspectiveRunSealIntegrityError(f'invalid prospective run-seal manifest: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ProspectiveRunSealIntegrityError('prospective run-seal manifest must use canonical JSON encoding')

    target_bytes = _read_regular_file(resolved / manifest.target_path, _MAX_MODEL_BYTES)
    try:
        target = ProspectiveRunSealTarget.model_validate_json(target_bytes)
    except ValueError as error:
        raise ProspectiveRunSealIntegrityError(f'invalid prospective run-seal target: {error}') from error
    if target_bytes != canonical_json_bytes(target):
        raise ProspectiveRunSealIntegrityError('prospective run-seal target must use canonical JSON encoding')
    if _sha256(target_bytes) != manifest.target_sha256 or len(target_bytes) != manifest.target_bytes:
        raise ProspectiveRunSealIntegrityError('prospective run-seal target does not match its manifest binding')

    proof_bytes = _read_regular_file(resolved / manifest.proof_path, _MAX_PROOF_BYTES)
    expected_target = build_prospective_run_seal_target(
        challenge=challenge,
        run=run,
        admission=admission,
        system=system,
        policy=policy,
        attempt_policy=attempt_policy,
    )
    if target != expected_target:
        raise ProspectiveRunSealIntegrityError('prospective run seal is bound to different run inputs')
    _require_valid_timestamp_proof(
        target=target,
        target_bytes=target_bytes,
        timestamp_proof=manifest.timestamp_proof,
        proof_bytes=proof_bytes,
        timestamp_verifier=timestamp_verifier,
        error_type=ProspectiveRunSealIntegrityError,
    )
    return LoadedProspectiveRunSeal(
        root=resolved,
        manifest=manifest,
        target=target,
        proof_bytes=proof_bytes,
        manifest_sha256=_sha256(manifest_bytes),
    )


def _require_bound_official_inputs(
    *,
    challenge: LoadedProspectiveChallengeBundle,
    run: LoadedRunArtifact,
    admission: ProspectiveChallengeAdmission,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    attempt_policy: ProspectiveAttemptPolicy,
) -> None:
    if not challenge.authority_proofs_reverified:
        raise ValueError('prospective challenge decision seals require trusted authority reverification')
    _require_fresh_challenge(challenge)
    _require_fresh_run(run)
    admission_sha256 = prospective_challenge_admission_sha256(admission)
    suite_sha256 = prospective_suite_manifest_sha256(challenge.suite)
    system_sha256 = _sha256(canonical_json_bytes(system))
    policy_sha256 = _sha256(canonical_json_bytes(policy))
    attempt_policy_sha256 = prospective_attempt_policy_sha256(attempt_policy)

    if challenge.admission != admission or challenge.manifest.prospective_admission_sha256 != admission_sha256:
        raise ValueError('prospective challenge is bound to a different admission')
    if challenge.manifest.prospective_suite_sha256 != suite_sha256 or admission.suite_sha256 != suite_sha256:
        raise ValueError('prospective admission, challenge, and suite hashes differ')
    if admission.attempt_policy_sha256 != attempt_policy_sha256:
        raise ValueError('attempt-policy hash does not match the prospective admission')
    if system.response_protocol != PROSPECTIVE_RESPONSE_PROTOCOL:
        raise ValueError('official prospective runs require the prospective response protocol')
    if policy.required_isolation != IsolationTier.OFFICIAL:
        raise ValueError('prospective run seals require an official runner policy')

    receipt = run.receipt
    controls = receipt.capabilities
    if not receipt.sealed or controls.isolation_tier != IsolationTier.OFFICIAL:
        raise ValueError('prospective run seals require an official sealed runner artifact')
    if not all(
        (
            controls.network_isolation,
            controls.host_filesystem_isolation,
            controls.read_only_root,
            controls.non_root_user,
            controls.capability_drop,
            controls.no_new_privileges,
            controls.process_limit,
            controls.memory_limit,
            controls.cpu_limit,
            controls.scratch_limit,
            controls.fresh_worker_per_episode,
        )
    ):
        raise ValueError('prospective run does not attest every mandatory official runner capability')
    if (
        receipt.challenge_id != challenge.manifest.challenge_id
        or receipt.challenge_bundle_sha256 != challenge.manifest_sha256
        or receipt.admission_sha256 != admission_sha256
        or receipt.suite_id != challenge.suite.suite_id
        or receipt.suite_manifest_sha256 != suite_sha256
    ):
        raise ValueError('run receipt does not bind the prospective admission, challenge, and suite')
    if (
        receipt.system_manifest_sha256 != system_sha256
        or receipt.image_ref != system.image_ref
        or receipt.policy_sha256 != policy_sha256
    ):
        raise ValueError('run receipt does not bind the supplied system and runner policy')
    if receipt.responses_sha256 != _sha256(run.responses) or receipt.responses_bytes != len(run.responses):
        raise ValueError('run response bytes do not match the run receipt')
    if receipt.finished_at > admission.run_deadline_at:
        raise ValueError('prospective run finished after the preregistered run deadline')


def _require_valid_timestamp_proof(
    *,
    target: ProspectiveRunSealTarget,
    target_bytes: bytes,
    timestamp_proof: ProspectiveRunTimestampProof,
    proof_bytes: bytes,
    timestamp_verifier: ProspectiveRunTimestampVerifier,
    error_type: type[ValueError],
) -> None:
    if timestamp_proof.authority_type not in _EXTERNAL_AUTHORITIES:
        raise error_type('run seal lacks an independent RFC 3161 or transparency-log witness')
    if (
        timestamp_proof.target_sha256 != _sha256(target_bytes)
        or timestamp_proof.target_bytes != len(target_bytes)
        or timestamp_proof.attempt_key_sha256 != target.attempt_key_sha256
    ):
        raise error_type('external timestamp proof does not bind the exact prospective run target')
    if timestamp_proof.proof_sha256 != _sha256(proof_bytes) or timestamp_proof.proof_bytes != len(proof_bytes):
        raise error_type('external timestamp proof bytes do not match their hash and size binding')
    if timestamp_proof.witnessed_at < target.run_finished_at:
        raise error_type('external run witness cannot predate run completion')
    if timestamp_proof.witnessed_at > target.run_deadline_at:
        raise error_type('external run witness arrived after the preregistered run deadline')
    try:
        accepted = timestamp_verifier(timestamp_proof, proof_bytes)
    except Exception as error:
        raise error_type(f'trusted external timestamp verifier failed: {error}') from error
    if not accepted:
        raise error_type('trusted external timestamp verifier rejected the run proof')


def _require_fresh_challenge(challenge: LoadedProspectiveChallengeBundle) -> None:
    try:
        structural = load_prospective_challenge_bundle(challenge.root)
    except ValueError as error:
        raise ValueError(f'prospective challenge changed on disk: {error}') from error
    if (
        structural.manifest != challenge.manifest
        or structural.suite != challenge.suite
        or structural.admission != challenge.admission
        or structural.envelopes != challenge.envelopes
        or structural.packages != challenge.packages
        or structural.seals != challenge.seals
        or structural.manifest_sha256 != challenge.manifest_sha256
    ):
        raise ValueError('prospective challenge changed after trusted loading')


def _require_fresh_run(run: LoadedRunArtifact) -> None:
    root = run.root
    if root.is_symlink() or not root.is_dir():
        raise ValueError('loaded run artifact root is unavailable or is a symlink')
    expected_root = {'run.json', 'run.hmac', 'responses.jsonl'}
    expected_audit = {
        f'{ordinal:06d}.{extension}'
        for ordinal in range(len(run.receipt.episodes))
        for extension in ('stdout', 'stderr')
    }
    _require_inventory(root, expected_files=expected_root, expected_directories={'audit'}, context='run artifact')
    _require_inventory(root / 'audit', expected_files=expected_audit, expected_directories=set(), context='run audit')
    receipt_bytes = _read_regular_file(root / 'run.json', _MAX_RUN_FILE_BYTES)
    response_bytes = _read_regular_file(root / 'responses.jsonl', _MAX_RUN_FILE_BYTES)
    hmac_bytes = _read_regular_file(root / 'run.hmac', 65)
    if receipt_bytes != canonical_json_bytes(run.receipt) or _sha256(receipt_bytes) != run.receipt_sha256:
        raise ValueError('loaded run receipt changed on disk')
    if response_bytes != run.responses:
        raise ValueError('loaded run responses changed on disk')
    if hmac_bytes != run.receipt_hmac_sha256.encode('ascii') + b'\n':
        raise ValueError('loaded organizer run HMAC changed on disk')
    for episode in run.receipt.episodes:
        stdout = _read_regular_file(root / 'audit' / f'{episode.ordinal:06d}.stdout', _MAX_RUN_FILE_BYTES)
        stderr = _read_regular_file(root / 'audit' / f'{episode.ordinal:06d}.stderr', _MAX_RUN_FILE_BYTES)
        if _sha256(stdout) != episode.captured_stdout_sha256 or len(stdout) != episode.captured_stdout_bytes:
            raise ValueError('loaded run stdout audit changed on disk')
        if _sha256(stderr) != episode.captured_stderr_sha256 or len(stderr) != episode.captured_stderr_bytes:
            raise ValueError('loaded run stderr audit changed on disk')


def _attempt_key_sha256(
    *,
    release_id: str,
    admission_sha256: str,
    system_sha256: str,
    submission_id: str,
) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.prospective-attempt-key.v0.1',
                'release_id': release_id,
                'prospective_admission_sha256': admission_sha256,
                'submission_id': submission_id,
                'system_manifest_sha256': system_sha256,
            }
        )
    )


def _resolve_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ProspectiveRunSealIntegrityError('prospective run-seal root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ProspectiveRunSealIntegrityError(f'prospective run-seal root does not exist: {resolved}')
    return resolved


def _require_exact_seal_inventory(root: Path) -> None:
    _require_inventory(
        root,
        expected_files={'seal.json', 'target.json', 'timestamp-proof.bin'},
        expected_directories=set(),
        context='prospective run seal',
        error_type=ProspectiveRunSealIntegrityError,
    )


def _require_inventory(
    root: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    context: str,
    error_type: type[ValueError] = ValueError,
) -> None:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise error_type(f'{context} cannot contain symlinks')
                if entry.is_file(follow_symlinks=False):
                    actual_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    actual_directories.add(entry.name)
                else:
                    raise error_type(f'{context} can contain only regular files and directories')
    except OSError as error:
        raise error_type(f'cannot inventory {context}: {error}') from error
    if actual_files != expected_files or actual_directories != expected_directories:
        raise error_type(f'{context} exact file allowlist mismatch')


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProspectiveRunSealIntegrityError(f'cannot open {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProspectiveRunSealIntegrityError(f'{path.name} is not a regular file')
        if metadata.st_size > maximum_bytes:
            raise ProspectiveRunSealIntegrityError(f'{path.name} exceeds its size limit')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ProspectiveRunSealIntegrityError(f'{path.name} exceeds its size limit')
    except OSError as error:
        raise ProspectiveRunSealIntegrityError(f'cannot read {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
