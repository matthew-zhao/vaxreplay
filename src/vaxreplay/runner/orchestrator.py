"""One-shot suite orchestration and one-way run-artifact verification."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Submission
from vaxreplay.prospective_schema import PROSPECTIVE_RESPONSE_PROTOCOL, ProspectiveSubmission
from vaxreplay.runner.backend import (
    IsolationBackend,
    RawExecutionResult,
    RawExecutionStatus,
)
from vaxreplay.runner.challenge import LoadedChallengeBundle
from vaxreplay.runner.prospective_challenge import LoadedProspectiveChallengeBundle
from vaxreplay.runner.schema import (
    BackendCapabilities,
    EpisodeRunReceipt,
    EpisodeRunStatus,
    IsolationTier,
    RunnerPolicy,
    SuiteRunReceipt,
    SystemSubmissionManifest,
)

_NULL_RESPONSE = b'null\n'
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_AGGREGATE_RESPONSES_BYTES = 128 * 1024 * 1024
_MAX_AGGREGATE_AUDIT_BYTES = 128 * 1024 * 1024

type RunnableChallengeBundle = LoadedChallengeBundle | LoadedProspectiveChallengeBundle


class RunArtifactIntegrityError(ValueError):
    """Raised when a worker output artifact is incomplete or not bound to its run."""


@dataclass(frozen=True)
class LoadedRunArtifact:
    root: Path
    receipt: SuiteRunReceipt
    receipt_sha256: str
    receipt_hmac_sha256: str
    responses: bytes
    response_records: tuple[bytes, ...]


def run_challenge_bundle(
    challenge: RunnableChallengeBundle,
    *,
    expected_challenge_sha256: str,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    output_dir: Path,
    backend: IsolationBackend,
) -> LoadedRunArtifact:
    """Run every episode once, without labels or scoring, then emit only answers and hash receipts."""

    if challenge.manifest_sha256 != expected_challenge_sha256:
        raise ValueError('challenge bundle does not match expected_challenge_sha256')
    response_protocols = {envelope.response_protocol for envelope in challenge.envelopes}
    if response_protocols != {system.response_protocol}:
        raise ValueError('system response protocol does not match every challenge envelope')
    maximum_responses = len(challenge.envelopes) * (policy.limits.max_response_bytes + 1)
    if maximum_responses > _MAX_AGGREGATE_RESPONSES_BYTES:
        raise ValueError('runner policy and suite size exceed the aggregate response limit')
    maximum_audit_bytes = len(challenge.envelopes) * (policy.limits.max_response_bytes + policy.limits.max_log_bytes)
    if maximum_audit_bytes > _MAX_AGGREGATE_AUDIT_BYTES:
        raise ValueError('runner policy and suite size exceed the aggregate audit-log limit')
    key_id = receipt_key_id(receipt_key)
    if key_id != expected_receipt_key_id:
        raise ValueError('receipt key does not match the preregistered organizer key ID')
    target, staging = _make_run_staging(output_dir)
    started_at = datetime.now(UTC)
    try:
        prepared = backend.prepare(system, policy)
        _validate_capabilities(prepared.capabilities, policy)
        system_sha256 = _sha256(canonical_json_bytes(system))
        policy_sha256 = _sha256(canonical_json_bytes(policy))
        episode_receipts: list[EpisodeRunReceipt] = []
        responses_digest = hashlib.sha256()
        responses_bytes = 0
        audit_dir = staging / 'audit'
        audit_dir.mkdir()
        with (staging / 'responses.jsonl').open('wb') as response_output:
            for file_binding, envelope in zip(
                _challenge_envelope_files(challenge),
                challenge.envelopes,
                strict=True,
            ):
                envelope_bytes = canonical_json_bytes(envelope)
                raw = backend.run(
                    input_bytes=envelope_bytes + b'\n',
                    system=system,
                    policy=policy,
                    prepared=prepared,
                )
                status, response_record = _normalize_worker_response(raw, envelope.response_protocol)
                response_output.write(response_record)
                responses_digest.update(response_record)
                responses_bytes += len(response_record)
                (audit_dir / f'{envelope.ordinal:06d}.stdout').write_bytes(raw.stdout)
                (audit_dir / f'{envelope.ordinal:06d}.stderr').write_bytes(raw.stderr)
                episode_receipts.append(
                    EpisodeRunReceipt(
                        ordinal=envelope.ordinal,
                        episode_id=envelope.binding.episode_id,
                        envelope_sha256=file_binding.envelope_sha256,
                        status=status,
                        exit_code=raw.exit_code,
                        duration_ms=raw.duration_ms,
                        captured_stdout_sha256=_sha256(raw.stdout),
                        captured_stdout_bytes=len(raw.stdout),
                        stdout_truncated=raw.stdout_truncated,
                        captured_stderr_sha256=_sha256(raw.stderr),
                        captured_stderr_bytes=len(raw.stderr),
                        stderr_truncated=raw.stderr_truncated,
                        response_record_sha256=_sha256(response_record),
                        response_record_bytes=len(response_record),
                    )
                )
        receipt = SuiteRunReceipt(
            run_id=uuid.uuid4().hex,
            challenge_id=challenge.manifest.challenge_id,
            challenge_bundle_sha256=challenge.manifest_sha256,
            admission_sha256=_challenge_admission_sha256(challenge),
            suite_id=challenge.suite.suite_id,
            suite_manifest_sha256=_challenge_suite_sha256(challenge),
            system_manifest_sha256=system_sha256,
            policy_sha256=policy_sha256,
            receipt_key_id=key_id,
            image_ref=system.image_ref,
            resolved_image_id=prepared.resolved_image_id,
            capabilities=prepared.capabilities,
            sealed=prepared.capabilities.isolation_tier == IsolationTier.OFFICIAL,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            responses_sha256=responses_digest.hexdigest(),
            responses_bytes=responses_bytes,
            episodes=tuple(episode_receipts),
        )
        receipt_bytes = canonical_json_bytes(receipt)
        (staging / 'run.json').write_bytes(receipt_bytes)
        (staging / 'run.hmac').write_text(
            receipt_hmac_sha256(receipt, receipt_key) + '\n',
            encoding='ascii',
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_run_artifact(
        target,
        challenge=challenge,
        system=system,
        policy=policy,
        receipt_key=receipt_key,
        expected_receipt_key_id=expected_receipt_key_id,
        require_sealed=policy.required_isolation == IsolationTier.OFFICIAL,
    )


def load_run_artifact(
    root: Path,
    *,
    challenge: RunnableChallengeBundle,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    receipt_key: bytes,
    expected_receipt_key_id: str,
    require_sealed: bool = True,
) -> LoadedRunArtifact:
    """Validate the scorer handoff without loading labels or executing contestant code."""

    supplied_root = root.expanduser()
    if supplied_root.is_symlink():
        raise RunArtifactIntegrityError('run artifact root cannot be a symlink')
    resolved_root = supplied_root.resolve()
    if not resolved_root.is_dir():
        raise RunArtifactIntegrityError(f'run artifact root does not exist: {resolved_root}')
    receipt_bytes = _read_limited(resolved_root / 'run.json', _MAX_RECEIPT_BYTES)
    try:
        receipt = SuiteRunReceipt.model_validate_json(receipt_bytes)
    except ValueError as error:
        raise RunArtifactIntegrityError(f'invalid run receipt: {error}') from error
    if receipt_bytes != canonical_json_bytes(receipt):
        raise RunArtifactIntegrityError('run receipt must use canonical JSON encoding')
    expected_key_id = receipt_key_id(receipt_key)
    if expected_key_id != expected_receipt_key_id:
        raise RunArtifactIntegrityError('receipt key does not match the preregistered organizer key ID')
    if receipt.receipt_key_id != expected_key_id:
        raise RunArtifactIntegrityError('run receipt was authenticated with a different organizer key')
    hmac_bytes = _read_limited(resolved_root / 'run.hmac', 65)
    expected_hmac = receipt_hmac_sha256(receipt, receipt_key)
    if not hmac.compare_digest(hmac_bytes, expected_hmac.encode('ascii') + b'\n'):
        raise RunArtifactIntegrityError('run receipt HMAC authentication failed')
    _validate_run_inventory(resolved_root, len(challenge.envelopes))
    if receipt.challenge_bundle_sha256 != challenge.manifest_sha256:
        raise RunArtifactIntegrityError('run receipt is bound to a different challenge bundle')
    if receipt.admission_sha256 != _challenge_admission_sha256(challenge):
        raise RunArtifactIntegrityError('run receipt is bound to a different challenge admission')
    if (
        receipt.challenge_id != challenge.manifest.challenge_id
        or receipt.suite_id != challenge.suite.suite_id
        or receipt.suite_manifest_sha256 != _challenge_suite_sha256(challenge)
    ):
        raise RunArtifactIntegrityError('run receipt suite metadata does not match the challenge')
    if receipt.system_manifest_sha256 != _sha256(canonical_json_bytes(system)) or receipt.image_ref != system.image_ref:
        raise RunArtifactIntegrityError('run receipt is bound to a different system manifest')
    if receipt.policy_sha256 != _sha256(canonical_json_bytes(policy)):
        raise RunArtifactIntegrityError('run receipt is bound to a different runner policy')
    if {envelope.response_protocol for envelope in challenge.envelopes} != {system.response_protocol}:
        raise RunArtifactIntegrityError('system response protocol does not match every challenge envelope')
    if require_sealed and not receipt.sealed:
        raise RunArtifactIntegrityError('official scoring refuses a development-tier run artifact')
    if policy.required_isolation == IsolationTier.OFFICIAL and not receipt.sealed:
        raise RunArtifactIntegrityError('run receipt does not satisfy the policy isolation tier')
    _validate_capabilities(receipt.capabilities, policy)

    maximum_response_file_bytes = min(
        len(challenge.envelopes) * (policy.limits.max_response_bytes + 1),
        _MAX_AGGREGATE_RESPONSES_BYTES,
    )
    responses = _read_limited(resolved_root / 'responses.jsonl', maximum_response_file_bytes)
    if not responses.endswith(b'\n'):
        raise RunArtifactIntegrityError('responses JSONL must end with a newline')
    if _sha256(responses) != receipt.responses_sha256 or len(responses) != receipt.responses_bytes:
        raise RunArtifactIntegrityError('responses JSONL does not match the run receipt')
    response_records = tuple(responses.splitlines(keepends=True))
    if len(response_records) != len(challenge.envelopes) or len(receipt.episodes) != len(challenge.envelopes):
        raise RunArtifactIntegrityError('run artifact must contain exactly one record per challenge episode')

    for envelope_file, envelope, episode_receipt, response_record in zip(
        _challenge_envelope_files(challenge),
        challenge.envelopes,
        receipt.episodes,
        response_records,
        strict=True,
    ):
        if (
            episode_receipt.ordinal != envelope.ordinal
            or episode_receipt.episode_id != envelope.binding.episode_id
            or episode_receipt.envelope_sha256 != envelope_file.envelope_sha256
            or envelope_file.envelope_sha256 != _canonical_model_sha256(envelope)
        ):
            raise RunArtifactIntegrityError('episode receipt does not match its challenge envelope')
        if (
            _sha256(response_record) != episode_receipt.response_record_sha256
            or len(response_record) != episode_receipt.response_record_bytes
        ):
            raise RunArtifactIntegrityError('response record does not match its episode receipt')
        stdout = _read_limited(
            resolved_root / 'audit' / f'{envelope.ordinal:06d}.stdout',
            policy.limits.max_response_bytes,
        )
        stderr = _read_limited(
            resolved_root / 'audit' / f'{envelope.ordinal:06d}.stderr',
            policy.limits.max_log_bytes,
        )
        if (
            _sha256(stdout) != episode_receipt.captured_stdout_sha256
            or len(stdout) != episode_receipt.captured_stdout_bytes
            or _sha256(stderr) != episode_receipt.captured_stderr_sha256
            or len(stderr) != episode_receipt.captured_stderr_bytes
        ):
            raise RunArtifactIntegrityError('private audit output does not match its episode receipt')
        if episode_receipt.status == EpisodeRunStatus.RESPONSE_LIMIT and not episode_receipt.stdout_truncated:
            raise RunArtifactIntegrityError('response-limit receipt must attest truncated stdout')
        if episode_receipt.status == EpisodeRunStatus.LOG_LIMIT and not episode_receipt.stderr_truncated:
            raise RunArtifactIntegrityError('log-limit receipt must attest truncated stderr')
        recomputed_status, recomputed_record = _normalize_worker_response(
            RawExecutionResult(
                status=_raw_status_for_receipt(episode_receipt.status),
                exit_code=episode_receipt.exit_code,
                duration_ms=episode_receipt.duration_ms,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=episode_receipt.stdout_truncated,
                stderr_truncated=episode_receipt.stderr_truncated,
            ),
            envelope.response_protocol,
        )
        if recomputed_status != episode_receipt.status or recomputed_record != response_record:
            raise RunArtifactIntegrityError('episode status or response is inconsistent with its private audit output')

    return LoadedRunArtifact(
        root=resolved_root,
        receipt=receipt,
        receipt_sha256=_sha256(receipt_bytes),
        receipt_hmac_sha256=expected_hmac,
        responses=responses,
        response_records=response_records,
    )


def _normalize_worker_response(raw: RawExecutionResult, response_protocol: str) -> tuple[EpisodeRunStatus, bytes]:
    if raw.stdout_truncated:
        return EpisodeRunStatus.RESPONSE_LIMIT, _NULL_RESPONSE
    if raw.stderr_truncated:
        return EpisodeRunStatus.LOG_LIMIT, _NULL_RESPONSE
    if raw.status == RawExecutionStatus.TIMED_OUT:
        return EpisodeRunStatus.TIMED_OUT, _NULL_RESPONSE
    if raw.status == RawExecutionStatus.RESPONSE_LIMIT:
        return EpisodeRunStatus.RESPONSE_LIMIT, _NULL_RESPONSE
    if raw.status == RawExecutionStatus.LOG_LIMIT:
        return EpisodeRunStatus.LOG_LIMIT, _NULL_RESPONSE
    if raw.status == RawExecutionStatus.BACKEND_ERROR:
        return EpisodeRunStatus.BACKEND_ERROR, _NULL_RESPONSE
    if raw.exit_code is None:
        return EpisodeRunStatus.BACKEND_ERROR, _NULL_RESPONSE
    if raw.exit_code != 0:
        return EpisodeRunStatus.NONZERO_EXIT, _NULL_RESPONSE
    try:
        raw.stdout.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return EpisodeRunStatus.INVALID_UTF8, _NULL_RESPONSE
    submission_model = ProspectiveSubmission if response_protocol == PROSPECTIVE_RESPONSE_PROTOCOL else Submission
    try:
        submission = submission_model.model_validate_json(raw.stdout)
    except ValidationError as error:
        if any(issue['type'] == 'json_invalid' for issue in error.errors()):
            return EpisodeRunStatus.INVALID_JSON, _NULL_RESPONSE
        return EpisodeRunStatus.INVALID_SUBMISSION, _NULL_RESPONSE
    return EpisodeRunStatus.ACCEPTED, canonical_json_bytes(submission) + b'\n'


def _raw_status_for_receipt(status: EpisodeRunStatus) -> RawExecutionStatus:
    if status == EpisodeRunStatus.TIMED_OUT:
        return RawExecutionStatus.TIMED_OUT
    if status == EpisodeRunStatus.RESPONSE_LIMIT:
        return RawExecutionStatus.RESPONSE_LIMIT
    if status == EpisodeRunStatus.LOG_LIMIT:
        return RawExecutionStatus.LOG_LIMIT
    if status == EpisodeRunStatus.BACKEND_ERROR:
        return RawExecutionStatus.BACKEND_ERROR
    return RawExecutionStatus.EXITED


def _validate_capabilities(capabilities: BackendCapabilities, policy: RunnerPolicy) -> None:
    required_controls = (
        capabilities.network_isolation,
        capabilities.host_filesystem_isolation,
        capabilities.read_only_root,
        capabilities.non_root_user,
        capabilities.capability_drop,
        capabilities.no_new_privileges,
        capabilities.process_limit,
        capabilities.memory_limit,
        capabilities.cpu_limit,
        capabilities.scratch_limit,
        capabilities.fresh_worker_per_episode,
    )
    if not all(required_controls):
        raise ValueError('backend does not attest every mandatory runner control')
    if policy.required_isolation == IsolationTier.OFFICIAL and capabilities.isolation_tier != IsolationTier.OFFICIAL:
        raise ValueError('backend does not satisfy the official isolation tier')


def _make_run_staging(output_dir: Path) -> tuple[Path, Path]:
    target = output_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise ValueError(f'run output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    return target, staging


def _validate_run_inventory(root: Path, episode_count: int) -> None:
    expected_root_files = {'responses.jsonl', 'run.json', 'run.hmac'}
    expected_audit_files = {
        path for ordinal in range(episode_count) for path in (f'{ordinal:06d}.stdout', f'{ordinal:06d}.stderr')
    }
    root_files: set[str] = set()
    root_directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise RunArtifactIntegrityError('run artifact cannot contain symlinks')
                if entry.is_file(follow_symlinks=False):
                    root_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    root_directories.add(entry.name)
                else:
                    raise RunArtifactIntegrityError('run artifact can contain only regular files')
    except OSError as error:
        raise RunArtifactIntegrityError(f'cannot inventory run artifact: {error}') from error
    if root_files != expected_root_files or root_directories != {'audit'}:
        raise RunArtifactIntegrityError('run artifact file allowlist mismatch')

    audit_files: set[str] = set()
    try:
        with os.scandir(root / 'audit') as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise RunArtifactIntegrityError('run audit can contain only regular files')
                audit_files.add(entry.name)
    except OSError as error:
        raise RunArtifactIntegrityError(f'cannot inventory run audit: {error}') from error
    if audit_files != expected_audit_files:
        raise RunArtifactIntegrityError('run artifact audit allowlist mismatch')


def _read_limited(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RunArtifactIntegrityError(f'cannot open {path.name}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RunArtifactIntegrityError(f'{path.name} is not a regular file')
        if metadata.st_size > maximum_bytes:
            raise RunArtifactIntegrityError(f'{path.name} exceeds its size limit')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise RunArtifactIntegrityError(f'{path.name} exceeds its size limit')
    except OSError as error:
        raise RunArtifactIntegrityError(f'cannot read {path.name}: {error}') from error
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_model_sha256(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _challenge_envelope_files(challenge: RunnableChallengeBundle):
    if isinstance(challenge, LoadedProspectiveChallengeBundle):
        return challenge.manifest.episodes
    return challenge.manifest.envelopes


def _challenge_admission_sha256(challenge: RunnableChallengeBundle) -> str | None:
    if isinstance(challenge, LoadedProspectiveChallengeBundle):
        return challenge.manifest.prospective_admission_sha256
    return challenge.manifest.admission_sha256


def _challenge_suite_sha256(challenge: RunnableChallengeBundle) -> str:
    if isinstance(challenge, LoadedProspectiveChallengeBundle):
        return challenge.manifest.prospective_suite_sha256
    return challenge.manifest.suite_manifest_sha256


def receipt_key_id(key: bytes) -> str:
    if len(key) < 32:
        raise ValueError('run receipt HMAC key must contain at least 32 bytes')
    return _sha256(key)


def receipt_hmac_sha256(receipt: SuiteRunReceipt, key: bytes) -> str:
    receipt_key_id(key)
    return hmac.new(key, canonical_json_bytes(receipt), hashlib.sha256).hexdigest()


def load_receipt_key(path: Path) -> bytes:
    try:
        raw = path.read_text(encoding='ascii').strip()
    except OSError as error:
        raise ValueError(f'cannot read receipt key {path}: {error}') from error
    try:
        key = bytes.fromhex(raw)
    except ValueError as error:
        raise ValueError(f'receipt key is not valid hexadecimal: {path}') from error
    receipt_key_id(key)
    return key
