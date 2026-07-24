from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tests.test_agentic_firecracker import _make_spec, _preflight
from vaxreplay.agentic.firecracker import FirecrackerPreflightError, firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification import (
    ARTIFACT_SHA256_FILE,
    QUALIFICATION_FILE,
    WORKER_SPEC_FILE,
    AuthenticatedFirecrackerQualification,
    FirecrackerFullSuiteEvidence,
    FirecrackerHostObservation,
    FirecrackerQualificationDrillEvidence,
    FirecrackerQualificationDrillId,
    FirecrackerQualificationError,
    FirecrackerQualificationRecord,
    FirecrackerQualificationStatus,
    decode_firecracker_qualification_key,
    firecracker_qualification_hmac,
    firecracker_qualification_key_id,
    inspect_and_retain_firecracker_host,
    load_firecracker_qualification,
    read_firecracker_qualification_key_fd,
    read_firecracker_qualification_key_file,
    required_firecracker_qualification_claims,
)
from vaxreplay.agentic.firecracker_qualification_cli import main as qualification_cli_main
from vaxreplay.bundle import canonical_json_bytes

_KEY = bytes.fromhex('42' * 32)
_QUALIFICATION_ID = '9' * 32


def _write_spec(root: Path):
    worker_root = root / 'worker'
    worker_root.mkdir()
    spec = _make_spec(worker_root)
    path = root / 'worker-spec.json'
    path.write_bytes(canonical_json_bytes(spec))
    return spec, path, firecracker_model_sha256(spec)


def _linux_observation() -> FirecrackerHostObservation:
    return FirecrackerHostObservation(
        collected_at=datetime.now(UTC),
        host_os='Linux',
        host_architecture='aarch64',
        host_kernel_release='test-linux-kernel',
        effective_uid=0,
        kvm_path_present=True,
        kvm_non_symlink_character_device=True,
        kvm_read_write_access=True,
        cgroup_v2_controller_file_present=True,
        cgroup_controllers=('cpu', 'memory', 'pids'),
    )


def _full_suite(spec_sha256: str, preflight_sha256: str, *, failed: FirecrackerQualificationDrillId | None = None):
    started = datetime.now(UTC)

    def drill(drill_id: FirecrackerQualificationDrillId, run_ids: tuple[str, ...]):
        required_claims = required_firecracker_qualification_claims(drill_id)
        failed_claims = (required_claims[0],) if drill_id == failed else ()
        verified_claims = tuple(claim for claim in required_claims if claim not in failed_claims)
        return FirecrackerQualificationDrillEvidence(
            drill_id=drill_id,
            passed=drill_id != failed,
            started_at=started,
            finished_at=started + timedelta(seconds=1),
            run_ids=run_ids,
            evidence_artifact_sha256=drill_id.value.encode().hex().ljust(64, '0')[:64],
            authenticated_worker_attestation_sha256='a' * 64,
            observer_executable_sha256='b' * 64,
            observation_count=len(run_ids),
            verified_claims=verified_claims,
            failed_claims=failed_claims,
        )

    first = ('1' * 32,)
    return FirecrackerFullSuiteEvidence(
        worker_spec_sha256=spec_sha256,
        host_preflight_sha256=preflight_sha256,
        collected_on_linux_kvm=True,
        live_boot=drill(FirecrackerQualificationDrillId.LIVE_BOOT, first),
        vsock_round_trip=drill(FirecrackerQualificationDrillId.VSOCK_ROUND_TRIP, first),
        guest_isolation=drill(FirecrackerQualificationDrillId.GUEST_ISOLATION, first),
        cgroup_enforcement=drill(FirecrackerQualificationDrillId.CGROUP_ENFORCEMENT, first),
        wall_timeout=drill(FirecrackerQualificationDrillId.WALL_TIMEOUT, ('2' * 32,)),
        teardown=drill(FirecrackerQualificationDrillId.TEARDOWN, first),
        load_canary=drill(FirecrackerQualificationDrillId.LOAD_CANARY, ('3' * 32, '4' * 32)),
    )


def test_unsupported_macos_result_is_authenticated_private_and_create_once(tmp_path: Path) -> None:
    spec, spec_path, spec_sha256 = _write_spec(tmp_path)
    output = tmp_path / 'unsupported'
    with (
        patch('vaxreplay.agentic.firecracker_qualification.platform.system', return_value='Darwin'),
        patch(
            'vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host',
            side_effect=FirecrackerPreflightError('requires Linux'),
        ),
    ):
        loaded = inspect_and_retain_firecracker_host(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=output,
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
            qualification_id=_QUALIFICATION_ID,
        )

    record = loaded.authenticated.record
    assert record.status == FirecrackerQualificationStatus.UNSUPPORTED_HOST_OS
    assert record.qualified is False
    assert record.preflight is None
    assert record.full_suite_evidence is None
    assert record.preflight_alone_is_full_runtime_qualification is False
    assert set(path.name for path in output.iterdir()) == {
        QUALIFICATION_FILE,
        WORKER_SPEC_FILE,
        ARTIFACT_SHA256_FILE,
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    retained = b''.join(path.read_bytes() for path in output.iterdir())
    assert _KEY not in retained
    assert b'key-file' not in retained

    reloaded = load_firecracker_qualification(
        output,
        qualification_key=_KEY,
        expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
        expected_worker_spec_sha256=spec_sha256,
        expected_artifact_sha256=loaded.artifact_sha256,
    )
    assert reloaded == loaded
    with pytest.raises(FirecrackerQualificationError, match='already exists'):
        inspect_and_retain_firecracker_host(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=output,
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
        )
    assert firecracker_model_sha256(spec) == spec_sha256


def test_successful_preflight_alone_is_explicitly_unqualified(tmp_path: Path) -> None:
    spec, spec_path, spec_sha256 = _write_spec(tmp_path)
    preflight = _preflight(spec)
    with patch('vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host', return_value=preflight):
        loaded = inspect_and_retain_firecracker_host(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=tmp_path / 'preflight-only',
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
        )
    assert loaded.authenticated.record.status == FirecrackerQualificationStatus.HOST_PREFLIGHT_PASSED_ONLY
    assert loaded.authenticated.record.preflight == preflight
    assert loaded.authenticated.record.qualified is False


def test_public_retention_api_rejects_unauthenticated_full_suite_evidence(tmp_path: Path) -> None:
    spec, spec_path, spec_sha256 = _write_spec(tmp_path)
    preflight = _preflight(spec)
    preflight_sha256 = firecracker_model_sha256(preflight)
    suite = _full_suite(spec_sha256, preflight_sha256)
    with (
        patch(
            'vaxreplay.agentic.firecracker_qualification.observe_firecracker_host',
            return_value=_linux_observation(),
        ),
        patch('vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host', return_value=preflight),
        pytest.raises(FirecrackerQualificationError, match='authenticated live collector'),
    ):
        inspect_and_retain_firecracker_host(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=tmp_path / 'qualified',
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
            full_suite_evidence=suite,
        )
    assert not (tmp_path / 'qualified').exists()


def test_loader_rejects_legacy_hmac_valid_positive_artifact(tmp_path: Path) -> None:
    """A legacy qualification HMAC cannot substitute for collector authentication."""

    spec, _, spec_sha256 = _write_spec(tmp_path)
    spec_bytes = canonical_json_bytes(spec)
    preflight = _preflight(spec)
    preflight_sha256 = firecracker_model_sha256(preflight)
    record = FirecrackerQualificationRecord(
        qualification_id=_QUALIFICATION_ID,
        worker_spec_sha256=spec_sha256,
        worker_spec_bytes=len(spec_bytes),
        qualifier_source_sha256='a' * 64,
        host_observation=_linux_observation(),
        status=FirecrackerQualificationStatus.FULL_RUNTIME_QUALIFIED,
        preflight=preflight,
        preflight_sha256=preflight_sha256,
        full_suite_evidence=_full_suite(spec_sha256, preflight_sha256),
        qualified=True,
        failure_summary=None,
        recorded_at=datetime.now(UTC),
    )
    authenticated = AuthenticatedFirecrackerQualification(
        record=record,
        qualification_key_id=firecracker_qualification_key_id(_KEY),
        qualification_hmac_sha256=firecracker_qualification_hmac(record, _KEY),
    )
    qualification_bytes = canonical_json_bytes(authenticated)
    artifact_sha256 = hashlib.sha256(qualification_bytes).hexdigest()
    artifact = tmp_path / 'legacy-qualified'
    artifact.mkdir(mode=0o700)
    files = {
        QUALIFICATION_FILE: qualification_bytes,
        WORKER_SPEC_FILE: spec_bytes,
        ARTIFACT_SHA256_FILE: (artifact_sha256 + '\n').encode('ascii'),
    }
    for name, content in files.items():
        path = artifact / name
        path.write_bytes(content)
        path.chmod(0o600)

    with pytest.raises(FirecrackerQualificationError, match='authenticated live collector'):
        load_firecracker_qualification(
            artifact,
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
            expected_worker_spec_sha256=spec_sha256,
            expected_artifact_sha256=artifact_sha256,
        )


def test_cli_rejects_caller_authored_full_suite_before_retaining_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, spec_path, spec_sha256 = _write_spec(tmp_path)
    suite_path = tmp_path / 'caller-authored-suite.json'
    suite_path.write_bytes(canonical_json_bytes(_full_suite(spec_sha256, firecracker_model_sha256(_preflight(spec)))))
    key_path = tmp_path / 'key'
    key_path.write_text(_KEY.hex() + '\n', encoding='ascii')
    key_path.chmod(0o600)
    output = tmp_path / 'must-not-exist'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-qualify',
            'inspect-host',
            '--worker-spec',
            str(spec_path),
            '--expected-worker-spec-sha256',
            spec_sha256,
            '--expected-qualification-key-id',
            firecracker_qualification_key_id(_KEY),
            '--key-file',
            str(key_path),
            '--output',
            str(output),
            '--full-suite-evidence',
            str(suite_path),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        qualification_cli_main()
    assert exit_info.value.code == 64
    assert 'authenticated live collector is required' in capsys.readouterr().err
    assert not output.exists()


def test_full_suite_schema_rejects_missing_load_coverage_and_wrong_drill() -> None:
    suite = _full_suite('c' * 64, 'd' * 64)
    one_run = suite.load_canary.model_copy(update={'run_ids': ('3' * 32,), 'observation_count': 1})
    with pytest.raises(ValidationError, match='at least two'):
        FirecrackerFullSuiteEvidence.model_validate(suite.model_dump() | {'load_canary': one_run.model_dump()})
    wrong = suite.vsock_round_trip.model_copy(update={'drill_id': FirecrackerQualificationDrillId.LIVE_BOOT})
    with pytest.raises(ValidationError, match='required live claim|corresponding drill IDs'):
        FirecrackerFullSuiteEvidence.model_validate(suite.model_dump() | {'vsock_round_trip': wrong.model_dump()})
    claims_lie = suite.wall_timeout.model_dump()
    claims_lie['verified_claims'] = claims_lie['verified_claims'][1:]
    claims_lie['failed_claims'] = (suite.wall_timeout.verified_claims[0],)
    with pytest.raises(ValidationError, match='pass state'):
        FirecrackerQualificationDrillEvidence.model_validate(claims_lie)


def test_tampering_or_wrong_external_pin_is_rejected(tmp_path: Path) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    with patch(
        'vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host',
        side_effect=FirecrackerPreflightError('requires Linux'),
    ):
        loaded = inspect_and_retain_firecracker_host(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=tmp_path / 'artifact',
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
        )
    with pytest.raises(FirecrackerQualificationError, match='external pin'):
        load_firecracker_qualification(
            Path(loaded.root),
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
            expected_worker_spec_sha256=spec_sha256,
            expected_artifact_sha256='f' * 64,
        )

    path = Path(loaded.root) / QUALIFICATION_FILE
    authenticated = AuthenticatedFirecrackerQualification.model_validate_json(path.read_bytes())
    forged_record = authenticated.record.model_copy(update={'failure_summary': 'forged'})
    forged = authenticated.model_copy(
        update={
            'record': forged_record,
            'qualification_hmac_sha256': firecracker_qualification_hmac(forged_record, _KEY),
        }
    )
    path.write_bytes(canonical_json_bytes(forged))
    path.chmod(0o600)
    with pytest.raises(FirecrackerQualificationError, match='digest'):
        load_firecracker_qualification(
            Path(loaded.root),
            qualification_key=_KEY,
            expected_qualification_key_id=firecracker_qualification_key_id(_KEY),
            expected_worker_spec_sha256=spec_sha256,
            expected_artifact_sha256=loaded.artifact_sha256,
        )


def test_cli_retains_unsupported_result_exits_nonzero_and_never_serializes_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    key_path = tmp_path / 'qualification-secret-do-not-serialize.key'
    key_path.write_text(_KEY.hex() + '\n', encoding='ascii')
    key_path.chmod(0o600)
    output = tmp_path / 'cli-artifact'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-qualify',
            'inspect-host',
            '--worker-spec',
            str(spec_path),
            '--expected-worker-spec-sha256',
            spec_sha256,
            '--expected-qualification-key-id',
            firecracker_qualification_key_id(_KEY),
            '--output',
            str(output),
            '--qualification-id',
            _QUALIFICATION_ID,
            '--key-file',
            str(key_path),
        ],
    )
    with (
        patch('vaxreplay.agentic.firecracker_qualification.platform.system', return_value='Darwin'),
        patch(
            'vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host',
            side_effect=FirecrackerPreflightError('requires Linux'),
        ),
        pytest.raises(SystemExit) as exit_info,
    ):
        qualification_cli_main()
    assert exit_info.value.code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary['qualified'] is False
    assert summary['status'] == 'unsupported_host_os'
    assert summary['preflight_alone_is_full_runtime_qualification'] is False
    retained = (output / QUALIFICATION_FILE).read_text(encoding='utf-8')
    assert str(key_path) not in retained
    assert _KEY.hex() not in retained


def test_cli_bounds_invalid_qualification_id_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    key_path = tmp_path / 'qualification-secret-do-not-serialize.key'
    key_path.write_text(_KEY.hex() + '\n', encoding='ascii')
    key_path.chmod(0o600)
    output = tmp_path / 'invalid-id-artifact'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-qualify',
            'inspect-host',
            '--worker-spec',
            str(spec_path),
            '--expected-worker-spec-sha256',
            spec_sha256,
            '--expected-qualification-key-id',
            firecracker_qualification_key_id(_KEY),
            '--output',
            str(output),
            '--qualification-id',
            'not-a-canonical-id',
            '--key-file',
            str(key_path),
        ],
    )
    with (
        patch('vaxreplay.agentic.firecracker_qualification.platform.system', return_value='Darwin'),
        patch(
            'vaxreplay.agentic.firecracker_qualification.preflight_firecracker_host',
            side_effect=FirecrackerPreflightError('requires Linux'),
        ),
        pytest.raises(SystemExit) as exit_info,
    ):
        qualification_cli_main()
    captured = capsys.readouterr()
    assert exit_info.value.code == 64
    assert captured.out == ''
    assert captured.err.startswith('firecracker qualification rejected: ')
    assert 'Traceback' not in captured.err
    assert str(key_path) not in captured.err
    assert _KEY.hex() not in captured.err
    assert not output.exists()


@pytest.mark.skipif(platform.system() == 'Linux', reason='exercises the real unsupported-host path')
def test_cli_on_current_non_linux_host_retains_honest_nonzero_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    key_path = tmp_path / 'key'
    key_path.write_text(_KEY.hex() + '\n', encoding='ascii')
    key_path.chmod(0o600)
    output = tmp_path / 'native-host-result'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-qualify',
            'inspect-host',
            '--worker-spec',
            str(spec_path),
            '--expected-worker-spec-sha256',
            spec_sha256,
            '--expected-qualification-key-id',
            firecracker_qualification_key_id(_KEY),
            '--output',
            str(output),
            '--key-file',
            str(key_path),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        qualification_cli_main()
    assert exit_info.value.code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == 'unsupported_host_os'
    assert summary['qualified'] is False
    assert (output / QUALIFICATION_FILE).is_file()


def test_key_is_read_only_from_private_file_or_inherited_descriptor(tmp_path: Path) -> None:
    key_path = tmp_path / 'key'
    key_path.write_text(_KEY.hex() + '\n', encoding='ascii')
    key_path.chmod(0o600)
    assert read_firecracker_qualification_key_file(key_path) == _KEY
    descriptor = os.open(key_path, os.O_RDONLY)
    try:
        assert read_firecracker_qualification_key_fd(descriptor) == _KEY
    finally:
        os.close(descriptor)
    key_path.chmod(0o644)
    with pytest.raises(FirecrackerQualificationError, match='owner-only'):
        read_firecracker_qualification_key_file(key_path)


def test_exported_qualification_key_decoder_enforces_the_documented_format() -> None:
    assert decode_firecracker_qualification_key(_KEY.hex().encode('ascii')) == _KEY
    assert decode_firecracker_qualification_key(_KEY.hex().encode('ascii') + b'\n') == _KEY
    with pytest.raises(FirecrackerQualificationError, match='trimmed hexadecimal'):
        decode_firecracker_qualification_key(b' ' + _KEY.hex().encode('ascii'))
    with pytest.raises(FirecrackerQualificationError, match='not valid hexadecimal'):
        decode_firecracker_qualification_key(_KEY.hex()[:2].encode('ascii') + b' ' + _KEY.hex()[2:].encode('ascii'))
    with pytest.raises(FirecrackerQualificationError, match='ASCII hexadecimal'):
        decode_firecracker_qualification_key(b'\xff' * 64)
