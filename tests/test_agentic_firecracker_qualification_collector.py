from __future__ import annotations

import json
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_agentic_firecracker import _make_spec
from vaxreplay.agentic.firecracker import firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification import (
    FirecrackerHostObservation,
    FirecrackerQualificationDrillId,
    FirecrackerQualificationError,
    load_firecracker_full_suite_evidence,
    required_firecracker_qualification_claims,
)
from vaxreplay.agentic.firecracker_qualification_collector import (
    COLLECTOR_PLAN_FILE,
    COLLECTOR_PLAN_SHA256_FILE,
    WORKER_SPEC_FILE,
    FirecrackerCollectorMissingCapability,
    FirecrackerQualificationCollectorError,
    FirecrackerQualificationCollectorPlan,
    FirecrackerQualificationCollectorStatus,
    build_firecracker_qualification_collector_plan,
    load_firecracker_qualification_collector_plan,
    retain_firecracker_qualification_collector_plan,
)
from vaxreplay.agentic.firecracker_qualification_collector_cli import main as collector_cli_main
from vaxreplay.bundle import canonical_json_bytes

_PLAN_ID = '7' * 32


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
        host_kernel_release='qualification-test-kernel',
        effective_uid=0,
        kvm_path_present=True,
        kvm_non_symlink_character_device=True,
        kvm_read_write_access=True,
        cgroup_v2_controller_file_present=True,
        cgroup_controllers=('cpu', 'memory', 'pids'),
    )


def test_plan_enumerates_all_drills_but_cannot_claim_execution_or_qualification(tmp_path: Path) -> None:
    spec, _, spec_sha256 = _write_spec(tmp_path)
    plan = build_firecracker_qualification_collector_plan(
        spec,
        plan_id=_PLAN_ID,
        host_observation=_linux_observation(),
    )

    assert plan.worker_spec_sha256 == spec_sha256
    assert plan.status == FirecrackerQualificationCollectorStatus.BLOCKED_MISSING_MEASURED_PROBES
    assert plan.host_linux_kvm_cgroup_prerequisites_observed is True
    assert tuple(drill.drill_id for drill in plan.drills) == tuple(FirecrackerQualificationDrillId)
    assert all(
        drill.required_claims == required_firecracker_qualification_claims(drill.drill_id)
        and drill.ready_to_collect is False
        and drill.caller_assertions_can_satisfy is False
        and drill.missing_capabilities
        for drill in plan.drills
    )
    assert set(plan.missing_capabilities) == set(FirecrackerCollectorMissingCapability)
    assert plan.guest_probe_protocol_id is None
    assert plan.task_guest_protocol_reused_for_qualification is False
    assert plan.live_vm_launched is False
    assert plan.host_preflight_executed is False
    assert plan.raw_observation_bundle_emitted is False
    assert plan.full_suite_evidence_emitted is False
    assert plan.caller_supplied_drill_evidence_accepted is False
    assert plan.qualified is False
    assert plan.official_leaderboard_execution_qualified is False


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('live_vm_launched', True),
        ('host_preflight_executed', True),
        ('raw_observation_bundle_emitted', True),
        ('full_suite_evidence_emitted', True),
        ('caller_supplied_drill_evidence_accepted', True),
        ('qualified', True),
        ('official_leaderboard_execution_qualified', True),
    ],
)
def test_plan_schema_rejects_positive_claims(tmp_path: Path, field: str, value: bool) -> None:
    spec, _, _ = _write_spec(tmp_path)
    plan = build_firecracker_qualification_collector_plan(spec, host_observation=_linux_observation())
    with pytest.raises(ValidationError):
        FirecrackerQualificationCollectorPlan.model_validate(plan.model_dump() | {field: value})


def test_plan_artifact_is_private_create_once_and_externally_pinned(tmp_path: Path) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    output = tmp_path / 'collector-plan'
    loaded = retain_firecracker_qualification_collector_plan(
        worker_spec_path=spec_path,
        expected_worker_spec_sha256=spec_sha256,
        output_root=output,
        plan_id=_PLAN_ID,
    )

    assert loaded.plan.qualified is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {path.name for path in output.iterdir()} == {
        COLLECTOR_PLAN_FILE,
        WORKER_SPEC_FILE,
        COLLECTOR_PLAN_SHA256_FILE,
    }
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    assert (
        load_firecracker_qualification_collector_plan(
            output,
            expected_plan_sha256=loaded.plan_sha256,
            expected_worker_spec_sha256=spec_sha256,
        )
        == loaded
    )
    with pytest.raises(FirecrackerQualificationCollectorError, match='already exists'):
        retain_firecracker_qualification_collector_plan(
            worker_spec_path=spec_path,
            expected_worker_spec_sha256=spec_sha256,
            output_root=output,
        )
    with pytest.raises(FirecrackerQualificationCollectorError, match='external pin'):
        load_firecracker_qualification_collector_plan(
            output,
            expected_plan_sha256='f' * 64,
            expected_worker_spec_sha256=spec_sha256,
        )


def test_collector_plan_is_not_full_suite_evidence(tmp_path: Path) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    loaded = retain_firecracker_qualification_collector_plan(
        worker_spec_path=spec_path,
        expected_worker_spec_sha256=spec_sha256,
        output_root=tmp_path / 'plan',
    )
    with pytest.raises(FirecrackerQualificationError, match='full-suite evidence input is invalid'):
        load_firecracker_full_suite_evidence(Path(loaded.root) / COLLECTOR_PLAN_FILE)


def test_collector_cli_retains_blocked_plan_nonzero_and_verifies_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, spec_path, spec_sha256 = _write_spec(tmp_path)
    output = tmp_path / 'cli-plan'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-collector',
            'plan',
            '--worker-spec',
            str(spec_path),
            '--expected-worker-spec-sha256',
            spec_sha256,
            '--output',
            str(output),
            '--plan-id',
            _PLAN_ID,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        collector_cli_main()
    assert exit_info.value.code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == FirecrackerQualificationCollectorStatus.BLOCKED_MISSING_MEASURED_PROBES
    assert summary['required_drill_count'] == 7
    assert summary['live_vm_launched'] is False
    assert summary['full_suite_evidence_emitted'] is False
    assert summary['qualified'] is False

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vaxreplay-firecracker-collector',
            'verify-plan',
            '--artifact',
            str(output),
            '--expected-plan-sha256',
            summary['collector_plan_sha256'],
            '--expected-worker-spec-sha256',
            spec_sha256,
        ],
    )
    collector_cli_main()
    verified = json.loads(capsys.readouterr().out)
    assert verified['collector_plan_sha256'] == summary['collector_plan_sha256']
    assert verified['qualified'] is False
