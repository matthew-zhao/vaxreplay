from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import vaxreplay.agentic.firecracker_qualification_cgroup_canary as canary_module
from tests.test_agentic_firecracker import _make_spec
from tests.test_agentic_firecracker_qualification_live_collector import _collect
from vaxreplay.agentic.firecracker import firecracker_model_sha256
from vaxreplay.agentic.firecracker_qualification import FirecrackerQualificationClaim
from vaxreplay.agentic.firecracker_qualification_cgroup_canary import (
    FirecrackerQualificationCgroupCanaryError,
    _PausedHelper,
    _require_safe_limits,
    _wait_for_memory_event_and_reap,
)
from vaxreplay.agentic.firecracker_qualification_probe import _derived_cgroup_claims


def _claims(tmp_path: Path, *, snapshots=None, canary=None):
    spec, _, _, _, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    drill = collection.drills[3]
    return _derived_cgroup_claims(
        drill.cgroup_snapshots if snapshots is None else snapshots,
        drill.host_cgroup_canary if canary is None else canary,
        collection,
        worker_spec=spec,
    )


def test_signed_host_canary_derives_memory_and_pids_controller_claims(tmp_path: Path) -> None:
    claims = _claims(tmp_path)
    assert FirecrackerQualificationClaim.MEMORY_LIMIT_OBSERVED in claims
    assert FirecrackerQualificationClaim.PIDS_LIMIT_OBSERVED in claims


def test_verifier_rejects_helper_left_in_exact_cgroup_even_with_rehashed_snapshot(tmp_path: Path) -> None:
    spec, _, _, _, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    drill = collection.drills[3]
    assert drill.host_cgroup_canary is not None
    cleanup = drill.cgroup_snapshots[-1].model_copy(
        update={
            'member_pids': tuple(
                sorted((*drill.cgroup_snapshots[-1].member_pids, drill.host_cgroup_canary.pids_helper_pid))
            )
        }
    )
    snapshots = (*drill.cgroup_snapshots[:-1], cleanup)
    canary = drill.host_cgroup_canary.model_copy(update={'cleanup_snapshot_sha256': firecracker_model_sha256(cleanup)})
    assert not _derived_cgroup_claims(snapshots, canary, collection, worker_spec=spec)


def test_verifier_rejects_rebound_cgroup_inode_even_with_matching_canary_hash(tmp_path: Path) -> None:
    spec, _, _, _, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    drill = collection.drills[3]
    assert drill.host_cgroup_canary is not None
    rebound = drill.cgroup_snapshots[2].model_copy(update={'cgroup_inode': drill.worker_bindings[0].cgroup_inode + 1})
    snapshots = (*drill.cgroup_snapshots[:2], rebound, *drill.cgroup_snapshots[3:])
    canary = drill.host_cgroup_canary.model_copy(
        update={'memory_armed_snapshot_sha256': firecracker_model_sha256(rebound)}
    )
    assert not _derived_cgroup_claims(snapshots, canary, collection, worker_spec=spec)


def test_verifier_rejects_unmeasured_helper_identity(tmp_path: Path) -> None:
    spec, _, _, _, loaded = _collect(tmp_path)
    collection = loaded.authenticated.collection
    drill = collection.drills[3]
    assert drill.host_cgroup_canary is not None
    forged = drill.host_cgroup_canary.model_copy(
        update={'pids_helper_descendant_pids': (*drill.host_cgroup_canary.pids_helper_descendant_pids, 999_999)}
    )
    assert not _derived_cgroup_claims(drill.cgroup_snapshots, forged, collection, worker_spec=spec)


def test_canary_schema_rejects_helper_pid_overlap(tmp_path: Path) -> None:
    _, _, _, _, loaded = _collect(tmp_path)
    canary = loaded.authenticated.collection.drills[3].host_cgroup_canary
    assert canary is not None
    with pytest.raises(ValidationError, match='overlap'):
        type(canary).model_validate(canary.model_dump() | {'memory_helper_pid': canary.pids_helper_pid})


def test_host_canary_rejects_unsafe_resource_fanout_before_fork(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    with pytest.raises(FirecrackerQualificationCgroupCanaryError, match='memory.max'):
        _require_safe_limits(spec.model_copy(update={'limits': spec.limits.model_copy(update={'memory_mib': 1025})}))
    with pytest.raises(FirecrackerQualificationCgroupCanaryError, match='pids.max'):
        _require_safe_limits(spec.model_copy(update={'limits': spec.limits.model_copy(update={'pids': 513})}))


def test_memory_canary_waits_for_delayed_oom_counters_after_sigkill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, loaded = _collect(tmp_path)
    snapshots = loaded.authenticated.collection.drills[3].cgroup_snapshots
    before, after = snapshots[2], snapshots[3]
    observations = iter((before, after))
    helper = _PausedHelper(pid=10, pidfd=11, command_fd=12, status_fd=13)
    monkeypatch.setattr(canary_module, '_wait_direct_child_nohang', lambda _helper: 9)
    monkeypatch.setattr(canary_module.time, 'sleep', lambda _seconds: None)
    ticks = iter(range(10))

    status = _wait_for_memory_event_and_reap(
        helper,
        before=before,
        snapshot=lambda: next(observations),
        deadline_ns=9,
        monotonic_ns=lambda: next(ticks),
    )

    assert status == 9


def test_memory_canary_reports_rlimit_exit_and_latest_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, _, loaded = _collect(tmp_path)
    before = loaded.authenticated.collection.drills[3].cgroup_snapshots[2]
    helper = _PausedHelper(pid=10, pidfd=11, command_fd=12, status_fd=13)
    monkeypatch.setattr(canary_module, '_wait_direct_child_nohang', lambda _helper: 71 << 8)

    with pytest.raises(
        FirecrackerQualificationCgroupCanaryError,
        match=r'exit=71; oom=0, oom_kill=0',
    ):
        _wait_for_memory_event_and_reap(
            helper,
            before=before,
            snapshot=lambda: before,
            deadline_ns=10,
            monotonic_ns=lambda: 0,
        )


def test_memory_helper_sets_oom_preference_before_drop_and_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _make_spec(tmp_path)
    events: list[str] = []

    class AllocationReached(Exception):
        pass

    monkeypatch.setattr(canary_module, '_set_and_verify_oom_score_adj', lambda: events.append('oom=1000'))
    monkeypatch.setattr(canary_module, '_drop_privileges', lambda _spec: events.append('drop'))

    def allocate(_limit: int) -> None:
        events.append('allocate')
        raise AllocationReached

    monkeypatch.setattr(canary_module, '_memory_child', allocate)

    with pytest.raises(AllocationReached):
        canary_module._run_released_helper_action(
            action='memory',
            spec=spec,
            command_read=10,
            status_write=11,
            memory_address_space_bytes=128 * 1024 * 1024,
        )

    assert events == ['oom=1000', 'drop', 'allocate']


def test_memory_helper_never_drops_or_allocates_when_oom_preference_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _make_spec(tmp_path)
    events: list[str] = []

    def reject_oom_preference() -> None:
        events.append('oom-rejected')
        raise FirecrackerQualificationCgroupCanaryError('kernel did not retain OOM preference')

    monkeypatch.setattr(canary_module, '_set_and_verify_oom_score_adj', reject_oom_preference)
    monkeypatch.setattr(canary_module, '_drop_privileges', lambda _spec: events.append('drop'))
    monkeypatch.setattr(canary_module, '_memory_child', lambda _limit: events.append('allocate'))

    with pytest.raises(FirecrackerQualificationCgroupCanaryError, match='did not retain'):
        canary_module._run_released_helper_action(
            action='memory',
            spec=spec,
            command_read=10,
            status_write=11,
            memory_address_space_bytes=128 * 1024 * 1024,
        )

    assert events == ['oom-rejected']
