from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_gold_adapter import _fixture
from tests.test_clinicaltrials_execution_workspace import _gold_by_nct, _plan
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_public_release import (
    ExecutionPublicReleaseError,
    ExecutionPublicReleaseReceipt,
    build_execution_public_release,
    verify_execution_public_release,
)
from vaxreplay.clinicaltrials.execution_workspace import ExecutionWorkspaceError, write_execution_workspace_build

_GOLD_KEY = bytes(range(32, 64))


def _workspace(root: Path):
    fixture = _fixture(root)
    plan = _plan(fixture)
    build = write_execution_workspace_build(
        plan=plan,
        gold_by_nct=_gold_by_nct(fixture, plan),
        private_gold_master_key=_GOLD_KEY,
        output_root=root / 'source-workspace',
    )
    receipt_sha256 = hashlib.sha256((build.root / 'BUILD-RECEIPT.json').read_bytes()).hexdigest()
    return build, receipt_sha256


def _release(root: Path):
    workspace, workspace_receipt_sha256 = _workspace(root)
    release = build_execution_public_release(
        source_workspace_root=workspace.root,
        expected_source_workspace_receipt_sha256=workspace_receipt_sha256,
        output_root=root / 'standalone-release',
        release_id='test-lane-a-public-release',
        expected_task_count=2,
    )
    return workspace, workspace_receipt_sha256, release


def test_builds_new_inode_public_only_release_with_exact_external_pins(tmp_path: Path) -> None:
    workspace, workspace_receipt_sha256, release = _release(tmp_path)
    verified = verify_execution_public_release(
        release.root,
        expected_receipt_sha256=release.receipt_sha256,
        expected_source_workspace_receipt_sha256=workspace_receipt_sha256,
        expected_task_count=2,
    )

    assert len(verified.tasks) == verified.receipt.task_count == 2
    assert verified.receipt.source_workspace_build_integrity_verified
    assert verified.receipt.source_workspace_external_receipt_pin_verified
    assert verified.receipt.standalone_public_tree
    assert verified.receipt.public_files_copied_to_new_inodes
    assert not verified.receipt.product_and_sponsor_names_removed
    assert verified.receipt.raw_intervention_and_sponsor_name_fields_omitted
    assert not verified.receipt.organizer_mappings_included
    assert not verified.receipt.private_gold_included
    assert not verified.receipt.secret_key_material_included
    assert not verified.receipt.redistribution_approved
    assert not verified.receipt.distribution_ready
    assert not verified.receipt.distribution_admitted
    assert not verified.receipt.leaderboard_admitted
    assert not verified.receipt.tier_b_admitted
    assert not verified.receipt.tier_a_official
    assert not verified.receipt.sealed_execution_supported
    assert not verified.receipt.identity_contamination_controlled
    assert verified.receipt.residual_model_weight_reidentification_risk
    assert not (release.root / 'public').exists()
    assert not (release.root / 'organizer').exists()
    assert not (release.root / 'private').exists()
    assert (release.root.stat().st_mode & 0o777) == 0o555
    assert all((path.stat().st_mode & 0o777) == 0o555 for path in release.root.rglob('*') if path.is_dir())
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in release.root.rglob('*') if path.is_file())

    source_binding_by_path = {
        item.relative_path.removeprefix('public/'): item
        for item in workspace.receipt.artifacts
        if item.role.value == 'public'
    }
    assert set(source_binding_by_path) == {item.relative_path for item in verified.receipt.artifacts}
    for artifact in verified.receipt.artifacts:
        source_path = workspace.root / artifact.source_workspace_relative_path
        release_path = release.root / artifact.relative_path
        assert source_path.read_bytes() == release_path.read_bytes()
        source_stat = source_path.stat()
        release_stat = release_path.stat()
        assert (source_stat.st_dev, source_stat.st_ino) != (release_stat.st_dev, release_stat.st_ino)


def test_release_contains_no_registry_ids_private_gold_or_raw_keys(tmp_path: Path) -> None:
    workspace, _workspace_receipt_sha256, release = _release(tmp_path)
    task_surface = b''.join((release.root / item.relative_path).read_bytes() for item in release.receipt.artifacts)

    assert b'NCT' not in task_surface.upper()
    assert b'organizer_private_nct_id' not in task_surface
    assert b'registry_outcome_class' not in task_surface
    assert all(canonical_json_bytes(gold) not in task_surface for gold in workspace.gold)
    private_keys = tuple(
        (workspace.root / item.relative_path).read_bytes()
        for item in workspace.receipt.artifacts
        if item.relative_path.endswith('/gold.key')
    )
    assert private_keys
    assert all(key not in task_surface for key in private_keys)
    assert all(
        not {'organizer', 'private'}.intersection(part.casefold() for part in Path(item.relative_path).parts)
        and not item.relative_path.endswith(('.key', '/gold.json'))
        for item in release.receipt.artifacts
    )


def test_release_rejects_wrong_source_pin_nested_output_and_existing_target(tmp_path: Path) -> None:
    workspace, workspace_receipt_sha256 = _workspace(tmp_path)
    with pytest.raises(ExecutionWorkspaceError, match='external pin'):
        build_execution_public_release(
            source_workspace_root=workspace.root,
            expected_source_workspace_receipt_sha256='0' * 64,
            output_root=tmp_path / 'wrong-pin',
            release_id='wrong-pin',
            expected_task_count=2,
        )
    with pytest.raises(ExecutionPublicReleaseError, match='separate from'):
        build_execution_public_release(
            source_workspace_root=workspace.root,
            expected_source_workspace_receipt_sha256=workspace_receipt_sha256,
            output_root=workspace.root / 'public-release',
            release_id='nested',
            expected_task_count=2,
        )

    target = tmp_path / 'already-there'
    target.mkdir()
    with pytest.raises(FileExistsError, match='already exists'):
        build_execution_public_release(
            source_workspace_root=workspace.root,
            expected_source_workspace_receipt_sha256=workspace_receipt_sha256,
            output_root=target,
            release_id='already-there',
            expected_task_count=2,
        )


def test_release_verifier_rejects_wrong_pin_tamper_extra_and_symlink(tmp_path: Path) -> None:
    _workspace_build, _workspace_receipt_sha256, release = _release(tmp_path)
    with pytest.raises(ExecutionPublicReleaseError, match='external pin'):
        verify_execution_public_release(release.root, expected_receipt_sha256='0' * 64)

    task_path = release.root / release.receipt.artifacts[0].relative_path
    original = task_path.read_bytes()
    task_path.chmod(0o644)
    task_path.write_bytes(original + b'\ntamper\n')
    task_path.chmod(0o444)
    with pytest.raises(ExecutionPublicReleaseError, match='does not match receipt'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    task_path.chmod(0o644)
    task_path.write_bytes(original)
    task_path.chmod(0o444)

    release.root.chmod(0o755)
    extra = release.root / 'extra.txt'
    extra.write_text('extra', encoding='utf-8')
    extra.chmod(0o444)
    release.root.chmod(0o555)
    with pytest.raises(ExecutionPublicReleaseError, match='missing or uncommitted'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    release.root.chmod(0o755)
    extra.unlink()
    leak = release.root / 'leak'
    leak.symlink_to(workspace_private := tmp_path / 'source-workspace' / 'private')
    release.root.chmod(0o555)
    assert workspace_private.is_dir()
    with pytest.raises(ExecutionPublicReleaseError, match='symbolic links'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    release.root.chmod(0o755)
    leak.unlink()
    release.root.chmod(0o555)


def test_release_verifier_rejects_extra_empty_directory_and_hardlink(tmp_path: Path) -> None:
    _workspace_build, _workspace_receipt_sha256, release = _release(tmp_path)
    release.root.chmod(0o755)
    extra = release.root / 'empty'
    extra.mkdir(mode=0o555)
    release.root.chmod(0o555)
    with pytest.raises(ExecutionPublicReleaseError, match='missing or uncommitted'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    release.root.chmod(0o755)
    extra.rmdir()
    release.root.chmod(0o555)

    task_path = release.root / release.receipt.artifacts[0].relative_path
    task_parent = task_path.parent
    task_parent.chmod(0o755)
    second_link = task_parent / 'second-link'
    os.link(task_path, second_link)
    task_parent.chmod(0o555)
    with pytest.raises(ExecutionPublicReleaseError, match='missing or uncommitted'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    task_parent.chmod(0o755)
    second_link.unlink()
    task_parent.chmod(0o555)
    outside_link = tmp_path / 'outside-hardlink'
    os.link(task_path, outside_link)
    with pytest.raises(ExecutionPublicReleaseError, match='hard linked'):
        verify_execution_public_release(release.root, expected_receipt_sha256=release.receipt_sha256)
    outside_link.unlink()


def test_release_schema_cannot_claim_distribution_or_admission(tmp_path: Path) -> None:
    _workspace_build, _workspace_receipt_sha256, release = _release(tmp_path)
    payload = release.receipt.model_dump(mode='json')
    for field in (
        'redistribution_approved',
        'distribution_ready',
        'distribution_admitted',
        'leaderboard_admitted',
        'tier_b_admitted',
        'tier_a_official',
        'sealed_execution_supported',
        'identity_contamination_controlled',
    ):
        with pytest.raises(ValidationError):
            ExecutionPublicReleaseReceipt.model_validate({**payload, field: True})
