from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_clinicaltrials_execution_gold_adapter import _fixture
from tests.test_clinicaltrials_execution_workspace import _gold_by_nct, _plan
from vaxreplay.clinicaltrials.execution_gold_adapter import (
    ExecutionGoldCohortTarget,
    ExecutionGoldCohortTargetSet,
    ExecutionPrivateGoldSet,
)
from vaxreplay.clinicaltrials.execution_workspace import verify_execution_workspace_build
from vaxreplay.clinicaltrials.execution_workspace_pipeline import (
    ExecutionWorkspacePipelineError,
    ExecutionWorkspacePreparationUpstream,
    ExecutionWorkspacePreparedCaseBinding,
    _write_preparation,
    finalize_execution_workspace,
    read_workspace_secret_key,
    verify_execution_workspace_preparation,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prepared(root: Path, fixture):
    plan = _plan(fixture)
    targets = ExecutionGoldCohortTargetSet(
        cohort_id='test-final-workspace-cohort',
        targets=tuple(
            sorted(
                (
                    ExecutionGoldCohortTarget(
                        organizer_private_nct_id=entry.organizer_private_nct_id,
                        context=entry.context,
                    )
                    for entry in plan.entries
                ),
                key=lambda item: (item.context.anchor_date, item.organizer_private_nct_id),
            )
        ),
        final_workspace_contexts_bound=True,
    )
    bindings = tuple(
        ExecutionWorkspacePreparedCaseBinding(
            organizer_private_nct_id=entry.organizer_private_nct_id,
            episode_id=entry.context.episode_id,
            context_sha256=entry.context_sha256,
            lineage_case_assignment_sha256=_sha256(entry.organizer_private_nct_id.encode()),
            lineage_group_id=entry.lineage_group_id,
            split=entry.split,
        )
        for entry in plan.entries
    )
    upstream = ExecutionWorkspacePreparationUpstream(
        merge_receipt_sha256='1' * 64,
        merged_inventory_artifact_sha256='2' * 64,
        relevance_review_receipt_sha256=plan.trusted_relevance_review_receipt_sha256,
        relevance_queue_artifact_sha256='3' * 64,
        relevance_adjudication_artifact_sha256='4' * 64,
        lineage_split_receipt_sha256='5' * 64,
        lineage_split_assignments_sha256=plan.split_manifest_sha256,
        lineage_split_policy_sha256='6' * 64,
        lineage_id_key_commitment_sha256='7' * 64,
    )
    return _write_preparation(
        plan=plan,
        targets=targets,
        case_bindings=bindings,
        upstream=upstream,
        output_root=root / 'prepared',
    )


def test_preparation_has_exact_private_tree_and_external_receipt_verification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _prepared(tmp_path, fixture)
    verified = verify_execution_workspace_preparation(
        built.root,
        expected_receipt_sha256=built.receipt_sha256,
    )

    assert verified.plan == built.plan
    assert verified.targets == built.targets
    assert verified.targets.final_workspace_contexts_bound
    assert verified.receipt.task_count == 2
    assert not verified.receipt.leaderboard_admitted
    assert {path.relative_to(built.root).as_posix() for path in built.root.rglob('*') if path.is_file()} == {
        'PREPARE-RECEIPT.json',
        'organizer/cohort-targets.json',
        'organizer/context-plan.json',
    }
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in built.root.rglob('*') if path.is_file())


def test_preparation_verifier_rejects_tamper_mode_extra_file_and_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    built = _prepared(tmp_path, fixture)
    plan_path = built.root / 'organizer' / 'context-plan.json'
    plan_path.chmod(0o640)
    with pytest.raises(ExecutionWorkspacePipelineError, match='mode 0600'):
        verify_execution_workspace_preparation(built.root, expected_receipt_sha256=built.receipt_sha256)
    plan_path.chmod(0o600)

    extra = built.root / 'organizer' / 'extra.json'
    extra.write_text('{}')
    os.chmod(extra, 0o600)
    with pytest.raises(ExecutionWorkspacePipelineError, match='missing or uncommitted'):
        verify_execution_workspace_preparation(built.root, expected_receipt_sha256=built.receipt_sha256)
    extra.unlink()

    (built.root / 'organizer' / 'link').symlink_to(plan_path)
    with pytest.raises(ExecutionWorkspacePipelineError, match='symbolic link'):
        verify_execution_workspace_preparation(built.root, expected_receipt_sha256=built.receipt_sha256)


def test_workspace_secret_key_requires_exact_mode_and_regular_non_symlink(tmp_path: Path) -> None:
    key = tmp_path / 'alias.key'
    key.write_bytes(b'a' * 32)
    os.chmod(key, 0o600)
    assert read_workspace_secret_key(key, purpose='alias') == b'a' * 32

    os.chmod(key, 0o640)
    with pytest.raises(ExecutionWorkspacePipelineError, match='mode 0600'):
        read_workspace_secret_key(key, purpose='alias')
    os.chmod(key, 0o600)
    link = tmp_path / 'alias-link.key'
    link.symlink_to(key)
    with pytest.raises(ExecutionWorkspacePipelineError, match='symbolic link'):
        read_workspace_secret_key(link, purpose='alias')


def test_finalize_accepts_only_exact_split_bound_cohort_derivation(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepared(tmp_path, fixture)
    gold = _gold_by_nct(fixture, prepared.plan)
    gold_set = ExecutionPrivateGoldSet(
        cohort_id=prepared.targets.cohort_id,
        records=tuple(sorted(gold.values(), key=lambda item: item.organizer_private_nct_id)),
    )
    case_receipts = tuple(
        SimpleNamespace(
            organizer_private_nct_id=binding.organizer_private_nct_id,
            episode_id=binding.episode_id,
            task_context_sha256=binding.context_sha256,
            lineage_case_assignment_sha256=binding.lineage_case_assignment_sha256,
            lineage_group_id=binding.lineage_group_id,
            split=binding.split,
        )
        for binding in prepared.receipt.case_bindings
    )
    split_binding = SimpleNamespace(
        split_receipt_sha256=prepared.receipt.lineage_split_receipt_sha256,
        split_assignments_sha256=prepared.receipt.lineage_split_assignments_sha256,
        split_policy_sha256=prepared.receipt.lineage_split_policy_sha256,
        id_key_commitment_sha256=prepared.receipt.lineage_id_key_commitment_sha256,
    )
    derivation = SimpleNamespace(
        targets=prepared.targets,
        private_gold=gold_set,
        receipt=SimpleNamespace(
            final_workspace_contexts_bound=True,
            split_inventory_bound=True,
            lineage_split_safe=True,
            split_binding=split_binding,
            case_receipts=case_receipts,
        ),
    )
    monkeypatch.setattr(
        'vaxreplay.clinicaltrials.execution_workspace_pipeline.load_execution_gold_cohort_derivation',
        lambda *_args, **_kwargs: SimpleNamespace(derivation=derivation),
    )
    finalized = finalize_execution_workspace(
        preparation_root=prepared.root,
        expected_preparation_receipt_sha256=prepared.receipt_sha256,
        gold_derivation_root=tmp_path / 'mock-gold',
        expected_gold_derivation_receipt_sha256='8' * 64,
        private_gold_master_key=b'g' * 32,
        output_root=tmp_path / 'workspace',
    )

    assert finalized.build.receipt.task_count == 2
    assert not finalized.externally_pinned_receipt_verified
    verify_execution_workspace_build(
        finalized.build.root,
        expected_receipt_sha256=finalized.receipt_sha256,
    )

    derivation.targets = ExecutionGoldCohortTargetSet(
        cohort_id='wrong-cohort',
        targets=prepared.targets.targets,
        final_workspace_contexts_bound=True,
    )
    with pytest.raises(ExecutionWorkspacePipelineError, match='exact prepared target set'):
        finalize_execution_workspace(
            preparation_root=prepared.root,
            expected_preparation_receipt_sha256=prepared.receipt_sha256,
            gold_derivation_root=tmp_path / 'mock-gold',
            expected_gold_derivation_receipt_sha256='8' * 64,
            private_gold_master_key=b'g' * 32,
            output_root=tmp_path / 'wrong-workspace',
        )
