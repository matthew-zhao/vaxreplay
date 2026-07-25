from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_gold_adapter import _fixture
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.execution_task import ExecutionPrivateGold
from vaxreplay.clinicaltrials.execution_workspace import (
    ExecutionWorkspaceArtifactBinding,
    ExecutionWorkspaceArtifactRole,
    ExecutionWorkspaceBuildReceipt,
    ExecutionWorkspaceError,
    _build_execution_workspace_context_plan,
    _tree_sha256,
    verify_execution_workspace_build,
    write_execution_workspace_build,
)

_ALIAS_KEY = bytes(range(32))
_GOLD_KEY = bytes(range(32, 64))


def _plan(fixture, *, alias_key: bytes = _ALIAS_KEY, cross_lineage: bool = False):
    merged, _archives, queue, adjudications, review_receipt, _trusted, _context = fixture
    assignments = {item.nct_id: item for item in merged.inventory.assignments}
    nct_ids = tuple(sorted(assignments))
    split_by_nct = {nct_ids[0]: Split.TRAIN, nct_ids[1]: Split.TEST}
    if cross_lineage:
        lineage_by_nct = {nct_id: 'lin-forced-shared' for nct_id in nct_ids}
    else:
        lineage_by_nct = {nct_id: assignments[nct_id].lineage_group_id for nct_id in nct_ids}
    return _build_execution_workspace_context_plan(
        inventory=merged.inventory,
        relevance_queue=queue,
        relevance_adjudications=adjudications,
        trusted_relevance_review_receipt_sha256=hashlib.sha256(canonical_json_bytes(review_receipt)).hexdigest(),
        split_manifest_sha256='e' * 64,
        split_by_nct=split_by_nct,
        lineage_by_nct=lineage_by_nct,
        alias_key=alias_key,
    )


def _gold_by_nct(fixture, plan):
    merged = fixture[0]
    labels = {item.nct_id: item for item in merged.labels.labels}
    return {
        entry.organizer_private_nct_id: ExecutionPrivateGold(
            episode_id=entry.context.episode_id,
            target_trial_id=entry.context.target_trial_id,
            organizer_private_nct_id=entry.organizer_private_nct_id,
            organizer_private_decision_record_sha256=entry.decision_source_record_sha256,
            task_context_sha256=entry.context_sha256,
            registry_outcome_class=labels[entry.organizer_private_nct_id].registry_outcome_class,
            enrollment_observation=labels[entry.organizer_private_nct_id].enrollment_observation,
            enrollment_ratio=labels[entry.organizer_private_nct_id].enrollment_ratio,
            primary_completion_observation=labels[entry.organizer_private_nct_id].primary_completion_observation,
            primary_completion_slippage_days=labels[entry.organizer_private_nct_id].primary_completion_slippage_days,
        )
        for entry in plan.entries
    }


def test_context_plan_is_deterministic_outcome_blind_and_identity_scrubbed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        first = _plan(fixture)
        second = _plan(fixture)
        changed_key = _plan(fixture, alias_key=b'x' * 32)

    assert first == second
    assert first.task_count == 2
    assert first.outcome_or_label_data_read is False
    assert first.lineage_split_isolated
    assert not first.public_tasks_created
    assert not first.identity_contamination_controlled
    assert {item.name: item.count for item in first.split_counts} == {'test': 1, 'train': 1}
    assert {item.context.episode_id for item in first.entries}.isdisjoint(
        {item.context.episode_id for item in changed_key.entries}
    )
    private_ids = {item.organizer_private_nct_id for item in first.entries}
    for entry in first.entries:
        assert entry.context.target_trial_id == 'trial-target'
        assert entry.context.cutoff_documents
        assert not entry.context.fact_questions
        public_text = ''.join(item.body for item in entry.context.cutoff_documents)
        assert 'NCT' not in public_text
        assert not any(private_id in public_text for private_id in private_ids)
        assert 'Fictional Biologics' not in public_text
        assert 'Candidate' not in public_text


def test_lineage_cannot_cross_workspace_splits() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        with pytest.raises(ValidationError, match='lineage cannot cross'):
            _plan(fixture, cross_lineage=True)


def test_workspace_build_separates_public_organizer_private_and_verifies_exact_bytes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        gold = _gold_by_nct(fixture, plan)
        build = write_execution_workspace_build(
            plan=plan,
            gold_by_nct=gold,
            private_gold_master_key=_GOLD_KEY,
            output_root=root / 'workspace',
        )
        receipt_sha256 = hashlib.sha256((build.root / 'BUILD-RECEIPT.json').read_bytes()).hexdigest()
        verified = verify_execution_workspace_build(build.root, expected_receipt_sha256=receipt_sha256)

        assert verified.receipt.task_count == 2
        assert len(verified.tasks) == len(verified.gold) == 2
        assert not verified.receipt.leaderboard_admitted
        assert not verified.receipt.sealed_execution_supported
        assert verified.receipt.residual_model_weight_reidentification_risk
        assert (build.root / 'public').is_dir()
        assert (build.root / 'organizer').is_dir()
        assert (build.root / 'private').is_dir()
        public_payload = b''.join(path.read_bytes() for path in (build.root / 'public').rglob('*') if path.is_file())
        assert b'NCT' not in public_payload
        assert b'Fictional Biologics' not in public_payload
        assert b'Candidate' not in public_payload

        task_path = next((build.root / 'public').glob('tasks/*/TASK.md'))
        task_path.chmod(0o644)
        task_path.write_bytes(task_path.read_bytes() + b'\ntamper\n')
        task_path.chmod(0o444)
        with pytest.raises(ExecutionWorkspaceError, match='does not match receipt'):
            verify_execution_workspace_build(build.root, expected_receipt_sha256=receipt_sha256)


def test_workspace_verifier_rejects_uncommitted_symbolic_links() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        build = write_execution_workspace_build(
            plan=plan,
            gold_by_nct=_gold_by_nct(fixture, plan),
            private_gold_master_key=_GOLD_KEY,
            output_root=root / 'workspace',
        )
        receipt_sha256 = hashlib.sha256((build.root / 'BUILD-RECEIPT.json').read_bytes()).hexdigest()
        (build.root / 'public' / 'uncommitted-link').symlink_to('/dev/null')

        with pytest.raises(ExecutionWorkspaceError, match='symbolic links'):
            verify_execution_workspace_build(build.root, expected_receipt_sha256=receipt_sha256)


def test_workspace_verifier_rejects_a_recommitted_extra_public_identity_file() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        build = write_execution_workspace_build(
            plan=plan,
            gold_by_nct=_gold_by_nct(fixture, plan),
            private_gold_master_key=_GOLD_KEY,
            output_root=root / 'workspace',
        )
        leak_payload = b'NCT00000001\n'
        leak_path = build.root / 'public' / 'leak.txt'
        leak_path.write_bytes(leak_payload)
        leak_path.chmod(0o444)
        original = ExecutionWorkspaceBuildReceipt.model_validate_json((build.root / 'BUILD-RECEIPT.json').read_bytes())
        leak_binding = ExecutionWorkspaceArtifactBinding(
            relative_path='public/leak.txt',
            role=ExecutionWorkspaceArtifactRole.PUBLIC,
            sha256=hashlib.sha256(leak_payload).hexdigest(),
            byte_count=len(leak_payload),
            mode='0444',
        )
        artifacts = tuple(sorted((*original.artifacts, leak_binding), key=lambda item: item.relative_path))
        recommitted = original.model_copy(
            update={
                'artifacts': artifacts,
                'public_tree_sha256': _tree_sha256(artifacts, ExecutionWorkspaceArtifactRole.PUBLIC),
            }
        )
        receipt_path = build.root / 'BUILD-RECEIPT.json'
        receipt_path.chmod(0o600)
        receipt_path.write_bytes(canonical_json_bytes(recommitted))
        recommitted_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

        with pytest.raises(ExecutionWorkspaceError, match='semantic artifact inventory'):
            verify_execution_workspace_build(build.root, expected_receipt_sha256=recommitted_sha256)


def test_workspace_build_requires_exact_gold_coverage_and_context_binding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        gold = _gold_by_nct(fixture, plan)
        gold.pop(next(iter(gold)))
        with pytest.raises(ExecutionWorkspaceError, match='every and only'):
            write_execution_workspace_build(
                plan=plan,
                gold_by_nct=gold,
                private_gold_master_key=_GOLD_KEY,
                output_root=root / 'workspace',
            )
