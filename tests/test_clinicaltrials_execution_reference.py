from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from tests.test_clinicaltrials_execution_gold_adapter import _fixture
from tests.test_clinicaltrials_execution_workspace import _gold_by_nct, _plan
from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.execution_reference import run_uniform_execution_reference
from vaxreplay.clinicaltrials.execution_workspace import ExecutionWorkspaceError, write_execution_workspace_build


def test_uniform_reference_scores_the_exact_complete_workspace_without_becoming_a_model_result() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        build = write_execution_workspace_build(
            plan=plan,
            gold_by_nct=_gold_by_nct(fixture, plan),
            private_gold_master_key=b'g' * 32,
            output_root=root / 'workspace',
        )
        receipt_sha256 = hashlib.sha256((build.root / 'BUILD-RECEIPT.json').read_bytes()).hexdigest()

        run = run_uniform_execution_reference(
            workspace_root=build.root,
            expected_workspace_receipt_sha256=receipt_sha256,
            lineage_split_manifest_sha256=plan.split_manifest_sha256,
            gold_derivation_receipt_sha256='a' * 64,
            cohort_id='fixture-uniform-reference',
            evaluation_split=Split.TEST,
        )

    assert run.manifest.task_count == 1
    assert len(run.submissions.submissions) == run.result.task_count == 1
    assert run.result.valid_task_count == 1
    assert run.result.evaluation_split == Split.TEST
    assert 0.0 <= run.result.metrics.mean_reward <= 1.0
    assert run.result.development_only and not run.result.leaderboard_admitted
    assert 'model' not in run.result.model_dump()


def test_uniform_reference_rejects_a_different_split_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        plan = _plan(fixture)
        build = write_execution_workspace_build(
            plan=plan,
            gold_by_nct=_gold_by_nct(fixture, plan),
            private_gold_master_key=b'g' * 32,
            output_root=root / 'workspace',
        )
        receipt_sha256 = hashlib.sha256((build.root / 'BUILD-RECEIPT.json').read_bytes()).hexdigest()

        with pytest.raises(ExecutionWorkspaceError, match='split manifest'):
            run_uniform_execution_reference(
                workspace_root=build.root,
                expected_workspace_receipt_sha256=receipt_sha256,
                lineage_split_manifest_sha256='f' * 64,
                gold_derivation_receipt_sha256='a' * 64,
                cohort_id='fixture-uniform-reference',
                evaluation_split=Split.TEST,
            )
