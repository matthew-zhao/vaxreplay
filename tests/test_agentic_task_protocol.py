from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_scoring import _case as _execution_case
from tests.test_clinicaltrials_execution_scoring import _submission as _execution_submission
from vaxreplay.agentic.protocol import AgenticExecutionPolicy
from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.schema import AgenticFactQuery, AgenticTaskEnvelope, AgenticValueType
from vaxreplay.agentic.scoring import (
    AgenticDecision,
    AgenticSubmissionV1,
    CandidateProbability,
    DecisionStatus,
    FactAnswer,
    FactAnswerStatus,
)
from vaxreplay.agentic.task_protocol import (
    AgenticTaskInvocation,
    agentic_task_invocation_sha256,
    parse_submission_for_invocation,
    submission_json_schema_for_invocation,
    submission_json_schema_sha256,
    validate_submission_for_invocation,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner.schema import IsolationTier

_WORKSPACE_SHA256 = '1' * 64


def _ranking_task() -> AgenticTaskEnvelope:
    return AgenticTaskEnvelope(
        task_id='ranking-task',
        episode_id='ranking-episode',
        episode_manifest_sha256='2' * 64,
        decision_at=datetime(2020, 2, 1, tzinfo=UTC),
        task_type='candidate_ranking',
        candidate_ids=('candidate-001', 'candidate-002'),
        portfolio_size=1,
        instructions='Rank the candidates using only cutoff-safe evidence.',
        fact_queries=(
            AgenticFactQuery(
                query_id='phase',
                description='Extract the phase.',
                value_type=AgenticValueType.STRING,
            ),
        ),
        historically_preregistered=False,
    )


def _ranking_submission() -> AgenticSubmissionV1:
    return AgenticSubmissionV1(
        task_id='ranking-task',
        workspace_manifest_sha256=_WORKSPACE_SHA256,
        fact_answers=(FactAnswer(query_id='phase', status=FactAnswerStatus.NOT_FOUND),),
        decision=AgenticDecision(
            status=DecisionStatus.INSUFFICIENT_EVIDENCE,
            ranking=('candidate-001', 'candidate-002'),
            advancement_probabilities=(
                CandidateProbability(candidate_id='candidate-001', probability=0.5),
                CandidateProbability(candidate_id='candidate-002', probability=0.5),
            ),
        ),
    )


def test_task_union_derives_closed_response_protocols_and_exact_commitments() -> None:
    ranking = AgenticTaskInvocation.from_task(
        _ranking_task(),
        workspace_manifest_sha256=_WORKSPACE_SHA256,
    )
    clinical_task, _ = _execution_case()
    clinical = AgenticTaskInvocation.from_task(
        clinical_task,
        workspace_manifest_sha256=_WORKSPACE_SHA256,
    )

    assert ranking.response_protocol == AgenticResponseProtocol.RANKING
    assert clinical.response_protocol == AgenticResponseProtocol.CLINICAL_EXECUTION
    assert len(agentic_task_invocation_sha256(ranking)) == 64
    assert AgenticTaskInvocation.model_validate_json(canonical_json_bytes(ranking)) == ranking
    assert AgenticTaskInvocation.model_validate_json(canonical_json_bytes(clinical)) == clinical
    ranking_schema = submission_json_schema_for_invocation(ranking)
    clinical_schema = submission_json_schema_for_invocation(clinical)
    assert 'decision' in ranking_schema['properties']
    assert 'registry_outcome_probabilities' not in ranking_schema['properties']
    assert 'registry_outcome_probabilities' in clinical_schema['properties']
    assert 'decision' not in clinical_schema['properties']
    assert len(submission_json_schema_sha256(ranking)) == 64
    assert submission_json_schema_sha256(ranking) != submission_json_schema_sha256(clinical)

    with pytest.raises(ValidationError, match='response protocol'):
        AgenticTaskInvocation.model_validate_json(
            canonical_json_bytes(
                {
                    **clinical.model_dump(mode='json'),
                    'response_protocol': AgenticResponseProtocol.RANKING.value,
                }
            )
        )


def test_submission_validation_is_task_bound_for_both_families() -> None:
    ranking = AgenticTaskInvocation.from_task(
        _ranking_task(),
        workspace_manifest_sha256=_WORKSPACE_SHA256,
    )
    clinical_task, _ = _execution_case()
    clinical_submission = _execution_submission(clinical_task)
    clinical = AgenticTaskInvocation.from_task(
        clinical_task,
        workspace_manifest_sha256=_WORKSPACE_SHA256,
    )

    validate_submission_for_invocation(ranking, _ranking_submission())
    validate_submission_for_invocation(clinical, clinical_submission)
    assert parse_submission_for_invocation(ranking, canonical_json_bytes(_ranking_submission())) == (
        _ranking_submission()
    )
    assert parse_submission_for_invocation(clinical, canonical_json_bytes(clinical_submission)) == (clinical_submission)

    with pytest.raises(ValueError, match='protocol'):
        validate_submission_for_invocation(ranking, clinical_submission)
    with pytest.raises(ValueError, match='different task or workspace'):
        validate_submission_for_invocation(
            ranking,
            _ranking_submission().model_copy(update={'workspace_manifest_sha256': '3' * 64}),
        )
    with pytest.raises(ValueError, match='public task contract'):
        validate_submission_for_invocation(
            clinical,
            clinical_submission.model_copy(update={'episode_id': 'different-episode'}),
        )


def test_clinical_execution_policy_is_explicitly_development_only() -> None:
    common = {
        'response_protocol': AgenticResponseProtocol.CLINICAL_EXECUTION,
        'required_workspace_broker_id': 'broker',
        'required_workspace_broker_version': '1',
        'required_workspace_broker_executable_sha256': '4' * 64,
    }
    policy = AgenticExecutionPolicy(required_isolation=IsolationTier.DEVELOPMENT, **common)
    assert policy.response_protocol == AgenticResponseProtocol.CLINICAL_EXECUTION

    with pytest.raises(ValidationError, match='development-only'):
        AgenticExecutionPolicy(required_isolation=IsolationTier.OFFICIAL, **common)
