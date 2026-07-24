from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.agentic_helpers import bind_episode_manifest, selection_contract
from vaxreplay.agentic.admission import AgenticAdmissionError, AgenticWorkspaceAdmission
from vaxreplay.agentic.schema import (
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticDerivedMetric,
    AgenticFactQuery,
    AgenticMediaType,
    AgenticTaskEnvelope,
    AgenticValueType,
    AgenticWorkspaceSource,
    ArtifactTemporalProof,
    AvailabilityInterval,
    TemporalProofKind,
    agentic_model_sha256,
)
from vaxreplay.agentic.scoring import (
    AGENTIC_REWARD_VERSION,
    MAX_CITATIONS_PER_FACT,
    MAX_SOURCE_SPAN_BYTES,
    MAX_TOTAL_CITATIONS,
    AgenticDecision,
    AgenticIssueCode,
    AgenticPrivateGoldV1,
    AgenticScoringContract,
    AgenticSubmissionEvaluator,
    AgenticSubmissionV1,
    CandidateProbability,
    DecisionStatus,
    FactAnswer,
    FactAnswerStatus,
    GoldCandidateDecision,
    GoldEvidenceGroup,
    GoldFactLabel,
    GoldMetricLabel,
    GoldSupportGroup,
    MetricAnswerStatus,
    SourceSpan,
    TypedValue,
    TypedValueKind,
    agentic_private_gold_commitment,
    oracle_submission,
    score_agentic_submission,
)
from vaxreplay.agentic.workspace import LoadedAgenticWorkspace, build_agentic_workspace
from vaxreplay.case_schema import ScoreStatus


@dataclass(frozen=True)
class _ScoringCase:
    workspace: LoadedAgenticWorkspace
    admission: AgenticWorkspaceAdmission
    admission_sha256: str
    gold: AgenticPrivateGoldV1
    gold_commitment_key: bytes
    gold_commitment_sha256: str
    relevant: bytes
    distractor: bytes

    @property
    def evaluator(self) -> AgenticSubmissionEvaluator:
        return AgenticSubmissionEvaluator(
            self.workspace,
            self.admission,
            self.admission_sha256,
            self.gold,
            self.gold_commitment_key,
            self.gold_commitment_sha256,
        )

    @property
    def oracle(self) -> AgenticSubmissionV1:
        return oracle_submission(AgenticScoringContract.from_workspace(self.workspace), self.gold)


def _number(value: float | int, unit: str) -> TypedValue:
    return TypedValue(kind=TypedValueKind.NUMBER, number=value, unit=unit)


def _interval() -> AvailabilityInterval:
    return AvailabilityInterval(
        lower_at=datetime(2020, 2, 1, tzinfo=UTC),
        upper_at=datetime(2020, 2, 1, 23, 59, 59, tzinfo=UTC),
        precision='day',
        timezone_basis='UTC upper bound',
    )


def _source(source_id: str, path: str, content: bytes) -> AgenticWorkspaceSource:
    digest = hashlib.sha256(content).hexdigest()
    interval = _interval()
    proof = ArtifactTemporalProof(
        proof_id=f'proof-{source_id}',
        kind=TemporalProofKind.FIXTURE,
        artifact_sha256=digest,
        artifact_bytes=len(content),
        witnessed=interval,
        authority_id='test-fixture',
        proof_sha256='f' * 64,
        proof_bytes=1,
        verification_uri='fixture://agentic-scoring',
    )
    return AgenticWorkspaceSource(
        source_id=source_id,
        path=path,
        display_title=f'Source {source_id.removeprefix("source-")}',
        artifact_kind=AgenticArtifactKind.RAW,
        media_type=AgenticMediaType.TEXT,
        sha256=digest,
        byte_count=len(content),
        source_url='fixture://agentic-scoring/source',
        license_id='test-fixture',
        retrieved_at=datetime(2020, 2, 1, tzinfo=UTC),
        temporal_proofs=(proof,),
        selected_proof_id=proof.proof_id,
        effective_available_at_upper=interval.upper_at,
    )


def _span(content: bytes, needle: bytes, source_id: str = 'source-002') -> SourceSpan:
    start = content.index(needle)
    return SourceSpan(source_id=source_id, start_byte=start, end_byte=start + len(needle))


def _build_case(
    tmp_path: Path,
    *,
    candidate_ids: tuple[str, str, str] = ('candidate-001', 'candidate-002', 'candidate-003'),
) -> _ScoringCase:
    relevant = (
        b'candidate-001 dose: 120 microgram\n'
        b'candidate-002 dose: 80 microgram\n'
        b'protocol context only\n' + (b'x' * (MAX_SOURCE_SPAN_BYTES + 100)) + '\nemoji marker: \U0001f489\n'.encode()
    )
    distractor = b'candidate-003 has no dose in the frozen protocol\n'
    sources = (
        _source('source-001', 'sources/source-001.txt', distractor),
        _source('source-002', 'sources/source-002.txt', relevant),
    )
    task = AgenticTaskEnvelope(
        task_id='agentic-scoring-task',
        episode_id='episode-1',
        episode_manifest_sha256='e' * 64,
        decision_at=datetime(2020, 2, 2, tzinfo=UTC),
        task_type='early_clinical_arm_prioritization',
        candidate_ids=candidate_ids,
        portfolio_size=1,
        instructions='Extract both doses, compute the difference, and prioritize one candidate.',
        fact_queries=(
            AgenticFactQuery(
                query_id='dose-a',
                description='Dose for candidate A',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
                candidate_id=candidate_ids[0],
            ),
            AgenticFactQuery(
                query_id='dose-b',
                description='Dose for candidate B',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
                candidate_id=candidate_ids[1],
            ),
        ),
        derived_metrics=(
            AgenticDerivedMetric(
                metric_id='dose-b-half',
                description='Candidate B dose divided by two',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
                formula_id='divide-by-two-v1',
                dependency_query_ids=('dose-b',),
            ),
        ),
        historically_preregistered=False,
    )
    task, episode_manifest = bind_episode_manifest(task)
    build_policy, discovery_manifest = selection_contract(task, sources)
    workspace = build_agentic_workspace(
        workspace_id='agentic-scoring-workspace',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        assurance_profile=AgenticAssuranceProfile.FIXTURE,
        sources=sources,
        transformations=(),
        source_bytes={
            'source-001': distractor,
            'source-002': relevant,
        },
        output_root=tmp_path / 'workspace',
    )
    admission = AgenticWorkspaceAdmission(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        build_policy_sha256=workspace.manifest.build_policy_sha256,
        discovery_manifest_sha256=workspace.manifest.discovery_manifest_sha256,
        alias_seed_commitment_sha256=workspace.manifest.alias_seed_commitment_sha256,
        temporal_admission_sha256='a' * 64,
        contamination_binding_sha256='b' * 64,
        contamination_admission_policy_sha256='d' * 64,
        contamination_audit_manifest_sha256='c' * 64,
        assurance_profile=AgenticAssuranceProfile.FIXTURE,
        admitted_use='fixture',
        official_release_eligible=False,
        selection_precommitted_before_cutoff=False,
        residual_retrospective_selection_contamination=True,
    )
    span_a = _span(relevant, b'candidate-001 dose: 120 microgram')
    span_b = _span(relevant, b'candidate-002 dose: 80 microgram')
    scoring_contract = AgenticScoringContract.from_workspace(workspace)
    gold = AgenticPrivateGoldV1(
        reward_version=AGENTIC_REWARD_VERSION,
        task_id=task.task_id,
        workspace_manifest_sha256=workspace.manifest_sha256,
        scoring_contract_sha256=agentic_model_sha256(scoring_contract),
        fact_labels=(
            GoldFactLabel(
                query_id='dose-a',
                status=FactAnswerStatus.OBSERVED,
                accepted_values=(_number(120, 'microgram'),),
                support_groups=(
                    GoldSupportGroup(
                        group_id='support-dose-a',
                        evidence_group_id='evidence-dose-a',
                        alternatives=(span_a,),
                    ),
                ),
            ),
            GoldFactLabel(
                query_id='dose-b',
                status=FactAnswerStatus.OBSERVED,
                accepted_values=(_number(80, 'microgram'),),
                support_groups=(
                    GoldSupportGroup(
                        group_id='support-dose-b',
                        evidence_group_id='evidence-dose-b',
                        alternatives=(span_b,),
                    ),
                ),
            ),
        ),
        metric_labels=(
            GoldMetricLabel(
                metric_id='dose-b-half',
                status=MetricAnswerStatus.COMPUTED,
                formula_id='divide-by-two-v1',
                dependency_query_ids=('dose-b',),
                accepted_values=(_number(40, 'microgram'),),
            ),
        ),
        evidence_groups=(
            GoldEvidenceGroup(group_id='evidence-dose-a', acceptable_source_ids=('source-002',)),
            GoldEvidenceGroup(group_id='evidence-dose-b', acceptable_source_ids=('source-002',)),
        ),
        decision_labels=(
            GoldCandidateDecision(
                candidate_id=candidate_ids[0],
                relevance=3,
                utility=1,
                historically_advanced=True,
            ),
            GoldCandidateDecision(
                candidate_id=candidate_ids[1],
                relevance=2,
                utility=0.5,
                historically_advanced=False,
            ),
            GoldCandidateDecision(
                candidate_id=candidate_ids[2],
                relevance=0,
                utility=0,
                historically_advanced=False,
            ),
        ),
    )
    gold_commitment_key = b'agentic-scoring-private-gold-key' + b'\x00' * 8
    return _ScoringCase(
        workspace=workspace,
        admission=admission,
        admission_sha256=agentic_model_sha256(admission),
        gold=gold,
        gold_commitment_key=gold_commitment_key,
        gold_commitment_sha256=agentic_private_gold_commitment(gold, gold_commitment_key),
        relevant=relevant,
        distractor=distractor,
    )


def _score(case: _ScoringCase, submission: AgenticSubmissionV1):
    return score_agentic_submission(
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        gold=case.gold,
        gold_commitment_key=case.gold_commitment_key,
        expected_gold_commitment_sha256=case.gold_commitment_sha256,
        submission=submission,
    )


def _score_with_gold(
    case: _ScoringCase,
    gold: AgenticPrivateGoldV1,
    submission: AgenticSubmissionV1,
):
    return score_agentic_submission(
        workspace=case.workspace,
        admission=case.admission,
        expected_admission_sha256=case.admission_sha256,
        gold=gold,
        gold_commitment_key=case.gold_commitment_key,
        expected_gold_commitment_sha256=agentic_private_gold_commitment(gold, case.gold_commitment_key),
        submission=submission,
    )


def _evaluator_with_gold(case: _ScoringCase, gold: AgenticPrivateGoldV1) -> AgenticSubmissionEvaluator:
    return AgenticSubmissionEvaluator(
        case.workspace,
        case.admission,
        case.admission_sha256,
        gold,
        case.gold_commitment_key,
        agentic_private_gold_commitment(gold, case.gold_commitment_key),
    )


def _replace_fact(
    submission: AgenticSubmissionV1,
    query_id: str,
    replacement: FactAnswer,
) -> AgenticSubmissionV1:
    return submission.model_copy(
        update={
            'fact_answers': tuple(
                replacement if answer.query_id == query_id else answer for answer in submission.fact_answers
            )
        }
    )


def test_oracle_is_one_and_forecasts_are_diagnostic_only(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    score = _score(case, oracle)

    assert score.status == ScoreStatus.VALID
    assert score.reward_version == AGENTIC_REWARD_VERSION
    assert score.scoring_contract_sha256 == agentic_model_sha256(AgenticScoringContract.from_workspace(case.workspace))
    assert score.private_gold_commitment_sha256 == case.gold_commitment_sha256
    assert score.private_gold_commitment_key_id == hashlib.sha256(case.gold_commitment_key).hexdigest()
    assert score.submission_sha256 == agentic_model_sha256(oracle)
    assert score.assurance_profile == AgenticAssuranceProfile.FIXTURE
    assert score.admitted_use == 'fixture'
    assert score.reward == pytest.approx(1.0)
    assert score.retrieval_f1 == pytest.approx(1.0)
    assert score.extraction_score == pytest.approx(1.0)
    assert score.analysis_score == pytest.approx(1.0)
    assert score.citation_f1 == pytest.approx(1.0)
    assert score.decision_score == pytest.approx(1.0)
    assert score.advancement_brier == pytest.approx(0.0)
    assert score.advancement_prevalence == pytest.approx(1 / 3)

    uncalibrated = oracle.model_copy(
        update={
            'decision': AgenticDecision(
                status=DecisionStatus.RECOMMEND,
                ranking=oracle.decision.ranking,
                portfolio=oracle.decision.portfolio,
                advancement_probabilities=tuple(
                    CandidateProbability(candidate_id=candidate_id, probability=0.5)
                    for candidate_id in oracle.decision.ranking
                ),
            )
        }
    )
    diagnostic = _score(case, uncalibrated)
    assert diagnostic.reward == pytest.approx(score.reward)
    assert diagnostic.advancement_brier > score.advancement_brier


def test_insufficient_evidence_cannot_get_oracle_decision_credit(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    insufficient = oracle.model_copy(
        update={
            'decision': AgenticDecision(
                status=DecisionStatus.INSUFFICIENT_EVIDENCE,
                ranking=oracle.decision.ranking,
                portfolio=(),
                advancement_probabilities=oracle.decision.advancement_probabilities,
            )
        }
    )

    score = _score(case, insufficient)
    assert score.status == ScoreStatus.VALID
    assert score.ndcg_at_k == 0.0
    assert score.top_k_utility == 0.0
    assert score.decision_score == 0.0
    assert score.reward == 0.0


def test_advancement_count_is_independent_of_portfolio_and_brier_uses_empirical_prevalence(
    tmp_path: Path,
) -> None:
    case = _build_case(tmp_path)
    labels = tuple(
        label.model_copy(update={'historically_advanced': label.candidate_id != case.workspace.task.candidate_ids[2]})
        for label in case.gold.decision_labels
    )
    gold = case.gold.model_copy(update={'decision_labels': labels})
    oracle = oracle_submission(AgenticScoringContract.from_workspace(case.workspace), gold)
    uniform = oracle.model_copy(
        update={
            'decision': oracle.decision.model_copy(
                update={
                    'advancement_probabilities': tuple(
                        CandidateProbability(candidate_id=candidate_id, probability=2 / 3)
                        for candidate_id in case.workspace.task.candidate_ids
                    )
                }
            )
        }
    )

    score = _score_with_gold(case, gold, uniform)
    assert sum(label.historically_advanced for label in labels) == 2
    assert case.workspace.task.portfolio_size == 1
    assert score.status == ScoreStatus.VALID
    assert score.advancement_prevalence == pytest.approx(2 / 3)
    assert score.advancement_brier == pytest.approx(2 / 9)
    assert score.advancement_brier_skill == pytest.approx(0.0)


def test_brier_skill_is_undefined_for_degenerate_empirical_prevalence(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    labels = tuple(label.model_copy(update={'historically_advanced': False}) for label in case.gold.decision_labels)
    gold = case.gold.model_copy(update={'decision_labels': labels})
    oracle = oracle_submission(AgenticScoringContract.from_workspace(case.workspace), gold)

    score = _score_with_gold(case, gold, oracle)
    assert score.status == ScoreStatus.VALID
    assert score.advancement_prevalence == 0.0
    assert score.advancement_brier == 0.0
    assert score.advancement_brier_skill is None


def test_candidate_alias_permutation_preserves_equivalent_scores(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    candidate_ids = case.workspace.task.candidate_ids
    base_probabilities = {candidate_ids[0]: 0.6, candidate_ids[1]: 0.3, candidate_ids[2]: 0.1}
    base_ranking = (candidate_ids[1], candidate_ids[0], candidate_ids[2])
    base_submission = case.oracle.model_copy(
        update={
            'decision': AgenticDecision(
                status=DecisionStatus.RECOMMEND,
                ranking=base_ranking,
                portfolio=base_ranking[:1],
                advancement_probabilities=tuple(
                    CandidateProbability(candidate_id=candidate_id, probability=base_probabilities[candidate_id])
                    for candidate_id in candidate_ids
                ),
            )
        }
    )
    base_score = _score(case, base_submission)

    # Reassign each private candidate identity to a different neutral public alias, then
    # transform the submission through that same permutation.
    alias_by_candidate = {
        candidate_ids[0]: candidate_ids[1],
        candidate_ids[1]: candidate_ids[2],
        candidate_ids[2]: candidate_ids[0],
    }
    permuted_labels = tuple(
        label.model_copy(update={'candidate_id': alias_by_candidate[label.candidate_id]})
        for label in case.gold.decision_labels
    )
    permuted_gold = case.gold.model_copy(update={'decision_labels': permuted_labels})
    permuted_ranking = tuple(alias_by_candidate[candidate_id] for candidate_id in base_ranking)
    permuted_submission = case.oracle.model_copy(
        update={
            'decision': AgenticDecision(
                status=DecisionStatus.RECOMMEND,
                ranking=permuted_ranking,
                portfolio=permuted_ranking[:1],
                advancement_probabilities=tuple(
                    CandidateProbability(
                        candidate_id=alias_by_candidate[candidate_id],
                        probability=base_probabilities[candidate_id],
                    )
                    for candidate_id in candidate_ids
                ),
            )
        }
    )
    permuted_score = _score_with_gold(case, permuted_gold, permuted_submission)

    assert 0.0 < base_score.reward < 1.0
    assert permuted_score.metrics() == pytest.approx(base_score.metrics())


def test_exact_query_and_metric_coverage_fails_closed(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle

    missing_fact = oracle.model_copy(update={'fact_answers': oracle.fact_answers[:1]})
    score = _score(case, missing_fact)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_FACT_COVERAGE in {issue.code for issue in score.issues}

    duplicated_fact = oracle.model_copy(update={'fact_answers': (*oracle.fact_answers, oracle.fact_answers[0])})
    score = _score(case, duplicated_fact)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_FACT_COVERAGE in {issue.code for issue in score.issues}

    missing_metric = oracle.model_copy(update={'derived_metrics': ()})
    score = _score(case, missing_metric)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_METRIC_COVERAGE in {issue.code for issue in score.issues}

    computed_without_dependency = _replace_fact(
        oracle,
        'dose-b',
        FactAnswer(query_id='dose-b', status=FactAnswerStatus.NOT_FOUND),
    )
    score = _score(case, computed_without_dependency)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_METRIC_COVERAGE in {issue.code for issue in score.issues}

    wrong_formula = oracle.model_copy(
        update={'derived_metrics': (oracle.derived_metrics[0].model_copy(update={'formula_id': 'divide-by-three-v1'}),)}
    )
    score = _score(case, wrong_formula)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_METRIC_COVERAGE in {issue.code for issue in score.issues}


def test_derived_metric_credit_requires_correct_dependency_facts(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    dose_b = next(answer for answer in oracle.fact_answers if answer.query_id == 'dose-b')
    wrong_dependency = _replace_fact(
        oracle,
        'dose-b',
        dose_b.model_copy(update={'value': _number(999, 'microgram')}),
    )

    asserted_score = _score(case, wrong_dependency)
    assert asserted_score.status == ScoreStatus.VALID
    assert asserted_score.analysis_signed_utility == -1.0
    assert asserted_score.analysis_score == 0.0
    assert asserted_score.reward == 0.0

    abstained_metric = wrong_dependency.model_copy(
        update={
            'derived_metrics': (
                wrong_dependency.derived_metrics[0].model_copy(
                    update={'status': MetricAnswerStatus.NOT_COMPUTABLE, 'value': None}
                ),
            )
        }
    )
    abstained_score = _score(case, abstained_metric)
    assert abstained_score.status == ScoreStatus.VALID
    assert abstained_score.analysis_signed_utility == 0.0
    assert abstained_score.analysis_score == 0.5
    assert asserted_score.analysis_signed_utility < abstained_score.analysis_signed_utility
    assert asserted_score.analysis_score < abstained_score.analysis_score


def test_typed_value_contract_is_enforced(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    original = oracle.fact_answers[0]
    wrong_type = FactAnswer(
        query_id='dose-a',
        status=FactAnswerStatus.OBSERVED,
        value=TypedValue(kind=TypedValueKind.STRING, text='120'),
        citations=original.citations,
    )
    score = _score(case, _replace_fact(oracle, 'dose-a', wrong_type))

    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.INVALID_FACT_VALUE_TYPE in {issue.code for issue in score.issues}


@pytest.mark.parametrize('span_kind', ['past_end', 'utf8_midpoint', 'overlong'])
def test_byte_spans_require_bounds_utf8_boundaries_and_length_cap(tmp_path: Path, span_kind: str) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    original = oracle.fact_answers[0]
    if span_kind == 'past_end':
        span = SourceSpan(
            source_id='source-002',
            start_byte=len(case.relevant) - 1,
            end_byte=len(case.relevant) + 1,
        )
        expected = AgenticIssueCode.INVALID_SPAN
    elif span_kind == 'utf8_midpoint':
        emoji = case.relevant.index('\U0001f489'.encode())
        span = SourceSpan(source_id='source-002', start_byte=emoji + 1, end_byte=emoji + 2)
        expected = AgenticIssueCode.INVALID_SPAN
    else:
        span = SourceSpan(source_id='source-002', start_byte=0, end_byte=MAX_SOURCE_SPAN_BYTES + 1)
        expected = AgenticIssueCode.OVERLONG_SPAN
    submission = _replace_fact(
        oracle,
        'dose-a',
        original.model_copy(update={'citations': (span,)}),
    )

    score = _score(case, submission)
    assert score.status == ScoreStatus.INVALID_SCHEMA
    assert expected in {issue.code for issue in score.issues}


def test_citation_cardinality_is_bounded_per_fact_and_per_submission() -> None:
    too_many = tuple(
        SourceSpan(source_id='source-001', start_byte=index, end_byte=index + 1)
        for index in range(MAX_CITATIONS_PER_FACT + 1)
    )
    with pytest.raises(ValidationError):
        FactAnswer(
            query_id='oversized-fact',
            status=FactAnswerStatus.OBSERVED,
            value=_number(1, 'microgram'),
            citations=too_many,
        )

    per_fact = too_many[:MAX_CITATIONS_PER_FACT]
    fact_count = MAX_TOTAL_CITATIONS // MAX_CITATIONS_PER_FACT + 1
    with pytest.raises(ValidationError, match=str(MAX_TOTAL_CITATIONS)):
        AgenticSubmissionV1(
            task_id='oversized-submission',
            workspace_manifest_sha256='0' * 64,
            fact_answers=tuple(
                FactAnswer(
                    query_id=f'fact-{index}',
                    status=FactAnswerStatus.OBSERVED,
                    value=_number(index, 'microgram'),
                    citations=per_fact,
                )
                for index in range(fact_count)
            ),
            decision=AgenticDecision(
                status=DecisionStatus.INSUFFICIENT_EVIDENCE,
                ranking=('candidate-001', 'candidate-002'),
                advancement_probabilities=(
                    CandidateProbability(candidate_id='candidate-001', probability=0.5),
                    CandidateProbability(candidate_id='candidate-002', probability=0.5),
                ),
            ),
        )


def test_half_of_an_atomic_gold_anchor_gets_no_support_credit(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    answer = oracle.fact_answers[0]
    anchor = answer.citations[0]
    half = SourceSpan(
        source_id=anchor.source_id,
        start_byte=anchor.start_byte,
        end_byte=anchor.start_byte + anchor.byte_count // 2,
    )
    submission = _replace_fact(oracle, answer.query_id, answer.model_copy(update={'citations': (half,)}))

    score = _score(case, submission)
    assert score.status == ScoreStatus.VALID
    assert score.citation_f1 == pytest.approx(0.5)
    assert score.citation_f1 < _score(case, oracle).citation_f1


def test_private_gold_support_groups_require_unique_feasible_anchors(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    span_a = case.oracle.fact_answers[0].citations[0]
    span_b = case.oracle.fact_answers[1].citations[0]

    with pytest.raises(ValidationError, match='unique'):
        GoldSupportGroup(
            group_id='duplicate-alternatives',
            evidence_group_id='evidence-a-first',
            alternatives=(span_a, span_a),
        )

    first = GoldSupportGroup(
        group_id='support-a-first',
        evidence_group_id='evidence-a-first',
        alternatives=(span_a, span_b),
    )
    second = GoldSupportGroup(
        group_id='support-a-second',
        evidence_group_id='evidence-a-second',
        alternatives=(span_a,),
    )
    conflict = GoldFactLabel(
        query_id='dose-a',
        status=FactAnswerStatus.CONFLICT,
        support_groups=(first, second),
    )
    evidence_groups = (
        GoldEvidenceGroup(group_id='evidence-a-first', acceptable_source_ids=('source-002',)),
        GoldEvidenceGroup(group_id='evidence-a-second', acceptable_source_ids=('source-002',)),
        next(group for group in case.gold.evidence_groups if group.group_id == 'evidence-dose-b'),
    )
    feasible_gold = case.gold.model_copy(
        update={
            'fact_labels': (conflict, case.gold.fact_labels[1]),
            'evidence_groups': evidence_groups,
        }
    )
    feasible_evaluator = _evaluator_with_gold(case, feasible_gold)
    feasible_oracle = oracle_submission(AgenticScoringContract.from_workspace(case.workspace), feasible_gold)
    feasible_score = feasible_evaluator.score(feasible_oracle)
    assert feasible_score.status == ScoreStatus.VALID
    assert feasible_score.reward == 1.0

    infeasible_conflict = conflict.model_copy(
        update={
            'support_groups': (
                first.model_copy(update={'alternatives': (span_a,)}),
                second,
            )
        }
    )
    infeasible_gold = feasible_gold.model_copy(update={'fact_labels': (infeasible_conflict, case.gold.fact_labels[1])})
    with pytest.raises(ValueError, match='unique oracle citations'):
        _evaluator_with_gold(case, infeasible_gold)


def test_unknown_source_is_a_hard_leakage_failure(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    answer = oracle.fact_answers[0].model_copy(
        update={'citations': (SourceSpan(source_id='future-results', start_byte=0, end_byte=10),)}
    )

    score = _score(case, _replace_fact(oracle, 'dose-a', answer))
    assert score.status == ScoreStatus.INVALID_LEAKAGE
    assert AgenticIssueCode.LEAK_UNKNOWN_SOURCE in {issue.code for issue in score.issues}


def test_duplicate_and_irrelevant_spans_cannot_inflate_reward(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    oracle_score = _score(case, oracle)
    answer = oracle.fact_answers[0]

    duplicated = _replace_fact(
        oracle,
        'dose-a',
        answer.model_copy(update={'citations': (answer.citations[0], answer.citations[0])}),
    )
    duplicate_score = _score(case, duplicated)
    assert duplicate_score.status == ScoreStatus.INVALID_SCHEMA
    assert AgenticIssueCode.DUPLICATE_SPAN in {issue.code for issue in duplicate_score.issues}

    irrelevant_span = SourceSpan(source_id='source-001', start_byte=0, end_byte=len(case.distractor) - 1)
    stuffed = _replace_fact(
        oracle,
        'dose-a',
        answer.model_copy(update={'citations': (*answer.citations, irrelevant_span)}),
    )
    stuffed_score = _score(case, stuffed)
    assert stuffed_score.status == ScoreStatus.VALID
    assert stuffed_score.reward < oracle_score.reward
    assert stuffed_score.retrieval_precision < oracle_score.retrieval_precision
    assert stuffed_score.citation_precision < oracle_score.citation_precision


def test_wrong_assertion_scores_below_honest_abstention(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    oracle = case.oracle
    answer = oracle.fact_answers[0]
    wrong = _replace_fact(
        oracle,
        'dose-a',
        answer.model_copy(update={'value': _number(999, 'microgram')}),
    )
    abstained = _replace_fact(
        oracle,
        'dose-a',
        FactAnswer(query_id='dose-a', status=FactAnswerStatus.NOT_FOUND),
    )

    wrong_score = _score(case, wrong)
    abstained_score = _score(case, abstained)
    assert wrong_score.status == ScoreStatus.VALID
    assert abstained_score.status == ScoreStatus.VALID
    assert wrong_score.extraction_signed_utility < abstained_score.extraction_signed_utility
    assert wrong_score.extraction_score < abstained_score.extraction_score
    assert wrong_score.reward < abstained_score.reward

    all_wrong = oracle.model_copy(
        update={
            'fact_answers': tuple(
                answer.model_copy(update={'value': _number(999, 'microgram')}) for answer in oracle.fact_answers
            )
        }
    )
    all_abstained = oracle.model_copy(
        update={
            'fact_answers': tuple(
                FactAnswer(query_id=answer.query_id, status=FactAnswerStatus.NOT_FOUND)
                for answer in oracle.fact_answers
            ),
            'derived_metrics': (
                oracle.derived_metrics[0].model_copy(
                    update={'status': MetricAnswerStatus.NOT_COMPUTABLE, 'value': None}
                ),
            ),
        }
    )
    all_wrong_score = _score(case, all_wrong)
    all_abstained_score = _score(case, all_abstained)
    assert all_wrong_score.extraction_signed_utility == -1.0
    assert all_wrong_score.extraction_score == 0.0
    assert all_abstained_score.extraction_signed_utility == 0.0
    assert all_abstained_score.extraction_score == 0.5


def test_scoring_requires_trusted_admission_commitment(tmp_path: Path) -> None:
    case = _build_case(tmp_path)

    with pytest.raises(AgenticAdmissionError, match='trusted release commitment'):
        score_agentic_submission(
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256='0' * 64,
            gold=case.gold,
            gold_commitment_key=case.gold_commitment_key,
            expected_gold_commitment_sha256=case.gold_commitment_sha256,
            submission=case.oracle,
        )


def test_mutated_private_gold_fails_its_expected_commitment_before_scoring(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    first = case.gold.fact_labels[0].model_copy(update={'accepted_values': (_number(999, 'microgram'),)})
    mutated = case.gold.model_copy(update={'fact_labels': (first, *case.gold.fact_labels[1:])})

    with pytest.raises(ValueError, match='trusted commitment'):
        score_agentic_submission(
            workspace=case.workspace,
            admission=case.admission,
            expected_admission_sha256=case.admission_sha256,
            gold=mutated,
            gold_commitment_key=case.gold_commitment_key,
            expected_gold_commitment_sha256=case.gold_commitment_sha256,
            submission=case.oracle,
        )


def test_private_gold_binds_reward_version_and_exact_scoring_contract(tmp_path: Path) -> None:
    case = _build_case(tmp_path)

    wrong_reward = case.gold.model_copy(update={'reward_version': 'vaxreplay.agentic-reward.future'})
    with pytest.raises(ValueError, match='reward_version'):
        _evaluator_with_gold(case, wrong_reward)

    wrong_contract = case.gold.model_copy(update={'scoring_contract_sha256': '0' * 64})
    with pytest.raises(ValueError, match='exact scoring contract'):
        _evaluator_with_gold(case, wrong_contract)
