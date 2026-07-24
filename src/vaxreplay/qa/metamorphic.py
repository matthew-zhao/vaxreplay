"""Deterministic metamorphic checks for scored VaxReplay submissions.

The utilities in this module compare already-produced responses and scores.  They
do not execute models, rewrite episode bundles, or decide training admission.
Keeping those responsibilities separate makes the expected metamorphic relation
explicit and lets an admission layer consume the resulting structured findings.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from vaxreplay.case_schema import Submission


class MetricProvider(Protocol):
    """Score-vector interface shared by V0 and V1 score models."""

    def metrics(self) -> dict[str, float]: ...


type ScoreSource = MetricProvider | Mapping[str, float]


class MetamorphicRelation(str, enum.Enum):
    CANDIDATE_EQUIVARIANCE = 'candidate_equivariance'
    NUISANCE_INVARIANCE = 'nuisance_invariance'
    EVIDENCE_EXPECTED_DIRECTION = 'evidence_expected_direction'


class ExpectedDirection(str, enum.Enum):
    """Direction of an intervention metric in raw metric coordinates."""

    INCREASE = 'increase'
    DECREASE = 'decrease'
    UNCHANGED = 'unchanged'


class ResponseExpectation(str, enum.Enum):
    CHANGE = 'change'
    UNCHANGED = 'unchanged'
    IGNORE = 'ignore'


class DecisionTargetKind(str, enum.Enum):
    """A response field whose causal sensitivity is checked explicitly."""

    RANKING_POSITION = 'ranking_position'
    FORECAST_PROBABILITY = 'forecast_probability'
    ASSESSMENT_CONCLUSION = 'assessment_conclusion'


@dataclass(frozen=True, slots=True)
class DecisionTarget:
    """One exact candidate-level decision path affected by an intervention."""

    kind: DecisionTargetKind
    candidate_id: str
    target_id: str | None = None
    horizon_days: int | None = None
    dimension: str | None = None
    minimum_change: float = 0.0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError('decision target candidate_id must be non-empty')
        if not math.isfinite(self.minimum_change) or self.minimum_change < 0.0:
            raise ValueError('decision target minimum_change must be finite and non-negative')
        if self.kind == DecisionTargetKind.RANKING_POSITION:
            if self.target_id is not None or self.horizon_days is not None or self.dimension is not None:
                raise ValueError('ranking-position targets cannot specify forecast or assessment fields')
            if self.minimum_change != 0.0:
                raise ValueError('ranking-position targets cannot specify minimum_change')
        elif self.kind == DecisionTargetKind.FORECAST_PROBABILITY:
            if not self.target_id or self.horizon_days is None or self.horizon_days <= 0:
                raise ValueError('forecast-probability targets require target_id and positive horizon_days')
            if self.dimension is not None:
                raise ValueError('forecast-probability targets cannot specify dimension')
        else:
            if not self.dimension:
                raise ValueError('assessment-conclusion targets require dimension')
            if self.target_id is not None or self.horizon_days is not None:
                raise ValueError('assessment-conclusion targets cannot specify forecast fields')
            if self.minimum_change != 0.0:
                raise ValueError('assessment-conclusion targets cannot specify minimum_change')

    @property
    def subject(self) -> str:
        if self.kind == DecisionTargetKind.RANKING_POSITION:
            return f'response.ranking[{self.candidate_id!r}].position'
        if self.kind == DecisionTargetKind.FORECAST_PROBABILITY:
            return f'response.forecasts[{self.candidate_id!r},{self.target_id!r},{self.horizon_days!r}].probability'
        return f'response.assessments[{self.candidate_id!r},{self.dimension!r}].conclusion'


@dataclass(frozen=True, slots=True)
class MetricExpectation:
    metric: str
    direction: ExpectedDirection
    minimum_change: float = 0.0
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError('metric must be non-empty')
        if not math.isfinite(self.minimum_change) or self.minimum_change < 0.0:
            raise ValueError('minimum_change must be finite and non-negative')
        if self.direction == ExpectedDirection.UNCHANGED and self.minimum_change != 0.0:
            raise ValueError('unchanged expectations cannot require minimum_change')
        _require_tolerance(self.tolerance)


@dataclass(frozen=True, slots=True)
class MetamorphicFinding:
    """One replayable comparison result.

    Fingerprints bind non-numeric response projections without embedding the
    response itself. Numeric metric comparisons additionally retain both values
    and the signed ``variant - reference`` delta.
    """

    audit_id: str
    relation: MetamorphicRelation
    subject: str
    passed: bool
    expected: str
    observed: str
    reference_fingerprint: str | None = None
    variant_fingerprint: str | None = None
    reference_value: float | None = None
    variant_value: float | None = None
    delta: float | None = None
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.audit_id:
            raise ValueError('audit_id must be non-empty')
        if not self.subject:
            raise ValueError('subject must be non-empty')
        _require_tolerance(self.tolerance)

    @property
    def finding_id(self) -> str:
        identity = {
            'audit_id': self.audit_id,
            'relation': self.relation.value,
            'subject': self.subject,
        }
        return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['relation'] = self.relation.value
        value['finding_id'] = self.finding_id
        return value


def audit_candidate_equivariance(
    reference: Submission,
    variant: Submission,
    *,
    variant_to_reference: Mapping[str, str] | None = None,
    reference_score: ScoreSource | None = None,
    variant_score: ScoreSource | None = None,
    metrics: Iterable[str] | None = None,
    tolerance: float = 1e-12,
    audit_id: str = 'candidate-equivariance',
) -> tuple[MetamorphicFinding, ...]:
    """Check response and score equivariance under candidate aliasing/permutation.

    ``variant_to_reference`` maps every candidate ID in ``variant`` back to its
    corresponding candidate ID in ``reference``.  Passing ``None`` selects the
    identity map, which is the appropriate check when only candidate presentation
    order was permuted.

    Episode IDs and manifest hashes are intentionally ignored: a transformed
    episode is expected to have different identity commitments.
    """

    return _audit_equivalence(
        reference,
        variant,
        relation=MetamorphicRelation.CANDIDATE_EQUIVARIANCE,
        variant_to_reference=variant_to_reference,
        reference_score=reference_score,
        variant_score=variant_score,
        metrics=metrics,
        tolerance=tolerance,
        audit_id=audit_id,
    )


def audit_nuisance_invariance(
    reference: Submission,
    variant: Submission,
    *,
    reference_score: ScoreSource | None = None,
    variant_score: ScoreSource | None = None,
    metrics: Iterable[str] | None = None,
    tolerance: float = 1e-12,
    audit_id: str = 'nuisance-invariance',
) -> tuple[MetamorphicFinding, ...]:
    """Check that a scientifically irrelevant transformation changes nothing."""

    return _audit_equivalence(
        reference,
        variant,
        relation=MetamorphicRelation.NUISANCE_INVARIANCE,
        variant_to_reference=None,
        reference_score=reference_score,
        variant_score=variant_score,
        metrics=metrics,
        tolerance=tolerance,
        audit_id=audit_id,
    )


def audit_evidence_intervention(
    reference: Submission,
    intervention: Submission,
    *,
    reference_score: ScoreSource,
    intervention_score: ScoreSource,
    expectations: Iterable[MetricExpectation],
    response_expectation: ResponseExpectation = ResponseExpectation.CHANGE,
    decision_targets: Iterable[DecisionTarget] | None = None,
    audit_id: str = 'evidence-intervention',
    response_tolerance: float = 1e-12,
) -> tuple[MetamorphicFinding, ...]:
    """Check response sensitivity and expected score directions after evidence changes.

    Positive response sensitivity must name exact ``decision_targets``. This
    prevents an unrelated forecast or assessment perturbation from satisfying a
    causal intervention check. Citations are not targetable, so merely deleting a
    citation cannot satisfy an expected decision change.

    Omitting ``decision_targets`` is retained only for whole-response
    ``UNCHANGED`` checks. A ``CHANGE`` check without an explicit target fails
    closed with ``ValueError``.
    """

    _require_tolerance(response_tolerance)
    metric_expectations = tuple(expectations)
    metric_names = [expectation.metric for expectation in metric_expectations]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError('metric expectations must be unique')

    targets = tuple(decision_targets or ())
    target_subjects = tuple(target.subject for target in targets)
    if len(target_subjects) != len(set(target_subjects)):
        raise ValueError('decision targets must be unique')
    if response_expectation == ResponseExpectation.CHANGE and not targets:
        raise ValueError('change expectations require at least one explicit decision target')

    findings: list[MetamorphicFinding] = []
    if response_expectation != ResponseExpectation.IGNORE:
        expected_equal = response_expectation == ResponseExpectation.UNCHANGED
        if targets:
            for target in targets:
                findings.append(
                    _audit_decision_target(
                        reference,
                        intervention,
                        target=target,
                        expected_equal=expected_equal,
                        audit_id=audit_id,
                        tolerance=response_tolerance,
                    )
                )
        else:
            reference_projection = _canonical_response(reference, {}, include_citations=False)
            intervention_projection = _canonical_response(intervention, {}, include_citations=False)
            equivalent = _response_projections_equal(
                reference_projection,
                intervention_projection,
                tolerance=response_tolerance,
            )
            findings.append(
                _response_finding(
                    audit_id=audit_id,
                    relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
                    subject='response.decision',
                    reference=reference_projection,
                    variant=intervention_projection,
                    passed=equivalent,
                    expected='unchanged',
                    observed='unchanged' if equivalent else 'changed',
                    tolerance=response_tolerance,
                )
            )

    reference_metrics = _score_metrics(reference_score)
    intervention_metrics = _score_metrics(intervention_score)
    for expectation in metric_expectations:
        metric = expectation.metric
        if metric not in reference_metrics or metric not in intervention_metrics:
            missing = [
                name
                for name, values in (
                    ('reference', reference_metrics),
                    ('intervention', intervention_metrics),
                )
                if metric not in values
            ]
            findings.append(
                MetamorphicFinding(
                    audit_id=audit_id,
                    relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
                    subject=f'score.{metric}',
                    passed=False,
                    expected=_expectation_description(expectation),
                    observed=f'metric missing from {" and ".join(missing)} score',
                    tolerance=expectation.tolerance,
                )
            )
            continue

        reference_value = reference_metrics[metric]
        intervention_value = intervention_metrics[metric]
        delta = intervention_value - reference_value
        passed = _direction_satisfied(delta, expectation)
        findings.append(
            MetamorphicFinding(
                audit_id=audit_id,
                relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
                subject=f'score.{metric}',
                passed=passed,
                expected=_expectation_description(expectation),
                observed=f'delta={delta:.17g}',
                reference_value=reference_value,
                variant_value=intervention_value,
                delta=delta,
                tolerance=expectation.tolerance,
            )
        )

    return tuple(findings)


def failed_findings(findings: Iterable[MetamorphicFinding]) -> tuple[MetamorphicFinding, ...]:
    """Return failures in input order for admission or reporting layers."""

    return tuple(finding for finding in findings if not finding.passed)


def _audit_equivalence(
    reference: Submission,
    variant: Submission,
    *,
    relation: MetamorphicRelation,
    variant_to_reference: Mapping[str, str] | None,
    reference_score: ScoreSource | None,
    variant_score: ScoreSource | None,
    metrics: Iterable[str] | None,
    tolerance: float,
    audit_id: str,
) -> tuple[MetamorphicFinding, ...]:
    _require_tolerance(tolerance)
    if (reference_score is None) != (variant_score is None):
        raise ValueError('reference_score and variant_score must be provided together')

    aliases = _validated_candidate_map(reference, variant, variant_to_reference)
    reference_projection = _canonical_response(reference, {}, include_citations=True)
    variant_projection = _canonical_response(variant, aliases, include_citations=True)

    findings: list[MetamorphicFinding] = []
    for subject in ('ranking', 'forecasts', 'assessments'):
        reference_value = reference_projection[subject]
        variant_value = variant_projection[subject]
        equivalent = _projection_values_equal(reference_value, variant_value, tolerance=tolerance)
        findings.append(
            _response_finding(
                audit_id=audit_id,
                relation=relation,
                subject=f'response.{subject}',
                reference=reference_value,
                variant=variant_value,
                passed=equivalent,
                expected='equal after candidate mapping',
                observed='equal' if equivalent else 'different',
                tolerance=tolerance,
            )
        )

    if reference_score is not None and variant_score is not None:
        findings.extend(
            _score_invariance_findings(
                reference_score,
                variant_score,
                relation=relation,
                metrics=metrics,
                tolerance=tolerance,
                audit_id=audit_id,
            )
        )
    elif metrics is not None:
        raise ValueError('metrics cannot be selected without score inputs')
    return tuple(findings)


def _score_invariance_findings(
    reference_score: ScoreSource,
    variant_score: ScoreSource,
    *,
    relation: MetamorphicRelation,
    metrics: Iterable[str] | None,
    tolerance: float,
    audit_id: str,
) -> tuple[MetamorphicFinding, ...]:
    reference_metrics = _score_metrics(reference_score)
    variant_metrics = _score_metrics(variant_score)
    metric_names = (
        tuple(metrics) if metrics is not None else tuple(sorted(set(reference_metrics) | set(variant_metrics)))
    )
    if len(metric_names) != len(set(metric_names)):
        raise ValueError('metrics must be unique')
    if any(not metric for metric in metric_names):
        raise ValueError('metric names must be non-empty')

    findings: list[MetamorphicFinding] = []
    for metric in metric_names:
        if metric not in reference_metrics or metric not in variant_metrics:
            missing = [
                name
                for name, values in (('reference', reference_metrics), ('variant', variant_metrics))
                if metric not in values
            ]
            findings.append(
                MetamorphicFinding(
                    audit_id=audit_id,
                    relation=relation,
                    subject=f'score.{metric}',
                    passed=False,
                    expected='equal',
                    observed=f'metric missing from {" and ".join(missing)} score',
                    tolerance=tolerance,
                )
            )
            continue
        reference_value = reference_metrics[metric]
        variant_value = variant_metrics[metric]
        delta = variant_value - reference_value
        passed = abs(delta) <= tolerance
        findings.append(
            MetamorphicFinding(
                audit_id=audit_id,
                relation=relation,
                subject=f'score.{metric}',
                passed=passed,
                expected=f'absolute delta <= {tolerance:.17g}',
                observed=f'delta={delta:.17g}',
                reference_value=reference_value,
                variant_value=variant_value,
                delta=delta,
                tolerance=tolerance,
            )
        )
    return tuple(findings)


def _validated_candidate_map(
    reference: Submission,
    variant: Submission,
    variant_to_reference: Mapping[str, str] | None,
) -> dict[str, str]:
    reference_ids = _candidate_ids(reference)
    variant_ids = _candidate_ids(variant)
    aliases = (
        {candidate_id: candidate_id for candidate_id in variant_ids}
        if variant_to_reference is None
        else dict(variant_to_reference)
    )
    if set(aliases) != variant_ids:
        missing = sorted(variant_ids - set(aliases))
        extra = sorted(set(aliases) - variant_ids)
        raise ValueError(f'candidate map must cover variant IDs exactly; missing={missing}, extra={extra}')
    if set(aliases.values()) != reference_ids or len(set(aliases.values())) != len(aliases):
        raise ValueError('candidate map must be a bijection onto the reference candidate IDs')
    return aliases


def _candidate_ids(submission: Submission) -> set[str]:
    return {
        *submission.ranking,
        *(forecast.candidate_id for forecast in submission.forecasts),
        *(assessment.candidate_id for assessment in submission.assessments),
    }


def _canonical_response(
    submission: Submission,
    candidate_map: Mapping[str, str],
    *,
    include_citations: bool,
) -> dict[str, Any]:
    def candidate(candidate_id: str) -> str:
        return candidate_map.get(candidate_id, candidate_id)

    ranking = tuple(candidate(candidate_id) for candidate_id in submission.ranking)
    forecasts = tuple(
        sorted(
            (
                candidate(forecast.candidate_id),
                forecast.target_id,
                forecast.horizon_days,
                forecast.probability,
            )
            for forecast in submission.forecasts
        )
    )
    assessments: list[tuple[Any, ...]] = []
    for assessment in submission.assessments:
        core: tuple[Any, ...] = (
            candidate(assessment.candidate_id),
            assessment.dimension,
            assessment.conclusion.value,
        )
        if include_citations:
            citations = tuple(
                sorted(
                    (citation.evidence_id, citation.stance.value, citation.quote) for citation in assessment.citations
                )
            )
            core = (*core, citations)
        assessments.append(core)
    return {
        'ranking': ranking,
        'forecasts': forecasts,
        'assessments': tuple(sorted(assessments)),
    }


def _response_projections_equal(
    reference: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    return all(
        _projection_values_equal(reference[subject], variant[subject], tolerance=tolerance)
        for subject in ('ranking', 'forecasts', 'assessments')
    )


def _audit_decision_target(
    reference: Submission,
    intervention: Submission,
    *,
    target: DecisionTarget,
    expected_equal: bool,
    audit_id: str,
    tolerance: float,
) -> MetamorphicFinding:
    reference_value, reference_error = _decision_target_value(reference, target)
    intervention_value, intervention_error = _decision_target_value(intervention, target)
    if reference_error is not None or intervention_error is not None:
        errors = tuple(
            detail
            for detail in (
                f'reference {reference_error}' if reference_error is not None else None,
                f'intervention {intervention_error}' if intervention_error is not None else None,
            )
            if detail is not None
        )
        return MetamorphicFinding(
            audit_id=audit_id,
            relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
            subject=target.subject,
            passed=False,
            expected='unchanged' if expected_equal else _target_change_description(target, tolerance),
            observed='; '.join(errors),
            tolerance=tolerance,
        )

    if target.kind == DecisionTargetKind.FORECAST_PROBABILITY:
        assert isinstance(reference_value, float)
        assert isinstance(intervention_value, float)
        delta = intervention_value - reference_value
        equivalent = abs(delta) <= tolerance
        if expected_equal:
            passed = equivalent
        else:
            passed = not equivalent and abs(delta) + tolerance >= target.minimum_change
        return MetamorphicFinding(
            audit_id=audit_id,
            relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
            subject=target.subject,
            passed=passed,
            expected='unchanged' if expected_equal else _target_change_description(target, tolerance),
            observed=f'delta={delta:.17g}',
            reference_value=reference_value,
            variant_value=intervention_value,
            delta=delta,
            tolerance=tolerance,
        )

    equivalent = reference_value == intervention_value
    return _response_finding(
        audit_id=audit_id,
        relation=MetamorphicRelation.EVIDENCE_EXPECTED_DIRECTION,
        subject=target.subject,
        reference=reference_value,
        variant=intervention_value,
        passed=equivalent == expected_equal,
        expected='unchanged' if expected_equal else 'changed',
        observed='unchanged' if equivalent else 'changed',
        tolerance=tolerance,
    )


def _decision_target_value(
    submission: Submission,
    target: DecisionTarget,
) -> tuple[int | float | str | None, str | None]:
    if target.kind == DecisionTargetKind.RANKING_POSITION:
        positions = tuple(
            index for index, candidate_id in enumerate(submission.ranking) if candidate_id == target.candidate_id
        )
        if len(positions) != 1:
            return None, f'contains {len(positions)} matching ranking entries; expected exactly one'
        return positions[0], None

    if target.kind == DecisionTargetKind.FORECAST_PROBABILITY:
        values = tuple(
            forecast.probability
            for forecast in submission.forecasts
            if (
                forecast.candidate_id == target.candidate_id
                and forecast.target_id == target.target_id
                and forecast.horizon_days == target.horizon_days
            )
        )
        if len(values) != 1:
            return None, f'contains {len(values)} matching forecasts; expected exactly one'
        return values[0], None

    values = tuple(
        assessment.conclusion.value
        for assessment in submission.assessments
        if (assessment.candidate_id == target.candidate_id and assessment.dimension == target.dimension)
    )
    if len(values) != 1:
        return None, f'contains {len(values)} matching assessments; expected exactly one'
    return values[0], None


def _target_change_description(target: DecisionTarget, tolerance: float) -> str:
    if target.kind != DecisionTargetKind.FORECAST_PROBABILITY:
        return 'changed'
    return f'absolute change of at least {target.minimum_change:.17g} (tolerance {tolerance:.17g})'


def _projection_values_equal(reference: Any, variant: Any, *, tolerance: float) -> bool:
    if isinstance(reference, float) and isinstance(variant, float):
        return abs(reference - variant) <= tolerance
    if isinstance(reference, tuple) and isinstance(variant, tuple):
        return len(reference) == len(variant) and all(
            _projection_values_equal(left, right, tolerance=tolerance)
            for left, right in zip(reference, variant, strict=True)
        )
    return reference == variant


def _response_finding(
    *,
    audit_id: str,
    relation: MetamorphicRelation,
    subject: str,
    reference: Any,
    variant: Any,
    passed: bool,
    expected: str,
    observed: str,
    tolerance: float,
) -> MetamorphicFinding:
    return MetamorphicFinding(
        audit_id=audit_id,
        relation=relation,
        subject=subject,
        passed=passed,
        expected=expected,
        observed=observed,
        reference_fingerprint=_fingerprint(reference),
        variant_fingerprint=_fingerprint(variant),
        tolerance=tolerance,
    )


def _score_metrics(score: ScoreSource) -> dict[str, float]:
    raw = dict(score) if isinstance(score, Mapping) else score.metrics()
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError('score metric names must be non-empty strings')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'score metric {name!r} must be a finite number')
        metrics[name] = float(value)
    return metrics


def _direction_satisfied(delta: float, expectation: MetricExpectation) -> bool:
    if expectation.direction == ExpectedDirection.UNCHANGED:
        return abs(delta) <= expectation.tolerance
    required_change = max(expectation.minimum_change, expectation.tolerance)
    if expectation.direction == ExpectedDirection.INCREASE:
        return delta > expectation.tolerance and delta + expectation.tolerance >= required_change
    return delta < -expectation.tolerance and -delta + expectation.tolerance >= required_change


def _expectation_description(expectation: MetricExpectation) -> str:
    if expectation.direction == ExpectedDirection.UNCHANGED:
        return f'absolute delta <= {expectation.tolerance:.17g}'
    return (
        f'{expectation.direction.value} by at least {expectation.minimum_change:.17g} '
        f'(tolerance {expectation.tolerance:.17g})'
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')


def _require_tolerance(tolerance: float) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('tolerance must be finite and non-negative')
