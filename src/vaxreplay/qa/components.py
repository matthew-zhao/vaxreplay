"""Non-collapsible component gates for VaxReplay training admission."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from vaxreplay.case_schema import ScoreStatus
from vaxreplay.qa.score_integrity import AnyScoreVector


@dataclass(frozen=True, slots=True)
class ComponentFloor:
    metric: str
    minimum: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError('component-floor metric must be non-empty')
        if not math.isfinite(self.minimum) or not 0.0 <= self.minimum <= 1.0:
            raise ValueError('component-floor minimum must be finite and between zero and one')


@dataclass(frozen=True, slots=True)
class ComponentFloorResult:
    metric: str
    minimum: float
    observed: float | None
    passed: bool
    reason: str


def normalized_component_floors(
    floors: Iterable[ComponentFloor] | Mapping[str, float],
) -> tuple[ComponentFloor, ...]:
    if isinstance(floors, Mapping):
        mapping = cast(Mapping[str, float], floors)
        values = tuple(ComponentFloor(metric=metric, minimum=minimum) for metric, minimum in mapping.items())
    else:
        values = tuple(floors)
    names = [floor.metric for floor in values]
    if len(names) != len(set(names)):
        raise ValueError('component-floor metrics must be unique')
    return tuple(sorted(values, key=lambda floor: floor.metric))


def audit_component_floors(
    score: AnyScoreVector,
    floors: Iterable[ComponentFloor] | Mapping[str, float],
) -> tuple[ComponentFloorResult, ...]:
    """Apply each scientific floor as a veto, never as a weighted bonus."""

    normalized = normalized_component_floors(floors)
    if score.status != ScoreStatus.VALID:
        return tuple(
            ComponentFloorResult(
                metric=floor.metric,
                minimum=floor.minimum,
                observed=None,
                passed=False,
                reason=f'score status is {score.status.value}, not valid',
            )
            for floor in normalized
        )

    metrics = score.metrics()
    results: list[ComponentFloorResult] = []
    for floor in normalized:
        observed = metrics.get(floor.metric)
        if observed is None:
            results.append(
                ComponentFloorResult(
                    metric=floor.metric,
                    minimum=floor.minimum,
                    observed=None,
                    passed=False,
                    reason='required component is absent from the score vector',
                )
            )
        else:
            passed = observed >= floor.minimum
            results.append(
                ComponentFloorResult(
                    metric=floor.metric,
                    minimum=floor.minimum,
                    observed=observed,
                    passed=passed,
                    reason=(
                        'component floor satisfied'
                        if passed
                        else f'{observed:.17g} is below required minimum {floor.minimum:.17g}'
                    ),
                )
            )
    return tuple(results)


def require_component_floors(
    score: AnyScoreVector,
    floors: Iterable[ComponentFloor] | Mapping[str, float],
) -> None:
    failures = tuple(result for result in audit_component_floors(score, floors) if not result.passed)
    if failures:
        detail = '; '.join(f'{failure.metric}: {failure.reason}' for failure in failures)
        raise ValueError(f'training reward failed non-collapsible component floors: {detail}')


def zero_grounding_reward_ceiling(reward_version: str) -> float:
    """Maximum published aggregate reward when grounding_reward is fixed to zero."""

    if reward_version not in {'v0.1', 'v1.0'}:
        raise ValueError(f'unsupported reward version {reward_version!r}')
    return 0.80
