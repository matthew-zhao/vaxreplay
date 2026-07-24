"""Fail-closed clock synchronization health gate for signed issuance.

The provider is deliberately deployment-neutral: a chrony sidecar, PTP monitor,
cloud time service, or hardware appliance can produce the observation.  Service
code validates the observation independently and does not accept a boolean alone.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

CLOCK_HEALTH_OBSERVATION_SCHEMA_VERSION = 'vaxreplay.clock-health-observation.v0.1'
CLOCK_HEALTH_POLICY_SCHEMA_VERSION = 'vaxreplay.clock-health-policy.v0.1'


class ClockHealthError(ValueError):
    """Clock health was unavailable, stale, unsynchronized, or out of policy."""


class ClockHealthObservation(StrictModel):
    """Exact output of a trusted local synchronization-health collector."""

    schema_version: Literal['vaxreplay.clock-health-observation.v0.1'] = CLOCK_HEALTH_OBSERVATION_SCHEMA_VERSION
    provider_id: str = Field(pattern=SAFE_ID_PATTERN)
    checked_at: datetime
    synchronized: bool
    leap_status: Literal['normal', 'insert_pending', 'delete_pending', 'unsynchronized']
    source_count: int = Field(ge=0, le=1024)
    absolute_offset_milliseconds: float = Field(ge=0, le=86_400_000)
    root_distance_milliseconds: float = Field(ge=0, le=86_400_000)
    sample_age_milliseconds: int = Field(ge=0, le=86_400_000)

    @field_validator('checked_at')
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'clock-health checked_at')

    @field_validator('absolute_offset_milliseconds', 'root_distance_milliseconds')
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError('clock-health measurements must be finite')
        return value

    @model_validator(mode='after')
    def validate_status(self) -> 'ClockHealthObservation':
        if self.synchronized == (self.leap_status == 'unsynchronized'):
            raise ValueError('clock synchronization flag conflicts with leap status')
        return self


class ClockHealthPolicy(StrictModel):
    """Pinned thresholds applied identically by every issuing service."""

    schema_version: Literal['vaxreplay.clock-health-policy.v0.1'] = CLOCK_HEALTH_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    provider_id: str = Field(pattern=SAFE_ID_PATTERN)
    max_observation_age_seconds: int = Field(ge=1, le=3600)
    max_future_skew_seconds: int = Field(default=5, ge=0, le=300)
    max_absolute_offset_milliseconds: float = Field(gt=0, le=60_000)
    max_root_distance_milliseconds: float = Field(gt=0, le=60_000)
    max_sample_age_milliseconds: int = Field(gt=0, le=3_600_000)
    minimum_source_count: int = Field(default=1, ge=1, le=1024)
    allowed_leap_statuses: tuple[Literal['normal', 'insert_pending', 'delete_pending'], ...] = ('normal',)

    @field_validator('allowed_leap_statuses')
    @classmethod
    def validate_statuses(
        cls,
        value: tuple[Literal['normal', 'insert_pending', 'delete_pending'], ...],
    ) -> tuple[Literal['normal', 'insert_pending', 'delete_pending'], ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError('allowed leap statuses must be nonempty and unique')
        return value


@runtime_checkable
class ClockHealthProvider(Protocol):
    def observe(self) -> ClockHealthObservation:
        """Return a current health observation without accepting caller time."""


class CallbackClockHealthProvider:
    """Adapter for a local chrony/PTP/cloud-time RPC collector."""

    def __init__(self, operation: Callable[[], ClockHealthObservation]) -> None:
        if not callable(operation):
            raise ClockHealthError('clock-health operation must be callable')
        self._operation = operation

    def observe(self) -> ClockHealthObservation:
        failed = False
        try:
            observation = self._operation()
        except BaseException:
            failed = True
            observation = None
        if failed:
            raise ClockHealthError('clock-health provider failed') from None
        invalid = False
        try:
            parsed = ClockHealthObservation.model_validate(observation)
        except (TypeError, ValueError):
            invalid = True
            parsed = None
        if invalid or parsed is None:
            raise ClockHealthError('clock-health provider returned an invalid observation') from None
        return parsed


class ClockHealthGate:
    """Validate synchronization immediately before a security timestamp is used."""

    def __init__(self, *, policy: ClockHealthPolicy, provider: ClockHealthProvider) -> None:
        self.policy = ClockHealthPolicy.model_validate(policy)
        if not isinstance(provider, ClockHealthProvider):
            raise ClockHealthError('clock-health provider does not implement the required interface')
        self.provider = provider

    def require_synchronized(self, *, security_time: datetime) -> ClockHealthObservation:
        now = aware_utc(security_time, 'security time')
        provider_failed = False
        try:
            raw_observation = self.provider.observe()
        except BaseException:
            provider_failed = True
            raw_observation = None
        if provider_failed:
            raise ClockHealthError('clock-health provider failed') from None
        invalid = False
        try:
            observation = ClockHealthObservation.model_validate(raw_observation)
        except (TypeError, ValueError):
            invalid = True
            observation = None
        if invalid or observation is None:
            raise ClockHealthError('clock-health provider returned an invalid observation') from None
        policy = self.policy
        if observation.provider_id != policy.provider_id:
            raise ClockHealthError('clock-health observation came from an untrusted provider')
        if observation.checked_at > now + timedelta(seconds=policy.max_future_skew_seconds):
            raise ClockHealthError('clock-health observation is implausibly in the future')
        if now - observation.checked_at > timedelta(seconds=policy.max_observation_age_seconds):
            raise ClockHealthError('clock-health observation is stale')
        if not observation.synchronized or observation.leap_status not in policy.allowed_leap_statuses:
            raise ClockHealthError('clock is not in an allowed synchronization state')
        if observation.source_count < policy.minimum_source_count:
            raise ClockHealthError('clock has too few synchronization sources')
        if observation.absolute_offset_milliseconds > policy.max_absolute_offset_milliseconds:
            raise ClockHealthError('clock offset exceeds policy')
        if observation.root_distance_milliseconds > policy.max_root_distance_milliseconds:
            raise ClockHealthError('clock root distance exceeds policy')
        if observation.sample_age_milliseconds > policy.max_sample_age_milliseconds:
            raise ClockHealthError('clock synchronization sample is too old')
        return observation


def require_clock_health(gate: ClockHealthGate | None, *, security_time: datetime) -> None:
    """Invoke a configured gate; ``None`` is the explicit development-only mode."""

    if gate is not None:
        gate.require_synchronized(security_time=security_time)


__all__ = [
    'CLOCK_HEALTH_OBSERVATION_SCHEMA_VERSION',
    'CLOCK_HEALTH_POLICY_SCHEMA_VERSION',
    'CallbackClockHealthProvider',
    'ClockHealthError',
    'ClockHealthGate',
    'ClockHealthObservation',
    'ClockHealthPolicy',
    'ClockHealthProvider',
    'require_clock_health',
]
