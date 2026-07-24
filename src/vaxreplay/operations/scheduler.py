"""Deterministic schedule arithmetic for externally invoked capture workers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from vaxreplay.operations.schema import CaptureJobSpec, aware_utc


class ScheduleError(ValueError):
    """Raised when a requested schedule window cannot be enumerated safely."""


class ScheduleBacklogError(ScheduleError):
    """Raised instead of silently truncating an unexpectedly large due-slot backlog."""

    def __init__(self, slot_count: int, max_slots: int) -> None:
        self.slot_count = slot_count
        self.max_slots = max_slots
        super().__init__(f'schedule window contains {slot_count} slots, exceeding max_slots={max_slots}')


class ScheduleHistoryGapError(ScheduleError):
    """Raised when bounded catch-up would hide an older unregistered slot."""

    def __init__(self, missing_slot: datetime, catch_up_boundary: datetime) -> None:
        self.missing_slot = aware_utc(missing_slot, 'missing_slot')
        self.catch_up_boundary = aware_utc(catch_up_boundary, 'catch_up_boundary')
        super().__init__(
            'registered schedule history has a gap at '
            f'{self.missing_slot.isoformat()} before catch-up boundary '
            f'{self.catch_up_boundary.isoformat()}; explicitly register/backfill the missing '
            'history with vaxreplay-ops register-due before run-static-due'
        )


def is_scheduled_slot(spec: CaptureJobSpec, scheduled_for: datetime) -> bool:
    """Return whether ``scheduled_for`` is exactly anchor + N intervals for N >= 0."""

    scheduled_for = aware_utc(scheduled_for, 'scheduled_for')
    anchor = spec.schedule_anchor_at
    if scheduled_for < anchor:
        return False
    delta = scheduled_for - anchor
    return _timedelta_microseconds(delta) % (spec.schedule_interval_seconds * 1_000_000) == 0


def enumerate_scheduled_slots(
    spec: CaptureJobSpec,
    *,
    window_start: datetime,
    window_end: datetime,
    max_slots: int = 1_000,
) -> tuple[datetime, ...]:
    """Enumerate every slot in one inclusive UTC window, failing on truncation.

    ``window_start`` is the explicit catch-up boundary.  Callers choose it from their
    committed operations policy; this function never silently drops an older slot.
    """

    if max_slots < 1:
        raise ScheduleError('max_slots must be at least one')
    start = aware_utc(window_start, 'window_start')
    end = aware_utc(window_end, 'window_end')
    if end < start:
        raise ScheduleError('window_end must be at or after window_start')

    anchor = spec.schedule_anchor_at
    if end < anchor:
        return ()
    interval_us = spec.schedule_interval_seconds * 1_000_000
    start_delta_us = _timedelta_microseconds(max(start, anchor) - anchor)
    first_ordinal = (start_delta_us + interval_us - 1) // interval_us
    end_ordinal = _timedelta_microseconds(end - anchor) // interval_us
    slot_count = max(0, end_ordinal - first_ordinal + 1)
    if slot_count > max_slots:
        raise ScheduleBacklogError(slot_count, max_slots)
    interval = timedelta(seconds=spec.schedule_interval_seconds)
    return tuple(anchor + ordinal * interval for ordinal in range(first_ordinal, end_ordinal + 1))


def latest_scheduled_slot(spec: CaptureJobSpec, at: datetime) -> datetime | None:
    """Return the latest scheduled slot at or before ``at``, or ``None`` pre-anchor."""

    at = aware_utc(at, 'at')
    anchor = spec.schedule_anchor_at
    if at < anchor:
        return None
    interval_us = spec.schedule_interval_seconds * 1_000_000
    ordinal = _timedelta_microseconds(at - anchor) // interval_us
    return anchor + ordinal * timedelta(seconds=spec.schedule_interval_seconds)


def first_unregistered_slot_before(
    spec: CaptureJobSpec,
    registered_slots: Iterable[datetime],
    *,
    before: datetime,
) -> datetime | None:
    """Return the first missing schedule slot strictly before ``before``.

    The check is linear in the supplied registered rows, not in the age of the schedule.
    This lets a worker fail closed on a very old missing prefix without first enumerating
    an attacker-sized backlog.
    """

    boundary = aware_utc(before, 'before')
    anchor = spec.schedule_anchor_at
    if boundary <= anchor:
        return None

    interval = timedelta(seconds=spec.schedule_interval_seconds)
    interval_us = spec.schedule_interval_seconds * 1_000_000
    boundary_delta_us = _timedelta_microseconds(boundary - anchor)
    required_slot_count = (boundary_delta_us + interval_us - 1) // interval_us

    ordinals: list[int] = []
    for value in registered_slots:
        scheduled_for = aware_utc(value, 'registered slot')
        if not is_scheduled_slot(spec, scheduled_for):
            raise ScheduleError('registered slot is not on the immutable job schedule')
        if scheduled_for >= boundary:
            continue
        ordinal = _timedelta_microseconds(scheduled_for - anchor) // interval_us
        ordinals.append(ordinal)

    ordinals.sort()
    expected = 0
    previous: int | None = None
    for ordinal in ordinals:
        if ordinal == previous:
            raise ScheduleError('registered slot inventory contains a duplicate schedule slot')
        previous = ordinal
        if ordinal != expected:
            return anchor + expected * interval
        expected += 1

    if expected < required_slot_count:
        return anchor + expected * interval
    return None


def require_complete_registered_history(
    spec: CaptureJobSpec,
    registered_slots: Iterable[datetime],
    *,
    before: datetime,
) -> None:
    """Fail if any schedule slot strictly before ``before`` is unregistered."""

    boundary = aware_utc(before, 'before')
    missing = first_unregistered_slot_before(spec, registered_slots, before=boundary)
    if missing is not None:
        raise ScheduleHistoryGapError(missing, boundary)


def _timedelta_microseconds(value: timedelta) -> int:
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds
