from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from vaxreplay.operations.scheduler import (
    ScheduleBacklogError,
    ScheduleError,
    ScheduleHistoryGapError,
    enumerate_scheduled_slots,
    first_unregistered_slot_before,
    is_scheduled_slot,
    latest_scheduled_slot,
    require_complete_registered_history,
)
from vaxreplay.operations.schema import CaptureJobSpec

_ANCHOR = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _job() -> CaptureJobSpec:
    return CaptureJobSpec(
        job_id='official-page-hourly',
        collector_id='https-exact-v0.1',
        schedule_anchor_at=_ANCHOR,
        schedule_interval_seconds=3600,
    )


class OperationsSchedulerTest(unittest.TestCase):
    def test_enumerates_inclusive_slots_without_rounding_now(self) -> None:
        slots = enumerate_scheduled_slots(
            _job(),
            window_start=_ANCHOR + timedelta(minutes=1),
            window_end=_ANCHOR + timedelta(hours=3, minutes=59),
        )
        self.assertEqual(
            slots,
            (
                _ANCHOR + timedelta(hours=1),
                _ANCHOR + timedelta(hours=2),
                _ANCHOR + timedelta(hours=3),
            ),
        )

    def test_normalizes_offsets_and_handles_pre_anchor_window(self) -> None:
        offset = timezone(timedelta(hours=-7))
        self.assertEqual(
            enumerate_scheduled_slots(
                _job(),
                window_start=(_ANCHOR - timedelta(hours=2)).astimezone(offset),
                window_end=(_ANCHOR + timedelta(hours=1)).astimezone(offset),
            ),
            (_ANCHOR, _ANCHOR + timedelta(hours=1)),
        )
        self.assertEqual(
            enumerate_scheduled_slots(
                _job(),
                window_start=_ANCHOR - timedelta(days=1),
                window_end=_ANCHOR - timedelta(seconds=1),
            ),
            (),
        )

    def test_refuses_to_silently_truncate_backlog(self) -> None:
        with self.assertRaises(ScheduleBacklogError) as raised:
            enumerate_scheduled_slots(
                _job(),
                window_start=_ANCHOR,
                window_end=_ANCHOR + timedelta(hours=10),
                max_slots=10,
            )
        self.assertEqual(raised.exception.slot_count, 11)

    def test_slot_membership_and_latest_slot_are_exact(self) -> None:
        job = _job()
        self.assertFalse(is_scheduled_slot(job, _ANCHOR - timedelta(hours=1)))
        self.assertTrue(is_scheduled_slot(job, _ANCHOR + timedelta(hours=4)))
        self.assertFalse(is_scheduled_slot(job, _ANCHOR + timedelta(hours=4, microseconds=1)))
        self.assertIsNone(latest_scheduled_slot(job, _ANCHOR - timedelta(microseconds=1)))
        self.assertEqual(
            latest_scheduled_slot(job, _ANCHOR + timedelta(hours=4, minutes=59)),
            _ANCHOR + timedelta(hours=4),
        )

    def test_finds_first_unregistered_slot_without_enumerating_schedule_age(self) -> None:
        job = _job()
        boundary = _ANCHOR + timedelta(hours=4)
        registered = (
            _ANCHOR,
            _ANCHOR + timedelta(hours=2),
            _ANCHOR + timedelta(hours=3),
        )
        self.assertEqual(
            first_unregistered_slot_before(job, registered, before=boundary),
            _ANCHOR + timedelta(hours=1),
        )
        with self.assertRaises(ScheduleHistoryGapError) as raised:
            require_complete_registered_history(job, registered, before=boundary)
        self.assertEqual(raised.exception.missing_slot, _ANCHOR + timedelta(hours=1))
        self.assertEqual(raised.exception.catch_up_boundary, boundary)
        self.assertIn('register-due', str(raised.exception))

    def test_history_boundary_is_strict_and_complete_prefix_passes(self) -> None:
        job = _job()
        registered = (_ANCHOR, _ANCHOR + timedelta(hours=1))
        self.assertIsNone(
            first_unregistered_slot_before(
                job,
                registered,
                before=_ANCHOR + timedelta(hours=2),
            )
        )
        self.assertEqual(
            first_unregistered_slot_before(
                job,
                registered,
                before=_ANCHOR + timedelta(hours=2, microseconds=1),
            ),
            _ANCHOR + timedelta(hours=2),
        )

    def test_rejects_invalid_window_and_naive_time(self) -> None:
        with self.assertRaisesRegex(ScheduleError, 'window_end'):
            enumerate_scheduled_slots(
                _job(),
                window_start=_ANCHOR,
                window_end=_ANCHOR - timedelta(seconds=1),
            )
        with self.assertRaisesRegex(ValueError, 'UTC offset'):
            latest_scheduled_slot(_job(), datetime(2026, 7, 13, 12, 0))


if __name__ == '__main__':
    unittest.main()
