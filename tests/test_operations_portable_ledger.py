from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.portable_ledger import (
    PortableLedgerIntegrityError,
    encode_ledger_prefix,
    export_ledger_prefix,
    ledger_prefix_sha256,
    parse_ledger_prefix,
    verify_ledger_prefix,
)
from vaxreplay.operations.schema import CaptureJobSpec, LedgerEventType
from vaxreplay.operations.store import OperationalStore

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_STORE_ID = '1234567890abcdef1234567890abcdef'


class PortableLedgerPrefixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = OperationalStore.initialize(
            Path(self.temporary_directory.name) / 'operations',
            created_at=_T0,
            store_id=_STORE_ID,
            trusted_lease_clock=None,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _artifact_checkpoint(self):
        self.store.put_bytes(b'first exact object', recorded_at=_T0 + timedelta(seconds=1))
        self.store.put_bytes(b'second exact object', recorded_at=_T0 + timedelta(seconds=2))
        return self.store.checkpoint(created_at=_T0 + timedelta(seconds=3))

    def _export(self, checkpoint) -> bytes:
        with self.store.verification_window():
            return export_ledger_prefix(self.store, checkpoint)

    def test_export_round_trip_is_canonical_portable_and_checkpoint_bound(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)
        events = parse_ledger_prefix(payload, checkpoint)

        self.assertEqual(events, self.store.events())
        self.assertEqual(len(events), checkpoint.through_sequence)
        self.assertEqual(payload, encode_ledger_prefix(events))
        self.assertEqual(ledger_prefix_sha256(payload), hashlib.sha256(payload).hexdigest())
        self.assertTrue(payload.endswith(b'\n'))
        self.assertNotIn(b'\n\n', payload)
        verify_ledger_prefix(events, checkpoint)

    def test_success_after_checkpoint_is_not_part_of_exported_membership_proof(self) -> None:
        job = self.store.register_job(
            CaptureJobSpec(
                job_id='portable-prefix-job',
                collector_id='test-collector-v1',
                schedule_anchor_at=_T0,
                schedule_interval_seconds=3600,
                configuration={},
            ),
            registered_at=_T0 + timedelta(seconds=1),
        )
        run = self.store.register_logical_run(
            job.spec_sha256,
            _T0,
            registered_at=_T0 + timedelta(seconds=2),
        )
        manifest = self.store.put_bytes(b'run manifest', recorded_at=_T0 + timedelta(seconds=3))
        attempt = self.store.begin_attempt(
            run.logical_run_id,
            owner_id='worker-1',
            now=_T0 + timedelta(seconds=4),
            lease_seconds=60,
            initial_artifacts={'run-manifest': manifest.sha256},
        )
        checkpoint = self.store.checkpoint(created_at=_T0 + timedelta(seconds=5))

        self.store.succeed_attempt(
            attempt.attempt_id,
            owner_id='worker-1',
            now=_T0 + timedelta(seconds=6),
        )
        payload = self._export(checkpoint)
        prefix = parse_ledger_prefix(payload, checkpoint)

        self.assertNotIn(LedgerEventType.ATTEMPT_SUCCEEDED, {event.event_type for event in prefix})
        self.assertEqual(prefix[-1].event_sha256, checkpoint.through_event_sha256)
        self.assertEqual(self.store.events()[-1].event_type, LedgerEventType.ATTEMPT_SUCCEEDED)
        self.assertGreater(len(self.store.events()), len(prefix))

    def test_noncanonical_event_and_invalid_event_digest_are_rejected(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)
        lines = payload.removesuffix(b'\n').split(b'\n')

        noncanonical = lines.copy()
        noncanonical[0] = noncanonical[0].replace(b':', b': ', 1)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'canonical JSON'):
            parse_ledger_prefix(b'\n'.join(noncanonical) + b'\n', checkpoint)

        changed = json.loads(lines[1])
        changed['payload']['byte_count'] += 1
        tampered = lines.copy()
        tampered[1] = canonical_json_bytes(changed)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'event 2 is invalid'):
            parse_ledger_prefix(b'\n'.join(tampered) + b'\n', checkpoint)

    def test_missing_reordered_and_wrong_head_events_are_rejected(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)
        lines = payload.removesuffix(b'\n').split(b'\n')

        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'event count'):
            parse_ledger_prefix(b'\n'.join(lines[:-1]) + b'\n', checkpoint)

        reordered = [lines[1], lines[0], *lines[2:]]
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'sequence'):
            parse_ledger_prefix(b'\n'.join(reordered) + b'\n', checkpoint)

        wrong_head = checkpoint.model_copy(update={'through_event_sha256': '0' * 64})
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'head'):
            parse_ledger_prefix(payload, wrong_head)

    def test_inventory_store_identity_and_event_time_must_match_checkpoint(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)

        wrong_count = checkpoint.model_copy(update={'object_count': checkpoint.object_count + 1})
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'object inventory'):
            parse_ledger_prefix(payload, wrong_count)

        wrong_inventory = checkpoint.model_copy(update={'object_inventory_sha256': '0' * 64})
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'object inventory'):
            parse_ledger_prefix(payload, wrong_inventory)

        wrong_store = checkpoint.model_copy(update={'store_id': 'f' * 32})
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'store identity'):
            parse_ledger_prefix(payload, wrong_store)

        predating = checkpoint.model_copy(update={'created_at': _T0 + timedelta(seconds=1)})
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'predates an event'):
            parse_ledger_prefix(payload, predating)

    def test_format_rejects_empty_blank_missing_newline_and_crlf(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)
        first_newline = payload.index(b'\n')

        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'cannot be empty'):
            parse_ledger_prefix(b'', checkpoint)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'blank lines'):
            parse_ledger_prefix(payload[: first_newline + 1] + b'\n' + payload[first_newline + 1 :], checkpoint)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'must end'):
            parse_ledger_prefix(payload.removesuffix(b'\n'), checkpoint)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'canonical JSON'):
            parse_ledger_prefix(payload.replace(b'\n', b'\r\n'), checkpoint)

    def test_event_count_event_size_and_total_size_bounds_fail_closed(self) -> None:
        checkpoint = self._artifact_checkpoint()
        payload = self._export(checkpoint)
        longest_line = max(len(line) for line in payload.splitlines())

        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'max_events'):
            parse_ledger_prefix(payload, checkpoint, max_events=checkpoint.through_sequence - 1)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'max_event_bytes'):
            parse_ledger_prefix(payload, checkpoint, max_event_bytes=longest_line - 1)
        with self.assertRaisesRegex(PortableLedgerIntegrityError, 'max_bytes'):
            parse_ledger_prefix(payload, checkpoint, max_bytes=len(payload) - 1)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            parse_ledger_prefix(payload, checkpoint, max_events=True)


if __name__ == '__main__':
    unittest.main()
