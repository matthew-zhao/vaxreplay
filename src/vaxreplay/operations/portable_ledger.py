"""Portable, offline-verifiable serialization of an operations-ledger prefix.

The SQLite operations store is the live coordination database, not a portable
membership proof.  This module exports the exact canonical :class:`LedgerEvent`
objects committed by a :class:`LedgerCheckpoint` as newline-delimited JSON and
verifies that representation without consulting SQLite.

Export is intentionally a small composable operation.  A caller that needs one
atomic snapshot across ledger export and other promotion work must hold
``OperationalStore.verification_window()`` for the entire operation.  The export
function does not acquire that window itself because nesting the store's
writer-blocking SQLite transaction would deadlock.
"""

from __future__ import annotations

import hashlib
import re

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.schema import (
    LedgerCheckpoint,
    LedgerEvent,
    LedgerEventType,
    ledger_event_sha256,
)
from vaxreplay.operations.store import OperationalStore

DEFAULT_MAX_LEDGER_PREFIX_EVENTS = 1_000_000
DEFAULT_MAX_LEDGER_EVENT_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_LEDGER_PREFIX_BYTES = 512 * 1024 * 1024

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


class PortableLedgerIntegrityError(ValueError):
    """A portable ledger prefix is malformed or disagrees with its checkpoint."""


def ledger_prefix_sha256(payload: bytes) -> str:
    """Return the SHA-256 digest of the exact portable prefix bytes."""

    if not isinstance(payload, bytes):
        raise TypeError('portable ledger prefix payload must be bytes')
    return hashlib.sha256(payload).hexdigest()


def encode_ledger_prefix(
    events: tuple[LedgerEvent, ...],
    *,
    max_events: int = DEFAULT_MAX_LEDGER_PREFIX_EVENTS,
    max_event_bytes: int = DEFAULT_MAX_LEDGER_EVENT_BYTES,
    max_bytes: int = DEFAULT_MAX_LEDGER_PREFIX_BYTES,
) -> bytes:
    """Encode a nonempty event tuple using the one allowed JSONL representation."""

    _validate_bounds(max_events=max_events, max_event_bytes=max_event_bytes, max_bytes=max_bytes)
    if not isinstance(events, tuple):
        raise TypeError('ledger events must be an immutable tuple')
    if not events:
        raise PortableLedgerIntegrityError('portable ledger prefix cannot be empty')
    if len(events) > max_events:
        raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_events={max_events}')

    lines: list[bytes] = []
    byte_count = 0
    for ordinal, event in enumerate(events, start=1):
        if not isinstance(event, LedgerEvent):
            raise TypeError(f'ledger event {ordinal} is not a LedgerEvent')
        line = canonical_json_bytes(event)
        if len(line) > max_event_bytes:
            raise PortableLedgerIntegrityError(f'ledger event {ordinal} exceeds max_event_bytes={max_event_bytes}')
        byte_count += len(line) + 1
        if byte_count > max_bytes:
            raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_bytes={max_bytes}')
        lines.append(line + b'\n')
    return b''.join(lines)


def verify_ledger_prefix(
    events: tuple[LedgerEvent, ...],
    checkpoint: LedgerCheckpoint,
    *,
    max_events: int = DEFAULT_MAX_LEDGER_PREFIX_EVENTS,
) -> None:
    """Verify an immutable event tuple as the complete prefix for ``checkpoint``.

    This verifies membership and the checkpoint's object-inventory commitment.  It
    does not replay collector-specific semantics or authenticate an external witness.
    """

    _validate_positive_bound(max_events, 'max_events')
    if not isinstance(events, tuple):
        raise TypeError('ledger events must be an immutable tuple')
    if not isinstance(checkpoint, LedgerCheckpoint):
        raise TypeError('checkpoint must be a LedgerCheckpoint')
    if not events:
        raise PortableLedgerIntegrityError('portable ledger prefix cannot be empty')
    if len(events) > max_events or checkpoint.through_sequence > max_events:
        raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_events={max_events}')
    if len(events) != checkpoint.through_sequence:
        raise PortableLedgerIntegrityError('ledger event count does not equal checkpoint through_sequence')

    previous_sha256: str | None = None
    inventory: dict[str, int] = {}
    for expected_sequence, event in enumerate(events, start=1):
        if not isinstance(event, LedgerEvent):
            raise TypeError(f'ledger event {expected_sequence} is not a LedgerEvent')
        if event.sequence != expected_sequence:
            raise PortableLedgerIntegrityError('ledger event sequence is not contiguous from one')
        if event.previous_event_sha256 != previous_sha256:
            raise PortableLedgerIntegrityError('ledger previous-event hash chain is broken')
        if ledger_event_sha256(event) != event.event_sha256:
            raise PortableLedgerIntegrityError('ledger event digest does not bind its canonical preimage')
        if event.occurred_at > checkpoint.created_at:
            raise PortableLedgerIntegrityError('checkpoint predates an event in its ledger prefix')
        if event.event_type is LedgerEventType.ARTIFACT_STORED:
            artifact_sha256 = event.payload.get('artifact_sha256')
            byte_count = event.payload.get('byte_count')
            if (
                not isinstance(artifact_sha256, str)
                or _SHA256_RE.fullmatch(artifact_sha256) is None
                or not isinstance(byte_count, int)
                or isinstance(byte_count, bool)
                or byte_count < 0
            ):
                raise PortableLedgerIntegrityError('artifact ledger event has malformed inventory fields')
            if artifact_sha256 in inventory:
                raise PortableLedgerIntegrityError('artifact ledger contains duplicate first-record events')
            inventory[artifact_sha256] = byte_count
        previous_sha256 = event.event_sha256

    first = events[0]
    if first.event_type is not LedgerEventType.STORE_INITIALIZED:
        raise PortableLedgerIntegrityError('first ledger event must initialize the operations store')
    if sum(event.event_type is LedgerEventType.STORE_INITIALIZED for event in events) != 1:
        raise PortableLedgerIntegrityError('store initialization event must be unique')
    if first.payload.get('store_id') != checkpoint.store_id:
        raise PortableLedgerIntegrityError('ledger store identity does not match checkpoint')

    last = events[-1]
    if last.sequence != checkpoint.through_sequence or last.event_sha256 != checkpoint.through_event_sha256:
        raise PortableLedgerIntegrityError('ledger head does not match checkpoint')

    canonical_inventory = tuple(sorted(inventory.items()))
    inventory_sha256 = hashlib.sha256(canonical_json_bytes(canonical_inventory)).hexdigest()
    if len(canonical_inventory) != checkpoint.object_count or inventory_sha256 != checkpoint.object_inventory_sha256:
        raise PortableLedgerIntegrityError('ledger object inventory does not match checkpoint')


def parse_ledger_prefix(
    payload: bytes,
    checkpoint: LedgerCheckpoint,
    *,
    max_events: int = DEFAULT_MAX_LEDGER_PREFIX_EVENTS,
    max_event_bytes: int = DEFAULT_MAX_LEDGER_EVENT_BYTES,
    max_bytes: int = DEFAULT_MAX_LEDGER_PREFIX_BYTES,
) -> tuple[LedgerEvent, ...]:
    """Parse and verify exact canonical JSONL bytes against ``checkpoint``."""

    _validate_bounds(max_events=max_events, max_event_bytes=max_event_bytes, max_bytes=max_bytes)
    if not isinstance(payload, bytes):
        raise TypeError('portable ledger prefix payload must be bytes')
    if not payload:
        raise PortableLedgerIntegrityError('portable ledger prefix cannot be empty')
    if len(payload) > max_bytes:
        raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_bytes={max_bytes}')
    if not payload.endswith(b'\n'):
        raise PortableLedgerIntegrityError('portable ledger prefix must end with exactly one record newline')

    raw_lines = payload.split(b'\n')
    if raw_lines[-1] != b'':  # Kept explicit so format changes cannot weaken the final-newline rule.
        raise PortableLedgerIntegrityError('portable ledger prefix has an invalid final record delimiter')
    raw_lines.pop()
    if not raw_lines or any(not line for line in raw_lines):
        raise PortableLedgerIntegrityError('portable ledger prefix cannot contain blank lines')
    if len(raw_lines) > max_events or checkpoint.through_sequence > max_events:
        raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_events={max_events}')

    events: list[LedgerEvent] = []
    for ordinal, line in enumerate(raw_lines, start=1):
        if len(line) > max_event_bytes:
            raise PortableLedgerIntegrityError(f'ledger event {ordinal} exceeds max_event_bytes={max_event_bytes}')
        try:
            event = LedgerEvent.model_validate_json(line)
        except ValueError as error:
            raise PortableLedgerIntegrityError(f'ledger event {ordinal} is invalid') from error
        if line != canonical_json_bytes(event):
            raise PortableLedgerIntegrityError(f'ledger event {ordinal} must use canonical JSON encoding')
        events.append(event)

    result = tuple(events)
    verify_ledger_prefix(result, checkpoint, max_events=max_events)
    return result


def export_ledger_prefix(
    store: OperationalStore,
    checkpoint: LedgerCheckpoint,
    *,
    max_events: int = DEFAULT_MAX_LEDGER_PREFIX_EVENTS,
    max_event_bytes: int = DEFAULT_MAX_LEDGER_EVENT_BYTES,
    max_bytes: int = DEFAULT_MAX_LEDGER_PREFIX_BYTES,
) -> bytes:
    """Export the exact local ledger prefix committed by ``checkpoint``.

    The caller must hold ``store.verification_window()`` while calling this
    function whenever the returned prefix is composed with other live-store reads.
    That caller-held window makes checkpoint verification and event selection one
    stable snapshot without forcing an uncomposable nested write reservation here.
    """

    if not isinstance(store, OperationalStore):
        raise TypeError('store must be an OperationalStore')
    _validate_bounds(max_events=max_events, max_event_bytes=max_event_bytes, max_bytes=max_bytes)
    if checkpoint.through_sequence > max_events:
        raise PortableLedgerIntegrityError(f'portable ledger prefix exceeds max_events={max_events}')

    store.verify_checkpoint(checkpoint)
    events = store.events()[: checkpoint.through_sequence]
    verify_ledger_prefix(events, checkpoint, max_events=max_events)
    payload = encode_ledger_prefix(
        events,
        max_events=max_events,
        max_event_bytes=max_event_bytes,
        max_bytes=max_bytes,
    )
    # Exercise the byte-level verifier before crossing the portable boundary.  This
    # also keeps export and offline-load acceptance rules exactly aligned.
    parse_ledger_prefix(
        payload,
        checkpoint,
        max_events=max_events,
        max_event_bytes=max_event_bytes,
        max_bytes=max_bytes,
    )
    return payload


def _validate_bounds(*, max_events: int, max_event_bytes: int, max_bytes: int) -> None:
    _validate_positive_bound(max_events, 'max_events')
    _validate_positive_bound(max_event_bytes, 'max_event_bytes')
    _validate_positive_bound(max_bytes, 'max_bytes')


def _validate_positive_bound(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{field_name} must be a positive integer')
