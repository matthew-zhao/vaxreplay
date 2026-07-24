"""Sealed case-universe and exhaustive selection-audit contracts for research releases."""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.temporal_schema import TemporalReceiptAuthority

CASE_UNIVERSE_SCHEMA_VERSION = 'vaxreplay.case-universe.v0.1'
CASE_UNIVERSE_SEAL_SCHEMA_VERSION = 'vaxreplay.case-universe-seal.v0.1'
CASE_SELECTION_AUDIT_SCHEMA_VERSION = 'vaxreplay.case-selection-audit.v0.1'

_INDEPENDENT_AUTHORITIES = {
    TemporalReceiptAuthority.RFC3161_TIMESTAMP,
    TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
    TemporalReceiptAuthority.SOURCE_SIGNED_VERSION,
    TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
}


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{name} must include a UTC offset')
    return value.astimezone(timezone.utc)


class CaseUniverseDisposition(str, enum.Enum):
    PREELIGIBLE = 'preeligible'
    EXCLUDED_PREDEFINED = 'excluded_predefined'


class CaseSelectionDisposition(str, enum.Enum):
    ADMITTED = 'admitted'
    EXCLUDED_PREDEFINED = 'excluded_predefined'
    UNSCORED_MISSING = 'unscored_missing'
    CONFLICT = 'conflict'
    QUARANTINED_CONTAMINATION = 'quarantined_contamination'


class CaseUniverseEntry(StrictModel):
    case_id: str = Field(min_length=1)
    lineage_group_id: str = Field(min_length=1)
    disposition: CaseUniverseDisposition
    decision_package_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    reason_code: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        if self.disposition == CaseUniverseDisposition.PREELIGIBLE:
            if self.decision_package_sha256 is None or self.reason_code is not None:
                raise ValueError('preeligible cases require a decision-package hash and no exclusion reason')
        elif self.decision_package_sha256 is not None or self.reason_code is None:
            raise ValueError('predefined exclusions require a reason and no decision-package hash')
        return self


class CaseUniverseSeal(StrictModel):
    schema_version: Literal['vaxreplay.case-universe-seal.v0.1'] = CASE_UNIVERSE_SEAL_SCHEMA_VERSION
    universe_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    witnessed_at: datetime
    authority_type: TemporalReceiptAuthority
    authority_id: str = Field(min_length=1)
    proof_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    proof_bytes: int = Field(gt=0)
    verification_uri: str = Field(min_length=1)

    @field_validator('witnessed_at')
    @classmethod
    def validate_witnessed_at(cls, value: datetime) -> datetime:
        return _aware(value, 'witnessed_at')

    @model_validator(mode='after')
    def validate_authority(self) -> Self:
        if self.authority_type not in _INDEPENDENT_AUTHORITIES:
            raise ValueError('case-universe seals require an independent authority')
        return self


def case_universe_content_sha256(
    *,
    universe_id: str,
    eligibility_protocol_sha256: str,
    entries: tuple[CaseUniverseEntry, ...],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                'universe_id': universe_id,
                'eligibility_protocol_sha256': eligibility_protocol_sha256,
                'entries': [entry.model_dump(mode='json') for entry in entries],
            }
        )
    ).hexdigest()


class CaseUniverseManifest(StrictModel):
    """All cases found by a pre-outcome, hash-bound eligibility protocol."""

    schema_version: Literal['vaxreplay.case-universe.v0.1'] = CASE_UNIVERSE_SCHEMA_VERSION
    universe_id: str = Field(min_length=1)
    eligibility_protocol_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    inventory_complete: Literal[True] = True
    entries: tuple[CaseUniverseEntry, ...] = Field(min_length=1)
    universe_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    seal: CaseUniverseSeal

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        case_ids = tuple(entry.case_id for entry in self.entries)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError('case-universe entries must have unique case IDs in sorted order')
        expected = case_universe_content_sha256(
            universe_id=self.universe_id,
            eligibility_protocol_sha256=self.eligibility_protocol_sha256,
            entries=self.entries,
        )
        if self.universe_content_sha256 != expected or self.seal.universe_content_sha256 != expected:
            raise ValueError('case-universe seal does not bind the exact universe content')
        return self


def case_universe_sha256(manifest: CaseUniverseManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


class CaseSelectionRecord(StrictModel):
    case_id: str = Field(min_length=1)
    disposition: CaseSelectionDisposition
    episode_id: str | None = Field(default=None, min_length=1)
    manifest_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    panel_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    reason_code: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.observed_count + self.missing_count + self.conflict_count != self.panel_count:
            raise ValueError('case-selection outcome counts must sum to the frozen panel count')
        if self.disposition == CaseSelectionDisposition.ADMITTED:
            if (
                self.episode_id is None
                or self.manifest_sha256 is None
                or self.reason_code is not None
                or self.panel_count == 0
                or self.observed_count != self.panel_count
            ):
                raise ValueError('admitted cases require a complete observed panel and episode binding')
        elif self.episode_id is not None or self.manifest_sha256 is not None or self.reason_code is None:
            raise ValueError('unadmitted cases require a reason and cannot bind a scoring episode')
        elif self.disposition == CaseSelectionDisposition.EXCLUDED_PREDEFINED and self.panel_count != 0:
            raise ValueError('predefined exclusions cannot contain a post-outcome panel')
        elif self.disposition == CaseSelectionDisposition.UNSCORED_MISSING and self.missing_count == 0:
            raise ValueError('missing cases must report at least one missing panel outcome')
        elif self.disposition == CaseSelectionDisposition.CONFLICT and self.conflict_count == 0:
            raise ValueError('conflicted cases must report at least one conflict')
        elif self.disposition == CaseSelectionDisposition.QUARANTINED_CONTAMINATION and self.panel_count == 0:
            raise ValueError('contamination-quarantined cases must report a non-empty frozen panel')
        return self


class CaseSelectionAudit(StrictModel):
    """One post-outcome disposition for every case in the sealed universe."""

    schema_version: Literal['vaxreplay.case-selection-audit.v0.1'] = CASE_SELECTION_AUDIT_SCHEMA_VERSION
    case_universe_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    selection_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    inventory_complete: Literal[True] = True
    records: tuple[CaseSelectionRecord, ...] = Field(min_length=1)

    @field_validator('records')
    @classmethod
    def validate_records(cls, value: tuple[CaseSelectionRecord, ...]) -> tuple[CaseSelectionRecord, ...]:
        case_ids = tuple(record.case_id for record in value)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError('case-selection records must have unique case IDs in sorted order')
        return value


def case_selection_audit_sha256(audit: CaseSelectionAudit) -> str:
    return hashlib.sha256(canonical_json_bytes(audit)).hexdigest()


def validate_case_selection_inventory(
    universe: CaseUniverseManifest,
    selection: CaseSelectionAudit,
    admitted_bundles: Iterable[EpisodeBundle],
) -> None:
    """Bind every built episode to one admitted record and expose every omitted case."""

    validate_case_selection_bindings(
        universe,
        selection,
        (
            (bundle.manifest.episode_id, bundle.manifest_sha256, bundle.manifest.lineage_group_id)
            for bundle in admitted_bundles
        ),
    )


def validate_case_selection_bindings(
    universe: CaseUniverseManifest,
    selection: CaseSelectionAudit,
    admitted_bindings: Iterable[tuple[str, str, str]],
) -> None:
    """Validate against ``(episode_id, manifest_sha256, lineage_group_id)`` bindings."""

    if selection.case_universe_sha256 != case_universe_sha256(universe):
        raise ValueError('case-selection audit does not bind the sealed case universe')
    universe_by_id = {entry.case_id: entry for entry in universe.entries}
    selection_by_id = {record.case_id: record for record in selection.records}
    if selection_by_id.keys() != universe_by_id.keys():
        raise ValueError('case-selection audit must cover every universe case exactly once')
    for case_id, entry in universe_by_id.items():
        disposition = selection_by_id[case_id].disposition
        if entry.disposition == CaseUniverseDisposition.EXCLUDED_PREDEFINED:
            if disposition != CaseSelectionDisposition.EXCLUDED_PREDEFINED:
                raise ValueError('predefined universe exclusions cannot change after outcomes')
        elif disposition == CaseSelectionDisposition.EXCLUDED_PREDEFINED:
            raise ValueError('preeligible cases cannot become predefined exclusions after outcomes')

    bindings = tuple(admitted_bindings)
    expected = {
        (record.episode_id, record.manifest_sha256)
        for record in selection.records
        if record.disposition == CaseSelectionDisposition.ADMITTED
    }
    actual = {(episode_id, manifest_sha256) for episode_id, manifest_sha256, _lineage in bindings}
    if expected != actual or len(expected) != len(bindings):
        raise ValueError('admitted case-selection records must exactly match the complete bundle inventory')
    lineage_by_episode = {
        record.episode_id: universe_by_id[record.case_id].lineage_group_id
        for record in selection.records
        if record.disposition == CaseSelectionDisposition.ADMITTED
    }
    for episode_id, _manifest_sha256, lineage_group_id in bindings:
        if lineage_by_episode[episode_id] != lineage_group_id:
            raise ValueError('case-universe lineage does not match its admitted scoring episode')
