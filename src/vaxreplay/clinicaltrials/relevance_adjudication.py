"""Decision-only active-vaccine relevance review for provisional AACT cohorts.

The review queue is deliberately built from the historical *decision* archives.  This module has
no label-schema imports and no parameter through which a later archive can be supplied.  It binds
the exact decision text shown to an organizer to the merged decision inventory, and then binds each
INCLUDE/EXCLUDE/HOLD decision to the hash of that text.
"""

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import io
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Self

from pydantic import Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_inventory import ExecutionInventoryError, audit_execution_inventory
from vaxreplay.clinicaltrials.execution_schema import ExecutionCohortInventory

RELEVANCE_POLICY_ID = 'aact-active-prophylactic-vaccine-relevance-v0.2'
REVIEW_QUEUE_SCHEMA_VERSION = 'vaxreplay.aact-vaccine-relevance-review-queue.v0.1'
ADJUDICATION_SCHEMA_VERSION = 'vaxreplay.aact-vaccine-relevance-adjudications.v0.1'
REVIEW_RECEIPT_SCHEMA_VERSION = 'vaxreplay.aact-vaccine-relevance-review-receipt.v0.1'
REVIEW_METHOD_ID = 'organizer-decision-only-manual-policy-review-v0.2'

ACTIVE_VACCINE_RELEVANCE_POLICY = {
    'policy_id': RELEVANCE_POLICY_ID,
    'scope': (
        'An infectious-disease intervention administered to induce active host immunity, where the '
        'record is studying an investigational vaccine candidate or candidate regimen.'
    ),
    'archive_side': 'decision_only',
    'future_registry_updates_or_execution_labels_permitted': False,
    'include': [
        'At least one administered intervention is an active prophylactic vaccine candidate.',
        'Challenge agents and licensed comparators may coexist with an otherwise eligible candidate.',
        'A licensed antigen plus an investigational vaccine adjuvant is an eligible candidate regimen when the '
        'decision-time record explicitly studies their co-administration as a vaccine regimen.',
        'Controlled-infection material may itself be the immunizing component only when the decision-time record '
        'explicitly describes repeated administration as vaccination and distinguishes it from a later challenge.',
        'A candidate may use nucleic acid, viral vector, protein, particle, inactivated, or live-attenuated platforms.',
    ],
    'exclude': [
        'Passive monoclonal or polyclonal antibody administration without an active vaccine candidate.',
        'Antibody-encoding prophylaxis whose intended product is antibody rather than antigen expression.',
        'Transferred immune-cell or other cellular therapy without an active vaccine candidate.',
        'Microbiome, probiotic, fecal-transplant, or non-vaccine decolonization without an active vaccine candidate.',
        'Challenge or controlled-infection material without an active vaccine candidate.',
        'Licensed-vaccine immune-response, uptake, delivery, schedule, or peri-vaccination immune-modulation research '
        'without an investigational vaccine candidate or investigational vaccine-regimen component.',
        'Therapeutic vaccination in infected or diseased participants without prophylactic candidate development.',
        'Any other intervention that does not administer an active prophylactic vaccine candidate.',
    ],
    'hold': [
        'Use HOLD when the decision-time record cannot establish active-vaccine product identity.',
        'A product code plus generic safety or immunogenicity language is not enough to establish vaccine identity.',
        'Use HOLD when decision-time text cannot distinguish candidate development from licensed-vaccine research.',
        'Use HOLD for mixed or internally conflicting decision-time evidence that changes the disposition.',
    ],
    'conservative_rule': 'Ambiguity is HOLD, never inferred INCLUDE.',
}

_MEMBER_FIELDS: dict[str, tuple[str, ...]] = {
    'brief_summaries.txt': ('description', 'nct_id'),
    'conditions.txt': ('name', 'nct_id'),
    'designs.txt': ('nct_id', 'primary_purpose'),
    'detailed_descriptions.txt': ('description', 'nct_id'),
    'interventions.txt': ('description', 'intervention_type', 'name', 'nct_id'),
    'sponsors.txt': ('agency_class', 'lead_or_collaborator', 'name', 'nct_id'),
    'studies.txt': ('acronym', 'brief_title', 'nct_id', 'official_title'),
}
_REQUIRED_MEMBERS = tuple(sorted(_MEMBER_FIELDS))


class RelevanceReviewError(ValueError):
    """A decision-only review artifact failed closed."""


class RelevanceDisposition(str, enum.Enum):
    INCLUDE = 'include'
    EXCLUDE = 'exclude'
    HOLD = 'hold'


class RelevanceReason(str, enum.Enum):
    INCLUDE_ACTIVE_PROPHYLACTIC_VACCINE_CANDIDATE = 'include_active_prophylactic_vaccine_candidate'
    EXCLUDE_PASSIVE_ANTIBODY = 'exclude_passive_antibody'
    EXCLUDE_ANTIBODY_ENCODING_PROPHYLAXIS = 'exclude_antibody_encoding_prophylaxis'
    EXCLUDE_CELLULAR_THERAPY = 'exclude_cellular_therapy'
    EXCLUDE_MICROBIOME_PROBIOTIC_OR_DECOLONIZATION = 'exclude_microbiome_probiotic_or_decolonization'
    EXCLUDE_CHALLENGE_ONLY = 'exclude_challenge_only'
    EXCLUDE_LICENSED_VACCINE_RESPONSE = 'exclude_licensed_vaccine_response'
    EXCLUDE_LICENSED_VACCINE_UPTAKE_OR_EDUCATION = 'exclude_licensed_vaccine_uptake_or_education'
    EXCLUDE_LICENSED_VACCINE_DELIVERY_OR_SCHEDULE = 'exclude_licensed_vaccine_delivery_or_schedule'
    EXCLUDE_LICENSED_VACCINE_IMMUNE_MODULATION = 'exclude_licensed_vaccine_immune_modulation'
    EXCLUDE_THERAPEUTIC_VACCINATION = 'exclude_therapeutic_vaccination'
    EXCLUDE_NOT_ACTIVE_PROPHYLACTIC_VACCINE = 'exclude_not_active_prophylactic_vaccine'
    HOLD_ACTIVE_PRODUCT_IDENTITY_AMBIGUOUS = 'hold_active_product_identity_ambiguous'
    HOLD_LICENSED_VERSUS_CANDIDATE_AMBIGUOUS = 'hold_licensed_versus_candidate_ambiguous'
    HOLD_MIXED_OR_CONFLICTING_EVIDENCE = 'hold_mixed_or_conflicting_evidence'
    HOLD_INSUFFICIENT_DECISION_EVIDENCE = 'hold_insufficient_decision_evidence'


class EvidenceSourceRow(StrictModel):
    member_path: str = Field(min_length=1)
    data_row_number: int = Field(gt=0)
    raw_row_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    fields_read: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_fields(self) -> Self:
        if self.fields_read != tuple(sorted(set(self.fields_read))):
            raise ValueError('source-row fields must be unique and sorted')
        return self


class InterventionEvidence(StrictModel):
    intervention_type: str
    name: str
    description: str


class SponsorEvidence(StrictModel):
    lead_or_collaborator: str
    agency_class: str
    name: str


class DecisionEvidenceBody(StrictModel):
    anchor_date: date
    snapshot_id: str = Field(min_length=1)
    decision_archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    brief_title: str
    official_title: str
    acronym: str
    primary_purposes: tuple[str, ...]
    conditions: tuple[str, ...]
    interventions: tuple[InterventionEvidence, ...]
    brief_summary: str
    detailed_description: str
    sponsors: tuple[SponsorEvidence, ...]
    source_rows: tuple[EvidenceSourceRow, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_evidence(self) -> Self:
        if not self.brief_title and not self.official_title:
            raise ValueError('decision evidence requires at least one historical title')
        if not self.interventions:
            raise ValueError('decision evidence requires at least one historical intervention')
        row_keys = tuple((row.member_path, row.data_row_number) for row in self.source_rows)
        if row_keys != tuple(sorted(set(row_keys))):
            raise ValueError('decision evidence source rows must be unique and sorted')
        return self


class DecisionEvidenceRecord(DecisionEvidenceBody):
    evidence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_evidence_hash(self) -> Self:
        body = DecisionEvidenceBody.model_validate(self.model_dump(exclude={'evidence_sha256'}))
        if self.evidence_sha256 != _model_sha256(body):
            raise ValueError('decision evidence hash does not match its exact content')
        return self


class ArchiveMemberEvidenceBinding(StrictModel):
    member_path: str = Field(min_length=1)
    member_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    crc32_hex: str = Field(pattern=r'^[0-9a-f]{8}$')
    data_row_count: int = Field(ge=0)
    selected_row_count: int = Field(ge=0)
    fields_read: tuple[str, ...] = Field(min_length=1)
    full_member_read_and_crc_verified: Literal[True] = True


class DecisionArchiveEvidenceBinding(StrictModel):
    anchor_date: date
    snapshot_id: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    archive_bytes: int = Field(gt=0)
    full_archive_hashed: Literal[True] = True
    members: tuple[ArchiveMemberEvidenceBinding, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_members(self) -> Self:
        paths = tuple(item.member_path for item in self.members)
        if paths != tuple(sorted(set(paths))) or paths != _REQUIRED_MEMBERS:
            raise ValueError('archive evidence must bind every required decision member exactly once')
        return self


class VaccineRelevanceReviewQueue(StrictModel):
    schema_version: Literal['vaxreplay.aact-vaccine-relevance-review-queue.v0.1'] = REVIEW_QUEUE_SCHEMA_VERSION
    policy_id: Literal['aact-active-prophylactic-vaccine-relevance-v0.2'] = RELEVANCE_POLICY_ID
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    merged_inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision_archives_only: Literal[True] = True
    later_archive_opened: Literal[False] = False
    execution_labels_read: Literal[False] = False
    source_archives: tuple[DecisionArchiveEvidenceBinding, ...] = Field(min_length=1)
    records: tuple[DecisionEvidenceRecord, ...] = Field(min_length=1)
    record_count: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_queue(self) -> Self:
        if self.policy_sha256 != relevance_policy_sha256():
            raise ValueError('review queue policy hash is not the fixed active-vaccine policy')
        archive_dates = tuple(item.anchor_date for item in self.source_archives)
        if archive_dates != tuple(sorted(set(archive_dates))):
            raise ValueError('review queue source archives must be unique and sorted')
        record_keys = tuple((item.anchor_date, item.nct_id) for item in self.records)
        if record_keys != tuple(sorted(set(record_keys))) or self.record_count != len(self.records):
            raise ValueError('review queue records/count must be unique and sorted')
        if set(anchor for anchor, _ in record_keys) != set(archive_dates):
            raise ValueError('review queue records must use exactly the bound archive anchors')
        return self


class RelevanceReviewInput(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    anchor_date: date
    evidence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    disposition: RelevanceDisposition
    reason_codes: tuple[RelevanceReason, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_reasons(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError('review reason codes must be unique and sorted')
        expected_prefix = f'{self.disposition.value}_'
        if any(not reason.value.startswith(expected_prefix) for reason in self.reason_codes):
            raise ValueError('review reason codes must match the disposition')
        return self


class VaccineRelevanceAdjudicationSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-vaccine-relevance-adjudications.v0.1'] = ADJUDICATION_SCHEMA_VERSION
    policy_id: Literal['aact-active-prophylactic-vaccine-relevance-v0.2'] = RELEVANCE_POLICY_ID
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    review_method_id: Literal['organizer-decision-only-manual-policy-review-v0.2'] = REVIEW_METHOD_ID
    review_queue_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    merged_inventory_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provisional: Literal[True] = True
    decision_only_relevance_review_complete: Literal[True] = True
    bound_to_scored_cohort: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    lineage_review_complete: Literal[False] = False
    lineage_split_safe: Literal[False] = False
    decisions: tuple[RelevanceReviewInput, ...] = Field(min_length=1)
    include_count: int = Field(ge=0)
    exclude_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_adjudications(self) -> Self:
        if self.policy_sha256 != relevance_policy_sha256():
            raise ValueError('adjudications are not bound to the fixed active-vaccine policy')
        keys = tuple((item.anchor_date, item.nct_id) for item in self.decisions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError('relevance decisions must be unique and sorted')
        expected = {
            RelevanceDisposition.INCLUDE: self.include_count,
            RelevanceDisposition.EXCLUDE: self.exclude_count,
            RelevanceDisposition.HOLD: self.hold_count,
        }
        for disposition, count in expected.items():
            if sum(item.disposition == disposition for item in self.decisions) != count:
                raise ValueError(f'{disposition.value} count does not match decisions')
        return self


class ReviewArtifactReceipt(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)
    organizer_private: Literal[True] = True


class VaccineRelevanceReviewReceipt(StrictModel):
    schema_version: Literal['vaxreplay.aact-vaccine-relevance-review-receipt.v0.1'] = REVIEW_RECEIPT_SCHEMA_VERSION
    policy_id: Literal['aact-active-prophylactic-vaccine-relevance-v0.2'] = RELEVANCE_POLICY_ID
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provisional: Literal[True] = True
    decision_archives_only: Literal[True] = True
    later_archive_opened: Literal[False] = False
    execution_labels_read: Literal[False] = False
    bound_to_scored_cohort: Literal[False] = False
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    lineage_split_safe: Literal[False] = False
    source_reverification_required: Literal[True] = True
    include_count: int = Field(ge=0)
    exclude_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    artifacts: tuple[ReviewArtifactReceipt, ...] = Field(min_length=3)

    @model_validator(mode='after')
    def validate_artifacts(self) -> Self:
        if self.policy_sha256 != relevance_policy_sha256():
            raise ValueError('review receipt is not bound to the fixed active-vaccine policy')
        paths = tuple(item.relative_path for item in self.artifacts)
        expected_paths = (
            'organizer/relevance-adjudications.json',
            'organizer/relevance-policy.json',
            'organizer/relevance-review-queue.json',
        )
        if paths != expected_paths:
            raise ValueError('review receipt must bind exactly the three organizer artifacts')
        return self


@dataclass(frozen=True)
class _RawRecord:
    data_row_number: int
    raw: bytes


@dataclass(frozen=True)
class _SelectedRow:
    values: dict[str, str]
    source: EvidenceSourceRow


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def relevance_policy_sha256() -> str:
    return _sha256_bytes(canonical_json_bytes(ACTIVE_VACCINE_RELEVANCE_POLICY))


def _hash_seekable_file(source: IO[bytes]) -> tuple[str, int]:
    source.seek(0)
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
    source.seek(0)
    return digest.hexdigest(), byte_count


def _stable_stat_fields(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _iter_raw_records(source: IO[bytes]) -> Iterator[_RawRecord]:
    buffered = bytearray()
    in_quotes = False
    data_row_number = -1
    for physical_line in source:
        buffered.extend(physical_line)
        if physical_line.count(b'"') % 2:
            in_quotes = not in_quotes
        if in_quotes:
            continue
        data_row_number += 1
        yield _RawRecord(data_row_number=data_row_number, raw=bytes(buffered))
        buffered.clear()
    if in_quotes:
        raise RelevanceReviewError('decision member has an unterminated quoted field')
    if buffered:
        data_row_number += 1
        yield _RawRecord(data_row_number=data_row_number, raw=bytes(buffered))


def _parse_values(raw: bytes, member: str, row_number: int) -> list[str]:
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise RelevanceReviewError(f'{member} row {row_number} is not UTF-8: {error}') from error
    try:
        parsed = list(csv.reader(io.StringIO(text, newline=''), delimiter='|', strict=True))
    except csv.Error as error:
        raise RelevanceReviewError(f'{member} row {row_number} is not valid pipe-delimited data: {error}') from error
    if len(parsed) != 1:
        raise RelevanceReviewError(f'{member} row {row_number} decoded to {len(parsed)} records')
    return parsed[0]


def _find_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_name: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if path.is_absolute() or '..' in path.parts or path.as_posix() != info.filename:
            raise RelevanceReviewError(f'unsafe decision ZIP member path: {info.filename}')
        if mode and stat.S_ISLNK(mode):
            raise RelevanceReviewError(f'decision ZIP member cannot be a symbolic link: {info.filename}')
        if info.flag_bits & 0x1:
            raise RelevanceReviewError(f'decision ZIP member cannot be encrypted: {info.filename}')
        if not info.is_dir():
            by_name[PurePosixPath(info.filename).name].append(info)
    selected: dict[str, zipfile.ZipInfo] = {}
    for member in _REQUIRED_MEMBERS:
        matches = [info for info in by_name.get(member, ()) if info.filename == member]
        if len(matches) != 1:
            raise RelevanceReviewError(f'decision archive must contain one canonical {member}; found {len(matches)}')
        selected[member] = matches[0]
    return selected


def _scan_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    selected_nct_ids: set[str],
) -> tuple[ArchiveMemberEvidenceBinding, dict[str, list[_SelectedRow]]]:
    member = info.filename
    fields_read = tuple(sorted(_MEMBER_FIELDS[member]))
    rows_by_nct: dict[str, list[_SelectedRow]] = defaultdict(list)
    digest = hashlib.sha256()
    data_count = 0
    try:
        with archive.open(info) as source:
            records = _iter_raw_records(source)
            header = next(records, None)
            if header is None:
                raise RelevanceReviewError(f'decision member is empty: {member}')
            digest.update(header.raw)
            columns = tuple(_parse_values(header.raw, member, 0))
            if not columns or len(columns) != len(set(columns)):
                raise RelevanceReviewError(f'decision member has an invalid header: {member}')
            missing = sorted(set(fields_read) - set(columns))
            if missing:
                raise RelevanceReviewError(f'{member} is missing review fields: {missing}')
            for raw_record in records:
                digest.update(raw_record.raw)
                data_count += 1
                values = _parse_values(raw_record.raw, member, raw_record.data_row_number)
                if len(values) != len(columns):
                    raise RelevanceReviewError(
                        f'{member} row {raw_record.data_row_number} has {len(values)} fields; expected {len(columns)}'
                    )
                row = dict(zip(columns, values, strict=True))
                nct_id = row['nct_id'].strip()
                if nct_id in selected_nct_ids:
                    rows_by_nct[nct_id].append(
                        _SelectedRow(
                            values={field: row[field] for field in fields_read},
                            source=EvidenceSourceRow(
                                member_path=member,
                                data_row_number=raw_record.data_row_number,
                                raw_row_sha256=_sha256_bytes(raw_record.raw),
                                fields_read=fields_read,
                            ),
                        )
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        if isinstance(error, RelevanceReviewError):
            raise
        raise RelevanceReviewError(f'cannot read exact decision member {member}: {error}') from error
    selected_count = sum(len(rows) for rows in rows_by_nct.values())
    return (
        ArchiveMemberEvidenceBinding(
            member_path=member,
            member_sha256=digest.hexdigest(),
            crc32_hex=f'{info.CRC:08x}',
            data_row_count=data_count,
            selected_row_count=selected_count,
            fields_read=fields_read,
        ),
        rows_by_nct,
    )


def _single_value(rows: Sequence[_SelectedRow], field: str, member: str, nct_id: str) -> str:
    if len(rows) > 1:
        raise RelevanceReviewError(f'{nct_id} has duplicate {member} rows')
    return rows[0].values[field] if rows else ''


def _evidence_record(
    *,
    anchor_date: date,
    snapshot_id: str,
    archive_sha256: str,
    nct_id: str,
    selected: Mapping[str, Mapping[str, Sequence[_SelectedRow]]],
) -> DecisionEvidenceRecord:
    rows = {member: tuple(selected[member].get(nct_id, ())) for member in _REQUIRED_MEMBERS}
    studies = rows['studies.txt']
    if len(studies) != 1:
        raise RelevanceReviewError(f'{nct_id} requires exactly one historical studies.txt row')
    source_rows = tuple(
        sorted(
            (row.source for member_rows in rows.values() for row in member_rows),
            key=lambda item: (item.member_path, item.data_row_number),
        )
    )
    body = DecisionEvidenceBody(
        anchor_date=anchor_date,
        snapshot_id=snapshot_id,
        decision_archive_sha256=archive_sha256,
        nct_id=nct_id,
        brief_title=studies[0].values['brief_title'],
        official_title=studies[0].values['official_title'],
        acronym=studies[0].values['acronym'],
        primary_purposes=tuple(sorted(row.values['primary_purpose'] for row in rows['designs.txt'])),
        conditions=tuple(sorted(row.values['name'] for row in rows['conditions.txt'])),
        interventions=tuple(
            InterventionEvidence(
                intervention_type=row.values['intervention_type'],
                name=row.values['name'],
                description=row.values['description'],
            )
            for row in sorted(
                rows['interventions.txt'],
                key=lambda item: (
                    item.values['intervention_type'],
                    item.values['name'],
                    item.values['description'],
                    item.source.data_row_number,
                ),
            )
        ),
        brief_summary=_single_value(rows['brief_summaries.txt'], 'description', 'brief_summaries.txt', nct_id),
        detailed_description=_single_value(
            rows['detailed_descriptions.txt'], 'description', 'detailed_descriptions.txt', nct_id
        ),
        sponsors=tuple(
            SponsorEvidence(
                lead_or_collaborator=row.values['lead_or_collaborator'],
                agency_class=row.values['agency_class'],
                name=row.values['name'],
            )
            for row in sorted(
                rows['sponsors.txt'],
                key=lambda item: (
                    item.values['lead_or_collaborator'],
                    item.values['agency_class'],
                    item.values['name'],
                    item.source.data_row_number,
                ),
            )
        ),
        source_rows=source_rows,
    )
    return DecisionEvidenceRecord(**body.model_dump(), evidence_sha256=_model_sha256(body))


def _scan_decision_archive(
    *,
    path: Path,
    anchor_date: date,
    snapshot_id: str,
    expected_sha256: str,
    selected_nct_ids: set[str],
) -> tuple[DecisionArchiveEvidenceBinding, tuple[DecisionEvidenceRecord, ...]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RelevanceReviewError(f'decision archive cannot be a symbolic link: {expanded}')
    resolved = expanded.resolve()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise RelevanceReviewError(f'cannot open decision archive {resolved}: {error}') from error
    try:
        with os.fdopen(descriptor, 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RelevanceReviewError(f'decision archive is not a regular file: {resolved}')
            archive_sha256, archive_bytes = _hash_seekable_file(source)
            if archive_sha256 != expected_sha256:
                raise RelevanceReviewError(
                    f'{anchor_date} decision archive SHA-256 does not match the merged inventory'
                )
            with zipfile.ZipFile(source) as archive:
                infos = _find_members(archive)
                member_bindings: list[ArchiveMemberEvidenceBinding] = []
                selected: dict[str, dict[str, list[_SelectedRow]]] = {}
                for member in _REQUIRED_MEMBERS:
                    binding, rows = _scan_member(archive, infos[member], selected_nct_ids)
                    member_bindings.append(binding)
                    selected[member] = rows
            after_sha256, after_bytes = _hash_seekable_file(source)
            after = os.fstat(source.fileno())
            try:
                path_after = os.stat(resolved, follow_symlinks=False)
            except OSError as error:
                raise RelevanceReviewError(f'decision archive path changed during verification: {resolved}') from error
            if (
                after_sha256 != archive_sha256
                or after_bytes != archive_bytes
                or _stable_stat_fields(after) != _stable_stat_fields(before)
                or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise RelevanceReviewError(f'decision archive changed during verification: {resolved}')
    except zipfile.BadZipFile as error:
        raise RelevanceReviewError(f'invalid decision archive ZIP {resolved}: {error}') from error
    except OSError as error:
        raise RelevanceReviewError(f'cannot read stable decision archive {resolved}: {error}') from error
    records = tuple(
        _evidence_record(
            anchor_date=anchor_date,
            snapshot_id=snapshot_id,
            archive_sha256=archive_sha256,
            nct_id=nct_id,
            selected=selected,
        )
        for nct_id in sorted(selected_nct_ids)
    )
    return (
        DecisionArchiveEvidenceBinding(
            anchor_date=anchor_date,
            snapshot_id=snapshot_id,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            members=tuple(member_bindings),
        ),
        records,
    )


def build_relevance_review_queue(
    *,
    inventory: ExecutionCohortInventory,
    merged_inventory_sha256: str,
    decision_archives: Mapping[date, Path],
) -> VaccineRelevanceReviewQueue:
    """Build exact decision evidence for every mechanically assigned trial.

    ``decision_archives`` is keyed by the historical anchor date.  There is intentionally no later
    archive or label argument.
    """

    try:
        audit_execution_inventory(inventory)
    except ExecutionInventoryError as error:
        raise RelevanceReviewError(f'merged decision inventory fails deterministic audit: {error}') from error
    if re.fullmatch(r'[0-9a-f]{64}', merged_inventory_sha256) is None:
        raise RelevanceReviewError('merged inventory SHA-256 must be a 64-character digest')
    canonical_inventory = canonical_json_bytes(inventory)
    permitted_inventory_hashes = {
        _sha256_bytes(canonical_inventory),
        _sha256_bytes(canonical_inventory + b'\n'),
    }
    if merged_inventory_sha256 not in permitted_inventory_hashes:
        raise RelevanceReviewError('merged inventory SHA-256 does not match its canonical decision inventory')
    binding_by_date = {binding.anchor_date: binding for binding in inventory.policy.anchors}
    if set(decision_archives) != set(binding_by_date):
        raise RelevanceReviewError('decision archive inputs must exactly cover the merged inventory anchors')
    ids_by_date: dict[date, set[str]] = defaultdict(set)
    for assignment in inventory.assignments:
        ids_by_date[assignment.anchor_date].add(assignment.nct_id)
    source_archives: list[DecisionArchiveEvidenceBinding] = []
    records: list[DecisionEvidenceRecord] = []
    for anchor_date in sorted(binding_by_date):
        binding = binding_by_date[anchor_date]
        source_binding, anchor_records = _scan_decision_archive(
            path=decision_archives[anchor_date],
            anchor_date=anchor_date,
            snapshot_id=binding.decision_snapshot_id,
            expected_sha256=binding.decision_archive_manifest_sha256,
            selected_nct_ids=ids_by_date[anchor_date],
        )
        source_archives.append(source_binding)
        records.extend(anchor_records)
    if len(records) != len(inventory.assignments):
        raise RelevanceReviewError('review evidence does not cover every mechanical assignment exactly once')
    return VaccineRelevanceReviewQueue(
        policy_sha256=relevance_policy_sha256(),
        merged_inventory_sha256=merged_inventory_sha256,
        source_archives=tuple(source_archives),
        records=tuple(sorted(records, key=lambda item: (item.anchor_date, item.nct_id))),
        record_count=len(records),
    )


def finalize_relevance_adjudications(
    *,
    queue: VaccineRelevanceReviewQueue,
    reviews: Iterable[RelevanceReviewInput],
) -> VaccineRelevanceAdjudicationSet:
    validated = tuple(RelevanceReviewInput.model_validate(item) for item in reviews)
    ordered = tuple(sorted(validated, key=lambda item: (item.anchor_date, item.nct_id)))
    queue_by_key = {(record.anchor_date, record.nct_id): record for record in queue.records}
    review_by_key = {(review.anchor_date, review.nct_id): review for review in ordered}
    if len(review_by_key) != len(ordered) or set(review_by_key) != set(queue_by_key):
        raise RelevanceReviewError('reviews must cover every queue record exactly once')
    for key, review in review_by_key.items():
        if review.evidence_sha256 != queue_by_key[key].evidence_sha256:
            raise RelevanceReviewError(f'review evidence hash does not match the fixed queue for {review.nct_id}')
    return VaccineRelevanceAdjudicationSet(
        policy_sha256=queue.policy_sha256,
        review_queue_sha256=_model_sha256(queue),
        merged_inventory_sha256=queue.merged_inventory_sha256,
        decisions=ordered,
        include_count=sum(item.disposition == RelevanceDisposition.INCLUDE for item in ordered),
        exclude_count=sum(item.disposition == RelevanceDisposition.EXCLUDE for item in ordered),
        hold_count=sum(item.disposition == RelevanceDisposition.HOLD for item in ordered),
    )


def _write_artifact(path: Path, value: object, relative_path: str) -> ReviewArtifactReceipt:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
    return ReviewArtifactReceipt(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
    )


def write_relevance_review_build(
    *,
    queue: VaccineRelevanceReviewQueue,
    reviews: Iterable[RelevanceReviewInput],
    output_root: Path,
) -> VaccineRelevanceReviewReceipt:
    adjudications = finalize_relevance_adjudications(queue=queue, reviews=reviews)
    target = output_root.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.tmp-', dir=target.parent))
    try:
        artifacts = [
            _write_artifact(
                temporary / 'organizer' / 'relevance-adjudications.json',
                adjudications,
                'organizer/relevance-adjudications.json',
            ),
            _write_artifact(
                temporary / 'organizer' / 'relevance-policy.json',
                ACTIVE_VACCINE_RELEVANCE_POLICY,
                'organizer/relevance-policy.json',
            ),
            _write_artifact(
                temporary / 'organizer' / 'relevance-review-queue.json',
                queue,
                'organizer/relevance-review-queue.json',
            ),
        ]
        receipt = VaccineRelevanceReviewReceipt(
            policy_sha256=relevance_policy_sha256(),
            include_count=adjudications.include_count,
            exclude_count=adjudications.exclude_count,
            hold_count=adjudications.hold_count,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        )
        receipt_payload = canonical_json_bytes(receipt)
        with (temporary / 'REVIEW-RECEIPT.json').open('xb') as sink:
            sink.write(receipt_payload)
            sink.flush()
            os.fsync(sink.fileno())
        fsync_directory(temporary / 'organizer')
        fsync_directory(temporary)
        rename_directory_noreplace(temporary, target)
        fsync_directory(target.parent)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_relevance_review_build(
    root: Path,
    *,
    expected_receipt_sha256: str,
    trusted_inventory_path: Path,
    trusted_decision_archives: Mapping[date, Path],
) -> VaccineRelevanceReviewReceipt:
    """Verify a review against external, trusted source files.

    Receipt and artifact hashes alone establish only that the files agree with one another.  They
    cannot establish that the embedded title, source-row hashes, inventory digest, or archive
    digest came from AACT, and a newly issued receipt cannot authenticate a manual review decision.
    This verifier therefore requires both an externally pinned receipt digest and the separately
    acquired inventory and exact decision archives, then reconstructs the full queue before
    accepting the build.
    """

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise RelevanceReviewError('review build root cannot be a symbolic link')
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise RelevanceReviewError('review build root must be a regular directory')
    try:
        receipt_payload = (resolved / 'REVIEW-RECEIPT.json').read_bytes()
        if re.fullmatch(r'[0-9a-f]{64}', expected_receipt_sha256) is None:
            raise RelevanceReviewError('expected review receipt SHA-256 must be a 64-character digest')
        if _sha256_bytes(receipt_payload) != expected_receipt_sha256:
            raise RelevanceReviewError('review receipt does not match the externally pinned digest')
        receipt = VaccineRelevanceReviewReceipt.model_validate_json(receipt_payload)
        artifact_payloads: dict[str, bytes] = {}
        for artifact in receipt.artifacts:
            path = resolved / artifact.relative_path
            if path.is_symlink() or not path.is_file():
                raise RelevanceReviewError(f'missing regular review artifact: {artifact.relative_path}')
            payload = path.read_bytes()
            if len(payload) != artifact.byte_count or _sha256_bytes(payload) != artifact.sha256:
                raise RelevanceReviewError(f'review artifact does not match receipt: {artifact.relative_path}')
            artifact_payloads[artifact.relative_path] = payload
        queue = VaccineRelevanceReviewQueue.model_validate_json(
            artifact_payloads['organizer/relevance-review-queue.json']
        )
        adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(
            artifact_payloads['organizer/relevance-adjudications.json']
        )
    except (OSError, ValueError) as error:
        if isinstance(error, RelevanceReviewError):
            raise
        raise RelevanceReviewError(f'invalid relevance review build: {error}') from error
    rebuilt = finalize_relevance_adjudications(queue=queue, reviews=adjudications.decisions)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(adjudications):
        raise RelevanceReviewError('adjudications do not reconstruct from their queue')
    if (
        receipt.include_count,
        receipt.exclude_count,
        receipt.hold_count,
    ) != (adjudications.include_count, adjudications.exclude_count, adjudications.hold_count):
        raise RelevanceReviewError('receipt counts do not match relevance adjudications')
    policy_payload = artifact_payloads['organizer/relevance-policy.json']
    if _sha256_bytes(policy_payload) != relevance_policy_sha256():
        raise RelevanceReviewError('review build does not contain the fixed policy')

    inventory_expanded = trusted_inventory_path.expanduser()
    if inventory_expanded.is_symlink():
        raise RelevanceReviewError('trusted inventory cannot be a symbolic link')
    inventory_resolved = inventory_expanded.resolve()
    if not inventory_resolved.is_file():
        raise RelevanceReviewError('trusted inventory must be a regular file')
    try:
        inventory_payload = inventory_resolved.read_bytes()
        inventory = ExecutionCohortInventory.model_validate_json(inventory_payload)
    except (OSError, ValueError) as error:
        raise RelevanceReviewError(f'invalid trusted decision inventory: {error}') from error
    reconstructed_queue = build_relevance_review_queue(
        inventory=inventory,
        merged_inventory_sha256=_sha256_bytes(inventory_payload),
        decision_archives=trusted_decision_archives,
    )
    if canonical_json_bytes(reconstructed_queue) != canonical_json_bytes(queue):
        raise RelevanceReviewError('review queue does not reconstruct from the trusted inventory and archives')
    return receipt


def _date_path(value: str) -> tuple[date, Path]:
    try:
        raw_date, raw_path = value.split('=', 1)
        return date.fromisoformat(raw_date), Path(raw_path)
    except ValueError as error:
        raise argparse.ArgumentTypeError('decision archive must be DATE=PATH') from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build decision-only AACT active-vaccine relevance artifacts')
    subparsers = parser.add_subparsers(dest='command', required=True)
    queue = subparsers.add_parser('queue')
    queue.add_argument('--inventory', type=Path, required=True)
    queue.add_argument('--decision-archive', type=_date_path, action='append', required=True)
    queue.add_argument('--output', type=Path, required=True)
    finalize = subparsers.add_parser('finalize')
    finalize.add_argument('--queue', type=Path, required=True)
    finalize.add_argument('--reviews', type=Path, required=True)
    finalize.add_argument('--output-root', type=Path, required=True)
    verify = subparsers.add_parser('verify')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--expected-receipt-sha256', required=True)
    verify.add_argument('--inventory', type=Path, required=True)
    verify.add_argument('--decision-archive', type=_date_path, action='append', required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'queue':
        payload = args.inventory.read_bytes()
        inventory = ExecutionCohortInventory.model_validate_json(payload)
        archive_pairs = dict(args.decision_archive)
        if len(archive_pairs) != len(args.decision_archive):
            raise RelevanceReviewError('decision archive dates must be unique')
        queue = build_relevance_review_queue(
            inventory=inventory,
            merged_inventory_sha256=_sha256_bytes(payload),
            decision_archives=archive_pairs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as sink:
            sink.write(canonical_json_bytes(queue))
        return 0
    if args.command == 'finalize':
        queue = VaccineRelevanceReviewQueue.model_validate_json(args.queue.read_bytes())
        reviews_payload = RelevanceReviewInputList.model_validate_json(args.reviews.read_bytes())
        write_relevance_review_build(queue=queue, reviews=reviews_payload.reviews, output_root=args.output_root)
        return 0
    archive_pairs = dict(args.decision_archive)
    if len(archive_pairs) != len(args.decision_archive):
        raise RelevanceReviewError('decision archive dates must be unique')
    verify_relevance_review_build(
        args.root,
        expected_receipt_sha256=args.expected_receipt_sha256,
        trusted_inventory_path=args.inventory,
        trusted_decision_archives=archive_pairs,
    )
    return 0


class RelevanceReviewInputList(StrictModel):
    schema_version: Literal['vaxreplay.aact-vaccine-relevance-review-input.v0.1'] = (
        'vaxreplay.aact-vaccine-relevance-review-input.v0.1'
    )
    reviews: tuple[RelevanceReviewInput, ...] = Field(min_length=1)


if __name__ == '__main__':
    raise SystemExit(main())
