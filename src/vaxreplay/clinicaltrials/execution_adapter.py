"""Organizer-only adapter from exact AACT ZIPs to registry-observed execution gold.

The adapter deliberately has two phases.  It fixes and audits the provisional decision inventory from the
historical decision archive before it opens or parses the later archive.  Only then does it read
the later ``studies.txt`` member and derive the private +48-month labels.  The resulting artifact
is a high-recall inventory for adjudication, not a scored cohort, a Tier-B/Tier-A release, a
split-safe dataset, or a measure of efficacy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_inventory import (
    audit_execution_inventory,
    audit_execution_label_set,
    bind_anchor_source,
    build_execution_inventory,
    derive_execution_labels,
)
from vaxreplay.clinicaltrials.execution_schema import (
    EXECUTION_TASK_ID,
    EXECUTION_TASK_SEMANTICS,
    AactExecutionDecisionRow,
    AactExecutionOutcomeRow,
    DiseaseStratum,
    ExecutionCohortInventory,
    ExecutionCohortPolicy,
    ExecutionLabelSet,
    ExecutionTaskSemantics,
    NormalizedPhase,
    NormalizedStudyType,
    RegistryStatus,
    RegistryValueType,
    add_calendar_months,
)
from vaxreplay.clinicaltrials.inventory_catalog import (
    AactCatalogIntegrityError,
    verify_archive_acquisition_plan,
)
from vaxreplay.clinicaltrials.inventory_schema import (
    AactArchiveAcquisitionPlan,
    AactArchiveAcquisitionRole,
    AactOfficialArchiveCatalog,
    aact_inventory_model_sha256,
)
from vaxreplay.clinicaltrials.schema import AactArchiveReceipt

ADAPTER_SCHEMA_VERSION = 'vaxreplay.aact-execution-adapter-receipt.v0.1'
ADAPTER_ID = 'aact-flatfile-registry-observed-execution-v0.1'
SELECTION_UNIVERSE_RULE_ID = 'aact-high-recall-vaccine-screen-v0.1'
LINEAGE_GROUPING_RULE_ID = 'aact-precutoff-exact-sponsor-product-lineage-v0.1'

_DECISION_MEMBERS = (
    'conditions.txt',
    'designs.txt',
    'interventions.txt',
    'keywords.txt',
    'sponsors.txt',
    'studies.txt',
)
_LABEL_MEMBERS = ('studies.txt',)

_FIELDS_READ: dict[str, tuple[str, ...]] = {
    'studies.txt': (
        'acronym',
        'brief_title',
        'enrollment',
        'enrollment_type',
        'nct_id',
        'official_title',
        'overall_status',
        'phase',
        'primary_completion_date',
        'primary_completion_date_type',
        'results_first_posted_date',
        'study_first_posted_date',
        'study_type',
    ),
    'interventions.txt': ('description', 'intervention_type', 'name', 'nct_id'),
    'conditions.txt': ('name', 'nct_id'),
    'keywords.txt': ('name', 'nct_id'),
    'sponsors.txt': ('agency_class', 'lead_or_collaborator', 'name', 'nct_id'),
    'designs.txt': ('nct_id', 'primary_purpose'),
}

_VACCINE_PATTERN = re.compile(
    r'\b(?:vaccin\w*|immuni[sz](?:ation|e|ing)\w*|immunogen(?:ic|icity)?\w*)\b',
    re.IGNORECASE,
)
_COVID_PATTERN = re.compile(
    r'\b(?:covid(?:-?19)?|sars[- ]?cov[- ]?2|2019[- ]?ncov|novel coronavirus)\b',
    re.IGNORECASE,
)
_INFECTIOUS_PATTERN = re.compile(
    r'\b(?:'
    r'infect(?:ion|ious|ive)\w*|pathogen\w*|bacteri\w*|vir(?:al|us|uses)\w*|'
    r'covid(?:-?19)?|coronavirus\w*|sars\w*|mers\w*|influenza\w*|flu|hiv|aids|'
    r'hepatitis\w*|dengue\w*|zika\w*|chikungunya\w*|malaria\w*|tuberculosis|'
    r'pertussis|pneumococ\w*|meningococ\w*|rotavirus\w*|norovirus\w*|'
    r'respiratory syncytial|rsv|measles|mumps|rubella|varicella|herpes\w*|'
    r'hpv|papilloma\w*|rabies|ebola|marburg|yellow fever|typhoid|cholera|'
    r'shigell\w*|salmonell\w*|clostrid\w*|cytomegalovirus|cmv|epstein[- ]barr|ebv|'
    r'mpox|monkeypox|smallpox|polio\w*|encephalitis|west nile|tick[- ]borne|lyme|'
    r'anthrax|plague'
    r')\b',
    re.IGNORECASE,
)

_SELECTION_RULE = {
    'rule_id': SELECTION_UNIVERSE_RULE_ID,
    'archive_side': 'decision_only',
    'screen_if': [
        'vaccine_or_immunization_lexical_signal_in_study_intervention_condition_or_keyword',
        'interventional_early_phase_record_with_at_least_one_biological_intervention',
    ],
    'early_phases': ['early_phase_1', 'phase_1', 'phase_1_phase_2'],
    'eligibility': 'execution_schema.aact-fixed-anchor-pre-results-phase1-vaccine-v0.1',
    'later_archive_fields_used': [],
}
# This checked-in rule is a public, contamination-exposed reference implementation. It must not be
# reused as the undisclosed selection policy for a held-out or commercial cohort.
_LINEAGE_RULE = {
    'rule_id': LINEAGE_GROUPING_RULE_ID,
    'archive_side': 'decision_only',
    'provisional_identity_key': ['normalized_lead_sponsor_names', 'normalized_biological_intervention_names'],
    'normalization': 'NFKC-casefold-alphanumeric-token-v1',
    'covid_stratum_separate': True,
    'missing_identity': 'singleton-needs-review',
    'all_assignments_require_manual_lineage_review': True,
    'split_safe': False,
}


class AactExecutionAdapterError(ValueError):
    """Raised when an archive or derived execution cohort fails closed."""


class ExactMemberReceipt(StrictModel):
    member_path: str = Field(min_length=1)
    member_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    uncompressed_bytes: int = Field(gt=0)
    compressed_bytes: int = Field(ge=0)
    crc32_hex: str = Field(pattern=r'^[0-9a-f]{8}$')
    header_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    columns: tuple[str, ...] = Field(min_length=1)
    data_row_count: int = Field(ge=0)
    exact_member_crc_verified: Literal[True] = True

    @model_validator(mode='after')
    def validate_columns(self) -> Self:
        if len(self.columns) != len(set(self.columns)):
            raise ValueError('AACT member columns must be unique')
        return self


class ExactArchiveReceipt(StrictModel):
    snapshot_id: str = Field(min_length=1)
    archive_date: date
    archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    archive_bytes: int = Field(gt=0)
    full_archive_bytes_hashed: Literal[True] = True
    zip_central_directory_valid: Literal[True] = True
    relevant_member_inventory_complete: Literal[True] = True
    members: tuple[ExactMemberReceipt, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_members(self) -> Self:
        paths = tuple(member.member_path for member in self.members)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('archive members must be unique and sorted by path')
        return self


class SourceRowProvenance(StrictModel):
    snapshot_id: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    member_path: str = Field(min_length=1)
    member_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    data_row_number: int = Field(gt=0)
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    raw_row_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    nct_id: str = Field(min_length=1)
    fields_read: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_provenance(self) -> Self:
        if self.byte_end <= self.byte_start:
            raise ValueError('source row byte range must be non-empty')
        if self.fields_read != tuple(sorted(self.fields_read)) or len(self.fields_read) != len(set(self.fields_read)):
            raise ValueError('fields_read must be unique and sorted')
        return self


class ScreenedDecisionRecord(StrictModel):
    nct_id: str = Field(min_length=1)
    discovery_signals: tuple[str, ...] = Field(min_length=1)
    normalized: bool
    normalization_exclusion_reasons: tuple[str, ...]
    normalization_notes: tuple[str, ...]
    normalized_decision_row_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    classification_basis_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_rows: tuple[SourceRowProvenance, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        for values, name in (
            (self.discovery_signals, 'discovery_signals'),
            (self.normalization_exclusion_reasons, 'normalization_exclusion_reasons'),
            (self.normalization_notes, 'normalization_notes'),
        ):
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f'{name} must be unique and sorted')
        if self.normalized:
            if self.normalization_exclusion_reasons or self.normalized_decision_row_sha256 is None:
                raise ValueError('normalized records require a row hash and no exclusion reasons')
        elif not self.normalization_exclusion_reasons or self.normalized_decision_row_sha256 is not None:
            raise ValueError('excluded records require reasons and cannot carry a normalized row hash')
        row_keys = tuple((row.member_path, row.data_row_number) for row in self.source_rows)
        if row_keys != tuple(sorted(row_keys)) or len(row_keys) != len(set(row_keys)):
            raise ValueError('source rows must be unique and sorted')
        return self


class LineageAssignmentReceipt(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    disease_stratum: DiseaseStratum
    lineage_group_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    identity_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    normalized_lead_sponsors: tuple[str, ...]
    normalized_product_names: tuple[str, ...]
    review_status: Literal[
        'needs_manual_lineage_review_exact_identity',
        'needs_review_missing_lead_sponsor',
        'needs_review_missing_product_identity',
        'needs_review_missing_sponsor_and_product_identity',
    ]


class OutcomeSourceRecord(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    record_present: bool
    source_row: SourceRowProvenance | None
    raw_overall_status: str
    raw_enrollment: str
    raw_enrollment_type: str
    raw_primary_completion_date: str
    raw_primary_completion_date_type: str
    normalization_notes: tuple[str, ...]

    @model_validator(mode='after')
    def validate_presence(self) -> Self:
        raw_values = (
            self.raw_overall_status,
            self.raw_enrollment,
            self.raw_enrollment_type,
            self.raw_primary_completion_date,
            self.raw_primary_completion_date_type,
        )
        if self.record_present != (self.source_row is not None):
            raise ValueError('outcome source presence must match its exact row provenance')
        if not self.record_present and any(raw_values):
            raise ValueError('missing outcome records cannot carry raw values')
        if self.normalization_notes != tuple(sorted(self.normalization_notes)) or len(self.normalization_notes) != len(
            set(self.normalization_notes)
        ):
            raise ValueError('normalization notes must be unique and sorted')
        return self


class ScreenedDecisionRecordSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-screened-decision-records.v0.1'] = (
        'vaxreplay.aact-screened-decision-records.v0.1'
    )
    records: tuple[ScreenedDecisionRecord, ...]


class LineageAssignmentSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-lineage-assignments.v0.1'] = 'vaxreplay.aact-lineage-assignments.v0.1'
    assignments: tuple[LineageAssignmentReceipt, ...]


class OutcomeSourceRecordSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-outcome-source-records.v0.1'] = 'vaxreplay.aact-outcome-source-records.v0.1'
    records: tuple[OutcomeSourceRecord, ...]


class TrustedAactSourceBinding(StrictModel):
    """Organizer-supplied authority binding for real-data mode.

    The official catalog proves that the requested dates are permanent AACT routes, the acquisition
    plan gives the two snapshots explicit decision/label roles, and the exact archive receipts bind
    those catalog entries to local byte counts and SHA-256 values.
    """

    catalog: AactOfficialArchiveCatalog
    acquisition_plan: AactArchiveAcquisitionPlan
    decision_archive_receipt: AactArchiveReceipt
    label_archive_receipt: AactArchiveReceipt


class ExecutionSourceBindingReceipt(StrictModel):
    mode: Literal['trusted_official_real', 'synthetic_test_only']
    catalog_id: str | None = None
    catalog_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    acquisition_plan_id: str | None = None
    acquisition_plan_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    decision_catalog_entry_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    label_catalog_entry_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    decision_acquisition_item_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    label_acquisition_item_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    decision_archive_receipt: AactArchiveReceipt | None = None
    label_archive_receipt: AactArchiveReceipt | None = None

    @model_validator(mode='after')
    def validate_mode(self) -> Self:
        trusted_values = (
            self.catalog_id,
            self.catalog_sha256,
            self.acquisition_plan_id,
            self.acquisition_plan_sha256,
            self.decision_catalog_entry_sha256,
            self.label_catalog_entry_sha256,
            self.decision_acquisition_item_sha256,
            self.label_acquisition_item_sha256,
            self.decision_archive_receipt,
            self.label_archive_receipt,
        )
        if self.mode == 'trusted_official_real' and any(value is None for value in trusted_values):
            raise ValueError('trusted real mode requires complete catalog, plan, entry, item, and archive bindings')
        if self.mode == 'synthetic_test_only' and any(value is not None for value in trusted_values):
            raise ValueError('synthetic test mode cannot carry or imply an official source binding')
        return self


class ArtifactReceipt(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)
    organizer_private: Literal[True] = True


class AactExecutionBuildReceipt(StrictModel):
    schema_version: Literal['vaxreplay.aact-execution-adapter-receipt.v0.1'] = ADAPTER_SCHEMA_VERSION
    adapter_id: Literal['aact-flatfile-registry-observed-execution-v0.1'] = ADAPTER_ID
    task_id: Literal['registry_observed_trial_execution'] = EXECUTION_TASK_ID
    task_semantics: ExecutionTaskSemantics = EXECUTION_TASK_SEMANTICS
    synthetic: bool
    release_status: Literal['provisional_high_recall_inventory', 'synthetic_test_only']
    tier_b_admitted: Literal[False] = False
    tier_a_official: Literal[False] = False
    biological_efficacy_claimed: Literal[False] = False
    mechanical_eligibility_only: Literal[True] = True
    active_vaccination_adjudication_bound: Literal[False] = False
    scored_cohort_eligible: Literal[False] = False
    manual_lineage_review_required: Literal[True] = True
    lineage_split_safe: Literal[False] = False
    selection_frozen_before_label_studies_member_opened: Literal[True] = True
    source_binding: ExecutionSourceBindingReceipt
    decision_archive: ExactArchiveReceipt
    label_archive: ExactArchiveReceipt
    selection_universe_rule_id: Literal['aact-high-recall-vaccine-screen-v0.1'] = SELECTION_UNIVERSE_RULE_ID
    selection_universe_rule_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    lineage_grouping_rule_id: Literal['aact-precutoff-exact-sponsor-product-lineage-v0.1'] = LINEAGE_GROUPING_RULE_ID
    lineage_grouping_rule_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    screened_record_count: int = Field(ge=0)
    normalized_record_count: int = Field(ge=0)
    normalization_exclusion_count: int = Field(ge=0)
    assigned_trial_count: int = Field(ge=0)
    covid_assigned_trial_count: int = Field(ge=0)
    non_covid_assigned_trial_count: int = Field(ge=0)
    missing_label_record_count: int = Field(ge=0)
    artifacts: tuple[ArtifactReceipt, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_counts(self) -> Self:
        if self.source_binding.mode == 'trusted_official_real':
            if self.synthetic or self.release_status != 'provisional_high_recall_inventory':
                raise ValueError('trusted official mode must be labeled as a provisional high-recall inventory')
        elif not self.synthetic or self.release_status != 'synthetic_test_only':
            raise ValueError('untrusted test mode must be labeled synthetic_test_only')
        if self.normalized_record_count + self.normalization_exclusion_count != self.screened_record_count:
            raise ValueError('screened disposition counts do not add up')
        if self.covid_assigned_trial_count + self.non_covid_assigned_trial_count != self.assigned_trial_count:
            raise ValueError('disease-stratum counts do not add up')
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('artifact receipts must be unique and sorted by path')
        return self


@dataclass(frozen=True)
class AactExecutionBuild:
    root: Path
    receipt: AactExecutionBuildReceipt
    inventory: ExecutionCohortInventory
    labels: ExecutionLabelSet


@dataclass(frozen=True)
class _ArchiveFingerprint:
    path: Path
    snapshot_id: str
    archive_date: date
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _RawRecord:
    data_row_number: int
    byte_start: int
    byte_end: int
    raw: bytes


@dataclass(frozen=True)
class _CapturedRow:
    values: dict[str, str]
    provenance: SourceRowProvenance


@dataclass
class _ExtractedTable:
    member_path: str
    path: Path
    member_sha256: str
    uncompressed_bytes: int
    compressed_bytes: int
    crc32_hex: str
    header_raw: bytes
    columns: tuple[str, ...]
    archive: _ArchiveFingerprint
    data_row_count: int | None = None

    def rows(self) -> Iterator[_CapturedRow]:
        for raw_record in _iter_data_records(self.path):
            values = _parse_record(raw_record.raw, self.columns, self.member_path, raw_record.data_row_number)
            nct_id = values.get('nct_id', '').strip()
            yield _CapturedRow(
                values=values,
                provenance=SourceRowProvenance(
                    snapshot_id=self.archive.snapshot_id,
                    archive_sha256=self.archive.sha256,
                    member_path=self.member_path,
                    member_sha256=self.member_sha256,
                    data_row_number=raw_record.data_row_number,
                    byte_start=raw_record.byte_start,
                    byte_end=raw_record.byte_end,
                    raw_row_sha256=_sha256_bytes(raw_record.raw),
                    nct_id=nct_id or '<missing>',
                    fields_read=tuple(sorted(_FIELDS_READ[self.member_path])),
                ),
            )

    def receipt(self) -> ExactMemberReceipt:
        if self.data_row_count is None:
            raise AactExecutionAdapterError(f'member was not fully scanned: {self.member_path}')
        return ExactMemberReceipt(
            member_path=self.member_path,
            member_sha256=self.member_sha256,
            uncompressed_bytes=self.uncompressed_bytes,
            compressed_bytes=self.compressed_bytes,
            crc32_hex=self.crc32_hex,
            header_sha256=_sha256_bytes(self.header_raw),
            columns=self.columns,
            data_row_count=self.data_row_count,
        )


@dataclass(frozen=True)
class _DecisionMaterial:
    inventory: ExecutionCohortInventory
    archive_receipt: ExactArchiveReceipt
    screened_records: tuple[ScreenedDecisionRecord, ...]
    lineage: tuple[LineageAssignmentReceipt, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_regular_file(path: Path, snapshot_id: str, archive_date: date) -> _ArchiveFingerprint:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AactExecutionAdapterError(f'AACT archive cannot be a symbolic link: {expanded}')
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise AactExecutionAdapterError(f'AACT archive must be a regular file: {resolved}')
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise AactExecutionAdapterError(f'cannot read AACT archive {resolved}: {error}') from error
    if byte_count <= 0:
        raise AactExecutionAdapterError(f'AACT archive is empty: {resolved}')
    return _ArchiveFingerprint(
        path=resolved,
        snapshot_id=snapshot_id,
        archive_date=archive_date,
        sha256=digest.hexdigest(),
        byte_count=byte_count,
    )


def _verify_trusted_source_binding(
    *,
    binding: TrustedAactSourceBinding,
    decision_fingerprint: _ArchiveFingerprint,
    label_fingerprint: _ArchiveFingerprint,
) -> ExecutionSourceBindingReceipt:
    try:
        verify_archive_acquisition_plan(binding.acquisition_plan, binding.catalog)
    except AactCatalogIntegrityError as error:
        raise AactExecutionAdapterError(f'untrusted AACT acquisition binding: {error}') from error

    entries = {entry.snapshot_id: entry for entry in binding.catalog.entries}
    items = {item.snapshot_id: item for item in binding.acquisition_plan.items}
    fingerprints_and_receipts = (
        ('decision', decision_fingerprint, binding.decision_archive_receipt),
        ('label', label_fingerprint, binding.label_archive_receipt),
    )
    selected_entries = {}
    selected_items = {}
    for role_name, fingerprint, archive_receipt in fingerprints_and_receipts:
        entry = entries.get(fingerprint.snapshot_id)
        item = items.get(fingerprint.snapshot_id)
        if entry is None or item is None:
            raise AactExecutionAdapterError(
                f'{role_name} snapshot is absent from the bound official catalog or acquisition plan'
            )
        required_roles = (
            {AactArchiveAcquisitionRole.DECISION_CANDIDATE, AactArchiveAcquisitionRole.DISCOVERY}
            if role_name == 'decision'
            else {AactArchiveAcquisitionRole.LABEL_CANDIDATE}
        )
        if required_roles.isdisjoint(item.roles):
            raise AactExecutionAdapterError(f'{role_name} snapshot lacks its required acquisition-plan role')
        if archive_receipt.snapshot_id != fingerprint.snapshot_id:
            raise AactExecutionAdapterError(f'{role_name} archive receipt snapshot ID does not match the build')
        if archive_receipt.archive_date != fingerprint.archive_date or entry.archive_date != fingerprint.archive_date:
            raise AactExecutionAdapterError(f'{role_name} archive date does not match its receipt and catalog entry')
        if archive_receipt.source_cutoff_at != entry.source_cutoff_at:
            raise AactExecutionAdapterError(f'{role_name} source cutoff differs from the official catalog entry')
        if archive_receipt.source_url != entry.source_url:
            raise AactExecutionAdapterError(f'{role_name} source URL differs from the official permanent-archive route')
        if archive_receipt.archive_sha256 != fingerprint.sha256:
            raise AactExecutionAdapterError(f'{role_name} archive SHA-256 does not match its trusted receipt')
        if archive_receipt.archive_bytes != fingerprint.byte_count:
            raise AactExecutionAdapterError(f'{role_name} archive size does not match its trusted receipt')
        selected_entries[role_name] = entry
        selected_items[role_name] = item

    return ExecutionSourceBindingReceipt(
        mode='trusted_official_real',
        catalog_id=binding.catalog.catalog_id,
        catalog_sha256=aact_inventory_model_sha256(binding.catalog),
        acquisition_plan_id=binding.acquisition_plan.plan_id,
        acquisition_plan_sha256=aact_inventory_model_sha256(binding.acquisition_plan),
        decision_catalog_entry_sha256=aact_inventory_model_sha256(selected_entries['decision']),
        label_catalog_entry_sha256=aact_inventory_model_sha256(selected_entries['label']),
        decision_acquisition_item_sha256=aact_inventory_model_sha256(selected_items['decision']),
        label_acquisition_item_sha256=aact_inventory_model_sha256(selected_items['label']),
        decision_archive_receipt=binding.decision_archive_receipt,
        label_archive_receipt=binding.label_archive_receipt,
    )


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != info.filename:
        raise AactExecutionAdapterError(f'unsafe AACT ZIP member path: {info.filename}')
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise AactExecutionAdapterError(f'AACT ZIP member cannot be a symbolic link: {info.filename}')
    if info.flag_bits & 0x1:
        raise AactExecutionAdapterError(f'AACT ZIP member cannot be encrypted: {info.filename}')


def _find_required_members(
    archive: zipfile.ZipFile,
    required_basenames: Sequence[str],
) -> dict[str, zipfile.ZipInfo]:
    by_basename: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        _validate_zip_member(info)
        if not info.is_dir():
            by_basename[PurePosixPath(info.filename).name].append(info)
    result: dict[str, zipfile.ZipInfo] = {}
    for basename in required_basenames:
        matches = [info for info in by_basename.get(basename, []) if info.filename == basename]
        if len(matches) != 1:
            raise AactExecutionAdapterError(
                f'AACT archive must contain exactly one canonical top-level {basename}; found {len(matches)}'
            )
        result[basename] = matches[0]
    return result


def _extract_tables(
    fingerprint: _ArchiveFingerprint,
    required_basenames: Sequence[str],
    destination: Path,
) -> dict[str, _ExtractedTable]:
    try:
        with zipfile.ZipFile(fingerprint.path) as archive:
            members = _find_required_members(archive, required_basenames)
            tables: dict[str, _ExtractedTable] = {}
            for basename in sorted(members):
                info = members[basename]
                target = destination / basename
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(info) as source, target.open('xb') as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                if byte_count != info.file_size:
                    raise AactExecutionAdapterError(f'uncompressed size mismatch for {info.filename}')
                header_record = next(_iter_raw_records(target), None)
                if header_record is None:
                    raise AactExecutionAdapterError(f'AACT member is empty: {info.filename}')
                columns = _parse_header(header_record.raw, info.filename)
                missing = sorted(set(_FIELDS_READ[basename]) - set(columns))
                if missing:
                    raise AactExecutionAdapterError(f'{info.filename} lacks required columns: {missing}')
                tables[basename] = _ExtractedTable(
                    member_path=info.filename,
                    path=target,
                    member_sha256=digest.hexdigest(),
                    uncompressed_bytes=byte_count,
                    compressed_bytes=info.compress_size,
                    crc32_hex=f'{info.CRC:08x}',
                    header_raw=header_record.raw,
                    columns=columns,
                    archive=fingerprint,
                )
            return tables
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, AactExecutionAdapterError):
            raise
        raise AactExecutionAdapterError(f'cannot verify/extract AACT archive {fingerprint.path}: {error}') from error


def _iter_raw_records(path: Path) -> Iterator[_RawRecord]:
    offset = 0
    start = 0
    data_row_number = -1
    buffered = bytearray()
    in_quotes = False
    try:
        with path.open('rb') as source:
            for physical_line in source:
                buffered.extend(physical_line)
                offset += len(physical_line)
                if physical_line.count(b'"') % 2:
                    in_quotes = not in_quotes
                if in_quotes:
                    continue
                data_row_number += 1
                yield _RawRecord(
                    data_row_number=data_row_number,
                    byte_start=start,
                    byte_end=offset,
                    raw=bytes(buffered),
                )
                buffered.clear()
                start = offset
    except OSError as error:
        raise AactExecutionAdapterError(f'cannot parse extracted AACT member {path}: {error}') from error
    if in_quotes:
        raise AactExecutionAdapterError(f'unterminated quoted field in {path.name}')
    if buffered:
        data_row_number += 1
        yield _RawRecord(
            data_row_number=data_row_number,
            byte_start=start,
            byte_end=offset,
            raw=bytes(buffered),
        )


def _iter_data_records(path: Path) -> Iterator[_RawRecord]:
    records = _iter_raw_records(path)
    if next(records, None) is None:
        raise AactExecutionAdapterError(f'AACT member is empty: {path}')
    yield from records


def _parse_csv_values(raw: bytes, member_path: str, row_label: str) -> list[str]:
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise AactExecutionAdapterError(f'{member_path} {row_label} is not UTF-8: {error}') from error
    try:
        rows = list(csv.reader(io.StringIO(text, newline=''), delimiter='|', strict=True))
    except csv.Error as error:
        raise AactExecutionAdapterError(
            f'invalid pipe-delimited record in {member_path} {row_label}: {error}'
        ) from error
    if len(rows) != 1:
        raise AactExecutionAdapterError(f'{member_path} {row_label} decoded to {len(rows)} records')
    return rows[0]


def _parse_header(raw: bytes, member_path: str) -> tuple[str, ...]:
    columns = tuple(_parse_csv_values(raw, member_path, 'header'))
    if not columns or any(not column for column in columns) or len(columns) != len(set(columns)):
        raise AactExecutionAdapterError(f'{member_path} has an invalid or duplicate header')
    return columns


def _parse_record(raw: bytes, columns: tuple[str, ...], member_path: str, row_number: int) -> dict[str, str]:
    values = _parse_csv_values(raw, member_path, f'data row {row_number}')
    if len(values) != len(columns):
        raise AactExecutionAdapterError(
            f'{member_path} data row {row_number} has {len(values)} fields; expected {len(columns)}'
        )
    return dict(zip(columns, values, strict=True))


def _normalize_identity(value: str) -> str:
    normalized = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+', normalized))


def _text(values: Sequence[str]) -> str:
    return ' '.join(value for value in values if value)


def _vaccine_signal(value: str) -> bool:
    return _VACCINE_PATTERN.search(value) is not None


def _normalize_phase(value: str) -> NormalizedPhase:
    normalized = _normalize_identity(value)
    mapping = {
        'early phase 1': NormalizedPhase.EARLY_PHASE_1,
        'phase 1': NormalizedPhase.PHASE_1,
        'phase 1 phase 2': NormalizedPhase.PHASE_1_PHASE_2,
    }
    return mapping.get(normalized, NormalizedPhase.OTHER)


def _normalize_study_type(value: str) -> NormalizedStudyType:
    return (
        NormalizedStudyType.INTERVENTIONAL
        if _normalize_identity(value) == 'interventional'
        else NormalizedStudyType.OTHER
    )


def _normalize_status(value: str) -> RegistryStatus:
    normalized = _normalize_identity(value)
    mapping = {
        'not yet recruiting': RegistryStatus.NOT_YET_RECRUITING,
        'recruiting': RegistryStatus.RECRUITING,
        'enrolling by invitation': RegistryStatus.ENROLLING_BY_INVITATION,
        'active not recruiting': RegistryStatus.ACTIVE_NOT_RECRUITING,
        'completed': RegistryStatus.COMPLETED,
        'terminated': RegistryStatus.TERMINATED,
        'withdrawn': RegistryStatus.WITHDRAWN,
        'suspended': RegistryStatus.SUSPENDED,
        'unknown status': RegistryStatus.UNKNOWN,
        'unknown': RegistryStatus.UNKNOWN,
    }
    return mapping.get(normalized, RegistryStatus.OTHER)


def _normalize_optional_status(value: str) -> RegistryStatus | None:
    return _normalize_status(value) if value.strip() else None


def _normalize_value_type(value: str) -> RegistryValueType | None:
    normalized = _normalize_identity(value)
    if normalized == 'anticipated':
        return RegistryValueType.ANTICIPATED
    if normalized == 'actual':
        return RegistryValueType.ACTUAL
    return None


def _parse_date(value: str) -> date | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return date.fromisoformat(stripped[:10])
    except ValueError:
        return None


def _normalize_enrollment_pair(
    raw_value: str, raw_type: str, prefix: str
) -> tuple[int | None, RegistryValueType | None, set[str]]:
    notes: set[str] = set()
    value_type = _normalize_value_type(raw_type)
    value: int | None = None
    if raw_value.strip():
        try:
            parsed = float(raw_value)
            if not parsed.is_integer() or parsed < 0:
                raise ValueError
            value = int(parsed)
        except ValueError:
            notes.add(f'{prefix}_value_unparseable')
    if raw_type.strip() and value_type is None:
        notes.add(f'{prefix}_type_unrecognized')
    if (value is None) != (value_type is None):
        notes.add(f'{prefix}_value_type_incomplete')
        return None, None, notes
    return value, value_type, notes


def _normalize_date_pair(
    raw_value: str,
    raw_type: str,
    prefix: str,
) -> tuple[date | None, RegistryValueType | None, set[str]]:
    notes: set[str] = set()
    value = _parse_date(raw_value)
    value_type = _normalize_value_type(raw_type)
    if raw_value.strip() and value is None:
        notes.add(f'{prefix}_value_unparseable')
    if raw_type.strip() and value_type is None:
        notes.add(f'{prefix}_type_unrecognized')
    if (value is None) != (value_type is None):
        notes.add(f'{prefix}_value_type_incomplete')
        return None, None, notes
    return value, value_type, notes


def _source_record_sha256(rows: Sequence[SourceRowProvenance]) -> str:
    material = [
        {
            'member_path': row.member_path,
            'data_row_number': row.data_row_number,
            'raw_row_sha256': row.raw_row_sha256,
            'fields_read': row.fields_read,
        }
        for row in sorted(rows, key=lambda item: (item.member_path, item.data_row_number))
    ]
    return _sha256_bytes(canonical_json_bytes(material))


def _archive_receipt(
    fingerprint: _ArchiveFingerprint,
    tables: dict[str, _ExtractedTable],
) -> ExactArchiveReceipt:
    return ExactArchiveReceipt(
        snapshot_id=fingerprint.snapshot_id,
        archive_date=fingerprint.archive_date,
        archive_sha256=fingerprint.sha256,
        archive_bytes=fingerprint.byte_count,
        members=tuple(sorted((table.receipt() for table in tables.values()), key=lambda item: item.member_path)),
    )


def _scan_decision_universe(
    tables: dict[str, _ExtractedTable],
) -> tuple[set[str], dict[str, set[str]]]:
    early_interventional: set[str] = set()
    biological: set[str] = set()
    signals: dict[str, set[str]] = defaultdict(set)

    studies_count = 0
    for captured in tables['studies.txt'].rows():
        studies_count += 1
        row = captured.values
        nct_id = row['nct_id'].strip()
        if not nct_id:
            continue
        if _normalize_study_type(row['study_type']) == NormalizedStudyType.INTERVENTIONAL and _normalize_phase(
            row['phase']
        ) in {NormalizedPhase.EARLY_PHASE_1, NormalizedPhase.PHASE_1, NormalizedPhase.PHASE_1_PHASE_2}:
            early_interventional.add(nct_id)
        if _vaccine_signal(_text((row['brief_title'], row['official_title'], row['acronym']))):
            signals[nct_id].add('study_text_vaccine_lexical')
    tables['studies.txt'].data_row_count = studies_count

    interventions_count = 0
    for captured in tables['interventions.txt'].rows():
        interventions_count += 1
        row = captured.values
        nct_id = row['nct_id'].strip()
        if not nct_id:
            continue
        if _normalize_identity(row['intervention_type']) == 'biological':
            biological.add(nct_id)
        if _vaccine_signal(_text((row['name'], row['description']))):
            signals[nct_id].add('intervention_text_vaccine_lexical')
    tables['interventions.txt'].data_row_count = interventions_count

    for member, signal_name in (
        ('conditions.txt', 'condition_text_vaccine_lexical'),
        ('keywords.txt', 'keyword_text_vaccine_lexical'),
    ):
        count = 0
        for captured in tables[member].rows():
            count += 1
            row = captured.values
            nct_id = row['nct_id'].strip()
            if nct_id and _vaccine_signal(row['name']):
                signals[nct_id].add(signal_name)
        tables[member].data_row_count = count

    for nct_id in sorted(early_interventional & biological):
        signals[nct_id].add('early_phase_interventional_with_biological')
    return set(signals), signals


def _capture_candidate_rows(
    tables: dict[str, _ExtractedTable],
    candidate_nct_ids: set[str],
) -> dict[str, dict[str, list[_CapturedRow]]]:
    captured_by_member: dict[str, dict[str, list[_CapturedRow]]] = {}
    for member in _DECISION_MEMBERS:
        by_nct: dict[str, list[_CapturedRow]] = defaultdict(list)
        count = 0
        for captured in tables[member].rows():
            count += 1
            nct_id = captured.values['nct_id'].strip()
            if nct_id in candidate_nct_ids:
                by_nct[nct_id].append(captured)
        if tables[member].data_row_count is None:
            tables[member].data_row_count = count
        elif tables[member].data_row_count != count:
            raise AactExecutionAdapterError(f'non-deterministic row count while rescanning {member}')
        captured_by_member[member] = by_nct
    return captured_by_member


def _lineage_assignment(
    *,
    nct_id: str,
    disease_stratum: DiseaseStratum,
    sponsor_rows: Sequence[_CapturedRow],
    intervention_rows: Sequence[_CapturedRow],
) -> LineageAssignmentReceipt:
    lead_sponsors = tuple(
        sorted(
            {
                normalized
                for captured in sponsor_rows
                if _normalize_identity(captured.values['lead_or_collaborator']) == 'lead'
                and (normalized := _normalize_identity(captured.values['name']))
            }
        )
    )
    biological_products = tuple(
        sorted(
            {
                normalized
                for captured in intervention_rows
                if _normalize_identity(captured.values['intervention_type']) == 'biological'
                and (normalized := _normalize_identity(captured.values['name']))
            }
        )
    )
    fallback_products = tuple(
        sorted({normalized for row in intervention_rows if (normalized := _normalize_identity(row.values['name']))})
    )
    products = biological_products or fallback_products
    identity_material = {'lead_sponsors': lead_sponsors, 'product_names': products}
    identity_sha256 = _sha256_bytes(canonical_json_bytes(identity_material))
    if lead_sponsors and biological_products:
        review_status = 'needs_manual_lineage_review_exact_identity'
        grouping_material = {
            **identity_material,
            'disease_stratum': disease_stratum.value,
            'rule_id': LINEAGE_GROUPING_RULE_ID,
        }
    else:
        if not lead_sponsors and not products:
            review_status = 'needs_review_missing_sponsor_and_product_identity'
        elif not lead_sponsors:
            review_status = 'needs_review_missing_lead_sponsor'
        else:
            review_status = 'needs_review_missing_product_identity'
        grouping_material = {
            **identity_material,
            'disease_stratum': disease_stratum.value,
            'nct_id': nct_id,
            'rule_id': LINEAGE_GROUPING_RULE_ID,
            'singleton': True,
        }
    group_sha256 = _sha256_bytes(canonical_json_bytes(grouping_material))
    return LineageAssignmentReceipt(
        nct_id=nct_id,
        disease_stratum=disease_stratum,
        lineage_group_id=f'lin-{group_sha256[:24]}',
        identity_sha256=identity_sha256,
        normalized_lead_sponsors=lead_sponsors,
        normalized_product_names=products,
        review_status=review_status,
    )


def _normalize_decision_records(
    *,
    fingerprint: _ArchiveFingerprint,
    candidates: set[str],
    signals: dict[str, set[str]],
    captured: dict[str, dict[str, list[_CapturedRow]]],
) -> tuple[
    tuple[AactExecutionDecisionRow, ...], tuple[ScreenedDecisionRecord, ...], tuple[LineageAssignmentReceipt, ...]
]:
    decision_rows: list[AactExecutionDecisionRow] = []
    screened_records: list[ScreenedDecisionRecord] = []
    lineage_receipts: list[LineageAssignmentReceipt] = []
    nct_pattern = re.compile(r'^NCT\d{8}$')

    for nct_id in sorted(candidates):
        rows_by_member = {member: tuple(captured[member].get(nct_id, ())) for member in _DECISION_MEMBERS}
        all_source_rows = tuple(
            sorted(
                (row.provenance for rows in rows_by_member.values() for row in rows),
                key=lambda item: (item.member_path, item.data_row_number),
            )
        )
        exclusions: set[str] = set()
        notes: set[str] = set()
        study_rows = rows_by_member['studies.txt']
        if not nct_pattern.fullmatch(nct_id):
            exclusions.add('invalid_nct_id')
        if len(study_rows) != 1:
            exclusions.add('missing_study_row' if not study_rows else 'duplicate_study_rows')

        text_values: list[str] = []
        for member, fields in (
            ('studies.txt', ('brief_title', 'official_title', 'acronym')),
            ('interventions.txt', ('name', 'description')),
            ('conditions.txt', ('name',)),
            ('keywords.txt', ('name',)),
        ):
            for captured_row in rows_by_member[member]:
                text_values.extend(captured_row.values[field] for field in fields)
        classification_text = _text(text_values)
        vaccine_lexical = _vaccine_signal(classification_text)
        infectious_signal = _INFECTIOUS_PATTERN.search(classification_text) is not None
        covid_signal = _COVID_PATTERN.search(classification_text) is not None
        purposes = {_normalize_identity(row.values['primary_purpose']) for row in rows_by_member['designs.txt']}
        prophylactic_intent = 'prevention' in purposes
        biological_intervention_count = sum(
            _normalize_identity(row.values['intervention_type']) == 'biological'
            for row in rows_by_member['interventions.txt']
        )
        infectious_disease_vaccine = infectious_signal and (vaccine_lexical or biological_intervention_count > 0)
        disease_stratum = DiseaseStratum.COVID_19 if covid_signal else DiseaseStratum.NON_COVID_INFECTIOUS
        classification_basis = {
            'vaccine_lexical': vaccine_lexical,
            'infectious_signal': infectious_signal,
            'covid_signal': covid_signal,
            'primary_purposes': sorted(purposes),
            'biological_intervention_count': biological_intervention_count,
            'source_row_hashes': [row.raw_row_sha256 for row in all_source_rows],
        }
        classification_basis_sha256 = _sha256_bytes(canonical_json_bytes(classification_basis))

        normalized_row: AactExecutionDecisionRow | None = None
        lineage: LineageAssignmentReceipt | None = None
        if len(study_rows) == 1 and not exclusions:
            study = study_rows[0].values
            first_posted = _parse_date(study['study_first_posted_date'])
            if first_posted is None:
                exclusions.add('study_first_posted_date_missing_or_unparseable')
            elif first_posted > fingerprint.archive_date:
                exclusions.add('study_first_posted_after_archive_date')
            results_first_posted = _parse_date(study['results_first_posted_date'])
            if study['results_first_posted_date'].strip() and results_first_posted is None:
                notes.add('results_first_posted_date_unparseable_treated_as_results_present')
            enrollment, enrollment_type, pair_notes = _normalize_enrollment_pair(
                study['enrollment'], study['enrollment_type'], 'enrollment'
            )
            notes.update(pair_notes)
            primary_completion, primary_completion_type, pair_notes = _normalize_date_pair(
                study['primary_completion_date'],
                study['primary_completion_date_type'],
                'primary_completion_date',
            )
            notes.update(pair_notes)
            lineage = _lineage_assignment(
                nct_id=nct_id,
                disease_stratum=disease_stratum,
                sponsor_rows=rows_by_member['sponsors.txt'],
                intervention_rows=rows_by_member['interventions.txt'],
            )
            if not exclusions and first_posted is not None:
                normalized_row = AactExecutionDecisionRow(
                    snapshot_id=fingerprint.snapshot_id,
                    archive_date=fingerprint.archive_date,
                    source_record_sha256=_source_record_sha256(all_source_rows),
                    nct_id=nct_id,
                    lineage_group_id=lineage.lineage_group_id,
                    disease_stratum=disease_stratum,
                    study_first_posted_date=first_posted,
                    study_type=_normalize_study_type(study['study_type']),
                    phase=_normalize_phase(study['phase']),
                    human=_normalize_study_type(study['study_type']) == NormalizedStudyType.INTERVENTIONAL,
                    prophylactic_intent=prophylactic_intent,
                    infectious_disease_vaccine=infectious_disease_vaccine,
                    biological_intervention_count=biological_intervention_count,
                    overall_status=_normalize_status(study['overall_status']),
                    results_first_posted_date=results_first_posted,
                    results_section_present=bool(study['results_first_posted_date'].strip()),
                    enrollment=enrollment,
                    enrollment_type=enrollment_type,
                    primary_completion_date=primary_completion,
                    primary_completion_date_type=primary_completion_type,
                )

        if normalized_row is not None and lineage is not None:
            decision_rows.append(normalized_row)
            lineage_receipts.append(lineage)
        screened_records.append(
            ScreenedDecisionRecord(
                nct_id=nct_id,
                discovery_signals=tuple(sorted(signals[nct_id])),
                normalized=normalized_row is not None,
                normalization_exclusion_reasons=tuple(sorted(exclusions)),
                normalization_notes=tuple(sorted(notes)),
                normalized_decision_row_sha256=(
                    _sha256_bytes(canonical_json_bytes(normalized_row)) if normalized_row is not None else None
                ),
                classification_basis_sha256=classification_basis_sha256,
                source_rows=all_source_rows,
            )
        )
    return tuple(decision_rows), tuple(screened_records), tuple(lineage_receipts)


def _build_decision_material(
    *,
    decision_fingerprint: _ArchiveFingerprint,
    label_fingerprint: _ArchiveFingerprint,
    workspace: Path,
    synthetic: bool,
) -> _DecisionMaterial:
    tables = _extract_tables(decision_fingerprint, _DECISION_MEMBERS, workspace)
    candidates, signals = _scan_decision_universe(tables)
    captured = _capture_candidate_rows(tables, candidates)
    decision_rows, screened_records, lineage = _normalize_decision_records(
        fingerprint=decision_fingerprint,
        candidates=candidates,
        signals=signals,
        captured=captured,
    )
    if not decision_rows:
        raise AactExecutionAdapterError('high-recall screen produced no normalizable decision records')
    decision_receipt = _archive_receipt(decision_fingerprint, tables)
    binding = bind_anchor_source(
        anchor_date=decision_fingerprint.archive_date,
        decision_snapshot_id=decision_fingerprint.snapshot_id,
        decision_archive_manifest_sha256=decision_fingerprint.sha256,
        label_snapshot_id=label_fingerprint.snapshot_id,
        label_archive_manifest_sha256=label_fingerprint.sha256,
        rows=decision_rows,
    )
    policy = ExecutionCohortPolicy(
        policy_id=f'aact-execution-{decision_fingerprint.archive_date.isoformat()}-development-v0.1',
        synthetic=synthetic,
        selection_universe_rule_id=SELECTION_UNIVERSE_RULE_ID,
        selection_universe_rule_sha256=_sha256_bytes(canonical_json_bytes(_SELECTION_RULE)),
        lineage_grouping_rule_id=LINEAGE_GROUPING_RULE_ID,
        lineage_grouping_rule_sha256=_sha256_bytes(canonical_json_bytes(_LINEAGE_RULE)),
        anchors=(binding,),
    )
    inventory = build_execution_inventory(policy=policy, decision_rows=decision_rows)
    audit_execution_inventory(inventory)
    if not inventory.assignments:
        raise AactExecutionAdapterError('fixed decision cohort has no eligible registry-execution trials')
    return _DecisionMaterial(
        inventory=inventory,
        archive_receipt=decision_receipt,
        screened_records=screened_records,
        lineage=lineage,
    )


def _derive_label_material(
    *,
    label_fingerprint: _ArchiveFingerprint,
    inventory: ExecutionCohortInventory,
    workspace: Path,
) -> tuple[ExecutionLabelSet, ExactArchiveReceipt, tuple[OutcomeSourceRecord, ...]]:
    # This is intentionally the first operation that opens the later archive or parses a later row.
    tables = _extract_tables(label_fingerprint, _LABEL_MEMBERS, workspace)
    studies = tables['studies.txt']
    assigned = {assignment.nct_id: assignment for assignment in inventory.assignments}
    found: dict[str, _CapturedRow] = {}
    count = 0
    for captured in studies.rows():
        count += 1
        nct_id = captured.values['nct_id'].strip()
        if nct_id not in assigned:
            continue
        if nct_id in found:
            raise AactExecutionAdapterError(f'label archive contains duplicate studies rows for {nct_id}')
        found[nct_id] = captured
    studies.data_row_count = count

    outcome_rows: list[AactExecutionOutcomeRow] = []
    source_records: list[OutcomeSourceRecord] = []
    for nct_id in sorted(assigned):
        captured = found.get(nct_id)
        if captured is None:
            outcome_rows.append(
                AactExecutionOutcomeRow(
                    snapshot_id=label_fingerprint.snapshot_id,
                    archive_date=label_fingerprint.archive_date,
                    nct_id=nct_id,
                    record_present=False,
                )
            )
            source_records.append(
                OutcomeSourceRecord(
                    nct_id=nct_id,
                    record_present=False,
                    source_row=None,
                    raw_overall_status='',
                    raw_enrollment='',
                    raw_enrollment_type='',
                    raw_primary_completion_date='',
                    raw_primary_completion_date_type='',
                    normalization_notes=(),
                )
            )
            continue
        row = captured.values
        notes: set[str] = set()
        enrollment, enrollment_type, pair_notes = _normalize_enrollment_pair(
            row['enrollment'], row['enrollment_type'], 'enrollment'
        )
        notes.update(pair_notes)
        primary_completion, primary_completion_type, pair_notes = _normalize_date_pair(
            row['primary_completion_date'],
            row['primary_completion_date_type'],
            'primary_completion_date',
        )
        notes.update(pair_notes)
        outcome_rows.append(
            AactExecutionOutcomeRow(
                snapshot_id=label_fingerprint.snapshot_id,
                archive_date=label_fingerprint.archive_date,
                nct_id=nct_id,
                record_present=True,
                source_record_sha256=captured.provenance.raw_row_sha256,
                overall_status=_normalize_optional_status(row['overall_status']),
                enrollment=enrollment,
                enrollment_type=enrollment_type,
                primary_completion_date=primary_completion,
                primary_completion_date_type=primary_completion_type,
            )
        )
        source_records.append(
            OutcomeSourceRecord(
                nct_id=nct_id,
                record_present=True,
                source_row=captured.provenance,
                raw_overall_status=row['overall_status'],
                raw_enrollment=row['enrollment'],
                raw_enrollment_type=row['enrollment_type'],
                raw_primary_completion_date=row['primary_completion_date'],
                raw_primary_completion_date_type=row['primary_completion_date_type'],
                normalization_notes=tuple(sorted(notes)),
            )
        )
    label_set = derive_execution_labels(inventory=inventory, outcome_rows=outcome_rows)
    audit_execution_label_set(inventory=inventory, label_set=label_set)
    return label_set, _archive_receipt(label_fingerprint, tables), tuple(source_records)


def _write_exact(path: Path, value: object, *, relative_path: str) -> ArtifactReceipt:
    payload = canonical_json_bytes(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ArtifactReceipt(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
    )


def build_aact_execution_cohort(
    *,
    decision_archive: Path,
    decision_archive_date: date,
    label_archive: Path,
    label_archive_date: date,
    output_root: Path,
    trusted_source_binding: TrustedAactSourceBinding | None = None,
    synthetic_test_only: bool = False,
) -> AactExecutionBuild:
    """Build one immutable, organizer-private provisional inventory with private observations.

    Real-data mode requires an independently supplied official catalog/acquisition binding plus
    exact archive receipts.  ``synthetic_test_only`` is an explicit escape hatch for unit fixtures;
    it cannot emit a real-data status.  Neither mode emits a scored or split-safe cohort.
    """

    if (trusted_source_binding is None) == (not synthetic_test_only):
        raise AactExecutionAdapterError(
            'provide exactly one source mode: trusted_source_binding or synthetic_test_only=True'
        )

    expected_label_date = add_calendar_months(decision_archive_date, 48)
    if label_archive_date != expected_label_date:
        raise AactExecutionAdapterError(
            f'label archive date must be exactly +48 calendar months: {expected_label_date.isoformat()}'
        )
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'immutable execution cohort output already exists: {destination}')
    decision_snapshot_id = f'aact-flatfiles-{decision_archive_date.isoformat()}'
    label_snapshot_id = f'aact-flatfiles-{label_archive_date.isoformat()}'
    decision_fingerprint = _hash_regular_file(decision_archive, decision_snapshot_id, decision_archive_date)
    # Hashing binds later bytes but does not open the ZIP or parse its studies member.
    label_fingerprint = _hash_regular_file(label_archive, label_snapshot_id, label_archive_date)
    source_binding_receipt = (
        _verify_trusted_source_binding(
            binding=trusted_source_binding,
            decision_fingerprint=decision_fingerprint,
            label_fingerprint=label_fingerprint,
        )
        if trusted_source_binding is not None
        else ExecutionSourceBindingReceipt(mode='synthetic_test_only')
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(0o700)
    try:
        with tempfile.TemporaryDirectory(prefix='vaxreplay-aact-decision-') as decision_tmp:
            decision_material = _build_decision_material(
                decision_fingerprint=decision_fingerprint,
                label_fingerprint=label_fingerprint,
                workspace=Path(decision_tmp),
                synthetic=synthetic_test_only,
            )
        # The inventory object has now been built and independently audited.  Only now may the
        # later archive be opened and its studies member parsed.
        with tempfile.TemporaryDirectory(prefix='vaxreplay-aact-label-') as label_tmp:
            labels, label_receipt, outcome_sources = _derive_label_material(
                label_fingerprint=label_fingerprint,
                inventory=decision_material.inventory,
                workspace=Path(label_tmp),
            )

        artifacts: list[ArtifactReceipt] = []
        values_by_path: tuple[tuple[str, object], ...] = (
            ('organizer/source-binding.json', source_binding_receipt),
            ('organizer/decision-archive.json', decision_material.archive_receipt),
            (
                'organizer/screened-decision-records.json',
                ScreenedDecisionRecordSet(records=decision_material.screened_records),
            ),
            (
                'organizer/lineage-assignments.json',
                LineageAssignmentSet(assignments=decision_material.lineage),
            ),
            ('organizer/cohort-inventory.json', decision_material.inventory),
            ('private/label-archive.json', label_receipt),
            ('private/outcome-source-records.json', OutcomeSourceRecordSet(records=outcome_sources)),
            ('private/execution-labels.json', labels),
        )
        for relative_path, value in values_by_path:
            artifacts.append(_write_exact(staging / relative_path, value, relative_path=relative_path))

        assigned = decision_material.inventory.assignments
        receipt = AactExecutionBuildReceipt(
            synthetic=synthetic_test_only,
            release_status=('synthetic_test_only' if synthetic_test_only else 'provisional_high_recall_inventory'),
            source_binding=source_binding_receipt,
            decision_archive=decision_material.archive_receipt,
            label_archive=label_receipt,
            selection_universe_rule_sha256=_sha256_bytes(canonical_json_bytes(_SELECTION_RULE)),
            lineage_grouping_rule_sha256=_sha256_bytes(canonical_json_bytes(_LINEAGE_RULE)),
            screened_record_count=len(decision_material.screened_records),
            normalized_record_count=sum(record.normalized for record in decision_material.screened_records),
            normalization_exclusion_count=sum(not record.normalized for record in decision_material.screened_records),
            assigned_trial_count=len(assigned),
            covid_assigned_trial_count=sum(
                assignment.disease_stratum == DiseaseStratum.COVID_19 for assignment in assigned
            ),
            non_covid_assigned_trial_count=sum(
                assignment.disease_stratum == DiseaseStratum.NON_COVID_INFECTIOUS for assignment in assigned
            ),
            missing_label_record_count=labels.missing_record_count,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        )
        _write_exact(staging / 'BUILD-RECEIPT.json', receipt, relative_path='BUILD-RECEIPT.json')
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()), key=lambda path: len(path.parts), reverse=True
        ):
            fsync_directory(directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return AactExecutionBuild(
        root=destination,
        receipt=receipt,
        inventory=decision_material.inventory,
        labels=labels,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-archive', required=True, type=Path)
    parser.add_argument('--decision-archive-date', type=date.fromisoformat, required=True)
    parser.add_argument('--label-archive', required=True, type=Path)
    parser.add_argument('--label-archive-date', type=date.fromisoformat, required=True)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--acquisition-plan', type=Path)
    parser.add_argument('--decision-archive-receipt', type=Path)
    parser.add_argument('--label-archive-receipt', type=Path)
    parser.add_argument('--synthetic-test-only', action='store_true')
    parser.add_argument('--output-root', required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding_paths = (
        args.catalog,
        args.acquisition_plan,
        args.decision_archive_receipt,
        args.label_archive_receipt,
    )
    if args.synthetic_test_only:
        if any(path is not None for path in binding_paths):
            raise AactExecutionAdapterError('synthetic test mode cannot accept trusted real-data binding files')
        trusted_source_binding = None
    else:
        if any(path is None for path in binding_paths):
            raise AactExecutionAdapterError(
                'real mode requires --catalog, --acquisition-plan, --decision-archive-receipt, '
                'and --label-archive-receipt'
            )
        try:
            trusted_source_binding = TrustedAactSourceBinding(
                catalog=AactOfficialArchiveCatalog.model_validate_json(args.catalog.read_bytes()),
                acquisition_plan=AactArchiveAcquisitionPlan.model_validate_json(args.acquisition_plan.read_bytes()),
                decision_archive_receipt=AactArchiveReceipt.model_validate_json(
                    args.decision_archive_receipt.read_bytes()
                ),
                label_archive_receipt=AactArchiveReceipt.model_validate_json(args.label_archive_receipt.read_bytes()),
            )
        except (OSError, ValueError) as error:
            raise AactExecutionAdapterError(f'cannot load trusted AACT source binding: {error}') from error
    build = build_aact_execution_cohort(
        decision_archive=args.decision_archive,
        decision_archive_date=args.decision_archive_date,
        label_archive=args.label_archive,
        label_archive_date=args.label_archive_date,
        output_root=args.output_root,
        trusted_source_binding=trusted_source_binding,
        synthetic_test_only=args.synthetic_test_only,
    )
    print(canonical_json_bytes(build.receipt).decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
