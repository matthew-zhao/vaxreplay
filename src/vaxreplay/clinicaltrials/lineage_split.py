"""Outcome-blind lineage isolation and frozen splits for AACT execution replay.

This stage consumes only the audited decision inventory and the decision-only active-vaccine
review.  It deliberately never opens the later execution-label artifact.  Included trials are
grouped at a conservative pathogen/target-family granularity before any split is assigned, so a
family cannot occur in more than one partition.  NCT IDs and biological family names remain in an
organizer-private artifact; prospective public task and lineage IDs are HMAC-derived with an
out-of-band random key.

The output is suitable as an input to task construction, but it is not an episode release and does
not assert that public workspaces are identity-masked, sealed, or leaderboard-admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay._atomic import fsync_directory, rename_directory_noreplace
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_inventory import ExecutionInventoryError, audit_execution_inventory
from vaxreplay.clinicaltrials.execution_merge import AactExecutionMultiAnchorMergeReceipt
from vaxreplay.clinicaltrials.execution_schema import ExecutionCohortInventory, TrialAnchorAssignment
from vaxreplay.clinicaltrials.relevance_adjudication import (
    ACTIVE_VACCINE_RELEVANCE_POLICY,
    DecisionEvidenceRecord,
    RelevanceDisposition,
    VaccineRelevanceAdjudicationSet,
    VaccineRelevanceReviewQueue,
    VaccineRelevanceReviewReceipt,
    finalize_relevance_adjudications,
)

LINEAGE_SPLIT_POLICY_SCHEMA_VERSION = 'vaxreplay.aact-lineage-split-policy.v0.1'
LINEAGE_SPLIT_ASSIGNMENTS_SCHEMA_VERSION = 'vaxreplay.aact-lineage-split-assignments.v0.1'
LINEAGE_SPLIT_RECEIPT_SCHEMA_VERSION = 'vaxreplay.aact-lineage-split-receipt.v0.1'
LINEAGE_SPLIT_POLICY_ID = 'aact-conservative-target-family-lineage-split-v0.1'
CLASSIFIER_ID = 'decision-conditions-first-fixed-target-family-v0.1'
SPLITTER_ID = 'largest-lineage-greedy-l2-hmac-tiebreak-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_FAMILY_PATTERN = r'^[a-z0-9][a-z0-9-]*$'
_TASK_ID_PATTERN = r'^vaxclin-[0-9a-f]{24}$'
_LINEAGE_ID_PATTERN = r'^clinlin-[0-9a-f]{24}$'
_TASK_ID_DOMAIN = b'vaxreplay/aact/public-task-id/v1\x00'
_LINEAGE_ID_DOMAIN = b'vaxreplay/aact/public-lineage-id/v1\x00'
_SPLIT_ORDER_DOMAIN = b'vaxreplay/aact/split-order/v1\x00'
_SPLIT_TIE_DOMAIN = b'vaxreplay/aact/split-tie/v1\x00'
_KEY_COMMITMENT_DOMAIN = b'vaxreplay/aact/id-key-commitment/v1\x00'
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_LINEAGE_ROOT_ENTRIES = frozenset({'LINEAGE-SPLIT-RECEIPT.json', 'organizer'})
_LINEAGE_ORGANIZER_ENTRIES = frozenset(
    {
        'lineage-split-assignments.json',
        'lineage-split-policy.json',
    }
)


class LineageSplitError(ValueError):
    """Trusted decision-only material cannot produce a closed lineage split."""


def _normalize_text(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


class TargetFamilyRule(StrictModel):
    family_id: str = Field(pattern=_SAFE_FAMILY_PATTERN)
    terms: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_terms(self) -> Self:
        if self.terms != tuple(sorted(set(self.terms))):
            raise ValueError('target-family terms must be unique and sorted')
        if any(term != _normalize_text(term) or not term for term in self.terms):
            raise ValueError('target-family terms must already be normalized')
        return self


class SplitWeight(StrictModel):
    split: Split
    weight: int = Field(gt=0)


class LineageSplitPolicy(StrictModel):
    schema_version: Literal['vaxreplay.aact-lineage-split-policy.v0.1'] = LINEAGE_SPLIT_POLICY_SCHEMA_VERSION
    policy_id: Literal['aact-conservative-target-family-lineage-split-v0.1'] = LINEAGE_SPLIT_POLICY_ID
    classifier_id: Literal['decision-conditions-first-fixed-target-family-v0.1'] = CLASSIFIER_ID
    splitter_id: Literal['largest-lineage-greedy-l2-hmac-tiebreak-v0.1'] = SPLITTER_ID
    grouping_granularity: Literal['conservative_pathogen_or_target_family'] = 'conservative_pathogen_or_target_family'
    target_family_rules: tuple[TargetFamilyRule, ...] = Field(min_length=1)
    split_weights: tuple[SplitWeight, ...] = Field(min_length=3, max_length=3)
    conditions_are_primary_classification_text: Literal[True] = True
    decision_text_fallback_only_when_conditions_unresolved: Literal[True] = True
    ambiguous_or_unresolved_classification_fails_closed: Literal[True] = True
    relevance_include_only: Literal[True] = True
    target_family_never_crosses_partitions: Literal[True] = True
    largest_groups_assigned_first: Literal[True] = True
    split_objective_uses_case_counts_only: Literal[True] = True
    hmac_tiebreak_uses_out_of_band_random_key: Literal[True] = True
    decision_time_evidence_only: Literal[True] = True
    later_archive_opened: Literal[False] = False
    execution_labels_read: Literal[False] = False
    outcome_conditioned_grouping_or_split: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_policy(self) -> Self:
        family_ids = tuple(rule.family_id for rule in self.target_family_rules)
        if family_ids != tuple(sorted(set(family_ids))):
            raise ValueError('target-family rules must have unique sorted IDs')
        expected_splits = (Split.TRAIN, Split.DEV, Split.TEST)
        if tuple(item.split for item in self.split_weights) != expected_splits:
            raise ValueError('split weights must be ordered train, dev, test')
        return self


def _rule(family_id: str, *terms: str) -> TargetFamilyRule:
    return TargetFamilyRule(family_id=family_id, terms=tuple(sorted(terms)))


# These deliberately broad families trade statistical independence for leakage isolation. They
# are public, contamination-exposed reference semantics—not an undisclosed selection policy for a
# held-out or commercial cohort. Such a cohort needs a newly frozen organizer-private policy.
TARGET_FAMILY_RULES = (
    _rule(
        'alphavirus',
        'alphavirus',
        'chikungunya',
        'eastern equine encephalitis',
        'encephalomyelitis, equine',
        'venezuelan equine encephalitis',
        'wevee',
        'western equine encephalitis',
    ),
    _rule('anthrax', 'anthrax', 'biothrax', 'nasoshield'),
    _rule('clostridioides-difficile', 'c. difficile', 'clostridium difficile', 'clostridium infections'),
    _rule('cytomegalovirus', 'cytomegalovirus'),
    _rule('diphtheria-pertussis-tetanus', 'diphtheria', 'pertussis', 'tdap', 'tetanus', 'whooping cough'),
    _rule(
        'enteric-bacterial',
        'enterotoxigenic escherichia coli',
        'etec',
        'gastroenteritis escherichia coli',
        'salmonella',
        'shigella',
        'shigellosis',
    ),
    _rule('enteric-viral', 'norovirus', 'norwalk', 'rotavirus'),
    _rule('filovirus', 'ebola', 'filovirus', 'marburg', 'multi-filo'),
    _rule('flavivirus', 'dengue', 'flavivirus', 'yellow fever', 'zika'),
    _rule('hantavirus', 'hantaan', 'hantavirus', 'puumala'),
    _rule('helminth', 'hookworm', 'schistosoma', 'schistosomiasis'),
    _rule('hepatitis', 'hepatitis b', 'hepatitis e', 'hev 239', 'hev239'),
    _rule('hiv', 'aidsvax', 'hiv', 'human immunodeficiency virus'),
    _rule('hpv', 'hpv', 'papillomavirus'),
    _rule('influenza', 'flu vaccine', 'h5n1', 'h7n9', 'influenza'),
    _rule('malaria', 'malaria', 'pfras', 'pfs230', 'pfs25', 'pfspz', 'plasmodium', 'r21 matrix', 'rh5'),
    _rule('mers-coronavirus', 'gls-5300', 'mers', 'middle east respiratory syndrome'),
    _rule('nipah-henipavirus', 'hev-sg-v', 'nipah'),
    _rule('parainfluenza-metapneumovirus', 'metapneumovirus', 'mrna-1653', 'parainfluenza', 'sendai virus vaccine'),
    _rule('pneumococcus', 'pcv13', 'pneumococcal', 'pneumovax', 'ppsv23', 'streptococcus pneumoniae'),
    _rule('polio', 'polio', 'poliomyelitis', 'poliovirus'),
    _rule('rabies', 'rabavert', 'rabies', 'rg sam'),
    _rule(
        'respiratory-syncytial-virus',
        'mva-bn-rsv',
        'respiratory syncytial virus',
        'rsv 276',
        'rsv infection',
        'rsv vaccine',
        'sevrsv',
    ),
    _rule('tuberculosis', 'ad5ag85a', 'chadox1 85a', 'mtbvac', 'mycobacterium tuberculosis', 'tuberculosis'),
)


LINEAGE_SPLIT_POLICY = LineageSplitPolicy(
    target_family_rules=TARGET_FAMILY_RULES,
    split_weights=(
        SplitWeight(split=Split.TRAIN, weight=3),
        SplitWeight(split=Split.DEV, weight=1),
        SplitWeight(split=Split.TEST, weight=1),
    ),
)


def lineage_split_policy_sha256() -> str:
    return _model_sha256(LINEAGE_SPLIT_POLICY)


class FamilyClassification(StrictModel):
    target_family_id: str = Field(pattern=_SAFE_FAMILY_PATTERN)
    match_basis: Literal['conditions', 'decision_text_fallback']
    matched_terms: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_terms(self) -> Self:
        if self.matched_terms != tuple(sorted(set(self.matched_terms))):
            raise ValueError('matched family terms must be unique and sorted')
        return self


class LineageCaseAssignment(StrictModel):
    nct_id: str = Field(pattern=r'^NCT\d{8}$')
    anchor_date: date
    opaque_task_id: str = Field(pattern=_TASK_ID_PATTERN)
    source_assignment_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_decision_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_family_id: str = Field(pattern=_SAFE_FAMILY_PATTERN)
    family_match_basis: Literal['conditions', 'decision_text_fallback']
    matched_terms: tuple[str, ...] = Field(min_length=1)
    lineage_group_id: str = Field(pattern=_LINEAGE_ID_PATTERN)
    split: Split
    organizer_private: Literal[True] = True
    public_identity_release_ready: Literal[False] = False

    @model_validator(mode='after')
    def validate_match_terms(self) -> Self:
        if self.matched_terms != tuple(sorted(set(self.matched_terms))):
            raise ValueError('case match terms must be unique and sorted')
        return self


class LineageGroupSummary(StrictModel):
    lineage_group_id: str = Field(pattern=_LINEAGE_ID_PATTERN)
    target_family_id: str = Field(pattern=_SAFE_FAMILY_PATTERN)
    split: Split
    member_count: int = Field(gt=0)
    nct_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_members(self) -> Self:
        if self.nct_ids != tuple(sorted(set(self.nct_ids))) or self.member_count != len(self.nct_ids):
            raise ValueError('lineage members must be unique, sorted, and match member_count')
        return self


class SplitCount(StrictModel):
    split: Split
    case_count: int = Field(ge=0)
    lineage_count: int = Field(ge=0)


class AnchorSplitCount(StrictModel):
    anchor_date: date
    split: Split
    case_count: int = Field(ge=0)


class LineageSplitAssignmentSet(StrictModel):
    schema_version: Literal['vaxreplay.aact-lineage-split-assignments.v0.1'] = LINEAGE_SPLIT_ASSIGNMENTS_SCHEMA_VERSION
    policy_id: Literal['aact-conservative-target-family-lineage-split-v0.1'] = LINEAGE_SPLIT_POLICY_ID
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    merge_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    merged_inventory_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_policy_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    id_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    upstream_include_count: int = Field(gt=0)
    upstream_exclude_count: int = Field(ge=0)
    upstream_hold_count: int = Field(ge=0)
    assignment_count: int = Field(gt=0)
    held_or_dropped_include_count: int = Field(ge=0)
    cases: tuple[LineageCaseAssignment, ...] = Field(min_length=1)
    lineages: tuple[LineageGroupSummary, ...] = Field(min_length=1)
    split_counts: tuple[SplitCount, ...] = Field(min_length=3, max_length=3)
    anchor_split_counts: tuple[AnchorSplitCount, ...] = Field(min_length=1)
    organizer_private: Literal[True] = True
    include_coverage_complete: Literal[True] = True
    lineage_classification_complete: Literal[True] = True
    lineage_split_isolated: Literal[True] = True
    conservative_target_family_isolation_claimed: Literal[True] = True
    exact_product_program_lineage_claimed: Literal[False] = False
    opaque_ids_hmac_sha256: Literal[True] = True
    id_secret_stored_in_artifact: Literal[False] = False
    public_ids_currently_safe_to_release: Literal[False] = False
    decision_time_evidence_only: Literal[True] = True
    later_archive_opened: Literal[False] = False
    execution_labels_read: Literal[False] = False
    outcome_conditioned_grouping_or_split: Literal[False] = False
    authenticated_upstream_receipt_hashes_required: Literal[True] = True
    decision_archive_bytes_reopened: Literal[False] = False
    upstream_source_reverification_required: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    public_task_workspaces_built: Literal[False] = False

    @model_validator(mode='after')
    def validate_assignment_set(self) -> Self:
        if self.policy_sha256 != lineage_split_policy_sha256():
            raise ValueError('lineage assignments do not use the fixed policy')
        case_keys = tuple((item.anchor_date, item.nct_id) for item in self.cases)
        if case_keys != tuple(sorted(set(case_keys))):
            raise ValueError('case assignments must be unique and sorted by anchor/NCT')
        if self.assignment_count != len(self.cases):
            raise ValueError('assignment_count does not equal the case count')
        if self.assignment_count + self.held_or_dropped_include_count != self.upstream_include_count:
            raise ValueError('included cases must be assigned or explicitly held/dropped')
        if self.held_or_dropped_include_count != 0:
            raise ValueError('a complete lineage split cannot silently hold or drop an included case')
        task_ids = tuple(item.opaque_task_id for item in self.cases)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError('opaque task IDs must be unique')
        lineage_ids = tuple(item.lineage_group_id for item in self.lineages)
        if lineage_ids != tuple(sorted(set(lineage_ids))):
            raise ValueError('lineage summaries must be unique and sorted by opaque lineage ID')
        summary_by_id = {item.lineage_group_id: item for item in self.lineages}
        cases_by_lineage: dict[str, list[LineageCaseAssignment]] = defaultdict(list)
        for item in self.cases:
            cases_by_lineage[item.lineage_group_id].append(item)
        if set(cases_by_lineage) != set(summary_by_id):
            raise ValueError('case lineages and lineage summaries do not cover the same groups')
        for lineage_id, members in cases_by_lineage.items():
            summary = summary_by_id[lineage_id]
            if {member.split for member in members} != {summary.split}:
                raise ValueError('a lineage crosses split partitions')
            if {member.target_family_id for member in members} != {summary.target_family_id}:
                raise ValueError('a lineage contains more than one target family')
            if tuple(sorted(member.nct_id for member in members)) != summary.nct_ids:
                raise ValueError('lineage summary members do not match case assignments')
        if tuple(item.split for item in self.split_counts) != (Split.TRAIN, Split.DEV, Split.TEST):
            raise ValueError('split counts must be ordered train, dev, test')
        for split_count in self.split_counts:
            if split_count.case_count != sum(item.split == split_count.split for item in self.cases):
                raise ValueError('split case count does not match assignments')
            if split_count.lineage_count != sum(item.split == split_count.split for item in self.lineages):
                raise ValueError('split lineage count does not match summaries')
        anchors = tuple(sorted({item.anchor_date for item in self.anchor_split_counts}))
        expected_anchor_keys = tuple(
            (anchor, split) for anchor in anchors for split in (Split.TRAIN, Split.DEV, Split.TEST)
        )
        anchor_keys = tuple((item.anchor_date, item.split) for item in self.anchor_split_counts)
        if anchor_keys != expected_anchor_keys:
            raise ValueError('anchor/split counts must completely cover anchors in fixed split order')
        for count in self.anchor_split_counts:
            observed = sum(item.anchor_date == count.anchor_date and item.split == count.split for item in self.cases)
            if count.case_count != observed:
                raise ValueError('anchor/split count does not match assignments')
        return self


class LineageSplitArtifactReceipt(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    organizer_private: Literal[True] = True


class LineageSplitReceipt(StrictModel):
    schema_version: Literal['vaxreplay.aact-lineage-split-receipt.v0.1'] = LINEAGE_SPLIT_RECEIPT_SCHEMA_VERSION
    policy_id: Literal['aact-conservative-target-family-lineage-split-v0.1'] = LINEAGE_SPLIT_POLICY_ID
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    merge_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    merged_inventory_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_review_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_queue_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_adjudication_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    id_key_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    assignment_count: int = Field(gt=0)
    lineage_count: int = Field(gt=0)
    upstream_exclude_count: int = Field(ge=0)
    upstream_hold_count: int = Field(ge=0)
    held_or_dropped_include_count: Literal[0] = 0
    split_counts: tuple[SplitCount, ...] = Field(min_length=3, max_length=3)
    include_coverage_complete: Literal[True] = True
    lineage_split_isolated: Literal[True] = True
    conservative_target_family_isolation_claimed: Literal[True] = True
    exact_product_program_lineage_claimed: Literal[False] = False
    execution_labels_read: Literal[False] = False
    outcome_conditioned_grouping_or_split: Literal[False] = False
    authenticated_upstream_receipt_hashes_required: Literal[True] = True
    decision_archive_bytes_reopened: Literal[False] = False
    upstream_source_reverification_required: Literal[True] = True
    public_ids_currently_safe_to_release: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False
    sealed_execution_supported: Literal[False] = False
    identity_contamination_controlled: Literal[False] = False
    public_task_workspaces_built: Literal[False] = False
    artifacts: tuple[LineageSplitArtifactReceipt, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.policy_sha256 != lineage_split_policy_sha256():
            raise ValueError('lineage receipt does not use the fixed policy')
        expected_paths = (
            'organizer/lineage-split-assignments.json',
            'organizer/lineage-split-policy.json',
        )
        if tuple(item.relative_path for item in self.artifacts) != expected_paths:
            raise ValueError('lineage receipt must bind exactly the assignment set and policy')
        return self


@dataclass(frozen=True)
class LineageSplitBuild:
    root: Path
    receipt: LineageSplitReceipt
    assignments: LineageSplitAssignmentSet


@dataclass(frozen=True)
class _TrustedInputs:
    merge_receipt_sha256: str
    inventory_artifact_sha256: str
    inventory: ExecutionCohortInventory
    relevance_receipt_sha256: str
    relevance_receipt: VaccineRelevanceReviewReceipt
    relevance_queue_artifact_sha256: str
    relevance_adjudication_artifact_sha256: str
    relevance_policy_artifact_sha256: str
    queue: VaccineRelevanceReviewQueue
    adjudications: VaccineRelevanceAdjudicationSet


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_sha256(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _term_present(term: str, value: str) -> bool:
    return re.search(rf'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])', value) is not None


def _family_matches(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    normalized = tuple(_normalize_text(value) for value in values)
    matches: dict[str, tuple[str, ...]] = {}
    for rule in LINEAGE_SPLIT_POLICY.target_family_rules:
        terms = tuple(sorted(term for term in rule.terms if any(_term_present(term, value) for value in normalized)))
        if terms:
            matches[rule.family_id] = terms
    return matches


def classify_target_family(record: DecisionEvidenceRecord) -> FamilyClassification:
    """Classify one included case from cutoff-safe text, failing closed on ambiguity."""

    matches = _family_matches(record.conditions)
    basis: Literal['conditions', 'decision_text_fallback'] = 'conditions'
    if not matches:
        fallback = (
            record.brief_title,
            record.official_title,
            record.brief_summary,
            record.detailed_description,
            *(f'{item.name} {item.description}' for item in record.interventions),
        )
        matches = _family_matches(fallback)
        basis = 'decision_text_fallback'
    if len(matches) != 1:
        raise LineageSplitError(
            f'target-family classification for {record.nct_id} resolved to {sorted(matches)}; '
            'unresolved or multi-family cases fail closed'
        )
    family_id, terms = next(iter(matches.items()))
    return FamilyClassification(target_family_id=family_id, match_basis=basis, matched_terms=terms)


def _read_regular(path: Path, *, description: str) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise LineageSplitError(f'{description} cannot be a symbolic link: {expanded}')
    try:
        before = os.stat(expanded, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise LineageSplitError(f'{description} must be a regular file: {expanded}')
        descriptor = os.open(expanded, _regular_open_flags())
    except OSError as error:
        raise LineageSplitError(f'cannot open {description}: {expanded}: {error}') from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(before) != _file_identity(opened):
            raise LineageSplitError(f'{description} changed while it was opened: {expanded}')
        payload = _read_descriptor(descriptor, description=description)
        after = os.fstat(descriptor)
        if _stable_file_metadata(opened) != _stable_file_metadata(after):
            raise LineageSplitError(f'{description} changed while it was read: {expanded}')
    finally:
        os.close(descriptor)
    try:
        current = os.stat(expanded, follow_symlinks=False)
    except OSError as error:
        raise LineageSplitError(f'cannot restat {description}: {expanded}: {error}') from error
    if _stable_file_metadata(after) != _stable_file_metadata(current):
        raise LineageSplitError(f'{description} changed while it was read: {expanded}')
    return payload


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_directory_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_descriptor(descriptor: int, *, description: str) -> bytes:
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as error:
        raise LineageSplitError(f'cannot read {description}: {error}') from error
    return b''.join(chunks)


def _require_exact_mode(metadata: os.stat_result, expected_mode: int, description: str) -> None:
    observed_mode = stat.S_IMODE(metadata.st_mode)
    if observed_mode != expected_mode:
        raise LineageSplitError(f'{description} must have mode {expected_mode:04o}, observed {observed_mode:04o}')


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)


def _regular_open_flags() -> int:
    return os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)


def _open_directory_at(parent_descriptor: int, name: str, *, description: str) -> int:
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise LineageSplitError(f'cannot stat {description}: {error}') from error
    if stat.S_ISLNK(before.st_mode):
        raise LineageSplitError(f'{description} cannot be a symbolic link')
    if not stat.S_ISDIR(before.st_mode):
        raise LineageSplitError(f'{description} must be a directory')
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)
    except OSError as error:
        raise LineageSplitError(f'cannot open {description}: {error}') from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise LineageSplitError(f'cannot inspect opened {description}: {error}') from error
    if not stat.S_ISDIR(opened.st_mode) or _file_identity(before) != _file_identity(opened):
        os.close(descriptor)
        raise LineageSplitError(f'{description} changed while it was opened')
    return descriptor


def _open_directory_chain(path: Path, *, description: str) -> tuple[Path, int]:
    absolute = _absolute_without_symlink_resolution(path)
    anchor = Path(absolute.anchor)
    try:
        descriptor = os.open(anchor, _directory_open_flags())
    except OSError as error:
        raise LineageSplitError(f'cannot open filesystem root for {description}: {error}') from error
    try:
        for index, component in enumerate(absolute.parts[1:], start=1):
            component_path = Path(*absolute.parts[: index + 1])
            child = _open_directory_at(
                descriptor,
                component,
                description=f'{description} path component {component_path}',
            )
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return absolute, descriptor


def _require_exact_directory_entries(
    descriptor: int,
    expected: frozenset[str],
    *,
    description: str,
) -> None:
    try:
        observed = frozenset(os.listdir(descriptor))
    except OSError as error:
        raise LineageSplitError(f'cannot enumerate {description}: {error}') from error
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise LineageSplitError(f'{description} must contain the exact closed tree; missing={missing}, extra={extra}')


def _read_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    description: str,
) -> bytes:
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise LineageSplitError(f'cannot stat {description}: {error}') from error
    if stat.S_ISLNK(before.st_mode):
        raise LineageSplitError(f'{description} cannot be a symbolic link')
    if not stat.S_ISREG(before.st_mode):
        raise LineageSplitError(f'{description} must be a regular file')
    if before.st_nlink != 1:
        raise LineageSplitError(f'{description} cannot have external hard links')
    _require_exact_mode(before, _PRIVATE_FILE_MODE, description)
    try:
        descriptor = os.open(name, _regular_open_flags(), dir_fd=parent_descriptor)
    except OSError as error:
        raise LineageSplitError(f'cannot open {description}: {error}') from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(before) != _file_identity(opened):
            raise LineageSplitError(f'{description} changed while it was opened')
        _require_exact_mode(opened, _PRIVATE_FILE_MODE, description)
        payload = _read_descriptor(descriptor, description=description)
        after = os.fstat(descriptor)
        if _stable_file_metadata(opened) != _stable_file_metadata(after):
            raise LineageSplitError(f'{description} changed while it was read')
    finally:
        os.close(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise LineageSplitError(f'cannot restat {description}: {error}') from error
    if _stable_file_metadata(after) != _stable_file_metadata(current):
        raise LineageSplitError(f'{description} changed while it was read')
    return payload


def _read_closed_lineage_tree(root: Path) -> tuple[Path, bytes, bytes, bytes]:
    resolved, root_descriptor = _open_directory_chain(root, description='lineage-split root')
    organizer_descriptor: int | None = None
    try:
        root_before = os.fstat(root_descriptor)
        _require_exact_mode(root_before, _PRIVATE_DIRECTORY_MODE, 'lineage-split root')
        _require_exact_directory_entries(
            root_descriptor,
            _LINEAGE_ROOT_ENTRIES,
            description='lineage-split root',
        )
        organizer_descriptor = _open_directory_at(
            root_descriptor,
            'organizer',
            description='lineage-split organizer directory',
        )
        organizer_before = os.fstat(organizer_descriptor)
        _require_exact_mode(
            organizer_before,
            _PRIVATE_DIRECTORY_MODE,
            'lineage-split organizer directory',
        )
        _require_exact_directory_entries(
            organizer_descriptor,
            _LINEAGE_ORGANIZER_ENTRIES,
            description='lineage-split organizer directory',
        )
        receipt_payload = _read_regular_at(
            root_descriptor,
            'LINEAGE-SPLIT-RECEIPT.json',
            description='lineage-split receipt',
        )
        assignment_payload = _read_regular_at(
            organizer_descriptor,
            'lineage-split-assignments.json',
            description='lineage-split assignments',
        )
        policy_payload = _read_regular_at(
            organizer_descriptor,
            'lineage-split-policy.json',
            description='lineage-split policy',
        )
        _require_exact_directory_entries(
            organizer_descriptor,
            _LINEAGE_ORGANIZER_ENTRIES,
            description='lineage-split organizer directory',
        )
        _require_exact_directory_entries(
            root_descriptor,
            _LINEAGE_ROOT_ENTRIES,
            description='lineage-split root',
        )
        organizer_after = os.fstat(organizer_descriptor)
        root_after = os.fstat(root_descriptor)
        if _stable_directory_metadata(organizer_before) != _stable_directory_metadata(organizer_after):
            raise LineageSplitError('lineage-split organizer directory changed during verification')
        if _stable_directory_metadata(root_before) != _stable_directory_metadata(root_after):
            raise LineageSplitError('lineage-split root changed during verification')
    finally:
        if organizer_descriptor is not None:
            os.close(organizer_descriptor)
        os.close(root_descriptor)
    return resolved, receipt_payload, assignment_payload, policy_payload


def _require_expected_hash(payload: bytes, expected: str, description: str) -> str:
    if re.fullmatch(_SHA256_PATTERN, expected) is None:
        raise LineageSplitError(f'expected {description} SHA-256 must be a 64-character digest')
    observed = _sha256_bytes(payload)
    if observed != expected:
        raise LineageSplitError(f'{description} does not match its externally pinned SHA-256')
    return observed


def _artifact_payload(
    root: Path,
    artifacts: Mapping[str, object],
    relative_path: str,
    *,
    description: str,
) -> tuple[bytes, str]:
    artifact = artifacts.get(relative_path)
    if artifact is None:
        raise LineageSplitError(f'{description} receipt does not bind {relative_path}')
    payload = _read_regular(root / relative_path, description=f'{description} artifact {relative_path}')
    _validate_artifact_payload(payload, artifacts, relative_path, description=description)
    return payload, getattr(artifact, 'sha256')


def _validate_artifact_payload(
    payload: bytes,
    artifacts: Mapping[str, object],
    relative_path: str,
    *,
    description: str,
) -> None:
    artifact = artifacts.get(relative_path)
    if artifact is None:
        raise LineageSplitError(f'{description} receipt does not bind {relative_path}')
    expected_bytes = getattr(artifact, 'byte_count')
    expected_sha256 = getattr(artifact, 'sha256')
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
        raise LineageSplitError(f'{description} artifact does not match its receipt: {relative_path}')


def _load_trusted_inputs(
    *,
    merge_root: Path,
    expected_merge_receipt_sha256: str,
    relevance_root: Path,
    expected_relevance_receipt_sha256: str,
) -> _TrustedInputs:
    merge_root = merge_root.expanduser().resolve()
    relevance_root = relevance_root.expanduser().resolve()
    merge_payload = _read_regular(merge_root / 'MERGE-RECEIPT.json', description='merge receipt')
    merge_receipt_sha256 = _require_expected_hash(merge_payload, expected_merge_receipt_sha256, 'merge receipt')
    try:
        merge_receipt = AactExecutionMultiAnchorMergeReceipt.model_validate_json(merge_payload)
    except ValueError as error:
        raise LineageSplitError(f'invalid merge receipt: {error}') from error
    merge_artifacts = {item.relative_path: item for item in merge_receipt.artifacts}
    inventory_payload, inventory_artifact_sha256 = _artifact_payload(
        merge_root,
        merge_artifacts,
        'organizer/cohort-inventory.json',
        description='merge',
    )
    try:
        inventory = ExecutionCohortInventory.model_validate_json(inventory_payload)
        audit_execution_inventory(inventory)
    except (ValueError, ExecutionInventoryError) as error:
        raise LineageSplitError(f'merged decision inventory fails deterministic audit: {error}') from error
    if merge_receipt.synthetic or merge_receipt.unique_nct_assignment_count != len(inventory.assignments):
        raise LineageSplitError('lineage build requires the trusted real merge with consistent assignments')

    relevance_payload = _read_regular(relevance_root / 'REVIEW-RECEIPT.json', description='relevance receipt')
    relevance_receipt_sha256 = _require_expected_hash(
        relevance_payload,
        expected_relevance_receipt_sha256,
        'relevance receipt',
    )
    try:
        relevance_receipt = VaccineRelevanceReviewReceipt.model_validate_json(relevance_payload)
    except ValueError as error:
        raise LineageSplitError(f'invalid relevance receipt: {error}') from error
    relevance_artifacts = {item.relative_path: item for item in relevance_receipt.artifacts}
    queue_payload, queue_sha256 = _artifact_payload(
        relevance_root,
        relevance_artifacts,
        'organizer/relevance-review-queue.json',
        description='relevance',
    )
    adjudication_payload, adjudication_sha256 = _artifact_payload(
        relevance_root,
        relevance_artifacts,
        'organizer/relevance-adjudications.json',
        description='relevance',
    )
    policy_payload, relevance_policy_sha256 = _artifact_payload(
        relevance_root,
        relevance_artifacts,
        'organizer/relevance-policy.json',
        description='relevance',
    )
    try:
        queue = VaccineRelevanceReviewQueue.model_validate_json(queue_payload)
        adjudications = VaccineRelevanceAdjudicationSet.model_validate_json(adjudication_payload)
    except ValueError as error:
        raise LineageSplitError(f'invalid relevance artifacts: {error}') from error
    if policy_payload != canonical_json_bytes(ACTIVE_VACCINE_RELEVANCE_POLICY):
        raise LineageSplitError('relevance artifact does not contain the fixed decision-only policy')
    rebuilt = finalize_relevance_adjudications(queue=queue, reviews=adjudications.decisions)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(adjudications):
        raise LineageSplitError('relevance adjudications do not reconstruct from the bound queue')
    if (
        relevance_receipt.include_count,
        relevance_receipt.exclude_count,
        relevance_receipt.hold_count,
    ) != (adjudications.include_count, adjudications.exclude_count, adjudications.hold_count):
        raise LineageSplitError('relevance receipt counts disagree with adjudications')
    if queue.merged_inventory_sha256 != inventory_artifact_sha256:
        raise LineageSplitError('relevance review is not bound to the exact merged inventory artifact')
    assignment_keys = {(item.anchor_date, item.nct_id) for item in inventory.assignments}
    queue_keys = {(item.anchor_date, item.nct_id) for item in queue.records}
    decision_keys = {(item.anchor_date, item.nct_id) for item in adjudications.decisions}
    if assignment_keys != queue_keys or assignment_keys != decision_keys:
        raise LineageSplitError('inventory, relevance queue, and adjudications do not have exact coverage')
    if queue.later_archive_opened or queue.execution_labels_read or relevance_receipt.execution_labels_read:
        raise LineageSplitError('lineage inputs are not outcome-blind')
    return _TrustedInputs(
        merge_receipt_sha256=merge_receipt_sha256,
        inventory_artifact_sha256=inventory_artifact_sha256,
        inventory=inventory,
        relevance_receipt_sha256=relevance_receipt_sha256,
        relevance_receipt=relevance_receipt,
        relevance_queue_artifact_sha256=queue_sha256,
        relevance_adjudication_artifact_sha256=adjudication_sha256,
        relevance_policy_artifact_sha256=relevance_policy_sha256,
        queue=queue,
        adjudications=adjudications,
    )


def _validate_id_key(id_key: bytes) -> None:
    if not isinstance(id_key, bytes) or len(id_key) < 32:
        raise LineageSplitError('opaque-ID HMAC key must contain at least 32 random bytes')


def _hmac_hex(id_key: bytes, domain: bytes, value: str, length: int = 24) -> str:
    return hmac.new(id_key, domain + value.encode('utf-8'), hashlib.sha256).hexdigest()[:length]


def _id_key_commitment(id_key: bytes) -> str:
    return _sha256_bytes(_KEY_COMMITMENT_DOMAIN + id_key)


def _target_counts(case_count: int) -> dict[Split, int]:
    weights = LINEAGE_SPLIT_POLICY.split_weights
    denominator = sum(item.weight for item in weights)
    counts = {item.split: case_count * item.weight // denominator for item in weights}
    remainder = case_count - sum(counts.values())
    fractional_order = sorted(
        weights,
        key=lambda item: (-(case_count * item.weight % denominator), item.split.value),
    )
    for item in fractional_order[:remainder]:
        counts[item.split] += 1
    return counts


def _assign_lineage_splits(family_counts: Mapping[str, int], id_key: bytes) -> dict[str, Split]:
    targets = _target_counts(sum(family_counts.values()))
    current = {split: 0 for split in (Split.TRAIN, Split.DEV, Split.TEST)}
    ordered_families = sorted(
        family_counts,
        key=lambda family_id: (
            -family_counts[family_id],
            _hmac_hex(id_key, _SPLIT_ORDER_DOMAIN, family_id, length=64),
        ),
    )
    result: dict[str, Split] = {}
    for family_id in ordered_families:
        size = family_counts[family_id]
        choices: list[tuple[tuple[int, int, str], Split]] = []
        for split in (Split.TRAIN, Split.DEV, Split.TEST):
            proposed = dict(current)
            proposed[split] += size
            squared_loss = sum((proposed[item] - targets[item]) ** 2 for item in proposed)
            overflow = sum(max(0, proposed[item] - targets[item]) for item in proposed)
            tie = _hmac_hex(id_key, _SPLIT_TIE_DOMAIN, f'{family_id}\x00{split.value}', length=64)
            choices.append(((squared_loss, overflow, tie), split))
        _, selected = min(choices)
        result[family_id] = selected
        current[selected] += size
    if any(current[split] == 0 for split in current):
        raise LineageSplitError('lineage allocation left an empty partition')
    return result


def _derive_assignment_set(inputs: _TrustedInputs, id_key: bytes) -> LineageSplitAssignmentSet:
    _validate_id_key(id_key)
    assignments_by_key = {(item.anchor_date, item.nct_id): item for item in inputs.inventory.assignments}
    records_by_key = {(item.anchor_date, item.nct_id): item for item in inputs.queue.records}
    included = tuple(
        item for item in inputs.adjudications.decisions if item.disposition == RelevanceDisposition.INCLUDE
    )
    classifications: dict[tuple[date, str], FamilyClassification] = {}
    family_counts: dict[str, int] = defaultdict(int)
    for decision in included:
        key = (decision.anchor_date, decision.nct_id)
        record = records_by_key[key]
        if decision.evidence_sha256 != record.evidence_sha256:
            raise LineageSplitError(f'relevance evidence mismatch for {decision.nct_id}')
        classification = classify_target_family(record)
        classifications[key] = classification
        family_counts[classification.target_family_id] += 1
    family_splits = _assign_lineage_splits(family_counts, id_key)
    cases: list[LineageCaseAssignment] = []
    for decision in included:
        key = (decision.anchor_date, decision.nct_id)
        assignment: TrialAnchorAssignment = assignments_by_key[key]
        classification = classifications[key]
        family_id = classification.target_family_id
        task_identity = f'{decision.anchor_date.isoformat()}\x00{decision.nct_id}'
        cases.append(
            LineageCaseAssignment(
                nct_id=decision.nct_id,
                anchor_date=decision.anchor_date,
                opaque_task_id=f'vaxclin-{_hmac_hex(id_key, _TASK_ID_DOMAIN, task_identity)}',
                source_assignment_sha256=_model_sha256(assignment),
                relevance_evidence_sha256=decision.evidence_sha256,
                relevance_decision_sha256=_model_sha256(decision),
                target_family_id=family_id,
                family_match_basis=classification.match_basis,
                matched_terms=classification.matched_terms,
                lineage_group_id=f'clinlin-{_hmac_hex(id_key, _LINEAGE_ID_DOMAIN, family_id)}',
                split=family_splits[family_id],
            )
        )
    ordered_cases = tuple(sorted(cases, key=lambda item: (item.anchor_date, item.nct_id)))
    cases_by_lineage: dict[str, list[LineageCaseAssignment]] = defaultdict(list)
    for case in ordered_cases:
        cases_by_lineage[case.lineage_group_id].append(case)
    lineages = tuple(
        sorted(
            (
                LineageGroupSummary(
                    lineage_group_id=lineage_id,
                    target_family_id=members[0].target_family_id,
                    split=members[0].split,
                    member_count=len(members),
                    nct_ids=tuple(sorted(item.nct_id for item in members)),
                )
                for lineage_id, members in cases_by_lineage.items()
            ),
            key=lambda item: item.lineage_group_id,
        )
    )
    split_counts = tuple(
        SplitCount(
            split=split,
            case_count=sum(item.split == split for item in ordered_cases),
            lineage_count=sum(item.split == split for item in lineages),
        )
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    )
    anchors = tuple(sorted({item.anchor_date for item in ordered_cases}))
    anchor_split_counts = tuple(
        AnchorSplitCount(
            anchor_date=anchor,
            split=split,
            case_count=sum(item.anchor_date == anchor and item.split == split for item in ordered_cases),
        )
        for anchor in anchors
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    )
    return LineageSplitAssignmentSet(
        policy_sha256=lineage_split_policy_sha256(),
        merge_receipt_sha256=inputs.merge_receipt_sha256,
        merged_inventory_artifact_sha256=inputs.inventory_artifact_sha256,
        relevance_review_receipt_sha256=inputs.relevance_receipt_sha256,
        relevance_queue_artifact_sha256=inputs.relevance_queue_artifact_sha256,
        relevance_adjudication_artifact_sha256=inputs.relevance_adjudication_artifact_sha256,
        relevance_policy_artifact_sha256=inputs.relevance_policy_artifact_sha256,
        id_key_commitment_sha256=_id_key_commitment(id_key),
        upstream_include_count=inputs.adjudications.include_count,
        upstream_exclude_count=inputs.adjudications.exclude_count,
        upstream_hold_count=inputs.adjudications.hold_count,
        assignment_count=len(ordered_cases),
        held_or_dropped_include_count=inputs.adjudications.include_count - len(ordered_cases),
        cases=ordered_cases,
        lineages=lineages,
        split_counts=split_counts,
        anchor_split_counts=anchor_split_counts,
    )


def _write_artifact(path: Path, value: object, relative_path: str) -> LineageSplitArtifactReceipt:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    path.parent.chmod(_PRIVATE_DIRECTORY_MODE)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return LineageSplitArtifactReceipt(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
    )


def build_lineage_split(
    *,
    merge_root: Path,
    expected_merge_receipt_sha256: str,
    relevance_root: Path,
    expected_relevance_receipt_sha256: str,
    id_key: bytes,
    output_root: Path,
) -> LineageSplitBuild:
    """Build an immutable organizer-private lineage mapping and split."""

    inputs = _load_trusted_inputs(
        merge_root=merge_root,
        expected_merge_receipt_sha256=expected_merge_receipt_sha256,
        relevance_root=relevance_root,
        expected_relevance_receipt_sha256=expected_relevance_receipt_sha256,
    )
    assignments = _derive_assignment_set(inputs, id_key)
    destination = output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'immutable lineage-split output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.staging-', dir=destination.parent))
    staging.chmod(_PRIVATE_DIRECTORY_MODE)
    try:
        artifacts = tuple(
            sorted(
                (
                    _write_artifact(
                        staging / 'organizer' / 'lineage-split-assignments.json',
                        assignments,
                        'organizer/lineage-split-assignments.json',
                    ),
                    _write_artifact(
                        staging / 'organizer' / 'lineage-split-policy.json',
                        LINEAGE_SPLIT_POLICY,
                        'organizer/lineage-split-policy.json',
                    ),
                ),
                key=lambda item: item.relative_path,
            )
        )
        receipt = LineageSplitReceipt(
            policy_sha256=assignments.policy_sha256,
            merge_receipt_sha256=assignments.merge_receipt_sha256,
            merged_inventory_artifact_sha256=assignments.merged_inventory_artifact_sha256,
            relevance_review_receipt_sha256=assignments.relevance_review_receipt_sha256,
            relevance_queue_artifact_sha256=assignments.relevance_queue_artifact_sha256,
            relevance_adjudication_artifact_sha256=assignments.relevance_adjudication_artifact_sha256,
            id_key_commitment_sha256=assignments.id_key_commitment_sha256,
            assignment_count=assignments.assignment_count,
            lineage_count=len(assignments.lineages),
            upstream_exclude_count=assignments.upstream_exclude_count,
            upstream_hold_count=assignments.upstream_hold_count,
            split_counts=assignments.split_counts,
            artifacts=artifacts,
        )
        _write_artifact(staging / 'LINEAGE-SPLIT-RECEIPT.json', receipt, 'LINEAGE-SPLIT-RECEIPT.json')
        fsync_directory(staging / 'organizer')
        fsync_directory(staging)
        rename_directory_noreplace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return LineageSplitBuild(root=destination, receipt=receipt, assignments=assignments)


def verify_lineage_split_build(
    root: Path,
    *,
    expected_receipt_sha256: str,
    merge_root: Path,
    expected_merge_receipt_sha256: str,
    relevance_root: Path,
    expected_relevance_receipt_sha256: str,
    id_key: bytes,
) -> LineageSplitBuild:
    """Reconstruct a split from external trust anchors and compare exact artifact bytes."""

    resolved, receipt_payload, assignment_payload, policy_payload = _read_closed_lineage_tree(root)
    _require_expected_hash(receipt_payload, expected_receipt_sha256, 'lineage-split receipt')
    try:
        receipt = LineageSplitReceipt.model_validate_json(receipt_payload)
    except ValueError as error:
        raise LineageSplitError(f'invalid lineage-split receipt: {error}') from error
    artifacts = {item.relative_path: item for item in receipt.artifacts}
    _validate_artifact_payload(
        assignment_payload,
        artifacts,
        'organizer/lineage-split-assignments.json',
        description='lineage split',
    )
    _validate_artifact_payload(
        policy_payload,
        artifacts,
        'organizer/lineage-split-policy.json',
        description='lineage split',
    )
    if policy_payload != canonical_json_bytes(LINEAGE_SPLIT_POLICY):
        raise LineageSplitError('lineage-split build does not contain the fixed policy')
    try:
        assignments = LineageSplitAssignmentSet.model_validate_json(assignment_payload)
    except ValueError as error:
        raise LineageSplitError(f'invalid lineage-split assignments: {error}') from error
    inputs = _load_trusted_inputs(
        merge_root=merge_root,
        expected_merge_receipt_sha256=expected_merge_receipt_sha256,
        relevance_root=relevance_root,
        expected_relevance_receipt_sha256=expected_relevance_receipt_sha256,
    )
    rebuilt = _derive_assignment_set(inputs, id_key)
    if canonical_json_bytes(rebuilt) != assignment_payload:
        raise LineageSplitError('lineage assignments do not reconstruct from trusted decision-only inputs')
    expected_receipt = LineageSplitReceipt(
        policy_sha256=rebuilt.policy_sha256,
        merge_receipt_sha256=rebuilt.merge_receipt_sha256,
        merged_inventory_artifact_sha256=rebuilt.merged_inventory_artifact_sha256,
        relevance_review_receipt_sha256=rebuilt.relevance_review_receipt_sha256,
        relevance_queue_artifact_sha256=rebuilt.relevance_queue_artifact_sha256,
        relevance_adjudication_artifact_sha256=rebuilt.relevance_adjudication_artifact_sha256,
        id_key_commitment_sha256=rebuilt.id_key_commitment_sha256,
        assignment_count=rebuilt.assignment_count,
        lineage_count=len(rebuilt.lineages),
        upstream_exclude_count=rebuilt.upstream_exclude_count,
        upstream_hold_count=rebuilt.upstream_hold_count,
        split_counts=rebuilt.split_counts,
        artifacts=receipt.artifacts,
    )
    if canonical_json_bytes(expected_receipt) != receipt_payload:
        raise LineageSplitError('lineage-split receipt does not match reconstructed assignments')
    return LineageSplitBuild(root=resolved, receipt=receipt, assignments=assignments)


def read_private_id_key(path: Path) -> bytes:
    """Read a raw HMAC key while rejecting symlinks, non-regular files, and broad permissions."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise LineageSplitError('opaque-ID key cannot be a symbolic link')
    resolved = expanded.resolve()
    try:
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise LineageSplitError(f'cannot stat opaque-ID key: {error}') from error
    if not stat.S_ISREG(metadata.st_mode):
        raise LineageSplitError('opaque-ID key must be a regular file')
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LineageSplitError('opaque-ID key must not be accessible by group or other users')
    key = _read_regular(resolved, description='opaque-ID key')
    _validate_id_key(key)
    return key


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'verify'):
        child = subparsers.add_parser(command)
        child.add_argument('--root' if command == 'verify' else '--output-root', type=Path, required=True)
        child.add_argument('--merge-root', type=Path, required=True)
        child.add_argument('--expected-merge-receipt-sha256', required=True)
        child.add_argument('--relevance-root', type=Path, required=True)
        child.add_argument('--expected-relevance-receipt-sha256', required=True)
        child.add_argument('--id-key-file', type=Path, required=True)
        if command == 'verify':
            child.add_argument('--expected-receipt-sha256', required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    key = read_private_id_key(args.id_key_file)
    common = {
        'merge_root': args.merge_root,
        'expected_merge_receipt_sha256': args.expected_merge_receipt_sha256,
        'relevance_root': args.relevance_root,
        'expected_relevance_receipt_sha256': args.expected_relevance_receipt_sha256,
        'id_key': key,
    }
    if args.command == 'build':
        build = build_lineage_split(output_root=args.output_root, **common)
    else:
        build = verify_lineage_split_build(
            args.root,
            expected_receipt_sha256=args.expected_receipt_sha256,
            **common,
        )
    print(canonical_json_bytes(build.receipt).decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
