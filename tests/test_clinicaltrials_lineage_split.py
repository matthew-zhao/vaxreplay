from __future__ import annotations

import hashlib
import os
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

import vaxreplay.clinicaltrials.lineage_split as lineage_split_module
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.clinicaltrials.lineage_split import (
    LINEAGE_SPLIT_POLICY,
    AnchorSplitCount,
    FamilyClassification,
    LineageCaseAssignment,
    LineageGroupSummary,
    LineageSplitAssignmentSet,
    LineageSplitError,
    SplitCount,
    _assign_lineage_splits,
    _target_counts,
    build_lineage_split,
    classify_target_family,
    lineage_split_policy_sha256,
    read_private_id_key,
    verify_lineage_split_build,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    DecisionEvidenceBody,
    DecisionEvidenceRecord,
    EvidenceSourceRow,
    InterventionEvidence,
    SponsorEvidence,
)


def _record(
    *,
    nct_id: str = 'NCT00000001',
    conditions: tuple[str, ...],
    title: str,
    intervention: str = 'Candidate V1',
    description: str = 'Investigational vaccine candidate',
) -> DecisionEvidenceRecord:
    body = DecisionEvidenceBody(
        anchor_date=date(2020, 2, 1),
        snapshot_id='aact-flatfiles-2020-02-01',
        decision_archive_sha256='a' * 64,
        nct_id=nct_id,
        brief_title=title,
        official_title=title,
        acronym='',
        primary_purposes=('Prevention',),
        conditions=conditions,
        interventions=(
            InterventionEvidence(
                intervention_type='Biological',
                name=intervention,
                description=description,
            ),
        ),
        brief_summary='',
        detailed_description='',
        sponsors=(
            SponsorEvidence(
                lead_or_collaborator='lead',
                agency_class='INDUSTRY',
                name='Example Sponsor',
            ),
        ),
        source_rows=(
            EvidenceSourceRow(
                member_path='studies.txt',
                data_row_number=1,
                raw_row_sha256='b' * 64,
                fields_read=('brief_title', 'nct_id'),
            ),
        ),
    )
    return DecisionEvidenceRecord(
        **body.model_dump(),
        evidence_sha256=hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    )


def _test_assignment_set() -> LineageSplitAssignmentSet:
    anchor = date(2020, 2, 1)
    case = LineageCaseAssignment(
        nct_id='NCT00000001',
        anchor_date=anchor,
        opaque_task_id=f'vaxclin-{"1" * 24}',
        source_assignment_sha256='1' * 64,
        relevance_evidence_sha256='2' * 64,
        relevance_decision_sha256='3' * 64,
        target_family_id='anthrax',
        family_match_basis='conditions',
        matched_terms=('anthrax',),
        lineage_group_id=f'clinlin-{"4" * 24}',
        split=Split.TRAIN,
    )
    lineage = LineageGroupSummary(
        lineage_group_id=case.lineage_group_id,
        target_family_id=case.target_family_id,
        split=case.split,
        member_count=1,
        nct_ids=(case.nct_id,),
    )
    return LineageSplitAssignmentSet(
        policy_sha256=lineage_split_policy_sha256(),
        merge_receipt_sha256='a' * 64,
        merged_inventory_artifact_sha256='b' * 64,
        relevance_review_receipt_sha256='c' * 64,
        relevance_queue_artifact_sha256='d' * 64,
        relevance_adjudication_artifact_sha256='e' * 64,
        relevance_policy_artifact_sha256='f' * 64,
        id_key_commitment_sha256='0' * 64,
        upstream_include_count=1,
        upstream_exclude_count=0,
        upstream_hold_count=0,
        assignment_count=1,
        held_or_dropped_include_count=0,
        cases=(case,),
        lineages=(lineage,),
        split_counts=(
            SplitCount(split=Split.TRAIN, case_count=1, lineage_count=1),
            SplitCount(split=Split.DEV, case_count=0, lineage_count=0),
            SplitCount(split=Split.TEST, case_count=0, lineage_count=0),
        ),
        anchor_split_counts=(
            AnchorSplitCount(anchor_date=anchor, split=Split.TRAIN, case_count=1),
            AnchorSplitCount(anchor_date=anchor, split=Split.DEV, case_count=0),
            AnchorSplitCount(anchor_date=anchor, split=Split.TEST, case_count=0),
        ),
    )


def _build_test_lineage_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> tuple[Path, str]:
    assignments = _test_assignment_set()
    monkeypatch.setattr(lineage_split_module, '_load_trusted_inputs', lambda **_: object())
    monkeypatch.setattr(lineage_split_module, '_derive_assignment_set', lambda *_: assignments)
    root = tmp_path / name
    build = build_lineage_split(
        merge_root=tmp_path / 'unused-merge',
        expected_merge_receipt_sha256='1' * 64,
        relevance_root=tmp_path / 'unused-relevance',
        expected_relevance_receipt_sha256='2' * 64,
        id_key=b'k' * 32,
        output_root=root,
    )
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(build.receipt)).hexdigest()
    return root, receipt_sha256


def _verify_test_lineage_tree(root: Path, receipt_sha256: str, tmp_path: Path) -> None:
    verify_lineage_split_build(
        root,
        expected_receipt_sha256=receipt_sha256,
        merge_root=tmp_path / 'unused-merge',
        expected_merge_receipt_sha256='1' * 64,
        relevance_root=tmp_path / 'unused-relevance',
        expected_relevance_receipt_sha256='2' * 64,
        id_key=b'k' * 32,
    )


def test_policy_and_taxonomy_are_content_addressed() -> None:
    assert lineage_split_policy_sha256() == hashlib.sha256(canonical_json_bytes(LINEAGE_SPLIT_POLICY)).hexdigest()
    assert lineage_split_policy_sha256() == 'ab40e20e3b377a7226c783432b15ca2b76518006c364ba77df51cb866cfb17d7'
    assert len(LINEAGE_SPLIT_POLICY.target_family_rules) == 24
    assert LINEAGE_SPLIT_POLICY.decision_time_evidence_only
    assert not LINEAGE_SPLIT_POLICY.execution_labels_read
    assert not LINEAGE_SPLIT_POLICY.leaderboard_admitted


def test_conditions_are_primary_and_token_boundaries_prevent_influenza_collision() -> None:
    record = _record(
        conditions=('Parainfluenza',),
        title='Influenza appears only in non-primary fallback text',
        intervention='Sendai virus vaccine',
    )
    assert classify_target_family(record) == FamilyClassification(
        target_family_id='parainfluenza-metapneumovirus',
        match_basis='conditions',
        matched_terms=('parainfluenza',),
    )


def test_generic_conditions_use_decision_text_fallback() -> None:
    record = _record(
        conditions=('Healthy Volunteers',),
        title='NasoShield Study of Safety and Immunogenicity',
        intervention='NasoShield',
        description='NasoShield is an adenovirus-vectored anthrax vaccine.',
    )
    classification = classify_target_family(record)
    assert classification.target_family_id == 'anthrax'
    assert classification.match_basis == 'decision_text_fallback'
    assert classification.matched_terms == ('anthrax', 'nasoshield')


def test_ambiguous_or_unresolved_target_family_fails_closed() -> None:
    ambiguous = _record(
        conditions=('HIV Infections', 'Malaria'),
        title='Ambiguous multi-target candidate',
    )
    with pytest.raises(LineageSplitError, match='resolved to'):
        classify_target_family(ambiguous)

    unresolved = _record(
        conditions=('Healthy Volunteers',),
        title='Candidate with no target identity',
    )
    with pytest.raises(LineageSplitError, match='resolved to'):
        classify_target_family(unresolved)


def test_real_cohort_family_sizes_produce_complete_79_26_26_split() -> None:
    family_counts = {
        'alphavirus': 4,
        'anthrax': 2,
        'clostridioides-difficile': 1,
        'cytomegalovirus': 1,
        'diphtheria-pertussis-tetanus': 3,
        'enteric-bacterial': 3,
        'enteric-viral': 3,
        'filovirus': 4,
        'flavivirus': 11,
        'hantavirus': 1,
        'helminth': 3,
        'hepatitis': 3,
        'hiv': 20,
        'hpv': 2,
        'influenza': 13,
        'malaria': 18,
        'mers-coronavirus': 5,
        'nipah-henipavirus': 1,
        'parainfluenza-metapneumovirus': 3,
        'pneumococcus': 8,
        'polio': 1,
        'rabies': 2,
        'respiratory-syncytial-virus': 14,
        'tuberculosis': 5,
    }
    key = b'fixed-test-only-hmac-key-material' * 2
    first = _assign_lineage_splits(family_counts, key)
    second = _assign_lineage_splits(dict(reversed(tuple(family_counts.items()))), key)
    assert first == second
    assert set(first) == set(family_counts)
    observed = Counter[Split]()
    for family_id, split in first.items():
        observed[split] += family_counts[family_id]
    assert _target_counts(sum(family_counts.values())) == {
        Split.TRAIN: 79,
        Split.DEV: 26,
        Split.TEST: 26,
    }
    assert observed == Counter({Split.TRAIN: 79, Split.DEV: 26, Split.TEST: 26})


def test_private_id_key_requires_entropy_and_private_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / 'opaque-id.key'
    key_path.write_bytes(b'k' * 32)
    os.chmod(key_path, 0o600)
    assert read_private_id_key(key_path) == b'k' * 32

    os.chmod(key_path, 0o644)
    with pytest.raises(LineageSplitError, match='group or other'):
        read_private_id_key(key_path)

    os.chmod(key_path, 0o600)
    key_path.write_bytes(b'too-short')
    with pytest.raises(LineageSplitError, match='at least 32'):
        read_private_id_key(key_path)


def test_lineage_verifier_accepts_only_the_exact_closed_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt_sha256 = _build_test_lineage_tree(tmp_path, monkeypatch, 'lineage-extra')
    _verify_test_lineage_tree(root, receipt_sha256, tmp_path)

    extra = root / 'organizer' / 'unbound.json'
    extra.write_bytes(b'{}')
    extra.chmod(0o600)
    with pytest.raises(LineageSplitError, match='exact closed tree'):
        _verify_test_lineage_tree(root, receipt_sha256, tmp_path)


def test_lineage_verifier_rejects_missing_and_nonregular_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_root, missing_sha256 = _build_test_lineage_tree(tmp_path, monkeypatch, 'lineage-missing')
    missing_policy = missing_root / 'organizer' / 'lineage-split-policy.json'
    missing_policy.unlink()
    with pytest.raises(LineageSplitError, match='exact closed tree'):
        _verify_test_lineage_tree(missing_root, missing_sha256, tmp_path)

    nonregular_root, nonregular_sha256 = _build_test_lineage_tree(
        tmp_path,
        monkeypatch,
        'lineage-nonregular',
    )
    nonregular_policy = nonregular_root / 'organizer' / 'lineage-split-policy.json'
    nonregular_policy.unlink()
    nonregular_policy.mkdir(mode=0o700)
    with pytest.raises(LineageSplitError, match='regular file'):
        _verify_test_lineage_tree(nonregular_root, nonregular_sha256, tmp_path)


def test_lineage_verifier_rejects_symlinked_root_or_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt_sha256 = _build_test_lineage_tree(tmp_path, monkeypatch, 'lineage-symlink-root')
    root_alias = tmp_path / 'lineage-root-alias'
    root_alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(LineageSplitError):
        _verify_test_lineage_tree(root_alias, receipt_sha256, tmp_path)

    parent_root, parent_sha256 = _build_test_lineage_tree(
        tmp_path,
        monkeypatch,
        'actual-parent/lineage',
    )
    parent_alias = tmp_path / 'parent-alias'
    parent_alias.symlink_to(parent_root.parent, target_is_directory=True)
    with pytest.raises(LineageSplitError):
        _verify_test_lineage_tree(parent_alias / parent_root.name, parent_sha256, tmp_path)

    artifact_root, artifact_sha256 = _build_test_lineage_tree(
        tmp_path,
        monkeypatch,
        'lineage-symlink-artifact',
    )
    artifact = artifact_root / 'organizer' / 'lineage-split-policy.json'
    external = tmp_path / 'external-policy.json'
    external.write_bytes(artifact.read_bytes())
    external.chmod(0o600)
    artifact.unlink()
    artifact.symlink_to(external)
    with pytest.raises(LineageSplitError, match='symbolic link'):
        _verify_test_lineage_tree(artifact_root, artifact_sha256, tmp_path)


def test_lineage_verifier_requires_exact_private_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt_sha256 = _build_test_lineage_tree(tmp_path, monkeypatch, 'lineage-modes')
    mutations = (
        (root, 0o755, 0o700),
        (root / 'organizer', 0o750, 0o700),
        (root / 'LINEAGE-SPLIT-RECEIPT.json', 0o644, 0o600),
        (root / 'organizer' / 'lineage-split-assignments.json', 0o640, 0o600),
    )
    for path, invalid_mode, valid_mode in mutations:
        path.chmod(invalid_mode)
        with pytest.raises(LineageSplitError, match='must have mode'):
            _verify_test_lineage_tree(root, receipt_sha256, tmp_path)
        path.chmod(valid_mode)

    _verify_test_lineage_tree(root, receipt_sha256, tmp_path)
