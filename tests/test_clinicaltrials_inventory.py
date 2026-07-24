from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.inventory import (
    AactInventoryIntegrityError,
    build_candidate_inventory,
    verify_candidate_inventory,
)
from vaxreplay.clinicaltrials.inventory_catalog import (
    AactCatalogIntegrityError,
    FrozenOfficialArchiveListing,
    build_archive_acquisition_plan,
    build_official_archive_catalog,
    verify_archive_acquisition_plan,
    verify_official_archive_catalog,
)
from vaxreplay.clinicaltrials.inventory_cli import main as inventory_cli_main
from vaxreplay.clinicaltrials.inventory_schema import (
    AactArchiveAcquisitionRole,
    AactCandidateChronology,
    AactCandidateDisposition,
    AactCandidateInventoryRecord,
    AactDecisionClass,
    AactGateResult,
    AactGateStatus,
    AactInventoryGate,
    AactInventoryReason,
    AactInventorySnapshotBinding,
    AactPostedDateType,
    AactScreenCounts,
    aact_inventory_model_sha256,
    aact_inventory_records_sha256,
    aact_mechanical_gate_expectations,
)

_HASH_A = 'a' * 64
_HASH_B = 'b' * 64
_HASH_C = 'c' * 64
_HASH_D = 'd' * 64
_RETRIEVED = datetime(2026, 7, 14, 22, 0, tzinfo=UTC)


def _row(date_text: str, file_name: str, size: str, *, href_date: str | None = None) -> str:
    iso_date = href_date or datetime.strptime(date_text, '%m-%d-%Y').date().isoformat()
    return f"""
      <div class="snapshots-grid-row">
        <div data-label="Date:"><strong>{date_text}</strong></div>
        <div data-label="File:">{file_name}</div>
        <div data-label="Size:">{size}</div>
        <div class="snapshots-grid-actions">
          <a href="/static/exported_files/daily/{iso_date}?source=web">Download</a>
        </div>
      </div>
    """


def _listing_html(year: int, rows: tuple[str, ...]) -> bytes:
    recent = _row('07-14-2026', '20260714_export_ctgov.zip', '2.31 GB')
    return f"""<!doctype html>
    <html><body>
      <h2 class="snapshots-section-title">Recent Daily Snapshots</h2>
      <div class="snapshots-grid-table">{recent}</div>
      <h2 class="snapshots-section-title">Monthly Archives</h2>
      <nav><a class="selected-year">{year}</a></nav>
      <div class="snapshots-grid-table">
        <div class="snapshots-grid-row snapshots-grid-header"><div>Date</div></div>
        {''.join(rows)}
      </div>
    </body></html>""".encode()


def _frozen_listing(year: int, rows: tuple[str, ...]) -> FrozenOfficialArchiveListing:
    return FrozenOfficialArchiveListing(
        year=year,
        source_url=(f'https://aact.ctti-clinicaltrials.org/downloads/snapshots?type=flatfiles&year={year}'),
        retrieved_at=_RETRIEVED,
        payload=_listing_html(year, rows),
    )


def _catalog():
    listing_2020 = _frozen_listing(
        2020,
        (
            _row('05-01-2020', '20200501_pipe-delimited-export.zip', '1.01 GB'),
            _row('02-01-2020', '20200201_pipe-delimited-export.zip', '996.84 MB'),
        ),
    )
    listing_2021 = _frozen_listing(
        2021,
        (_row('10-04-2021', '20211004_pipe-delimited-export.zip', '1.21 GB'),),
    )
    catalog = build_official_archive_catalog(
        catalog_id='aact-inventory-2026-07',
        generated_at=_RETRIEVED,
        parser_implementation_sha256=_HASH_A,
        listings=(listing_2021, listing_2020),
    )
    return catalog, (listing_2020, listing_2021)


def _plan(catalog):
    return build_archive_acquisition_plan(
        plan_id='aact-acquisition-2026-07',
        created_at=_RETRIEVED,
        catalog=catalog,
        screening_policy_sha256=_HASH_B,
        requested_roles={
            'aact-flatfiles-2020-02-01': (
                AactArchiveAcquisitionRole.DECISION_CANDIDATE,
                AactArchiveAcquisitionRole.DISCOVERY,
            ),
            'aact-flatfiles-2020-05-01': (AactArchiveAcquisitionRole.LABEL_CANDIDATE,),
            'aact-flatfiles-2021-10-04': (AactArchiveAcquisitionRole.CONFIRMATION_CANDIDATE,),
        },
    )


def _binding(catalog, snapshot_id: str, *, sliced: bool = False) -> AactInventorySnapshotBinding:
    entry = next(entry for entry in catalog.entries if entry.snapshot_id == snapshot_id)
    return AactInventorySnapshotBinding(
        snapshot_id=entry.snapshot_id,
        archive_date=entry.archive_date,
        catalog_entry_sha256=aact_inventory_model_sha256(entry),
        slice_receipt_sha256=_HASH_D if sliced else None,
    )


def _counts() -> AactScreenCounts:
    return AactScreenCounts(
        design_group_count=4,
        experimental_group_count=3,
        comparator_like_group_count=1,
        biological_linked_group_count=3,
        planned_immune_endpoint_count=2,
        reported_immune_endpoint_count=2,
        outcome_result_group_count=4,
        numeric_outcome_group_count=4,
    )


def _reason_union(results: tuple[AactGateResult, ...]) -> tuple[AactInventoryReason, ...]:
    return tuple(
        sorted({reason for result in results for reason in result.reason_codes}, key=lambda reason: reason.value)
    )


def _gate_results(
    chronology: AactCandidateChronology,
    *,
    fully_reviewed: bool,
    study_type: str = 'Interventional',
    allocation: str = 'Randomized',
    intervention_model: str = 'Parallel Assignment',
    primary_purpose: str = 'Prevention',
    phase: str = 'Phase 1',
    counts: AactScreenCounts | None = None,
) -> tuple[AactGateResult, ...]:
    resolved_counts = counts or _counts()
    mechanical = aact_mechanical_gate_expectations(
        study_type=study_type,
        allocation=allocation,
        intervention_model=intervention_model,
        primary_purpose=primary_purpose,
        phase=phase,
        counts=resolved_counts,
        chronology=chronology,
    )
    results: list[AactGateResult] = []
    for gate in sorted(AactInventoryGate, key=lambda item: item.value):
        if gate in mechanical:
            status, reasons = mechanical[gate]
            results.append(
                AactGateResult(
                    gate=gate,
                    status=status,
                    reason_codes=reasons,
                    evidence_sha256=(_HASH_A,),
                )
            )
        elif fully_reviewed:
            results.append(AactGateResult(gate=gate, status=AactGateStatus.PASS, evidence_sha256=(_HASH_A,)))
        else:
            results.append(
                AactGateResult(
                    gate=gate,
                    status=AactGateStatus.NOT_ASSESSED,
                    reason_codes=(AactInventoryReason.GATE_NOT_ASSESSED,),
                )
            )
    return tuple(results)


def _held_record(catalog) -> AactCandidateInventoryRecord:
    chronology = AactCandidateChronology(
        study_first_posted_date=date(2019, 1, 24),
        start_date_lower_bound=date(2019, 1, 21),
        start_date_lower_bound_source_sha256=_HASH_A,
        later_actual_start_date=date(2019, 1, 21),
        later_actual_start_date_source_sha256=_HASH_B,
        results_first_posted_date=date(2020, 3, 1),
        results_first_posted_date_type=AactPostedDateType.ACTUAL,
        results_first_posted_date_source_sha256=_HASH_C,
    )
    gates = _gate_results(chronology, fully_reviewed=False)
    return AactCandidateInventoryRecord(
        inventory_id='aact-candidates-2026-07',
        nct_id='NCT00000001',
        discovery_snapshot=_binding(catalog, 'aact-flatfiles-2020-02-01'),
        chronology=chronology,
        study_type='Interventional',
        allocation='Randomized',
        intervention_model='Parallel Assignment',
        primary_purpose='Prevention',
        phase='Phase 1',
        counts=_counts(),
        gate_results=gates,
        reason_codes=_reason_union(gates),
        disposition=AactCandidateDisposition.HOLD,
    )


def _admitted_record(catalog) -> AactCandidateInventoryRecord:
    held = _held_record(catalog)
    chronology = AactCandidateChronology(
        study_first_posted_date=date(2020, 1, 1),
        start_date_lower_bound=date(2020, 3, 1),
        start_date_lower_bound_source_sha256=_HASH_A,
        later_actual_start_date=date(2020, 3, 15),
        later_actual_start_date_source_sha256=_HASH_B,
        results_first_posted_date=date(2020, 4, 1),
        results_first_posted_date_type=AactPostedDateType.ACTUAL,
        results_first_posted_date_source_sha256=_HASH_C,
    )
    return AactCandidateInventoryRecord.model_validate(
        {
            **held.model_dump(mode='python'),
            'chronology': chronology,
            'decision_class': AactDecisionClass.PRE_ENROLLMENT,
            'decision_snapshot': _binding(catalog, 'aact-flatfiles-2020-02-01', sliced=True),
            'label_snapshot': _binding(catalog, 'aact-flatfiles-2020-05-01', sliced=True),
            'confirmation_snapshot': _binding(catalog, 'aact-flatfiles-2021-10-04', sliced=True),
            'value_hidden_mapping_sha256': _HASH_A,
            'label_stability_sha256': _HASH_B,
            'pre_cutoff_evidence_sha256': _HASH_C,
            'lineage_group_id': 'program-001',
            'lineage_adjudication_sha256': _HASH_D,
            'gate_results': _gate_results(chronology, fully_reviewed=True),
            'reason_codes': (),
            'disposition': AactCandidateDisposition.ADMIT_PRE_ENROLLMENT,
        }
    )


def test_catalog_parser_uses_only_permanent_rows_and_canonicalizes_order() -> None:
    catalog, listings = _catalog()

    assert [entry.snapshot_id for entry in catalog.entries] == [
        'aact-flatfiles-2020-02-01',
        'aact-flatfiles-2020-05-01',
        'aact-flatfiles-2021-10-04',
    ]
    assert all(entry.archive_date != date(2026, 7, 14) for entry in catalog.entries)
    assert (
        catalog.source_pages_sha256
        == hashlib.sha256(b''.join(canonical_json_bytes(page) + b'\n' for page in catalog.pages)).hexdigest()
    )
    assert (
        catalog.entries_sha256
        == hashlib.sha256(b''.join(canonical_json_bytes(entry) + b'\n' for entry in catalog.entries)).hexdigest()
    )
    verify_official_archive_catalog(catalog, listings)

    rebuilt = build_official_archive_catalog(
        catalog_id=catalog.catalog_id,
        generated_at=catalog.generated_at,
        parser_implementation_sha256=catalog.parser_implementation_sha256,
        listings=tuple(reversed(listings)),
    )
    assert canonical_json_bytes(rebuilt) == canonical_json_bytes(catalog)


def test_catalog_verification_rejects_changed_listing_bytes() -> None:
    catalog, listings = _catalog()
    changed = listings[0].payload.replace(b'996.84 MB', b'996.85 MB')
    altered = FrozenOfficialArchiveListing(
        year=listings[0].year,
        source_url=listings[0].source_url,
        retrieved_at=listings[0].retrieved_at,
        payload=changed,
    )

    with pytest.raises(AactCatalogIntegrityError, match='does not match'):
        verify_official_archive_catalog(catalog, (altered, listings[1]))


@pytest.mark.parametrize(
    ('row', 'message'),
    [
        (
            _row(
                '02-01-2020',
                '20200201_pipe-delimited-export.zip',
                '1 GB',
                href_date='2020-03-01',
            ),
            'download path',
        ),
        (
            _row('02-01-2021', '20210201_pipe-delimited-export.zip', '1 GB'),
            'outside its selected listing year',
        ),
    ],
)
def test_catalog_parser_rejects_mismatched_official_rows(row: str, message: str) -> None:
    listing = _frozen_listing(2020, (row,))
    with pytest.raises(AactCatalogIntegrityError, match=message):
        build_official_archive_catalog(
            catalog_id='catalog-test',
            generated_at=_RETRIEVED,
            parser_implementation_sha256=_HASH_A,
            listings=(listing,),
        )


def test_catalog_parser_preserves_official_file_name_date_discrepancy() -> None:
    listing = _frozen_listing(
        2020,
        (_row('02-01-2020', '20200131_pipe-delimited-export.zip', '1 GB'),),
    )
    catalog = build_official_archive_catalog(
        catalog_id='catalog-date-discrepancy',
        generated_at=_RETRIEVED,
        parser_implementation_sha256=_HASH_A,
        listings=(listing,),
    )

    entry = catalog.entries[0]
    assert entry.archive_date == date(2020, 2, 1)
    assert entry.file_name_date == date(2020, 1, 31)
    assert entry.file_name_date_matches_archive_date is False
    assert entry.source_url.endswith('/2020-02-01?source=web')


def test_acquisition_plan_binds_every_item_to_catalog() -> None:
    catalog, _ = _catalog()
    plan = _plan(catalog)

    assert [item.snapshot_id for item in plan.items] == [
        'aact-flatfiles-2020-02-01',
        'aact-flatfiles-2020-05-01',
        'aact-flatfiles-2021-10-04',
    ]
    verify_archive_acquisition_plan(plan, catalog)

    altered_item = plan.items[0].model_copy(update={'catalog_entry_sha256': _HASH_C})
    altered_plan = plan.model_copy(update={'items': (altered_item, *plan.items[1:])})
    with pytest.raises(AactCatalogIntegrityError, match='hash differs'):
        verify_archive_acquisition_plan(altered_plan, catalog)

    with pytest.raises(AactCatalogIntegrityError, match='unknown snapshots'):
        build_archive_acquisition_plan(
            plan_id='bad-plan',
            created_at=_RETRIEVED,
            catalog=catalog,
            screening_policy_sha256=_HASH_B,
            requested_roles={
                'aact-flatfiles-1999-01-01': (AactArchiveAcquisitionRole.DISCOVERY,),
            },
        )


def test_gate_results_require_gate_specific_explicit_reasons() -> None:
    with pytest.raises(ValueError, match='require an explicit reason'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.NOT_ASSESSED,
        )
    with pytest.raises(ValueError, match='not valid for gate'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.FAIL,
            reason_codes=(AactInventoryReason.ENDPOINTS_INCOMPARABLE,),
        )
    with pytest.raises(ValueError, match='passing gates cannot'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.PASS,
            reason_codes=(AactInventoryReason.ARM_MAPPING_AMBIGUOUS,),
        )
    with pytest.raises(ValueError, match='failed gates cannot use'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.FAIL,
            reason_codes=(AactInventoryReason.GATE_NOT_ASSESSED,),
        )
    with pytest.raises(ValueError, match='require exact evidence hashes'):
        AactGateResult(gate=AactInventoryGate.ARM_MAPPING, status=AactGateStatus.PASS)
    with pytest.raises(ValueError, match='require exact evidence hashes'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.FAIL,
            reason_codes=(AactInventoryReason.ARM_MAPPING_AMBIGUOUS,),
        )
    with pytest.raises(ValueError, match='cannot claim reviewed evidence'):
        AactGateResult(
            gate=AactInventoryGate.ARM_MAPPING,
            status=AactGateStatus.NOT_ASSESSED,
            reason_codes=(AactInventoryReason.ARM_MAPPING_NOT_ASSESSED,),
            evidence_sha256=(_HASH_A,),
        )


def test_candidate_inventory_is_exhaustive_sorted_and_cross_bound() -> None:
    catalog, _ = _catalog()
    plan = _plan(catalog)
    second = _held_record(catalog).model_copy(update={'nct_id': 'NCT00000002'})
    first = _held_record(catalog)
    inventory = build_candidate_inventory(
        inventory_id='aact-candidates-2026-07',
        created_at=_RETRIEVED,
        catalog=catalog,
        acquisition_plan=plan,
        screening_policy_sha256=_HASH_B,
        gate_policy_sha256=_HASH_A,
        lineage_policy_sha256=_HASH_C,
        masking_policy_sha256=_HASH_D,
        records=(second, first),
    )

    assert [record.nct_id for record in inventory.records] == ['NCT00000001', 'NCT00000002']
    assert inventory.record_count == 2
    verify_candidate_inventory(inventory, catalog, plan)

    wrong_binding = first.discovery_snapshot.model_copy(update={'catalog_entry_sha256': _HASH_C})
    wrong_record = first.model_copy(update={'discovery_snapshot': wrong_binding})
    wrong_inventory = inventory.model_copy(update={'records': (wrong_record, second)})
    with pytest.raises(AactInventoryIntegrityError, match='differs from the official catalog'):
        verify_candidate_inventory(wrong_inventory, catalog, plan)

    tampered_item = plan.items[0].model_copy(update={'target_relative_path': 'archives/forged.zip'})
    tampered_items = (tampered_item, *plan.items[1:])
    tampered_plan = plan.model_copy(
        update={
            'items': tampered_items,
            'items_sha256': aact_inventory_records_sha256(tampered_items),
        }
    )
    inventory_bound_to_tampered_plan = inventory.model_copy(
        update={'acquisition_plan_sha256': aact_inventory_model_sha256(tampered_plan)}
    )
    with pytest.raises(AactInventoryIntegrityError, match='acquisition target path differs'):
        verify_candidate_inventory(inventory_bound_to_tampered_plan, catalog, tampered_plan)


def test_forged_observational_zero_arm_record_cannot_be_admitted() -> None:
    catalog, _ = _catalog()
    plan = _plan(catalog)
    admitted = _admitted_record(catalog)
    inventory = build_candidate_inventory(
        inventory_id=admitted.inventory_id,
        created_at=_RETRIEVED,
        catalog=catalog,
        acquisition_plan=plan,
        screening_policy_sha256=_HASH_B,
        gate_policy_sha256=_HASH_A,
        lineage_policy_sha256=_HASH_C,
        masking_policy_sha256=_HASH_D,
        records=(admitted,),
    )
    zero_counts = AactScreenCounts(
        design_group_count=0,
        experimental_group_count=0,
        comparator_like_group_count=0,
        biological_linked_group_count=0,
        planned_immune_endpoint_count=0,
        reported_immune_endpoint_count=0,
        outcome_result_group_count=0,
        numeric_outcome_group_count=0,
    )
    forged_record = admitted.model_copy(update={'study_type': 'Observational', 'counts': zero_counts})
    forged_records = (forged_record,)
    forged_inventory = inventory.model_copy(
        update={
            'records': forged_records,
            'records_sha256': aact_inventory_records_sha256(forged_records),
        }
    )

    with pytest.raises(AactInventoryIntegrityError, match='forged or inconsistent interventional gate'):
        verify_candidate_inventory(forged_inventory, catalog, plan)


def test_candidate_disposition_and_admission_require_complete_proof_state() -> None:
    catalog, _ = _catalog()
    held = _held_record(catalog)
    with pytest.raises(ValueError, match='excluded candidates require'):
        AactCandidateInventoryRecord.model_validate(
            {**held.model_dump(mode='python'), 'disposition': AactCandidateDisposition.EXCLUDE}
        )

    passing = _gate_results(held.chronology, fully_reviewed=True)
    historically_late = {
        **held.model_dump(mode='python'),
        'decision_class': AactDecisionClass.PRE_ENROLLMENT,
        'decision_snapshot': _binding(catalog, 'aact-flatfiles-2020-02-01', sliced=True),
        'gate_results': passing,
        'reason_codes': (),
        'disposition': AactCandidateDisposition.ADMIT_PRE_ENROLLMENT,
    }
    with pytest.raises(ValueError, match='strictly predate the start lower bound'):
        AactCandidateInventoryRecord.model_validate(historically_late)

    admitted = _admitted_record(catalog).model_dump(mode='python')
    validated = AactCandidateInventoryRecord.model_validate(admitted)
    assert validated.disposition == AactCandidateDisposition.ADMIT_PRE_ENROLLMENT

    admitted['value_hidden_mapping_sha256'] = None
    with pytest.raises(ValueError, match='mapping, evidence, stability, and lineage'):
        AactCandidateInventoryRecord.model_validate(admitted)


def test_admission_chronology_is_source_bound_and_conservative() -> None:
    catalog, _ = _catalog()
    admitted = _admitted_record(catalog)

    with pytest.raises(ValueError, match='start-date lower bound and exact source hash'):
        AactCandidateChronology(
            study_first_posted_date=date(2020, 1, 1),
            start_date_lower_bound=date(2020, 3, 1),
            results_first_posted_date=date(2020, 4, 1),
            results_first_posted_date_type=AactPostedDateType.ACTUAL,
            results_first_posted_date_source_sha256=_HASH_A,
        )
    with pytest.raises(ValueError, match='requires an exact source hash'):
        AactCandidateChronology(
            study_first_posted_date=date(2020, 1, 1),
            results_first_posted_date=date(2020, 4, 1),
            results_first_posted_date_type=AactPostedDateType.ACTUAL,
        )

    contradicted_chronology = admitted.chronology.model_copy(update={'later_actual_start_date': date(2020, 1, 15)})
    contradicted = admitted.model_dump(mode='python')
    contradicted['chronology'] = contradicted_chronology
    contradicted['gate_results'] = _gate_results(contradicted_chronology, fully_reviewed=True)
    with pytest.raises(ValueError, match='conservative actual-start adjudication'):
        AactCandidateInventoryRecord.model_validate(contradicted)

    results_after_label = admitted.chronology.model_copy(update={'results_first_posted_date': date(2020, 6, 1)})
    early_label = admitted.model_dump(mode='python')
    early_label['chronology'] = results_after_label
    early_label['gate_results'] = _gate_results(results_after_label, fully_reviewed=True)
    with pytest.raises(ValueError, match='label snapshot cannot predate'):
        AactCandidateInventoryRecord.model_validate(early_label)

    too_early_confirmation = admitted.model_dump(mode='python')
    assert admitted.confirmation_snapshot is not None
    too_early_confirmation['confirmation_snapshot'] = admitted.confirmation_snapshot.model_copy(
        update={
            'snapshot_id': 'aact-flatfiles-2020-06-01',
            'archive_date': date(2020, 6, 1),
        }
    )
    with pytest.raises(ValueError, match='at least 90 days'):
        AactCandidateInventoryRecord.model_validate(too_early_confirmation)


def test_offline_organizer_cli_builds_and_reverifies_catalog_and_plan(tmp_path) -> None:
    html_path = tmp_path / 'listing-2020.html'
    html_path.write_bytes(
        _listing_html(
            2020,
            (
                _row('05-01-2020', '20200501_pipe-delimited-export.zip', '1.01 GB'),
                _row('02-01-2020', '20200201_pipe-delimited-export.zip', '996.84 MB'),
            ),
        )
    )
    catalog_spec = tmp_path / 'catalog-spec.json'
    catalog_spec.write_text(
        json.dumps(
            {
                'catalog_id': 'cli-catalog',
                'generated_at': _RETRIEVED.isoformat(),
                'parser_implementation_sha256': _HASH_A,
                'listings': [
                    {
                        'year': 2020,
                        'source_url': (
                            'https://aact.ctti-clinicaltrials.org/downloads/snapshots?type=flatfiles&year=2020'
                        ),
                        'retrieved_at': _RETRIEVED.isoformat(),
                        'path': html_path.name,
                    }
                ],
            }
        ),
        encoding='utf-8',
    )
    catalog_path = tmp_path / 'catalog.json'
    assert inventory_cli_main(['build-catalog', '--spec', str(catalog_spec), '--output', str(catalog_path)]) == 0
    assert inventory_cli_main(['verify-catalog', '--spec', str(catalog_spec), '--catalog', str(catalog_path)]) == 0

    plan_spec = tmp_path / 'plan-spec.json'
    plan_spec.write_text(
        json.dumps(
            {
                'plan_id': 'cli-plan',
                'created_at': _RETRIEVED.isoformat(),
                'screening_policy_sha256': _HASH_B,
                'requested_archives': [
                    {
                        'snapshot_id': 'aact-flatfiles-2020-02-01',
                        'roles': ['decision_candidate', 'discovery'],
                    },
                    {
                        'snapshot_id': 'aact-flatfiles-2020-05-01',
                        'roles': ['label_candidate'],
                    },
                ],
            }
        ),
        encoding='utf-8',
    )
    plan_path = tmp_path / 'plan.json'
    assert (
        inventory_cli_main(
            ['build-plan', '--catalog', str(catalog_path), '--spec', str(plan_spec), '--output', str(plan_path)]
        )
        == 0
    )
    assert inventory_cli_main(['verify-plan', '--catalog', str(catalog_path), '--plan', str(plan_path)]) == 0
