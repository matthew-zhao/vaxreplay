"""Construction and independent cross-verification of AACT candidate inventories."""

from __future__ import annotations

from datetime import datetime

from vaxreplay.clinicaltrials.inventory_catalog import (
    AactCatalogIntegrityError,
    verify_archive_acquisition_plan,
)
from vaxreplay.clinicaltrials.inventory_schema import (
    AactArchiveAcquisitionPlan,
    AactArchiveAcquisitionRole,
    AactCandidateInventory,
    AactCandidateInventoryRecord,
    AactInventoryGate,
    AactOfficialArchiveCatalog,
    aact_inventory_model_sha256,
    aact_inventory_records_sha256,
    aact_mechanical_gate_expectations,
)


class AactInventoryIntegrityError(ValueError):
    """A candidate inventory or one of its catalog/plan bindings failed closed."""


def build_candidate_inventory(
    *,
    inventory_id: str,
    created_at: datetime,
    catalog: AactOfficialArchiveCatalog,
    acquisition_plan: AactArchiveAcquisitionPlan,
    screening_policy_sha256: str,
    gate_policy_sha256: str,
    lineage_policy_sha256: str,
    masking_policy_sha256: str,
    records: tuple[AactCandidateInventoryRecord, ...],
) -> AactCandidateInventory:
    """Create a deterministic NCT-sorted candidate inventory and verify all source bindings."""

    ordered_records = tuple(sorted(records, key=lambda record: record.nct_id))
    try:
        inventory = AactCandidateInventory(
            inventory_id=inventory_id,
            created_at=created_at,
            catalog_sha256=aact_inventory_model_sha256(catalog),
            acquisition_plan_sha256=aact_inventory_model_sha256(acquisition_plan),
            screening_policy_sha256=screening_policy_sha256,
            gate_policy_sha256=gate_policy_sha256,
            lineage_policy_sha256=lineage_policy_sha256,
            masking_policy_sha256=masking_policy_sha256,
            records=ordered_records,
            record_count=len(ordered_records),
            records_sha256=aact_inventory_records_sha256(ordered_records),
        )
    except ValueError as error:
        raise AactInventoryIntegrityError(f'invalid AACT candidate inventory: {error}') from error
    verify_candidate_inventory(inventory, catalog, acquisition_plan)
    return inventory


def verify_candidate_inventory(
    inventory: AactCandidateInventory,
    catalog: AactOfficialArchiveCatalog,
    acquisition_plan: AactArchiveAcquisitionPlan,
) -> None:
    """Independently cross-bind every candidate snapshot to the catalog and acquisition plan."""

    # The plan is an input authority, not a trusted helper object.  Validate all of it before using
    # its hash, roles, or paths to validate the inventory.
    try:
        verify_archive_acquisition_plan(acquisition_plan, catalog)
    except AactCatalogIntegrityError as error:
        raise AactInventoryIntegrityError(f'invalid archive acquisition plan: {error}') from error

    if inventory.catalog_sha256 != aact_inventory_model_sha256(catalog):
        raise AactInventoryIntegrityError('candidate inventory binds a different official archive catalog')
    if inventory.acquisition_plan_sha256 != aact_inventory_model_sha256(acquisition_plan):
        raise AactInventoryIntegrityError('candidate inventory binds a different acquisition plan')
    if inventory.created_at < acquisition_plan.created_at:
        raise AactInventoryIntegrityError('candidate inventory cannot predate its acquisition plan')
    if inventory.screening_policy_sha256 != acquisition_plan.screening_policy_sha256:
        raise AactInventoryIntegrityError('inventory and acquisition plan use different screening policies')

    entry_by_snapshot = {entry.snapshot_id: entry for entry in catalog.entries}
    plan_item_by_snapshot = {item.snapshot_id: item for item in acquisition_plan.items}
    for record in inventory.records:
        gate_values = tuple(result.gate.value for result in record.gate_results)
        expected_gate_values = tuple(sorted(gate.value for gate in AactInventoryGate))
        if gate_values != expected_gate_values:
            raise AactInventoryIntegrityError(
                f'candidate {record.nct_id} does not contain every gate exactly once in sorted order'
            )
        result_by_gate = {result.gate: result for result in record.gate_results}
        for gate, (expected_status, expected_reason_codes) in aact_mechanical_gate_expectations(
            study_type=record.study_type,
            allocation=record.allocation,
            intervention_model=record.intervention_model,
            primary_purpose=record.primary_purpose,
            phase=record.phase,
            counts=record.counts,
            chronology=record.chronology,
        ).items():
            result = result_by_gate[gate]
            if result.status != expected_status or result.reason_codes != expected_reason_codes:
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} has a forged or inconsistent {gate.value} gate result'
                )
        expected_roles = (
            (record.discovery_snapshot, AactArchiveAcquisitionRole.DISCOVERY),
            (record.decision_snapshot, AactArchiveAcquisitionRole.DECISION_CANDIDATE),
            (record.label_snapshot, AactArchiveAcquisitionRole.LABEL_CANDIDATE),
            (record.confirmation_snapshot, AactArchiveAcquisitionRole.CONFIRMATION_CANDIDATE),
        )
        for binding, expected_role in expected_roles:
            if binding is None:
                continue
            entry = entry_by_snapshot.get(binding.snapshot_id)
            if entry is None:
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} references unknown snapshot {binding.snapshot_id}'
                )
            plan_item = plan_item_by_snapshot.get(binding.snapshot_id)
            if plan_item is None:
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} references an archive outside the acquisition plan'
                )
            if expected_role not in plan_item.roles:
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} snapshot lacks its required {expected_role.value} role'
                )
            if binding.archive_date != entry.archive_date:
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} snapshot date differs from the official catalog'
                )
            if binding.catalog_entry_sha256 != aact_inventory_model_sha256(entry):
                raise AactInventoryIntegrityError(
                    f'candidate {record.nct_id} snapshot commitment differs from the official catalog'
                )

    # Re-run strict model validators last so model-copy bypasses of aggregate hashes and structural
    # invariants also fail, after more specific source/gate errors have had a chance to surface.
    try:
        AactCandidateInventory.model_validate(inventory.model_dump(mode='python'))
    except ValueError as error:
        raise AactInventoryIntegrityError(f'invalid AACT candidate inventory: {error}') from error


def inventory_catalog_and_plan_sha256(
    catalog: AactOfficialArchiveCatalog,
    acquisition_plan: AactArchiveAcquisitionPlan,
) -> tuple[str, str]:
    """Return the two public commitments used when handing an inventory to later stages."""

    try:
        verify_archive_acquisition_plan(acquisition_plan, catalog)
    except AactCatalogIntegrityError as error:
        raise AactInventoryIntegrityError(str(error)) from error
    return aact_inventory_model_sha256(catalog), aact_inventory_model_sha256(acquisition_plan)
