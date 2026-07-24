from __future__ import annotations

import hashlib
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import pytest

from tests.test_clinicaltrials_execution_adapter import (
    _table_bytes,
    _write_decision_zip,
    _write_label_zip,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_adapter import build_aact_execution_cohort
from vaxreplay.clinicaltrials.execution_gold_adapter import (
    ExecutionGoldCohortDerivation,
    ExecutionGoldCohortTarget,
    ExecutionGoldCohortTargetSet,
    ExecutionGoldDerivation,
    ExecutionGoldDerivationError,
    TrustedExecutionGoldSourceHashes,
    TrustedSourceBuildHashes,
    derive_execution_private_gold,
    derive_execution_private_gold_cohort,
    execution_forecast_spec_policy,
    execution_forecast_spec_policy_sha256,
    load_execution_gold_cohort_derivation,
    write_execution_gold_cohort_derivation,
    write_execution_gold_derivation,
)
from vaxreplay.clinicaltrials.execution_merge import merge_aact_execution_builds
from vaxreplay.clinicaltrials.execution_schema import ObservationState, RegistryOutcomeClass
from vaxreplay.clinicaltrials.execution_task import (
    ContinuousForecastSpec,
    CutoffDocument,
    ExecutionTaskContext,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    RelevanceDisposition,
    RelevanceReason,
    RelevanceReviewInput,
    build_relevance_review_queue,
    finalize_relevance_adjudications,
    write_relevance_review_build,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _enrich_for_relevance_review(path: Path) -> None:
    """Add the two decision-only text members that the relevance reviewer also requires."""

    with zipfile.ZipFile(path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    nct_ids = ('NCT00000001', 'NCT00000002')
    members['brief_summaries.txt'] = _table_bytes(
        ('nct_id', 'description'),
        [{'nct_id': nct_id, 'description': f'Decision-time candidate summary for {nct_id}'} for nct_id in nct_ids],
    )
    members['detailed_descriptions.txt'] = _table_bytes(
        ('nct_id', 'description'),
        [
            {'nct_id': nct_id, 'description': 'Investigational active prophylactic vaccine candidate.'}
            for nct_id in nct_ids
        ],
    )
    replacement = path.with_suffix('.replacement.zip')
    with zipfile.ZipFile(replacement, 'x', compression=zipfile.ZIP_DEFLATED) as destination:
        for name, payload in sorted(members.items()):
            destination.writestr(name, payload)
    path.unlink()
    replacement.rename(path)


def _fixture(root: Path):
    decision_2018 = root / 'decision-2018.zip'
    label_2022 = root / 'label-2022.zip'
    decision_2020 = root / 'decision-2020.zip'
    label_2024 = root / 'label-2024.zip'
    _write_decision_zip(
        decision_2018,
        decision_anchor=date(2018, 4, 1),
        include_second_study=False,
        first_lead_sponsor='Moderna',
    )
    _write_label_zip(label_2022)
    _write_decision_zip(decision_2020, first_lead_sponsor='Moderna, Inc.')
    _write_label_zip(label_2024)
    _enrich_for_relevance_review(decision_2018)
    _enrich_for_relevance_review(decision_2020)

    source_2018 = root / 'source-2018'
    source_2020 = root / 'source-2020'
    build_aact_execution_cohort(
        decision_archive=decision_2018,
        decision_archive_date=date(2018, 4, 1),
        label_archive=label_2022,
        label_archive_date=date(2022, 4, 1),
        output_root=source_2018,
        synthetic_test_only=True,
    )
    build_aact_execution_cohort(
        decision_archive=decision_2020,
        decision_archive_date=date(2020, 2, 1),
        label_archive=label_2024,
        label_archive_date=date(2024, 2, 1),
        output_root=source_2020,
        synthetic_test_only=True,
    )
    merged = merge_aact_execution_builds(
        source_roots=(source_2020, source_2018),
        output_root=root / 'merged',
    )
    decision_archives = {
        date(2018, 4, 1): decision_2018,
        date(2020, 2, 1): decision_2020,
    }
    inventory_payload = (merged.root / 'organizer' / 'cohort-inventory.json').read_bytes()
    queue = build_relevance_review_queue(
        inventory=merged.inventory,
        merged_inventory_sha256=_sha256(inventory_payload),
        decision_archives=decision_archives,
    )
    reviews = tuple(
        RelevanceReviewInput(
            nct_id=record.nct_id,
            anchor_date=record.anchor_date,
            evidence_sha256=record.evidence_sha256,
            disposition=RelevanceDisposition.INCLUDE,
            reason_codes=(RelevanceReason.INCLUDE_ACTIVE_PROPHYLACTIC_VACCINE_CANDIDATE,),
            rationale='Decision-time text identifies an investigational active prophylactic vaccine candidate.',
        )
        for record in queue.records
    )
    adjudications = finalize_relevance_adjudications(queue=queue, reviews=reviews)
    review_receipt = write_relevance_review_build(
        queue=queue,
        reviews=reviews,
        output_root=root / 'relevance-review',
    )
    trusted = TrustedExecutionGoldSourceHashes(
        merge_receipt_artifact_sha256=_sha256((merged.root / 'MERGE-RECEIPT.json').read_bytes()),
        inventory_artifact_sha256=_sha256(inventory_payload),
        label_set_artifact_sha256=_sha256((merged.root / 'private' / 'execution-labels.json').read_bytes()),
        relevance_queue_sha256=_sha256(canonical_json_bytes(queue)),
        relevance_adjudication_sha256=_sha256(canonical_json_bytes(adjudications)),
        relevance_review_receipt_sha256=_sha256(canonical_json_bytes(review_receipt)),
        source_builds=tuple(
            TrustedSourceBuildHashes(
                anchor_date=source.anchor_date,
                build_receipt_sha256=source.build_receipt_sha256,
                decision_archive_sha256=source.decision_archive_sha256,
                label_archive_sha256=source.label_archive_sha256,
            )
            for source in merged.receipt.source_builds
        ),
    )
    assignment = next(item for item in merged.inventory.assignments if item.nct_id == 'NCT00000001')
    policy = execution_forecast_spec_policy()
    context = ExecutionTaskContext(
        episode_id='execution-private-dev-001',
        target_trial_id='trial-private-001',
        decision_snapshot_id=assignment.decision_snapshot_id,
        anchor_date=assignment.anchor_date,
        label_snapshot_id=assignment.label_snapshot_id,
        label_archive_date=assignment.label_archive_date,
        planned_enrollment=assignment.planned_enrollment,
        planned_primary_completion_date=assignment.planned_primary_completion_date,
        enrollment_ratio_spec=policy.enrollment_ratio_spec,
        primary_completion_slippage_days_spec=policy.primary_completion_slippage_days_spec,
    )
    return merged, decision_archives, queue, adjudications, review_receipt, trusted, context


def _derive(fixture):
    merged, decision_archives, queue, adjudications, review_receipt, trusted, context = fixture
    return derive_execution_private_gold(
        nct_id='NCT00000001',
        context=context,
        inventory=merged.inventory,
        label_set=merged.labels,
        merge_receipt=merged.receipt,
        trusted_source_hashes=trusted,
        decision_archives=decision_archives,
        relevance_queue=queue,
        relevance_adjudications=adjudications,
        relevance_review_receipt=review_receipt,
    )


def _cohort_targets(fixture, *, include_cutoff_document: bool = False) -> ExecutionGoldCohortTargetSet:
    merged, _, _, adjudications, _, _, _ = fixture
    include_nct_ids = {
        (item.anchor_date, item.nct_id)
        for item in adjudications.decisions
        if item.disposition == RelevanceDisposition.INCLUDE
    }
    policy = execution_forecast_spec_policy()
    targets = []
    for index, assignment in enumerate(merged.inventory.assignments, start=1):
        if (assignment.anchor_date, assignment.nct_id) not in include_nct_ids:
            continue
        documents = ()
        if include_cutoff_document:
            body = 'Outcome-blind decision-time vaccine candidate evidence.'
            documents = (
                CutoffDocument(
                    document_id='decision-evidence-001',
                    available_on=assignment.anchor_date,
                    body=body,
                    body_sha256=_sha256(body.encode('utf-8')),
                ),
            )
        targets.append(
            ExecutionGoldCohortTarget(
                organizer_private_nct_id=assignment.nct_id,
                context=ExecutionTaskContext(
                    episode_id=f'execution-cohort-dev-{index:03d}',
                    target_trial_id='trial-target',
                    decision_snapshot_id=assignment.decision_snapshot_id,
                    anchor_date=assignment.anchor_date,
                    label_snapshot_id=assignment.label_snapshot_id,
                    label_archive_date=assignment.label_archive_date,
                    planned_enrollment=assignment.planned_enrollment,
                    planned_primary_completion_date=assignment.planned_primary_completion_date,
                    enrollment_ratio_spec=policy.enrollment_ratio_spec,
                    primary_completion_slippage_days_spec=policy.primary_completion_slippage_days_spec,
                    cutoff_documents=documents,
                ),
            )
        )
    return ExecutionGoldCohortTargetSet(
        cohort_id='test-execution-cohort',
        targets=tuple(targets),
        final_workspace_contexts_bound=include_cutoff_document,
    )


def _derive_cohort(fixture, *, include_cutoff_document: bool = False):
    merged, decision_archives, queue, adjudications, review_receipt, trusted, _ = fixture
    return derive_execution_private_gold_cohort(
        targets=_cohort_targets(fixture, include_cutoff_document=include_cutoff_document),
        inventory=merged.inventory,
        label_set=merged.labels,
        merge_receipt=merged.receipt,
        trusted_source_hashes=trusted,
        decision_archives=decision_archives,
        relevance_queue=queue,
        relevance_adjudications=adjudications,
        relevance_review_receipt=review_receipt,
    )


def test_derives_gold_only_from_reaudited_sources_and_retains_raw_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        derivation = _derive(_fixture(Path(temporary)))

    gold = derivation.private_gold
    source = derivation.receipt.source_observation
    assert gold.registry_outcome_class == RegistryOutcomeClass.TERMINATED
    assert gold.enrollment_observation == ObservationState.OBSERVED_ACTUAL
    assert gold.enrollment_ratio == 0.8
    assert gold.primary_completion_observation == ObservationState.OBSERVED_ACTUAL
    assert source.raw_overall_status is not None
    assert source.raw_enrollment == 80
    assert source.planned_enrollment == 100
    assert source.enrollment_ratio == round(source.raw_enrollment / source.planned_enrollment, 12)
    assert (
        source.primary_completion_slippage_days
        == (source.raw_primary_completion_date - source.planned_primary_completion_date).days
    )
    assert source.decision_archive_sha256 in {
        item.decision_archive_sha256 for item in derivation.receipt.trusted_source_hashes.source_builds
    }
    assert source.outcome_source_record_sha256 is not None
    assert derivation.receipt.forecast_spec_policy_sha256 == execution_forecast_spec_policy_sha256()
    assert derivation.receipt.relevance.decision_only_queue_rebuilt_from_exact_archives
    assert not derivation.receipt.leaderboard_admitted
    assert not derivation.receipt.tier_b_admitted
    assert not derivation.receipt.tier_a_official
    assert not derivation.receipt.public_workspace_created
    assert not derivation.receipt.identity_masking_claimed
    assert not derivation.receipt.sealed_execution_claimed
    assert not derivation.receipt.later_data_used_for_selection


def test_writer_emits_only_organizer_private_material_and_refuses_tampered_gold() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        derivation = _derive(_fixture(root))
        build = write_execution_gold_derivation(derivation=derivation, output_root=root / 'private-build')
        assert tuple(item.relative_path for item in build.artifacts) == (
            'organizer/forecast-spec-policy.json',
            'organizer/source-derivation-receipt.json',
            'private/execution-private-gold.json',
        )
        assert not (build.root / 'public').exists()
        tampered_gold = derivation.private_gold.model_copy(update={'enrollment_ratio': 1.5})
        with pytest.raises(ExecutionGoldDerivationError, match='does not match'):
            write_execution_gold_derivation(
                derivation=ExecutionGoldDerivation(receipt=derivation.receipt, private_gold=tampered_gold),
                output_root=root / 'tampered-build',
            )


def test_stale_hash_model_copy_and_source_archive_mismatch_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = _fixture(root)
        merged, decision_archives, queue, adjudications, review_receipt, trusted, context = fixture

        malicious_source = merged.receipt.source_builds[0].model_copy(update={'decision_archive_sha256': 'f' * 64})
        copied_receipt = merged.receipt.model_copy(
            update={'source_builds': (malicious_source, *merged.receipt.source_builds[1:])}
        )
        with pytest.raises(ExecutionGoldDerivationError, match='external trusted hash'):
            derive_execution_private_gold(
                nct_id='NCT00000001',
                context=context,
                inventory=merged.inventory,
                label_set=merged.labels,
                merge_receipt=copied_receipt,
                trusted_source_hashes=trusted,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=adjudications,
                relevance_review_receipt=review_receipt,
            )

        stale_inventory_hashes = trusted.model_copy(update={'inventory_artifact_sha256': 'a' * 64})
        with pytest.raises(ExecutionGoldDerivationError, match='merged inventory'):
            derive_execution_private_gold(
                nct_id='NCT00000001',
                context=context,
                inventory=merged.inventory,
                label_set=merged.labels,
                merge_receipt=merged.receipt,
                trusted_source_hashes=stale_inventory_hashes,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=adjudications,
                relevance_review_receipt=review_receipt,
            )

        forged_review_receipt = review_receipt.model_copy(update={'include_count': review_receipt.include_count + 1})
        with pytest.raises(ExecutionGoldDerivationError, match='review receipt does not match'):
            derive_execution_private_gold(
                nct_id='NCT00000001',
                context=context,
                inventory=merged.inventory,
                label_set=merged.labels,
                merge_receipt=merged.receipt,
                trusted_source_hashes=trusted,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=adjudications,
                relevance_review_receipt=forged_review_receipt,
            )

        decision_archives[date(2018, 4, 1)].write_bytes(decision_archives[date(2018, 4, 1)].read_bytes() + b'tamper')
        with pytest.raises(ExecutionGoldDerivationError, match='exact archive audit'):
            _derive((merged, decision_archives, queue, adjudications, review_receipt, trusted, context))


def test_caller_asserted_label_forecast_bounds_and_noninclude_relevance_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        merged, decision_archives, queue, adjudications, review_receipt, trusted, context = fixture

        changed_label = merged.labels.labels[0].model_copy(update={'enrollment_ratio': 1.7})
        copied_labels = merged.labels.model_copy(update={'labels': (changed_label, *merged.labels.labels[1:])})
        copied_label_hashes = trusted.model_copy(
            update={'label_set_artifact_sha256': _sha256(canonical_json_bytes(copied_labels) + b'\n')}
        )
        with pytest.raises(ExecutionGoldDerivationError, match='execution label set failed schema audit'):
            derive_execution_private_gold(
                nct_id='NCT00000001',
                context=context,
                inventory=merged.inventory,
                label_set=copied_labels,
                merge_receipt=merged.receipt,
                trusted_source_hashes=copied_label_hashes,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=adjudications,
                relevance_review_receipt=review_receipt,
            )

        outcome_chosen_context = context.model_copy(
            update={
                'enrollment_ratio_spec': ContinuousForecastSpec(
                    forecast_kind='point',
                    lower_bound=0.0,
                    upper_bound=0.81,
                )
            }
        )
        with pytest.raises(ExecutionGoldDerivationError, match='fixed forecast-spec'):
            derive_execution_private_gold(
                nct_id='NCT00000001',
                context=outcome_chosen_context,
                inventory=merged.inventory,
                label_set=merged.labels,
                merge_receipt=merged.receipt,
                trusted_source_hashes=trusted,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=adjudications,
                relevance_review_receipt=review_receipt,
            )

        selected = adjudications.decisions[0]
        exclusion = RelevanceReviewInput(
            nct_id=selected.nct_id,
            anchor_date=selected.anchor_date,
            evidence_sha256=selected.evidence_sha256,
            disposition=RelevanceDisposition.EXCLUDE,
            reason_codes=(RelevanceReason.EXCLUDE_NOT_ACTIVE_PROPHYLACTIC_VACCINE,),
            rationale='Organizer adjudicated this record out of the active-vaccine cohort.',
        )
        excluded_reviews = (exclusion, *adjudications.decisions[1:])
        excluded_adjudications = finalize_relevance_adjudications(queue=queue, reviews=excluded_reviews)
        excluded_review_receipt = write_relevance_review_build(
            queue=queue,
            reviews=excluded_reviews,
            output_root=Path(temporary) / 'relevance-review-excluded',
        )
        excluded_hashes = trusted.model_copy(
            update={
                'relevance_adjudication_sha256': _sha256(canonical_json_bytes(excluded_adjudications)),
                'relevance_review_receipt_sha256': _sha256(canonical_json_bytes(excluded_review_receipt)),
            }
        )
        with pytest.raises(ExecutionGoldDerivationError, match='only for relevance INCLUDE'):
            derive_execution_private_gold(
                nct_id=selected.nct_id,
                context=context,
                inventory=merged.inventory,
                label_set=merged.labels,
                merge_receipt=merged.receipt,
                trusted_source_hashes=excluded_hashes,
                decision_archives=decision_archives,
                relevance_queue=queue,
                relevance_adjudications=excluded_adjudications,
                relevance_review_receipt=excluded_review_receipt,
            )


def test_cohort_derives_every_include_once_and_binds_caller_workspace_contexts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        derivation = _derive_cohort(fixture, include_cutoff_document=True)

    assert derivation.receipt.eligible_include_count == 2
    assert derivation.receipt.derived_case_count == 2
    assert len(derivation.receipt.case_receipts) == 2
    assert len(derivation.private_gold.records) == 2
    assert derivation.receipt.source_models_verified_once
    assert derivation.receipt.exact_include_coverage_verified
    assert derivation.receipt.every_raw_outcome_recomputed
    assert all(target.context.cutoff_documents for target in derivation.targets.targets)
    assert derivation.receipt.final_workspace_contexts_bound
    assert not derivation.receipt.split_inventory_bound
    assert not derivation.receipt.lineage_split_safe
    assert not derivation.receipt.leaderboard_admitted
    for case in derivation.receipt.case_receipts:
        gold = next(
            item
            for item in derivation.private_gold.records
            if item.organizer_private_nct_id == case.organizer_private_nct_id
        )
        assert case.private_gold_sha256 == _sha256(canonical_json_bytes(gold))
        assert case.source_observation.registry_outcome_class == gold.registry_outcome_class


def test_cohort_rejects_missing_extra_and_duplicate_target_cases() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        merged, decision_archives, queue, adjudications, review_receipt, trusted, _ = fixture
        complete = _cohort_targets(fixture)
        missing = complete.model_copy(update={'targets': complete.targets[:-1]})
        common = {
            'inventory': merged.inventory,
            'label_set': merged.labels,
            'merge_receipt': merged.receipt,
            'trusted_source_hashes': trusted,
            'decision_archives': decision_archives,
            'relevance_queue': queue,
            'relevance_adjudications': adjudications,
            'relevance_review_receipt': review_receipt,
        }
        with pytest.raises(ExecutionGoldDerivationError, match='exactly cover'):
            derive_execution_private_gold_cohort(targets=missing, **common)

        first = complete.targets[0]
        extra_context = first.context.model_copy(
            update={
                'episode_id': 'execution-cohort-dev-extra',
                'target_trial_id': 'trial-cohort-dev-extra',
            }
        )
        extra = ExecutionGoldCohortTargetSet(
            cohort_id=complete.cohort_id,
            targets=tuple(
                sorted(
                    (
                        *complete.targets,
                        ExecutionGoldCohortTarget(
                            organizer_private_nct_id='NCT99999999',
                            context=extra_context,
                        ),
                    ),
                    key=lambda item: (item.context.anchor_date, item.organizer_private_nct_id),
                )
            ),
        )
        with pytest.raises(ExecutionGoldDerivationError, match='exactly cover'):
            derive_execution_private_gold_cohort(targets=extra, **common)

        with pytest.raises(ValueError, match='unique ascending'):
            ExecutionGoldCohortTargetSet(
                cohort_id=complete.cohort_id,
                targets=(complete.targets[0], complete.targets[0]),
            )


def test_cohort_writer_is_private_and_rejects_cross_artifact_tampering() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        derivation = _derive_cohort(_fixture(root))
        build = write_execution_gold_cohort_derivation(
            derivation=derivation,
            output_root=root / 'cohort-private-build',
        )
        assert tuple(item.relative_path for item in build.artifacts) == (
            'organizer/cohort-source-derivation-receipt.json',
            'organizer/cohort-targets.json',
            'organizer/forecast-spec-policy.json',
            'private/execution-private-gold-set.json',
        )
        assert not (build.root / 'public').exists()
        receipt_path = build.root / 'organizer' / 'cohort-source-derivation-receipt.json'
        receipt_sha256 = _sha256(receipt_path.read_bytes())
        loaded = load_execution_gold_cohort_derivation(
            build.root,
            expected_receipt_sha256=receipt_sha256,
        )
        assert loaded.derivation == build.derivation
        with pytest.raises(ExecutionGoldDerivationError, match='external hash'):
            load_execution_gold_cohort_derivation(
                build.root,
                expected_receipt_sha256='0' * 64,
            )
        changed = derivation.private_gold.records[0].model_copy(update={'enrollment_ratio': 1.5})
        tampered_set = derivation.private_gold.model_copy(
            update={'records': (changed, *derivation.private_gold.records[1:])}
        )
        with pytest.raises(ExecutionGoldDerivationError, match='does not match'):
            write_execution_gold_cohort_derivation(
                derivation=ExecutionGoldCohortDerivation(
                    receipt=derivation.receipt,
                    targets=derivation.targets,
                    private_gold=tampered_set,
                ),
                output_root=root / 'tampered-cohort',
            )

        # Updating every caller-controlled gold hash still cannot detach the reward from the raw
        # source observation retained in the compact case receipt.
        changed_records = (changed, *derivation.private_gold.records[1:])
        self_consistent_set = derivation.private_gold.model_copy(update={'records': changed_records})
        changed_case = next(
            item
            for item in derivation.receipt.case_receipts
            if item.organizer_private_nct_id == changed.organizer_private_nct_id
        ).model_copy(update={'private_gold_sha256': _sha256(canonical_json_bytes(changed))})
        changed_cases = tuple(
            changed_case if item.organizer_private_nct_id == changed.organizer_private_nct_id else item
            for item in derivation.receipt.case_receipts
        )
        self_consistent_receipt = derivation.receipt.model_copy(
            update={
                'case_receipts': changed_cases,
                'private_gold_set_sha256': _sha256(canonical_json_bytes(self_consistent_set)),
            }
        )
        with pytest.raises(ExecutionGoldDerivationError, match='recomputed source'):
            write_execution_gold_cohort_derivation(
                derivation=ExecutionGoldCohortDerivation(
                    receipt=self_consistent_receipt,
                    targets=derivation.targets,
                    private_gold=self_consistent_set,
                ),
                output_root=root / 'self-consistent-tampered-cohort',
            )


def test_cohort_loader_rejects_extra_and_noncanonical_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        derivation = _derive_cohort(_fixture(root))
        build = write_execution_gold_cohort_derivation(
            derivation=derivation,
            output_root=root / 'cohort-private-build',
        )
        receipt_path = build.root / 'organizer' / 'cohort-source-derivation-receipt.json'
        receipt_sha256 = _sha256(receipt_path.read_bytes())
        extra = build.root / 'organizer' / 'unbound.json'
        extra.write_bytes(b'{}')
        extra.chmod(0o600)
        with pytest.raises(ExecutionGoldDerivationError, match='exactly'):
            load_execution_gold_cohort_derivation(build.root, expected_receipt_sha256=receipt_sha256)
        extra.unlink()

        targets_path = build.root / 'organizer' / 'cohort-targets.json'
        targets_path.chmod(0o644)
        with pytest.raises(ExecutionGoldDerivationError, match='mode 0600'):
            load_execution_gold_cohort_derivation(build.root, expected_receipt_sha256=receipt_sha256)
        targets_path.chmod(0o600)
        targets_path.write_bytes(targets_path.read_bytes() + b'\n')
        with pytest.raises(ExecutionGoldDerivationError, match='not canonical'):
            load_execution_gold_cohort_derivation(build.root, expected_receipt_sha256=receipt_sha256)
