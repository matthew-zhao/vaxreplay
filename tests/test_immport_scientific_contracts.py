from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, ForecastTarget
from vaxreplay.operations.promotion import (
    AdapterSpec,
    LoadedCapturePromotion,
)
from vaxreplay.prospective import LoadedProspectiveDecisionPackage
from vaxreplay.prospective_admission import (
    ImmportScientificContractBinding,
    PromotionArchivePolicyEntry,
    PromotionArchiveVerificationSpec,
    ProspectiveAdmissionError,
    _verify_immport_scientific_decision,
)
from vaxreplay.sources.immport import (
    ImmportArmAdapterPolicy,
    ImmportArmCandidateMap,
    ImmportArmCandidateMapEntry,
    ImmportArmCandidateSetDefinition,
    ImmportPromotionLayout,
    ImmportScientificContractError,
    ImmportSourceVerifierPolicy,
    ImmportStudyUniverseEntry,
    ImmportStudyUniverseRegistry,
    immport_study_universe_bytes,
    verify_immport_study_universe_precommit,
)
from vaxreplay.sources.immport_outcomes import (
    ImmportFutureOutcomeCapture,
    ImmportFutureOutcomeDisposition,
    ImmportOutcomeContractError,
    ImmportOutcomeTargetDefinition,
    ImmportProspectiveOutcomeAdjudicationSpec,
    immport_outcome_adjudication_spec_bytes,
    verify_and_join_immport_future_outcomes,
)

_T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
_FIRST_SLOT = _T0 + timedelta(days=1)
_STUDY = 'SDY00000000'
_SOURCE = 'immport:science-contract-test'


def _registry(*, studies: tuple[ImmportStudyUniverseEntry, ...] | None = None):
    return ImmportStudyUniverseRegistry(
        registry_id='immport-sdy-universe-v1',
        campaign_id='immport-campaign-v1',
        frozen_at=_T0,
        first_capture_not_before=_FIRST_SLOT,
        studies=studies
        or (
            ImmportStudyUniverseEntry(
                study_accession=_STUDY,
                disposition='selected',
                reason_code='predeclared_eligible_study',
            ),
            ImmportStudyUniverseEntry(
                study_accession='SDY00000001',
                disposition='excluded',
                reason_code='administrative_scope_exclusion',
            ),
        ),
    )


def _policies(registry_sha256: str):
    layout = ImmportPromotionLayout(
        study_accession=_STUDY,
        openapi_before_artifact_id='a01-openapi-before',
        study_before_artifact_id='a02-study-before',
        manifest_before_artifact_id='a03-manifest-before',
        arm_artifact_id='a04-arms',
        experiment_artifact_id='a05-experiments',
        link_artifact_id='a06-links',
        manifest_after_artifact_id='a07-manifest-after',
        study_after_artifact_id='a08-study-after',
        openapi_after_artifact_id='a09-openapi-after',
    )
    source = ImmportSourceVerifierPolicy(
        policy_id='immport-source-v1',
        source_id=_SOURCE,
        study_universe_registry_sha256=registry_sha256,
        layout=layout,
        expected_openapi_sha256='1' * 64,
        expected_openapi_info_version='v1',
        expected_latest_release_version='DR65',
        expected_latest_release_date=date(2026, 6, 25),
        expected_collector_id='immport-secret-broker-collector',
        expected_collector_implementation_sha256='2' * 64,
        expected_collector_execution_environment_sha256='3' * 64,
    )
    adapter = ImmportArmAdapterPolicy(
        policy_id='immport-arm-policy-v1',
        source_id=_SOURCE,
        episode_id='immport-arm-episode-v1',
        study_accession=_STUDY,
        study_universe_registry_sha256=registry_sha256,
        outcome_adjudication_spec_sha256='4' * 64,
        decision_at=_FIRST_SLOT + timedelta(days=1),
    )
    return source, adapter


def test_study_universe_exactly_cross_binds_selected_sdy_and_worker_policies() -> None:
    registry = _registry()
    registry_bytes = immport_study_universe_bytes(registry)
    source, adapter = _policies(hashlib.sha256(registry_bytes).hexdigest())

    assert (
        verify_immport_study_universe_precommit(
            registry_bytes,
            source_policies=(source,),
            adapter_policies=(adapter,),
            campaign_id=registry.campaign_id,
            first_scheduled_for=_FIRST_SLOT,
        )
        == registry
    )


def test_study_universe_rejects_post_selection_substitution_and_cherry_picking() -> None:
    registry_bytes = immport_study_universe_bytes(_registry())
    source, adapter = _policies(hashlib.sha256(registry_bytes).hexdigest())
    substituted = immport_study_universe_bytes(
        _registry(
            studies=(
                ImmportStudyUniverseEntry(
                    study_accession='SDY00000001',
                    disposition='selected',
                    reason_code='predeclared_eligible_study',
                ),
            )
        )
    )
    with pytest.raises(ImmportScientificContractError, match='paired source and adapter'):
        verify_immport_study_universe_precommit(
            registry_bytes,
            source_policies=(),
            adapter_policies=(),
            campaign_id='immport-campaign-v1',
            first_scheduled_for=_FIRST_SLOT,
        )
    with pytest.raises(ImmportScientificContractError, match='exactly cover'):
        verify_immport_study_universe_precommit(
            substituted,
            source_policies=(source,),
            adapter_policies=(adapter,),
            campaign_id='immport-campaign-v1',
            first_scheduled_for=_FIRST_SLOT,
        )
    with pytest.raises(ImmportScientificContractError, match='first-slot boundary'):
        verify_immport_study_universe_precommit(
            registry_bytes,
            source_policies=(source,),
            adapter_policies=(adapter,),
            campaign_id='immport-campaign-v1',
            first_scheduled_for=_FIRST_SLOT + timedelta(seconds=1),
        )


def _outcome_materials():
    registry_sha256 = '5' * 64
    spec = ImmportProspectiveOutcomeAdjudicationSpec(
        spec_id='immport-outcomes-v1',
        episode_id='immport-arm-episode-v1',
        decision_at=_T0,
        adapter_policy_id='immport-arm-policy-v1',
        study_universe_registry_sha256=registry_sha256,
        targets=(
            ImmportOutcomeTargetDefinition(
                target_id='binary-primary-endpoint',
                horizon_days=30,
                binary_success_definition_sha256='6' * 64,
                utility_if_failure=0.0,
                utility_if_success=1.0,
            ),
        ),
        allowed_censor_reasons=(
            'adjudicators_cannot_reach_consensus',
            'endpoint_not_reported_by_deadline',
        ),
    )
    spec_bytes = immport_outcome_adjudication_spec_bytes(spec)
    candidate_map = ImmportArmCandidateMap(
        policy_id=spec.adapter_policy_id,
        episode_id=spec.episode_id,
        study_universe_registry_sha256=registry_sha256,
        outcome_adjudication_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        candidates=tuple(
            sorted(
                (
                    ImmportArmCandidateMapEntry(
                        candidate_id=f'cand-immport-{ordinal:032x}',
                        study_accession=_STUDY,
                        arm_accession=f'ARM{ordinal}',
                        latest_release_version='DR65',
                        arm_role=role,
                        decision_disposition=disposition,
                    )
                    for ordinal, role, disposition in (
                        (1, 'intervention', 'rankable_intervention_arm'),
                        (2, 'intervention', 'rankable_intervention_arm'),
                        (3, 'control', 'contextual_control_not_ranked'),
                    )
                ),
                key=lambda item: item.candidate_id,
            )
        ),
    )
    map_bytes = canonical_json_bytes(candidate_map)
    revealed_at = _T0 + timedelta(days=31)
    capture = ImmportFutureOutcomeCapture(
        capture_id='immport-future-outcomes-v1',
        episode_id=spec.episode_id,
        outcome_adjudication_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        organizer_candidate_map_sha256=hashlib.sha256(map_bytes).hexdigest(),
        captured_at=revealed_at,
        witnessed_at=revealed_at + timedelta(seconds=1),
        dispositions=tuple(
            ImmportFutureOutcomeDisposition(
                candidate_id=f'cand-immport-{ordinal:032x}',
                target_id='binary-primary-endpoint',
                horizon_days=30,
                status='observed',
                binary_outcome=outcome,
                revealed_at=revealed_at,
                source_record_sha256=f'{ordinal}' * 64,
                adjudication_evidence_sha256=f'{ordinal + 3}' * 64,
                adjudicator_ids=('reviewer-a', 'reviewer-b'),
            )
            for ordinal, outcome in ((1, 1), (2, 0))
        ),
    )
    return spec_bytes, map_bytes, capture


def test_future_outcome_join_is_exhaustive_mature_and_control_free() -> None:
    spec_bytes, map_bytes, capture = _outcome_materials()
    labels = verify_and_join_immport_future_outcomes(
        spec_bytes=spec_bytes,
        candidate_map_bytes=map_bytes,
        capture_bytes=canonical_json_bytes(capture),
    )
    assert tuple(item.outcome for item in labels) == (1, 0)
    assert tuple(item.candidate_utility for item in labels) == (1.0, 0.0)


def test_outcome_spec_rejects_multiple_targets_without_a_utility_aggregation_rule() -> None:
    spec_bytes, _map_bytes, _capture = _outcome_materials()
    spec = ImmportProspectiveOutcomeAdjudicationSpec.model_validate_json(spec_bytes)
    second = spec.targets[0].model_copy(update={'target_id': 'second-endpoint'})
    with pytest.raises(ValueError, match='at most 1'):
        ImmportProspectiveOutcomeAdjudicationSpec.model_validate(
            spec.model_dump() | {'targets': (*spec.targets, second)}
        )


def test_future_outcome_join_rejects_missing_extra_control_and_early_rows() -> None:
    spec_bytes, map_bytes, capture = _outcome_materials()
    with pytest.raises(ImmportOutcomeContractError, match='exactly cover'):
        verify_and_join_immport_future_outcomes(
            spec_bytes=spec_bytes,
            candidate_map_bytes=map_bytes,
            capture_bytes=canonical_json_bytes(capture.model_copy(update={'dispositions': capture.dispositions[:1]})),
        )
    control = capture.dispositions[0].model_copy(
        update={'candidate_id': 'cand-immport-00000000000000000000000000000003'}
    )
    with pytest.raises(ImmportOutcomeContractError, match='exactly cover'):
        verify_and_join_immport_future_outcomes(
            spec_bytes=spec_bytes,
            candidate_map_bytes=map_bytes,
            capture_bytes=canonical_json_bytes(
                capture.model_copy(update={'dispositions': (*capture.dispositions, control)})
            ),
        )
    early = _T0 + timedelta(days=29)
    early_rows = tuple(item.model_copy(update={'revealed_at': early}) for item in capture.dispositions)
    early_capture = capture.model_copy(
        update={
            'captured_at': early,
            'witnessed_at': early + timedelta(seconds=1),
            'dispositions': early_rows,
        }
    )
    with pytest.raises(ImmportOutcomeContractError, match='before its target matured'):
        verify_and_join_immport_future_outcomes(
            spec_bytes=spec_bytes,
            candidate_map_bytes=map_bytes,
            capture_bytes=canonical_json_bytes(early_capture),
        )


def test_future_outcome_join_rejects_uncommitted_censor_and_noncanonical_bytes() -> None:
    spec_bytes, map_bytes, capture = _outcome_materials()
    censored = capture.dispositions[0].model_copy(
        update={
            'status': 'censored',
            'binary_outcome': None,
            'censor_reason': 'organizer_discretion',
        }
    )
    censored_capture = capture.model_copy(update={'dispositions': (censored, capture.dispositions[1])})
    with pytest.raises(ImmportOutcomeContractError, match='uncommitted censor'):
        verify_and_join_immport_future_outcomes(
            spec_bytes=spec_bytes,
            candidate_map_bytes=map_bytes,
            capture_bytes=canonical_json_bytes(censored_capture),
        )
    with pytest.raises(ImmportOutcomeContractError, match='canonical JSON'):
        verify_and_join_immport_future_outcomes(
            spec_bytes=b' ' + spec_bytes,
            candidate_map_bytes=map_bytes,
            capture_bytes=canonical_json_bytes(capture),
        )


def _decision_admission_fixture():
    spec_bytes, map_bytes, _capture = _outcome_materials()
    candidate_map = ImmportArmCandidateMap.model_validate_json(map_bytes)
    intervention_ids = tuple(item.candidate_id for item in candidate_map.candidates if item.arm_role == 'intervention')
    control_ids = tuple(item.candidate_id for item in candidate_map.candidates if item.arm_role == 'control')
    definition = ImmportArmCandidateSetDefinition(
        policy_id=candidate_map.policy_id,
        episode_id=candidate_map.episode_id,
        study_universe_registry_sha256=candidate_map.study_universe_registry_sha256,
        organizer_candidate_map_sha256=hashlib.sha256(map_bytes).hexdigest(),
        outcome_adjudication_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        intervention_candidate_ids=intervention_ids,
        contextual_control_ids=control_ids,
    )
    definition_bytes = canonical_json_bytes(definition)
    candidates = tuple(
        CandidateRecord(
            episode_id=candidate_map.episode_id,
            candidate_id=item.candidate_id,
            eligible=item.arm_role == 'intervention',
        )
        for item in candidate_map.candidates
    )
    loaded = LoadedCapturePromotion(
        root=Path('/tmp/immport-science-test'),
        manifest=cast(object, None),
        manifest_sha256='0' * 64,
        index=cast(object, None),
        index_bytes=b'{}',
        handoff_descriptor=cast(object, None),
        handoff_descriptor_bytes=b'{}',
        ledger_events=(),
        candidates=candidates,
        evidence=(),
        dispositions=(),
        auxiliary_outputs=MappingProxyType(
            {
                'immport-arm-candidate-map': map_bytes,
                'immport-candidate-set-definition': definition_bytes,
            }
        ),
        source_captures=(),
    )
    binding = ImmportScientificContractBinding(
        study_universe_registry_sha256=candidate_map.study_universe_registry_sha256,
        candidate_set_definition_sha256=hashlib.sha256(definition_bytes).hexdigest(),
        outcome_adjudication_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
    )
    entry = PromotionArchivePolicyEntry(
        promotion_id='immport-promotion-v1',
        source_id='promotion:immport-promotion-v1',
        promotion_manifest_sha256='1' * 64,
        capture_index_sha256='2' * 64,
        handoff_descriptor_sha256='3' * 64,
        scope_policy_sha256='4' * 64,
        scope_precommit_sha256='5' * 64,
        campaign_id='immport-campaign-v1',
        selection_key='immport-selection-v1',
        selection_policy_sha256='6' * 64,
        selection_policy_artifact_sha256='7' * 64,
        selection_manifest_sha256='8' * 64,
        immport_scientific_contract=binding,
    )
    verifier_spec = cast(
        PromotionArchiveVerificationSpec,
        SimpleNamespace(
            adapter=AdapterSpec(
                adapter_id='immport-complete-study-arm-catalog-adapter',
                adapter_version='v0.2',
                implementation_bytes=b'i',
                policy_bytes=b'p',
                execution_environment_bytes=b'e',
            ),
            immport_outcome_adjudication_spec_bytes=spec_bytes,
        ),
    )
    config = SimpleNamespace(
        task_type='early_clinical_arm_prioritization',
        episode_id=candidate_map.episode_id,
        decision_at=_T0,
        candidate_ids=intervention_ids,
        forecast_targets=(ForecastTarget(target_id='binary-primary-endpoint', horizon_days=30),),
    )
    package = cast(
        LoadedProspectiveDecisionPackage,
        SimpleNamespace(
            manifest=SimpleNamespace(episode=SimpleNamespace(decision_snapshot=SimpleNamespace(config=config))),
            protocol_artifacts={
                'candidate_set_definition': definition_bytes,
                'outcome_adjudication_spec': spec_bytes,
            },
        ),
    )
    return entry, verifier_spec, loaded, package


def test_promotion_decision_admission_binds_task_roles_and_protocol_bytes() -> None:
    entry, spec, loaded, package = _decision_admission_fixture()
    _verify_immport_scientific_decision(
        entry=entry,
        spec=spec,
        loaded=loaded,
        package=package,
    )

    wrong_task = cast(
        LoadedProspectiveDecisionPackage,
        SimpleNamespace(
            manifest=SimpleNamespace(
                episode=SimpleNamespace(
                    decision_snapshot=SimpleNamespace(
                        config=SimpleNamespace(
                            **(
                                package.manifest.episode.decision_snapshot.config.__dict__
                                | {'task_type': 'preclinical_candidate_advancement'}
                            )
                        )
                    )
                )
            ),
            protocol_artifacts=package.protocol_artifacts,
        ),
    )
    with pytest.raises(ProspectiveAdmissionError, match='exact ImmPort arm task'):
        _verify_immport_scientific_decision(
            entry=entry,
            spec=spec,
            loaded=loaded,
            package=wrong_task,
        )

    forged_candidates = tuple(item.model_copy(update={'eligible': True}) for item in loaded.candidates)
    forged_loaded = LoadedCapturePromotion(**(loaded.__dict__ | {'candidates': forged_candidates}))
    with pytest.raises(ProspectiveAdmissionError, match='incomplete, inconsistent'):
        _verify_immport_scientific_decision(
            entry=entry,
            spec=spec,
            loaded=forged_loaded,
            package=package,
        )
