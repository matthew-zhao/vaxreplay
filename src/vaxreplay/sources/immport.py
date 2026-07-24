"""Fail-closed ImmPort study/arm normalization for prospective Tier-A campaigns.

This module intentionally implements a narrow scientific claim.  One study accession is
precommitted; identical ``study`` and ``release_file`` manifest snapshots bracket the complete
study-specific ``arm``, ``experiment``, and ``link`` responses; identical pinned OpenAPI
documents bracket that panel; and every returned arm/cohort becomes a candidate.  No result
endpoint, participant row, ranking, top-N rule, or outcome-dependent filter is available to
the worker.

ImmPort requires authentication for the study-specific endpoints.  The raw bearer value must
never enter a promotion archive.  :class:`ImmportSanitizedCaptureReceipt` is the exact contract
expected from a separately deployed secret-safe collector: it records that authentication was
applied while making the credential value structurally unrepresentable.  It remains a collector
assertion, not a publisher signature.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, SourceType, StrictModel
from vaxreplay.operations.promotion import (
    AdapterRunResult,
    AdapterSourceInput,
    SourceVerificationInput,
    SourceVerifierRunResult,
)
from vaxreplay.operations.promotion_schema import (
    AuthoritativeReleaseBasis,
    AuthoritativeSourceRelease,
    NormalizedRecordReference,
    SourceRecordBinding,
    SourceRecordDisposition,
    SourceVerificationResult,
    SourceVerifierIdentity,
)
from vaxreplay.operations.schema import aware_utc

IMMPORT_LAYOUT_SCHEMA_VERSION = 'vaxreplay.immport-study-layout.v0.1'
IMMPORT_RECEIPT_SCHEMA_VERSION = 'vaxreplay.immport-sanitized-capture-receipt.v0.1'
IMMPORT_SOURCE_POLICY_SCHEMA_VERSION = 'vaxreplay.immport-source-verifier-policy.v0.2'
IMMPORT_ARM_ADAPTER_POLICY_SCHEMA_VERSION = 'vaxreplay.immport-arm-adapter-policy.v0.2'
IMMPORT_ARM_MAP_SCHEMA_VERSION = 'vaxreplay.immport-arm-candidate-map.v0.2'
IMMPORT_STUDY_UNIVERSE_SCHEMA_VERSION = 'vaxreplay.immport-study-universe.v0.1'
IMMPORT_CANDIDATE_SET_DEFINITION_SCHEMA_VERSION = 'vaxreplay.immport-arm-candidate-set-definition.v0.1'
IMMPORT_SOURCE_VERIFIER_ID = 'immport-shared-data-study-panel-offline-verifier'
IMMPORT_SOURCE_VERIFIER_VERSION = 'v0.2'
IMMPORT_ARM_ADAPTER_ID = 'immport-complete-study-arm-catalog-adapter'
IMMPORT_ARM_ADAPTER_VERSION = 'v0.2'
IMMPORT_ARM_ADAPTER_EXCLUSION_REASON_CODES = (
    'non_exact_clinical_trial_registry_link',
    'source_metadata_record',
)

_ORIGIN = 'https://www.immport.org'
_OPENAPI_URL = f'{_ORIGIN}/data/query/v3/api-docs'
_ARTIFACT_ID_PATTERN = r'^[a-z][a-z0-9._-]{0,109}$'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MD5_PATTERN = r'^[0-9a-f]{32}$'
_UUID_PATTERN = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
_STUDY_RE = re.compile(r'^SDY[0-9]{1,12}$')
_ARM_RE = re.compile(r'^ARM[0-9]{1,12}$')
_EXPERIMENT_RE = re.compile(r'^EXP[0-9]{1,12}$')
_RELEASE_RE = re.compile(r'^DR[0-9]+(?:\.[0-9]+)?$')
_CTGOV_PATH_RE = re.compile(r'^/(?:study|ct2/show|show)/(NCT[0-9]{8})$')
_PUBLIC_SOURCE_IDENTIFIER_RE = re.compile(
    r'(?<![A-Za-z0-9])(?:SDY|ARM|EXP|NCT)[0-9]+(?![A-Za-z0-9])',
    re.IGNORECASE,
)
_PUBLIC_POLICY_IDENTIFIER_RE = re.compile(r'(?:SDY|ARM|EXP|NCT)[0-9]+', re.IGNORECASE)
_PUBLIC_DOI_RE = re.compile(r'(?<![A-Za-z0-9])10\.[0-9]{4,9}/[^\s]+', re.IGNORECASE)
_PUBLIC_URL_RE = re.compile(r'https?://[^\s]+', re.IGNORECASE)
_MAX_TEXT_CHARS = 8_192


class ImmportProductionSourceError(ValueError):
    """An ImmPort capture cannot support the committed source claim."""


class ImmportProductionAdapterError(ValueError):
    """Verified ImmPort records cannot support the committed arm universe."""


class ImmportScientificContractError(ValueError):
    """ImmPort scientific selection or candidate semantics are not exactly precommitted."""


class ImmportStudyUniverseEntry(StrictModel):
    """One accession disposition frozen before any covered capture.

    Exclusion reasons are deliberately closed and cannot mention an observed arm, result, or
    outcome.  The registry decides the study universe; it must not be reconstructed by filtering
    the later capture.
    """

    study_accession: str = Field(pattern=r'^SDY[0-9]{1,12}$')
    disposition: Literal['selected', 'excluded']
    reason_code: Literal[
        'predeclared_eligible_study',
        'administrative_scope_exclusion',
        'deduplicated_by_predeclared_rule',
        'not_a_shared_clinical_trial_at_registry_freeze',
        'out_of_scope_pathogen_at_registry_freeze',
    ]

    @model_validator(mode='after')
    def validate_reason(self) -> Self:
        if (self.disposition == 'selected') != (self.reason_code == 'predeclared_eligible_study'):
            raise ValueError('only selected studies use the predeclared eligible-study reason')
        return self


class ImmportStudyUniverseRegistry(StrictModel):
    """Canonical SDY universe whose digest is pinned into both hermetic policies."""

    schema_version: Literal['vaxreplay.immport-study-universe.v0.1'] = IMMPORT_STUDY_UNIVERSE_SCHEMA_VERSION
    registry_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    campaign_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    frozen_at: datetime
    first_capture_not_before: datetime
    task_type: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    enumeration_semantics: Literal['externally_curated_sdy_accessions_exhaustively_dispositioned_before_capture'] = (
        'externally_curated_sdy_accessions_exhaustively_dispositioned_before_capture'
    )
    prohibited_selection_inputs: tuple[
        Literal['arm_payload', 'participant_payload', 'result_payload', 'post_freeze_outcome'], ...
    ] = ('arm_payload', 'participant_payload', 'post_freeze_outcome', 'result_payload')
    studies: tuple[ImmportStudyUniverseEntry, ...] = Field(min_length=1)

    @field_validator('frozen_at', 'first_capture_not_before')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort study-universe timestamp')

    @field_validator('prohibited_selection_inputs')
    @classmethod
    def validate_prohibited_inputs(
        cls,
        value: tuple[
            Literal['arm_payload', 'participant_payload', 'result_payload', 'post_freeze_outcome'],
            ...,
        ],
    ) -> tuple[
        Literal['arm_payload', 'participant_payload', 'result_payload', 'post_freeze_outcome'],
        ...,
    ]:
        expected = ('arm_payload', 'participant_payload', 'post_freeze_outcome', 'result_payload')
        if value != expected:
            raise ValueError('study-universe registry must prohibit every post-selection input class')
        return value

    @field_validator('studies')
    @classmethod
    def validate_studies(
        cls,
        value: tuple[ImmportStudyUniverseEntry, ...],
    ) -> tuple[ImmportStudyUniverseEntry, ...]:
        accessions = tuple(item.study_accession for item in value)
        if accessions != tuple(sorted(accessions)) or len(accessions) != len(set(accessions)):
            raise ValueError('study-universe entries must use sorted unique SDY accessions')
        if not any(item.disposition == 'selected' for item in value):
            raise ValueError('study-universe registry must select at least one study')
        return value

    @model_validator(mode='after')
    def validate_freeze_order(self) -> Self:
        if self.frozen_at >= self.first_capture_not_before:
            raise ValueError('study-universe registry must be frozen before its first capture slot')
        return self


def immport_study_universe_bytes(registry: ImmportStudyUniverseRegistry) -> bytes:
    if not isinstance(registry, ImmportStudyUniverseRegistry):
        raise TypeError('registry must be an ImmportStudyUniverseRegistry')
    return canonical_json_bytes(registry)


def immport_study_universe_sha256(registry: ImmportStudyUniverseRegistry) -> str:
    return hashlib.sha256(immport_study_universe_bytes(registry)).hexdigest()


class ImmportPromotionLayout(StrictModel):
    """Exact one-study endpoint inventory in serial, release-bracketed collector order."""

    schema_version: Literal['vaxreplay.immport-study-layout.v0.1'] = IMMPORT_LAYOUT_SCHEMA_VERSION
    study_accession: str = Field(pattern=r'^SDY[0-9]{1,12}$')
    openapi_before_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    study_before_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    manifest_before_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    arm_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    experiment_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    link_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    manifest_after_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    study_after_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    openapi_after_artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)

    @model_validator(mode='after')
    def validate_collection_order(self) -> Self:
        identifiers = self.artifact_ids
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('every ImmPort artifact ID must be unique')
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError('ImmPort artifact IDs must encode the serial OpenAPI/study/manifest data bracket')
        return self

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.openapi_before_artifact_id,
            self.study_before_artifact_id,
            self.manifest_before_artifact_id,
            self.arm_artifact_id,
            self.experiment_artifact_id,
            self.link_artifact_id,
            self.manifest_after_artifact_id,
            self.study_after_artifact_id,
            self.openapi_after_artifact_id,
        )

    @property
    def endpoint_artifact_ids(self) -> tuple[str, ...]:
        return (
            self.study_before_artifact_id,
            self.manifest_before_artifact_id,
            self.arm_artifact_id,
            self.experiment_artifact_id,
            self.link_artifact_id,
            self.manifest_after_artifact_id,
            self.study_after_artifact_id,
        )


class ImmportTlsPeerBinding(StrictModel):
    """Minimal TLS facts retained by the credentialed collector.

    Arbitrary peer-address/cipher strings are deliberately absent so they cannot become a covert
    path for bearer material into an archive.
    """

    server_name: Literal['www.immport.org'] = 'www.immport.org'
    tls_version: Literal['TLSv1.2', 'TLSv1.3']
    certificate_der_sha256: str = Field(pattern=_SHA256_PATTERN)


class ImmportSanitizedCaptureReceipt(StrictModel):
    """Credential-free receipt emitted by the dedicated authenticated collector.

    There is deliberately no arbitrary request-header or redirect URL field.  A bearer value,
    cookie, or expiring object-store URL therefore cannot be represented by this schema.
    """

    schema_version: Literal['vaxreplay.immport-sanitized-capture-receipt.v0.1'] = IMMPORT_RECEIPT_SCHEMA_VERSION
    method: Literal['GET'] = 'GET'
    requested_url: str = Field(min_length=1, max_length=8192)
    final_url: str = Field(min_length=1, max_length=8192)
    authentication: Literal['none', 'immport_scoped_api_key_bearer_redacted']
    authorization_applied: bool
    credential_source: Literal['not_applicable', 'runtime_secret_broker']
    credential_material_persisted: Literal[False] = False
    cookie_material_persisted: Literal[False] = False
    presigned_url_persisted: Literal[False] = False
    request_accept: Literal['application/json'] = 'application/json'
    request_accept_encoding: Literal['identity'] = 'identity'
    status_code: Literal[200] = 200
    response_content_type: Literal[
        'application/json',
        'application/json;charset=UTF-8',
        'application/json;charset=utf-8',
        'application/json; charset=UTF-8',
        'application/json; charset=utf-8',
    ]
    body_sha256: str = Field(pattern=_SHA256_PATTERN)
    body_byte_count: int = Field(ge=0, le=2 * 1024 * 1024 * 1024)
    started_at: datetime
    completed_at: datetime
    tls_peer: ImmportTlsPeerBinding
    collector_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('started_at', 'completed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort collector timestamp')

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError('ImmPort receipt completion cannot precede its start')
        if self.final_url != self.requested_url:
            raise ValueError('ImmPort authenticated collection rejects redirects')
        _validate_official_url(self.requested_url)
        required_auth = self.requested_url != _OPENAPI_URL
        if required_auth != self.authorization_applied:
            raise ValueError('ImmPort receipt authorization flag differs from the endpoint profile')
        expected_authentication = 'immport_scoped_api_key_bearer_redacted' if required_auth else 'none'
        if self.authentication != expected_authentication:
            raise ValueError('ImmPort receipt authentication mode differs from the endpoint profile')
        expected_credential_source = 'runtime_secret_broker' if required_auth else 'not_applicable'
        if self.credential_source != expected_credential_source:
            raise ValueError('ImmPort receipt credential source differs from the endpoint profile')
        return self


class ImmportSourceVerifierPolicy(StrictModel):
    """Precommitted source contract for one shared clinical study."""

    schema_version: Literal['vaxreplay.immport-source-verifier-policy.v0.2'] = IMMPORT_SOURCE_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    layout: ImmportPromotionLayout
    expected_openapi_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_openapi_info_version: str = Field(min_length=1, max_length=100)
    expected_latest_release_version: str = Field(pattern=r'^DR[0-9]+(?:\.[0-9]+)?$')
    expected_latest_release_date: date
    expected_collector_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    expected_collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_collector_execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    accepted_tls_versions: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...] = ('TLSv1.2', 'TLSv1.3')
    require_shared_study: Literal[True] = True
    require_clinical_trial: Literal[True] = True
    captures_per_verification: Literal[1] = 1
    source_authentication: Literal['sanitized_authenticated_collector_assertion_with_system_ca_tls'] = (
        'sanitized_authenticated_collector_assertion_with_system_ca_tls'
    )
    completeness_semantics: Literal['complete_one_study_api_arrays_and_release_file_manifest'] = (
        'complete_one_study_api_arrays_and_release_file_manifest'
    )
    source_release_semantics: Literal[
        'study_latest_release_date_end_of_day_america_new_york_upper_bound_not_scientific_change'
    ] = 'study_latest_release_date_end_of_day_america_new_york_upper_bound_not_scientific_change'
    forbidden_endpoint_classes: tuple[Literal['participant', 'result', 'subject', 'download'], ...] = (
        'download',
        'participant',
        'result',
        'subject',
    )

    @field_validator('accepted_tls_versions')
    @classmethod
    def validate_tls_versions(
        cls,
        value: tuple[Literal['TLSv1.2', 'TLSv1.3'], ...],
    ) -> tuple[Literal['TLSv1.2', 'TLSv1.3'], ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('accepted TLS versions must be sorted and unique')
        return value

    @field_validator('forbidden_endpoint_classes')
    @classmethod
    def validate_forbidden_endpoint_classes(
        cls,
        value: tuple[Literal['participant', 'result', 'subject', 'download'], ...],
    ) -> tuple[Literal['participant', 'result', 'subject', 'download'], ...]:
        if value != ('download', 'participant', 'result', 'subject'):
            raise ValueError('the production ImmPort profile must forbid participant/result/download APIs')
        return value


class ImmportArmAdapterPolicy(StrictModel):
    """All-arm normalization policy with no rank/count/type filtering."""

    schema_version: Literal['vaxreplay.immport-arm-adapter-policy.v0.2'] = IMMPORT_ARM_ADAPTER_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    episode_id: str = Field(min_length=1, max_length=1024)
    study_accession: str = Field(pattern=r'^SDY[0-9]{1,12}$')
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_adjudication_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_type: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    decision_at: datetime
    minimum_candidate_count: int = Field(default=2, ge=2, le=10_000)
    license_id: Literal['ImmPort User Agreement'] = 'ImmPort User Agreement'
    candidate_universe_semantics: Literal['every_returned_arm_or_cohort_without_ranking_sampling_or_type_filter'] = (
        'every_returned_arm_or_cohort_without_ranking_sampling_or_type_filter'
    )
    candidate_unit_semantics: Literal['trial_intervention_arm_not_vaccine_construct'] = (
        'trial_intervention_arm_not_vaccine_construct'
    )
    vaccine_construct_mapping: Literal['unverified_not_claimed_and_not_usable_for_construct_level_evaluation'] = (
        'unverified_not_claimed_and_not_usable_for_construct_level_evaluation'
    )
    arm_role_field: Literal['typePreferred'] = 'typePreferred'
    intervention_arm_types: tuple[str, ...] = ('Experimental Arm',)
    control_arm_types: tuple[str, ...] = (
        'Active Comparator Arm',
        'No Intervention Arm',
        'Placebo Comparator Arm',
    )
    control_handling: Literal['retained_as_ineligible_context_and_never_ranked'] = (
        'retained_as_ineligible_context_and_never_ranked'
    )
    public_identity_semantics: Literal['public_evidence_omits_sdy_arm_nct_doi_title_and_result_descriptions'] = (
        'public_evidence_omits_sdy_arm_nct_doi_title_and_result_descriptions'
    )
    experiment_semantics: Literal['study_level_measurement_techniques_only_not_arm_linked'] = (
        'study_level_measurement_techniques_only_not_arm_linked'
    )

    @field_validator('decision_at')
    @classmethod
    def validate_decision_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'ImmPort arm decision_at')

    @field_validator('episode_id')
    @classmethod
    def validate_public_episode_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError('ImmPort public episode_id must be bounded clean text')
        if (
            _PUBLIC_URL_RE.search(value) is not None
            or _PUBLIC_DOI_RE.search(value) is not None
            or _PUBLIC_POLICY_IDENTIFIER_RE.search(value) is not None
        ):
            raise ValueError('ImmPort public episode_id cannot contain a source, registry, DOI, or URL identifier')
        return value

    @field_validator('intervention_arm_types', 'control_arm_types')
    @classmethod
    def validate_arm_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('ImmPort arm-type allowlists must be nonempty, sorted, and unique')
        if any(
            not item
            or item != item.strip()
            or len(item) > 256
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in item)
            for item in value
        ):
            raise ValueError('ImmPort arm-type allowlists require bounded clean exact strings')
        return value

    @model_validator(mode='after')
    def validate_arm_type_partition(self) -> Self:
        if set(self.intervention_arm_types) & set(self.control_arm_types):
            raise ValueError('intervention and control arm types must be disjoint')
        return self


class ImmportArmCandidateMapEntry(StrictModel):
    candidate_id: str = Field(pattern=r'^cand-immport-[0-9a-f]{32}$')
    study_accession: str = Field(pattern=r'^SDY[0-9]{1,12}$')
    arm_accession: str = Field(pattern=r'^ARM[0-9]{1,12}$')
    latest_release_version: str = Field(pattern=r'^DR[0-9]+(?:\.[0-9]+)?$')
    nct_ids: tuple[str, ...] = ()
    arm_role: Literal['intervention', 'control', 'unclassified']
    decision_disposition: Literal[
        'rankable_intervention_arm',
        'contextual_control_not_ranked',
        'blocked_unclassified_arm_type',
    ]
    vaccine_construct_mapping: Literal['unverified_not_claimed'] = 'unverified_not_claimed'

    @field_validator('nct_ids')
    @classmethod
    def validate_nct_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('ImmPort NCT IDs must be sorted and unique')
        if any(re.fullmatch(r'NCT[0-9]{8}', item) is None for item in value):
            raise ValueError('ImmPort NCT IDs must use canonical uppercase syntax')
        return value

    @model_validator(mode='after')
    def validate_disposition(self) -> Self:
        expected = {
            'intervention': 'rankable_intervention_arm',
            'control': 'contextual_control_not_ranked',
            'unclassified': 'blocked_unclassified_arm_type',
        }
        if self.decision_disposition != expected[self.arm_role]:
            raise ValueError('ImmPort arm role and decision disposition disagree')
        return self


class ImmportArmCandidateMap(StrictModel):
    schema_version: Literal['vaxreplay.immport-arm-candidate-map.v0.2'] = IMMPORT_ARM_MAP_SCHEMA_VERSION
    policy_id: str
    episode_id: str
    task_type: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_adjudication_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_unit_semantics: Literal['trial_intervention_arm_not_vaccine_construct'] = (
        'trial_intervention_arm_not_vaccine_construct'
    )
    organizer_only: Literal[True] = True
    candidates: tuple[ImmportArmCandidateMapEntry, ...] = Field(min_length=2)

    @field_validator('candidates')
    @classmethod
    def validate_candidates(
        cls,
        value: tuple[ImmportArmCandidateMapEntry, ...],
    ) -> tuple[ImmportArmCandidateMapEntry, ...]:
        keys = tuple((item.candidate_id, item.study_accession, item.arm_accession) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('ImmPort candidate map must be canonically sorted and unique')
        if len({item.arm_accession for item in value}) != len(value):
            raise ValueError('ImmPort candidate map arm accessions must be unique')
        return value


class ImmportArmCandidateSetDefinition(StrictModel):
    """Public, label-free decision protocol produced by the pinned adapter."""

    schema_version: Literal['vaxreplay.immport-arm-candidate-set-definition.v0.1'] = (
        IMMPORT_CANDIDATE_SET_DEFINITION_SCHEMA_VERSION
    )
    policy_id: str
    episode_id: str
    task_type: Literal['early_clinical_arm_prioritization'] = 'early_clinical_arm_prioritization'
    study_universe_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_candidate_map_sha256: str = Field(pattern=_SHA256_PATTERN)
    outcome_adjudication_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_unit_semantics: Literal['trial_intervention_arm_not_vaccine_construct'] = (
        'trial_intervention_arm_not_vaccine_construct'
    )
    vaccine_construct_mapping: Literal['unverified_not_claimed_and_not_usable_for_construct_level_evaluation'] = (
        'unverified_not_claimed_and_not_usable_for_construct_level_evaluation'
    )
    intervention_candidate_ids: tuple[str, ...] = ()
    contextual_control_ids: tuple[str, ...] = ()
    blocked_unclassified_ids: tuple[str, ...] = ()

    @field_validator(
        'intervention_candidate_ids',
        'contextual_control_ids',
        'blocked_unclassified_ids',
    )
    @classmethod
    def validate_candidate_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('ImmPort candidate-set ID partitions must be sorted and unique')
        if any(re.fullmatch(r'cand-immport-[0-9a-f]{32}', item) is None for item in value):
            raise ValueError('ImmPort candidate-set IDs must use opaque adapter identity syntax')
        return value

    @model_validator(mode='after')
    def validate_partitions(self) -> Self:
        partitions = (
            set(self.intervention_candidate_ids),
            set(self.contextual_control_ids),
            set(self.blocked_unclassified_ids),
        )
        if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
            raise ValueError('ImmPort candidate-set ID partitions must be disjoint')
        if not self.intervention_candidate_ids:
            raise ValueError('ImmPort early-clinical task requires at least one intervention arm')
        return self


@dataclass(frozen=True)
class _ArmCandidate:
    source: SourceRecordBinding
    raw: dict[str, Any]
    candidate: CandidateRecord
    evidence: EvidenceRecord
    arm_accession: str
    arm_role: Literal['intervention', 'control', 'unclassified']


def immport_source_verifier_policy_bytes(policy: ImmportSourceVerifierPolicy) -> bytes:
    if not isinstance(policy, ImmportSourceVerifierPolicy):
        raise TypeError('policy must be an ImmportSourceVerifierPolicy')
    return canonical_json_bytes(policy)


def immport_arm_adapter_policy_bytes(policy: ImmportArmAdapterPolicy) -> bytes:
    if not isinstance(policy, ImmportArmAdapterPolicy):
        raise TypeError('policy must be an ImmportArmAdapterPolicy')
    return canonical_json_bytes(policy)


def parse_immport_study_universe(payload: bytes) -> ImmportStudyUniverseRegistry:
    """Parse only exact canonical registry bytes suitable for a precommitment digest."""

    return _canonical_model(
        payload,
        ImmportStudyUniverseRegistry,
        ImmportScientificContractError,
        'study-universe registry',
    )


def verify_immport_study_universe_precommit(
    registry_bytes: bytes,
    *,
    source_policies: tuple[ImmportSourceVerifierPolicy, ...],
    adapter_policies: tuple[ImmportArmAdapterPolicy, ...],
    campaign_id: str,
    first_scheduled_for: datetime,
) -> ImmportStudyUniverseRegistry:
    """Cross-bind a frozen SDY registry to every exact hermetic policy.

    A caller must obtain ``registry_bytes`` independently.  Both worker policy hashes are already
    covered by the witnessed pre-capture plan; requiring their embedded registry digest makes a
    later substitute registry or post-hoc study subset fail closed.
    """

    registry = parse_immport_study_universe(registry_bytes)
    first_slot = aware_utc(first_scheduled_for, 'first ImmPort scheduled capture')
    digest = hashlib.sha256(registry_bytes).hexdigest()
    if registry.campaign_id != campaign_id:
        raise ImmportScientificContractError('ImmPort study registry belongs to another campaign')
    if registry.first_capture_not_before != first_slot or registry.frozen_at >= first_slot:
        raise ImmportScientificContractError(
            'ImmPort study registry freeze or first-slot boundary differs from the capture plan'
        )
    if not source_policies or len(source_policies) != len(adapter_policies):
        raise ImmportScientificContractError('ImmPort study registry requires paired source and adapter policies')
    source_by_study = {item.layout.study_accession: item for item in source_policies}
    adapter_by_study = {item.study_accession: item for item in adapter_policies}
    if len(source_by_study) != len(source_policies) or len(adapter_by_study) != len(adapter_policies):
        raise ImmportScientificContractError('ImmPort worker policies duplicate an SDY accession')
    selected = {item.study_accession for item in registry.studies if item.disposition == 'selected'}
    if set(source_by_study) != selected or set(adapter_by_study) != selected:
        raise ImmportScientificContractError(
            'ImmPort worker policies must exactly cover every registry-selected SDY accession'
        )
    for accession in sorted(selected):
        source = source_by_study[accession]
        adapter = adapter_by_study[accession]
        if (
            source.study_universe_registry_sha256 != digest
            or adapter.study_universe_registry_sha256 != digest
            or source.source_id != adapter.source_id
            or adapter.task_type != registry.task_type
        ):
            raise ImmportScientificContractError(
                'ImmPort source/adapter policy differs from its registry or task precommitment'
            )
    return registry


def verify_tier_a_immport_source(
    verifier_input: SourceVerificationInput,
    policy_bytes: bytes,
    *,
    implementation_sha256: str,
    execution_environment_sha256: str,
) -> SourceVerifierRunResult:
    """Verify a stable, authenticated, complete study-specific ImmPort panel."""

    policy = _canonical_source_policy(policy_bytes)
    _require_sha256(implementation_sha256, 'implementation_sha256')
    _require_sha256(execution_environment_sha256, 'execution_environment_sha256')
    if verifier_input.source_id != policy.source_id or len(verifier_input.captures) != 1:
        raise ImmportProductionSourceError('ImmPort verification requires its one committed capture')
    capture = verifier_input.captures[0]
    if capture.binding.source_id != policy.source_id:
        raise ImmportProductionSourceError('ImmPort capture belongs to a different source')

    artifacts = {item.role: item for item in capture.artifacts}
    if len(artifacts) != len(capture.artifacts):
        raise ImmportProductionSourceError('ImmPort capture contains duplicate artifact roles')
    expected_bodies = {f'body.{item}' for item in policy.layout.artifact_ids}
    expected_receipts = {f'receipt.{item}' for item in policy.layout.artifact_ids}
    expected_source_roles = expected_bodies | expected_receipts
    structural_roles = {'collection-plan', 'run-manifest'}
    extra_roles = set(artifacts) - expected_source_roles
    if (
        {role for role in artifacts if role.startswith('body.')} != expected_bodies
        or {role for role in artifacts if role.startswith('receipt.')} != expected_receipts
        or extra_roles not in (set(), structural_roles)
    ):
        raise ImmportProductionSourceError(
            'ImmPort artifact inventory differs from its exact source and structural role set'
        )

    receipts: dict[str, ImmportSanitizedCaptureReceipt] = {}
    for artifact_id in policy.layout.artifact_ids:
        body = artifacts[f'body.{artifact_id}']
        if hashlib.sha256(body.payload).hexdigest() != body.sha256 or len(body.payload) != body.byte_count:
            raise ImmportProductionSourceError('ImmPort body differs from its promoted binding')
        receipts[artifact_id] = _verify_receipt(
            artifacts[f'receipt.{artifact_id}'].payload,
            body_sha256=body.sha256,
            body_bytes=body.byte_count,
            expected_url=_artifact_url(policy.layout, artifact_id),
            captured_at=capture.binding.captured_at,
            accepted_tls_versions=policy.accepted_tls_versions,
            expected_collector_id=policy.expected_collector_id,
            expected_collector_implementation_sha256=(policy.expected_collector_implementation_sha256),
            expected_collector_execution_environment_sha256=(policy.expected_collector_execution_environment_sha256),
        )

    ordered_receipts = tuple(receipts[item] for item in policy.layout.artifact_ids)
    if any(
        left.completed_at > right.started_at
        for left, right in zip(ordered_receipts[:-1], ordered_receipts[1:], strict=True)
    ):
        raise ImmportProductionSourceError('ImmPort receipt times do not prove the committed serial release bracket')

    before_artifact = artifacts[f'body.{policy.layout.openapi_before_artifact_id}']
    after_artifact = artifacts[f'body.{policy.layout.openapi_after_artifact_id}']
    if before_artifact.payload != after_artifact.payload:
        raise ImmportProductionSourceError('ImmPort OpenAPI bytes changed during capture')
    if before_artifact.sha256 != policy.expected_openapi_sha256:
        raise ImmportProductionSourceError('ImmPort OpenAPI bytes differ from the precommitment')
    openapi = _strict_json_object(before_artifact.payload, 'OpenAPI document')
    schema_properties = _validate_openapi_contract(openapi, policy.expected_openapi_info_version)

    study_before_artifact = artifacts[f'body.{policy.layout.study_before_artifact_id}']
    study_after_artifact = artifacts[f'body.{policy.layout.study_after_artifact_id}']
    if study_before_artifact.payload != study_after_artifact.payload:
        raise ImmportProductionSourceError('ImmPort study bytes changed during capture')
    manifest_before_artifact = artifacts[f'body.{policy.layout.manifest_before_artifact_id}']
    manifest_after_artifact = artifacts[f'body.{policy.layout.manifest_after_artifact_id}']
    if manifest_before_artifact.payload != manifest_after_artifact.payload:
        raise ImmportProductionSourceError('ImmPort release_file manifest bytes changed during capture')

    raw_by_kind: dict[str, tuple[dict[str, Any], ...]] = {}
    for kind, artifact_id in _endpoint_artifacts(policy.layout).items():
        value = _strict_json_value(artifacts[f'body.{artifact_id}'].payload, f'{kind} response')
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ImmportProductionSourceError(f'ImmPort {kind} response must be an object array')
        rows = tuple(value)
        if len({hashlib.sha256(canonical_json_bytes(item)).hexdigest() for item in rows}) != len(rows):
            raise ImmportProductionSourceError(f'ImmPort {kind} response contains duplicate rows')
        raw_by_kind[kind] = rows

    studies = raw_by_kind['study']
    if len(studies) != 1:
        raise ImmportProductionSourceError('ImmPort study endpoint must return exactly one study')
    study = studies[0]
    _validate_schema_keys(study, schema_properties['StudyApi'], 'study')
    study_accession = _required_accession(study, 'studyAccession', _STUDY_RE, 'study')
    if study_accession != policy.layout.study_accession:
        raise ImmportProductionSourceError('ImmPort study response differs from the committed accession')
    if study.get('sharedStudy') != 'Y':
        raise ImmportProductionSourceError('ImmPort production profile admits only shared studies')
    if study.get('clinicalTrial') != 'Y':
        raise ImmportProductionSourceError('ImmPort production arm profile requires a clinical trial')
    latest_version = study.get('latestDataReleaseVersion')
    latest_date_text = study.get('latestDataReleaseDate')
    if latest_version != policy.expected_latest_release_version:
        raise ImmportProductionSourceError('ImmPort latest release version differs from the precommitment')
    latest_date = _parse_date(latest_date_text, 'latestDataReleaseDate')
    if latest_date != policy.expected_latest_release_date:
        raise ImmportProductionSourceError('ImmPort latest release date differs from the precommitment')
    source_release_at = _conservative_release_upper_bound(latest_date)
    if source_release_at > capture.binding.captured_at:
        raise ImmportProductionSourceError(
            'ImmPort conservative release-date upper bound is after the selected capture'
        )

    for arm in raw_by_kind['arm']:
        _validate_schema_keys(arm, schema_properties['StudyArmApi'], 'arm')
        _require_same_study(arm, study_accession, 'arm')
        _required_accession(arm, 'armAccession', _ARM_RE, 'arm')
    _require_unique_field(raw_by_kind['arm'], 'armAccession', 'arm')

    for experiment in raw_by_kind['experiment']:
        _validate_schema_keys(experiment, schema_properties['StudyExperimentApi'], 'experiment')
        _require_same_study(experiment, study_accession, 'experiment')
        _required_accession(experiment, 'experimentAccession', _EXPERIMENT_RE, 'experiment')
    _require_unique_field(raw_by_kind['experiment'], 'experimentAccession', 'experiment')

    for link in raw_by_kind['link']:
        _validate_schema_keys(link, schema_properties['StudyLinkApi'], 'link')
        _require_same_study(link, study_accession, 'link')
        link_id = link.get('studyLinkId')
        if isinstance(link_id, bool) or not isinstance(link_id, int) or link_id < 0:
            raise ImmportProductionSourceError('ImmPort link has an invalid studyLinkId')
    _require_unique_field(raw_by_kind['link'], 'studyLinkId', 'link')

    manifest_rows = raw_by_kind['manifest']
    if not manifest_rows:
        raise ImmportProductionSourceError('ImmPort release_file manifest cannot be empty')
    for entry in manifest_rows:
        _validate_schema_keys(entry, schema_properties['FileDetails'], 'manifest')
        _validate_manifest_entry(entry, study_accession)
    _require_unique_field(manifest_rows, 'fileUUID', 'manifest')
    _require_unique_field(manifest_rows, 'path', 'manifest')
    if not any(
        _contains_exact_release_token(
            str(entry.get('fileName', '')),
            policy.expected_latest_release_version,
        )
        or _contains_exact_release_token(
            str(entry.get('path', '')),
            policy.expected_latest_release_version,
        )
        for entry in manifest_rows
    ):
        raise ImmportProductionSourceError(
            'ImmPort release_file manifest does not bind the committed release identifier'
        )

    records: list[SourceRecordBinding] = []
    _append_record(
        records,
        source_id=policy.source_id,
        record_id=f'api_contract:{before_artifact.sha256}',
        raw=openapi,
        artifact_sha256=before_artifact.sha256,
        locator=f'{_OPENAPI_URL}#info.version',
    )
    study_artifact = artifacts[f'body.{policy.layout.study_before_artifact_id}']
    _append_record(
        records,
        source_id=policy.source_id,
        record_id=f'study:{study_accession}',
        raw=study,
        artifact_sha256=study_artifact.sha256,
        locator=f'{_study_url(study_accession)}#studyAccession={study_accession}',
    )
    for kind, id_field, prefix in (
        ('arm', 'armAccession', 'arm'),
        ('experiment', 'experimentAccession', 'experiment'),
        ('link', 'studyLinkId', 'link'),
        ('manifest', 'fileUUID', 'manifest'),
    ):
        artifact_id = _endpoint_artifacts(policy.layout)[kind]
        body = artifacts[f'body.{artifact_id}']
        url = _artifact_url(policy.layout, artifact_id)
        for row in raw_by_kind[kind]:
            identity = str(row[id_field])
            _append_record(
                records,
                source_id=policy.source_id,
                record_id=f'{prefix}:{identity}',
                raw=row,
                artifact_sha256=body.sha256,
                locator=f'{url}#{id_field}={identity}',
            )

    ordered = tuple(sorted(records, key=lambda item: (item.source_id, item.source_record_id)))
    records_bytes = _jsonl(ordered, ImmportProductionSourceError)
    study_binding = next(item for item in ordered if item.source_record_id == f'study:{study_accession}')
    result = SourceVerificationResult(
        source_id=policy.source_id,
        verifier=SourceVerifierIdentity(
            verifier_id=IMMPORT_SOURCE_VERIFIER_ID,
            verifier_version=IMMPORT_SOURCE_VERIFIER_VERSION,
            implementation_sha256=implementation_sha256,
            execution_environment_sha256=execution_environment_sha256,
        ),
        verifier_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        verified_attempt_ids=(capture.binding.attempt_id,),
        source_release=AuthoritativeSourceRelease(
            source_release_at=source_release_at,
            basis=AuthoritativeReleaseBasis.SOURCE_API_PUBLICATION_FIELD,
            authority_locator=f'{_study_url(study_accession)}#latestDataReleaseDate',
            authority_field=(
                'latestDataReleaseDate end-of-day America/New_York upper bound; not evidence of a scientific change'
            ),
            evidence_attempt_id=capture.binding.attempt_id,
            evidence_role=f'body.{policy.layout.study_before_artifact_id}',
            evidence_sha256=study_binding.source_artifact_sha256,
            evidence_source_record_id=study_binding.source_record_id,
            evidence_source_record_sha256=study_binding.source_record_sha256,
        ),
        verified_capture_inventory_sha256=verifier_input.capture_inventory_sha256,
        verified_source_record_inventory_sha256=hashlib.sha256(records_bytes).hexdigest(),
        verified_source_record_count=len(ordered),
        result_codes=(
            'complete_one_study_api_arrays_and_release_file_manifest',
            'openapi_contract_stable_during_capture',
            'participant_result_subject_and_download_endpoints_forbidden',
            'release_identity_and_manifest_stable_during_capture',
            'sanitized_authenticated_collector_assertion_with_system_ca_tls',
            'study_release_metadata_is_availability_only_not_scientific_change',
            'study_selection_bound_to_precommitted_registry',
        ),
    )
    return SourceVerifierRunResult(result=result, verified_records=records_bytes)


def adapt_tier_a_immport_arms(
    inputs: tuple[AdapterSourceInput, ...],
    policy_bytes: bytes,
) -> AdapterRunResult:
    """Emit every returned arm/cohort using only a blinded structural projection."""

    policy = _canonical_adapter_policy(policy_bytes)
    if len(inputs) != 1 or inputs[0].source_id != policy.source_id:
        raise ImmportProductionAdapterError('ImmPort adaptation requires its one committed source')
    source_input = inputs[0]
    verification = source_input.verification_result
    if verification.source_id != policy.source_id:
        raise ImmportProductionAdapterError('ImmPort verification belongs to a different source')
    if (
        verification.verifier.verifier_id != IMMPORT_SOURCE_VERIFIER_ID
        or verification.verifier.verifier_version != IMMPORT_SOURCE_VERIFIER_VERSION
    ):
        raise ImmportProductionAdapterError('ImmPort adapter requires the production source profile')
    required_result_codes = {
        'complete_one_study_api_arrays_and_release_file_manifest',
        'participant_result_subject_and_download_endpoints_forbidden',
        'study_selection_bound_to_precommitted_registry',
        'study_release_metadata_is_availability_only_not_scientific_change',
    }
    if not required_result_codes.issubset(verification.result_codes):
        raise ImmportProductionAdapterError('ImmPort verification lacks required production-profile result codes')
    if verification.source_release.source_release_at > policy.decision_at:
        raise ImmportProductionAdapterError('ImmPort source release is after the decision cutoff')
    if any(capture.binding.captured_at > policy.decision_at for capture in source_input.captures):
        raise ImmportProductionAdapterError('ImmPort selected capture is after the decision cutoff')
    inventory_bytes = _jsonl(source_input.verified_records, ImmportProductionAdapterError)
    if (
        len(source_input.verified_records) != verification.verified_source_record_count
        or hashlib.sha256(inventory_bytes).hexdigest() != verification.verified_source_record_inventory_sha256
    ):
        raise ImmportProductionAdapterError('ImmPort adapter input differs from verified records')
    if any(record.source_id != policy.source_id for record in source_input.verified_records):
        raise ImmportProductionAdapterError('ImmPort adapter input contains a foreign source row')

    raw_rows = _rebind_rows(source_input)
    by_prefix: dict[str, list[tuple[SourceRecordBinding, dict[str, Any]]]] = {}
    for source in source_input.verified_records:
        prefix = source.source_record_id.partition(':')[0]
        by_prefix.setdefault(prefix, []).append((source, raw_rows[(source.source_id, source.source_record_id)]))
    study_rows = by_prefix.get('study', [])
    if len(study_rows) != 1:
        raise ImmportProductionAdapterError('ImmPort adapter requires exactly one study row')
    _study_source, study = study_rows[0]
    if study.get('studyAccession') != policy.study_accession:
        raise ImmportProductionAdapterError('ImmPort adapter study differs from its precommitment')
    latest_release_version = study.get('latestDataReleaseVersion')
    if not isinstance(latest_release_version, str) or _RELEASE_RE.fullmatch(latest_release_version) is None:
        raise ImmportProductionAdapterError('ImmPort study has an invalid latest release version')

    nct_ids: set[str] = set()
    exact_link_sources: list[SourceRecordBinding] = []
    excluded: dict[tuple[str, str], str] = {}
    for source, raw in by_prefix.get('link', []):
        values = _extract_nct_ids(raw)
        if len(values) == 1:
            nct_ids.update(values)
            exact_link_sources.append(source)
        else:
            excluded[(source.source_id, source.source_record_id)] = 'non_exact_clinical_trial_registry_link'

    measurement_techniques = tuple(
        sorted(
            {
                _clean_public_text(value, 'measurementTechnique', max_chars=256)
                for _source, raw in by_prefix.get('experiment', [])
                for value in (raw.get('measurementTechnique'),)
                if isinstance(value, str) and value.strip()
            }
        )
    )
    arms: list[_ArmCandidate] = []
    for source, raw in by_prefix.get('arm', []):
        arm_accession = _required_adapter_accession(raw, 'armAccession', _ARM_RE, 'arm')
        if raw.get('studyAccession') != policy.study_accession:
            raise ImmportProductionAdapterError('ImmPort arm belongs to another study')
        candidate_id = _candidate_id(policy, arm_accession)
        arm_role = _arm_role(policy, raw)
        candidate = CandidateRecord(
            episode_id=policy.episode_id,
            candidate_id=candidate_id,
            eligible=arm_role == 'intervention',
        )
        body = _render_arm_projection(raw, candidate_id, measurement_techniques)
        evidence_seed = canonical_json_bytes(
            {
                'adapter_id': IMMPORT_ARM_ADAPTER_ID,
                'adapter_version': IMMPORT_ARM_ADAPTER_VERSION,
                'episode_id': policy.episode_id,
                'source_record_id': source.source_record_id,
                'source_record_sha256': source.source_record_sha256,
            }
        )
        evidence = EvidenceRecord(
            episode_id=policy.episode_id,
            evidence_id=f'immport-evidence-{hashlib.sha256(evidence_seed).hexdigest()}',
            source_type=SourceType.EXPERIMENTAL,
            collected_at=None,
            available_at=verification.source_release.source_release_at,
            title=f'Blinded ImmPort arm evidence for {candidate_id}',
            body=body,
            body_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest(),
            related_candidate_ids=[candidate_id],
            provenance_url=f'{_ORIGIN}/data/query/',
            license_id=policy.license_id,
            derivation=(
                f'Deterministic {IMMPORT_ARM_ADAPTER_ID} {IMMPORT_ARM_ADAPTER_VERSION} '
                'projection over every returned arm. Public evidence omits source accessions, '
                'registry IDs, study titles/descriptions, experiment descriptions, manifests, '
                'participant rows, and result fields. Measurement techniques are study-level '
                'context and are not asserted to be linked to a particular arm.'
            ),
        )
        arms.append(_ArmCandidate(source, raw, candidate, evidence, arm_accession, arm_role))

    if len(arms) < policy.minimum_candidate_count:
        raise ImmportProductionAdapterError('complete ImmPort study panel has fewer arms than the committed minimum')
    if len({item.arm_accession for item in arms}) != len(arms):
        raise ImmportProductionAdapterError('ImmPort arm identities are duplicated')
    candidates = tuple(sorted((item.candidate for item in arms), key=lambda item: item.candidate_id))
    evidence = tuple(sorted((item.evidence for item in arms), key=lambda item: item.evidence_id))
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ImmportProductionAdapterError('ImmPort candidate identities are duplicated')

    candidate_refs = {
        item.candidate_id: _normalized_reference(item.episode_id, item.candidate_id, item) for item in candidates
    }
    evidence_refs = {
        item.evidence_id: _normalized_reference(item.episode_id, item.evidence_id, item) for item in evidence
    }
    all_candidate_refs = tuple(sorted(candidate_refs.values(), key=lambda item: (item.episode_id, item.record_id)))
    all_evidence_refs = tuple(sorted(evidence_refs.values(), key=lambda item: (item.episode_id, item.record_id)))
    normalized: dict[tuple[str, str], SourceRecordDisposition] = {}
    for item in arms:
        normalized[(item.source.source_id, item.source.source_record_id)] = SourceRecordDisposition(
            source_id=item.source.source_id,
            source_record_id=item.source.source_record_id,
            source_record_sha256=item.source.source_record_sha256,
            source_artifact_sha256=item.source.source_artifact_sha256,
            disposition='normalized',
            candidate_record_refs=(candidate_refs[item.candidate.candidate_id],),
            evidence_record_refs=(evidence_refs[item.evidence.evidence_id],),
        )
    shared_sources = [
        *(source for source, _raw in study_rows),
        *(source for source, _raw in by_prefix.get('experiment', [])),
        *exact_link_sources,
    ]
    for source in shared_sources:
        normalized[(source.source_id, source.source_record_id)] = SourceRecordDisposition(
            source_id=source.source_id,
            source_record_id=source.source_record_id,
            source_record_sha256=source.source_record_sha256,
            source_artifact_sha256=source.source_artifact_sha256,
            disposition='normalized',
            candidate_record_refs=all_candidate_refs,
            evidence_record_refs=all_evidence_refs,
        )
    for source in source_input.verified_records:
        key = (source.source_id, source.source_record_id)
        if source.source_record_id.startswith(('api_contract:', 'manifest:')):
            excluded[key] = 'source_metadata_record'
    if set(normalized) & set(excluded):
        raise ImmportProductionAdapterError('ImmPort source record has conflicting dispositions')
    all_source_keys = {(source.source_id, source.source_record_id) for source in source_input.verified_records}
    unresolved = sorted(all_source_keys - set(normalized) - set(excluded))
    if unresolved:
        raise ImmportProductionAdapterError(
            f'ImmPort adapter has no disposition for verified source records: {unresolved!r}'
        )

    dispositions = tuple(
        sorted(
            (
                normalized[key]
                if key in normalized
                else SourceRecordDisposition(
                    source_id=source.source_id,
                    source_record_id=source.source_record_id,
                    source_record_sha256=source.source_record_sha256,
                    source_artifact_sha256=source.source_artifact_sha256,
                    disposition='excluded',
                    reason_code=excluded[key],
                )
                for source in source_input.verified_records
                for key in ((source.source_id, source.source_record_id),)
            ),
            key=lambda item: (item.source_id, item.source_record_id),
        )
    )
    disposition_by_role: dict[
        Literal['intervention', 'control', 'unclassified'],
        Literal[
            'rankable_intervention_arm',
            'contextual_control_not_ranked',
            'blocked_unclassified_arm_type',
        ],
    ] = {
        'intervention': 'rankable_intervention_arm',
        'control': 'contextual_control_not_ranked',
        'unclassified': 'blocked_unclassified_arm_type',
    }
    candidate_map = ImmportArmCandidateMap(
        policy_id=policy.policy_id,
        episode_id=policy.episode_id,
        study_universe_registry_sha256=policy.study_universe_registry_sha256,
        outcome_adjudication_spec_sha256=policy.outcome_adjudication_spec_sha256,
        candidates=tuple(
            sorted(
                (
                    ImmportArmCandidateMapEntry(
                        candidate_id=item.candidate.candidate_id,
                        study_accession=policy.study_accession,
                        arm_accession=item.arm_accession,
                        latest_release_version=latest_release_version,
                        nct_ids=tuple(sorted(nct_ids)),
                        arm_role=item.arm_role,
                        decision_disposition=disposition_by_role[item.arm_role],
                    )
                    for item in arms
                ),
                key=lambda item: (item.candidate_id, item.study_accession, item.arm_accession),
            )
        ),
    )
    candidate_map_bytes = canonical_json_bytes(candidate_map)
    candidate_set = ImmportArmCandidateSetDefinition(
        policy_id=policy.policy_id,
        episode_id=policy.episode_id,
        study_universe_registry_sha256=policy.study_universe_registry_sha256,
        organizer_candidate_map_sha256=hashlib.sha256(candidate_map_bytes).hexdigest(),
        outcome_adjudication_spec_sha256=policy.outcome_adjudication_spec_sha256,
        intervention_candidate_ids=tuple(
            sorted(item.candidate.candidate_id for item in arms if item.arm_role == 'intervention')
        ),
        contextual_control_ids=tuple(
            sorted(item.candidate.candidate_id for item in arms if item.arm_role == 'control')
        ),
        blocked_unclassified_ids=tuple(
            sorted(item.candidate.candidate_id for item in arms if item.arm_role == 'unclassified')
        ),
    )
    return AdapterRunResult(
        candidate_records=_jsonl(candidates, ImmportProductionAdapterError),
        evidence_records=_jsonl(evidence, ImmportProductionAdapterError),
        dispositions=_jsonl(dispositions, ImmportProductionAdapterError),
        auxiliary_outputs={
            'immport-arm-candidate-map': candidate_map_bytes,
            'immport-candidate-set-definition': canonical_json_bytes(candidate_set),
        },
    )


def _canonical_source_policy(payload: bytes) -> ImmportSourceVerifierPolicy:
    return _canonical_model(
        payload,
        ImmportSourceVerifierPolicy,
        ImmportProductionSourceError,
        'source verifier',
    )


def _canonical_adapter_policy(payload: bytes) -> ImmportArmAdapterPolicy:
    return _canonical_model(
        payload,
        ImmportArmAdapterPolicy,
        ImmportProductionAdapterError,
        'arm adapter',
    )


def _canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    error_type: type[ValueError],
    label: str,
) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise error_type(f'ImmPort {label} policy must be nonempty exact bytes')
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise error_type(f'invalid ImmPort {label} policy: {error}') from error
    if payload != canonical_json_bytes(value):
        raise error_type(f'ImmPort {label} policy must use canonical JSON')
    return value


def _study_url(study_accession: str) -> str:
    return f'{_ORIGIN}/data/query/api/study/{study_accession}?format=json'


def _artifact_url(layout: ImmportPromotionLayout, artifact_id: str) -> str:
    if artifact_id in {layout.openapi_before_artifact_id, layout.openapi_after_artifact_id}:
        return _OPENAPI_URL
    accession = layout.study_accession
    urls = {
        layout.study_before_artifact_id: _study_url(accession),
        layout.study_after_artifact_id: _study_url(accession),
        layout.arm_artifact_id: f'{_ORIGIN}/data/query/api/study/arm/{accession}?format=json',
        layout.experiment_artifact_id: (f'{_ORIGIN}/data/query/api/study/experiment/{accession}?format=json'),
        layout.link_artifact_id: f'{_ORIGIN}/data/query/api/study/link/{accession}?format=json',
        layout.manifest_before_artifact_id: (
            f'{_ORIGIN}/data/query/api/study/manifest/{accession}?fileType=release_file&format=json'
        ),
        layout.manifest_after_artifact_id: (
            f'{_ORIGIN}/data/query/api/study/manifest/{accession}?fileType=release_file&format=json'
        ),
    }
    try:
        return urls[artifact_id]
    except KeyError as error:
        raise ImmportProductionSourceError(f'unknown ImmPort artifact ID: {artifact_id}') from error


def _endpoint_artifacts(layout: ImmportPromotionLayout) -> dict[str, str]:
    return {
        'study': layout.study_before_artifact_id,
        'arm': layout.arm_artifact_id,
        'experiment': layout.experiment_artifact_id,
        'link': layout.link_artifact_id,
        'manifest': layout.manifest_before_artifact_id,
    }


def _validate_official_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or parsed.netloc != 'www.immport.org'
        or parsed.hostname != 'www.immport.org'
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError('ImmPort receipt URL must use the exact official HTTPS origin')
    pairs = tuple(parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True))
    if len({name for name, _item in pairs}) != len(pairs):
        raise ValueError('ImmPort receipt URL query parameters must be unique')
    allowed = {'format', 'fileType'}
    if any(name not in allowed for name, _item in pairs):
        raise ValueError('ImmPort receipt URL contains an unapproved query parameter')
    if any(not item for _name, item in pairs):
        raise ValueError('ImmPort receipt URL query values must be nonempty')
    query = dict(pairs)
    if 'format' in query and query['format'] != 'json':
        raise ValueError('ImmPort receipt format query must be exactly json')
    if 'fileType' in query and query['fileType'] != 'release_file':
        raise ValueError('ImmPort receipt fileType query must be exactly release_file')


def _verify_receipt(
    payload: bytes,
    *,
    body_sha256: str,
    body_bytes: int,
    expected_url: str,
    captured_at: datetime,
    accepted_tls_versions: tuple[str, ...],
    expected_collector_id: str,
    expected_collector_implementation_sha256: str,
    expected_collector_execution_environment_sha256: str,
) -> ImmportSanitizedCaptureReceipt:
    receipt: ImmportSanitizedCaptureReceipt | None = None
    try:
        receipt = ImmportSanitizedCaptureReceipt.model_validate_json(payload)
    except ValueError:
        pass
    if receipt is None:
        raise ImmportProductionSourceError('invalid ImmPort collector receipt')
    if payload != canonical_json_bytes(receipt):
        raise ImmportProductionSourceError('ImmPort collector receipt is not canonical JSON')
    if (
        receipt.requested_url != expected_url
        or receipt.body_sha256 != body_sha256
        or receipt.body_byte_count != body_bytes
        or receipt.completed_at > _capture_upper_bound(captured_at)
        or receipt.collector_id != expected_collector_id
        or receipt.collector_implementation_sha256 != expected_collector_implementation_sha256
        or receipt.collector_execution_environment_sha256 != expected_collector_execution_environment_sha256
    ):
        raise ImmportProductionSourceError(
            'ImmPort receipt differs from its body, URL, capture, or committed collector'
        )
    peer = receipt.tls_peer
    if (
        peer.server_name != 'www.immport.org'
        or peer.certificate_der_sha256 is None
        or peer.tls_version not in accepted_tls_versions
    ):
        raise ImmportProductionSourceError('ImmPort capture lacks collector-reported official-origin TLS peer metadata')
    return receipt


def _capture_upper_bound(value: datetime) -> datetime:
    """Normalize a promotion capture timestamp for receipt comparisons."""

    return aware_utc(value, 'ImmPort capture timestamp')


def _strict_json_value(payload: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ImmportProductionSourceError(f'{label} contains duplicate JSON key {name!r}')
            result[name] = value
        return result

    def reject_constant(value: str) -> None:
        raise ImmportProductionSourceError(f'{label} contains non-finite JSON number {value}')

    if payload.startswith(b'\xef\xbb\xbf') or b'\x00' in payload:
        raise ImmportProductionSourceError(f'{label} contains a BOM or NUL byte')
    try:
        return json.loads(
            payload.decode('utf-8'),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImmportProductionSourceError(f'{label} is not strict UTF-8 JSON: {error}') from error


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_value(payload, label)
    if not isinstance(value, dict):
        raise ImmportProductionSourceError(f'{label} must contain one JSON object')
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ImmportProductionSourceError(f'ImmPort {label} must be an object')
    return value


def _validate_openapi_contract(
    root: dict[str, Any],
    expected_info_version: str,
) -> dict[str, frozenset[str]]:
    openapi_version = root.get('openapi')
    info = _mapping(root.get('info'), 'OpenAPI info')
    if not isinstance(openapi_version, str) or not openapi_version.startswith('3.'):
        raise ImmportProductionSourceError('ImmPort API contract is not OpenAPI 3')
    if info.get('version') != expected_info_version:
        raise ImmportProductionSourceError('ImmPort OpenAPI info.version differs from policy')
    paths = _mapping(root.get('paths'), 'OpenAPI paths')
    expected_paths = {
        '/api/study/{studyAccession}': ('StudyApi', True),
        '/api/study/arm/{studyAccession}': ('StudyArmApi', True),
        '/api/study/experiment/{studyAccession}': ('StudyExperimentApi', True),
        '/api/study/link/{studyAccession}': ('StudyLinkApi', True),
        '/api/study/manifest/{studyAccession}': ('FileDetails', True),
    }
    for path, (schema_name, requires_auth) in expected_paths.items():
        operation = _mapping(_mapping(paths.get(path), f'OpenAPI path {path}').get('get'), f'GET {path}')
        responses = _mapping(operation.get('responses'), f'GET {path} responses')
        response = _mapping(responses.get('200'), f'GET {path} 200 response')
        content = _mapping(response.get('content'), f'GET {path} response content')
        json_content = _mapping(content.get('application/json'), f'GET {path} JSON response')
        schema = _mapping(json_content.get('schema'), f'GET {path} response schema')
        items = _mapping(schema.get('items'), f'GET {path} response items')
        if schema.get('type') != 'array' or items.get('$ref') != f'#/components/schemas/{schema_name}':
            raise ImmportProductionSourceError(f'ImmPort OpenAPI response contract drifted for {path}')
        if requires_auth and not operation.get('security'):
            raise ImmportProductionSourceError(f'ImmPort OpenAPI no longer marks {path} authenticated')
    components = _mapping(root.get('components'), 'OpenAPI components')
    schemas = _mapping(components.get('schemas'), 'OpenAPI schemas')
    properties: dict[str, frozenset[str]] = {}
    for schema_name in ('StudyApi', 'StudyArmApi', 'StudyExperimentApi', 'StudyLinkApi', 'FileDetails'):
        schema = _mapping(schemas.get(schema_name), f'OpenAPI schema {schema_name}')
        raw_properties = _mapping(schema.get('properties'), f'OpenAPI schema {schema_name} properties')
        properties[schema_name] = frozenset(raw_properties)
    return properties


def _validate_schema_keys(raw: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ImmportProductionSourceError(f'ImmPort {label} contains fields absent from pinned OpenAPI: {unknown}')


def _required_accession(
    raw: dict[str, Any],
    field: str,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ImmportProductionSourceError(f'ImmPort {label} has an invalid {field}')
    return value


def _required_adapter_accession(
    raw: dict[str, Any],
    field: str,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ImmportProductionAdapterError(f'ImmPort {label} has an invalid {field}')
    return value


def _require_same_study(raw: dict[str, Any], study_accession: str, label: str) -> None:
    if raw.get('studyAccession') != study_accession:
        raise ImmportProductionSourceError(f'ImmPort {label} belongs to another study')


def _require_unique_field(rows: tuple[dict[str, Any], ...], field: str, label: str) -> None:
    values = tuple(row.get(field) for row in rows)
    if len(values) != len(set(values)):
        raise ImmportProductionSourceError(f'ImmPort {label} contains duplicate {field} values')


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ImmportProductionSourceError(f'ImmPort {label} must be an ISO date')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ImmportProductionSourceError(f'ImmPort {label} must be an ISO date') from error
    if parsed.isoformat() != value:
        raise ImmportProductionSourceError(f'ImmPort {label} must use canonical ISO date syntax')
    return parsed


def _conservative_release_upper_bound(value: date) -> datetime:
    eastern = ZoneInfo('America/New_York')
    return datetime.combine(value, time.max, tzinfo=eastern).astimezone(timezone.utc)


def _contains_exact_release_token(value: str, release_version: str) -> bool:
    """Match one release identifier, never a prefix of another release identifier."""

    boundary_pattern = re.compile(
        rf'(?<![A-Za-z0-9.]){re.escape(release_version)}(?![A-Za-z0-9.])',
        re.IGNORECASE,
    )
    return boundary_pattern.search(value) is not None


def _validate_manifest_entry(entry: dict[str, Any], study_accession: str) -> None:
    if entry.get('studyAccession') != study_accession or entry.get('fileType') != 'release_file':
        raise ImmportProductionSourceError('ImmPort manifest contains a foreign study or file type')
    generated_md5 = entry.get('generatedMD5')
    size = entry.get('filesizeBytes')
    file_uuid = entry.get('fileUUID')
    path_value = entry.get('path')
    filename = entry.get('fileName')
    if not isinstance(generated_md5, str) or re.fullmatch(_MD5_PATTERN, generated_md5) is None:
        raise ImmportProductionSourceError('ImmPort manifest entry lacks a generated MD5')
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ImmportProductionSourceError('ImmPort manifest entry has an invalid byte size')
    if not isinstance(file_uuid, str) or re.fullmatch(_UUID_PATTERN, file_uuid) is None:
        raise ImmportProductionSourceError('ImmPort manifest entry has an invalid DRS UUID')
    if entry.get('drsObjectCreated') != 'Y':
        raise ImmportProductionSourceError('ImmPort manifest entry lacks a created DRS object')
    if not isinstance(path_value, str) or not isinstance(filename, str) or not filename:
        raise ImmportProductionSourceError('ImmPort manifest entry lacks its file path/name')
    path = PurePosixPath(path_value)
    if (
        path.is_absolute()
        or '..' in path.parts
        or path.as_posix() != path_value
        or '\\' in path_value
        or path.name != filename
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in path_value)
    ):
        raise ImmportProductionSourceError('ImmPort manifest entry path is unsafe or inconsistent')


def _append_record(
    records: list[SourceRecordBinding],
    *,
    source_id: str,
    record_id: str,
    raw: dict[str, Any],
    artifact_sha256: str,
    locator: str,
) -> None:
    if any(item.source_record_id == record_id for item in records):
        raise ImmportProductionSourceError(f'duplicate ImmPort source record ID: {record_id}')
    records.append(
        SourceRecordBinding(
            source_id=source_id,
            source_record_id=record_id,
            source_record_sha256=hashlib.sha256(canonical_json_bytes(raw)).hexdigest(),
            source_artifact_sha256=artifact_sha256,
            source_locator=locator,
        )
    )


def _rebind_rows(source_input: AdapterSourceInput) -> dict[tuple[str, str], dict[str, Any]]:
    row_index_by_artifact: dict[str, dict[str, dict[str, Any]]] = {}
    for capture in source_input.captures:
        for artifact in capture.artifacts:
            if not artifact.role.startswith('body.'):
                continue
            digest = hashlib.sha256(artifact.payload).hexdigest()
            if digest != artifact.sha256:
                raise ImmportProductionAdapterError('ImmPort body differs from promoted bytes')
            if digest in row_index_by_artifact:
                continue
            try:
                root = _strict_json_value(artifact.payload, f'captured body {digest}')
            except ImmportProductionSourceError as error:
                raise ImmportProductionAdapterError(str(error)) from error
            rows = root if isinstance(root, list) else [root]
            if any(not isinstance(item, dict) for item in rows):
                raise ImmportProductionAdapterError('ImmPort captured body contains a non-object row')
            row_index: dict[str, dict[str, Any]] = {}
            for raw in rows:
                row_sha256 = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
                if row_sha256 in row_index:
                    raise ImmportProductionAdapterError('ImmPort captured body contains duplicate canonical rows')
                row_index[row_sha256] = raw
            row_index_by_artifact[digest] = row_index
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for source in source_input.verified_records:
        raw = row_index_by_artifact.get(source.source_artifact_sha256, {}).get(source.source_record_sha256)
        if raw is None:
            raise ImmportProductionAdapterError(
                f'ImmPort row {source.source_record_id!r} cannot be rebound to captured bytes'
            )
        resolved[(source.source_id, source.source_record_id)] = raw
    return resolved


def _extract_nct_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    name = raw.get('name')
    link_type = raw.get('type')
    value = raw.get('value')
    if (
        not isinstance(name, str)
        or name.casefold() != 'clinicaltrials.gov'
        or not isinstance(link_type, str)
        or link_type.casefold() != 'website'
        or not isinstance(value, str)
        or value != value.strip()
    ):
        return ()
    parsed = urlsplit(value)
    if (
        parsed.scheme != 'https'
        or parsed.netloc != 'clinicaltrials.gov'
        or parsed.hostname != 'clinicaltrials.gov'
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return ()
    match = _CTGOV_PATH_RE.fullmatch(parsed.path)
    return () if match is None else (match.group(1),)


def _candidate_id(policy: ImmportArmAdapterPolicy, arm_accession: str) -> str:
    seed = canonical_json_bytes(
        {
            'adapter_id': IMMPORT_ARM_ADAPTER_ID,
            'episode_id': policy.episode_id,
            'policy_id': policy.policy_id,
            'study_accession': policy.study_accession,
            'arm_accession': arm_accession,
        }
    )
    return f'cand-immport-{hashlib.sha256(seed).hexdigest()[:32]}'


def _arm_role(
    policy: ImmportArmAdapterPolicy,
    arm: dict[str, Any],
) -> Literal['intervention', 'control', 'unclassified']:
    """Classify only an exact publisher vocabulary value under the pinned partition."""

    value = arm.get(policy.arm_role_field)
    if value in policy.intervention_arm_types:
        return 'intervention'
    if value in policy.control_arm_types:
        return 'control'
    return 'unclassified'


def _render_arm_projection(
    arm: dict[str, Any],
    candidate_id: str,
    measurement_techniques: tuple[str, ...],
) -> str:
    name = _clean_public_text(arm.get('name'), 'arm name', max_chars=512)
    description = _clean_public_text(arm.get('description'), 'arm description', max_chars=4_000)
    reported = _optional_public_text(arm.get('typeReported'), 'reported arm type', max_chars=256)
    preferred = _optional_public_text(arm.get('typePreferred'), 'preferred arm type', max_chars=256)
    lines = [
        f'Candidate ID: {candidate_id}.',
        f'Arm/cohort name: {name}.',
        f'Arm/cohort description: {description}.',
    ]
    if reported is not None:
        lines.append(f'Investigator-reported arm type: {reported}.')
    if preferred is not None:
        lines.append(f'ImmPort preferred arm type: {preferred}.')
    if measurement_techniques:
        lines.append(
            f'Study-level measurement techniques (not asserted as arm-linked): {"; ".join(measurement_techniques)}.'
        )
    return '\n'.join(lines)


def _optional_public_text(value: Any, label: str, *, max_chars: int) -> str | None:
    if value is None or value == '':
        return None
    return _clean_public_text(value, label, max_chars=max_chars)


def _clean_public_text(value: Any, label: str, *, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise ImmportProductionAdapterError(f'ImmPort {label} must be text')
    cleaned = ' '.join(value.split())
    if not cleaned or len(cleaned) > max_chars:
        raise ImmportProductionAdapterError(f'ImmPort {label} is empty or exceeds its bound')
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in cleaned):
        raise ImmportProductionAdapterError(f'ImmPort {label} contains control characters')
    blinded = _PUBLIC_URL_RE.sub('[url redacted]', cleaned)
    blinded = _PUBLIC_DOI_RE.sub('[doi redacted]', blinded)
    blinded = _PUBLIC_SOURCE_IDENTIFIER_RE.sub('[source identifier redacted]', blinded)
    return blinded


def _normalized_reference(
    episode_id: str,
    record_id: str,
    record: StrictModel,
) -> NormalizedRecordReference:
    return NormalizedRecordReference(
        episode_id=episode_id,
        record_id=record_id,
        record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
    )


def _jsonl(records: tuple[StrictModel, ...], error_type: type[ValueError]) -> bytes:
    if not records:
        raise error_type('ImmPort record inventory cannot be empty')
    return b''.join(canonical_json_bytes(record) + b'\n' for record in records)


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ImmportProductionSourceError(f'{label} must be a lowercase SHA-256 digest')
