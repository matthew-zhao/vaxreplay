"""Outcome-blind identity probes and conservative model-memory diagnostics.

This module intentionally keeps four questions separate:

* whether the mounted workspace itself contains a protected identity or later fact;
* whether decision-time metadata is sufficient to reidentify a trial;
* whether one pinned model/harness recalls an exact identity and protected later facts; and
* whether a model makes a good forecast from legitimate cutoff evidence.

Only the first question can globally exclude a case under the fixed policy below.  Target-system
probe results are annotations and never change that system's score denominator.  A negative probe
means only ``no_signal``; it is not evidence that model weights are uncontaminated.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
import re
import unicodedata
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_schema import ObservationState, RegistryOutcomeClass
from vaxreplay.clinicaltrials.execution_task import (
    ExecutionPrivateGold,
    ExecutionTask,
    validate_execution_task_gold,
)

EXECUTION_PROBE_POLICY_SCHEMA_VERSION = 'vaxreplay.clinical-execution-contamination-policy.dev-v0.1'
EXECUTION_PROBE_CHALLENGE_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-challenge.dev-v0.1'
EXECUTION_PROBE_RESPONSE_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-response.dev-v0.1'
EXECUTION_PROBE_PROMPT_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-prompt.dev-v0.1'
EXECUTION_PROBE_GOLD_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-gold.dev-v0.1'
EXECUTION_PROBE_RECEIPT_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-receipt.dev-v0.1'
EXECUTION_PROBE_EVALUATION_SCHEMA_VERSION = 'vaxreplay.clinical-execution-probe-evaluation.dev-v0.1'
EXECUTION_CASE_RISK_SCHEMA_VERSION = 'vaxreplay.clinical-execution-case-contamination-risk.dev-v0.1'
EXECUTION_CASE_STRATA_MANIFEST_SCHEMA_VERSION = 'vaxreplay.clinical-execution-contamination-strata-manifest.dev-v0.1'
EXECUTION_SYSTEM_PROBE_MANIFEST_SCHEMA_VERSION = 'vaxreplay.clinical-execution-system-probe-manifest.dev-v0.1'
EXECUTION_PROBE_POLICY_ID = 'aact-identity-and-memory-diagnostics-v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_NCT_PATTERN = re.compile(r'NCT\d{8}', re.IGNORECASE)
_NCT_EXACT_PATTERN = re.compile(r'^NCT\d{8}$', re.IGNORECASE)
_PROBE_GOLD_HMAC_DOMAIN = b'vaxreplay.clinical-execution-probe-gold.dev-v0.1\x00'
_CHALLENGE_ID_HMAC_DOMAIN = b'vaxreplay.clinical-execution-probe-challenge-id.dev-v0.1\x00'
_FUTURE_FIELD_ORDER = (
    'registry_outcome_class',
    'enrollment_observation',
    'enrollment_ratio',
    'primary_completion_observation',
    'primary_completion_slippage_days',
)
_HIGH_SPECIFICITY_FUTURE_FIELDS = frozenset({'enrollment_ratio', 'primary_completion_slippage_days'})


class ExecutionContaminationControlError(ValueError):
    """A probe or contamination-strata record failed closed."""


class ExecutionProbeKind(str, enum.Enum):
    REIDENTIFICATION = 'reidentification'
    PARAMETRIC_RECALL = 'parametric_recall'


class ExecutionProbeSurfaceVariant(str, enum.Enum):
    RELEASED_TASK = 'released_task'
    IDENTITY_SCRUBBED = 'identity_scrubbed'
    MINIMAL_FINGERPRINT = 'minimal_fingerprint'


class ProbeClaimBasis(str, enum.Enum):
    RECOGNIZED_FROM_MEMORY = 'recognized_from_memory'
    INFERRED_FROM_VISIBLE_METADATA = 'inferred_from_visible_metadata'
    UNCERTAIN = 'uncertain'
    UNKNOWN = 'unknown'


class WorkspaceLeakageStatus(str, enum.Enum):
    NO_SIGNAL_UNDER_PINNED_AUDIT = 'no_signal_under_pinned_audit'
    LEAK_DETECTED = 'leak_detected'
    AUDIT_INCOMPLETE = 'audit_incomplete'


class ExecutionCaseRiskStratum(str, enum.Enum):
    WORKSPACE_LEAK_EXCLUDED = 'workspace_leak_excluded'
    WORKSPACE_AUDIT_INCOMPLETE = 'workspace_audit_incomplete'
    IDENTITY_FINGERPRINTABLE = 'identity_fingerprintable'
    NO_IDENTITY_SIGNAL = 'no_identity_signal'
    REIDENTIFICATION_UNMEASURED = 'reidentification_unmeasured'


class ExecutionSystemExposureStatus(str, enum.Enum):
    PROBE_POSITIVE_MEMORY_SIGNAL = 'probe_positive_memory_signal'
    IDENTITY_SIGNAL_ONLY = 'identity_signal_only'
    FUTURE_MATCH_WITHOUT_IDENTITY_AMBIGUOUS = 'future_match_without_identity_ambiguous'
    NO_SIGNAL = 'no_signal'
    UNKNOWN = 'unknown'


class ExecutionProbePolicy(StrictModel):
    """Fixed conservative interpretation policy; it does not promise decontamination."""

    schema_version: Literal['vaxreplay.clinical-execution-contamination-policy.dev-v0.1'] = (
        EXECUTION_PROBE_POLICY_SCHEMA_VERSION
    )
    policy_id: Literal['aact-identity-and-memory-diagnostics-v0.1'] = EXECUTION_PROBE_POLICY_ID
    minimum_independent_organizer_attackers: Literal[2] = 2
    minimum_future_fact_matches_for_memory_signal: Literal[2] = 2
    high_specificity_future_match_required: Literal[True] = True
    exact_identity_match_required: Literal[True] = True
    workspace_leak_or_incomplete_audit_globally_excluded: Literal[True] = True
    fingerprintable_cases_retained_in_historical_all: Literal[True] = True
    fingerprintable_cases_excluded_from_common_low_risk: Literal[True] = True
    case_strata_fixed_before_target_system_runs: Literal[True] = True
    target_system_probe_never_changes_score_denominator: Literal[True] = True
    probe_correctness_feedback_withheld: Literal[True] = True
    no_signal_is_not_proof_of_clean_weights: Literal[True] = True
    machine_unlearning_required: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False
    residual_model_weight_contamination_possible: Literal[True] = True
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False


EXECUTION_PROBE_POLICY = ExecutionProbePolicy()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def execution_probe_policy_sha256() -> str:
    return _sha256(EXECUTION_PROBE_POLICY)


def normalize_identity_claim(value: str) -> str:
    """Apply the pinned exact-match normalization used for private identity aliases."""

    if not isinstance(value, str):
        raise TypeError('identity claims must be strings')
    normalized = unicodedata.normalize('NFKC', value).casefold()
    normalized = ' '.join(normalized.split())
    if not normalized or '\x00' in normalized:
        raise ValueError('identity claims must contain non-NUL text')
    return normalized


class ExecutionProbePrivateGold(StrictModel):
    """Organizer-only identity and later facts committed by an HMAC in the public challenge."""

    schema_version: Literal['vaxreplay.clinical-execution-probe-gold.dev-v0.1'] = EXECUTION_PROBE_GOLD_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_private_nct_id: str = Field(pattern=r'^NCT\d{8}$')
    normalized_identity_aliases: tuple[str, ...] = ()
    registry_outcome_class: RegistryOutcomeClass
    enrollment_observation: ObservationState
    enrollment_ratio: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    primary_completion_observation: ObservationState
    primary_completion_slippage_days: int | None = None
    organizer_private: Literal[True] = True
    participant_visible: Literal[False] = False

    @field_validator('normalized_identity_aliases')
    @classmethod
    def validate_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError('private identity aliases must be unique and sorted')
        for alias in value:
            if alias != normalize_identity_claim(alias):
                raise ValueError('private identity aliases must use the pinned normalization')
            if _NCT_PATTERN.search(alias):
                raise ValueError('the registry ID belongs in organizer_private_nct_id, not aliases')
        return value

    @model_validator(mode='after')
    def validate_future_observability(self) -> Self:
        if (self.enrollment_ratio is not None) != (self.enrollment_observation == ObservationState.OBSERVED_ACTUAL):
            raise ValueError('probe enrollment ratio must follow the execution-gold observability rule')
        if (self.primary_completion_slippage_days is not None) != (
            self.primary_completion_observation == ObservationState.OBSERVED_ACTUAL
        ):
            raise ValueError('probe completion slippage must follow the execution-gold observability rule')
        return self


def make_execution_probe_private_gold(
    *,
    task: ExecutionTask,
    execution_gold: ExecutionPrivateGold,
    execution_gold_key: bytes,
    identity_aliases: Iterable[str] = (),
) -> ExecutionProbePrivateGold:
    """Derive probe gold only after authenticating the task's existing private gold."""

    validate_execution_task_gold(task, execution_gold, execution_gold_key)
    aliases = tuple(sorted({normalize_identity_claim(value) for value in identity_aliases}))
    return ExecutionProbePrivateGold(
        episode_id=execution_gold.episode_id,
        task_context_sha256=execution_gold.task_context_sha256,
        execution_private_gold_commitment_sha256=task.private_gold_commitment_sha256,
        organizer_private_nct_id=execution_gold.organizer_private_nct_id,
        normalized_identity_aliases=aliases,
        registry_outcome_class=execution_gold.registry_outcome_class,
        enrollment_observation=execution_gold.enrollment_observation,
        enrollment_ratio=execution_gold.enrollment_ratio,
        primary_completion_observation=execution_gold.primary_completion_observation,
        primary_completion_slippage_days=execution_gold.primary_completion_slippage_days,
    )


def execution_probe_private_gold_commitment(gold: ExecutionProbePrivateGold, key: bytes) -> str:
    if len(key) < 32:
        raise ValueError('probe private-gold HMAC key must contain at least 32 bytes')
    validated = ExecutionProbePrivateGold.model_validate_json(canonical_json_bytes(gold))
    return hmac.new(
        key,
        _PROBE_GOLD_HMAC_DOMAIN + canonical_json_bytes(validated),
        hashlib.sha256,
    ).hexdigest()


class ExecutionProbeChallenge(StrictModel):
    """Participant-visible outcome-blind probe bound to one exact model-facing surface."""

    schema_version: Literal['vaxreplay.clinical-execution-probe-challenge.dev-v0.1'] = (
        EXECUTION_PROBE_CHALLENGE_SCHEMA_VERSION
    )
    challenge_id: str = Field(pattern=r'^probe-[0-9a-f]{24}$')
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_private_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_kind: ExecutionProbeKind
    surface_variant: ExecutionProbeSurfaceVariant
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    instructions_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_probe_gold_commitment_scheme: Literal['hmac-sha256'] = 'hmac-sha256'
    private_probe_gold_commitment_sha256: str = Field(pattern=_SHA256_PATTERN)
    private_probe_gold_key_id: str = Field(pattern=_SHA256_PATTERN)
    outcome_blind_prompt: Literal[True] = True
    participant_sees_private_identity: Literal[False] = False
    participant_sees_private_gold: Literal[False] = False
    fresh_stateless_session_required: Literal[True] = True
    network_access_permitted: Literal[False] = False
    external_search_tools_permitted: Literal[False] = False
    probe_correctness_feedback_returned: Literal[False] = False
    isolation_enforcement_proven_here: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @model_validator(mode='after')
    def validate_public_contract(self) -> Self:
        if self.probe_policy_sha256 != execution_probe_policy_sha256():
            raise ValueError('probe challenge does not bind the fixed contamination policy')
        if (
            self.instructions_sha256
            != hashlib.sha256(execution_probe_instructions(self.probe_kind).encode('utf-8')).hexdigest()
        ):
            raise ValueError('probe challenge does not bind the fixed instructions')
        for value in (self.challenge_id, self.episode_id):
            if _NCT_PATTERN.search(value):
                raise ValueError('public probe identifiers cannot expose an NCT identifier')
        return self


def execution_probe_challenge_sha256(challenge: ExecutionProbeChallenge) -> str:
    validated = ExecutionProbeChallenge.model_validate_json(canonical_json_bytes(challenge))
    return _sha256(validated)


def execution_probe_instructions(kind: ExecutionProbeKind) -> str:
    shared = (
        'Use only the exact mounted task surface in a fresh no-network, no-search session. '
        'The organizer will return no correctness feedback. Abstain rather than inventing an identity. '
    )
    if kind == ExecutionProbeKind.REIDENTIFICATION:
        return shared + (
            'Attempt to identify the underlying registry study from the visible decision-time metadata. '
            'Return an exact registry identifier and/or study-name claims, but no later outcome claims.'
        )
    return shared + (
        'Report an identity and later registry facts only when you believe you recognize and remember them; '
        'do not turn the probe into a forecast from cutoff evidence. This self-report is diagnostic and is not '
        'trusted as proof of how the answer was obtained.'
    )


def execution_probe_prompt_bytes(challenge: ExecutionProbeChallenge) -> bytes:
    """Render the complete outcome-blind request without opening organizer-private gold.

    The exact task surface is mounted separately and is bound by ``public_surface_sha256``.  The
    response contract names possible fields but supplies no identity, later value, or correctness
    signal.
    """

    challenge = ExecutionProbeChallenge.model_validate_json(canonical_json_bytes(challenge))
    later_contract: dict[str, object] | None = None
    if challenge.probe_kind == ExecutionProbeKind.PARAMETRIC_RECALL:
        later_contract = {
            'registry_outcome_class': 'enum value or null',
            'enrollment_observation': 'enum value or null',
            'enrollment_ratio': 'number or null',
            'primary_completion_observation': 'enum value or null',
            'primary_completion_slippage_days': 'integer or null',
        }
    payload = {
        'schema_version': EXECUTION_PROBE_PROMPT_SCHEMA_VERSION,
        'challenge': challenge.model_dump(mode='json'),
        'instructions': execution_probe_instructions(challenge.probe_kind),
        'output_contract': {
            'challenge_id': challenge.challenge_id,
            'challenge_sha256': execution_probe_challenge_sha256(challenge),
            'probe_kind': challenge.probe_kind.value,
            'system_manifest_sha256': challenge.system_manifest_sha256,
            'session_isolation_receipt_sha256': 'runner-supplied SHA-256 binding',
            'abstained': 'boolean',
            'declared_basis': [item.value for item in ProbeClaimBasis],
            'registry_identifier_claim': 'string or null',
            'identity_name_claims': 'array of strings',
            'later_registry_claim': later_contract,
        },
        'correctness_feedback': None,
    }
    rendered = canonical_json_bytes(payload)
    if _NCT_PATTERN.search(rendered.decode('utf-8')):
        raise ExecutionContaminationControlError('public probe prompt unexpectedly contains an NCT identifier')
    return rendered


def make_execution_probe_challenge(
    *,
    task: ExecutionTask,
    probe_gold: ExecutionProbePrivateGold,
    probe_gold_key: bytes,
    public_surface: bytes,
    probe_kind: ExecutionProbeKind,
    surface_variant: ExecutionProbeSurfaceVariant,
    system_manifest_sha256: str,
) -> ExecutionProbeChallenge:
    """Create an opaque challenge without copying a private identity or outcome into public bytes."""

    task = ExecutionTask.model_validate_json(canonical_json_bytes(task))
    probe_gold = ExecutionProbePrivateGold.model_validate_json(canonical_json_bytes(probe_gold))
    if not isinstance(public_surface, bytes) or not public_surface:
        raise ExecutionContaminationControlError('probe public surface must be non-empty exact bytes')
    try:
        public_text = public_surface.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ExecutionContaminationControlError('probe public surface must be UTF-8') from error
    if _NCT_PATTERN.search(public_text):
        raise ExecutionContaminationControlError('probe public surface exposes an NCT identifier')
    if (probe_gold.episode_id, probe_gold.task_context_sha256) != (
        task.context.episode_id,
        task.context_sha256,
    ):
        raise ExecutionContaminationControlError('probe gold does not bind the exact public task context')
    if not hmac.compare_digest(
        probe_gold.execution_private_gold_commitment_sha256,
        task.private_gold_commitment_sha256,
    ):
        raise ExecutionContaminationControlError('probe gold does not bind the task private-gold commitment')
    if len(probe_gold_key) < 32:
        raise ExecutionContaminationControlError('probe private-gold HMAC key must contain at least 32 bytes')
    key_id = hashlib.sha256(probe_gold_key).hexdigest()
    public_surface_sha256 = hashlib.sha256(public_surface).hexdigest()
    instructions_sha256 = hashlib.sha256(execution_probe_instructions(probe_kind).encode('utf-8')).hexdigest()
    id_material = canonical_json_bytes(
        {
            'episode_id': task.context.episode_id,
            'task_context_sha256': task.context_sha256,
            'public_surface_sha256': public_surface_sha256,
            'probe_kind': probe_kind.value,
            'surface_variant': surface_variant.value,
            'system_manifest_sha256': system_manifest_sha256,
            'policy_sha256': execution_probe_policy_sha256(),
        }
    )
    challenge_id = (
        'probe-'
        + hmac.new(
            probe_gold_key,
            _CHALLENGE_ID_HMAC_DOMAIN + id_material,
            hashlib.sha256,
        ).hexdigest()[:24]
    )
    return ExecutionProbeChallenge(
        challenge_id=challenge_id,
        episode_id=task.context.episode_id,
        task_context_sha256=task.context_sha256,
        execution_private_gold_commitment_sha256=task.private_gold_commitment_sha256,
        public_surface_sha256=public_surface_sha256,
        probe_kind=probe_kind,
        surface_variant=surface_variant,
        system_manifest_sha256=system_manifest_sha256,
        probe_policy_sha256=execution_probe_policy_sha256(),
        instructions_sha256=instructions_sha256,
        private_probe_gold_commitment_sha256=execution_probe_private_gold_commitment(
            probe_gold,
            probe_gold_key,
        ),
        private_probe_gold_key_id=key_id,
    )


class LaterRegistryRecallClaim(StrictModel):
    registry_outcome_class: RegistryOutcomeClass | None = None
    enrollment_observation: ObservationState | None = None
    enrollment_ratio: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    primary_completion_observation: ObservationState | None = None
    primary_completion_slippage_days: int | None = None

    @model_validator(mode='after')
    def require_a_claim(self) -> Self:
        if all(getattr(self, field) is None for field in _FUTURE_FIELD_ORDER):
            raise ValueError('later registry recall requires at least one claimed field')
        return self


class ExecutionProbeResponse(StrictModel):
    """Untrusted model response. Identity and later-fact values are never echoed in public feedback."""

    schema_version: Literal['vaxreplay.clinical-execution-probe-response.dev-v0.1'] = (
        EXECUTION_PROBE_RESPONSE_SCHEMA_VERSION
    )
    challenge_id: str = Field(pattern=r'^probe-[0-9a-f]{24}$')
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    probe_kind: ExecutionProbeKind
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_isolation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    abstained: bool
    declared_basis: ProbeClaimBasis
    registry_identifier_claim: str | None = Field(default=None, min_length=1, max_length=128)
    identity_name_claims: tuple[str, ...] = Field(default=(), max_length=32)
    later_registry_claim: LaterRegistryRecallClaim | None = None

    @field_validator('registry_identifier_claim')
    @classmethod
    def validate_registry_claim(cls, value: str | None) -> str | None:
        if value is not None and (value != value.strip() or '\x00' in value):
            raise ValueError('registry identifier claims must be trimmed and cannot contain NUL')
        return value

    @field_validator('identity_name_claims')
    @classmethod
    def validate_name_claims(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(normalize_identity_claim(item) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError('identity name claims must be unique after pinned normalization')
        return value

    @model_validator(mode='after')
    def validate_abstention(self) -> Self:
        has_claim = bool(
            self.registry_identifier_claim or self.identity_name_claims or self.later_registry_claim is not None
        )
        if self.abstained == has_claim:
            raise ValueError('an abstention has no claims; a non-abstention requires at least one claim')
        if self.abstained and self.declared_basis != ProbeClaimBasis.UNKNOWN:
            raise ValueError('an abstention must use the unknown declared basis')
        return self


class ExecutionProbeAcceptanceReceipt(StrictModel):
    """Participant-safe syntax receipt containing hashes but no correctness signal."""

    schema_version: Literal['vaxreplay.clinical-execution-probe-receipt.dev-v0.1'] = (
        EXECUTION_PROBE_RECEIPT_SCHEMA_VERSION
    )
    challenge_id: str = Field(pattern=r'^probe-[0-9a-f]{24}$')
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_isolation_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    syntax_accepted: Literal[True] = True
    correctness_evaluated_in_this_receipt: Literal[False] = False
    correctness_feedback_returned: Literal[False] = False
    private_identity_returned: Literal[False] = False
    private_outcome_returned: Literal[False] = False


def accept_execution_probe_response(
    challenge: ExecutionProbeChallenge,
    response: ExecutionProbeResponse,
) -> ExecutionProbeAcceptanceReceipt:
    """Validate only public bindings; this step never opens private probe gold."""

    challenge = ExecutionProbeChallenge.model_validate_json(canonical_json_bytes(challenge))
    response = ExecutionProbeResponse.model_validate_json(canonical_json_bytes(response))
    challenge_sha256 = execution_probe_challenge_sha256(challenge)
    if (response.challenge_id, response.challenge_sha256) != (
        challenge.challenge_id,
        challenge_sha256,
    ):
        raise ExecutionContaminationControlError('probe response does not bind the exact challenge')
    if response.system_manifest_sha256 != challenge.system_manifest_sha256:
        raise ExecutionContaminationControlError('probe response is bound to a different system manifest')
    if response.probe_kind != challenge.probe_kind:
        raise ExecutionContaminationControlError('probe response kind does not match the challenge')
    if challenge.probe_kind == ExecutionProbeKind.REIDENTIFICATION and response.later_registry_claim is not None:
        raise ExecutionContaminationControlError('the outcome-blind reidentification probe cannot carry later facts')
    return ExecutionProbeAcceptanceReceipt(
        challenge_id=challenge.challenge_id,
        challenge_sha256=challenge_sha256,
        response_sha256=_sha256(response),
        system_manifest_sha256=response.system_manifest_sha256,
        session_isolation_receipt_sha256=response.session_isolation_receipt_sha256,
    )


class ExecutionPrivateProbeEvaluation(StrictModel):
    """Organizer-private diagnostic. It is not participant feedback or a score correction."""

    schema_version: Literal['vaxreplay.clinical-execution-probe-evaluation.dev-v0.1'] = (
        EXECUTION_PROBE_EVALUATION_SCHEMA_VERSION
    )
    challenge_id: str = Field(pattern=r'^probe-[0-9a-f]{24}$')
    challenge_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_isolation_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    probe_kind: ExecutionProbeKind
    surface_variant: ExecutionProbeSurfaceVariant
    exact_registry_identifier_match: bool
    exact_identity_alias_match_count: int = Field(ge=0)
    identity_recovered: bool
    matched_future_fields: tuple[str, ...]
    high_specificity_future_match: bool
    exposure_status: ExecutionSystemExposureStatus
    organizer_private: Literal[True] = True
    participant_visible: Literal[False] = False
    participant_correctness_feedback_returned: Literal[False] = False
    score_denominator_changed: Literal[False] = False
    no_signal_is_not_proof_of_clean_weights: Literal[True] = True
    proves_training_membership: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False
    residual_model_weight_contamination_possible: Literal[True] = True

    @model_validator(mode='after')
    def validate_evaluation(self) -> Self:
        if self.matched_future_fields != tuple(
            field for field in _FUTURE_FIELD_ORDER if field in set(self.matched_future_fields)
        ):
            raise ValueError('matched future fields must be unique and use fixed field order')
        expected_identity = self.exact_registry_identifier_match or self.exact_identity_alias_match_count > 0
        if self.identity_recovered != expected_identity:
            raise ValueError('identity_recovered is inconsistent with exact match signals')
        expected_high_specificity = bool(_HIGH_SPECIFICITY_FUTURE_FIELDS.intersection(self.matched_future_fields))
        if self.high_specificity_future_match != expected_high_specificity:
            raise ValueError('high-specificity future-match flag is inconsistent')
        if self.response_sha256 is None:
            if self.session_isolation_receipt_sha256 is not None:
                raise ValueError('a missing response cannot carry a session receipt')
            if self.identity_recovered or self.matched_future_fields:
                raise ValueError('a missing response cannot carry detected signals')
            expected_status = ExecutionSystemExposureStatus.UNKNOWN
        elif (
            self.probe_kind == ExecutionProbeKind.PARAMETRIC_RECALL
            and self.identity_recovered
            and len(self.matched_future_fields) >= EXECUTION_PROBE_POLICY.minimum_future_fact_matches_for_memory_signal
            and self.high_specificity_future_match
        ):
            expected_status = ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL
        elif self.identity_recovered:
            expected_status = ExecutionSystemExposureStatus.IDENTITY_SIGNAL_ONLY
        elif self.matched_future_fields:
            expected_status = ExecutionSystemExposureStatus.FUTURE_MATCH_WITHOUT_IDENTITY_AMBIGUOUS
        else:
            expected_status = ExecutionSystemExposureStatus.NO_SIGNAL
        if self.exposure_status != expected_status:
            raise ValueError('system exposure status does not follow the fixed interpretation policy')
        return self


def _validate_probe_gold_binding(
    challenge: ExecutionProbeChallenge,
    probe_gold: ExecutionProbePrivateGold,
    probe_gold_key: bytes,
) -> None:
    if len(probe_gold_key) < 32:
        raise ExecutionContaminationControlError('probe private-gold HMAC key must contain at least 32 bytes')
    if not hmac.compare_digest(
        hashlib.sha256(probe_gold_key).hexdigest(),
        challenge.private_probe_gold_key_id,
    ):
        raise ExecutionContaminationControlError('probe private-gold key does not match the challenge')
    if not hmac.compare_digest(
        execution_probe_private_gold_commitment(probe_gold, probe_gold_key),
        challenge.private_probe_gold_commitment_sha256,
    ):
        raise ExecutionContaminationControlError('probe private gold does not match the challenge HMAC')
    if (
        probe_gold.episode_id,
        probe_gold.task_context_sha256,
        probe_gold.execution_private_gold_commitment_sha256,
    ) != (
        challenge.episode_id,
        challenge.task_context_sha256,
        challenge.execution_private_gold_commitment_sha256,
    ):
        raise ExecutionContaminationControlError('probe private gold does not bind the challenge task')


def _future_matches(
    claim: LaterRegistryRecallClaim | None,
    gold: ExecutionProbePrivateGold,
) -> tuple[str, ...]:
    if claim is None:
        return ()
    matches: list[str] = []
    for field in _FUTURE_FIELD_ORDER:
        claimed = getattr(claim, field)
        expected = getattr(gold, field)
        if claimed is None:
            continue
        if isinstance(claimed, float):
            equal = expected is not None and math.isclose(claimed, expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = claimed == expected
        if equal:
            matches.append(field)
    return tuple(matches)


def evaluate_execution_probe_response(
    *,
    challenge: ExecutionProbeChallenge,
    probe_gold: ExecutionProbePrivateGold,
    probe_gold_key: bytes,
    response: ExecutionProbeResponse | None,
) -> ExecutionPrivateProbeEvaluation:
    """Open private gold only after public syntax acceptance and emit organizer-only flags."""

    challenge = ExecutionProbeChallenge.model_validate_json(canonical_json_bytes(challenge))
    probe_gold = ExecutionProbePrivateGold.model_validate_json(canonical_json_bytes(probe_gold))
    _validate_probe_gold_binding(challenge, probe_gold, probe_gold_key)
    if response is None:
        return ExecutionPrivateProbeEvaluation(
            challenge_id=challenge.challenge_id,
            challenge_sha256=execution_probe_challenge_sha256(challenge),
            response_sha256=None,
            episode_id=challenge.episode_id,
            task_context_sha256=challenge.task_context_sha256,
            public_surface_sha256=challenge.public_surface_sha256,
            system_manifest_sha256=challenge.system_manifest_sha256,
            session_isolation_receipt_sha256=None,
            probe_kind=challenge.probe_kind,
            surface_variant=challenge.surface_variant,
            exact_registry_identifier_match=False,
            exact_identity_alias_match_count=0,
            identity_recovered=False,
            matched_future_fields=(),
            high_specificity_future_match=False,
            exposure_status=ExecutionSystemExposureStatus.UNKNOWN,
        )

    receipt = accept_execution_probe_response(challenge, response)
    exact_registry = bool(
        response.registry_identifier_claim
        and _NCT_EXACT_PATTERN.fullmatch(response.registry_identifier_claim)
        and hmac.compare_digest(
            response.registry_identifier_claim.upper(),
            probe_gold.organizer_private_nct_id,
        )
    )
    normalized_claims = {normalize_identity_claim(value) for value in response.identity_name_claims}
    alias_matches = len(normalized_claims.intersection(probe_gold.normalized_identity_aliases))
    identity_recovered = exact_registry or alias_matches > 0
    matches = _future_matches(response.later_registry_claim, probe_gold)
    high_specificity = bool(_HIGH_SPECIFICITY_FUTURE_FIELDS.intersection(matches))
    if (
        challenge.probe_kind == ExecutionProbeKind.PARAMETRIC_RECALL
        and identity_recovered
        and len(matches) >= EXECUTION_PROBE_POLICY.minimum_future_fact_matches_for_memory_signal
        and high_specificity
    ):
        status = ExecutionSystemExposureStatus.PROBE_POSITIVE_MEMORY_SIGNAL
    elif identity_recovered:
        status = ExecutionSystemExposureStatus.IDENTITY_SIGNAL_ONLY
    elif matches:
        status = ExecutionSystemExposureStatus.FUTURE_MATCH_WITHOUT_IDENTITY_AMBIGUOUS
    else:
        status = ExecutionSystemExposureStatus.NO_SIGNAL
    return ExecutionPrivateProbeEvaluation(
        challenge_id=challenge.challenge_id,
        challenge_sha256=receipt.challenge_sha256,
        response_sha256=receipt.response_sha256,
        episode_id=challenge.episode_id,
        task_context_sha256=challenge.task_context_sha256,
        public_surface_sha256=challenge.public_surface_sha256,
        system_manifest_sha256=challenge.system_manifest_sha256,
        session_isolation_receipt_sha256=receipt.session_isolation_receipt_sha256,
        probe_kind=challenge.probe_kind,
        surface_variant=challenge.surface_variant,
        exact_registry_identifier_match=exact_registry,
        exact_identity_alias_match_count=alias_matches,
        identity_recovered=identity_recovered,
        matched_future_fields=matches,
        high_specificity_future_match=high_specificity,
        exposure_status=status,
    )


class ExecutionCaseSurfaceBinding(StrictModel):
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_surface_sha256: str = Field(pattern=_SHA256_PATTERN)


class ExecutionCaseContaminationEvidence(StrictModel):
    """Organizer evidence used to freeze a case stratum before target-system runs."""

    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_leakage_status: WorkspaceLeakageStatus
    workspace_audit_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    organizer_probe_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    precommitted_organizer_attacker_system_manifest_sha256s: tuple[str, ...] = Field(min_length=2)
    organizer_reidentification_evaluations: tuple[ExecutionPrivateProbeEvaluation, ...] = ()
    target_system_results_seen: Literal[False] = False

    @model_validator(mode='after')
    def validate_evidence(self) -> Self:
        if self.workspace_leakage_status == WorkspaceLeakageStatus.AUDIT_INCOMPLETE:
            if self.workspace_audit_receipt_sha256 is not None:
                raise ValueError('an incomplete workspace audit cannot carry a completed receipt')
        elif self.workspace_audit_receipt_sha256 is None:
            raise ValueError('a completed workspace audit status requires its receipt hash')
        expected_systems = self.precommitted_organizer_attacker_system_manifest_sha256s
        if expected_systems != tuple(sorted(set(expected_systems))) or any(
            re.fullmatch(_SHA256_PATTERN, value) is None for value in expected_systems
        ):
            raise ValueError('precommitted organizer attackers must use unique sorted system hashes')
        systems: list[str] = []
        for evaluation in self.organizer_reidentification_evaluations:
            if evaluation.probe_kind != ExecutionProbeKind.REIDENTIFICATION:
                raise ValueError('case fingerprinting accepts only outcome-blind reidentification probes')
            if evaluation.exposure_status == ExecutionSystemExposureStatus.UNKNOWN:
                raise ValueError('missing or invalid organizer probes cannot count as completed attackers')
            if (
                evaluation.episode_id,
                evaluation.task_context_sha256,
                evaluation.public_surface_sha256,
            ) != (self.episode_id, self.task_context_sha256, self.public_surface_sha256):
                raise ValueError('organizer reidentification evaluation binds a different case surface')
            systems.append(evaluation.system_manifest_sha256)
        if len(systems) != len(set(systems)):
            raise ValueError('organizer reidentification attackers must use distinct system manifests')
        if tuple(systems) != tuple(sorted(systems)):
            raise ValueError('organizer reidentification evaluations must use ascending system hashes')
        if not set(systems).issubset(expected_systems):
            raise ValueError('an organizer reidentification result was not precommitted in the probe batch')
        return self


class ExecutionCaseContaminationRisk(StrictModel):
    schema_version: Literal['vaxreplay.clinical-execution-case-contamination-risk.dev-v0.1'] = (
        EXECUTION_CASE_RISK_SCHEMA_VERSION
    )
    episode_id: str = Field(min_length=1)
    task_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_surface_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_leakage_status: WorkspaceLeakageStatus
    workspace_audit_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    organizer_probe_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    precommitted_organizer_attacker_system_manifest_sha256s: tuple[str, ...] = Field(min_length=2)
    completed_organizer_attacker_system_manifest_sha256s: tuple[str, ...]
    independent_organizer_attacker_count: int = Field(ge=0)
    identity_recovery_count: int = Field(ge=0)
    stratum: ExecutionCaseRiskStratum
    included_in_historical_all: bool
    included_in_historical_common_low_risk: bool
    exclusion_reason: str | None = None
    target_system_results_used_for_case_selection: Literal[False] = False
    fixed_before_target_system_runs: Literal[True] = True
    proves_identity_unrecoverable: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False
    residual_reidentification_risk: Literal[True] = True

    @model_validator(mode='after')
    def validate_fixed_policy(self) -> Self:
        expected_systems = self.precommitted_organizer_attacker_system_manifest_sha256s
        completed_systems = self.completed_organizer_attacker_system_manifest_sha256s
        if expected_systems != tuple(sorted(set(expected_systems))) or any(
            re.fullmatch(_SHA256_PATTERN, value) is None for value in expected_systems
        ):
            raise ValueError('precommitted organizer attackers must use unique sorted system hashes')
        if completed_systems != tuple(sorted(set(completed_systems))) or not set(completed_systems).issubset(
            expected_systems
        ):
            raise ValueError('completed organizer attackers must be a unique sorted precommitted subset')
        if self.independent_organizer_attacker_count != len(completed_systems):
            raise ValueError('organizer attacker count does not match completed system hashes')
        if self.identity_recovery_count > self.independent_organizer_attacker_count:
            raise ValueError('identity recoveries cannot exceed organizer attacker count')
        expected = _case_disposition(
            self.workspace_leakage_status,
            self.independent_organizer_attacker_count,
            self.identity_recovery_count,
            organizer_probe_batch_complete=completed_systems == expected_systems,
        )
        if (
            self.stratum,
            self.included_in_historical_all,
            self.included_in_historical_common_low_risk,
            self.exclusion_reason,
        ) != expected:
            raise ValueError('case risk does not follow the fixed exclusion and strata policy')
        return self


def _case_disposition(
    workspace_status: WorkspaceLeakageStatus,
    attacker_count: int,
    recovery_count: int,
    *,
    organizer_probe_batch_complete: bool,
) -> tuple[ExecutionCaseRiskStratum, bool, bool, str | None]:
    if workspace_status == WorkspaceLeakageStatus.LEAK_DETECTED:
        return (
            ExecutionCaseRiskStratum.WORKSPACE_LEAK_EXCLUDED,
            False,
            False,
            'workspace_leak_detected',
        )
    if workspace_status == WorkspaceLeakageStatus.AUDIT_INCOMPLETE:
        return (
            ExecutionCaseRiskStratum.WORKSPACE_AUDIT_INCOMPLETE,
            False,
            False,
            'workspace_audit_incomplete',
        )
    if (
        attacker_count < EXECUTION_PROBE_POLICY.minimum_independent_organizer_attackers
        or not organizer_probe_batch_complete
    ):
        return ExecutionCaseRiskStratum.REIDENTIFICATION_UNMEASURED, True, False, None
    if recovery_count:
        return ExecutionCaseRiskStratum.IDENTITY_FINGERPRINTABLE, True, False, None
    return ExecutionCaseRiskStratum.NO_IDENTITY_SIGNAL, True, True, None


def assess_execution_case_contamination(
    evidence: ExecutionCaseContaminationEvidence,
) -> ExecutionCaseContaminationRisk:
    evidence = ExecutionCaseContaminationEvidence.model_validate_json(canonical_json_bytes(evidence))
    attacker_count = len(evidence.organizer_reidentification_evaluations)
    recovery_count = sum(item.identity_recovered for item in evidence.organizer_reidentification_evaluations)
    completed_systems = tuple(item.system_manifest_sha256 for item in evidence.organizer_reidentification_evaluations)
    stratum, historical_all, common_low_risk, exclusion = _case_disposition(
        evidence.workspace_leakage_status,
        attacker_count,
        recovery_count,
        organizer_probe_batch_complete=(
            completed_systems == evidence.precommitted_organizer_attacker_system_manifest_sha256s
        ),
    )
    return ExecutionCaseContaminationRisk(
        episode_id=evidence.episode_id,
        task_context_sha256=evidence.task_context_sha256,
        public_surface_sha256=evidence.public_surface_sha256,
        workspace_leakage_status=evidence.workspace_leakage_status,
        workspace_audit_receipt_sha256=evidence.workspace_audit_receipt_sha256,
        organizer_probe_batch_sha256=evidence.organizer_probe_batch_sha256,
        precommitted_organizer_attacker_system_manifest_sha256s=(
            evidence.precommitted_organizer_attacker_system_manifest_sha256s
        ),
        completed_organizer_attacker_system_manifest_sha256s=completed_systems,
        independent_organizer_attacker_count=attacker_count,
        identity_recovery_count=recovery_count,
        stratum=stratum,
        included_in_historical_all=historical_all,
        included_in_historical_common_low_risk=common_low_risk,
        exclusion_reason=exclusion,
    )


class ExecutionCaseStratumCount(StrictModel):
    stratum: ExecutionCaseRiskStratum
    case_count: int = Field(ge=0)


class ExecutionContaminationStrataManifest(StrictModel):
    """Complete, globally fixed case strata. This manifest contains no target-system probes."""

    schema_version: Literal['vaxreplay.clinical-execution-contamination-strata-manifest.dev-v0.1'] = (
        EXECUTION_CASE_STRATA_MANIFEST_SCHEMA_VERSION
    )
    manifest_id: str = Field(pattern=r'^[a-z0-9][a-z0-9._-]*$')
    probe_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_universe_sha256: str = Field(pattern=_SHA256_PATTERN)
    organizer_probe_batch_sha256: str = Field(pattern=_SHA256_PATTERN)
    precommitted_organizer_attacker_system_manifest_sha256s: tuple[str, ...] = Field(min_length=2)
    cases: tuple[ExecutionCaseContaminationRisk, ...] = Field(min_length=1)
    case_count: int = Field(gt=0)
    historical_all_count: int = Field(ge=0)
    historical_common_low_risk_count: int = Field(ge=0)
    stratum_counts: tuple[ExecutionCaseStratumCount, ...]
    complete_case_universe_covered: Literal[True] = True
    fixed_before_target_system_runs: Literal[True] = True
    target_system_results_used_for_case_selection: Literal[False] = False
    target_specific_denominators_prohibited: Literal[True] = True
    model_weight_contamination_eliminated: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False
    development_only: Literal[True] = True
    leaderboard_admitted: Literal[False] = False

    @field_validator('cases')
    @classmethod
    def validate_cases(
        cls,
        value: tuple[ExecutionCaseContaminationRisk, ...],
    ) -> tuple[ExecutionCaseContaminationRisk, ...]:
        episode_ids = tuple(item.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('contamination case risks must use unique ascending episode IDs')
        return value

    @model_validator(mode='after')
    def validate_manifest(self) -> Self:
        if self.probe_policy_sha256 != execution_probe_policy_sha256():
            raise ValueError('contamination strata manifest does not bind the fixed policy')
        expected_systems = self.precommitted_organizer_attacker_system_manifest_sha256s
        if expected_systems != tuple(sorted(set(expected_systems))) or any(
            re.fullmatch(_SHA256_PATTERN, value) is None for value in expected_systems
        ):
            raise ValueError('strata manifest organizer attackers must use unique sorted system hashes')
        if any(
            item.organizer_probe_batch_sha256 != self.organizer_probe_batch_sha256
            or item.precommitted_organizer_attacker_system_manifest_sha256s != expected_systems
            for item in self.cases
        ):
            raise ValueError('every case must use the same precommitted organizer probe batch and attackers')
        if self.case_count != len(self.cases):
            raise ValueError('contamination strata case_count is inconsistent')
        if self.historical_all_count != sum(item.included_in_historical_all for item in self.cases):
            raise ValueError('historical_all_count is inconsistent')
        if self.historical_common_low_risk_count != sum(
            item.included_in_historical_common_low_risk for item in self.cases
        ):
            raise ValueError('historical_common_low_risk_count is inconsistent')
        expected_counts = tuple(
            ExecutionCaseStratumCount(
                stratum=stratum,
                case_count=sum(item.stratum == stratum for item in self.cases),
            )
            for stratum in ExecutionCaseRiskStratum
        )
        if self.stratum_counts != expected_counts:
            raise ValueError('contamination stratum counts are inconsistent')
        return self


def execution_case_universe_sha256(bindings: Iterable[ExecutionCaseSurfaceBinding]) -> str:
    validated = tuple(
        sorted(
            (ExecutionCaseSurfaceBinding.model_validate_json(canonical_json_bytes(binding)) for binding in bindings),
            key=lambda item: item.episode_id,
        )
    )
    episode_ids = tuple(item.episode_id for item in validated)
    if not validated or episode_ids != tuple(sorted(set(episode_ids))):
        raise ExecutionContaminationControlError('case universe must contain unique episode IDs')
    return _sha256([item.model_dump(mode='json') for item in validated])


def build_execution_contamination_strata_manifest(
    *,
    manifest_id: str,
    case_universe: Iterable[ExecutionCaseSurfaceBinding],
    case_risks: Iterable[ExecutionCaseContaminationRisk],
) -> ExecutionContaminationStrataManifest:
    universe = tuple(
        sorted(
            (
                ExecutionCaseSurfaceBinding.model_validate_json(canonical_json_bytes(binding))
                for binding in case_universe
            ),
            key=lambda item: item.episode_id,
        )
    )
    risks = tuple(
        sorted(
            (ExecutionCaseContaminationRisk.model_validate_json(canonical_json_bytes(risk)) for risk in case_risks),
            key=lambda item: item.episode_id,
        )
    )
    expected = tuple((item.episode_id, item.task_context_sha256, item.public_surface_sha256) for item in universe)
    observed = tuple((item.episode_id, item.task_context_sha256, item.public_surface_sha256) for item in risks)
    if expected != observed:
        raise ExecutionContaminationControlError('case risk records do not exactly cover the bound case universe')
    if not risks:
        raise ExecutionContaminationControlError('contamination strata require at least one case risk')
    probe_batch_sha256 = risks[0].organizer_probe_batch_sha256
    attacker_systems = risks[0].precommitted_organizer_attacker_system_manifest_sha256s
    if any(
        item.organizer_probe_batch_sha256 != probe_batch_sha256
        or item.precommitted_organizer_attacker_system_manifest_sha256s != attacker_systems
        for item in risks
    ):
        raise ExecutionContaminationControlError(
            'every case must use one globally precommitted organizer probe batch and attacker set'
        )
    counts = tuple(
        ExecutionCaseStratumCount(
            stratum=stratum,
            case_count=sum(item.stratum == stratum for item in risks),
        )
        for stratum in ExecutionCaseRiskStratum
    )
    return ExecutionContaminationStrataManifest(
        manifest_id=manifest_id,
        probe_policy_sha256=execution_probe_policy_sha256(),
        case_universe_sha256=execution_case_universe_sha256(universe),
        organizer_probe_batch_sha256=probe_batch_sha256,
        precommitted_organizer_attacker_system_manifest_sha256s=attacker_systems,
        cases=risks,
        case_count=len(risks),
        historical_all_count=sum(item.included_in_historical_all for item in risks),
        historical_common_low_risk_count=sum(item.included_in_historical_common_low_risk for item in risks),
        stratum_counts=counts,
    )


class ExecutionSystemProbeStatusCount(StrictModel):
    status: ExecutionSystemExposureStatus
    case_count: int = Field(ge=0)


class ExecutionSystemProbeManifest(StrictModel):
    """One target system's private diagnostics over the fixed historical-all denominator."""

    schema_version: Literal['vaxreplay.clinical-execution-system-probe-manifest.dev-v0.1'] = (
        EXECUTION_SYSTEM_PROBE_MANIFEST_SCHEMA_VERSION
    )
    system_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_strata_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluations: tuple[ExecutionPrivateProbeEvaluation, ...]
    evaluated_case_count: int = Field(ge=0)
    status_counts: tuple[ExecutionSystemProbeStatusCount, ...]
    score_denominator_changed: Literal[False] = False
    score_corrected_using_probe: Literal[False] = False
    probe_reported_beside_score: Literal[True] = True
    no_signal_is_not_proof_of_clean_weights: Literal[True] = True
    proves_training_membership: Literal[False] = False
    proves_absence_of_contamination: Literal[False] = False
    residual_model_weight_contamination_possible: Literal[True] = True
    organizer_private: Literal[True] = True

    @field_validator('evaluations')
    @classmethod
    def validate_evaluations(
        cls,
        value: tuple[ExecutionPrivateProbeEvaluation, ...],
    ) -> tuple[ExecutionPrivateProbeEvaluation, ...]:
        episode_ids = tuple(item.episode_id for item in value)
        if episode_ids != tuple(sorted(set(episode_ids))):
            raise ValueError('system probe evaluations must use unique ascending episode IDs')
        if any(item.probe_kind != ExecutionProbeKind.PARAMETRIC_RECALL for item in value):
            raise ValueError('system probe manifest accepts one parametric-recall probe per case')
        return value

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        if self.evaluated_case_count != len(self.evaluations):
            raise ValueError('system probe evaluated_case_count is inconsistent')
        if any(item.system_manifest_sha256 != self.system_manifest_sha256 for item in self.evaluations):
            raise ValueError('system probe evaluations bind a different system')
        expected_counts = tuple(
            ExecutionSystemProbeStatusCount(
                status=status,
                case_count=sum(item.exposure_status == status for item in self.evaluations),
            )
            for status in ExecutionSystemExposureStatus
        )
        if self.status_counts != expected_counts:
            raise ValueError('system probe status counts are inconsistent')
        return self


def execution_contamination_strata_manifest_sha256(
    manifest: ExecutionContaminationStrataManifest,
) -> str:
    validated = ExecutionContaminationStrataManifest.model_validate_json(canonical_json_bytes(manifest))
    return _sha256(validated)


def build_execution_system_probe_manifest(
    *,
    case_strata_manifest: ExecutionContaminationStrataManifest,
    system_manifest_sha256: str,
    evaluations: Iterable[ExecutionPrivateProbeEvaluation],
) -> ExecutionSystemProbeManifest:
    strata = ExecutionContaminationStrataManifest.model_validate_json(canonical_json_bytes(case_strata_manifest))
    ordered = tuple(
        sorted(
            (ExecutionPrivateProbeEvaluation.model_validate_json(canonical_json_bytes(item)) for item in evaluations),
            key=lambda item: item.episode_id,
        )
    )
    expected_episodes = tuple(item.episode_id for item in strata.cases if item.included_in_historical_all)
    if tuple(item.episode_id for item in ordered) != expected_episodes:
        raise ExecutionContaminationControlError(
            'system probes must exactly cover the fixed historical-all case denominator'
        )
    fixed_surfaces = {
        item.episode_id: (item.task_context_sha256, item.public_surface_sha256)
        for item in strata.cases
        if item.included_in_historical_all
    }
    if any(
        (item.task_context_sha256, item.public_surface_sha256) != fixed_surfaces[item.episode_id] for item in ordered
    ):
        raise ExecutionContaminationControlError(
            'system probes must bind the exact task context and surface in the fixed denominator'
        )
    counts = tuple(
        ExecutionSystemProbeStatusCount(
            status=status,
            case_count=sum(item.exposure_status == status for item in ordered),
        )
        for status in ExecutionSystemExposureStatus
    )
    return ExecutionSystemProbeManifest(
        system_manifest_sha256=system_manifest_sha256,
        case_strata_manifest_sha256=execution_contamination_strata_manifest_sha256(strata),
        evaluations=ordered,
        evaluated_case_count=len(ordered),
        status_counts=counts,
    )


__all__ = [
    'EXECUTION_PROBE_POLICY',
    'EXECUTION_PROBE_POLICY_ID',
    'ExecutionCaseContaminationEvidence',
    'ExecutionCaseContaminationRisk',
    'ExecutionCaseRiskStratum',
    'ExecutionCaseStratumCount',
    'ExecutionCaseSurfaceBinding',
    'ExecutionContaminationControlError',
    'ExecutionContaminationStrataManifest',
    'ExecutionPrivateProbeEvaluation',
    'ExecutionProbeAcceptanceReceipt',
    'ExecutionProbeChallenge',
    'ExecutionProbeKind',
    'ExecutionProbePolicy',
    'ExecutionProbePrivateGold',
    'ExecutionProbeResponse',
    'ExecutionProbeSurfaceVariant',
    'ExecutionSystemExposureStatus',
    'ExecutionSystemProbeManifest',
    'ExecutionSystemProbeStatusCount',
    'LaterRegistryRecallClaim',
    'ProbeClaimBasis',
    'WorkspaceLeakageStatus',
    'accept_execution_probe_response',
    'assess_execution_case_contamination',
    'build_execution_contamination_strata_manifest',
    'build_execution_system_probe_manifest',
    'evaluate_execution_probe_response',
    'execution_case_universe_sha256',
    'execution_contamination_strata_manifest_sha256',
    'execution_probe_challenge_sha256',
    'execution_probe_instructions',
    'execution_probe_policy_sha256',
    'execution_probe_prompt_bytes',
    'execution_probe_private_gold_commitment',
    'make_execution_probe_challenge',
    'make_execution_probe_private_gold',
    'normalize_identity_claim',
]
