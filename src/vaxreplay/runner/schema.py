"""Strict contracts for public challenges, executable systems, and run receipts."""

from __future__ import annotations

import enum
import re
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.aggregation import SuiteEpisodeBinding
from vaxreplay.case_schema import StrictModel
from vaxreplay.prompt import PromptVariant

CHALLENGE_ENVELOPE_SCHEMA_VERSION = 'vaxreplay.challenge-envelope.v0.1'
CHALLENGE_BUNDLE_SCHEMA_VERSION = 'vaxreplay.challenge-bundle.v0.2'
RUNNER_POLICY_SCHEMA_VERSION = 'vaxreplay.runner-policy.v0.1'
SYSTEM_SUBMISSION_SCHEMA_VERSION = 'vaxreplay.system-submission.v0.1'
RUN_RECEIPT_SCHEMA_VERSION = 'vaxreplay.run-receipt.v0.2'
RESPONSE_PROTOCOL = 'vaxreplay.submission-json-stdout.v0.1'
RECEIPT_AUTHENTICATION = 'hmac-sha256'

_IMMUTABLE_IMAGE_PATTERN = re.compile(r'^(?:sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64})$')


class IsolationTier(str, enum.Enum):
    """Strength of the host boundary, not a claim about knowledge in model weights."""

    DEVELOPMENT = 'development'
    OFFICIAL = 'official'


class EpisodeRunStatus(str, enum.Enum):
    ACCEPTED = 'accepted'
    NONZERO_EXIT = 'nonzero_exit'
    TIMED_OUT = 'timed_out'
    RESPONSE_LIMIT = 'response_limit'
    LOG_LIMIT = 'log_limit'
    BACKEND_ERROR = 'backend_error'
    INVALID_UTF8 = 'invalid_utf8'
    INVALID_JSON = 'invalid_json'
    INVALID_SUBMISSION = 'invalid_submission'


class ChatMessage(StrictModel):
    role: Literal['system', 'user']
    content: str = Field(min_length=1)


class ChallengeEnvelope(StrictModel):
    """The complete public input delivered to one fresh worker over stdin."""

    schema_version: Literal['vaxreplay.challenge-envelope.v0.1'] = CHALLENGE_ENVELOPE_SCHEMA_VERSION
    challenge_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    ordinal: int = Field(ge=0)
    sample_index: int = Field(default=0, ge=0)
    prompt_variant: PromptVariant = PromptVariant.FULL
    binding: SuiteEpisodeBinding
    messages: tuple[ChatMessage, ChatMessage]
    response_protocol: Literal['vaxreplay.submission-json-stdout.v0.1'] = RESPONSE_PROTOCOL

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, value: tuple[ChatMessage, ChatMessage]) -> tuple[ChatMessage, ChatMessage]:
        if tuple(message.role for message in value) != ('system', 'user'):
            raise ValueError('challenge messages must contain exactly one system message followed by one user message')
        return value


class ChallengeEnvelopeFile(StrictModel):
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    path: str = Field(pattern=r'^episodes/[0-9]{6}\.json$')
    envelope_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ChallengeBundleManifest(StrictModel):
    """Canonical allowlist for every file in a public challenge bundle."""

    schema_version: Literal['vaxreplay.challenge-bundle.v0.2'] = CHALLENGE_BUNDLE_SCHEMA_VERSION
    challenge_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    suite_path: Literal['suite.json'] = 'suite.json'
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    prompt_variant: PromptVariant = PromptVariant.FULL
    admission_path: Literal['admission.json'] | None = None
    admission_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    envelopes: tuple[ChallengeEnvelopeFile, ...] = Field(min_length=1, max_length=4_096)

    @model_validator(mode='after')
    def validate_admission_binding(self) -> Self:
        if (self.admission_path is None) != (self.admission_sha256 is None):
            raise ValueError('challenge admission path and hash must be declared together')
        return self

    @field_validator('envelopes')
    @classmethod
    def validate_envelopes(cls, value: tuple[ChallengeEnvelopeFile, ...]) -> tuple[ChallengeEnvelopeFile, ...]:
        ordinals = tuple(binding.ordinal for binding in value)
        if ordinals != tuple(range(len(value))):
            raise ValueError('challenge envelope ordinals must be contiguous and start at zero')
        episode_ids = tuple(binding.episode_id for binding in value)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('challenge envelope episode IDs must be unique')
        paths = tuple(binding.path for binding in value)
        if len(paths) != len(set(paths)):
            raise ValueError('challenge envelope paths must be unique')
        return value


class RunLimits(StrictModel):
    wall_seconds: int = Field(default=600, ge=1, le=86_400)
    max_response_bytes: int = Field(default=1_048_576, ge=1_024, le=67_108_864)
    max_log_bytes: int = Field(default=1_048_576, ge=1_024, le=67_108_864)
    memory_mib: int = Field(default=8_192, ge=128, le=1_048_576)
    cpus: float = Field(default=4.0, gt=0.0, le=1024.0, allow_inf_nan=False)
    pids: int = Field(default=256, ge=16, le=1_048_576)
    scratch_mib: int = Field(default=1_024, ge=16, le=1_048_576)
    shared_memory_mib: int = Field(default=64, ge=1, le=65_536)
    open_files: int = Field(default=1_024, ge=64, le=1_048_576)
    gpu_count: int = Field(default=0, ge=0, le=64)


class RunnerPolicy(StrictModel):
    """Organizer-owned execution policy. Official is the fail-closed default."""

    schema_version: Literal['vaxreplay.runner-policy.v0.1'] = RUNNER_POLICY_SCHEMA_VERSION
    required_isolation: IsolationTier = IsolationTier.OFFICIAL
    network_allowed: Literal[False] = False
    root_filesystem_read_only: Literal[True] = True
    run_as_non_root: Literal[True] = True
    privileged: Literal[False] = False
    inherited_host_mounts: Literal[False] = False
    input_transport: Literal['stdin'] = 'stdin'
    output_transport: Literal['stdout'] = 'stdout'
    limits: RunLimits = Field(default_factory=RunLimits)


class SystemSubmissionManifest(StrictModel):
    """Immutable executable system admitted to the harness-plus-model track."""

    schema_version: Literal['vaxreplay.system-submission.v0.1'] = SYSTEM_SUBMISSION_SCHEMA_VERSION
    submission_id: str = Field(min_length=1)
    image_ref: str = Field(min_length=1)
    entrypoint: tuple[str, ...] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harness_id: str = Field(min_length=1)
    response_protocol: Literal[
        'vaxreplay.submission-json-stdout.v0.1',
        'vaxreplay.prospective-submission-json-stdout.v0.1',
    ] = RESPONSE_PROTOCOL

    @field_validator('image_ref')
    @classmethod
    def validate_image_ref(cls, value: str) -> str:
        if not _IMMUTABLE_IMAGE_PATTERN.fullmatch(value):
            raise ValueError('image_ref must be an OCI image ID or named image pinned by sha256 digest')
        return value

    @field_validator('entrypoint')
    @classmethod
    def validate_entrypoint(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value[0].startswith('/'):
            raise ValueError('entrypoint executable must be an absolute path inside the image')
        if any(not argument or '\x00' in argument for argument in value):
            raise ValueError('entrypoint arguments must be non-empty and cannot contain NUL')
        return value


class BackendCapabilities(StrictModel):
    backend_id: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    isolation_tier: IsolationTier
    network_isolation: bool
    host_filesystem_isolation: bool
    read_only_root: bool
    non_root_user: bool
    capability_drop: bool
    no_new_privileges: bool
    process_limit: bool
    memory_limit: bool
    cpu_limit: bool
    scratch_limit: bool
    fresh_worker_per_episode: bool


class EpisodeRunReceipt(StrictModel):
    ordinal: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    envelope_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    status: EpisodeRunStatus
    exit_code: int | None = None
    duration_ms: int = Field(ge=0)
    captured_stdout_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    captured_stdout_bytes: int = Field(ge=0)
    stdout_truncated: bool
    captured_stderr_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    captured_stderr_bytes: int = Field(ge=0)
    stderr_truncated: bool
    response_record_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    response_record_bytes: int = Field(gt=0)


class SuiteRunReceipt(StrictModel):
    """Authenticated audit record; it intentionally contains no model-controlled log text."""

    schema_version: Literal['vaxreplay.run-receipt.v0.2'] = RUN_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    challenge_id: str = Field(min_length=1)
    challenge_bundle_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    admission_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    suite_id: str = Field(min_length=1)
    suite_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    system_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    receipt_authentication: Literal['hmac-sha256'] = RECEIPT_AUTHENTICATION
    receipt_key_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    image_ref: str = Field(min_length=1)
    resolved_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    capabilities: BackendCapabilities
    sealed: bool
    started_at: datetime
    finished_at: datetime
    responses_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    responses_bytes: int = Field(gt=0)
    episodes: tuple[EpisodeRunReceipt, ...] = Field(min_length=1)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('run receipt timestamps must include a UTC offset')
        return value

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('finished_at cannot precede started_at')
        expected_sealed = self.capabilities.isolation_tier == IsolationTier.OFFICIAL
        if self.sealed != expected_sealed:
            raise ValueError('sealed must reflect the backend isolation tier')
        ordinals = tuple(receipt.ordinal for receipt in self.episodes)
        if ordinals != tuple(range(len(self.episodes))):
            raise ValueError('episode receipt ordinals must be contiguous and start at zero')
        episode_ids = tuple(receipt.episode_id for receipt in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError('episode receipt IDs must be unique')
        return self
