"""Canonical stdin/stdout payloads for hermetic promotion callbacks.

The OCI runner in :mod:`vaxreplay.operations.hermetic_execution` authenticates an
opaque input and output byte string.  This module gives those byte strings a strict,
source-independent meaning.  In particular, every captured artifact is embedded and
re-bound to its promotion metadata; workers cannot obtain ambient files or network
state, and the host can compare their exact output with the portable promotion.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.promotion_schema import (
    PromotedCaptureBinding,
    PromotedRawArtifactBinding,
    SourceVerificationResult,
)

HERMETIC_SOURCE_INPUT_SCHEMA_VERSION = 'vaxreplay.hermetic-source-verifier-input.v0.1'
HERMETIC_SOURCE_OUTPUT_SCHEMA_VERSION = 'vaxreplay.hermetic-source-verifier-output.v0.1'
HERMETIC_ADAPTER_INPUT_SCHEMA_VERSION = 'vaxreplay.hermetic-adapter-input.v0.1'
HERMETIC_ADAPTER_OUTPUT_SCHEMA_VERSION = 'vaxreplay.hermetic-adapter-output.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_OUTPUT_NAME_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._@+-]{0,199}$'


class HermeticCallbackProtocolError(ValueError):
    """A callback input/output is noncanonical or breaks an exact byte binding."""


def encode_callback_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError('callback payload must be bytes')
    return base64.b64encode(payload).decode('ascii')


def decode_callback_bytes(value: str) -> bytes:
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HermeticCallbackProtocolError('callback payload must use canonical base64') from error
    if encode_callback_bytes(payload) != value:
        raise HermeticCallbackProtocolError('callback payload must use canonical padded base64')
    return payload


class HermeticArtifactPayload(StrictModel):
    binding: PromotedRawArtifactBinding
    payload_base64: str

    @field_validator('payload_base64')
    @classmethod
    def validate_base64(cls, value: str) -> str:
        decode_callback_bytes(value)
        return value

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        payload = decode_callback_bytes(self.payload_base64)
        if (
            len(payload) != self.binding.file.byte_count
            or hashlib.sha256(payload).hexdigest() != self.binding.file.sha256
        ):
            raise ValueError('embedded artifact bytes differ from their promoted file binding')
        return self


class HermeticCapturePayload(StrictModel):
    binding: PromotedCaptureBinding
    artifacts: tuple[HermeticArtifactPayload, ...] = Field(min_length=1)

    @field_validator('artifacts')
    @classmethod
    def validate_artifact_order(
        cls,
        value: tuple[HermeticArtifactPayload, ...],
    ) -> tuple[HermeticArtifactPayload, ...]:
        roles = tuple(item.binding.role for item in value)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError('hermetic capture artifacts must use sorted unique roles')
        return value

    @model_validator(mode='after')
    def validate_capture_inventory(self) -> Self:
        if tuple(item.binding for item in self.artifacts) != self.binding.artifacts:
            raise ValueError('hermetic artifact inventory differs from the promoted capture binding')
        return self


class HermeticSourceVerifierInput(StrictModel):
    schema_version: Literal['vaxreplay.hermetic-source-verifier-input.v0.1'] = HERMETIC_SOURCE_INPUT_SCHEMA_VERSION
    source_id: str = Field(min_length=1, max_length=200)
    capture_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    captures: tuple[HermeticCapturePayload, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_sources(self) -> Self:
        if any(capture.binding.source_id != self.source_id for capture in self.captures):
            raise ValueError('hermetic source-verifier captures belong to another source')
        identities = tuple(capture.binding.attempt_id for capture in self.captures)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError('hermetic source-verifier captures must use sorted unique attempt IDs')
        return self


class HermeticSourceVerifierOutput(StrictModel):
    schema_version: Literal['vaxreplay.hermetic-source-verifier-output.v0.1'] = HERMETIC_SOURCE_OUTPUT_SCHEMA_VERSION
    result: SourceVerificationResult
    verified_records_base64: str

    @field_validator('verified_records_base64')
    @classmethod
    def validate_records(cls, value: str) -> str:
        if not decode_callback_bytes(value):
            raise ValueError('verified source-record output cannot be empty')
        return value

    @model_validator(mode='after')
    def validate_record_binding(self) -> Self:
        records = decode_callback_bytes(self.verified_records_base64)
        if hashlib.sha256(records).hexdigest() != self.result.verified_source_record_inventory_sha256:
            raise ValueError('source-verifier output records differ from the result inventory digest')
        return self


class HermeticAdapterSourceInput(StrictModel):
    source_id: str = Field(min_length=1, max_length=200)
    captures: tuple[HermeticCapturePayload, ...] = Field(min_length=1)
    verification_result: SourceVerificationResult
    verified_records_base64: str

    @field_validator('verified_records_base64')
    @classmethod
    def validate_records(cls, value: str) -> str:
        if not decode_callback_bytes(value):
            raise ValueError('adapter source records cannot be empty')
        return value

    @model_validator(mode='after')
    def validate_source(self) -> Self:
        records = decode_callback_bytes(self.verified_records_base64)
        if (
            self.verification_result.source_id != self.source_id
            or any(capture.binding.source_id != self.source_id for capture in self.captures)
            or hashlib.sha256(records).hexdigest() != self.verification_result.verified_source_record_inventory_sha256
        ):
            raise ValueError('hermetic adapter source input breaks its source-verification binding')
        return self


class HermeticAdapterInput(StrictModel):
    schema_version: Literal['vaxreplay.hermetic-adapter-input.v0.1'] = HERMETIC_ADAPTER_INPUT_SCHEMA_VERSION
    sources: tuple[HermeticAdapterSourceInput, ...] = Field(min_length=1)

    @field_validator('sources')
    @classmethod
    def validate_source_order(
        cls,
        value: tuple[HermeticAdapterSourceInput, ...],
    ) -> tuple[HermeticAdapterSourceInput, ...]:
        source_ids = tuple(item.source_id for item in value)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError('hermetic adapter sources must use sorted unique source IDs')
        return value


class HermeticNamedOutput(StrictModel):
    name: str = Field(pattern=_OUTPUT_NAME_PATTERN)
    payload_base64: str

    @field_validator('payload_base64')
    @classmethod
    def validate_payload(cls, value: str) -> str:
        if not decode_callback_bytes(value):
            raise ValueError('named adapter output cannot be empty')
        return value


class HermeticAdapterOutput(StrictModel):
    schema_version: Literal['vaxreplay.hermetic-adapter-output.v0.1'] = HERMETIC_ADAPTER_OUTPUT_SCHEMA_VERSION
    candidate_records_base64: str
    evidence_records_base64: str
    dispositions_base64: str
    auxiliary_outputs: tuple[HermeticNamedOutput, ...] = ()

    @field_validator('candidate_records_base64', 'evidence_records_base64', 'dispositions_base64')
    @classmethod
    def validate_required_output(cls, value: str) -> str:
        if not decode_callback_bytes(value):
            raise ValueError('required adapter outputs cannot be empty')
        return value

    @field_validator('auxiliary_outputs')
    @classmethod
    def validate_auxiliary_order(
        cls,
        value: tuple[HermeticNamedOutput, ...],
    ) -> tuple[HermeticNamedOutput, ...]:
        names = tuple(item.name for item in value)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError('auxiliary adapter outputs must use sorted unique names')
        return value


def parse_source_input(payload: bytes) -> HermeticSourceVerifierInput:
    return _parse_canonical(payload, HermeticSourceVerifierInput, 'source-verifier input')


def parse_source_output(payload: bytes) -> HermeticSourceVerifierOutput:
    return _parse_canonical(payload, HermeticSourceVerifierOutput, 'source-verifier output')


def parse_adapter_input(payload: bytes) -> HermeticAdapterInput:
    return _parse_canonical(payload, HermeticAdapterInput, 'adapter input')


def parse_adapter_output(payload: bytes) -> HermeticAdapterOutput:
    return _parse_canonical(payload, HermeticAdapterOutput, 'adapter output')


def _parse_canonical[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise HermeticCallbackProtocolError(f'{label} must be nonempty exact bytes')
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise HermeticCallbackProtocolError(f'invalid {label}: {error}') from error
    if canonical_json_bytes(value) != payload:
        raise HermeticCallbackProtocolError(f'{label} must use canonical JSON bytes')
    return value
