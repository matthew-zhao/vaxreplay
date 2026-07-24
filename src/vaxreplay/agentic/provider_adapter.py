"""Credential-free provider adapter boundary for the authenticated inference gateway."""

from __future__ import annotations

import enum
import hashlib
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.gateway import AgenticGatewayUsage, AgenticModelRequest
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

PROVIDER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION = 'vaxreplay.provider-adapter-descriptor.v0.1'
PROVIDER_CALL_RESULT_SCHEMA_VERSION = 'vaxreplay.provider-call-result.v0.1'


class ProviderFailureCode(str, enum.Enum):
    TIMEOUT = 'provider_timeout'
    RATE_LIMIT = 'provider_rate_limit'
    REJECTED = 'provider_rejected'
    PROTOCOL = 'provider_protocol'
    INTERNAL = 'provider_internal'


class ProviderCallFailure(RuntimeError):
    """Safe provider error; upstream response bodies and exception strings stay private."""

    def __init__(self, code: ProviderFailureCode):
        super().__init__(code.value)
        self.code = code


class ProviderAdapterDescriptor(StrictModel):
    schema_version: Literal['vaxreplay.provider-adapter-descriptor.v0.1'] = PROVIDER_ADAPTER_DESCRIPTOR_SCHEMA_VERSION
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider: str = Field(min_length=1)
    synchronous_non_streaming: Literal[True] = True
    automatic_retries: Literal[False] = False
    credential_passed_in_request: Literal[False] = False
    input_estimate_is_conservative_upper_bound: Literal[True] = True


class ProviderCallResult(StrictModel):
    schema_version: Literal['vaxreplay.provider-call-result.v0.1'] = PROVIDER_CALL_RESULT_SCHEMA_VERSION
    resolved_model_id: str = Field(min_length=1)
    provider_reported_model_id: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=2_000_000)
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal']
    usage: AgenticGatewayUsage
    provider_request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_request_bytes: int = Field(gt=0)
    provider_response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_response_bytes: int = Field(gt=0)
    provider_request_id: str = Field(min_length=1, max_length=500)
    http_status: int = Field(ge=200, le=299)
    started_at: datetime
    finished_at: datetime
    provider_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator('started_at', 'finished_at')
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f'{info.field_name} must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_interval(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError('provider call cannot finish before it starts')
        if self.resolved_model_id != self.provider_reported_model_id:
            raise ValueError('resolved model ID must be the exact provider-reported model ID')
        return self


class ProviderModelRoute(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def provider_model_id(self) -> str: ...

    @property
    def resolved_model_id(self) -> str: ...

    @property
    def accepted_provider_model_ids(self) -> tuple[str, ...]: ...

    @property
    def endpoint_origin(self) -> str: ...

    @property
    def endpoint_path(self) -> str: ...

    @property
    def fixed_parameters_sha256(self) -> str: ...

    @property
    def provider_storage_disabled(self) -> bool: ...

    @property
    def provider_data_control(self) -> str: ...

    @property
    def provider_data_control_attested(self) -> bool: ...

    @property
    def provider_data_control_attestation_sha256(self) -> str | None: ...


class ProviderAdapter(Protocol):
    @property
    def descriptor(self) -> ProviderAdapterDescriptor: ...

    def estimate_input_tokens(self, request: AgenticModelRequest, route: ProviderModelRoute) -> int: ...

    def generate(
        self,
        request: AgenticModelRequest,
        route: ProviderModelRoute,
        *,
        timeout_seconds: float,
    ) -> ProviderCallResult: ...


@dataclass(frozen=True)
class ScriptedProviderTurn:
    content: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = 0
    estimated_input_tokens: int | None = None
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal'] = 'completed'
    provider_reported_model_id: str | None = None
    failure: ProviderFailureCode | None = None
    provider_cost_usd: float | None = None


class ScriptedProviderAdapter:
    """Deterministic adapter used to exercise the real authenticated gateway state machine."""

    def __init__(
        self,
        *,
        provider: str,
        adapter_id: str,
        adapter_version: str,
        executable_sha256: str,
        config_sha256: str,
        turns: Iterable[ScriptedProviderTurn],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._descriptor = ProviderAdapterDescriptor(
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            executable_sha256=executable_sha256,
            config_sha256=config_sha256,
            provider=provider,
        )
        self._turns = tuple(turns)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def descriptor(self) -> ProviderAdapterDescriptor:
        return self._descriptor

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def estimate_input_tokens(self, request: AgenticModelRequest, route: ProviderModelRoute) -> int:
        del request, route
        with self._lock:
            index = self._call_count
            if index >= len(self._turns):
                return 1
            turn = self._turns[index]
        return turn.estimated_input_tokens if turn.estimated_input_tokens is not None else turn.input_tokens

    def generate(
        self,
        request: AgenticModelRequest,
        route: ProviderModelRoute,
        *,
        timeout_seconds: float,
    ) -> ProviderCallResult:
        if timeout_seconds <= 0:
            raise ProviderCallFailure(ProviderFailureCode.TIMEOUT)
        with self._lock:
            index = self._call_count
            self._call_count += 1
        if index >= len(self._turns):
            raise ProviderCallFailure(ProviderFailureCode.INTERNAL)
        turn = self._turns[index]
        if turn.failure is not None:
            raise ProviderCallFailure(turn.failure)
        started = self._clock()
        request_body = canonical_json_bytes(
            {
                'model': route.provider_model_id,
                'messages_sha256': hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
                'max_output_tokens': request.max_output_tokens,
                'store': False,
            }
        )
        reported_model = turn.provider_reported_model_id or route.resolved_model_id
        response_body = canonical_json_bytes(
            {
                'id': f'fixture-{request.run_id}-{request.call_index}',
                'model': reported_model,
                'content': turn.content,
                'input_tokens': turn.input_tokens,
                'output_tokens': turn.output_tokens,
                'reasoning_tokens': turn.reasoning_tokens,
            }
        )
        return ProviderCallResult(
            resolved_model_id=reported_model,
            provider_reported_model_id=reported_model,
            content=turn.content,
            stop_reason=turn.stop_reason,
            usage=AgenticGatewayUsage(
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                reasoning_tokens=turn.reasoning_tokens,
            ),
            provider_request_sha256=hashlib.sha256(request_body).hexdigest(),
            provider_request_bytes=len(request_body),
            provider_response_sha256=hashlib.sha256(response_body).hexdigest(),
            provider_response_bytes=len(response_body),
            provider_request_id=f'fixture-{request.run_id}-{request.call_index}',
            http_status=200,
            started_at=started,
            finished_at=self._clock(),
            provider_cost_usd=turn.provider_cost_usd,
        )
