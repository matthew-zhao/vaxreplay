"""Narrow, metered model-RPC contracts and a deterministic conformance gateway.

The fake gateway is deliberately in-process.  It exercises call ordering, server-selected model
identity, budget enforcement, and transcript binding without exposing a provider credential or a
general HTTP proxy.  A production Unix-socket transport can implement the same ``ModelGateway``
protocol inside an isolated worker boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from vaxreplay.agentic.protocol import AgenticRunLimits
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

MODEL_REQUEST_SCHEMA_VERSION = 'vaxreplay.agentic-model-request.v0.1'
MODEL_RESPONSE_SCHEMA_VERSION = 'vaxreplay.agentic-model-response.v0.1'
MODEL_CALL_RECEIPT_SCHEMA_VERSION = 'vaxreplay.agentic-model-call-receipt.v0.1'
MODEL_EXCHANGE_SCHEMA_VERSION = 'vaxreplay.agentic-model-exchange.v0.1'
GATEWAY_TRANSCRIPT_SCHEMA_VERSION = 'vaxreplay.agentic-gateway-transcript.v0.1'


class GatewayBudgetError(RuntimeError):
    """Raised before or after a call whose authoritative usage exceeds the sealed budget."""


class GatewayProtocolError(ValueError):
    """Raised when a harness request violates the narrow model-RPC contract."""


class AgenticModelMessage(StrictModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str = Field(min_length=1, max_length=2_000_000)


class AgenticModelRequest(StrictModel):
    schema_version: Literal['vaxreplay.agentic-model-request.v0.1'] = MODEL_REQUEST_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    call_index: int = Field(ge=0)
    messages: tuple[AgenticModelMessage, ...] = Field(min_length=1, max_length=1_000)
    max_output_tokens: int = Field(gt=0)
    response_schema_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_request(self) -> Self:
        if self.messages[0].role != 'system':
            raise ValueError('the first gateway message must be a system message')
        return self


class AgenticGatewayUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class AgenticModelResponse(StrictModel):
    schema_version: Literal['vaxreplay.agentic-model-response.v0.1'] = MODEL_RESPONSE_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    call_index: int = Field(ge=0)
    resolved_model_id: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=2_000_000)
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal', 'provider_error']
    usage: AgenticGatewayUsage


class AgenticModelCallReceipt(StrictModel):
    schema_version: Literal['vaxreplay.agentic-model-call-receipt.v0.1'] = MODEL_CALL_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    call_index: int = Field(ge=0)
    request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    resolved_model_id: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal', 'provider_error']
    metering_authoritative: Literal[True] = True


class AgenticModelExchange(StrictModel):
    schema_version: Literal['vaxreplay.agentic-model-exchange.v0.1'] = MODEL_EXCHANGE_SCHEMA_VERSION
    request: AgenticModelRequest
    response: AgenticModelResponse
    receipt: AgenticModelCallReceipt

    @model_validator(mode='after')
    def validate_exchange(self) -> Self:
        if (
            self.request.run_id,
            self.request.call_index,
            self.response.run_id,
            self.response.call_index,
            self.receipt.run_id,
            self.receipt.call_index,
        ) != (
            self.receipt.run_id,
            self.receipt.call_index,
            self.receipt.run_id,
            self.receipt.call_index,
            self.receipt.run_id,
            self.receipt.call_index,
        ):
            raise ValueError('gateway exchange run and call identities must match')
        if self.receipt.request_sha256 != hashlib.sha256(canonical_json_bytes(self.request)).hexdigest():
            raise ValueError('gateway receipt does not bind the exact request')
        if self.receipt.response_sha256 != hashlib.sha256(canonical_json_bytes(self.response)).hexdigest():
            raise ValueError('gateway receipt does not bind the exact response')
        if (
            self.receipt.resolved_model_id,
            self.receipt.input_tokens,
            self.receipt.output_tokens,
            self.receipt.reasoning_tokens,
            self.receipt.stop_reason,
        ) != (
            self.response.resolved_model_id,
            self.response.usage.input_tokens,
            self.response.usage.output_tokens,
            self.response.usage.reasoning_tokens,
            self.response.stop_reason,
        ):
            raise ValueError('gateway receipt does not match authoritative response metadata')
        return self


class AgenticGatewayTranscript(StrictModel):
    schema_version: Literal['vaxreplay.agentic-gateway-transcript.v0.1'] = GATEWAY_TRANSCRIPT_SCHEMA_VERSION
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    resolved_model_id: str | None = Field(default=None, min_length=1)
    exchanges: tuple[AgenticModelExchange, ...] = ()
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    metering_authoritative: Literal[True] = True

    @model_validator(mode='after')
    def validate_transcript(self) -> Self:
        if tuple(exchange.request.call_index for exchange in self.exchanges) != tuple(range(len(self.exchanges))):
            raise ValueError('gateway transcript call indexes must be contiguous')
        if any(exchange.request.run_id != self.run_id for exchange in self.exchanges):
            raise ValueError('gateway transcript contains a different run')
        models = {exchange.response.resolved_model_id for exchange in self.exchanges}
        if (not self.exchanges and self.resolved_model_id is not None) or (
            self.exchanges and models != {self.resolved_model_id}
        ):
            raise ValueError('gateway transcript must resolve exactly one server-selected model')
        expected_input = sum(exchange.response.usage.input_tokens for exchange in self.exchanges)
        expected_output = sum(exchange.response.usage.output_tokens for exchange in self.exchanges)
        reasoning = tuple(exchange.response.usage.reasoning_tokens for exchange in self.exchanges)
        expected_reasoning = (
            None if any(value is None for value in reasoning) else sum(value or 0 for value in reasoning)
        )
        if (self.input_tokens, self.output_tokens, self.reasoning_tokens) != (
            expected_input,
            expected_output,
            expected_reasoning,
        ):
            raise ValueError('gateway transcript usage does not equal its authoritative exchanges')
        return self


class ModelGateway(Protocol):
    def generate(self, request: AgenticModelRequest) -> AgenticModelResponse: ...


@dataclass(frozen=True)
class ScriptedGatewayTurn:
    content: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None
    stop_reason: Literal['completed', 'max_output_tokens', 'refusal', 'provider_error'] = 'completed'


class MeteredFakeGateway:
    """Deterministic conformance implementation with server-owned model selection and usage."""

    def __init__(
        self,
        *,
        run_id: str,
        resolved_model_id: str,
        limits: AgenticRunLimits,
        scripted_turns: Iterable[ScriptedGatewayTurn],
    ):
        if len(run_id) != 32 or any(character not in '0123456789abcdef' for character in run_id):
            raise ValueError('run_id must contain exactly 32 lowercase hexadecimal characters')
        if not resolved_model_id:
            raise ValueError('resolved_model_id cannot be empty')
        self._run_id = run_id
        self._resolved_model_id = resolved_model_id
        self._limits = limits
        self._turns = tuple(scripted_turns)
        self._receipts: list[AgenticModelCallReceipt] = []
        self._exchanges: list[AgenticModelExchange] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._reasoning_complete = True

    @property
    def receipts(self) -> tuple[AgenticModelCallReceipt, ...]:
        return tuple(self._receipts)

    @property
    def exchanges(self) -> tuple[AgenticModelExchange, ...]:
        return tuple(self._exchanges)

    @property
    def transcript(self) -> AgenticGatewayTranscript:
        return AgenticGatewayTranscript(
            run_id=self._run_id,
            resolved_model_id=self._resolved_model_id if self._exchanges else None,
            exchanges=tuple(self._exchanges),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            reasoning_tokens=self._reasoning_tokens if self._reasoning_complete else None,
        )

    @property
    def transcript_bytes(self) -> bytes:
        return canonical_json_bytes(self.transcript)

    @property
    def transcript_sha256(self) -> str:
        return hashlib.sha256(self.transcript_bytes).hexdigest()

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def reasoning_tokens(self) -> int | None:
        return self._reasoning_tokens if self._reasoning_complete else None

    def generate(self, request: AgenticModelRequest) -> AgenticModelResponse:
        expected_index = len(self._receipts)
        if request.run_id != self._run_id or request.call_index != expected_index:
            raise GatewayProtocolError('gateway requests must match the run and use contiguous call indexes')
        if expected_index >= self._limits.max_model_calls or expected_index >= len(self._turns):
            raise GatewayBudgetError('model-call budget exhausted')
        if request.max_output_tokens > self._limits.max_output_tokens:
            raise GatewayBudgetError('request max_output_tokens exceeds the run output-token budget')

        turn = self._turns[expected_index]
        next_input = self._input_tokens + turn.input_tokens
        next_output = self._output_tokens + turn.output_tokens
        next_reasoning = self._reasoning_tokens + (turn.reasoning_tokens or 0)
        if next_input > self._limits.max_input_tokens or next_output > self._limits.max_output_tokens:
            raise GatewayBudgetError('authoritative provider usage exceeds the run token budget')
        if turn.reasoning_tokens is not None and next_reasoning > self._limits.max_reasoning_tokens:
            raise GatewayBudgetError('authoritative provider reasoning usage exceeds the run token budget')
        if turn.output_tokens > request.max_output_tokens:
            raise GatewayProtocolError('scripted authoritative usage exceeds the per-call request limit')

        response = AgenticModelResponse(
            run_id=self._run_id,
            call_index=expected_index,
            resolved_model_id=self._resolved_model_id,
            content=turn.content,
            stop_reason=turn.stop_reason,
            usage=AgenticGatewayUsage(
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                reasoning_tokens=turn.reasoning_tokens,
            ),
        )
        receipt = AgenticModelCallReceipt(
            run_id=self._run_id,
            call_index=expected_index,
            request_sha256=hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
            response_sha256=hashlib.sha256(canonical_json_bytes(response)).hexdigest(),
            resolved_model_id=self._resolved_model_id,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            stop_reason=turn.stop_reason,
        )
        self._receipts.append(receipt)
        self._exchanges.append(
            AgenticModelExchange(
                request=request,
                response=response,
                receipt=receipt,
            )
        )
        self._input_tokens = next_input
        self._output_tokens = next_output
        self._reasoning_tokens = next_reasoning
        if turn.reasoning_tokens is None:
            self._reasoning_complete = False
        return response
