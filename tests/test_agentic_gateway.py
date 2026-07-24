from __future__ import annotations

import pytest

from vaxreplay.agentic.gateway import (
    AgenticGatewayTranscript,
    AgenticModelMessage,
    AgenticModelRequest,
    GatewayBudgetError,
    GatewayProtocolError,
    MeteredFakeGateway,
    ScriptedGatewayTurn,
)
from vaxreplay.agentic.protocol import AgenticRunLimits


def _request(index: int, *, max_output_tokens: int = 10) -> AgenticModelRequest:
    return AgenticModelRequest(
        run_id='a' * 32,
        call_index=index,
        messages=(
            AgenticModelMessage(role='system', content='Use only the frozen workspace.'),
            AgenticModelMessage(role='user', content='Analyze the next source.'),
        ),
        max_output_tokens=max_output_tokens,
    )


def test_fake_gateway_enforces_order_and_binds_authoritative_usage() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='organizer-selected-snapshot',
        limits=AgenticRunLimits(max_model_calls=2, max_input_tokens=20, max_output_tokens=10),
        scripted_turns=(
            ScriptedGatewayTurn(content='first', input_tokens=6, output_tokens=3),
            ScriptedGatewayTurn(content='second', input_tokens=7, output_tokens=4),
        ),
    )

    first = gateway.generate(_request(0))
    second = gateway.generate(_request(1))

    assert first.resolved_model_id == 'organizer-selected-snapshot'
    assert second.call_index == 1
    assert gateway.input_tokens == 13
    assert gateway.output_tokens == 7
    assert len(gateway.receipts) == 2
    assert len(gateway.transcript_sha256) == 64
    with pytest.raises(GatewayBudgetError, match='exhausted'):
        gateway.generate(_request(2))


def test_fake_gateway_rejects_cross_run_and_skipped_calls() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='model',
        limits=AgenticRunLimits(),
        scripted_turns=(ScriptedGatewayTurn(content='x', input_tokens=1, output_tokens=1),),
    )
    with pytest.raises(GatewayProtocolError, match='contiguous'):
        gateway.generate(_request(1))
    wrong_run = _request(0).model_copy(update={'run_id': 'b' * 32})
    with pytest.raises(GatewayProtocolError, match='match the run'):
        gateway.generate(wrong_run)


def test_fake_gateway_fails_closed_on_authoritative_token_overrun() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='model',
        limits=AgenticRunLimits(max_input_tokens=5, max_output_tokens=5),
        scripted_turns=(ScriptedGatewayTurn(content='too much', input_tokens=6, output_tokens=1),),
    )
    with pytest.raises(GatewayBudgetError, match='provider usage'):
        gateway.generate(_request(0, max_output_tokens=5))


def test_fake_gateway_rejects_per_call_usage_above_request() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='model',
        limits=AgenticRunLimits(max_output_tokens=100),
        scripted_turns=(ScriptedGatewayTurn(content='large', input_tokens=1, output_tokens=11),),
    )
    with pytest.raises(GatewayProtocolError, match='per-call'):
        gateway.generate(_request(0, max_output_tokens=10))


def test_transcript_binds_exact_requests_responses_and_usage() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='model-snapshot',
        limits=AgenticRunLimits(max_reasoning_tokens=10),
        scripted_turns=(ScriptedGatewayTurn(content='answer', input_tokens=3, output_tokens=2, reasoning_tokens=4),),
    )
    gateway.generate(_request(0))

    parsed = AgenticGatewayTranscript.model_validate_json(gateway.transcript_bytes)
    assert parsed.reasoning_tokens == 4
    assert parsed.exchanges[0].request == _request(0)
    with pytest.raises(ValueError, match='exact response'):
        parsed.exchanges[0].model_copy(
            update={'receipt': parsed.exchanges[0].receipt.model_copy(update={'response_sha256': '0' * 64})}
        ).model_validate(
            {
                **parsed.exchanges[0].model_dump(),
                'receipt': {
                    **parsed.exchanges[0].receipt.model_dump(),
                    'response_sha256': '0' * 64,
                },
            }
        )


def test_fake_gateway_enforces_reported_reasoning_budget() -> None:
    gateway = MeteredFakeGateway(
        run_id='a' * 32,
        resolved_model_id='model',
        limits=AgenticRunLimits(max_reasoning_tokens=3),
        scripted_turns=(ScriptedGatewayTurn(content='x', input_tokens=1, output_tokens=1, reasoning_tokens=4),),
    )
    with pytest.raises(GatewayBudgetError, match='reasoning usage'):
        gateway.generate(_request(0))
