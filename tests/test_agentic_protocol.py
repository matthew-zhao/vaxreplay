from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from vaxreplay.agentic.protocol import (
    AgenticExecutionPolicy,
    AgenticModelUsage,
    AgenticRunFailureCode,
    AgenticRunReceipt,
    AgenticTool,
    AgenticToolPolicy,
    agentic_policy_sha256,
    agentic_receipt_key_id,
    agentic_run_receipt_hmac,
)
from vaxreplay.runner.schema import IsolationTier


def _policy() -> AgenticExecutionPolicy:
    return AgenticExecutionPolicy(
        required_workspace_broker_id='fixture-exact-byte-broker',
        required_workspace_broker_version='1',
        required_workspace_broker_executable_sha256='b' * 64,
    )


def _receipt(**updates) -> AgenticRunReceipt:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    base = AgenticRunReceipt(
        run_id='a' * 32,
        task_id='task-1',
        episode_manifest_sha256='0' * 64,
        workspace_manifest_sha256='1' * 64,
        workspace_tree_sha256='2' * 64,
        model_visible_surface_sha256='3' * 64,
        build_policy_sha256='b' * 64,
        discovery_manifest_sha256='c' * 64,
        alias_seed_commitment_sha256='d' * 64,
        alias_permutation_receipt_sha256='1' * 64,
        temporal_admission_sha256='4' * 64,
        contamination_admission_sha256='5' * 64,
        workspace_admission_sha256='e' * 64,
        attempt_reservation_sha256='f' * 64,
        policy_sha256=agentic_policy_sha256(_policy()),
        receipt_key_id='0' * 64,
        harness_id='reference-agent',
        harness_version='1.0',
        harness_image_or_commitment='sha256:' + '6' * 64,
        harness_manifest_sha256='b' * 64,
        harness_behavior_sha256='c' * 64,
        harness_execution_mode='fixed_model_loop',
        requested_model_id='model-v1',
        resolved_model_id='model-v1-snapshot',
        adapter_id='gateway-reference-v1',
        isolation_tier=IsolationTier.OFFICIAL,
        sealed=True,
        network_isolation=True,
        host_filesystem_isolation=True,
        gateway_channel_isolation=True,
        tool_tracing_authoritative=True,
        development_only=False,
        started_at=started,
        finished_at=started + timedelta(seconds=10),
        duration_ms=10_000,
        usage=AgenticModelUsage(
            model_calls=3,
            input_tokens=100,
            output_tokens=20,
            gateway_metering_authoritative=True,
        ),
        transcript_sha256='7' * 64,
        tool_events_sha256='8' * 64,
        workspace_broker_attestation_sha256='6' * 64,
        scratch_tree_sha256='9' * 64,
        final_submission_sha256='a' * 64,
        final_submission_bytes=123,
        accepted=True,
        residual_retrospective_selection_contamination=True,
    )
    return AgenticRunReceipt.model_validate({**base.model_dump(), **updates})


def test_policy_is_task_level_and_fail_closed() -> None:
    policy = _policy()

    assert policy.limits.max_model_calls == 20
    assert policy.limits.wall_seconds == 1200
    assert AgenticTool.MODEL_GENERATE in policy.tool_policy.allowed_tools
    assert policy.tool_policy.general_network_allowed is False
    assert policy.intermediate_scoring_feedback is False
    assert policy.workspace_filesystem_mounted_to_worker is False
    assert policy.workspace_metadata_exposed_to_worker is False


def test_tool_policy_rejects_missing_retrieval_or_noncanonical_order() -> None:
    with pytest.raises(ValidationError, match='requires workspace retrieval'):
        AgenticToolPolicy(
            allowed_tools=(
                AgenticTool.LIST_WORKSPACE,
                AgenticTool.READ_WORKSPACE,
                AgenticTool.WRITE_SCRATCH,
                AgenticTool.MODEL_GENERATE,
            )
        )
    with pytest.raises(ValidationError, match='canonical protocol order'):
        AgenticToolPolicy(allowed_tools=tuple(reversed(tuple(AgenticTool))))


def test_official_receipt_requires_authoritative_controls() -> None:
    with pytest.raises(ValidationError, match='every isolation and metering control'):
        _receipt(network_isolation=False)

    development = _receipt(
        isolation_tier=IsolationTier.DEVELOPMENT,
        sealed=False,
        development_only=True,
        network_isolation=False,
        host_filesystem_isolation=False,
        gateway_channel_isolation=False,
        tool_tracing_authoritative=False,
        usage=AgenticModelUsage(
            model_calls=1,
            input_tokens=1,
            output_tokens=1,
            gateway_metering_authoritative=False,
        ),
    )
    assert development.development_only
    assert development.residual_model_weight_contamination


def test_rejected_receipt_requires_failure_and_empty_output_is_allowed() -> None:
    rejected = _receipt(
        accepted=False,
        failure_code=AgenticRunFailureCode.INVALID_SUBMISSION,
        final_submission_bytes=0,
        final_submission_sha256='0' * 64,
    )
    assert rejected.failure_code == AgenticRunFailureCode.INVALID_SUBMISSION
    with pytest.raises(ValidationError, match='rejected receipts require one'):
        _receipt(accepted=False, final_submission_bytes=0)


def test_receipt_hmac_is_bound_and_requires_a_real_key() -> None:
    receipt = _receipt()
    key = bytes.fromhex('ab' * 32)

    assert len(agentic_run_receipt_hmac(receipt, key)) == 64
    assert len(agentic_receipt_key_id(key)) == 64
    with pytest.raises(ValueError, match='at least 32 bytes'):
        agentic_run_receipt_hmac(receipt, b'short')
