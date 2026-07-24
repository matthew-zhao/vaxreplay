"""Task-bound invocation and final-submission unions for Agentic Replay.

This module is deliberately a narrow bridge between the existing candidate-ranking track and the
development-only registry-observed clinical-execution track.  The invocation is organizer-owned;
the guest may submit only the response family and task identity committed by that invocation.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from vaxreplay.agentic.response_protocol import AgenticResponseProtocol
from vaxreplay.agentic.schema import AgenticTaskEnvelope
from vaxreplay.agentic.scoring import AgenticSubmissionV1
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_scoring import validate_execution_submission
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission, ExecutionTask

AGENTIC_TASK_INVOCATION_SCHEMA_VERSION = 'vaxreplay.agentic-task-invocation.dev-v0.1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'

type AgenticRuntimeTask = Annotated[
    AgenticTaskEnvelope | ExecutionTask,
    Field(discriminator='schema_version'),
]
type AgenticRuntimeSubmission = Annotated[
    AgenticSubmissionV1 | ExecutionSubmission,
    Field(discriminator='schema_version'),
]

_TASK_ADAPTER = TypeAdapter(AgenticRuntimeTask)
_SUBMISSION_ADAPTER = TypeAdapter(AgenticRuntimeSubmission)


def response_protocol_for_task(task: AgenticRuntimeTask) -> AgenticResponseProtocol:
    if isinstance(task, AgenticTaskEnvelope):
        return AgenticResponseProtocol.RANKING
    if isinstance(task, ExecutionTask):
        return AgenticResponseProtocol.CLINICAL_EXECUTION
    raise TypeError('unsupported Agentic runtime task')


def response_protocol_for_submission(submission: AgenticRuntimeSubmission) -> AgenticResponseProtocol:
    if isinstance(submission, AgenticSubmissionV1):
        return AgenticResponseProtocol.RANKING
    if isinstance(submission, ExecutionSubmission):
        return AgenticResponseProtocol.CLINICAL_EXECUTION
    raise TypeError('unsupported Agentic runtime submission')


class AgenticTaskInvocation(StrictModel):
    """Public task plus the exact workspace and response family trusted by one run."""

    schema_version: Literal['vaxreplay.agentic-task-invocation.dev-v0.1'] = AGENTIC_TASK_INVOCATION_SCHEMA_VERSION
    task: AgenticRuntimeTask
    task_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_protocol: AgenticResponseProtocol

    @model_validator(mode='after')
    def validate_binding(self) -> Self:
        if hashlib.sha256(canonical_json_bytes(self.task)).hexdigest() != self.task_sha256:
            raise ValueError('task invocation does not bind the exact public task')
        if response_protocol_for_task(self.task) != self.response_protocol:
            raise ValueError('task invocation response protocol does not match its task family')
        return self

    @classmethod
    def from_task(
        cls,
        task: AgenticRuntimeTask,
        *,
        workspace_manifest_sha256: str,
    ) -> AgenticTaskInvocation:
        canonical_task = _TASK_ADAPTER.validate_json(canonical_json_bytes(task))
        return cls(
            task=canonical_task,
            task_sha256=hashlib.sha256(canonical_json_bytes(canonical_task)).hexdigest(),
            workspace_manifest_sha256=workspace_manifest_sha256,
            response_protocol=response_protocol_for_task(canonical_task),
        )


def agentic_task_invocation_sha256(invocation: AgenticTaskInvocation) -> str:
    canonical = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(invocation))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def submission_json_schema_for_invocation(invocation: AgenticTaskInvocation) -> dict[str, Any]:
    """Return only the response schema selected by the organizer-owned task invocation."""

    invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(invocation))
    if invocation.response_protocol == AgenticResponseProtocol.RANKING:
        return AgenticSubmissionV1.model_json_schema()
    if invocation.response_protocol == AgenticResponseProtocol.CLINICAL_EXECUTION:
        return ExecutionSubmission.model_json_schema()
    raise TypeError('unsupported Agentic response protocol')


def submission_json_schema_sha256(invocation: AgenticTaskInvocation) -> str:
    return hashlib.sha256(canonical_json_bytes(submission_json_schema_for_invocation(invocation))).hexdigest()


def parse_submission_for_invocation(
    invocation: AgenticTaskInvocation,
    payload: bytes,
) -> AgenticRuntimeSubmission:
    """Parse provider/harness JSON through the closed union and enforce its public task binding."""

    submission = _SUBMISSION_ADAPTER.validate_json(payload)
    validate_submission_for_invocation(invocation, submission)
    return submission


def validate_submission_for_invocation(
    invocation: AgenticTaskInvocation,
    submission: AgenticRuntimeSubmission,
) -> None:
    """Reject a strict submission unless it is bound to the organizer-owned invocation."""

    invocation = AgenticTaskInvocation.model_validate_json(canonical_json_bytes(invocation))
    submission = _SUBMISSION_ADAPTER.validate_json(canonical_json_bytes(submission))
    if response_protocol_for_submission(submission) != invocation.response_protocol:
        raise ValueError('submission protocol does not match the task invocation')

    task = invocation.task
    if isinstance(task, AgenticTaskEnvelope) and isinstance(submission, AgenticSubmissionV1):
        if (
            submission.task_id,
            submission.workspace_manifest_sha256,
        ) != (
            task.task_id,
            invocation.workspace_manifest_sha256,
        ):
            raise ValueError('ranking submission is bound to a different task or workspace')
        return

    if isinstance(task, ExecutionTask) and isinstance(submission, ExecutionSubmission):
        issues = validate_execution_submission(task, submission)
        if issues:
            raise ValueError('clinical-execution submission violates its public task contract')
        return

    raise ValueError('submission and invocation task families do not match')


__all__ = [
    'AGENTIC_TASK_INVOCATION_SCHEMA_VERSION',
    'AgenticRuntimeSubmission',
    'AgenticRuntimeTask',
    'AgenticTaskInvocation',
    'agentic_task_invocation_sha256',
    'parse_submission_for_invocation',
    'response_protocol_for_submission',
    'response_protocol_for_task',
    'submission_json_schema_for_invocation',
    'submission_json_schema_sha256',
    'validate_submission_for_invocation',
]
