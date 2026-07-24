"""Pluggable isolation backend boundary for the VaxReplay runner."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol

from vaxreplay.runner.schema import (
    BackendCapabilities,
    RunnerPolicy,
    SystemSubmissionManifest,
)


class BackendPolicyError(ValueError):
    """Raised before launch when a backend cannot meet the requested isolation policy."""


class IsolationCleanupError(RuntimeError):
    """Raised when the runner cannot prove that an isolation unit was destroyed."""


class RawExecutionStatus(str, enum.Enum):
    EXITED = 'exited'
    TIMED_OUT = 'timed_out'
    RESPONSE_LIMIT = 'response_limit'
    LOG_LIMIT = 'log_limit'
    BACKEND_ERROR = 'backend_error'


@dataclass(frozen=True)
class PreparedBackend:
    capabilities: BackendCapabilities
    resolved_image_id: str


@dataclass(frozen=True)
class RawExecutionResult:
    status: RawExecutionStatus
    exit_code: int | None
    duration_ms: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool


class IsolationBackend(Protocol):
    def prepare(self, system: SystemSubmissionManifest, policy: RunnerPolicy) -> PreparedBackend: ...

    def run(
        self,
        *,
        input_bytes: bytes,
        system: SystemSubmissionManifest,
        policy: RunnerPolicy,
        prepared: PreparedBackend,
    ) -> RawExecutionResult: ...
