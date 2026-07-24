"""Closed response-protocol identifiers for task-level Agentic Replay runs.

The protocol identifier is part of the trusted invocation boundary.  It is not inferred from a
model-controlled submission because doing so would let one task family terminate a run with the
other family's answer shape.
"""

from __future__ import annotations

import enum

AGENTIC_RANKING_RESPONSE_PROTOCOL = 'vaxreplay.agentic-submission-file.v0.1'
CLINICAL_EXECUTION_RESPONSE_PROTOCOL = 'vaxreplay.clinical-execution-submission.dev-v0.1'


class AgenticResponseProtocol(str, enum.Enum):
    """Every final-submission protocol accepted by the task-level runner."""

    RANKING = AGENTIC_RANKING_RESPONSE_PROTOCOL
    CLINICAL_EXECUTION = CLINICAL_EXECUTION_RESPONSE_PROTOCOL


__all__ = [
    'AGENTIC_RANKING_RESPONSE_PROTOCOL',
    'CLINICAL_EXECUTION_RESPONSE_PROTOCOL',
    'AgenticResponseProtocol',
]
