"""Development-only local CLI baselines for VaxReplay challenges."""

from vaxreplay.reference_harness.runner import (
    ReferenceHarnessInputError,
    VerifiedChallengeEnvelope,
    canonical_receipt_bytes,
    load_verified_challenge_envelope,
    render_challenge_prompt,
    render_cursor_challenge_prompt,
    run_reference_harness,
    verify_challenge_envelope,
)
from vaxreplay.reference_harness.schema import (
    CursorEventKindObservation,
    CursorParseConsistencyFlags,
    CursorParseFailureInventory,
    ReferenceHarnessFailure,
    ReferenceHarnessFailureCode,
    ReferenceHarnessLimits,
    ReferenceHarnessName,
    ReferenceHarnessReceipt,
    ReferenceHarnessRuntimeIdentity,
)

__all__ = [
    'CursorEventKindObservation',
    'CursorParseConsistencyFlags',
    'CursorParseFailureInventory',
    'ReferenceHarnessFailure',
    'ReferenceHarnessFailureCode',
    'ReferenceHarnessInputError',
    'ReferenceHarnessLimits',
    'ReferenceHarnessName',
    'ReferenceHarnessReceipt',
    'ReferenceHarnessRuntimeIdentity',
    'VerifiedChallengeEnvelope',
    'canonical_receipt_bytes',
    'load_verified_challenge_envelope',
    'render_challenge_prompt',
    'render_cursor_challenge_prompt',
    'run_reference_harness',
    'verify_challenge_envelope',
]
