"""Fail-closed boundary for the not-yet-implemented Tinker training collector."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import NoReturn, Protocol

from tinker_cookbook import renderers

from vaxreplay.bundle import EpisodeBundle
from vaxreplay.qa.admission import AdmissionTokenConsumer
from vaxreplay.qa.authority import SignedGradientAdmission
from vaxreplay.qa.firewall import (
    QuarantinedBatch,
    TrainingRewardFirewall,
)


class UnadmittedTrainingDisabled(RuntimeError):
    """Raised before Tinker can receive an unverified per-trajectory reward."""


class GradientAdmissionBroker(Protocol):
    """Out-of-process QA authority reserved for a batch-aware collector."""

    async def admit(self, batch: QuarantinedBatch) -> SignedGradientAdmission: ...


def make_tinker_env(
    renderer: renderers.Renderer,
    bundle: EpisodeBundle,
    *,
    trajectory_id: str | None = None,
    firewall: TrainingRewardFirewall | None = None,
    admission_broker: GradientAdmissionBroker | None = None,
    trusted_public_key_bytes: bytes | None = None,
    consume_token: AdmissionTokenConsumer | None = None,
    trusted_now: Callable[[], datetime] | None = None,
    model_sha256: str | None = None,
    harness_sha256: str | None = None,
    tool_policy_sha256: str | None = None,
    environment_sha256: str | None = None,
    dataset_sha256: str | None = None,
    optimizer_config_sha256: str | None = None,
    max_trajectory_tokens: int | None = None,
    max_generation_tokens: int | None = None,
) -> NoReturn:
    """Refuse per-environment training until a batch-aware collector exists.

    ``EnvFromMessageEnv`` can create parse-error and context-overflow rewards
    without calling the message environment.  Releasing an admitted reward from
    ``MessageEnv.step`` would also admit one trajectory before Tinker has frozen
    the correlated rollout group or optimizer batch.  Both paths violate the
    reward-firewall contract, so supplying all legacy admission dependencies
    deliberately does not enable this adapter.
    """

    raise UnadmittedTrainingDisabled(
        'VaxReplay Tinker training is disabled: per-environment MessageEnv rewards '
        'cannot enforce reward-firewall admission because Tinker may emit parse/overflow '
        'rewards before MessageEnv.step and the complete optimizer batch is not yet frozen. '
        'A batch-aware collector must quarantine every trajectory, reject unsigned wrapper '
        'terminations, obtain one signed admission for the exact frozen optimizer batch, and '
        'atomically release rewards immediately before the optimizer update.'
    )
