"""Fail-closed semantic dispatch for every promotion-capable collector."""

from __future__ import annotations

from typing import Any

from vaxreplay.operations.collector import (
    StaticCollectionError,
    load_static_run_manifest,
    verify_static_attempt_policy,
)
from vaxreplay.operations.immport_capture import (
    load_immport_authenticated_run_manifest,
    verify_immport_attempt_policy,
)
from vaxreplay.operations.policy import (
    IMMPORT_AUTHENTICATED_COLLECTOR_ID,
    STATIC_HTTPS_COLLECTOR_ID,
    parse_supported_collector_job_configuration,
)
from vaxreplay.operations.store import OperationalStore


def load_supported_run_manifest(store: OperationalStore, attempt_id: str) -> Any:
    """Replay one successful run through the exact registered collector verifier."""

    attempt = store.get_attempt(attempt_id)
    run = store.get_logical_run(attempt.logical_run_id)
    job = store.get_job(run.job_spec_sha256)
    if job.spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
        return load_static_run_manifest(store, attempt_id)
    if job.spec.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        return load_immport_authenticated_run_manifest(store, attempt_id)
    raise StaticCollectionError(
        f'no semantic verifier is registered for successful collector {job.spec.collector_id!r}'
    )


def verify_all_supported_run_manifests(store: OperationalStore) -> tuple[Any, ...]:
    """Replay every successful run and reject unknown collector semantics.

    All supported job configurations are parsed even before success.  Unknown registered jobs
    remain inert operational records, but an unknown *successful* collector cannot be
    checkpointed or promoted.
    """

    for job in store.list_jobs():
        if job.spec.collector_id in {
            STATIC_HTTPS_COLLECTOR_ID,
            IMMPORT_AUTHENTICATED_COLLECTOR_ID,
        }:
            configuration_valid = True
            try:
                parse_supported_collector_job_configuration(
                    job.spec.collector_id,
                    job.spec.configuration,
                )
            except ValueError:
                configuration_valid = False
            if not configuration_valid:
                # Validation errors can render rejected values.  Do not retain them as a cause
                # or implicit context on this operational trust boundary.
                raise StaticCollectionError(f'collector {job.spec.collector_id!r} has invalid immutable configuration')
    manifests = []
    for run in store.list_logical_runs():
        job = store.get_job(run.job_spec_sha256)
        if job.spec.collector_id == STATIC_HTTPS_COLLECTOR_ID:
            verify_static_attempt_policy(store, run)
        elif job.spec.collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
            verify_immport_attempt_policy(store, run)
        if run.successful_attempt_id is not None:
            manifests.append(load_supported_run_manifest(store, run.successful_attempt_id))
    return tuple(manifests)
