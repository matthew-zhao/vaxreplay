"""Shared immutable collector policy parsed at operational trust boundaries."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay.case_schema import StrictModel

STATIC_HTTPS_COLLECTOR_ID = 'static-https-v0.2'
IMMPORT_AUTHENTICATED_COLLECTOR_ID = 'immport-secret-broker-collector'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class StaticHttpsJobConfiguration(StrictModel):
    """Exact policy for a public static-HTTPS job revision."""

    collection_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    lease_seconds: int = Field(ge=1, le=24 * 60 * 60)
    max_attempts_per_slot: int = Field(ge=1, le=100)
    plan_deadline_seconds: int = Field(ge=1, le=24 * 60 * 60)
    request_deadline_seconds: int = Field(ge=1, le=60 * 60)
    dns_resolution_timeout_seconds: int = Field(ge=1, le=5 * 60)
    dns_resolution_attempts: Literal[1]
    max_dns_addresses: int = Field(ge=1, le=64)
    max_total_body_bytes: int = Field(gt=0, le=16 * 1024 * 1024 * 1024)
    catch_up_seconds: int | None = Field(default=None, ge=0, le=366 * 24 * 60 * 60)
    max_slots_per_wake: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode='after')
    def validate_deadline_hierarchy(self) -> Self:
        if self.plan_deadline_seconds > self.lease_seconds:
            raise ValueError('plan_deadline_seconds cannot exceed lease_seconds')
        if self.request_deadline_seconds > self.plan_deadline_seconds:
            raise ValueError('request_deadline_seconds cannot exceed plan_deadline_seconds')
        if self.dns_resolution_timeout_seconds > self.request_deadline_seconds:
            raise ValueError('dns_resolution_timeout_seconds cannot exceed request_deadline_seconds')
        return self


class ImmportAuthenticatedJobConfiguration(StrictModel):
    """Exact public policy for one credentialed, secret-brokered ImmPort job.

    The broker's credential and even its secret identifier are deliberately absent.  The
    immutable job binds only the public collection plan and the reviewed collector code and
    execution-environment digests.
    """

    collection_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,199}$')
    lease_seconds: int = Field(ge=1, le=24 * 60 * 60)
    max_attempts_per_slot: int = Field(ge=1, le=100)
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)


def parse_static_job_configuration(
    configuration: dict[str, str | int | bool],
) -> StaticHttpsJobConfiguration:
    """Parse a static job's exact allowlist or raise Pydantic's validation error."""

    return StaticHttpsJobConfiguration.model_validate(configuration)


def parse_immport_authenticated_job_configuration(
    configuration: dict[str, str | int | bool],
) -> ImmportAuthenticatedJobConfiguration:
    """Parse the authenticated ImmPort collector's exact public allowlist."""

    return ImmportAuthenticatedJobConfiguration.model_validate(configuration)


type SupportedCollectorJobConfiguration = StaticHttpsJobConfiguration | ImmportAuthenticatedJobConfiguration


def parse_supported_collector_job_configuration(
    collector_id: str,
    configuration: dict[str, str | int | bool],
) -> SupportedCollectorJobConfiguration:
    """Typed dispatch for every collector admitted to Tier-A promotion."""

    if collector_id == STATIC_HTTPS_COLLECTOR_ID:
        return parse_static_job_configuration(configuration)
    if collector_id == IMMPORT_AUTHENTICATED_COLLECTOR_ID:
        return parse_immport_authenticated_job_configuration(configuration)
    raise ValueError(f'no semantic verifier is registered for collector {collector_id!r}')
