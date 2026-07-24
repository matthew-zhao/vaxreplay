"""Fail-closed deployment contract for the credential-bearing ImmPort producer.

The contract is an offline policy artifact.  It makes unsafe workload specifications
unrepresentable, but it does not implement or attest the external supervisor that must
obtain a scoped credential and exec the one-shot producer with that credential already
open as file descriptor 3.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal, Self

from pydantic import Field, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.immport_producer import IMMPORT_CREDENTIAL_FD
from vaxreplay.operations.schema import SAFE_ID_PATTERN
from vaxreplay.operations.supply_chain import (
    OciPlatform,
    SourceWorkerBuildProvenance,
    SourceWorkerSupplyChainReport,
)

IMMPORT_PRODUCER_WORKLOAD_SCHEMA_VERSION = 'vaxreplay.immport-producer-workload-policy.v0.1'
IMMPORT_PRODUCER_WORKLOAD_REPORT_SCHEMA_VERSION = 'vaxreplay.immport-producer-workload-report.v0.1'
IMMPORT_PRODUCER_ENVIRONMENT_SCHEMA_VERSION = 'vaxreplay.immport-producer-execution-environment.v0.1'
IMMPORT_PRODUCER_ENVIRONMENT_REPORT_SCHEMA_VERSION = 'vaxreplay.immport-producer-execution-environment-report.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_DIGEST_REF = re.compile(r'^.+@sha256:([0-9a-f]{64})$')
_MAX_WORKLOAD_POLICY_BYTES = 1024 * 1024
_MAX_EXECUTION_ENVIRONMENT_BYTES = 8 * 1024 * 1024
_PROXY_NAMES = (
    'ALL_PROXY',
    'HTTP_PROXY',
    'HTTPS_PROXY',
    'NO_PROXY',
    'all_proxy',
    'http_proxy',
    'https_proxy',
    'no_proxy',
)


class ImmportProducerDeploymentError(ValueError):
    """The producer workload policy is malformed or differs from precommitted identity."""


class ImmportDeploymentMaterial(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=1)


class ImmportProducerWorkloadPolicy(StrictModel):
    schema_version: Literal['vaxreplay.immport-producer-workload-policy.v0.1'] = (
        IMMPORT_PRODUCER_WORKLOAD_SCHEMA_VERSION
    )
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    image_ref: str = Field(min_length=1, max_length=2048)
    expected_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    platform: Literal['linux/amd64', 'linux/arm64']
    entrypoint: tuple[
        Literal['/usr/local/bin/python'],
        Literal['-I'],
        Literal['-m'],
        Literal['vaxreplay.operations.immport_producer_cli'],
    ] = (
        '/usr/local/bin/python',
        '-I',
        '-m',
        'vaxreplay.operations.immport_producer_cli',
    )
    credential_fd: Literal[3] = IMMPORT_CREDENTIAL_FD
    credential_delivery: Literal['deployment-supplied-supervisor-preopened-fd'] = (
        'deployment-supplied-supervisor-preopened-fd'
    )
    fd_broker_supervisor: ImmportDeploymentMaterial
    application_network_policy: ImmportDeploymentMaterial
    dns_resolver_policy: ImmportDeploymentMaterial
    tls_trust_bundle: ImmportDeploymentMaterial
    host_memory_policy: ImmportDeploymentMaterial
    repository_supplies_fd_broker_integration: Literal[False] = False
    fd_broker_handoff_evidence_required: Literal[True] = True
    credential_environment_allowed: Literal[False] = False
    credential_argv_allowed: Literal[False] = False
    credential_path_allowed: Literal[False] = False
    kubernetes_secret_projection_allowed: Literal[False] = False
    secret_volume_allowed: Literal[False] = False
    proxy_environment_allowed: Literal[False] = False
    forbidden_proxy_environment_names: tuple[str, ...] = _PROXY_NAMES
    runtime_environment: tuple[()] = ()
    application_egress_allowlist: tuple[Literal['www.immport.org:443']] = ('www.immport.org:443',)
    dns_egress_policy: Literal['deployment-pinned-recursive-resolver-only'] = (
        'deployment-pinned-recursive-resolver-only'
    )
    redirects_allowed: Literal[False] = False
    service_account_token_automount: Literal[False] = False
    enable_service_links: Literal[False] = False
    ambient_environment_injection_allowed: Literal[False] = False
    run_as_user: Literal[65532] = 65532
    run_as_group: Literal[65532] = 65532
    run_as_non_root: Literal[True] = True
    read_only_root_filesystem: Literal[True] = True
    allow_privilege_escalation: Literal[False] = False
    capabilities_drop: tuple[Literal['ALL']] = ('ALL',)
    seccomp_profile: Literal['RuntimeDefault'] = 'RuntimeDefault'
    core_dumps_enabled: Literal[False] = False
    process_dumpable: Literal[False] = False
    ptrace_allowed: Literal[False] = False
    swap_policy: Literal['encrypted-or-disabled'] = 'encrypted-or-disabled'
    stdin_public_request_only: Literal[True] = True
    stdout_sanitized_response_only: Literal[True] = True
    restart_policy: Literal['Never'] = 'Never'
    one_shot_process: Literal[True] = True
    plan_panel_deadline_seconds: int = Field(ge=9, le=60 * 60)
    supervisor_hard_deadline_seconds: int = Field(ge=10, le=60 * 60 + 300)

    @model_validator(mode='after')
    def validate_identity_and_deadline(self) -> Self:
        image_match = _DIGEST_REF.fullmatch(self.image_ref)
        if image_match is None:
            raise ValueError('producer image_ref must be a named SHA-256 digest reference')
        margin = self.supervisor_hard_deadline_seconds - self.plan_panel_deadline_seconds
        if margin < 1 or margin > 300:
            raise ValueError('supervisor hard deadline must be 1-300 seconds above the panel deadline')
        if self.forbidden_proxy_environment_names != _PROXY_NAMES:
            raise ValueError('producer workload must forbid the complete fixed proxy environment set')
        return self


class ImmportProducerExecutionEnvironment(StrictModel):
    """Canonical composite identity committed by the ImmPort job configuration.

    The envelope intentionally does not claim that the image was observed running or
    that the retained source/SBOM/layers were reverified at slot execution time.  It
    embeds the canonical output of that separate supply-chain verification and the
    canonical build provenance so their identities cannot drift from the workload.
    """

    schema_version: Literal['vaxreplay.immport-producer-execution-environment.v0.1'] = (
        IMMPORT_PRODUCER_ENVIRONMENT_SCHEMA_VERSION
    )
    environment_id: str = Field(pattern=SAFE_ID_PATTERN)
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    workload_policy: ImmportDeploymentMaterial
    image_ref: str = Field(min_length=1, max_length=2048)
    expected_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    platform: OciPlatform
    entrypoint: tuple[
        Literal['/usr/local/bin/python'],
        Literal['-I'],
        Literal['-m'],
        Literal['vaxreplay.operations.immport_producer_cli'],
    ] = (
        '/usr/local/bin/python',
        '-I',
        '-m',
        'vaxreplay.operations.immport_producer_cli',
    )
    supply_chain_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    supply_chain_report: SourceWorkerSupplyChainReport
    build_provenance: SourceWorkerBuildProvenance

    @model_validator(mode='after')
    def validate_composite_identity(self) -> Self:
        report_bytes = canonical_json_bytes(self.supply_chain_report)
        provenance_bytes = canonical_json_bytes(self.build_provenance)
        if hashlib.sha256(report_bytes).hexdigest() != self.supply_chain_report_sha256:
            raise ValueError('embedded producer supply-chain report differs from its digest')
        if hashlib.sha256(provenance_bytes).hexdigest() != self.build_provenance_sha256:
            raise ValueError('embedded producer build provenance differs from its digest')
        if (
            self.supply_chain_report.worker_name != 'immport-authenticated-producer'
            or self.build_provenance.worker_name != 'immport-authenticated-producer'
        ):
            raise ValueError('execution environment must bind the authenticated ImmPort producer')
        if self.supply_chain_report.provenance_sha256 != self.build_provenance_sha256:
            raise ValueError('producer supply-chain report binds different build provenance')
        report_build_identity = (
            self.supply_chain_report.provenance_id,
            self.supply_chain_report.source_archive_sha256,
            self.supply_chain_report.primary_package_sha256,
            self.supply_chain_report.runtime_lock_sha256,
            self.supply_chain_report.build_recipe_sha256,
            self.supply_chain_report.sbom_sha256,
        )
        provenance_build_identity = (
            self.build_provenance.provenance_id,
            self.build_provenance.source_archive.sha256,
            self.build_provenance.primary_package.sha256,
            self.build_provenance.runtime_lock.sha256,
            self.build_provenance.build_recipe.sha256,
            self.build_provenance.sbom.sha256,
        )
        if report_build_identity != provenance_build_identity:
            raise ValueError('producer supply-chain report and provenance bind different build materials')
        expected_identity = (
            self.image_ref,
            self.expected_image_id,
            self.platform,
        )
        if expected_identity != (
            self.supply_chain_report.image_ref,
            self.supply_chain_report.resolved_image_id,
            self.supply_chain_report.platform,
        ) or expected_identity != (
            self.build_provenance.image_ref,
            self.build_provenance.resolved_image_id,
            self.build_provenance.platform,
        ):
            raise ValueError('producer envelope, supply-chain report, and provenance identify different images')
        if self.entrypoint != self.build_provenance.entrypoint:
            raise ValueError('producer envelope and provenance bind different entrypoints')
        if self.collector_implementation_sha256 != self.build_provenance.implementation_sha256:
            raise ValueError('producer envelope and provenance bind different implementations')
        return self


class ImmportProducerWorkloadVerificationReport(StrictModel):
    schema_version: Literal['vaxreplay.immport-producer-workload-report.v0.1'] = (
        IMMPORT_PRODUCER_WORKLOAD_REPORT_SCHEMA_VERSION
    )
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    image_ref: str = Field(min_length=1, max_length=2048)
    expected_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    credential_fd: Literal[3] = 3
    application_egress_policy: Literal['www.immport.org:443'] = 'www.immport.org:443'
    hard_deadline_seconds: int = Field(ge=10, le=60 * 60 + 300)
    control_artifact_bytes_verified: Literal[True] = True
    unsafe_secret_channels_forbidden_by_policy: Literal[True] = True
    proxy_environment_forbidden_by_policy: Literal[True] = True
    non_root_read_only_profile_required_by_policy: Literal[True] = True
    memory_remanence_controls_required_by_policy: Literal[True] = True
    fd_broker_integration_verified: Literal[False] = False
    network_policy_enforcement_verified: Literal[False] = False
    host_swap_and_dump_controls_verified: Literal[False] = False
    runtime_security_context_enforcement_verified: Literal[False] = False
    ambient_environment_enforcement_verified: Literal[False] = False


class ImmportProducerExecutionEnvironmentVerificationReport(StrictModel):
    schema_version: Literal['vaxreplay.immport-producer-execution-environment-report.v0.1'] = (
        IMMPORT_PRODUCER_ENVIRONMENT_REPORT_SCHEMA_VERSION
    )
    environment_id: str = Field(pattern=SAFE_ID_PATTERN)
    execution_environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    workload_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    supply_chain_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    collector_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    image_ref: str = Field(min_length=1, max_length=2048)
    expected_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    platform: OciPlatform
    entrypoint: tuple[
        Literal['/usr/local/bin/python'],
        Literal['-I'],
        Literal['-m'],
        Literal['vaxreplay.operations.immport_producer_cli'],
    ]
    exact_environment_precommit_verified: Literal[True] = True
    exact_workload_precommit_verified: Literal[True] = True
    embedded_supply_chain_identity_verified: Literal[True] = True
    retained_build_materials_reverified: Literal[False] = False
    external_builder_identity_verified: Literal[False] = False
    runtime_image_observed: Literal[False] = False


def parse_immport_producer_execution_environment(
    environment_bytes: bytes,
    *,
    expected_environment_sha256: str,
) -> ImmportProducerExecutionEnvironment:
    """Load the exact canonical envelope selected by a job's environment digest."""

    environment = _parse_canonical_model(
        environment_bytes,
        ImmportProducerExecutionEnvironment,
        'producer execution environment',
        maximum=_MAX_EXECUTION_ENVIRONMENT_BYTES,
    )
    if (
        re.fullmatch(_SHA256_PATTERN, expected_environment_sha256) is None
        or hashlib.sha256(environment_bytes).hexdigest() != expected_environment_sha256
    ):
        raise ImmportProducerDeploymentError('producer execution environment differs from its registered digest')
    return environment


def verify_immport_producer_execution_environment(
    environment_bytes: bytes,
    *,
    expected_environment_sha256: str,
    workload_policy_bytes: bytes,
    expected_workload_policy_sha256: str,
    expected_collector_implementation_sha256: str,
) -> ImmportProducerExecutionEnvironmentVerificationReport:
    """Cross-bind the registered composite environment to one separately pinned workload.

    ``expected_environment_sha256`` is the immutable job configuration value.
    ``expected_workload_policy_sha256`` must arrive through an independent operational
    trust channel; accepting a policy hash from the environment itself would allow an
    attacker to replace both policy and image-control declarations together.
    """

    environment = parse_immport_producer_execution_environment(
        environment_bytes,
        expected_environment_sha256=expected_environment_sha256,
    )
    workload = _parse_canonical_model(
        workload_policy_bytes,
        ImmportProducerWorkloadPolicy,
        'producer workload policy',
        maximum=_MAX_WORKLOAD_POLICY_BYTES,
    )
    workload_digest = hashlib.sha256(workload_policy_bytes).hexdigest()
    if (
        re.fullmatch(_SHA256_PATTERN, expected_workload_policy_sha256) is None
        or workload_digest != expected_workload_policy_sha256
    ):
        raise ImmportProducerDeploymentError('producer workload policy differs from its out-of-band digest')
    if (
        re.fullmatch(_SHA256_PATTERN, expected_collector_implementation_sha256) is None
        or environment.collector_implementation_sha256 != expected_collector_implementation_sha256
    ):
        raise ImmportProducerDeploymentError(
            'producer execution environment differs from the registered implementation'
        )
    if environment.workload_policy.sha256 != workload_digest or environment.workload_policy.byte_count != len(
        workload_policy_bytes
    ):
        raise ImmportProducerDeploymentError('producer workload policy differs from the execution-environment binding')
    if (
        workload.image_ref != environment.image_ref
        or workload.expected_image_id != environment.expected_image_id
        or workload.platform != environment.platform
        or workload.entrypoint != environment.entrypoint
    ):
        raise ImmportProducerDeploymentError(
            'producer workload and execution environment identify different runtime images'
        )
    return ImmportProducerExecutionEnvironmentVerificationReport(
        environment_id=environment.environment_id,
        execution_environment_sha256=hashlib.sha256(environment_bytes).hexdigest(),
        workload_policy_sha256=workload_digest,
        supply_chain_report_sha256=environment.supply_chain_report_sha256,
        build_provenance_sha256=environment.build_provenance_sha256,
        collector_implementation_sha256=environment.collector_implementation_sha256,
        image_ref=environment.image_ref,
        expected_image_id=environment.expected_image_id,
        platform=environment.platform,
        entrypoint=environment.entrypoint,
    )


def verify_immport_producer_workload_policy(
    policy_bytes: bytes,
    *,
    expected_policy_sha256: str,
    expected_image_ref: str,
    expected_image_id: str,
    expected_platform: Literal['linux/amd64', 'linux/arm64'],
    expected_plan_panel_deadline_seconds: int,
    fd_broker_supervisor_bytes: bytes,
    application_network_policy_bytes: bytes,
    dns_resolver_policy_bytes: bytes,
    tls_trust_bundle_bytes: bytes,
    host_memory_policy_bytes: bytes,
) -> ImmportProducerWorkloadVerificationReport:
    policy = _parse_canonical_model(
        policy_bytes,
        ImmportProducerWorkloadPolicy,
        'producer workload policy',
        maximum=_MAX_WORKLOAD_POLICY_BYTES,
    )
    if (
        re.fullmatch(_SHA256_PATTERN, expected_policy_sha256) is None
        or hashlib.sha256(policy_bytes).hexdigest() != expected_policy_sha256
    ):
        raise ImmportProducerDeploymentError('producer workload policy differs from its out-of-band digest')
    if (
        policy.image_ref != expected_image_ref
        or policy.expected_image_id != expected_image_id
        or policy.platform != expected_platform
        or policy.plan_panel_deadline_seconds != expected_plan_panel_deadline_seconds
    ):
        raise ImmportProducerDeploymentError(
            'producer workload policy differs from precommitted image, platform, or capture deadline'
        )
    for binding, payload, label in (
        (policy.fd_broker_supervisor, fd_broker_supervisor_bytes, 'FD broker supervisor'),
        (policy.application_network_policy, application_network_policy_bytes, 'application network policy'),
        (policy.dns_resolver_policy, dns_resolver_policy_bytes, 'DNS resolver policy'),
        (policy.tls_trust_bundle, tls_trust_bundle_bytes, 'TLS trust bundle'),
        (policy.host_memory_policy, host_memory_policy_bytes, 'host memory policy'),
    ):
        if (
            not isinstance(payload, bytes)
            or len(payload) != binding.byte_count
            or hashlib.sha256(payload).hexdigest() != binding.sha256
        ):
            raise ImmportProducerDeploymentError(f'{label} differs from its exact workload binding')
    return ImmportProducerWorkloadVerificationReport(
        policy_id=policy.policy_id,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        image_ref=policy.image_ref,
        expected_image_id=policy.expected_image_id,
        hard_deadline_seconds=policy.supervisor_hard_deadline_seconds,
    )


def _parse_canonical_model[ModelT: StrictModel](
    payload: bytes,
    model: type[ModelT],
    label: str,
    *,
    maximum: int,
) -> ModelT:
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise ImmportProducerDeploymentError(f'{label} must be bounded nonempty exact bytes')
    try:
        parsed = model.model_validate_json(payload)
    except ValueError as error:
        raise ImmportProducerDeploymentError(f'{label} does not match its strict schema') from error
    if payload != canonical_json_bytes(parsed):
        raise ImmportProducerDeploymentError(f'{label} must use exact canonical JSON bytes')
    return parsed
