"""Digest-pinned OCI execution and offline-verifiable callback receipts.

This module removes Python callback execution from the trusted promotion process.  A
source verifier or adapter is executed in a locally present, digest-pinned OCI image
with no network, no host mounts, a read-only root, an explicit seccomp profile, and
bounded resources.  Exact canonical request, response, and signed receipt bytes are
returned so a caller can preserve them in a release archive.

The receipt attests what the configured runner observed.  It is not remote attestation
of the host kernel, Docker daemon, image build, or signing-key custody.  A Tier A
deployment must operate those components as audited infrastructure and pin the public
receipt key outside the release being verified.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc
from vaxreplay.runner._process import BoundedProcessResult, run_bounded_process

OCI_ENVIRONMENT_SCHEMA_VERSION = 'vaxreplay.hermetic-oci-environment.v0.1'
SANDBOX_POLICY_SCHEMA_VERSION = 'vaxreplay.hermetic-sandbox-policy.v0.1'
EXECUTION_REQUEST_SCHEMA_VERSION = 'vaxreplay.hermetic-execution-request.v0.1'
EXECUTION_RESPONSE_SCHEMA_VERSION = 'vaxreplay.hermetic-execution-response.v0.1'
EXECUTION_ATTESTATION_SCHEMA_VERSION = 'vaxreplay.hermetic-execution-attestation.v0.1'
EXECUTION_RECEIPT_SCHEMA_VERSION = 'vaxreplay.hermetic-execution-receipt.v0.1'
WORKER_PROTOCOL = 'vaxreplay.hermetic-worker-stdio.v0.1'

IMPLEMENTATION_LABEL = 'org.vaxreplay.implementation-sha256'
_SIGNATURE_DOMAIN = b'vaxreplay.hermetic-execution-attestation.v0.1\x00'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_DIGEST_IMAGE_RE = re.compile(r'^.+@sha256:[0-9a-f]{64}$')
_ENV_NAME_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,127}$')
_MAX_QUERY_STDOUT = 4 * 1024 * 1024
_MAX_QUERY_STDERR = 256 * 1024

ExecutionPurpose = Literal['source_verifier', 'adapter']


class HermeticExecutionError(ValueError):
    """Hermetic execution, validation, cleanup, or offline verification failed closed."""


class HermeticOciEnvironment(StrictModel):
    """Canonical, precommittable identity of the callback execution environment."""

    schema_version: Literal['vaxreplay.hermetic-oci-environment.v0.1'] = OCI_ENVIRONMENT_SCHEMA_VERSION
    environment_id: str = Field(pattern=SAFE_ID_PATTERN)
    image_ref: str = Field(min_length=1, max_length=2048)
    expected_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    platform: Literal['linux/amd64', 'linux/arm64']
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)
    worker_protocol: Literal['vaxreplay.hermetic-worker-stdio.v0.1'] = WORKER_PROTOCOL
    implementation_label: Literal['org.vaxreplay.implementation-sha256'] = IMPLEMENTATION_LABEL

    @field_validator('image_ref')
    @classmethod
    def validate_image_ref(cls, value: str) -> str:
        if value != value.strip() or not _DIGEST_IMAGE_RE.fullmatch(value):
            raise ValueError('image_ref must be an exact sha256 digest reference')
        return value

    @field_validator('entrypoint')
    @classmethod
    def validate_entrypoint(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or '\x00' in item or len(item.encode('utf-8')) > 4096 for item in value):
            raise ValueError('entrypoint arguments must be nonempty, NUL-free, and bounded')
        return value


class HermeticSandboxPolicy(StrictModel):
    """Canonical resource, isolation, and receipt-authority policy."""

    schema_version: Literal['vaxreplay.hermetic-sandbox-policy.v0.1'] = SANDBOX_POLICY_SCHEMA_VERSION
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)
    seccomp_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    wall_seconds: int = Field(ge=1, le=3600)
    memory_mib: int = Field(ge=32, le=1024 * 1024)
    milli_cpus: int = Field(ge=100, le=64_000)
    pids: int = Field(ge=1, le=4096)
    scratch_mib: int = Field(ge=1, le=16 * 1024)
    shared_memory_mib: int = Field(default=16, ge=1, le=16 * 1024)
    open_files: int = Field(ge=16, le=65_536)
    max_input_bytes: int = Field(ge=1, le=512 * 1024 * 1024)
    max_callback_policy_bytes: int = Field(ge=1, le=64 * 1024 * 1024)
    max_output_bytes: int = Field(ge=1, le=512 * 1024 * 1024)
    max_worker_response_bytes: int = Field(ge=1024, le=1024 * 1024 * 1024)
    max_log_bytes: int = Field(ge=1, le=64 * 1024 * 1024)
    deterministic_environment: tuple[str, ...] = (
        'HOME=/nonexistent',
        'LANG=C.UTF-8',
        'LC_ALL=C.UTF-8',
        'PATH=/usr/local/bin:/usr/bin:/bin',
        'PYTHONHASHSEED=0',
        'SOURCE_DATE_EPOCH=0',
        'TZ=UTC',
        'VAXREPLAY_INPUT=stdin',
        'VAXREPLAY_OUTPUT=stdout',
        f'VAXREPLAY_WORKER_PROTOCOL={WORKER_PROTOCOL}',
    )
    network_disabled: Literal[True] = True
    read_only_root: Literal[True] = True
    no_host_mounts: Literal[True] = True
    non_root_user: Literal[True] = True
    all_capabilities_dropped: Literal[True] = True
    no_new_privileges: Literal[True] = True
    seccomp_required: Literal[True] = True
    resource_limits_required: Literal[True] = True
    fresh_container_per_invocation: Literal[True] = True

    @field_validator('deterministic_environment')
    @classmethod
    def validate_environment(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('deterministic environment entries must be sorted and unique')
        names: set[str] = set()
        for item in value:
            if '\x00' in item or '=' not in item:
                raise ValueError('deterministic environment entries must be NUL-free NAME=value strings')
            name, _separator, _setting = item.partition('=')
            if not _ENV_NAME_RE.fullmatch(name) or name in names:
                raise ValueError('deterministic environment variable names must be safe and unique')
            names.add(name)
        required = {'LANG', 'LC_ALL', 'PATH', 'TZ', 'VAXREPLAY_INPUT', 'VAXREPLAY_OUTPUT'}
        if not required.issubset(names):
            raise ValueError('deterministic environment omits a required worker setting')
        return value

    @model_validator(mode='after')
    def validate_response_bound(self) -> Self:
        minimum_response = ((self.max_output_bytes + 2) // 3) * 4 + 1024
        if self.max_worker_response_bytes < minimum_response:
            raise ValueError('max_worker_response_bytes cannot contain the configured base64 output')
        return self


class HermeticMaterialBinding(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)


class HermeticExecutionRequest(StrictModel):
    """Canonical stdin request understood by a digest-pinned callback worker."""

    schema_version: Literal['vaxreplay.hermetic-execution-request.v0.1'] = EXECUTION_REQUEST_SCHEMA_VERSION
    worker_protocol: Literal['vaxreplay.hermetic-worker-stdio.v0.1'] = WORKER_PROTOCOL
    invocation_id: str = Field(pattern=SAFE_ID_PATTERN)
    invocation_index: int = Field(ge=0)
    purpose: ExecutionPurpose
    implementation: HermeticMaterialBinding
    execution_environment: HermeticMaterialBinding
    callback_policy: HermeticMaterialBinding
    sandbox_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    input: HermeticMaterialBinding
    callback_policy_base64: str = Field(min_length=4)
    input_base64: str = Field(min_length=4)

    @field_validator('callback_policy_base64', 'input_base64')
    @classmethod
    def validate_base64(cls, value: str) -> str:
        _decode_canonical_base64(value)
        return value

    @model_validator(mode='after')
    def validate_embedded_bytes(self) -> Self:
        _require_binding(_decode_canonical_base64(self.callback_policy_base64), self.callback_policy, 'callback policy')
        _require_binding(_decode_canonical_base64(self.input_base64), self.input, 'input')
        return self


class HermeticExecutionResponse(StrictModel):
    """Canonical stdout response produced by the isolated worker."""

    schema_version: Literal['vaxreplay.hermetic-execution-response.v0.1'] = EXECUTION_RESPONSE_SCHEMA_VERSION
    worker_protocol: Literal['vaxreplay.hermetic-worker-stdio.v0.1'] = WORKER_PROTOCOL
    invocation_id: str = Field(pattern=SAFE_ID_PATTERN)
    invocation_index: int = Field(ge=0)
    purpose: ExecutionPurpose
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: Literal['ok'] = 'ok'
    output: HermeticMaterialBinding
    output_base64: str = Field(min_length=4)

    @field_validator('output_base64')
    @classmethod
    def validate_output_base64(cls, value: str) -> str:
        _decode_canonical_base64(value)
        return value

    @model_validator(mode='after')
    def validate_output_binding(self) -> Self:
        _require_binding(_decode_canonical_base64(self.output_base64), self.output, 'output')
        return self


class HermeticExecutionAttestation(StrictModel):
    """Security facts signed by the independently operated execution authority."""

    schema_version: Literal['vaxreplay.hermetic-execution-attestation.v0.1'] = EXECUTION_ATTESTATION_SCHEMA_VERSION
    receipt_id: str = Field(pattern=SAFE_ID_PATTERN)
    issued_at: datetime
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    signing_key_id: str = Field(pattern=SAFE_ID_PATTERN)
    invocation_id: str = Field(pattern=SAFE_ID_PATTERN)
    invocation_index: int = Field(ge=0)
    purpose: ExecutionPurpose
    request: HermeticMaterialBinding
    response: HermeticMaterialBinding
    output: HermeticMaterialBinding
    implementation: HermeticMaterialBinding
    execution_environment: HermeticMaterialBinding
    callback_policy: HermeticMaterialBinding
    sandbox_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    image_ref: str = Field(min_length=1, max_length=2048)
    resolved_image_id: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    image_inspection_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_id: Literal['docker-oci'] = 'docker-oci'
    runtime_version: str = Field(min_length=1, max_length=200)
    platform: Literal['linux/amd64', 'linux/arm64']
    exit_code: Literal[0] = 0
    duration_ms: int = Field(ge=0)
    network_disabled: Literal[True] = True
    read_only_root: Literal[True] = True
    no_host_mounts: Literal[True] = True
    input_via_stdin_only: Literal[True] = True
    deterministic_environment: Literal[True] = True
    resource_limits_enforced: Literal[True] = True
    explicit_seccomp_profile: Literal[True] = True
    cleanup_verified: Literal[True] = True

    @field_validator('issued_at')
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'hermetic receipt issued_at')

    @field_validator('image_ref')
    @classmethod
    def validate_image_ref(cls, value: str) -> str:
        if not _DIGEST_IMAGE_RE.fullmatch(value):
            raise ValueError('attested image_ref must be digest pinned')
        return value


class SignedHermeticExecutionReceipt(StrictModel):
    schema_version: Literal['vaxreplay.hermetic-execution-receipt.v0.1'] = EXECUTION_RECEIPT_SCHEMA_VERSION
    signature_algorithm: Literal['ed25519'] = 'ed25519'
    signed_payload_domain: Literal['vaxreplay.hermetic-execution-attestation.v0.1'] = (
        'vaxreplay.hermetic-execution-attestation.v0.1'
    )
    attestation: HermeticExecutionAttestation
    signature_base64: str = Field(min_length=88, max_length=88)

    @field_validator('signature_base64')
    @classmethod
    def validate_signature(cls, value: str) -> str:
        if len(_decode_canonical_base64(value)) != 64:
            raise ValueError('Ed25519 signatures must contain exactly 64 bytes')
        return value


class ReceiptSigner(Protocol):
    @property
    def key_id(self) -> str: ...

    @property
    def public_key_bytes(self) -> bytes: ...

    def sign(self, payload: bytes) -> bytes: ...


class Ed25519ReceiptSigner:
    """Ed25519 receipt signer; production should back this interface with a KMS/HSM."""

    def __init__(self, *, key_id: str, private_key_bytes: bytes):
        if not re.fullmatch(SAFE_ID_PATTERN, key_id):
            raise ValueError('key_id is invalid')
        if not isinstance(private_key_bytes, bytes) or len(private_key_bytes) != 32:
            raise ValueError('Ed25519 private key seed must contain exactly 32 bytes')
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError as error:  # pragma: no cover - exercised only in minimal installations
            raise RuntimeError('cryptography is required for Ed25519 execution receipts') from error
        self._key_id = key_id
        self._key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        self._public_key_bytes = self._key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key_bytes(self) -> bytes:
        return self._public_key_bytes

    def sign(self, payload: bytes) -> bytes:
        return self._key.sign(payload)


@dataclass(frozen=True)
class HermeticCallbackMaterials:
    implementation_bytes: bytes
    execution_environment_bytes: bytes
    callback_policy_bytes: bytes


@dataclass(frozen=True)
class HermeticExecutionBundle:
    """Exact bytes suitable for embedding unchanged in a promotion archive."""

    request: HermeticExecutionRequest
    request_bytes: bytes
    response: HermeticExecutionResponse
    response_bytes: bytes
    receipt: SignedHermeticExecutionReceipt
    receipt_bytes: bytes
    output_bytes: bytes
    image_inspection_bytes: bytes


class HermeticCallbackExecutor(Protocol):
    def execute(
        self,
        *,
        purpose: ExecutionPurpose,
        invocation_id: str,
        invocation_index: int,
        input_bytes: bytes,
        materials: HermeticCallbackMaterials,
    ) -> HermeticExecutionBundle: ...


class OciHermeticCallbackExecutor:
    """Execute one callback in a fresh digest-pinned Docker OCI container."""

    def __init__(
        self,
        *,
        sandbox_policy: HermeticSandboxPolicy,
        seccomp_profile_bytes: bytes,
        signer: ReceiptSigner,
        runtime: str = 'docker',
        clock: Callable[[], datetime] | None = None,
    ):
        self._runtime = _resolve_docker(runtime)
        self._sandbox_policy = _validate_model_bytes(
            canonical_json_bytes(sandbox_policy), HermeticSandboxPolicy, 'sandbox policy'
        )
        if not isinstance(seccomp_profile_bytes, bytes) or not seccomp_profile_bytes:
            raise ValueError('seccomp profile bytes must be nonempty')
        if _sha256(seccomp_profile_bytes) != sandbox_policy.seccomp_profile_sha256:
            raise ValueError('seccomp profile bytes differ from the sandbox policy')
        if signer.key_id != sandbox_policy.signing_key_id:
            raise ValueError('receipt signer key ID differs from the sandbox policy')
        if _sha256(signer.public_key_bytes) != sandbox_policy.signing_public_key_sha256:
            raise ValueError('receipt signer public key differs from the sandbox policy')
        self._seccomp_profile_bytes = bytes(seccomp_profile_bytes)
        self._signer = signer
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def sandbox_policy_bytes(self) -> bytes:
        return canonical_json_bytes(self._sandbox_policy)

    @property
    def seccomp_profile_bytes(self) -> bytes:
        return self._seccomp_profile_bytes

    def execute(
        self,
        *,
        purpose: ExecutionPurpose,
        invocation_id: str,
        invocation_index: int,
        input_bytes: bytes,
        materials: HermeticCallbackMaterials,
    ) -> HermeticExecutionBundle:
        policy = self._sandbox_policy
        _require_nonempty_bounded(input_bytes, policy.max_input_bytes, 'input')
        _require_nonempty_bounded(materials.callback_policy_bytes, policy.max_callback_policy_bytes, 'callback policy')
        _require_nonempty_bounded(materials.implementation_bytes, 512 * 1024 * 1024, 'implementation')
        _require_nonempty_bounded(materials.execution_environment_bytes, 16 * 1024 * 1024, 'execution environment')
        environment = _validate_model_bytes(
            materials.execution_environment_bytes,
            HermeticOciEnvironment,
            'execution environment',
        )
        request = HermeticExecutionRequest(
            invocation_id=invocation_id,
            invocation_index=invocation_index,
            purpose=purpose,
            implementation=_binding(materials.implementation_bytes),
            execution_environment=_binding(materials.execution_environment_bytes),
            callback_policy=_binding(materials.callback_policy_bytes),
            sandbox_policy_sha256=_sha256(self.sandbox_policy_bytes),
            input=_binding(input_bytes),
            callback_policy_base64=_encode_base64(materials.callback_policy_bytes),
            input_base64=_encode_base64(input_bytes),
        )
        request_bytes = canonical_json_bytes(request)
        runtime_version, inspection_bytes = self._preflight(environment, request.implementation.sha256)

        container_name = f'vaxreplay-hermetic-{uuid.uuid4().hex}'
        with tempfile.TemporaryDirectory(prefix='vaxreplay-seccomp-') as temp_dir:
            seccomp_path = Path(temp_dir) / 'profile.json'
            _write_private_file(seccomp_path, self._seccomp_profile_bytes)
            argv = build_hermetic_docker_argv(
                runtime=self._runtime,
                container_name=container_name,
                environment=environment,
                policy=policy,
                seccomp_profile_path=seccomp_path,
            )
            try:
                container_id = self._query(tuple(argv[1:]), label='cannot create hermetic callback container')
                _validate_container_id(container_id.decode('ascii', errors='strict').strip())
                result = run_bounded_process(
                    (self._runtime, 'start', '--attach', '--interactive', container_name),
                    input_bytes=request_bytes,
                    wall_seconds=policy.wall_seconds,
                    max_stdout_bytes=policy.max_worker_response_bytes,
                    max_stderr_bytes=policy.max_log_bytes,
                    on_abort=lambda: self._request_remove(container_name),
                    env=_minimal_host_environment(),
                )
            except (OSError, RuntimeError, UnicodeError, HermeticExecutionError) as error:
                if not self._cleanup(container_name):
                    raise HermeticExecutionError('cannot prove failed hermetic container cleanup') from error
                raise HermeticExecutionError('hermetic callback runtime failed closed') from error
            if not self._cleanup(container_name):
                raise HermeticExecutionError('cannot prove hermetic callback container cleanup')

        response, response_bytes, output_bytes = _validate_worker_result(result, request, policy)
        try:
            issued_at = aware_utc(self._clock(), 'hermetic execution authority clock')
        except (AttributeError, TypeError, ValueError) as error:
            raise HermeticExecutionError('hermetic execution authority clock is invalid') from error
        attestation = HermeticExecutionAttestation(
            receipt_id=f'execution-{uuid.uuid4().hex}',
            issued_at=issued_at,
            authority_id=policy.authority_id,
            signing_key_id=policy.signing_key_id,
            invocation_id=request.invocation_id,
            invocation_index=request.invocation_index,
            purpose=request.purpose,
            request=_binding(request_bytes),
            response=_binding(response_bytes),
            output=_binding(output_bytes),
            implementation=request.implementation,
            execution_environment=request.execution_environment,
            callback_policy=request.callback_policy,
            sandbox_policy_sha256=request.sandbox_policy_sha256,
            image_ref=environment.image_ref,
            resolved_image_id=environment.expected_image_id,
            image_inspection_sha256=_sha256(inspection_bytes),
            runtime_version=runtime_version,
            platform=environment.platform,
            duration_ms=result.duration_ms,
        )
        signature = self._signer.sign(_signed_attestation_bytes(attestation))
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise HermeticExecutionError('receipt signer returned an invalid Ed25519 signature')
        receipt = SignedHermeticExecutionReceipt(
            attestation=attestation,
            signature_base64=_encode_base64(signature),
        )
        receipt_bytes = canonical_json_bytes(receipt)
        return verify_hermetic_execution_bundle(
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            receipt_bytes=receipt_bytes,
            expected_materials=materials,
            expected_sandbox_policy_bytes=self.sandbox_policy_bytes,
            expected_seccomp_profile_bytes=self.seccomp_profile_bytes,
            trusted_public_key_bytes=self._signer.public_key_bytes,
            image_inspection_bytes=inspection_bytes,
        )

    def _preflight(self, environment: HermeticOciEnvironment, implementation_sha256: str) -> tuple[str, bytes]:
        version_bytes = self._query(
            ('version', '--format', '{{.Server.Os}}|{{.Server.Version}}'),
            label='cannot query Docker server version',
        )
        try:
            server_os, runtime_version = version_bytes.decode('utf-8', errors='strict').strip().split('|', 1)
        except (UnicodeError, ValueError) as error:
            raise HermeticExecutionError('Docker returned an invalid server version') from error
        if server_os != 'linux' or not runtime_version:
            raise HermeticExecutionError('hermetic execution requires a reachable Linux Docker server')
        inspection_bytes = self._query(
            ('image', 'inspect', '--format', '{{json .}}', environment.image_ref),
            label='cannot inspect the locally present digest-pinned callback image',
        )
        inspection = _parse_inspection(inspection_bytes)
        _validate_inspection_bindings(
            inspection,
            environment=environment,
            implementation_sha256=implementation_sha256,
        )
        canonical_inspection = canonical_json_bytes(inspection)
        return runtime_version, canonical_inspection

    def _query(self, arguments: Sequence[str], *, label: str) -> bytes:
        try:
            result = run_bounded_process(
                (self._runtime, *arguments),
                input_bytes=b'',
                wall_seconds=30,
                max_stdout_bytes=_MAX_QUERY_STDOUT,
                max_stderr_bytes=_MAX_QUERY_STDERR,
                on_abort=lambda: None,
                env=_minimal_host_environment(),
            )
        except OSError as error:
            raise HermeticExecutionError(f'{label}: {error}') from error
        if result.termination != 'exited' or result.exit_code != 0:
            detail = result.stderr.decode('utf-8', errors='replace').strip()[:500]
            raise HermeticExecutionError(f'{label}: {detail or result.termination}')
        return result.stdout

    def _request_remove(self, container_name: str) -> None:
        try:
            subprocess.run(  # noqa: S603 - trusted absolute Docker argv, never a shell
                (self._runtime, 'rm', '--force', container_name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                close_fds=True,
                shell=False,
                env=_minimal_host_environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _cleanup(self, container_name: str) -> bool:
        self._request_remove(container_name)
        try:
            remaining = self._query(
                ('ps', '--all', '--quiet', '--filter', f'name=^{container_name}$'),
                label='cannot verify hermetic container cleanup',
            )
        except HermeticExecutionError:
            return False
        return not remaining.strip()


def build_hermetic_docker_argv(
    *,
    runtime: str,
    container_name: str,
    environment: HermeticOciEnvironment,
    policy: HermeticSandboxPolicy,
    seccomp_profile_path: Path,
) -> tuple[str, ...]:
    """Return the complete no-mount Docker create argv for audit and testing."""

    if not runtime or '\x00' in runtime or not container_name or '\x00' in container_name:
        raise ValueError('runtime and container name must be nonempty and NUL-free')
    if not seccomp_profile_path.is_absolute() or not seccomp_profile_path.is_file():
        raise ValueError('seccomp profile must be an existing absolute regular file')
    argv = [
        runtime,
        'create',
        '--interactive',
        '--name',
        container_name,
        '--hostname',
        'vaxreplay-hermetic',
        '--pull',
        'never',
        '--platform',
        environment.platform,
        '--network',
        'none',
        '--read-only',
        '--cap-drop',
        'ALL',
        '--security-opt',
        'no-new-privileges:true',
        '--security-opt',
        f'seccomp={seccomp_profile_path}',
        '--user',
        '65532:65532',
        '--pids-limit',
        str(policy.pids),
        '--memory',
        f'{policy.memory_mib}m',
        '--memory-swap',
        f'{policy.memory_mib}m',
        '--cpus',
        _format_cpus(policy.milli_cpus),
        '--ipc',
        'private',
        '--shm-size',
        f'{policy.shared_memory_mib}m',
        '--cgroupns',
        'private',
        '--tmpfs',
        f'/tmp:rw,noexec,nosuid,nodev,size={policy.scratch_mib}m',
        '--workdir',
        '/tmp',
        '--ulimit',
        f'nofile={policy.open_files}:{policy.open_files}',
        '--ulimit',
        'core=0:0',
        '--no-healthcheck',
        '--log-driver',
        'none',
        '--init',
    ]
    for item in policy.deterministic_environment:
        argv.extend(('--env', item))
    argv.extend(('--entrypoint', environment.entrypoint[0], environment.expected_image_id, *environment.entrypoint[1:]))
    return tuple(argv)


def verify_hermetic_execution_bundle(
    *,
    request_bytes: bytes,
    response_bytes: bytes,
    receipt_bytes: bytes,
    expected_materials: HermeticCallbackMaterials,
    expected_sandbox_policy_bytes: bytes,
    expected_seccomp_profile_bytes: bytes,
    trusted_public_key_bytes: bytes,
    image_inspection_bytes: bytes,
) -> HermeticExecutionBundle:
    """Offline-verify exact bytes under separately pinned materials and public key."""

    request = _validate_model_bytes(request_bytes, HermeticExecutionRequest, 'execution request')
    response = _validate_model_bytes(response_bytes, HermeticExecutionResponse, 'execution response')
    receipt = _validate_model_bytes(receipt_bytes, SignedHermeticExecutionReceipt, 'execution receipt')
    sandbox_policy = _validate_model_bytes(
        expected_sandbox_policy_bytes, HermeticSandboxPolicy, 'expected sandbox policy'
    )
    environment = _validate_model_bytes(
        expected_materials.execution_environment_bytes,
        HermeticOciEnvironment,
        'expected execution environment',
    )
    if len(trusted_public_key_bytes) != 32:
        raise HermeticExecutionError('trusted Ed25519 public key must contain exactly 32 bytes')
    if _sha256(trusted_public_key_bytes) != sandbox_policy.signing_public_key_sha256:
        raise HermeticExecutionError('trusted receipt public key differs from the sandbox policy')
    if not isinstance(expected_seccomp_profile_bytes, bytes) or not expected_seccomp_profile_bytes:
        raise HermeticExecutionError('expected seccomp profile bytes must be nonempty exact bytes')
    if _sha256(expected_seccomp_profile_bytes) != sandbox_policy.seccomp_profile_sha256:
        raise HermeticExecutionError('expected seccomp profile differs from the sandbox policy')
    if receipt.attestation.authority_id != sandbox_policy.authority_id:
        raise HermeticExecutionError('execution receipt authority differs from the sandbox policy')
    if receipt.attestation.signing_key_id != sandbox_policy.signing_key_id:
        raise HermeticExecutionError('execution receipt key differs from the sandbox policy')
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:  # pragma: no cover - exercised only in minimal installations
        raise RuntimeError('cryptography is required to verify Ed25519 execution receipts') from error
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key_bytes).verify(
            _decode_canonical_base64(receipt.signature_base64),
            _signed_attestation_bytes(receipt.attestation),
        )
    except (InvalidSignature, ValueError) as error:
        raise HermeticExecutionError('execution receipt signature verification failed') from error

    expected = (
        (request.implementation, expected_materials.implementation_bytes, 'implementation'),
        (request.execution_environment, expected_materials.execution_environment_bytes, 'execution environment'),
        (request.callback_policy, expected_materials.callback_policy_bytes, 'callback policy'),
    )
    for binding, payload, label in expected:
        _require_binding(payload, binding, label)
    if len(_decode_canonical_base64(request.input_base64)) > sandbox_policy.max_input_bytes:
        raise HermeticExecutionError('execution request input exceeds the expected sandbox policy')
    if len(_decode_canonical_base64(request.callback_policy_base64)) > sandbox_policy.max_callback_policy_bytes:
        raise HermeticExecutionError('execution request callback policy exceeds the expected sandbox policy')
    if len(response_bytes) > sandbox_policy.max_worker_response_bytes:
        raise HermeticExecutionError('execution response exceeds the expected sandbox policy')
    if request.sandbox_policy_sha256 != _sha256(expected_sandbox_policy_bytes):
        raise HermeticExecutionError('execution request differs from the expected sandbox policy')
    if response.request_sha256 != _sha256(request_bytes):
        raise HermeticExecutionError('worker response binds a different request')
    if (
        response.invocation_id != request.invocation_id
        or response.invocation_index != request.invocation_index
        or response.purpose != request.purpose
    ):
        raise HermeticExecutionError('worker response invocation identity differs from its request')
    output_bytes = _decode_canonical_base64(response.output_base64)
    if len(output_bytes) > sandbox_policy.max_output_bytes:
        raise HermeticExecutionError('execution output exceeds the expected sandbox policy')
    attestation = receipt.attestation
    for binding, payload, label in (
        (attestation.request, request_bytes, 'attested request'),
        (attestation.response, response_bytes, 'attested response'),
        (attestation.output, output_bytes, 'attested output'),
    ):
        _require_binding(payload, binding, label)
    if (
        attestation.invocation_id != request.invocation_id
        or attestation.invocation_index != request.invocation_index
        or attestation.purpose != request.purpose
        or attestation.implementation != request.implementation
        or attestation.execution_environment != request.execution_environment
        or attestation.callback_policy != request.callback_policy
        or attestation.sandbox_policy_sha256 != request.sandbox_policy_sha256
    ):
        raise HermeticExecutionError('execution receipt differs from the exact request materials or invocation')
    if (
        attestation.image_ref != environment.image_ref
        or attestation.resolved_image_id != environment.expected_image_id
        or attestation.platform != environment.platform
    ):
        raise HermeticExecutionError('execution receipt differs from the expected OCI environment')
    canonical_inspection = canonical_json_bytes(_parse_inspection(image_inspection_bytes))
    if canonical_inspection != image_inspection_bytes:
        raise HermeticExecutionError('image inspection must use exact canonical JSON bytes')
    _validate_inspection_bindings(
        _parse_inspection(canonical_inspection),
        environment=environment,
        implementation_sha256=request.implementation.sha256,
    )
    if _sha256(canonical_inspection) != attestation.image_inspection_sha256:
        raise HermeticExecutionError('execution receipt differs from exact image inspection bytes')
    return HermeticExecutionBundle(
        request=request,
        request_bytes=request_bytes,
        response=response,
        response_bytes=response_bytes,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        output_bytes=output_bytes,
        image_inspection_bytes=canonical_inspection,
    )


def parse_hermetic_worker_request(request_bytes: bytes) -> HermeticExecutionRequest:
    """Worker-side strict parser for one exact canonical stdin request."""

    return _validate_model_bytes(request_bytes, HermeticExecutionRequest, 'execution request')


def build_hermetic_worker_response(request_bytes: bytes, output_bytes: bytes) -> bytes:
    """Worker-side helper for emitting one exact canonical successful stdout response."""

    request = parse_hermetic_worker_request(request_bytes)
    if not isinstance(output_bytes, bytes) or not output_bytes:
        raise ValueError('worker output bytes must be nonempty')
    return canonical_json_bytes(
        HermeticExecutionResponse(
            invocation_id=request.invocation_id,
            invocation_index=request.invocation_index,
            purpose=request.purpose,
            request_sha256=_sha256(request_bytes),
            output=_binding(output_bytes),
            output_base64=_encode_base64(output_bytes),
        )
    )


def _validate_worker_result(
    result: BoundedProcessResult,
    request: HermeticExecutionRequest,
    policy: HermeticSandboxPolicy,
) -> tuple[HermeticExecutionResponse, bytes, bytes]:
    if result.termination != 'exited':
        raise HermeticExecutionError(f'hermetic worker did not exit normally: {result.termination}')
    if result.exit_code != 0:
        raise HermeticExecutionError(f'hermetic worker failed with exit code {result.exit_code}')
    response = _validate_model_bytes(result.stdout, HermeticExecutionResponse, 'worker response')
    response_bytes = canonical_json_bytes(response)
    if response.request_sha256 != _sha256(canonical_json_bytes(request)):
        raise HermeticExecutionError('worker response binds a different request')
    if (
        response.invocation_id != request.invocation_id
        or response.invocation_index != request.invocation_index
        or response.purpose != request.purpose
    ):
        raise HermeticExecutionError('worker response invocation differs from its request')
    output = _decode_canonical_base64(response.output_base64)
    if len(output) > policy.max_output_bytes:
        raise HermeticExecutionError('hermetic worker output exceeds the configured limit')
    return response, response_bytes, output


def _signed_attestation_bytes(attestation: HermeticExecutionAttestation) -> bytes:
    return _SIGNATURE_DOMAIN + canonical_json_bytes(attestation)


def _validate_model_bytes[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    if not isinstance(payload, bytes) or not payload:
        raise HermeticExecutionError(f'{label} bytes must be nonempty')
    try:
        value = model.model_validate_json(payload)
    except ValueError as error:
        raise HermeticExecutionError(f'{label} does not match its strict schema') from error
    if canonical_json_bytes(value) != payload:
        raise HermeticExecutionError(f'{label} must use exact canonical JSON bytes')
    return value


def _binding(payload: bytes) -> HermeticMaterialBinding:
    return HermeticMaterialBinding(sha256=_sha256(payload), byte_count=len(payload))


def _require_binding(payload: bytes, binding: HermeticMaterialBinding, label: str) -> None:
    if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
        raise HermeticExecutionError(f'{label} bytes differ from their exact material binding')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode('ascii')


def _decode_canonical_base64(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError('value must be canonical base64') from error
    if _encode_base64(decoded) != value:
        raise ValueError('value must be canonical padded base64')
    return decoded


def _parse_inspection(payload: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HermeticExecutionError('Docker returned invalid image inspection JSON') from error
    if not isinstance(value, dict):
        raise HermeticExecutionError('Docker image inspection must be one JSON object')
    return value


def _validate_inspection_bindings(
    inspection: Mapping[str, object],
    *,
    environment: HermeticOciEnvironment,
    implementation_sha256: str,
) -> None:
    if inspection.get('Id') != environment.expected_image_id:
        raise HermeticExecutionError('resolved callback image ID differs from the precommitted environment')
    expected_os, expected_arch = environment.platform.split('/', 1)
    if inspection.get('Os') != expected_os or inspection.get('Architecture') != expected_arch:
        raise HermeticExecutionError('resolved callback image platform differs from the precommitted environment')
    repo_digests = inspection.get('RepoDigests')
    if not isinstance(repo_digests, list) or environment.image_ref not in repo_digests:
        raise HermeticExecutionError('Docker did not prove the requested digest reference belongs to the image')
    config = inspection.get('Config')
    if not isinstance(config, dict):
        raise HermeticExecutionError('Docker image inspection omitted configuration')
    if config.get('Volumes') not in (None, {}):
        raise HermeticExecutionError('hermetic callback images cannot declare volumes')
    if config.get('Cmd') not in (None, []):
        raise HermeticExecutionError('hermetic callback images cannot declare an implicit command')
    if config.get('Env') not in (None, []):
        raise HermeticExecutionError('hermetic callback images cannot declare ambient environment variables')
    labels = config.get('Labels')
    if not isinstance(labels, dict) or labels.get(environment.implementation_label) != implementation_sha256:
        raise HermeticExecutionError('callback image metadata label differs from the reviewed implementation digest')


def _write_private_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short seccomp profile write')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_nonempty_bounded(payload: bytes, maximum: int, label: str) -> None:
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise ValueError(f'{label} bytes must be nonempty and at most {maximum} bytes')


def _resolve_docker(runtime: str) -> str:
    if not runtime or '\x00' in runtime:
        raise ValueError('runtime must be a nonempty NUL-free path or command')
    resolved = shutil.which(runtime)
    if resolved is None:
        raise HermeticExecutionError(f'cannot find Docker runtime: {runtime}')
    path = str(Path(resolved).resolve())
    if Path(path).name != 'docker':
        raise HermeticExecutionError('hermetic OCI execution currently supports only the Docker CLI')
    return path


def _validate_container_id(value: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise HermeticExecutionError('Docker create returned an invalid container ID')


def _format_cpus(milli_cpus: int) -> str:
    whole, fraction = divmod(milli_cpus, 1000)
    return str(whole) if fraction == 0 else f'{whole}.{fraction:03d}'.rstrip('0')


def _minimal_host_environment() -> dict[str, str]:
    return {'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'TZ': 'UTC'}
