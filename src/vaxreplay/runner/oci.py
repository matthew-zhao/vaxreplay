"""Hardened OCI development backend with explicit non-official capability claims."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from vaxreplay.runner._process import run_bounded_process
from vaxreplay.runner.backend import (
    BackendPolicyError,
    IsolationCleanupError,
    PreparedBackend,
    RawExecutionResult,
    RawExecutionStatus,
)
from vaxreplay.runner.schema import (
    BackendCapabilities,
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)

_QUERY_STDOUT_LIMIT = 1024 * 1024
_QUERY_STDERR_LIMIT = 128 * 1024


class OciDevelopmentBackend:
    """Run one digest-pinned system per episode using Docker isolation controls.

    This backend is intentionally classified as ``development``. An official benchmark must plug
    in a backend backed by a dedicated ephemeral microVM or an equivalently audited hostile-code
    boundary. Policy negotiation never silently upgrades this backend's claim.
    """

    def __init__(self, runtime: str = 'docker'):
        self._runtime = _resolve_runtime(runtime)

    def prepare(self, system: SystemSubmissionManifest, policy: RunnerPolicy) -> PreparedBackend:
        if policy.required_isolation != IsolationTier.DEVELOPMENT:
            raise BackendPolicyError(
                'the OCI backend is development-only; official policy requires a dedicated strong sandbox backend'
            )
        version_output = (
            self._query_runtime(
                ('version', '--format', '{{.Server.Os}}|{{.Server.Version}}'),
                error_label='cannot query Docker server version',
            )
            .decode('utf-8', errors='strict')
            .strip()
        )
        try:
            server_os, server_version = version_output.split('|', maxsplit=1)
        except ValueError as error:
            raise BackendPolicyError('Docker server returned an unexpected version response') from error
        if server_os != 'linux' or not server_version:
            raise BackendPolicyError('the OCI development backend requires a reachable Linux Docker server')

        inspect_output = self._query_runtime(
            ('image', 'inspect', '--format', '{{json .}}', system.image_ref),
            error_label='cannot inspect the digest-pinned system image; images are never pulled at evaluation time',
        ).decode('utf-8', errors='strict')
        try:
            image_metadata = json.loads(inspect_output)
        except json.JSONDecodeError as error:
            raise BackendPolicyError('Docker returned invalid image metadata') from error
        if not isinstance(image_metadata, dict):
            raise BackendPolicyError('Docker image inspection returned an unexpected response')
        resolved_image_id = image_metadata.get('Id')
        config = image_metadata.get('Config')
        if not isinstance(config, dict):
            raise BackendPolicyError('Docker image inspection did not return image configuration')
        if not isinstance(resolved_image_id, str) or not _is_image_id(resolved_image_id):
            raise BackendPolicyError('Docker did not resolve the system image to a sha256 image ID')
        declared_volumes = config.get('Volumes')
        if declared_volumes not in (None, {}):
            raise BackendPolicyError('system images cannot declare VOLUME mounts')
        declared_command = config.get('Cmd')
        if declared_command not in (None, []):
            raise BackendPolicyError(
                'system images cannot declare CMD; the manifest entrypoint is the complete command'
            )

        return PreparedBackend(
            capabilities=BackendCapabilities(
                backend_id='docker-oci-development',
                backend_version=server_version,
                isolation_tier=IsolationTier.DEVELOPMENT,
                network_isolation=True,
                host_filesystem_isolation=True,
                read_only_root=True,
                non_root_user=True,
                capability_drop=True,
                no_new_privileges=True,
                process_limit=True,
                memory_limit=True,
                cpu_limit=True,
                scratch_limit=True,
                fresh_worker_per_episode=True,
            ),
            resolved_image_id=resolved_image_id,
        )

    def run(
        self,
        *,
        input_bytes: bytes,
        system: SystemSubmissionManifest,
        policy: RunnerPolicy,
        prepared: PreparedBackend,
    ) -> RawExecutionResult:
        if prepared.capabilities.backend_id != 'docker-oci-development':
            raise ValueError('prepared backend does not belong to the OCI development backend')
        container_name = f'vaxreplay-{uuid.uuid4().hex}'
        create_argv = build_docker_argv(
            runtime=self._runtime,
            container_name=container_name,
            system=system,
            policy=policy,
            resolved_image_id=prepared.resolved_image_id,
        )
        try:
            created_id = (
                self._query_runtime(
                    tuple(create_argv[1:]),
                    error_label='cannot create the isolated worker container',
                )
                .decode('ascii', errors='strict')
                .strip()
            )
            if len(created_id) != 64 or any(character not in '0123456789abcdef' for character in created_id):
                raise RuntimeError('Docker create returned an invalid container ID')
            result = run_bounded_process(
                (self._runtime, 'start', '--attach', '--interactive', container_name),
                input_bytes=input_bytes,
                wall_seconds=policy.limits.wall_seconds,
                max_stdout_bytes=policy.limits.max_response_bytes,
                max_stderr_bytes=policy.limits.max_log_bytes,
                on_abort=lambda: self._request_remove(container_name),
            )
        except (BackendPolicyError, OSError, RuntimeError):
            if not self._cleanup_container(container_name):
                raise IsolationCleanupError('cannot prove failed OCI worker cleanup')
            return RawExecutionResult(
                status=RawExecutionStatus.BACKEND_ERROR,
                exit_code=None,
                duration_ms=0,
                stdout=b'',
                stderr=b'',
                stdout_truncated=False,
                stderr_truncated=False,
            )
        if not self._cleanup_container(container_name):
            raise IsolationCleanupError('cannot prove OCI worker cleanup')

        status = {
            'exited': RawExecutionStatus.EXITED,
            'timed_out': RawExecutionStatus.TIMED_OUT,
            'response_limit': RawExecutionStatus.RESPONSE_LIMIT,
            'log_limit': RawExecutionStatus.LOG_LIMIT,
        }[result.termination]
        return RawExecutionResult(
            status=status,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    def _query_runtime(self, arguments: tuple[str, ...], *, error_label: str) -> bytes:
        try:
            result = run_bounded_process(
                (self._runtime, *arguments),
                input_bytes=b'',
                wall_seconds=15,
                max_stdout_bytes=_QUERY_STDOUT_LIMIT,
                max_stderr_bytes=_QUERY_STDERR_LIMIT,
                on_abort=lambda: None,
            )
        except OSError as error:
            raise BackendPolicyError(f'{error_label}: {error}') from error
        if result.termination != 'exited' or result.exit_code != 0:
            detail = result.stderr.decode('utf-8', errors='replace').strip()[:500]
            raise BackendPolicyError(f'{error_label}: {detail or result.termination}')
        return result.stdout

    def _request_remove(self, container_name: str) -> None:
        try:
            subprocess.run(  # noqa: S603 - trusted, absolute runtime argv; never a shell
                (self._runtime, 'rm', '--force', container_name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                shell=False,
                close_fds=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _cleanup_container(self, container_name: str) -> bool:
        self._request_remove(container_name)
        try:
            remaining = self._query_runtime(
                ('ps', '--all', '--quiet', '--filter', f'name={container_name}'),
                error_label='cannot verify isolated worker cleanup',
            )
        except BackendPolicyError:
            return False
        return not remaining.strip()


def build_docker_argv(
    *,
    runtime: str,
    container_name: str,
    system: SystemSubmissionManifest,
    policy: RunnerPolicy,
    resolved_image_id: str,
) -> tuple[str, ...]:
    """Return the complete no-mount Docker argv so its controls can be unit-tested."""

    limits = policy.limits
    argv = [
        runtime,
        'create',
        '--interactive',
        '--name',
        container_name,
        '--hostname',
        'vaxreplay-worker',
        '--pull',
        'never',
        '--network',
        'none',
        '--read-only',
        '--cap-drop',
        'ALL',
        '--security-opt',
        'no-new-privileges:true',
        '--user',
        '65532:65532',
        '--pids-limit',
        str(limits.pids),
        '--memory',
        f'{limits.memory_mib}m',
        '--memory-swap',
        f'{limits.memory_mib}m',
        '--cpus',
        str(limits.cpus),
        '--ipc',
        'private',
        '--shm-size',
        f'{limits.shared_memory_mib}m',
        '--cgroupns',
        'private',
        '--tmpfs',
        f'/tmp:rw,noexec,nosuid,nodev,size={limits.scratch_mib}m',
        '--workdir',
        '/tmp',
        '--ulimit',
        f'nofile={limits.open_files}:{limits.open_files}',
        '--ulimit',
        'core=0:0',
        '--env',
        'LANG=C.UTF-8',
        '--env',
        'LC_ALL=C.UTF-8',
        '--env',
        'TZ=UTC',
        '--env',
        'VAXREPLAY_INPUT=stdin',
        '--env',
        'VAXREPLAY_OUTPUT=stdout',
        '--env',
        f'VAXREPLAY_RESPONSE_PROTOCOL={system.response_protocol}',
        '--no-healthcheck',
        '--log-driver',
        'none',
        '--init',
    ]
    if limits.gpu_count:
        argv.extend(('--gpus', str(limits.gpu_count)))
    if not _is_image_id(resolved_image_id):
        raise ValueError('resolved_image_id must be a sha256 image ID')
    argv.extend(('--entrypoint', system.entrypoint[0], resolved_image_id, *system.entrypoint[1:]))
    return tuple(argv)


def _resolve_runtime(runtime: str) -> str:
    if not runtime or '\x00' in runtime:
        raise ValueError('runtime must be a non-empty path or command name')
    resolved = shutil.which(runtime)
    if resolved is None:
        raise BackendPolicyError(f'cannot find OCI runtime command: {runtime}')
    path = str(Path(resolved).resolve())
    if os.path.basename(path) != 'docker':
        raise BackendPolicyError('V0 supports only the Docker CLI as its OCI development backend')
    return path


def _is_image_id(value: str) -> bool:
    return (
        value.startswith('sha256:')
        and len(value) == len('sha256:') + 64
        and all(character in '0123456789abcdef' for character in value.removeprefix('sha256:'))
    )
