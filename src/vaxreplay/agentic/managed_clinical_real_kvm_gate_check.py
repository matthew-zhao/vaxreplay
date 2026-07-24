"""Pytest-free frozen-runtime check for the managed provider observation gate.

This is deliberately narrower than the live managed Firecracker drill.  It verifies the exact
installed collector runtime closure, starts only the pinned network-free deterministic provider
child, proves that call zero remains blocked until one canonical create-once gate release exists,
and retains a canonical receipt.  It never opens KVM, starts Firecracker, contacts a provider, or
writes the fixed ``/etc/vaxreplay/lane-a-managed`` deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

import vaxreplay.agentic.managed_clinical_real_kvm_collector as collector
from vaxreplay.agentic.firecracker_qualification_runtime_closure import (
    LoadedQualificationDriverRuntimeClosure,
    verify_qualification_driver_runtime_closure,
)
from vaxreplay.agentic.gateway import AgenticModelMessage, AgenticModelRequest
from vaxreplay.agentic.managed_clinical_real_kvm_drill import (
    ManagedClinicalRealKvmObservationGateRelease,
)
from vaxreplay.agentic.provider_adapter import ProviderAdapterDescriptor
from vaxreplay.agentic.provider_gateway import GatewayModelRoute
from vaxreplay.agentic.provider_subprocess import ProviderSubprocessRequest, ProviderSubprocessResponse
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

_CLEAN_ENVIRONMENT = {
    'LANG': 'C',
    'LC_ALL': 'C',
    'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
}
_DRILL_ID = 'a' * 32
_CHALLENGE_NONCE_HEX = 'b' * 64
_RUN_ID = 'c' * 32
_GATE_BINDING_TOKEN = bytes.fromhex('d' * 64)
_CONTENT = '{"action":"submit","submission":{"scores":[]}}'


class ManagedClinicalRealKvmGateCheckError(RuntimeError):
    """The frozen-runtime provider-gate check failed closed."""


class ManagedClinicalRealKvmGateCheckReceipt(StrictModel):
    """Exact non-qualifying receipt from the local provider-gate contract check."""

    schema_version: Literal['vaxreplay.managed-clinical-real-kvm-gate-check.v0.1'] = (
        'vaxreplay.managed-clinical-real-kvm-gate-check.v0.1'
    )
    check_id: Literal['linux-root-frozen-provider-gate-v1'] = 'linux-root-frozen-provider-gate-v1'
    passed: Literal[True] = True
    linux_root_observed: Literal[True] = True
    isolated_python_observed: Literal[True] = True
    runtime_closure_verified: Literal[True] = True
    reproducible_build_claimed: Literal[False] = False
    self_contained_executable_claimed: Literal[False] = False
    native_operating_system_libraries_pinned: Literal[False] = False
    runtime_closure_id: str = Field(min_length=1, max_length=200)
    runtime_closure_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    runtime_closure_receipt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    runtime_closure_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    runtime_closure_entry_count: int = Field(gt=0)
    interpreter_path: str = Field(min_length=2, max_length=4096)
    checker_module_path: str = Field(min_length=2, max_length=4096)
    checker_module_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    collector_module_path: str = Field(min_length=2, max_length=4096)
    collector_module_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    output_root: str = Field(min_length=2, max_length=4096)
    provider_plan_path: str = Field(min_length=2, max_length=4096)
    provider_plan_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_child_path: str = Field(min_length=2, max_length=4096)
    provider_child_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_request_path: str = Field(min_length=2, max_length=4096)
    provider_request_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    observation_gate_path: str = Field(min_length=2, max_length=4096)
    observation_gate_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_response_path: str = Field(min_length=2, max_length=4096)
    provider_response_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    child_blocked_before_gate_release: Literal[True] = True
    child_process_group_isolated: Literal[True] = True
    child_process_group_reaped: Literal[True] = True
    credential_descriptor_passed: Literal[True] = True
    credential_bytes_supplied: Literal[0] = 0
    deterministic_network_free_fixture_used: Literal[True] = True
    nonqualifying_sentinel_gate_evidence_used: Literal[True] = True
    external_provider_called: Literal[False] = False
    real_kvm_run_performed: Literal[False] = False
    firecracker_process_started: Literal[False] = False
    fixed_deployment_written: Literal[False] = False
    fixed_deployment_state_unchanged: Literal[True] = True
    receipt_authenticated: Literal[False] = False
    live_qualification_claimed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class GateFixture:
    plan: bytes
    child: bytes
    request: bytes
    release: ManagedClinicalRealKvmObservationGateRelease
    expected_content: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one pytest-free, non-qualifying frozen-runtime provider-gate check',
    )
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--runtime-closure-root', required=True, type=Path)
    parser.add_argument('--expected-runtime-closure-manifest-sha256', required=True)
    parser.add_argument('--expected-runtime-closure-receipt-sha256', required=True)
    parser.add_argument('--expected-runtime-closure-sha256', required=True)
    return parser


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_execution_environment() -> None:
    if sys.platform != 'linux' or platform.system() != 'Linux' or os.geteuid() != 0:
        raise ManagedClinicalRealKvmGateCheckError('gate check requires effective UID 0 on Linux')
    if (
        sys.flags.isolated != 1
        or not sys.dont_write_bytecode
        or Path.cwd() != Path('/')
        or dict(os.environ) != _CLEAN_ENVIRONMENT
    ):
        raise ManagedClinicalRealKvmGateCheckError(
            'gate check requires -I -B, cwd /, and the exact empty production environment'
        )


def _normalized_fresh_output_root(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded != Path(os.path.abspath(expanded)):
        raise ManagedClinicalRealKvmGateCheckError('gate-check output root must be normalized and absolute')
    if expanded.exists() or expanded.is_symlink():
        raise ManagedClinicalRealKvmGateCheckError('gate-check output root must not already exist')
    return expanded


def _closure_entry_sha256(closure: LoadedQualificationDriverRuntimeClosure, path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ManagedClinicalRealKvmGateCheckError('frozen checker module is unavailable') from None
    matches = [entry for entry in closure.manifest.entries if entry.path == str(resolved)]
    if len(matches) != 1 or matches[0].kind != 'regular_file' or matches[0].sha256 is None:
        raise ManagedClinicalRealKvmGateCheckError('frozen checker module is absent from the runtime closure')
    return matches[0].sha256


def _path_state_sha256(path: Path) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        state: object = {'state': 'absent'}
    except OSError:
        raise ManagedClinicalRealKvmGateCheckError('fixed deployment path state could not be observed') from None
    else:
        state = {
            'state': 'present',
            'device': metadata.st_dev,
            'inode': metadata.st_ino,
            'mode': metadata.st_mode,
            'uid': metadata.st_uid,
            'gid': metadata.st_gid,
            'link_count': metadata.st_nlink,
            'size': metadata.st_size,
            'mtime_ns': metadata.st_mtime_ns,
            'ctime_ns': metadata.st_ctime_ns,
        }
    return _sha256(canonical_json_bytes(state))


def _build_fixture(*, output_root: Path, interpreter: Path) -> GateFixture:
    gate_path = output_root / 'observation-gate.json'
    child = collector.render_deterministic_provider_child(interpreter)
    child_sha256 = _sha256(child)
    plan = canonical_json_bytes(
        {
            'adapter_id': collector._ADAPTER_ID,
            'adapter_version': collector._ADAPTER_VERSION,
            'logical_model_id': collector._PUBLIC_MODEL,
            'provider': collector._PUBLIC_PROVIDER,
            'provider_model_id': collector._PUBLIC_MODEL,
            'schema_version': 'vaxreplay.managed-real-kvm-provider-plan.dev-v0.1',
            'observation_gate': {
                'binding_token_sha256': _sha256(_GATE_BINDING_TOKEN),
                'challenge_nonce_hex': _CHALLENGE_NONCE_HEX,
                'drill_id': _DRILL_ID,
                'path': str(gate_path),
                'provider_call_index': 0,
                'timeout_seconds': collector.OBSERVATION_GATE_TIMEOUT_SECONDS,
            },
            'turns': [
                {
                    'call_index': 0,
                    'content': _CONTENT,
                    'input_tokens': 1,
                    'output_tokens': 1,
                    'reasoning_tokens': 0,
                }
            ],
        }
    )
    plan_sha256 = _sha256(plan)
    route = GatewayModelRoute(
        route_id='deterministic-frozen-gate-check-route',
        logical_model_id=collector._PUBLIC_MODEL,
        provider=collector._PUBLIC_PROVIDER,
        provider_model_id=collector._PUBLIC_MODEL,
        resolved_model_id=collector._PUBLIC_MODEL,
        accepted_provider_model_ids=(collector._PUBLIC_MODEL,),
        adapter_id=collector._ADAPTER_ID,
        adapter_version=collector._ADAPTER_VERSION,
        adapter_executable_sha256=child_sha256,
        adapter_config_sha256=plan_sha256,
        endpoint_origin='https://fixture.invalid',
        endpoint_path='/never-called',
        fixed_parameters_sha256=_sha256(b'vaxreplay-gate-check-fixed-parameters-v1'),
        max_context_tokens=4096,
        max_output_tokens=1024,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )
    adapter = ProviderAdapterDescriptor(
        adapter_id=collector._ADAPTER_ID,
        adapter_version=collector._ADAPTER_VERSION,
        executable_sha256=child_sha256,
        config_sha256=plan_sha256,
        provider=collector._PUBLIC_PROVIDER,
    )
    request = ProviderSubprocessRequest(
        request=AgenticModelRequest(
            run_id=_RUN_ID,
            call_index=0,
            messages=(AgenticModelMessage(role='system', content='Use only the sealed gate-check fixture.'),),
            max_output_tokens=32,
        ),
        route=route,
        adapter=adapter,
        timeout_milliseconds=10_000,
    )
    observed_at = datetime.now(UTC)
    release = ManagedClinicalRealKvmObservationGateRelease(
        drill_id=_DRILL_ID,
        challenge_nonce_hex=_CHALLENGE_NONCE_HEX,
        challenge_sha256=_sha256(b'vaxreplay-gate-check-no-live-challenge-v1'),
        run_id=_RUN_ID,
        ownership_envelope_sha256=_sha256(b'vaxreplay-gate-check-no-live-ownership-v1'),
        live_process_observation_sha256=_sha256(b'vaxreplay-gate-check-no-live-process-observation-v1'),
        gate_binding_token_hex=_GATE_BINDING_TOKEN.hex(),
        observed_at=observed_at,
        released_at=datetime.now(UTC),
        persisted_path=str(gate_path),
    )
    return GateFixture(
        plan=plan,
        child=child,
        request=canonical_json_bytes(request),
        release=release,
        expected_content=_CONTENT,
    )


def _start_child(
    *,
    child_path: Path,
    plan_path: Path,
    plan_sha256: str,
    credential_descriptor: int,
) -> collector.RunningManagedInvocation:
    process = subprocess.Popen(
        (
            str(child_path),
            '--plan',
            str(plan_path),
            '--expected-plan-sha256',
            plan_sha256,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/',
        env={
            **_CLEAN_ENVIRONMENT,
            'VAXREPLAY_PROVIDER_CREDENTIAL_FD': str(credential_descriptor),
        },
        pass_fds=(credential_descriptor,),
        close_fds=True,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None or process.stdin is None:
        collector._abort_managed_invocation_setup(process, process_group_id=process.pid)
        raise ManagedClinicalRealKvmGateCheckError('gate-check child lacks exact bounded pipes')
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group_id = -1
    if process_group_id != process.pid:
        collector._abort_managed_invocation_setup(process, process_group_id=process_group_id)
        raise ManagedClinicalRealKvmGateCheckError('gate-check child lacks one isolated process group')
    stdout_drain: collector.BoundedPipeDrain | None = None
    stderr_drain: collector.BoundedPipeDrain | None = None
    try:
        stdout_drain = collector._start_bounded_pipe_drain(process.stdout, label='gate-check stdout')
        stderr_drain = collector._start_bounded_pipe_drain(process.stderr, label='gate-check stderr')
    except BaseException as error:
        collector._abort_managed_invocation_setup(
            process,
            process_group_id=process_group_id,
            stdout_drain=stdout_drain,
            stderr_drain=stderr_drain,
        )
        raise ManagedClinicalRealKvmGateCheckError('gate-check bounded drains could not be started') from error
    return collector.RunningManagedInvocation(
        process=process,
        process_group_id=process_group_id,
        stdout_drain=stdout_drain,
        stderr_drain=stderr_drain,
    )


def _exercise_fixture(*, output_root: Path, fixture: GateFixture) -> tuple[bytes, str]:
    plan_path = output_root / 'provider-plan.json'
    child_path = output_root / 'provider-child'
    request_path = output_root / 'provider-request.json'
    gate_path = output_root / 'observation-gate.json'
    collector._write_create_once(plan_path, fixture.plan)
    collector._write_create_once(child_path, fixture.child, mode=0o500)
    collector._write_create_once(request_path, fixture.request)
    credential_read, credential_write = os.pipe()
    running: collector.RunningManagedInvocation | None = None
    completed: collector.CompletedManagedInvocation | None = None
    try:
        running = _start_child(
            child_path=child_path,
            plan_path=plan_path,
            plan_sha256=_sha256(fixture.plan),
            credential_descriptor=credential_read,
        )
        if running.process.stdin is None:
            raise ManagedClinicalRealKvmGateCheckError('gate-check child lost its request pipe')
        running.process.stdin.write(fixture.request)
        running.process.stdin.close()
        time.sleep(0.1)
        if running.process.poll() is not None:
            completed = collector._finish_managed_invocation(running, timeout_seconds=1)
            running = None
            raise ManagedClinicalRealKvmGateCheckError(
                'gate-check child returned before the observation gate was released: '
                + completed.stderr[:1000].decode('utf-8', errors='replace')
            )
        collector._write_create_once(gate_path, canonical_json_bytes(fixture.release))
        completed = collector._finish_managed_invocation(running, timeout_seconds=3)
        process_group_reaped = not collector._managed_process_group_exists(running.process_group_id)
        running = None
    except BaseException:
        if running is not None:
            collector._abort_managed_invocation_setup(
                running.process,
                process_group_id=running.process_group_id,
                stdout_drain=running.stdout_drain,
                stderr_drain=running.stderr_drain,
            )
        raise
    finally:
        os.close(credential_read)
        os.close(credential_write)
    if completed is None or completed.return_code != 0 or completed.stderr:
        raise ManagedClinicalRealKvmGateCheckError('gate-check child failed its exact provider contract')
    try:
        response = ProviderSubprocessResponse.model_validate_json(completed.stdout)
    except ValueError:
        raise ManagedClinicalRealKvmGateCheckError('gate-check child returned an invalid response') from None
    if (
        canonical_json_bytes(response) != completed.stdout
        or not response.succeeded
        or response.result is None
        or response.result.content != fixture.expected_content
        or response.result.provider_request_sha256 != _sha256(fixture.request)
        or not process_group_reaped
    ):
        raise ManagedClinicalRealKvmGateCheckError('gate-check child response differs from the pinned request')
    return completed.stdout, _sha256(canonical_json_bytes(fixture.release))


def _run(arguments: argparse.Namespace) -> ManagedClinicalRealKvmGateCheckReceipt:
    _require_execution_environment()
    output_root = _normalized_fresh_output_root(arguments.output_root)
    closure = verify_qualification_driver_runtime_closure(
        arguments.runtime_closure_root,
        expected_manifest_sha256=arguments.expected_runtime_closure_manifest_sha256,
        expected_receipt_sha256=arguments.expected_runtime_closure_receipt_sha256,
        expected_closure_sha256=arguments.expected_runtime_closure_sha256,
        require_root_owned=True,
    )
    if Path(sys.executable) != Path(closure.manifest.interpreter_path):
        raise ManagedClinicalRealKvmGateCheckError('gate check is not using the closure-pinned interpreter')
    collector._verify_loaded_module_runtime_binding(closure)
    before_fixed_deployment = _path_state_sha256(collector.FIXED_CONFIG_ROOT)
    collector._require_root_directory_path(output_root.parent)
    collector._create_root_directory(output_root)
    fixture = _build_fixture(output_root=output_root, interpreter=Path(sys.executable))
    response, gate_sha256 = _exercise_fixture(output_root=output_root, fixture=fixture)
    response_path = output_root / 'provider-response.json'
    collector._write_create_once(response_path, response)
    after_fixed_deployment = _path_state_sha256(collector.FIXED_CONFIG_ROOT)
    if before_fixed_deployment != after_fixed_deployment:
        raise ManagedClinicalRealKvmGateCheckError('gate check observed a fixed deployment path state change')
    checker_path = Path(__file__)
    collector_path = Path(str(collector.__file__))
    receipt = ManagedClinicalRealKvmGateCheckReceipt(
        runtime_closure_id=closure.manifest.closure_id,
        runtime_closure_manifest_sha256=closure.manifest_sha256,
        runtime_closure_receipt_sha256=closure.receipt_sha256,
        runtime_closure_sha256=closure.closure_sha256,
        runtime_closure_entry_count=closure.manifest.entry_count,
        interpreter_path=closure.manifest.interpreter_path,
        checker_module_path=str(checker_path),
        checker_module_sha256=_closure_entry_sha256(closure, checker_path),
        collector_module_path=str(collector_path),
        collector_module_sha256=_closure_entry_sha256(closure, collector_path),
        output_root=str(output_root),
        provider_plan_path=str(output_root / 'provider-plan.json'),
        provider_plan_sha256=_sha256(fixture.plan),
        provider_child_path=str(output_root / 'provider-child'),
        provider_child_sha256=_sha256(fixture.child),
        provider_request_path=str(output_root / 'provider-request.json'),
        provider_request_sha256=_sha256(fixture.request),
        observation_gate_path=str(output_root / 'observation-gate.json'),
        observation_gate_sha256=gate_sha256,
        provider_response_path=str(response_path),
        provider_response_sha256=_sha256(response),
    )
    receipt_bytes = canonical_json_bytes(receipt)
    collector._write_create_once(output_root / 'CHECK-RECEIPT.json', receipt_bytes)
    collector._write_create_once(output_root / 'CHECK-RECEIPT.sha256', (_sha256(receipt_bytes) + '\n').encode('ascii'))
    return receipt


def main() -> None:
    receipt = _run(_parser().parse_args())
    sys.stdout.buffer.write(canonical_json_bytes(receipt) + b'\n')


if __name__ == '__main__':
    main()
