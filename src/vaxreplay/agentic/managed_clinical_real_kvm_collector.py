"""Collect one bounded managed Lane A task run on a real Firecracker microVM.

The live command consumes an organizer-issued drill ID and challenge, provisions one disposable fixed
``/etc/vaxreplay/lane-a-managed`` deployment, reserves one real public clinical task through the
managed Unix authority, invokes the production no-argument entrypoint twice, and retains signed
evidence.  The first invocation must succeed.  The second must be denied before worker/provider
execution, proving global one-attempt enforcement after a real service-process restart.

The provider is a four-turn deterministic subprocess fixture.  It receives the ordinary fake
credential descriptor but its exact pinned source never reads it and performs no network call.
This is a development integration drill, not model/provider or leaderboard qualification.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Literal, TypeVar, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import field_validator

import vaxreplay.agentic.clinical_launcher as clinical_launcher_module
import vaxreplay.agentic.clinical_operator as clinical_operator_module
import vaxreplay.agentic.clinical_production_run_v02 as clinical_production_run_v02_module
import vaxreplay.agentic.firecracker_clinical_runtime as firecracker_clinical_runtime_module
import vaxreplay.agentic.provider_subprocess as provider_subprocess_module
from vaxreplay._atomic import rename_directory_noreplace
from vaxreplay.agentic.clinical_execution_bridge import (
    LoadedClinicalAgenticWorkspace,
    build_clinical_agentic_workspace,
    clinical_workspace_receipt_key_id,
    load_clinical_agentic_workspace,
)
from vaxreplay.agentic.clinical_guest_bootstrap import (
    clinical_guest_bootstrap_authorization_key_id,
    clinical_guest_bootstrap_receipt_key_id,
)
from vaxreplay.agentic.clinical_guest_executable import (
    LaneAClinicalGuestConfig,
    load_lane_a_clinical_guest_config,
)
from vaxreplay.agentic.clinical_launcher import (
    CanonicalClinicalLauncherDeployment,
    ClinicalLauncherFailureCode,
    clinical_launcher_failure_key_id,
)
from vaxreplay.agentic.clinical_operator import (
    CanonicalClinicalOperatorManifest,
    expected_system_identity,
)
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionReservationContext,
    ClinicalProductionSystemIdentity,
    ClinicalProductionTaskRecord,
    ClinicalProductionTerminalCode,
    SqliteClinicalProductionRegistry,
    clinical_production_system_identity_sha256,
)
from vaxreplay.agentic.clinical_production_run_v02 import (
    LoadedClinicalProductionRunV02,
    load_clinical_production_run_v02,
)
from vaxreplay.agentic.firecracker import (
    FirecrackerWorkerSpec,
)
from vaxreplay.agentic.firecracker_clinical_runtime import (
    FirecrackerClinicalRuntimeConfig,
    FirecrackerClinicalRuntimeKeys,
    firecracker_clinical_runtime_config_sha256,
    reconcile_firecracker_clinical_startup_without_execution,
)
from vaxreplay.agentic.firecracker_qualification import (
    decode_firecracker_qualification_key,
    firecracker_qualification_key_id,
    load_firecracker_qualification,
    load_pinned_firecracker_worker_spec,
)
from vaxreplay.agentic.firecracker_qualification_probe import (
    AuthenticatedFirecrackerQualificationRawCollection,
)
from vaxreplay.agentic.firecracker_qualification_runtime_closure import (
    LoadedQualificationDriverRuntimeClosure,
    verify_qualification_driver_runtime_closure,
)
from vaxreplay.agentic.guest_disk_build import (
    LaneAGuestDiskBuildReceipt,
    load_lane_a_guest_disk_build_receipt,
)
from vaxreplay.agentic.guest_rpc import (
    GuestRpcPolicy,
    guest_rpc_policy_sha256,
)
from vaxreplay.agentic.managed_clinical_deployment import (
    LoadedManagedClinicalDeployment,
    ManagedClinicalDeploymentSecrets,
    ManagedClinicalStandaloneDeployment,
    validate_managed_clinical_deployment_binding,
)
from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalOwnershipLedger,
    LinuxManagedClinicalHostAdapter,
    ManagedClinicalOwnershipConfig,
    authenticated_managed_clinical_ownership_sha256,
    managed_clinical_ownership_config_sha256,
    read_linux_process_identity,
)
from vaxreplay.agentic.managed_clinical_real_kvm_drill import (
    AuthenticatedManagedClinicalRealKvmDrill,
    ManagedClinicalRealKvmDrillEvidence,
    ManagedClinicalRealKvmExternalPins,
    ManagedClinicalRealKvmObservationGateRelease,
    ManagedClinicalRealKvmProcessObservation,
    ManagedClinicalRealKvmVerifierKeys,
    ManagedClinicalRegistryDrillObservation,
    ManagedClinicalStartupCleanupDrillObservation,
    authenticate_managed_clinical_real_kvm_drill,
    independently_verify_authenticated_managed_clinical_real_kvm_drill,
    managed_clinical_real_kvm_authority_id,
    managed_clinical_real_kvm_challenge_sha256,
    managed_clinical_real_kvm_collector_key_id,
    managed_clinical_real_kvm_deployment_id,
    managed_clinical_real_kvm_registered_entry_id,
    managed_clinical_real_kvm_release_pins_sha256,
    verify_managed_clinical_real_kvm_drill_from_persisted_state,
)
from vaxreplay.agentic.managed_clinical_registry import (
    LINUX_AF_UNIX_PATHNAME_MAX_BYTES,
    AuthenticatedManagedClinicalRegistryAudit,
    ManagedClaimRequest,
    ManagedClinicalRegistryClient,
    ManagedClinicalRegistryConfig,
    ManagedClinicalRegistryService,
    ManagedRecordRunRequest,
    ManagedWorkspaceReference,
    load_authenticated_managed_registry_audit_chain,
    managed_clinical_registry_config_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    AuthenticatedManagedClinicalStartupCleanup,
    ManagedClinicalStartupConfig,
    ManagedClinicalStartupReconciler,
    load_authenticated_managed_cleanup,
    managed_clinical_cleanup_key_id,
    managed_clinical_startup_config_sha256,
)
from vaxreplay.agentic.managed_gateway_capability import (
    RestartVisibleManagedGatewayCapabilityLedger,
)
from vaxreplay.agentic.protocol import AgenticExecutionPolicy, agentic_policy_sha256
from vaxreplay.agentic.provider_adapter import ProviderAdapterDescriptor
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayPolicy,
    GatewayCapabilityRevocation,
    GatewayModelRoute,
    SqliteGatewayLedger,
    authenticated_gateway_policy_sha256,
    gateway_model_route_sha256,
    gateway_session_key_id,
)
from vaxreplay.agentic.provider_subprocess import ProviderSubprocessSpec
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
    make_agentic_harness_identity,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split, StrictModel
from vaxreplay.clinicaltrials.execution_aggregation import (
    ExecutionCohortManifest,
    ExecutionCohortSplitCount,
    ExecutionCohortTaskBinding,
    execution_cohort_aggregation_policy_sha256,
)
from vaxreplay.clinicaltrials.execution_baselines import uniform_execution_submission
from vaxreplay.clinicaltrials.execution_task import ExecutionTask

STATE_PARENT = Path('/var/lib/vaxreplay/managed-real-kvm-drills')
FIXED_CONFIG_ROOT = Path('/etc/vaxreplay/lane-a-managed')
# Linux pathname AF_UNIX sockets have a 108-byte ``sun_path`` field, including
# the terminating NUL.  Keep the human-auditable full challenge namespace while
# using a deliberately compact volatile parent and socket basename.
RUNTIME_SOCKET_PARENT = Path('/run/vrk')
OBSERVATION_GATE_PARENT = Path('/var/lib/vaxreplay/managed-real-kvm-observation-gates')
EVIDENCE_FILE = 'authenticated-evidence.json'
EVIDENCE_SHA256_FILE = 'EVIDENCE.sha256'
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_MANAGED_OUTPUT_BYTES = 1024 * 1024
MAX_EVIDENCE_SNAPSHOT_ENTRIES = 10_000
MAX_EVIDENCE_SNAPSHOT_BYTES = 512 * 1024 * 1024
OBSERVATION_GATE_TIMEOUT_SECONDS = 10
MANAGED_PROVIDER_CALL_SECONDS = 15
MANAGED_MINIMUM_WALL_SECONDS = 30
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_PUBLIC_PROVIDER = 'vaxreplay-public-deterministic-fixture'
_PUBLIC_MODEL = 'vaxreplay-public-deterministic-v0'
_ADAPTER_ID = 'vaxreplay-public-deterministic-subprocess'
_ADAPTER_VERSION = 'dev-v0.1'
_OBSERVATION_GATE_TOKEN_DOMAIN = b'vaxreplay.managed-real-kvm-observation-gate-token.dev-v0.1\x00'
_VERIFIER_KEY_FILES = frozenset(
    {
        'bootstrap-receipt.key',
        'gateway-receipt.key',
        'guest-rpc-receipt.key',
        'launcher-failure-receipt.key',
        'managed-authority.key',
        'production-receipt.key',
        'qualification.key',
        'worker-attestation.key',
        'workspace-receipt.key',
    }
)
_EXTERNAL_PIN_CHALLENGE_FIELDS = frozenset(
    {
        'drill_id',
        'challenge_nonce_hex',
        'challenge_issued_at',
        'release_pins_sha256',
        'challenge_sha256',
    }
)
_STABLE_RELEASE_PIN_FIELDS = frozenset(ManagedClinicalRealKvmExternalPins.model_fields) - _EXTERNAL_PIN_CHALLENGE_FIELDS

ModelT = TypeVar('ModelT', bound=StrictModel)


class ManagedClinicalRealKvmDrillError(RuntimeError):
    """The bounded drill was unsafe, incomplete, or contradicted its claimed evidence."""


class _ManagedClinicalCleanupIncompleteError(ManagedClinicalRealKvmDrillError):
    """A failed mutation left missing, replaced, foreign, or durability-ambiguous state."""


def _require_linux_pathname_socket_path(path: Path) -> Path:
    """Reject a pathname that cannot fit in Linux ``sockaddr_un.sun_path``."""

    encoded = os.fsencode(path)
    if not path.is_absolute() or not encoded or b'\x00' in encoded:
        raise ManagedClinicalRealKvmDrillError('managed registry socket must be one absolute pathname')
    if len(encoded) > LINUX_AF_UNIX_PATHNAME_MAX_BYTES:
        raise ManagedClinicalRealKvmDrillError('managed registry AF_UNIX pathname exceeds the Linux 107-byte limit')
    return path


@dataclass(frozen=True, slots=True)
class DrillPaths:
    root: Path
    drill_id: str
    private_root: Path
    config_root: Path
    operator_secret_root: Path
    managed_secret_root: Path
    workspace_root: Path
    evidence_root: Path
    ownership_root: Path
    startup_receipt_root: Path
    gateway_database: Path
    registry_database: Path
    registry_socket: Path
    protocol_audit_root: Path
    provider_child: Path
    provider_plan: Path
    collector_runtime_closure_root: Path
    authorization_receipt: Path

    @classmethod
    def live(cls, drill_id: str, challenge_sha256: str) -> 'DrillPaths':
        if len(drill_id) != 32 or any(character not in '0123456789abcdef' for character in drill_id):
            raise ManagedClinicalRealKvmDrillError('organizer drill ID is invalid')
        _require_sha256(challenge_sha256, 'organizer challenge digest')
        namespace = f'{drill_id}-{challenge_sha256[:32]}'
        root = STATE_PARENT / namespace
        registry_socket = _require_linux_pathname_socket_path(RUNTIME_SOCKET_PARENT / f'{namespace}.sock')
        return cls(
            root=root,
            drill_id=drill_id,
            private_root=root / 'private',
            config_root=FIXED_CONFIG_ROOT,
            operator_secret_root=FIXED_CONFIG_ROOT / 'operator-secrets',
            managed_secret_root=FIXED_CONFIG_ROOT / 'managed-secrets',
            workspace_root=root / 'workspace',
            evidence_root=root / 'production-evidence',
            ownership_root=root / 'ownership',
            startup_receipt_root=root / 'startup-receipts',
            gateway_database=root / 'gateway' / 'gateway.sqlite3',
            registry_database=root / 'registry' / 'attempts.sqlite3',
            registry_socket=registry_socket,
            protocol_audit_root=root / 'registry-audit',
            provider_child=root / 'provider-fixture' / 'provider-child',
            provider_plan=root / 'provider-fixture' / 'provider-plan.json',
            collector_runtime_closure_root=root / 'collector-runtime-closure',
            authorization_receipt=root / 'challenge-authorization.json',
        )


@dataclass(frozen=True, slots=True)
class DrillAuthorization:
    drill_id: str
    challenge_nonce_hex: str
    challenge_issued_at: datetime
    release_pins_sha256: str
    challenge_sha256: str
    registry_authority_id: str
    deployment_id: str
    registered_entry_id: str
    external_pins: ManagedClinicalRealKvmExternalPins
    observation_gate_path: Path
    observation_gate_binding_token: bytes


class ManagedClinicalRealKvmChallengeAuthorizationReceipt(StrictModel):
    """Canonical public prelaunch receipt; it is deliberately not an authenticated result."""

    schema_version: Literal['vaxreplay.managed-real-kvm-challenge-authorization.dev-v0.1'] = (
        'vaxreplay.managed-real-kvm-challenge-authorization.dev-v0.1'
    )
    drill_id: str
    challenge_nonce_hex: str
    challenge_issued_at: datetime
    release_pins_sha256: str
    challenge_sha256: str
    registry_authority_id: str
    deployment_id: str
    registered_entry_id: str
    observation_gate_path: str
    observation_gate_binding_token_sha256: str
    state_namespace: str
    collector_selected_identity_or_nonce: Literal[False] = False
    fresh_create_once_namespace: Literal[True] = True
    development_only: Literal[True] = True
    official_execution_qualified: Literal[False] = False
    authenticated: Literal[False] = False
    vm_started: Literal[False] = False
    registry_started: Literal[False] = False
    provider_started: Literal[False] = False

    @field_validator('challenge_issued_at')
    @classmethod
    def validate_challenge_issued_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('challenge authorization time requires a UTC offset')
        return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _CreatedLivePath:
    path: Path
    device_id: int
    inode: int
    owner_uid: int
    owner_gid: int
    mode: int
    kind: Literal['directory', 'regular_file']


@dataclass(frozen=True, slots=True)
class DrillKeys:
    workspace: bytes
    worker: bytes
    gateway: bytes
    guest_rpc: bytes
    bootstrap_receipt: bytes
    production: bytes
    launcher_failure: bytes
    managed: bytes

    @classmethod
    def generate(cls) -> 'DrillKeys':
        return cls(*(secrets.token_bytes(32) for _ in range(8)))

    @property
    def runtime(self) -> FirecrackerClinicalRuntimeKeys:
        return FirecrackerClinicalRuntimeKeys(
            workspace_receipt_key=self.workspace,
            worker_attestation_key=self.worker,
            gateway_receipt_key=self.gateway,
            guest_rpc_receipt_key=self.guest_rpc,
            clinical_guest_bootstrap_receipt_key=self.bootstrap_receipt,
            production_receipt_key=self.production,
        )

    def verifier(self, *, qualification_key: bytes) -> ManagedClinicalRealKvmVerifierKeys:
        return ManagedClinicalRealKvmVerifierKeys(
            workspace_receipt_key=self.workspace,
            worker_attestation_key=self.worker,
            gateway_receipt_key=self.gateway,
            guest_rpc_receipt_key=self.guest_rpc,
            bootstrap_receipt_key=self.bootstrap_receipt,
            production_receipt_key=self.production,
            qualification_key=qualification_key,
            ownership_key=self.managed,
            startup_cleanup_key=self.managed,
        )


@dataclass(frozen=True, slots=True)
class PublicInputs:
    spec: FirecrackerWorkerSpec
    policy: AgenticExecutionPolicy
    rpc_policy: GuestRpcPolicy
    guest_config: LaneAClinicalGuestConfig
    disk_receipt: LaneAGuestDiskBuildReceipt
    task: ExecutionTask
    qualification: Any
    qualification_key: bytes
    qualification_raw: AuthenticatedFirecrackerQualificationRawCollection
    collector_runtime_closure: LoadedQualificationDriverRuntimeClosure


@dataclass(frozen=True, slots=True)
class Composition:
    workspace: LoadedClinicalAgenticWorkspace
    cohort: ExecutionCohortManifest
    runtime_config: FirecrackerClinicalRuntimeConfig
    gateway_policy: AuthenticatedGatewayPolicy
    gateway_route: GatewayModelRoute
    provider_adapter: ProviderAdapterDescriptor
    provider_subprocess: ProviderSubprocessSpec
    submitted_harness: SubmittedHarnessManifest
    provisional_manifest: CanonicalClinicalOperatorManifest
    system: ClinicalProductionSystemIdentity
    startup_config: ManagedClinicalStartupConfig
    ownership_config: ManagedClinicalOwnershipConfig
    registry_config: ManagedClinicalRegistryConfig


@dataclass(slots=True)
class RunningRegistry:
    service: ManagedClinicalRegistryService
    client: ManagedClinicalRegistryClient
    stop_event: threading.Event
    thread: threading.Thread
    errors: list[BaseException]

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(10)
        if self.thread.is_alive():
            raise ManagedClinicalRealKvmDrillError('managed registry did not stop')
        if self.errors:
            raise ManagedClinicalRealKvmDrillError('managed registry service failed') from self.errors[0]


@dataclass(frozen=True, slots=True)
class FinalComposition:
    base: Composition
    reservation_context: ClinicalProductionReservationContext
    manifest: CanonicalClinicalOperatorManifest
    deployment: ManagedClinicalStandaloneDeployment


@dataclass(slots=True)
class RunningManagedInvocation:
    process: subprocess.Popen[bytes]
    process_group_id: int
    stdout_drain: 'BoundedPipeDrain'
    stderr_drain: 'BoundedPipeDrain'


@dataclass(slots=True)
class BoundedPipeDrain:
    label: str
    pipe: Any
    content: bytearray
    overflowed: threading.Event
    done: threading.Event
    errors: list[BaseException]
    thread: threading.Thread


@dataclass(frozen=True, slots=True)
class CompletedManagedInvocation:
    return_code: int
    stdout: bytes
    stderr: bytes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_model(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _sha256_label(value: str) -> str:
    return _sha256_bytes(('vaxreplay-managed-real-kvm:' + value).encode('utf-8'))


def _module_sha256(module: ModuleType) -> str:
    path = Path(cast(str, module.__file__))
    return _sha256_file(path)


def _verify_loaded_module_runtime_binding(
    closure: LoadedQualificationDriverRuntimeClosure,
) -> None:
    """Require every loaded file-backed module to be one exact closure entry."""

    roots = tuple(Path(value) for value in closure.manifest.runtime_roots)
    entries = {item.path: item for item in closure.manifest.entries}
    observed: dict[Path, str] = {}
    for name, module in tuple(sys.modules.items()):
        raw = getattr(module, '__file__', None)
        if raw is None:
            continue
        if not isinstance(raw, str) or not raw or raw.startswith('<'):
            raise ManagedClinicalRealKvmDrillError('collector loaded a module with an unrepresentable source path')
        supplied = Path(raw)
        try:
            resolved = supplied.resolve(strict=True)
        except OSError:
            raise ManagedClinicalRealKvmDrillError('collector loaded a module whose source is unavailable') from None
        if (
            supplied != resolved
            or supplied.suffix in {'.pyc', '.pyo'}
            or not any(resolved == root or root in resolved.parents for root in roots)
        ):
            raise ManagedClinicalRealKvmDrillError('collector loaded code outside its isolated runtime closure')
        entry = entries.get(str(resolved))
        if entry is None or entry.kind != 'regular_file' or entry.sha256 is None:
            raise ManagedClinicalRealKvmDrillError('collector loaded code absent from its runtime inventory')
        digest = observed.setdefault(resolved, _sha256_file(resolved))
        if not hmac.compare_digest(digest, entry.sha256):
            raise ManagedClinicalRealKvmDrillError(f'collector module {name!r} differs from its runtime inventory')


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ManagedClinicalRealKvmDrillError(f'{label} must be one lowercase SHA-256 digest')


def _observation_gate_inputs(arguments: argparse.Namespace) -> tuple[Path, bytes]:
    drill_id = cast(str, arguments.drill_id)
    nonce = cast(str, arguments.challenge_nonce_hex)
    if (
        len(drill_id) != 32
        or any(character not in '0123456789abcdef' for character in drill_id)
        or len(nonce) != 64
        or any(character not in '0123456789abcdef' for character in nonce)
    ):
        raise ManagedClinicalRealKvmDrillError(
            'organizer drill ID and challenge nonce must be exact lowercase hexadecimal values'
        )
    token = hashlib.sha256(
        _OBSERVATION_GATE_TOKEN_DOMAIN + drill_id.encode('ascii') + b'\x00' + nonce.encode('ascii')
    ).digest()
    return OBSERVATION_GATE_PARENT / f'{drill_id}-{nonce}.json', token


def _canonical_provider_plan(
    task: ExecutionTask,
    *,
    observation_gate: tuple[Path, bytes, str, str] | None = None,
) -> bytes:
    documents = task.context.cutoff_documents
    if not documents or not documents[0].body:
        raise ManagedClinicalRealKvmDrillError('deterministic provider fixture requires one nonempty cutoff document')
    source_paths = tuple(f'sources/source-{index:03d}.txt' for index in range(1, min(len(documents), 8) + 1))
    match = re.search(r'[A-Za-z0-9_]{4,64}', documents[0].body)
    if match is None:
        raise ManagedClinicalRealKvmDrillError('task lacks a bounded fixture search token')
    submission = uniform_execution_submission(task)
    actions: tuple[dict[str, object], ...] = (
        {'action': 'list_workspace', 'cursor': 0, 'limit': 100},
        {
            'action': 'search_workspace',
            'needle': match.group(0),
            'paths': source_paths,
            'max_results': 20,
        },
        {
            'action': 'read_workspace',
            'path': source_paths[0],
            'offset': 0,
            'limit': min(len(documents[0].body.encode('utf-8')), 32_768),
        },
        {
            'action': 'submit',
            'submission': submission.model_dump(mode='json'),
        },
    )
    turns = []
    for index, action in enumerate(actions):
        content = canonical_json_bytes(action).decode('utf-8')
        turns.append(
            {
                'call_index': index,
                'content': content,
                'input_tokens': 4096,
                'output_tokens': max(1, (len(content.encode('utf-8')) + 3) // 4),
                'reasoning_tokens': 0,
            }
        )
    gate = None
    if observation_gate is not None:
        gate_path, binding_token, drill_id, challenge_nonce_hex = observation_gate
        if not gate_path.is_absolute() or gate_path != Path(os.path.abspath(gate_path)):
            raise ManagedClinicalRealKvmDrillError('observation gate path must be normalized and absolute')
        gate = {
            'binding_token_sha256': _sha256_bytes(binding_token),
            'challenge_nonce_hex': challenge_nonce_hex,
            'drill_id': drill_id,
            'path': str(gate_path),
            'provider_call_index': 0,
            'timeout_seconds': OBSERVATION_GATE_TIMEOUT_SECONDS,
        }
    return canonical_json_bytes(
        {
            'adapter_id': _ADAPTER_ID,
            'adapter_version': _ADAPTER_VERSION,
            'logical_model_id': _PUBLIC_MODEL,
            'provider': _PUBLIC_PROVIDER,
            'provider_model_id': _PUBLIC_MODEL,
            'schema_version': 'vaxreplay.managed-real-kvm-provider-plan.dev-v0.1',
            'observation_gate': gate,
            'turns': turns,
        }
    )


def render_deterministic_provider_child(interpreter: Path) -> bytes:
    """Render the exact credential-ignoring, network-free provider fixture executable."""

    path = PurePosixPath(str(interpreter))
    if not path.is_absolute() or '..' in path.parts or str(path) != str(interpreter):
        raise ValueError('provider child interpreter path must be normalized and absolute')
    body = r"""from __future__ import annotations
import argparse
import hashlib
import json
import os
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

MAXIMUM = 16 * 1024 * 1024
CREDENTIAL_FD_NAME = 'VAXREPLAY_PROVIDER_CREDENTIAL_FD'

def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def fail():
    raise SystemExit(1)

def main():
    credential_descriptor = os.environ.get(CREDENTIAL_FD_NAME, '')
    expected_environment = {
        'LANG': 'C',
        'LC_ALL': 'C',
        'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
        CREDENTIAL_FD_NAME: credential_descriptor,
    }
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
        or Path.cwd() != Path('/')
        or dict(os.environ) != expected_environment
        or not credential_descriptor.isascii()
        or not credential_descriptor.isdigit()
        or int(credential_descriptor) < 3
    ):
        fail()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    plan_path = PurePosixPath(args.plan)
    if not plan_path.is_absolute() or '..' in plan_path.parts or str(plan_path) != args.plan:
        fail()
    path = Path(plan_path)
    if path.is_symlink():
        fail()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or not 0 < metadata.st_size <= MAXIMUM
        ):
            fail()
        plan_bytes = b''
        while len(plan_bytes) <= MAXIMUM:
            block = os.read(descriptor, min(1024 * 1024, MAXIMUM + 1 - len(plan_bytes)))
            if not block:
                break
            plan_bytes += block
    finally:
        os.close(descriptor)
    if hashlib.sha256(plan_bytes).hexdigest() != args.expected_plan_sha256:
        fail()
    try:
        plan = json.loads(plan_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail()
    if (
        canonical(plan) != plan_bytes
        or plan.get('schema_version')
        != 'vaxreplay.managed-real-kvm-provider-plan.dev-v0.1'
    ):
        fail()
    request_bytes = sys.stdin.buffer.read(MAXIMUM + 1)
    if not request_bytes or len(request_bytes) > MAXIMUM:
        fail()
    try:
        envelope = json.loads(request_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail()
    if (
        canonical(envelope) != request_bytes
        or envelope.get('schema_version')
        != 'vaxreplay.provider-subprocess-request.dev-v0.1'
    ):
        fail()
    request = envelope.get('request')
    route = envelope.get('route')
    adapter = envelope.get('adapter')
    if not isinstance(request, dict) or not isinstance(route, dict) or not isinstance(adapter, dict):
        fail()
    if (
        route.get('provider'),
        route.get('logical_model_id'),
        route.get('provider_model_id'),
        route.get('adapter_id'),
        route.get('adapter_version'),
        route.get('adapter_config_sha256'),
    ) != (
        plan.get('provider'),
        plan.get('logical_model_id'),
        plan.get('provider_model_id'),
        plan.get('adapter_id'),
        plan.get('adapter_version'),
        args.expected_plan_sha256,
    ):
        fail()
    if (
        adapter.get('provider'),
        adapter.get('adapter_id'),
        adapter.get('adapter_version'),
        adapter.get('config_sha256'),
    ) != (
        plan.get('provider'),
        plan.get('adapter_id'),
        plan.get('adapter_version'),
        args.expected_plan_sha256,
    ):
        fail()
    index = request.get('call_index')
    turns = plan.get('turns')
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not isinstance(turns, list)
        or index < 0
        or index >= len(turns)
    ):
        fail()
    turn = turns[index]
    if not isinstance(turn, dict) or turn.get('call_index') != index:
        fail()
    run_id = request.get('run_id')
    if not isinstance(run_id, str) or len(run_id) != 32:
        fail()
    gate = plan.get('observation_gate')
    if index == 0 and gate is not None:
        if not isinstance(gate, dict) or set(gate) != {
            'binding_token_sha256',
            'challenge_nonce_hex',
            'drill_id',
            'path',
            'provider_call_index',
            'timeout_seconds',
        }:
            fail()
        gate_path = gate.get('path')
        gate_timeout = gate.get('timeout_seconds')
        if (
            not isinstance(gate_path, str)
            or not isinstance(gate_timeout, int)
            or isinstance(gate_timeout, bool)
            or gate_timeout != 10
            or gate.get('provider_call_index') != 0
        ):
            fail()
        pure_gate_path = PurePosixPath(gate_path)
        if not pure_gate_path.is_absolute() or '..' in pure_gate_path.parts or str(pure_gate_path) != gate_path:
            fail()
        deadline = time.monotonic() + gate_timeout
        gate_bytes = None
        while time.monotonic() < deadline:
            try:
                gate_fd = os.open(
                    Path(pure_gate_path),
                    os.O_RDONLY
                    | getattr(os, 'O_NOFOLLOW', 0)
                    | getattr(os, 'O_CLOEXEC', 0),
                )
            except FileNotFoundError:
                time.sleep(0.01)
                continue
            except OSError:
                fail()
            try:
                gate_before = os.fstat(gate_fd)
                if (
                    not stat.S_ISREG(gate_before.st_mode)
                    or gate_before.st_uid != os.geteuid()
                    or gate_before.st_nlink != 1
                    or stat.S_IMODE(gate_before.st_mode) != 0o600
                    or not 0 < gate_before.st_size <= 64 * 1024
                ):
                    fail()
                candidate = b''
                while len(candidate) <= 64 * 1024:
                    block = os.read(gate_fd, min(4096, 64 * 1024 + 1 - len(candidate)))
                    if not block:
                        break
                    candidate += block
                gate_after = os.fstat(gate_fd)
                if (
                    len(candidate) > 64 * 1024
                    or (
                        gate_before.st_dev,
                        gate_before.st_ino,
                        gate_before.st_size,
                        gate_before.st_mtime_ns,
                        gate_before.st_ctime_ns,
                    )
                    != (
                        gate_after.st_dev,
                        gate_after.st_ino,
                        gate_after.st_size,
                        gate_after.st_mtime_ns,
                        gate_after.st_ctime_ns,
                    )
                ):
                    fail()
                gate_bytes = candidate
            finally:
                os.close(gate_fd)
            break
        if gate_bytes is None:
            fail()
        try:
            release = json.loads(gate_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            fail()
        token = release.get('gate_binding_token_hex') if isinstance(release, dict) else None
        if (
            not isinstance(release, dict)
            or canonical(release) != gate_bytes
            or set(release) != {
                'gate_binding_token_hex',
                'challenge_nonce_hex',
                'challenge_sha256',
                'create_once',
                'drill_id',
                'file_fsynced',
                'file_mode',
                'live_process_observation_sha256',
                'observed_at',
                'ownership_envelope_sha256',
                'parent_directory_fsynced',
                'persisted_path',
                'provider_call_index',
                'released_at',
                'root_owned',
                'run_id',
                'schema_version',
            }
            or release.get('schema_version') != 'vaxreplay.managed-clinical-real-kvm-observation-gate.dev-v0.1'
            or release.get('drill_id') != gate.get('drill_id')
            or release.get('challenge_nonce_hex') != gate.get('challenge_nonce_hex')
            or release.get('run_id') != run_id
            or release.get('provider_call_index') != 0
            or release.get('persisted_path') != gate_path
            or not isinstance(token, str)
            or len(token) != 64
            or any(character not in '0123456789abcdef' for character in token)
            or hashlib.sha256(bytes.fromhex(token)).hexdigest() != gate.get('binding_token_sha256')
            or release.get('create_once') is not True
            or release.get('root_owned') is not True
            or release.get('file_mode') != 0o600
            or release.get('file_fsynced') is not True
            or release.get('parent_directory_fsynced') is not True
        ):
            fail()
    elif gate is not None and index != 0:
        if not isinstance(gate, dict) or gate.get('provider_call_index') != 0:
            fail()
    content = turn.get('content')
    if not isinstance(content, str) or not content:
        fail()
    request_id = 'fixture-' + run_id + '-' + str(index)
    response_material = canonical({'content': content, 'request_id': request_id})
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    result = {
        'content': content,
        'finished_at': now,
        'http_status': 200,
        'provider_cost_usd': None,
        'provider_reported_model_id': plan['provider_model_id'],
        'provider_request_bytes': len(request_bytes),
        'provider_request_id': request_id,
        'provider_request_sha256': hashlib.sha256(request_bytes).hexdigest(),
        'provider_response_bytes': len(response_material),
        'provider_response_sha256': hashlib.sha256(response_material).hexdigest(),
        'resolved_model_id': plan['provider_model_id'],
        'schema_version': 'vaxreplay.provider-call-result.v0.1',
        'started_at': now,
        'stop_reason': 'completed',
        'usage': {
            'input_tokens': turn['input_tokens'],
            'output_tokens': turn['output_tokens'],
            'reasoning_tokens': turn['reasoning_tokens'],
        },
    }
    response = {
        'error_code': None,
        'result': result,
        'schema_version': 'vaxreplay.provider-subprocess-response.dev-v0.1',
        'succeeded': True,
    }
    sys.stdout.buffer.write(canonical(response))

if __name__ == '__main__':
    main()
"""
    return f'#!{interpreter} -ISB\n'.encode('utf-8') + body.encode('utf-8')


def _close_descriptor_best_effort(descriptor: int) -> bool:
    try:
        os.close(descriptor)
    except OSError:
        return False
    return True


def _write_create_once(path: Path, payload: bytes, *, mode: int = 0o600) -> _CreatedLivePath:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor: int | None = None
    open_attempted = False
    created_identity: tuple[int, int] | None = None
    created_owner: tuple[int, int] | None = None
    created_mode: int | None = None
    try:
        open_attempted = True
        descriptor = os.open(path, flags, mode)
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        created_owner = (created.st_uid, created.st_gid)
        created_mode = stat.S_IMODE(created.st_mode)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_uid != os.geteuid()
            or created.st_nlink != 1
            or stat.S_IMODE(created.st_mode) != mode
        ):
            raise ManagedClinicalRealKvmDrillError('create-once drill file has unsafe initial metadata')
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        closing_descriptor = descriptor
        descriptor = None
        if not _close_descriptor_best_effort(closing_descriptor):
            raise _ManagedClinicalCleanupIncompleteError(
                'create-once drill file descriptor could not be closed; ambiguous state was preserved'
            )
        _fsync_directory(path.parent)
        retained = path.lstat()
        if (
            (retained.st_dev, retained.st_ino) != created_identity
            or not stat.S_ISREG(retained.st_mode)
            or (retained.st_uid, retained.st_gid) != created_owner
            or retained.st_nlink != 1
            or stat.S_IMODE(retained.st_mode) != mode
        ):
            raise ManagedClinicalRealKvmDrillError('create-once drill file changed identity after persistence')
        return _CreatedLivePath(
            path=path,
            device_id=retained.st_dev,
            inode=retained.st_ino,
            owner_uid=retained.st_uid,
            owner_gid=retained.st_gid,
            mode=stat.S_IMODE(retained.st_mode),
            kind='regular_file',
        )
    except BaseException as error:
        close_failed = False
        if descriptor is not None:
            close_failed = not _close_descriptor_best_effort(descriptor)
            descriptor = None
        identity_changed = False
        removed_created_file = False
        if created_identity is not None:
            try:
                retained = path.lstat()
            except FileNotFoundError:
                identity_changed = True
            except OSError:
                identity_changed = True
            else:
                if (
                    (retained.st_dev, retained.st_ino) == created_identity
                    and stat.S_ISREG(retained.st_mode)
                    and (retained.st_uid, retained.st_gid) == created_owner
                    and retained.st_nlink == 1
                    and stat.S_IMODE(retained.st_mode) == created_mode
                ):
                    try:
                        path.unlink()
                        removed_created_file = True
                    except OSError:
                        identity_changed = True
                else:
                    identity_changed = True
        if removed_created_file:
            try:
                _fsync_directory(path.parent)
            except OSError:
                identity_changed = True
        if created_identity is None and open_attempted:
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            except OSError:
                identity_changed = True
            else:
                identity_changed = True
        if identity_changed or close_failed or isinstance(error, _ManagedClinicalCleanupIncompleteError):
            raise _ManagedClinicalCleanupIncompleteError(
                'create-once drill file failed and foreign or changed state was preserved'
            ) from error
        if isinstance(error, ManagedClinicalRealKvmDrillError):
            raise
        raise ManagedClinicalRealKvmDrillError('create-once drill file could not be persisted') from error
    finally:
        if descriptor is not None:
            _close_descriptor_best_effort(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_directory_names(
    path: Path,
    *,
    maximum_entries: int,
    label: str,
) -> frozenset[str]:
    names: set[str] = set()
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if len(names) >= maximum_entries:
                    raise ManagedClinicalRealKvmDrillError(f'{label} exceeds its fixed entry bound')
                names.add(entry.name)
    except OSError:
        raise ManagedClinicalRealKvmDrillError(f'{label} is unavailable') from None
    return frozenset(names)


def _private_directory(path: Path, *, create: bool = True) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    resolved = path.resolve(strict=True)
    metadata = resolved.lstat()
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalRealKvmDrillError('drill directories must be owned, normalized, and mode-0700')
    return resolved


def _require_root_directory_path(
    path: Path,
    *,
    exact_mode: int | None = None,
) -> Path:
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ManagedClinicalRealKvmDrillError('live directory paths must be normalized and absolute')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            raise ManagedClinicalRealKvmDrillError('live directory path is unavailable') from None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ManagedClinicalRealKvmDrillError(
                'live directories and every ancestor must be root-owned non-writable directories'
            )
    metadata = path.lstat()
    if exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode:
        raise ManagedClinicalRealKvmDrillError('live private directory has the wrong exact mode')
    return path


def _postvalidate_created_directory(
    path: Path,
    *,
    directory_descriptor: int,
    created_identity: tuple[int, int],
    created_owner: tuple[int, int],
    mode: int,
) -> _CreatedLivePath:
    retained_descriptor = os.fstat(directory_descriptor)
    retained_name = path.lstat()
    if (
        not stat.S_ISDIR(retained_descriptor.st_mode)
        or not stat.S_ISDIR(retained_name.st_mode)
        or (retained_descriptor.st_dev, retained_descriptor.st_ino) != created_identity
        or (retained_name.st_dev, retained_name.st_ino) != created_identity
        or (retained_descriptor.st_uid, retained_descriptor.st_gid) != created_owner
        or (retained_name.st_uid, retained_name.st_gid) != created_owner
        or stat.S_IMODE(retained_descriptor.st_mode) != mode
        or stat.S_IMODE(retained_name.st_mode) != mode
    ):
        raise ManagedClinicalRealKvmDrillError('fresh challenge directory changed identity after persistence')
    return _CreatedLivePath(
        path=path,
        device_id=retained_descriptor.st_dev,
        inode=retained_descriptor.st_ino,
        owner_uid=retained_descriptor.st_uid,
        owner_gid=retained_descriptor.st_gid,
        mode=stat.S_IMODE(retained_descriptor.st_mode),
        kind='directory',
    )


def _create_root_directory(path: Path, *, mode: int = 0o700) -> _CreatedLivePath:
    if path.exists() or path.is_symlink():
        raise ManagedClinicalRealKvmDrillError('fresh challenge directory already exists')
    _require_root_directory_path(path.parent)
    directory_descriptor: int | None = None
    mkdir_attempted = False
    created_name = False
    created_identity: tuple[int, int] | None = None
    created_owner: tuple[int, int] | None = None
    cleanup_mode: int | None = None
    try:
        mkdir_attempted = True
        os.mkdir(path, mode)
        created_name = True
        named_after_mkdir = path.lstat()
        created_identity = (named_after_mkdir.st_dev, named_after_mkdir.st_ino)
        created_owner = (named_after_mkdir.st_uid, named_after_mkdir.st_gid)
        cleanup_mode = stat.S_IMODE(named_after_mkdir.st_mode)
        if not stat.S_ISDIR(named_after_mkdir.st_mode) or named_after_mkdir.st_uid != os.geteuid():
            raise ManagedClinicalRealKvmDrillError('fresh challenge directory has unsafe initial metadata')
        directory_descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0),
        )
        created = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(created.st_mode)
            or (created.st_dev, created.st_ino) != created_identity
            or (created.st_uid, created.st_gid) != created_owner
        ):
            raise ManagedClinicalRealKvmDrillError('fresh challenge directory changed identity while opening')
        os.fchmod(directory_descriptor, mode)
        cleanup_mode = mode
        os.fsync(directory_descriptor)
        _fsync_directory(path.parent)
        result = _postvalidate_created_directory(
            path,
            directory_descriptor=directory_descriptor,
            created_identity=created_identity,
            created_owner=created_owner,
            mode=mode,
        )
        closing_descriptor = directory_descriptor
        directory_descriptor = None
        if not _close_descriptor_best_effort(closing_descriptor):
            raise _ManagedClinicalCleanupIncompleteError(
                'fresh challenge directory descriptor could not be closed; ambiguous state was preserved'
            )
        return result
    except BaseException as error:
        close_failed = False
        if directory_descriptor is not None:
            close_failed = not _close_descriptor_best_effort(directory_descriptor)
            directory_descriptor = None
        if created_identity is not None:
            try:
                observed = path.lstat()
            except FileNotFoundError:
                raise _ManagedClinicalCleanupIncompleteError(
                    'fresh challenge directory disappeared after creation; ambiguous state was preserved'
                ) from error
            except OSError:
                raise _ManagedClinicalCleanupIncompleteError(
                    'fresh challenge directory became unavailable; ambiguous state was preserved'
                ) from error
            else:
                if (
                    stat.S_ISDIR(observed.st_mode)
                    and (observed.st_dev, observed.st_ino) == created_identity
                    and (observed.st_uid, observed.st_gid) == created_owner
                    and stat.S_IMODE(observed.st_mode) == cleanup_mode
                ):
                    try:
                        path.rmdir()
                        _fsync_directory(path.parent)
                    except OSError:
                        raise _ManagedClinicalCleanupIncompleteError(
                            'fresh challenge directory failed and could not be removed'
                        ) from None
                else:
                    raise _ManagedClinicalCleanupIncompleteError(
                        'fresh challenge directory changed identity; foreign state was preserved'
                    ) from error
        elif created_name:
            raise _ManagedClinicalCleanupIncompleteError(
                'fresh challenge directory identity could not be established; ambiguous state was preserved'
            ) from error
        elif mkdir_attempted:
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            except OSError:
                raise _ManagedClinicalCleanupIncompleteError(
                    'fresh challenge directory mkdir outcome is unavailable; ambiguous state was preserved'
                ) from error
            else:
                raise _ManagedClinicalCleanupIncompleteError(
                    'fresh challenge directory mkdir outcome is ambiguous; untracked state was preserved'
                ) from error
        if close_failed:
            raise _ManagedClinicalCleanupIncompleteError(
                'fresh challenge directory descriptor close failed; ambiguous state was preserved'
            ) from error
        if isinstance(error, ManagedClinicalRealKvmDrillError):
            raise
        raise ManagedClinicalRealKvmDrillError('fresh challenge directory could not be created') from error
    finally:
        if directory_descriptor is not None:
            _close_descriptor_best_effort(directory_descriptor)


def _ensure_private_namespace_parent(
    path: Path,
    *,
    created_directories: list[_CreatedLivePath],
) -> Path:
    try:
        path.lstat()
    except FileNotFoundError:
        parent = path.parent
        try:
            parent.lstat()
        except FileNotFoundError:
            _ensure_private_namespace_parent(
                parent,
                created_directories=created_directories,
            )
        created_directories.append(_create_root_directory(path))
        return path
    except OSError:
        raise ManagedClinicalRealKvmDrillError('prelaunch namespace parent is unavailable') from None
    return _require_root_directory_path(path, exact_mode=0o700)


def _create_tracked_live_directory(
    path: Path,
    *,
    created_directories: list[_CreatedLivePath],
) -> Path:
    created_directories.append(_create_root_directory(path))
    return path


def _rollback_created_live_paths(created: list[_CreatedLivePath]) -> tuple[str, ...]:
    """Remove only exact paths created by this prelaunch, leaf-first and without recursion.

    Threat boundary: the final identity check and pathname ``unlink``/``rmdir`` are separate
    syscalls.  Production therefore requires these namespaces and every ancestor to remain
    root-owned, mode-0700/non-writable, and exclusively controlled by the trusted collector.  A
    second hostile UID-0 writer could swap a name in that final syscall window and is outside this
    evidence boundary.  Every ambiguity we can observe is preserved and reported incomplete.
    """

    issues: list[str] = []
    for item in reversed(created):
        try:
            metadata = item.path.lstat()
        except FileNotFoundError:
            issues.append('created-path-missing')
            continue
        except OSError:
            issues.append('created-path-unavailable')
            continue
        expected_kind = stat.S_ISDIR(metadata.st_mode) if item.kind == 'directory' else stat.S_ISREG(metadata.st_mode)
        expected_mode = 0o700 if item.kind == 'directory' else 0o600
        if (
            not expected_kind
            or (metadata.st_dev, metadata.st_ino) != (item.device_id, item.inode)
            or (metadata.st_uid, metadata.st_gid) != (item.owner_uid, item.owner_gid)
            or stat.S_IMODE(metadata.st_mode) != item.mode
            or item.mode != expected_mode
        ):
            issues.append('foreign-or-changed-path-preserved')
            continue
        try:
            if item.kind == 'directory':
                item.path.rmdir()
            else:
                item.path.unlink()
            _fsync_directory(item.path.parent)
        except OSError:
            issues.append('nonempty-or-unremovable-path-preserved')
    return tuple(issues)


def _initialize_live_paths(
    paths: DrillPaths,
    *,
    authorization: DrillAuthorization,
) -> None:
    # This check must precede every filesystem mutation in this transaction.
    _require_linux_pathname_socket_path(paths.registry_socket)
    if authorization.challenge_issued_at.tzinfo is None or authorization.challenge_issued_at.utcoffset() is None:
        raise ManagedClinicalRealKvmDrillError('prelaunch authorization time must include a UTC offset')
    receipt = ManagedClinicalRealKvmChallengeAuthorizationReceipt(
        drill_id=authorization.drill_id,
        challenge_nonce_hex=authorization.challenge_nonce_hex,
        challenge_issued_at=authorization.challenge_issued_at.astimezone(UTC),
        release_pins_sha256=authorization.release_pins_sha256,
        challenge_sha256=authorization.challenge_sha256,
        registry_authority_id=authorization.registry_authority_id,
        deployment_id=authorization.deployment_id,
        registered_entry_id=authorization.registered_entry_id,
        observation_gate_path=str(authorization.observation_gate_path),
        observation_gate_binding_token_sha256=_sha256_bytes(authorization.observation_gate_binding_token),
        state_namespace=str(paths.root),
    )
    receipt_bytes = canonical_json_bytes(receipt)
    try:
        reloaded_receipt = ManagedClinicalRealKvmChallengeAuthorizationReceipt.model_validate_json(receipt_bytes)
    except ValueError:
        raise ManagedClinicalRealKvmDrillError('prelaunch authorization receipt is not valid canonical JSON') from None
    if canonical_json_bytes(reloaded_receipt) != receipt_bytes or not receipt_bytes:
        raise ManagedClinicalRealKvmDrillError('prelaunch authorization receipt is not exact canonical JSON')
    created: list[_CreatedLivePath] = []
    stage = 'occupancy-preflight'
    try:
        occupied = (
            FIXED_CONFIG_ROOT,
            authorization.observation_gate_path,
            paths.root,
            paths.registry_socket,
        )
        if any(path.exists() or path.is_symlink() for path in occupied):
            raise ManagedClinicalRealKvmDrillError('one fixed prelaunch path is already occupied')
        stage = 'shared-namespace-parents'
        for parent in (STATE_PARENT, RUNTIME_SOCKET_PARENT, OBSERVATION_GATE_PARENT):
            _ensure_private_namespace_parent(
                parent,
                created_directories=created,
            )
        stage = 'challenge-state-root'
        _create_tracked_live_directory(paths.root, created_directories=created)
        stage = 'challenge-state-children'
        for directory in (
            paths.private_root,
            paths.workspace_root.parent,
            paths.evidence_root,
            paths.ownership_root,
            paths.startup_receipt_root,
            paths.gateway_database.parent,
            paths.registry_database.parent,
            paths.protocol_audit_root,
            paths.provider_child.parent,
        ):
            if directory == paths.root:
                continue
            try:
                directory.lstat()
            except FileNotFoundError:
                _create_tracked_live_directory(
                    directory,
                    created_directories=created,
                )
            except OSError:
                raise ManagedClinicalRealKvmDrillError('prelaunch child path is unavailable') from None
            else:
                raise ManagedClinicalRealKvmDrillError('prelaunch child path unexpectedly already exists')
        stage = 'authorization-receipt-write'
        created.append(_write_create_once(paths.authorization_receipt, receipt_bytes))
        stage = 'authorization-receipt-reload'
        if (
            _read_pinned_file(
                paths.authorization_receipt,
                expected_sha256=_sha256_bytes(receipt_bytes),
                maximum_bytes=64 * 1024,
                require_root_owner=True,
            )
            != receipt_bytes
        ):
            raise ManagedClinicalRealKvmDrillError('prelaunch authorization receipt changed after persistence')
    except BaseException as error:
        cleanup_issues = _rollback_created_live_paths(created)
        cleanup_incomplete = bool(cleanup_issues) or isinstance(error, _ManagedClinicalCleanupIncompleteError)
        cleanup = 'incomplete-foreign-state-preserved' if cleanup_incomplete else 'complete'
        raise ManagedClinicalRealKvmDrillError(
            'unauthenticated prelaunch failure; '
            f'stage={stage}; cleanup={cleanup}; vm_started=false; registry_started=false; provider_started=false'
        ) from error


def _require_linux_root_kvm() -> None:
    if sys.platform != 'linux' or platform.system() != 'Linux' or os.geteuid() != 0:
        raise ManagedClinicalRealKvmDrillError('managed real-Firecracker collection requires effective UID 0 on Linux')
    try:
        kvm = Path('/dev/kvm').lstat()
        controllers = Path('/sys/fs/cgroup/cgroup.controllers').read_text(encoding='ascii').split()
    except OSError:
        raise ManagedClinicalRealKvmDrillError('managed collection requires usable /dev/kvm and cgroup v2') from None
    if (
        stat.S_ISLNK(kvm.st_mode)
        or not stat.S_ISCHR(kvm.st_mode)
        or not os.access('/dev/kvm', os.R_OK | os.W_OK)
        or not {'cpu', 'memory', 'pids'}.issubset(controllers)
    ):
        raise ManagedClinicalRealKvmDrillError(
            'managed collection requires a usable KVM character device and cgroup v2 controllers'
        )


def _read_pinned_file(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int = MAX_INPUT_BYTES,
    require_root_owner: bool,
    allow_empty: bool = False,
) -> bytes:
    _require_sha256(expected_sha256, 'external file pin')
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ManagedClinicalRealKvmDrillError('pinned input cannot be a symbolic link')
    try:
        resolved = supplied.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise ManagedClinicalRealKvmDrillError('pinned input is unavailable') from None
    allowed_uid = 0 if require_root_owner else os.geteuid()
    if (
        resolved != supplied
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in ({0} if require_root_owner else {0, allowed_uid})
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or metadata.st_size > maximum_bytes
        or (metadata.st_size == 0 and not allow_empty)
    ):
        raise ManagedClinicalRealKvmDrillError('pinned input has unsafe metadata')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ManagedClinicalRealKvmDrillError('pinned input changed while opening')
        body = bytearray()
        while len(body) <= maximum_bytes:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(body)))
            if not block:
                break
            body.extend(block)
    finally:
        os.close(descriptor)
    if len(body) > maximum_bytes or not hmac.compare_digest(_sha256_bytes(bytes(body)), expected_sha256):
        raise ManagedClinicalRealKvmDrillError('pinned input differs from its external digest')
    return bytes(body)


def _load_model(
    path: Path,
    *,
    expected_sha256: str,
    model: type[ModelT],
    require_root_owner: bool,
) -> ModelT:
    body = _read_pinned_file(
        path,
        expected_sha256=expected_sha256,
        require_root_owner=require_root_owner,
    )
    try:
        value = model.model_validate_json(body)
    except ValueError:
        raise ManagedClinicalRealKvmDrillError('pinned input has an invalid strict schema') from None
    if canonical_json_bytes(value) != body:
        raise ManagedClinicalRealKvmDrillError('pinned input is not exact canonical JSON')
    return value


def _add_public_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--worker-spec', required=True, type=Path)
    parser.add_argument('--expected-worker-spec-sha256', required=True)
    parser.add_argument('--execution-policy', required=True, type=Path)
    parser.add_argument('--expected-execution-policy-sha256', required=True)
    parser.add_argument('--guest-rpc-policy', required=True, type=Path)
    parser.add_argument('--expected-guest-rpc-policy-sha256', required=True)
    parser.add_argument('--guest-config', required=True, type=Path)
    parser.add_argument('--expected-guest-config-sha256', required=True)
    parser.add_argument('--disk-build-receipt', required=True, type=Path)
    parser.add_argument('--expected-disk-build-receipt-sha256', required=True)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--expected-task-sha256', required=True)
    parser.add_argument('--qualification-root', required=True, type=Path)
    parser.add_argument('--qualification-key-file', required=True, type=Path)
    parser.add_argument('--expected-qualification-key-id', required=True)
    parser.add_argument('--expected-qualification-artifact-sha256', required=True)
    parser.add_argument('--expected-qualification-collector-evidence-sha256', required=True)
    parser.add_argument('--expected-qualification-probe-manifest-sha256', required=True)
    parser.add_argument('--expected-qualification-runtime-closure-manifest-sha256', required=True)
    parser.add_argument('--expected-qualification-runtime-closure-receipt-sha256', required=True)
    parser.add_argument('--expected-qualification-runtime-closure-sha256', required=True)
    parser.add_argument('--expected-qualification-collector-public-key-hex', required=True)
    parser.add_argument('--expected-qualification-collector-key-id', required=True)
    parser.add_argument('--expected-qualification-verifier-source-sha256', required=True)
    parser.add_argument('--collector-runtime-closure-root', required=True, type=Path)
    parser.add_argument('--expected-collector-runtime-closure-manifest-sha256', required=True)
    parser.add_argument('--expected-collector-runtime-closure-receipt-sha256', required=True)
    parser.add_argument('--expected-collector-runtime-closure-sha256', required=True)
    parser.add_argument('--expected-collector-entrypoint-sha256', required=True)
    parser.add_argument('--expected-collector-interpreter-sha256', required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    dry = subparsers.add_parser(
        'dry-run-config',
        help='generate and exercise the deterministic provider plan without Linux/KVM or fixed /etc',
    )
    dry.add_argument('--task', required=True, type=Path)
    dry.add_argument('--expected-task-sha256', required=True)
    dry.add_argument('--output-root', required=True, type=Path)
    dry.add_argument('--interpreter', default=Path(sys.executable), type=Path)
    prepare = subparsers.add_parser(
        'prepare-live',
        help=(
            'compute the exact gated plan, complete stable pins, challenge, and '
            'isolated collect invocation without launching KVM or writing /etc'
        ),
    )
    _add_public_input_arguments(prepare)
    prepare.add_argument('--drill-id', required=True)
    prepare.add_argument('--challenge-nonce-hex', required=True)
    prepare.add_argument('--challenge-issued-at', required=True)
    prepare.add_argument('--collector-key-file', required=True, type=Path)
    prepare.add_argument('--expected-collector-public-key-hex', required=True)
    prepare.add_argument('--expected-collector-key-id', required=True)
    prepare.add_argument('--expected-bootstrap-authorization-key-id', required=True)
    prepare.add_argument('--bootstrap-authorization-seed-file', required=True, type=Path)
    prepare.add_argument('--output-root', required=True, type=Path)
    collect = subparsers.add_parser(
        'collect',
        help='run the privileged fixed-deployment managed+real-Firecracker drill',
    )
    _add_public_input_arguments(collect)
    collect.add_argument('--drill-id', required=True)
    collect.add_argument('--challenge-nonce-hex', required=True)
    collect.add_argument('--challenge-issued-at', required=True)
    collect.add_argument('--expected-release-pins-sha256', required=True)
    collect.add_argument('--expected-challenge-sha256', required=True)
    collect.add_argument('--bootstrap-authorization-seed-file', required=True, type=Path)
    collect.add_argument('--collector-key-file', required=True, type=Path)
    collect.add_argument('--expected-collector-public-key-hex', required=True)
    collect.add_argument('--expected-collector-key-id', required=True)
    collect.add_argument('--expected-bootstrap-authorization-key-id', required=True)
    return parser


def _dry_run_config(arguments: argparse.Namespace) -> dict[str, object]:
    task = _load_model(
        arguments.task,
        expected_sha256=arguments.expected_task_sha256,
        model=ExecutionTask,
        require_root_owner=False,
    )
    output_root = arguments.output_root.expanduser().resolve(strict=False)
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ManagedClinicalRealKvmDrillError('dry-run config output root must be one new normalized absolute path')
    output_root.mkdir(mode=0o700, parents=False)
    plan = _canonical_provider_plan(task)
    interpreter = arguments.interpreter.expanduser().resolve(strict=True)
    child = render_deterministic_provider_child(interpreter)
    plan_path = output_root / 'provider-plan.json'
    child_path = output_root / 'provider-child'
    _write_create_once(plan_path, plan)
    _write_create_once(child_path, child, mode=0o500)
    plan_sha256 = _sha256_bytes(plan)
    child_sha256 = _sha256_bytes(child)
    provider_spec = ProviderSubprocessSpec(
        executable_path=str(child_path),
        executable_sha256=child_sha256,
        argv_suffix=(
            '--plan',
            str(plan_path),
            '--expected-plan-sha256',
            plan_sha256,
        ),
        maximum_call_seconds=5,
    )
    receipt = {
        'schema_version': 'vaxreplay.managed-real-kvm-dry-run-config.dev-v0.1',
        'task_sha256': arguments.expected_task_sha256,
        'provider_plan_path': str(plan_path),
        'provider_plan_sha256': plan_sha256,
        'provider_child_path': str(child_path),
        'provider_child_sha256': child_sha256,
        'provider_subprocess': provider_spec.model_dump(mode='json'),
        'provider_turn_count': 4,
        'external_provider_called': False,
        'credential_read_by_fixture_source': False,
        'linux_kvm_run_performed': False,
        'fixed_deployment_written': False,
        'development_only': True,
    }
    receipt_path = output_root / 'DRY-RUN-RECEIPT.json'
    _write_create_once(receipt_path, canonical_json_bytes(receipt))
    return {**receipt, 'receipt_path': str(receipt_path)}


def _collect_exec_argv(
    arguments: argparse.Namespace,
    *,
    inputs: PublicInputs,
    authorization: DrillAuthorization,
) -> tuple[str, ...]:
    entrypoint = inputs.collector_runtime_closure.manifest.driver_entrypoint_path
    interpreter = inputs.collector_runtime_closure.manifest.interpreter_path
    values = (
        ('--worker-spec', arguments.worker_spec),
        ('--expected-worker-spec-sha256', arguments.expected_worker_spec_sha256),
        ('--execution-policy', arguments.execution_policy),
        (
            '--expected-execution-policy-sha256',
            arguments.expected_execution_policy_sha256,
        ),
        ('--guest-rpc-policy', arguments.guest_rpc_policy),
        (
            '--expected-guest-rpc-policy-sha256',
            arguments.expected_guest_rpc_policy_sha256,
        ),
        ('--guest-config', arguments.guest_config),
        ('--expected-guest-config-sha256', arguments.expected_guest_config_sha256),
        ('--disk-build-receipt', arguments.disk_build_receipt),
        (
            '--expected-disk-build-receipt-sha256',
            arguments.expected_disk_build_receipt_sha256,
        ),
        ('--task', arguments.task),
        ('--expected-task-sha256', arguments.expected_task_sha256),
        ('--qualification-root', arguments.qualification_root),
        ('--qualification-key-file', arguments.qualification_key_file),
        (
            '--expected-qualification-key-id',
            arguments.expected_qualification_key_id,
        ),
        (
            '--expected-qualification-artifact-sha256',
            arguments.expected_qualification_artifact_sha256,
        ),
        (
            '--expected-qualification-collector-evidence-sha256',
            arguments.expected_qualification_collector_evidence_sha256,
        ),
        (
            '--expected-qualification-probe-manifest-sha256',
            arguments.expected_qualification_probe_manifest_sha256,
        ),
        (
            '--expected-qualification-runtime-closure-manifest-sha256',
            arguments.expected_qualification_runtime_closure_manifest_sha256,
        ),
        (
            '--expected-qualification-runtime-closure-receipt-sha256',
            arguments.expected_qualification_runtime_closure_receipt_sha256,
        ),
        (
            '--expected-qualification-runtime-closure-sha256',
            arguments.expected_qualification_runtime_closure_sha256,
        ),
        (
            '--expected-qualification-collector-public-key-hex',
            arguments.expected_qualification_collector_public_key_hex,
        ),
        (
            '--expected-qualification-collector-key-id',
            arguments.expected_qualification_collector_key_id,
        ),
        (
            '--expected-qualification-verifier-source-sha256',
            arguments.expected_qualification_verifier_source_sha256,
        ),
        (
            '--collector-runtime-closure-root',
            arguments.collector_runtime_closure_root,
        ),
        (
            '--expected-collector-runtime-closure-manifest-sha256',
            arguments.expected_collector_runtime_closure_manifest_sha256,
        ),
        (
            '--expected-collector-runtime-closure-receipt-sha256',
            arguments.expected_collector_runtime_closure_receipt_sha256,
        ),
        (
            '--expected-collector-runtime-closure-sha256',
            arguments.expected_collector_runtime_closure_sha256,
        ),
        (
            '--expected-collector-entrypoint-sha256',
            arguments.expected_collector_entrypoint_sha256,
        ),
        (
            '--expected-collector-interpreter-sha256',
            arguments.expected_collector_interpreter_sha256,
        ),
        ('--drill-id', authorization.drill_id),
        ('--challenge-nonce-hex', authorization.challenge_nonce_hex),
        (
            '--challenge-issued-at',
            authorization.challenge_issued_at.isoformat().replace('+00:00', 'Z'),
        ),
        ('--expected-release-pins-sha256', authorization.release_pins_sha256),
        ('--expected-challenge-sha256', authorization.challenge_sha256),
        (
            '--bootstrap-authorization-seed-file',
            arguments.bootstrap_authorization_seed_file,
        ),
        ('--collector-key-file', arguments.collector_key_file),
        (
            '--expected-collector-public-key-hex',
            arguments.expected_collector_public_key_hex,
        ),
        ('--expected-collector-key-id', arguments.expected_collector_key_id),
        (
            '--expected-bootstrap-authorization-key-id',
            arguments.expected_bootstrap_authorization_key_id,
        ),
    )
    flattened = tuple(item for option, value in values for item in (option, str(value)))
    return (
        interpreter,
        '-I',
        '-B',
        entrypoint,
        'collect',
        *flattened,
    )


def _prepare_live(
    arguments: argparse.Namespace,
    *,
    preverified_collector_runtime_closure: LoadedQualificationDriverRuntimeClosure,
) -> dict[str, object]:
    _require_linux_root_kvm()
    inputs = _load_public_inputs(
        arguments,
        preverified_collector_runtime_closure=(preverified_collector_runtime_closure),
    )
    _verify_loaded_module_runtime_binding(inputs.collector_runtime_closure)
    collector_private_key = _load_collector_private_key(arguments)
    _load_bootstrap_seed(
        arguments.bootstrap_authorization_seed_file,
        guest_config=inputs.guest_config,
    )
    gate_path, binding_token = _observation_gate_inputs(arguments)
    provider_plan = _canonical_provider_plan(
        inputs.task,
        observation_gate=(
            gate_path,
            binding_token,
            arguments.drill_id,
            arguments.challenge_nonce_hex,
        ),
    )
    provider_child = render_deterministic_provider_child(
        Path(inputs.collector_runtime_closure.manifest.interpreter_path)
    )
    authorization = _challenge_authorization(
        arguments,
        inputs=inputs,
        provider_plan=provider_plan,
        provider_child=provider_child,
        collector_private_key=collector_private_key,
        observation_gate_path=gate_path,
        observation_gate_binding_token=binding_token,
    )
    # Fail before publishing a runnable collect invocation if the task/challenge
    # projection no longer satisfies the frozen execution-cohort schema.
    _execution_cohort(
        inputs.task,
        challenge_sha256=authorization.challenge_sha256,
    )
    # Validate the exact live pathname before publishing any preparation files.
    DrillPaths.live(
        authorization.drill_id,
        authorization.challenge_sha256,
    )
    supplied_output_root = arguments.output_root.expanduser()
    output_root = Path(os.path.abspath(supplied_output_root))
    if (
        not supplied_output_root.is_absolute()
        or supplied_output_root != output_root
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise ManagedClinicalRealKvmDrillError('live preparation output root must be one new normalized absolute path')
    _require_root_directory_path(output_root.parent)
    _create_root_directory(output_root)
    plan_path = output_root / 'provider-plan.json'
    child_path = output_root / 'provider-child'
    external_pins_path = output_root / 'external-pins.json'
    invocation_path = output_root / 'collect-invocation.json'
    _write_create_once(plan_path, provider_plan)
    _write_create_once(child_path, provider_child, mode=0o500)
    external_pins_bytes = canonical_json_bytes(authorization.external_pins)
    _write_create_once(external_pins_path, external_pins_bytes)
    collect_argv = _collect_exec_argv(
        arguments,
        inputs=inputs,
        authorization=authorization,
    )
    invocation = {
        'schema_version': 'vaxreplay.managed-real-kvm-collect-invocation.dev-v0.1',
        'executable': collect_argv[0],
        'argv': collect_argv,
        'cwd': '/',
        'environment': {
            'LANG': 'C',
            'LC_ALL': 'C',
            'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
        },
        'effective_uid': 0,
        'requires_linux_kvm': True,
    }
    invocation_bytes = canonical_json_bytes(invocation)
    _write_create_once(invocation_path, invocation_bytes)
    receipt = {
        'schema_version': 'vaxreplay.managed-real-kvm-live-preparation.dev-v0.1',
        'drill_id': authorization.drill_id,
        'challenge_nonce_hex': authorization.challenge_nonce_hex,
        'challenge_issued_at': (authorization.challenge_issued_at.isoformat().replace('+00:00', 'Z')),
        'release_pins_sha256': authorization.release_pins_sha256,
        'release_pin_count': len(_STABLE_RELEASE_PIN_FIELDS),
        'challenge_sha256': authorization.challenge_sha256,
        'provider_plan_path': str(plan_path),
        'provider_plan_sha256': _sha256_bytes(provider_plan),
        'provider_child_path': str(child_path),
        'provider_child_sha256': _sha256_bytes(provider_child),
        'external_pins_path': str(external_pins_path),
        'external_pins_sha256': _sha256_bytes(external_pins_bytes),
        'collect_invocation_path': str(invocation_path),
        'collect_invocation_sha256': _sha256_bytes(invocation_bytes),
        'observation_gate_path': str(authorization.observation_gate_path),
        'observation_gate_binding_token_sha256': _sha256_bytes(authorization.observation_gate_binding_token),
        'linux_kvm_run_performed': False,
        'fixed_deployment_written': False,
        'development_only': True,
        'official_leaderboard_execution_qualified': False,
    }
    receipt_path = output_root / 'PREPARE-LIVE-RECEIPT.json'
    receipt_bytes = canonical_json_bytes(receipt)
    _write_create_once(receipt_path, receipt_bytes)
    return {
        **receipt,
        'receipt_path': str(receipt_path),
        'receipt_sha256': _sha256_bytes(receipt_bytes),
    }


def _read_private_file(path: Path, *, maximum_bytes: int = 4096) -> bytes:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ManagedClinicalRealKvmDrillError('private input cannot be a symbolic link')
    try:
        resolved = supplied.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise ManagedClinicalRealKvmDrillError('private input is unavailable') from None
    if (
        resolved != supplied
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= maximum_bytes
    ):
        raise ManagedClinicalRealKvmDrillError('live private inputs must be root-owned, single-link, mode-0600 files')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ManagedClinicalRealKvmDrillError('private input changed while opening')
        body = bytearray()
        while len(body) <= maximum_bytes:
            block = os.read(
                descriptor,
                min(4096, maximum_bytes + 1 - len(body)),
            )
            if not block:
                break
            body.extend(block)
        closed = os.fstat(descriptor)
        if len(body) > maximum_bytes or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
            closed.st_size,
            closed.st_mtime_ns,
            closed.st_ctime_ns,
        ):
            raise ManagedClinicalRealKvmDrillError('private input changed while reading')
        return bytes(body)
    finally:
        os.close(descriptor)


def _load_collector_private_key(arguments: argparse.Namespace) -> Ed25519PrivateKey:
    encoded = _read_private_file(arguments.collector_key_file, maximum_bytes=65)
    if encoded.endswith(b'\n'):
        encoded = encoded[:-1]
    if len(encoded) != 64:
        raise ManagedClinicalRealKvmDrillError('collector key must be one 32-byte lowercase hexadecimal seed')
    try:
        seed = bytes.fromhex(encoded.decode('ascii'))
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
    except (UnicodeDecodeError, ValueError):
        raise ManagedClinicalRealKvmDrillError('collector key seed is invalid') from None
    if encoded != seed.hex().encode('ascii'):
        raise ManagedClinicalRealKvmDrillError('collector key seed is not canonical lowercase hexadecimal')
    public_key = private_key.public_key().public_bytes_raw()
    if (
        public_key.hex() != arguments.expected_collector_public_key_hex
        or managed_clinical_real_kvm_collector_key_id(public_key) != arguments.expected_collector_key_id
    ):
        raise ManagedClinicalRealKvmDrillError('collector signing key differs from its external public identity')
    return private_key


def _persist_verifier_keys(
    paths: DrillPaths,
    *,
    keys: DrillKeys,
    qualification_key: bytes,
) -> None:
    root = _private_directory(paths.private_root, create=False)
    if _bounded_directory_names(
        root,
        maximum_entries=1,
        label='challenge-private verifier-key directory',
    ):
        raise ManagedClinicalRealKvmDrillError('challenge-private verifier-key directory is not empty')
    values = {
        'workspace-receipt.key': keys.workspace,
        'worker-attestation.key': keys.worker,
        'gateway-receipt.key': keys.gateway,
        'guest-rpc-receipt.key': keys.guest_rpc,
        'bootstrap-receipt.key': keys.bootstrap_receipt,
        'production-receipt.key': keys.production,
        'launcher-failure-receipt.key': keys.launcher_failure,
        'managed-authority.key': keys.managed,
        'qualification.key': qualification_key,
    }
    if set(values) != _VERIFIER_KEY_FILES or any(len(value) != 32 for value in values.values()):
        raise ManagedClinicalRealKvmDrillError('challenge-private verifier-key inventory is invalid')
    for name in sorted(values):
        _write_create_once(root / name, values[name])
    _fsync_directory(root)


def _load_persisted_verifier_keys(paths: DrillPaths) -> tuple[DrillKeys, bytes]:
    root = _private_directory(paths.private_root, create=False)
    observed = _bounded_directory_names(
        root,
        maximum_entries=len(_VERIFIER_KEY_FILES),
        label='challenge-private verifier-key directory',
    )
    if observed != _VERIFIER_KEY_FILES:
        raise ManagedClinicalRealKvmDrillError('challenge-private verifier-key inventory changed after persistence')
    values = {name: _read_private_file(root / name, maximum_bytes=32) for name in sorted(_VERIFIER_KEY_FILES)}
    if any(len(value) != 32 for value in values.values()):
        raise ManagedClinicalRealKvmDrillError('challenge-private verifier key has the wrong exact length')
    return (
        DrillKeys(
            workspace=values['workspace-receipt.key'],
            worker=values['worker-attestation.key'],
            gateway=values['gateway-receipt.key'],
            guest_rpc=values['guest-rpc-receipt.key'],
            bootstrap_receipt=values['bootstrap-receipt.key'],
            production=values['production-receipt.key'],
            launcher_failure=values['launcher-failure-receipt.key'],
            managed=values['managed-authority.key'],
        ),
        values['qualification.key'],
    )


def _load_public_inputs(
    arguments: argparse.Namespace,
    *,
    preverified_collector_runtime_closure: LoadedQualificationDriverRuntimeClosure,
) -> PublicInputs:
    for name, value in vars(arguments).items():
        if name.startswith('expected_') and name.endswith('sha256'):
            _require_sha256(cast(str, value), name.replace('_', ' '))
    spec, spec_bytes = load_pinned_firecracker_worker_spec(
        arguments.worker_spec,
        expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
    )
    if spec_bytes != _read_pinned_file(
        arguments.worker_spec,
        expected_sha256=arguments.expected_worker_spec_sha256,
        require_root_owner=True,
    ):
        raise ManagedClinicalRealKvmDrillError('worker-spec loaders observed different bytes')
    policy = _load_model(
        arguments.execution_policy,
        expected_sha256=arguments.expected_execution_policy_sha256,
        model=AgenticExecutionPolicy,
        require_root_owner=True,
    )
    rpc_policy = _load_model(
        arguments.guest_rpc_policy,
        expected_sha256=arguments.expected_guest_rpc_policy_sha256,
        model=GuestRpcPolicy,
        require_root_owner=True,
    )
    guest_config = load_lane_a_clinical_guest_config(
        arguments.guest_config,
        expected_sha256=arguments.expected_guest_config_sha256,
    )
    if canonical_json_bytes(guest_config) != _read_pinned_file(
        arguments.guest_config,
        expected_sha256=arguments.expected_guest_config_sha256,
        require_root_owner=True,
    ):
        raise ManagedClinicalRealKvmDrillError('guest-config loaders observed different bytes')
    receipt = load_lane_a_guest_disk_build_receipt(
        arguments.disk_build_receipt,
        expected_receipt_sha256=arguments.expected_disk_build_receipt_sha256,
    )
    if canonical_json_bytes(receipt) != _read_pinned_file(
        arguments.disk_build_receipt,
        expected_sha256=arguments.expected_disk_build_receipt_sha256,
        require_root_owner=True,
    ):
        raise ManagedClinicalRealKvmDrillError('disk-receipt loaders observed different bytes')
    task = _load_model(
        arguments.task,
        expected_sha256=arguments.expected_task_sha256,
        model=ExecutionTask,
        require_root_owner=True,
    )
    if agentic_policy_sha256(policy) != arguments.expected_execution_policy_sha256 or (
        guest_rpc_policy_sha256(rpc_policy) != arguments.expected_guest_rpc_policy_sha256
    ):
        raise ManagedClinicalRealKvmDrillError('policy model differs from exact external bytes')
    qualification_key = decode_firecracker_qualification_key(_read_private_file(arguments.qualification_key_file))
    if firecracker_qualification_key_id(qualification_key) != arguments.expected_qualification_key_id:
        raise ManagedClinicalRealKvmDrillError('qualification key differs from external key ID')
    qualification = load_firecracker_qualification(
        arguments.qualification_root,
        qualification_key=qualification_key,
        expected_qualification_key_id=arguments.expected_qualification_key_id,
        expected_worker_spec_sha256=arguments.expected_worker_spec_sha256,
        expected_artifact_sha256=arguments.expected_qualification_artifact_sha256,
        expected_collector_evidence_sha256=(arguments.expected_qualification_collector_evidence_sha256),
        expected_probe_manifest_sha256=(arguments.expected_qualification_probe_manifest_sha256),
        expected_driver_runtime_closure_manifest_sha256=(
            arguments.expected_qualification_runtime_closure_manifest_sha256
        ),
        expected_driver_runtime_closure_receipt_sha256=(
            arguments.expected_qualification_runtime_closure_receipt_sha256
        ),
        expected_driver_runtime_closure_sha256=(arguments.expected_qualification_runtime_closure_sha256),
        expected_collector_public_key_hex=(arguments.expected_qualification_collector_public_key_hex),
        expected_collector_key_id=arguments.expected_qualification_collector_key_id,
        expected_verifier_source_sha256=(arguments.expected_qualification_verifier_source_sha256),
    )
    raw_bytes = _read_pinned_file(
        arguments.qualification_root / 'collector-evidence.json',
        expected_sha256=arguments.expected_qualification_collector_evidence_sha256,
        require_root_owner=True,
    )
    qualification_raw = AuthenticatedFirecrackerQualificationRawCollection.model_validate_json(raw_bytes)
    closure = verify_qualification_driver_runtime_closure(
        arguments.collector_runtime_closure_root,
        expected_manifest_sha256=(arguments.expected_collector_runtime_closure_manifest_sha256),
        expected_receipt_sha256=(arguments.expected_collector_runtime_closure_receipt_sha256),
        expected_closure_sha256=arguments.expected_collector_runtime_closure_sha256,
    )
    if (
        closure != preverified_collector_runtime_closure
        or Path(closure.root) != arguments.collector_runtime_closure_root
        or closure.manifest.driver_entrypoint_sha256 != arguments.expected_collector_entrypoint_sha256
        or closure.manifest.interpreter_sha256 != arguments.expected_collector_interpreter_sha256
        or Path(closure.manifest.driver_entrypoint_path) != Path(sys.argv[0])
        or Path(closure.manifest.interpreter_path) != Path(sys.executable)
        or _sha256_file(Path('/proc/self/exe')) != arguments.expected_collector_interpreter_sha256
        or sys.flags.isolated != 1
        or not sys.dont_write_bytecode
    ):
        raise ManagedClinicalRealKvmDrillError(
            'collector process differs from exact entrypoint/interpreter closure pins'
        )
    return PublicInputs(
        spec=spec,
        policy=policy,
        rpc_policy=rpc_policy,
        guest_config=guest_config,
        disk_receipt=receipt,
        task=task,
        qualification=qualification,
        qualification_key=qualification_key,
        qualification_raw=qualification_raw,
        collector_runtime_closure=closure,
    )


def _challenge_authorization(
    arguments: argparse.Namespace,
    *,
    inputs: PublicInputs,
    provider_plan: bytes,
    provider_child: bytes,
    collector_private_key: Ed25519PrivateKey,
    observation_gate_path: Path,
    observation_gate_binding_token: bytes,
) -> DrillAuthorization:
    if (
        len(arguments.drill_id) != 32
        or any(character not in '0123456789abcdef' for character in arguments.drill_id)
        or len(arguments.challenge_nonce_hex) != 64
        or any(character not in '0123456789abcdef' for character in arguments.challenge_nonce_hex)
    ):
        raise ManagedClinicalRealKvmDrillError(
            'organizer drill ID and challenge nonce must be exact lowercase hexadecimal values'
        )
    expected_release_pins_sha256 = getattr(
        arguments,
        'expected_release_pins_sha256',
        None,
    )
    expected_challenge_sha256 = getattr(
        arguments,
        'expected_challenge_sha256',
        None,
    )
    if (expected_release_pins_sha256 is None) != (expected_challenge_sha256 is None):
        raise ManagedClinicalRealKvmDrillError('organizer release-pins and challenge digests must be supplied together')
    if expected_release_pins_sha256 is not None:
        _require_sha256(
            expected_release_pins_sha256,
            'organizer release-pins digest',
        )
        _require_sha256(
            cast(str, expected_challenge_sha256),
            'organizer challenge digest',
        )
    try:
        issued_at = datetime.fromisoformat(arguments.challenge_issued_at.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        raise ManagedClinicalRealKvmDrillError(
            'challenge-issued-at must be one timezone-aware ISO-8601 timestamp'
        ) from None
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ManagedClinicalRealKvmDrillError('challenge-issued-at must include a UTC offset')
    issued_at = issued_at.astimezone(UTC)
    if issued_at > datetime.now(UTC):
        raise ManagedClinicalRealKvmDrillError('organizer challenge cannot be issued in the future')
    collector_public_key = collector_private_key.public_key().public_bytes_raw()
    if arguments.expected_bootstrap_authorization_key_id != inputs.guest_config.trust_anchor.authorization_key_id:
        raise ManagedClinicalRealKvmDrillError('guest bootstrap trust anchor differs from its external release pin')
    release_pins_sha256 = managed_clinical_real_kvm_release_pins_sha256(
        worker_spec_sha256=arguments.expected_worker_spec_sha256,
        execution_policy_sha256=arguments.expected_execution_policy_sha256,
        guest_rpc_policy_sha256=arguments.expected_guest_rpc_policy_sha256,
        guest_config_sha256=arguments.expected_guest_config_sha256,
        disk_build_receipt_sha256=arguments.expected_disk_build_receipt_sha256,
        qualification_key_id=arguments.expected_qualification_key_id,
        qualification_artifact_sha256=arguments.expected_qualification_artifact_sha256,
        qualification_collector_evidence_sha256=(arguments.expected_qualification_collector_evidence_sha256),
        qualification_probe_manifest_sha256=(arguments.expected_qualification_probe_manifest_sha256),
        qualification_runtime_closure_manifest_sha256=(
            arguments.expected_qualification_runtime_closure_manifest_sha256
        ),
        qualification_runtime_closure_receipt_sha256=(arguments.expected_qualification_runtime_closure_receipt_sha256),
        qualification_runtime_closure_sha256=(arguments.expected_qualification_runtime_closure_sha256),
        qualification_collector_public_key_hex=(arguments.expected_qualification_collector_public_key_hex),
        qualification_collector_key_id=(arguments.expected_qualification_collector_key_id),
        qualification_verifier_source_sha256=(arguments.expected_qualification_verifier_source_sha256),
        task_sha256=arguments.expected_task_sha256,
        provider_child_executable_sha256=_sha256_bytes(provider_child),
        provider_plan_sha256=_sha256_bytes(provider_plan),
        collector_entrypoint_sha256=arguments.expected_collector_entrypoint_sha256,
        collector_interpreter_sha256=arguments.expected_collector_interpreter_sha256,
        collector_runtime_closure_manifest_sha256=(arguments.expected_collector_runtime_closure_manifest_sha256),
        collector_runtime_closure_receipt_sha256=(arguments.expected_collector_runtime_closure_receipt_sha256),
        collector_runtime_closure_sha256=(arguments.expected_collector_runtime_closure_sha256),
        collector_public_key_hex=collector_public_key.hex(),
        collector_key_id=managed_clinical_real_kvm_collector_key_id(collector_public_key),
        launcher_process_executable_sha256=arguments.expected_collector_interpreter_sha256,
        bootstrap_authorization_key_id=(arguments.expected_bootstrap_authorization_key_id),
    )
    challenge_sha256 = managed_clinical_real_kvm_challenge_sha256(
        drill_id=arguments.drill_id,
        challenge_nonce_hex=arguments.challenge_nonce_hex,
        challenge_issued_at=issued_at,
        release_pins_sha256=release_pins_sha256,
    )
    if expected_release_pins_sha256 is not None and (
        not hmac.compare_digest(
            release_pins_sha256,
            expected_release_pins_sha256,
        )
        or not hmac.compare_digest(
            challenge_sha256,
            cast(str, expected_challenge_sha256),
        )
    ):
        raise ManagedClinicalRealKvmDrillError('organizer challenge differs from the complete stable release pins')
    external_pins = ManagedClinicalRealKvmExternalPins(
        drill_id=arguments.drill_id,
        challenge_nonce_hex=arguments.challenge_nonce_hex,
        challenge_issued_at=issued_at,
        release_pins_sha256=release_pins_sha256,
        challenge_sha256=challenge_sha256,
        worker_spec_sha256=arguments.expected_worker_spec_sha256,
        execution_policy_sha256=arguments.expected_execution_policy_sha256,
        guest_rpc_policy_sha256=arguments.expected_guest_rpc_policy_sha256,
        guest_config_sha256=arguments.expected_guest_config_sha256,
        disk_build_receipt_sha256=arguments.expected_disk_build_receipt_sha256,
        qualification_key_id=arguments.expected_qualification_key_id,
        qualification_artifact_sha256=arguments.expected_qualification_artifact_sha256,
        qualification_collector_evidence_sha256=(arguments.expected_qualification_collector_evidence_sha256),
        qualification_probe_manifest_sha256=(arguments.expected_qualification_probe_manifest_sha256),
        qualification_runtime_closure_manifest_sha256=(
            arguments.expected_qualification_runtime_closure_manifest_sha256
        ),
        qualification_runtime_closure_receipt_sha256=(arguments.expected_qualification_runtime_closure_receipt_sha256),
        qualification_runtime_closure_sha256=(arguments.expected_qualification_runtime_closure_sha256),
        qualification_collector_public_key_hex=(arguments.expected_qualification_collector_public_key_hex),
        qualification_collector_key_id=(arguments.expected_qualification_collector_key_id),
        qualification_verifier_source_sha256=(arguments.expected_qualification_verifier_source_sha256),
        task_sha256=arguments.expected_task_sha256,
        provider_child_executable_sha256=_sha256_bytes(provider_child),
        provider_plan_sha256=_sha256_bytes(provider_plan),
        collector_entrypoint_sha256=arguments.expected_collector_entrypoint_sha256,
        collector_interpreter_sha256=arguments.expected_collector_interpreter_sha256,
        collector_runtime_closure_manifest_sha256=(arguments.expected_collector_runtime_closure_manifest_sha256),
        collector_runtime_closure_receipt_sha256=(arguments.expected_collector_runtime_closure_receipt_sha256),
        collector_runtime_closure_sha256=(arguments.expected_collector_runtime_closure_sha256),
        collector_public_key_hex=collector_public_key.hex(),
        collector_key_id=managed_clinical_real_kvm_collector_key_id(collector_public_key),
        launcher_process_executable_sha256=arguments.expected_collector_interpreter_sha256,
        bootstrap_authorization_key_id=(arguments.expected_bootstrap_authorization_key_id),
    )
    return DrillAuthorization(
        drill_id=arguments.drill_id,
        challenge_nonce_hex=arguments.challenge_nonce_hex,
        challenge_issued_at=issued_at,
        release_pins_sha256=release_pins_sha256,
        challenge_sha256=challenge_sha256,
        registry_authority_id=managed_clinical_real_kvm_authority_id(challenge_sha256=challenge_sha256),
        deployment_id=managed_clinical_real_kvm_deployment_id(challenge_sha256=challenge_sha256),
        registered_entry_id=managed_clinical_real_kvm_registered_entry_id(challenge_sha256=challenge_sha256),
        external_pins=external_pins,
        observation_gate_path=observation_gate_path,
        observation_gate_binding_token=(observation_gate_binding_token),
    )


def _execution_cohort(
    task: ExecutionTask,
    *,
    challenge_sha256: str,
) -> ExecutionCohortManifest:
    task_sha256 = _sha256_model(task)
    binding = ExecutionCohortTaskBinding(
        episode_id=task.context.episode_id,
        target_trial_id=task.context.target_trial_id,
        split=Split.DEV,
        public_lineage_id=f'lineage-{challenge_sha256[:20]}',
        task_sha256=task_sha256,
        task_context_sha256=task.context_sha256,
        private_gold_sha256=_sha256_label(f'private-gold:{challenge_sha256}'),
        private_gold_commitment_sha256=task.private_gold_commitment_sha256,
        private_gold_commitment_key_id=task.private_gold_commitment_key_id,
        cutoff_facts_configured=bool(task.context.fact_questions),
    )
    return ExecutionCohortManifest(
        cohort_id=f'managed-real-kvm-{challenge_sha256[:32]}',
        aggregation_policy_sha256=execution_cohort_aggregation_policy_sha256(),
        lineage_split_manifest_sha256=_sha256_label(f'lineage-split:{challenge_sha256}'),
        workspace_build_receipt_sha256=_sha256_label(f'workspace-build:{challenge_sha256}'),
        gold_derivation_receipt_sha256=_sha256_label(f'gold-derivation:{challenge_sha256}'),
        evaluation_split=Split.DEV,
        tasks=(binding,),
        task_count=1,
        lineage_count=1,
        split_counts=(
            ExecutionCohortSplitCount(split=Split.TRAIN, task_count=0, lineage_count=0),
            ExecutionCohortSplitCount(split=Split.DEV, task_count=1, lineage_count=1),
            ExecutionCohortSplitCount(split=Split.TEST, task_count=0, lineage_count=0),
        ),
    )


def _write_provider_fixture(
    paths: DrillPaths,
    *,
    plan: bytes,
    child: bytes,
) -> None:
    _write_create_once(paths.provider_plan, plan)
    _write_create_once(paths.provider_child, child, mode=0o500)


def _build_pre_reservation_composition(
    arguments: argparse.Namespace,
    *,
    inputs: PublicInputs,
    paths: DrillPaths,
    keys: DrillKeys,
    authorization: DrillAuthorization,
    provider_plan: bytes,
    provider_child: bytes,
) -> Composition:
    # Workspace construction is the first mutation in this function.  Fail on
    # an unbindable Linux AF_UNIX pathname before it can create that workspace.
    _require_linux_pathname_socket_path(paths.registry_socket)
    if (
        min(
            inputs.spec.limits.wall_seconds,
            inputs.policy.limits.wall_seconds,
        )
        < MANAGED_MINIMUM_WALL_SECONDS
    ):
        raise ManagedClinicalRealKvmDrillError(
            'worker and execution-policy wall limits are too short for bootstrap, gate, and cleanup'
        )
    workspace = build_clinical_agentic_workspace(
        task=inputs.task,
        workspace_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        output_root=paths.workspace_root,
        receipt_key=keys.workspace,
        expected_receipt_key_id=clinical_workspace_receipt_key_id(keys.workspace),
    )
    cohort = _execution_cohort(
        inputs.task,
        challenge_sha256=authorization.challenge_sha256,
    )
    plan_sha256 = _sha256_bytes(provider_plan)
    child_sha256 = _sha256_bytes(provider_child)
    provider_subprocess = ProviderSubprocessSpec(
        executable_path=str(paths.provider_child),
        executable_sha256=child_sha256,
        argv_suffix=(
            '--plan',
            str(paths.provider_plan),
            '--expected-plan-sha256',
            plan_sha256,
        ),
        maximum_call_seconds=MANAGED_PROVIDER_CALL_SECONDS,
    )
    provider_adapter = ProviderAdapterDescriptor(
        adapter_id=_ADAPTER_ID,
        adapter_version=_ADAPTER_VERSION,
        executable_sha256=child_sha256,
        config_sha256=plan_sha256,
        provider=_PUBLIC_PROVIDER,
    )
    gateway_policy = AuthenticatedGatewayPolicy(
        gateway_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        gateway_version='dev-v0.1',
        gateway_executable_sha256=_sha256_label('managed-provider-gateway'),
        gateway_config_sha256=_sha256_label(f'gateway-config:{authorization.challenge_sha256}'),
        model_registry_sha256=_sha256_label(f'model-registry:{authorization.challenge_sha256}'),
        receipt_key_id=gateway_session_key_id(keys.gateway),
        maximum_provider_call_seconds=MANAGED_PROVIDER_CALL_SECONDS,
    )
    gateway_route = GatewayModelRoute(
        route_id=f'managed-real-kvm-public-{authorization.challenge_sha256[:20]}',
        logical_model_id=_PUBLIC_MODEL,
        provider=_PUBLIC_PROVIDER,
        provider_model_id=_PUBLIC_MODEL,
        resolved_model_id=_PUBLIC_MODEL,
        accepted_provider_model_ids=(_PUBLIC_MODEL,),
        adapter_id=_ADAPTER_ID,
        adapter_version=_ADAPTER_VERSION,
        adapter_executable_sha256=child_sha256,
        adapter_config_sha256=plan_sha256,
        endpoint_origin='https://fixture.invalid',
        endpoint_path='/v1/deterministic-no-network',
        fixed_parameters_sha256=plan_sha256,
        max_context_tokens=131_072,
        max_output_tokens=4096,
        input_preflight='conservative_upper_bound',
        reasoning_accounting='reported',
        provider_data_control='default',
    )
    receipt = inputs.disk_receipt
    submitted_harness = SubmittedHarnessManifest(
        harness_id='vaxreplay-native-clinical-task-guest',
        harness_version='clinical-v2',
        family=HarnessFamily.VAXREPLAY_NATIVE,
        execution_mode=HarnessExecutionMode.FIXED_MODEL_LOOP,
        runtime_support=HarnessRuntimeSupport.RUNTIME_INTEGRATED,
        harness_image_sha256=receipt.harness.sha256,
        harness_image_byte_count=receipt.harness.byte_count,
        normalized_runtime_tree_sha256=receipt.harness.normalized_tree_sha256,
        guest_executable_path=receipt.guest_executable_path,
        guest_executable_sha256=receipt.guest_executable_sha256,
        guest_argv=receipt.fixed_guest_argv,
        baked_config_sha256=receipt.guest_config_sha256,
        dependency_closure_sha256=receipt.dependency_closure_sha256,
        reproducible_build_receipt_sha256=(arguments.expected_disk_build_receipt_sha256),
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=False,
            guest_local_shell_allowed=False,
        ),
        display_name='VaxReplay native clinical task guest',
        submitter='vaxreplay-organizer',
    )
    harness = make_agentic_harness_identity(
        manifest=submitted_harness,
        requested_model_id=_PUBLIC_MODEL,
        adapter_id=_ADAPTER_ID,
    )
    runtime_config = FirecrackerClinicalRuntimeConfig(
        runtime_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        runtime_version='dev-v0.1',
        runtime_executable_sha256=_module_sha256(firecracker_clinical_runtime_module),
        bootstrap_authorization_key_id=(inputs.guest_config.trust_anchor.authorization_key_id),
        bootstrap_receipt_key_id=clinical_guest_bootstrap_receipt_key_id(keys.bootstrap_receipt),
        # ``serve_one`` separately caps the accept/handshake at the worker wall
        # deadline.  Do not impose a shorter five-second sub-deadline here: the
        # pinned real task guest has a measured cold-boot-to-bootstrap interval
        # longer than five seconds on the qualified nested-KVM development host.
        bootstrap_connection_timeout_seconds=min(
            30.0,
            float(inputs.spec.limits.wall_seconds),
        ),
        bootstrap_validity_seconds=30,
        cleanup_grace_seconds=3,
    )
    runtime_config_sha256 = firecracker_clinical_runtime_config_sha256(runtime_config)
    cgroup_root = Path('/sys/fs/cgroup').joinpath(*inputs.spec.cgroup_parent.split('/'))
    jail_root = Path(inputs.spec.chroot_base_dir) / Path(inputs.spec.runtime.firecracker.source_path).name
    startup_config = ManagedClinicalStartupConfig(
        reconciler_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        reconciler_version='dev-v0.2',
        registry_authority_id=authorization.registry_authority_id,
        runtime_config_sha256=runtime_config_sha256,
        worker_spec_sha256=arguments.expected_worker_spec_sha256,
        cleanup_receipt_key_id=managed_clinical_cleanup_key_id(keys.managed),
        cgroup_root=str(cgroup_root),
        jail_root=str(jail_root),
        vsock_root=str(jail_root),
        receipt_root=str(paths.startup_receipt_root),
        cleanup_grace_seconds=3,
    )
    ownership_config = ManagedClinicalOwnershipConfig(
        ledger_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        ledger_version='dev-v0.2',
        registry_authority_id=authorization.registry_authority_id,
        worker_spec_sha256=arguments.expected_worker_spec_sha256,
        firecracker_executable_sha256=inputs.spec.runtime.firecracker.sha256,
        firecracker_executable_name=Path(inputs.spec.runtime.firecracker.source_path).name,
        ownership_key_id=managed_clinical_cleanup_key_id(keys.managed),
        ledger_root=str(paths.ownership_root),
        jail_namespace_root=str(jail_root),
        cgroup_namespace_root=str(cgroup_root),
    )
    interpreter_sha256 = arguments.expected_collector_interpreter_sha256
    registry_config = ManagedClinicalRegistryConfig(
        service_id=f'managed-real-kvm-{authorization.challenge_sha256[:32]}',
        service_version='dev-v0.4',
        registry_authority_id=authorization.registry_authority_id,
        database_path=str(paths.registry_database),
        socket_path=str(paths.registry_socket),
        production_evidence_root=str(paths.evidence_root),
        protocol_audit_root=str(paths.protocol_audit_root),
        canonical_launcher_id='vaxreplay-lane-a-canonical-operator',
        canonical_launcher_executable_sha256=_module_sha256(clinical_launcher_module),
        launcher_process_executable_sha256=interpreter_sha256,
        service_process_executable_sha256=interpreter_sha256,
        startup_config_sha256=managed_clinical_startup_config_sha256(startup_config),
        startup_cleanup_receipt_key_id=startup_config.cleanup_receipt_key_id,
        connection_timeout_seconds=5,
    )
    if receipt.tool_runtime_closure_manifest_sha256 is None:
        raise ManagedClinicalRealKvmDrillError('managed drill requires the pinned production guest-disk tool closure')
    launcher_deployment = CanonicalClinicalLauncherDeployment(
        registry_authority_id=authorization.registry_authority_id,
        canonical_launcher_id=registry_config.canonical_launcher_id,
        canonical_launcher_executable_sha256=(registry_config.canonical_launcher_executable_sha256),
        expected_system_identity_sha256='0' * 64,
        runtime_id=runtime_config.runtime_id,
        runtime_version=runtime_config.runtime_version,
        runtime_executable_sha256=runtime_config.runtime_executable_sha256,
        runtime_config_sha256=runtime_config_sha256,
        failure_receipt_key_id=clinical_launcher_failure_key_id(keys.launcher_failure),
    )
    manifest_values: dict[str, object] = {
        'operator_executable_sha256': _module_sha256(clinical_operator_module),
        'strict_evidence_loader_executable_sha256': _module_sha256(clinical_production_run_v02_module),
        'provider_subprocess_module_source_sha256': _module_sha256(provider_subprocess_module),
        'deployment': launcher_deployment,
        'runtime_config': runtime_config,
        'worker_spec_path': str(arguments.worker_spec),
        'expected_worker_spec_sha256': arguments.expected_worker_spec_sha256,
        'guest_disk_build_receipt_path': str(arguments.disk_build_receipt),
        'expected_guest_disk_build_receipt_sha256': (arguments.expected_disk_build_receipt_sha256),
        'expected_guest_disk_builder_source_sha256': receipt.builder_source_sha256,
        'expected_base_rootfs_source_sha256': receipt.base_rootfs_source.sha256,
        'expected_harness_payload_source_sha256': receipt.harness_payload_source.sha256,
        'expected_mke2fs_sha256': receipt.mke2fs.sha256,
        'expected_e2fsck_sha256': receipt.e2fsck.sha256,
        'expected_debugfs_sha256': receipt.debugfs.sha256,
        'expected_tool_runtime_closure_manifest_sha256': (receipt.tool_runtime_closure_manifest_sha256),
        'qualification_root': str(arguments.qualification_root),
        'expected_qualification_artifact_sha256': (arguments.expected_qualification_artifact_sha256),
        'expected_qualification_key_id': arguments.expected_qualification_key_id,
        'expected_collector_evidence_sha256': (arguments.expected_qualification_collector_evidence_sha256),
        'expected_probe_manifest_sha256': (arguments.expected_qualification_probe_manifest_sha256),
        'expected_driver_runtime_closure_manifest_sha256': (
            arguments.expected_qualification_runtime_closure_manifest_sha256
        ),
        'expected_driver_runtime_closure_receipt_sha256': (
            arguments.expected_qualification_runtime_closure_receipt_sha256
        ),
        'expected_driver_runtime_closure_sha256': (arguments.expected_qualification_runtime_closure_sha256),
        'expected_collector_public_key_hex': (arguments.expected_qualification_collector_public_key_hex),
        'expected_collector_key_id': (arguments.expected_qualification_collector_key_id),
        'expected_qualification_verifier_source_sha256': (arguments.expected_qualification_verifier_source_sha256),
        'registry_path': str(paths.registry_database),
        'registry_execution_mode': 'managed-unix-authority',
        'managed_registry_config_sha256': managed_clinical_registry_config_sha256(registry_config),
        'managed_startup_config_sha256': managed_clinical_startup_config_sha256(startup_config),
        'managed_ownership_config_sha256': managed_clinical_ownership_config_sha256(ownership_config),
        'gateway_ledger_path': str(paths.gateway_database),
        'evidence_root': str(paths.evidence_root),
        'reservation_sha256': '0' * 64,
        'episode_id': inputs.task.context.episode_id,
        'workspace_root': str(workspace.root),
        'expected_authenticated_workspace_receipt_sha256': (workspace.authenticated_receipt_sha256),
        'expected_workspace_receipt_key_id': clinical_workspace_receipt_key_id(keys.workspace),
        'execution_policy': inputs.policy,
        'gateway_policy': gateway_policy,
        'gateway_route': gateway_route,
        'guest_rpc_policy': inputs.rpc_policy,
        'harness': harness,
        'submitted_harness': submitted_harness,
        'guest_boot_dispatch': receipt.guest_boot_dispatch,
        'provider_adapter': provider_adapter,
        'provider_subprocess': provider_subprocess,
        'bootstrap_trust_anchor': inputs.guest_config.trust_anchor,
    }
    provisional = CanonicalClinicalOperatorManifest.model_validate(manifest_values)
    system = expected_system_identity(provisional, keys.runtime)
    system_sha256 = clinical_production_system_identity_sha256(system)
    launcher_deployment = launcher_deployment.model_copy(update={'expected_system_identity_sha256': system_sha256})
    provisional = CanonicalClinicalOperatorManifest.model_validate(
        {**manifest_values, 'deployment': launcher_deployment}
    )
    if expected_system_identity(provisional, keys.runtime) != system:
        raise ManagedClinicalRealKvmDrillError(
            'managed composition system identity changed while closing its deployment binding'
        )
    return Composition(
        workspace=workspace,
        cohort=cohort,
        runtime_config=runtime_config,
        gateway_policy=gateway_policy,
        gateway_route=gateway_route,
        provider_adapter=provider_adapter,
        provider_subprocess=provider_subprocess,
        submitted_harness=submitted_harness,
        provisional_manifest=provisional,
        system=system,
        startup_config=startup_config,
        ownership_config=ownership_config,
        registry_config=registry_config,
    )


@contextlib.contextmanager
def _running_registry_service(
    *,
    composition: Composition,
    keys: DrillKeys,
) -> Iterator[RunningRegistry]:
    stop_event = threading.Event()
    ready_event = threading.Event()
    errors: list[BaseException] = []
    service = ManagedClinicalRegistryService(
        config=composition.registry_config,
        workspace_receipt_keys_by_id={
            composition.provisional_manifest.expected_workspace_receipt_key_id: (keys.workspace)
        },
        startup_config=composition.startup_config,
        startup_cleanup_receipt_key=keys.managed,
    )

    def serve() -> None:
        try:
            service.serve_until(stop_event=stop_event, ready_event=ready_event)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(
        target=serve,
        name='vaxreplay-managed-real-kvm-provisioning-authority',
        daemon=False,
    )
    thread.start()
    if not ready_event.wait(10) or errors or not thread.is_alive():
        stop_event.set()
        thread.join(10)
        raise ManagedClinicalRealKvmDrillError('provisioning registry authority did not publish readiness') from (
            errors[0] if errors else None
        )
    running = RunningRegistry(
        service=service,
        client=ManagedClinicalRegistryClient(
            composition.registry_config,
            expected_config_sha256=managed_clinical_registry_config_sha256(composition.registry_config),
        ),
        stop_event=stop_event,
        thread=thread,
        errors=errors,
    )
    try:
        yield running
    finally:
        running.close()


def _provision_reservation(
    *,
    composition: Composition,
    keys: DrillKeys,
    inputs: PublicInputs,
    authorization: DrillAuthorization,
) -> tuple[
    ClinicalProductionReservationContext,
    AuthenticatedManagedClinicalStartupCleanup,
]:
    ownership = DurableManagedClinicalOwnershipLedger(
        config=composition.ownership_config,
        ownership_key=keys.managed,
    )
    host = LinuxManagedClinicalHostAdapter(
        config=composition.startup_config,
        ownership=ownership,
        ownership_key=keys.managed,
    )
    gateway_ledger = SqliteGatewayLedger(Path(composition.provisional_manifest.gateway_ledger_path))
    capabilities = RestartVisibleManagedGatewayCapabilityLedger(
        ownership=ownership,
        ownership_key=keys.managed,
        gateway_ledger=gateway_ledger,
        expected_model_route_sha256=gateway_model_route_sha256(composition.gateway_route),
    )
    if (
        ownership.active()
        or host.scan_process_groups()
        or host.scan_cgroups()
        or host.scan_jail_roots()
        or host.scan_vsock_endpoints()
        or capabilities.inventory()
    ):
        raise ManagedClinicalRealKvmDrillError(
            'fresh challenge namespace contains pre-existing managed execution state'
        )
    with _running_registry_service(composition=composition, keys=keys) as running:
        running.client.begin_reconciliation()
        reconciler = ManagedClinicalStartupReconciler(
            config=composition.startup_config,
            host=host,
            capabilities=capabilities,
            attempts=running.client,
            cleanup_receipt_key=keys.managed,
            reconciliation_complete=running.client.finish_reconciliation,
        )
        reconcile_firecracker_clinical_startup_without_execution(
            config=composition.runtime_config,
            execution_policy_sha256=agentic_policy_sha256(inputs.policy),
            worker_spec=inputs.spec,
            gateway_policy_sha256=authenticated_gateway_policy_sha256(composition.gateway_policy),
            gateway_route_sha256=gateway_model_route_sha256(composition.gateway_route),
            guest_rpc_policy=inputs.rpc_policy,
            bootstrap_receipt_key=keys.bootstrap_receipt,
            bootstrap_trust_anchor=inputs.guest_config.trust_anchor,
            evidence_root=Path(composition.provisional_manifest.evidence_root),
            reconciler=reconciler,
        )
        cleanup = reconciler.last_authenticated_receipt
        if cleanup is None or (cleanup.reconciliation_request.requested_at < authorization.challenge_issued_at):
            raise ManagedClinicalRealKvmDrillError(
                'reservation provisioning lacks its post-challenge authenticated cleanup'
            )
        reservation = running.client.reserve_managed(
            manifest=composition.cohort,
            workspaces=(
                ManagedWorkspaceReference(
                    root=str(composition.workspace.root),
                    expected_authenticated_receipt_sha256=(composition.workspace.authenticated_receipt_sha256),
                    expected_receipt_key_id=(composition.provisional_manifest.expected_workspace_receipt_key_id),
                ),
            ),
            system=composition.system,
            registered_entry_id=authorization.registered_entry_id,
            reserved_at=datetime.now(UTC),
        )
    return reservation, cleanup


def _finalize_composition(
    *,
    composition: Composition,
    reservation: ClinicalProductionReservationContext,
    paths: DrillPaths,
    authorization: DrillAuthorization,
    keys: DrillKeys,
) -> FinalComposition:
    manifest = composition.provisional_manifest.model_copy(
        update={'reservation_sha256': reservation.reservation_sha256}
    )
    manifest = CanonicalClinicalOperatorManifest.model_validate_json(canonical_json_bytes(manifest))
    if (
        reservation.reservation.registry_authority_id != authorization.registry_authority_id
        or reservation.reservation.registered_entry_id != authorization.registered_entry_id
        or reservation.reservation.system != composition.system
        or reservation.reservation.system_identity_sha256 != manifest.deployment.expected_system_identity_sha256
    ):
        raise ManagedClinicalRealKvmDrillError(
            'managed reservation differs from the challenge-bound operator composition'
        )
    manifest_sha256 = _sha256_model(manifest)
    deployment = ManagedClinicalStandaloneDeployment(
        deployment_id=authorization.deployment_id,
        deployment_version='dev-v0.2',
        registry_config_path=str(paths.config_root / 'registry.json'),
        registry_config_sha256=managed_clinical_registry_config_sha256(composition.registry_config),
        startup_config_path=str(paths.config_root / 'startup.json'),
        startup_config_sha256=managed_clinical_startup_config_sha256(composition.startup_config),
        ownership_config_path=str(paths.config_root / 'ownership.json'),
        ownership_config_sha256=managed_clinical_ownership_config_sha256(composition.ownership_config),
        operator_manifest_path=str(paths.config_root / 'operator.json'),
        operator_manifest_sha256=manifest_sha256,
        operator_secret_root=str(paths.operator_secret_root),
        managed_secret_root=str(paths.managed_secret_root),
        service_startup_timeout_seconds=10,
        service_shutdown_timeout_seconds=10,
    )
    validate_managed_clinical_deployment_binding(
        LoadedManagedClinicalDeployment(
            deployment=deployment,
            registry_config=composition.registry_config,
            startup_config=composition.startup_config,
            ownership_config=composition.ownership_config,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            secrets=ManagedClinicalDeploymentSecrets(
                startup_cleanup_key=keys.managed,
                ownership_key=keys.managed,
            ),
        )
    )
    return FinalComposition(
        base=composition,
        reservation_context=reservation,
        manifest=manifest,
        deployment=deployment,
    )


def _load_bootstrap_seed(
    path: Path,
    *,
    guest_config: LaneAClinicalGuestConfig,
) -> bytes:
    seed = _read_private_file(path, maximum_bytes=32)
    if len(seed) != 32:
        raise ManagedClinicalRealKvmDrillError('bootstrap authorization seed must contain exactly 32 bytes')
    try:
        public_key = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    except ValueError:
        raise ManagedClinicalRealKvmDrillError('bootstrap authorization seed is invalid') from None
    anchor = guest_config.trust_anchor
    if (
        public_key.hex() != anchor.ed25519_public_key_hex
        or clinical_guest_bootstrap_authorization_key_id(public_key) != anchor.authorization_key_id
    ):
        raise ManagedClinicalRealKvmDrillError('bootstrap authorization seed differs from the guest trust anchor')
    return seed


def _write_fixed_deployment(
    final: FinalComposition,
    *,
    paths: DrillPaths,
    keys: DrillKeys,
    inputs: PublicInputs,
    bootstrap_seed: bytes,
) -> tuple[int, int]:
    parent = paths.config_root.parent
    if not parent.exists():
        _create_root_directory(parent, mode=0o755)
    else:
        _require_root_directory_path(parent)
    staging_root = parent / f'.lane-a-managed.staging-{paths.drill_id}'
    if (
        paths.config_root.exists()
        or paths.config_root.is_symlink()
        or staging_root.exists()
        or staging_root.is_symlink()
    ):
        raise ManagedClinicalRealKvmDrillError('fixed deployment or its challenge staging path is already occupied')
    staging_operator_secrets = staging_root / paths.operator_secret_root.name
    staging_managed_secrets = staging_root / paths.managed_secret_root.name
    published = False
    staging_identity: tuple[int, int] | None = None
    config_files = {
        'registry.json': canonical_json_bytes(final.base.registry_config),
        'startup.json': canonical_json_bytes(final.base.startup_config),
        'ownership.json': canonical_json_bytes(final.base.ownership_config),
        'operator.json': canonical_json_bytes(final.manifest),
        'deployment.json': canonical_json_bytes(final.deployment),
    }
    operator_secrets = {
        'workspace-receipt.key': keys.workspace,
        'worker-attestation.key': keys.worker,
        'gateway-receipt.key': keys.gateway,
        'guest-rpc-receipt.key': keys.guest_rpc,
        'bootstrap-receipt.key': keys.bootstrap_receipt,
        'production-receipt.key': keys.production,
        'launcher-failure-receipt.key': keys.launcher_failure,
        'qualification.key': inputs.qualification_key.hex().encode('ascii') + b'\n',
        'bootstrap-authorization.seed': bootstrap_seed,
        'provider-credential': b'fixture-credential-never-read',
    }
    try:
        _create_root_directory(staging_root)
        staging_metadata = staging_root.lstat()
        staging_identity = (staging_metadata.st_dev, staging_metadata.st_ino)
        _create_root_directory(staging_operator_secrets)
        _create_root_directory(staging_managed_secrets)
        for name, body in config_files.items():
            _write_create_once(staging_root / name, body)
        for name, body in operator_secrets.items():
            _write_create_once(staging_operator_secrets / name, body)
        for name in ('startup-cleanup.key', 'ownership.key'):
            _write_create_once(staging_managed_secrets / name, keys.managed)
        if (
            _bounded_directory_names(
                staging_root,
                maximum_entries=len(config_files) + 2,
                label='staged fixed-deployment root',
            )
            != {
                *config_files,
                paths.operator_secret_root.name,
                paths.managed_secret_root.name,
            }
            or _bounded_directory_names(
                staging_operator_secrets,
                maximum_entries=len(operator_secrets),
                label='staged operator-secret directory',
            )
            != set(operator_secrets)
            or _bounded_directory_names(
                staging_managed_secrets,
                maximum_entries=2,
                label='staged managed-secret directory',
            )
            != {
                'startup-cleanup.key',
                'ownership.key',
            }
        ):
            raise ManagedClinicalRealKvmDrillError('staged fixed deployment has an unexpected file inventory')
        _fsync_directory(staging_operator_secrets)
        _fsync_directory(staging_managed_secrets)
        _fsync_directory(staging_root)
        _fsync_directory(parent)
        try:
            rename_directory_noreplace(staging_root, paths.config_root)
        except OSError:
            raise ManagedClinicalRealKvmDrillError(
                'fixed deployment could not be atomically published without replacement'
            ) from None
        published = True
        _fsync_directory(parent)
        metadata = paths.config_root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (metadata.st_dev, metadata.st_ino) != staging_identity
        ):
            raise ManagedClinicalRealKvmDrillError('atomically published fixed deployment changed identity')
        return metadata.st_dev, metadata.st_ino
    except BaseException:
        if staging_identity is None:
            if staging_root.exists() or staging_root.is_symlink():
                raise ManagedClinicalRealKvmDrillError(
                    'failed fixed-deployment staging was not transactionally removed'
                ) from None
            raise
        cleanup_path = paths.config_root if published else staging_root
        try:
            metadata = cleanup_path.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                stat.S_ISDIR(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and stat.S_IMODE(metadata.st_mode) == 0o700
                and (metadata.st_dev, metadata.st_ino) == staging_identity
            ):
                shutil.rmtree(cleanup_path)
                _fsync_directory(parent)
            else:
                raise ManagedClinicalRealKvmDrillError(
                    'failed fixed-deployment staging changed identity before cleanup'
                ) from None
        raise


def _remove_fixed_deployment(
    paths: DrillPaths,
    *,
    expected_identity: tuple[int, int],
) -> None:
    try:
        metadata = paths.config_root.lstat()
    except OSError:
        raise ManagedClinicalRealKvmDrillError('fixed deployment disappeared before scoped cleanup') from None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalRealKvmDrillError('fixed deployment changed identity before scoped cleanup')
    shutil.rmtree(paths.config_root)
    _fsync_directory(paths.config_root.parent)
    if paths.config_root.exists() or paths.config_root.is_symlink():
        raise ManagedClinicalRealKvmDrillError('fixed deployment cleanup was incomplete')


def _start_managed_invocation(
    paths: DrillPaths,
    *,
    label: Literal['first', 'retry', 'recovery'],
) -> RunningManagedInvocation:
    module = (
        'vaxreplay.agentic.managed_clinical_recovery_cli'
        if label == 'recovery'
        else 'vaxreplay.agentic.managed_clinical_deployment_cli'
    )
    try:
        process = subprocess.Popen(
            (sys.executable, '-I', '-B', '-m', module),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd='/',
            env={
                'LANG': 'C',
                'LC_ALL': 'C',
                'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
            },
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        raise ManagedClinicalRealKvmDrillError(f'{label} fixed managed entrypoint could not be started') from None
    if process.stdout is None or process.stderr is None:
        _abort_managed_invocation_setup(
            process,
            process_group_id=process.pid,
        )
        raise ManagedClinicalRealKvmDrillError(f'{label} fixed managed entrypoint lacks bounded output pipes')
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group_id = -1
    if process_group_id != process.pid:
        _abort_managed_invocation_setup(
            process,
            process_group_id=process_group_id,
        )
        raise ManagedClinicalRealKvmDrillError(f'{label} fixed managed entrypoint lacks its isolated process group')
    stdout_drain: BoundedPipeDrain | None = None
    stderr_drain: BoundedPipeDrain | None = None
    try:
        stdout_drain = _start_bounded_pipe_drain(
            process.stdout,
            label=f'{label} stdout',
        )
        stderr_drain = _start_bounded_pipe_drain(
            process.stderr,
            label=f'{label} stderr',
        )
    except BaseException as error:
        _abort_managed_invocation_setup(
            process,
            process_group_id=process_group_id,
            stdout_drain=stdout_drain,
            stderr_drain=stderr_drain,
        )
        raise ManagedClinicalRealKvmDrillError(
            f'{label} fixed managed entrypoint output drains could not be started'
        ) from error
    return RunningManagedInvocation(
        process=process,
        process_group_id=process_group_id,
        stdout_drain=stdout_drain,
        stderr_drain=stderr_drain,
    )


def _abort_managed_invocation_setup(
    process: subprocess.Popen[bytes],
    *,
    process_group_id: int,
    stdout_drain: BoundedPipeDrain | None = None,
    stderr_drain: BoundedPipeDrain | None = None,
) -> None:
    cleanup_errors: list[BaseException] = []
    try:
        if process_group_id == process.pid:
            _kill_managed_process_group(
                process,
                process_group_id=process_group_id,
            )
        else:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
    except BaseException as error:
        cleanup_errors.append(error)
    drains = (stdout_drain, stderr_drain)
    pipes = (process.stdout, process.stderr)
    for drain, pipe in zip(drains, pipes, strict=True):
        if drain is None:
            if pipe is not None:
                try:
                    pipe.close()
                except BaseException as error:
                    cleanup_errors.append(error)
            continue
        drain.thread.join(5)
        if drain.thread.is_alive() or not drain.done.is_set():
            cleanup_errors.append(ManagedClinicalRealKvmDrillError(f'{drain.label} remained live after setup abort'))
    try:
        process.wait(timeout=3)
    except BaseException as error:
        cleanup_errors.append(error)
    if cleanup_errors:
        raise ManagedClinicalRealKvmDrillError(
            'managed entrypoint setup failed and could not be completely reaped'
        ) from cleanup_errors[0]


def _start_bounded_pipe_drain(pipe: Any, *, label: str) -> BoundedPipeDrain:
    content = bytearray()
    overflowed = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def drain() -> None:
        try:
            while True:
                block = pipe.read(64 * 1024)
                if not block:
                    break
                remaining = MAX_MANAGED_OUTPUT_BYTES + 1 - len(content)
                if remaining > 0:
                    content.extend(block[:remaining])
                if len(content) > MAX_MANAGED_OUTPUT_BYTES or len(block) > remaining:
                    overflowed.set()
                    break
        except BaseException as error:
            errors.append(error)
        finally:
            try:
                pipe.close()
            except BaseException as error:
                errors.append(error)
            done.set()

    thread = threading.Thread(
        target=drain,
        name=f'vaxreplay-bounded-{label.replace(" ", "-")}',
        daemon=False,
    )
    result = BoundedPipeDrain(
        label=label,
        pipe=pipe,
        content=content,
        overflowed=overflowed,
        done=done,
        errors=errors,
        thread=thread,
    )
    thread.start()
    return result


def _kill_managed_process_group(
    process: subprocess.Popen[bytes],
    *,
    process_group_id: int,
) -> None:
    if process_group_id != process.pid:
        raise ManagedClinicalRealKvmDrillError('managed entrypoint does not own its expected isolated process group')
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            raise ManagedClinicalRealKvmDrillError(
                'managed entrypoint disappeared from its process group but was not reaped'
            ) from None
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        # Reap the leader as soon as it exits.  In particular, Darwin may report
        # EPERM for a group containing only an unreaped leader, while Linux would
        # otherwise keep us waiting on that zombie until the grace deadline.
        process.poll()
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            # Darwin can transiently return EPERM while a delivered signal is
            # taking effect.  Never interpret that as absence; retry until the
            # group is reaped or the bounded grace period expires.
            time.sleep(0.01)
            continue
        time.sleep(0.01)
    else:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            raise ManagedClinicalRealKvmDrillError(
                'managed entrypoint process group could not be killed after its grace period'
            ) from None
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        raise ManagedClinicalRealKvmDrillError('managed entrypoint process-group leader did not terminate') from None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            time.sleep(0.01)
            continue
        time.sleep(0.01)
    raise ManagedClinicalRealKvmDrillError('managed entrypoint process group remained after SIGKILL')


def _managed_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _join_bounded_drain(drain: BoundedPipeDrain) -> bytes:
    drain.thread.join(5)
    if drain.thread.is_alive() or not drain.done.is_set() or drain.errors:
        raise ManagedClinicalRealKvmDrillError(f'{drain.label} bounded output drain did not close safely') from (
            drain.errors[0] if drain.errors else None
        )
    if drain.overflowed.is_set() or len(drain.content) > MAX_MANAGED_OUTPUT_BYTES:
        raise ManagedClinicalRealKvmDrillError(f'{drain.label} exceeded its in-memory output bound')
    return bytes(drain.content)


def _finish_managed_invocation(
    running: RunningManagedInvocation,
    *,
    timeout_seconds: float,
) -> CompletedManagedInvocation:
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    descendant_leak = False
    while True:
        if running.stdout_drain.overflowed.is_set() or running.stderr_drain.overflowed.is_set():
            _kill_managed_process_group(
                running.process,
                process_group_id=running.process_group_id,
            )
            break
        return_code = running.process.poll()
        if return_code is not None:
            if _managed_process_group_exists(running.process_group_id):
                descendant_leak = True
                _kill_managed_process_group(
                    running.process,
                    process_group_id=running.process_group_id,
                )
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _kill_managed_process_group(
                running.process,
                process_group_id=running.process_group_id,
            )
            break
        time.sleep(0.01)
    stdout = _join_bounded_drain(running.stdout_drain)
    stderr = _join_bounded_drain(running.stderr_drain)
    if timed_out:
        raise ManagedClinicalRealKvmDrillError('fixed managed entrypoint exceeded its bounded deadline')
    if descendant_leak:
        raise ManagedClinicalRealKvmDrillError(
            'fixed managed entrypoint left a descendant in its isolated process group'
        )
    if running.stdout_drain.overflowed.is_set() or running.stderr_drain.overflowed.is_set():
        raise ManagedClinicalRealKvmDrillError('fixed managed entrypoint exceeded its bounded output budget')
    return CompletedManagedInvocation(
        return_code=running.process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _read_proc_file(path: Path, *, maximum_bytes: int = 64 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManagedClinicalRealKvmDrillError('live procfs evidence is not a regular file')
        body = bytearray()
        while len(body) <= maximum_bytes:
            block = os.read(descriptor, min(4096, maximum_bytes + 1 - len(body)))
            if not block:
                break
            body.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(body) > maximum_bytes or (before.st_dev, before.st_ino, before.st_mode, before.st_uid) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
    ):
        raise ManagedClinicalRealKvmDrillError('live procfs evidence changed while reading')
    return bytes(body)


def _hash_live_process_executable(
    pid: int,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> str:
    descriptor = os.open(Path('/proc') / str(pid) / 'exe', os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ManagedClinicalRealKvmDrillError('live Firecracker executable has unsafe type or size')
        digest = hashlib.sha256()
        consumed = 0
        while block := os.read(descriptor, 1024 * 1024):
            consumed += len(block)
            if consumed > maximum_bytes:
                raise ManagedClinicalRealKvmDrillError('live Firecracker executable exceeds its artifact bound')
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if (before.st_dev, before.st_ino, before.st_mode, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
    ) or not hmac.compare_digest(observed, expected_sha256):
        raise ManagedClinicalRealKvmDrillError('live Firecracker executable changed or differs from its pin')
    return observed


def _read_live_pid_file(
    path: Path,
    *,
    pid: int,
    expected_device: int,
    expected_inode: int,
    expected_owner_uid: int,
    expected_mode: int,
) -> os.stat_result:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        content = os.read(descriptor, 129)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    expected = (
        expected_device,
        expected_inode,
        expected_owner_uid,
        expected_mode,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (
            before.st_dev,
            before.st_ino,
            before.st_uid,
            stat.S_IMODE(before.st_mode),
        )
        != expected
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        or (named.st_dev, named.st_ino, named.st_uid, stat.S_IMODE(named.st_mode)) != expected
        or content != str(pid).encode('ascii')
    ):
        raise ManagedClinicalRealKvmDrillError(
            'live Firecracker PID file differs from its authenticated ownership record'
        )
    return before


def _observe_live_firecracker(
    *,
    ledger: DurableManagedClinicalOwnershipLedger,
    running_envelope: Any,
    spec: FirecrackerWorkerSpec,
) -> ManagedClinicalRealKvmProcessObservation:
    running = running_envelope.record
    required = (
        running.firecracker_pid,
        running.firecracker_start_time_ticks,
        running.process_group_id,
        running.process_group_session_id,
        running.firecracker_executable_sha256,
        running.cgroup_device_id,
        running.cgroup_inode,
        running.firecracker_pid_file_path,
        running.firecracker_pid_file_device_id,
        running.firecracker_pid_file_inode,
        running.firecracker_pid_file_owner_uid,
        running.firecracker_pid_file_mode,
    )
    if any(value is None for value in required):
        raise ManagedClinicalRealKvmDrillError('running ownership record lacks an exact Firecracker identity')
    pid = cast(int, running.firecracker_pid)
    identity_before = read_linux_process_identity(pid)
    if identity_before.process_state == 'Z' or (
        identity_before.pid,
        identity_before.process_group_id,
        identity_before.session_id,
        identity_before.start_time_ticks,
    ) != (
        pid,
        running.process_group_id,
        running.process_group_session_id,
        running.firecracker_start_time_ticks,
    ):
        raise ManagedClinicalRealKvmDrillError('live Firecracker process identity differs from authenticated ownership')
    executable_sha256 = _hash_live_process_executable(
        pid,
        expected_sha256=cast(str, running.firecracker_executable_sha256),
        maximum_bytes=max(spec.runtime.firecracker.byte_count, 1),
    )
    cgroup_path = Path(running.cgroup_path)
    cgroup_descriptor = os.open(
        cgroup_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        cgroup_metadata = os.fstat(cgroup_descriptor)
        if not stat.S_ISDIR(cgroup_metadata.st_mode) or (cgroup_metadata.st_dev, cgroup_metadata.st_ino) != (
            running.cgroup_device_id,
            running.cgroup_inode,
        ):
            raise ManagedClinicalRealKvmDrillError('live Firecracker cgroup differs from authenticated ownership')
        member_descriptor = os.open(
            'cgroup.procs',
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=cgroup_descriptor,
        )
        try:
            members = os.read(member_descriptor, 64 * 1024 + 1)
        finally:
            os.close(member_descriptor)
        try:
            member_pids = {int(value) for value in members.split()}
        except ValueError:
            raise ManagedClinicalRealKvmDrillError('live Firecracker cgroup membership is invalid') from None
        if len(members) > 64 * 1024 or pid not in member_pids:
            raise ManagedClinicalRealKvmDrillError('live Firecracker PID is absent from its exact cgroup')
    finally:
        os.close(cgroup_descriptor)
    relative_cgroup = '/' + cgroup_path.relative_to('/sys/fs/cgroup').as_posix()
    proc_cgroup = _read_proc_file(Path('/proc') / str(pid) / 'cgroup')
    try:
        proc_cgroup_lines = proc_cgroup.decode('ascii').splitlines()
    except UnicodeDecodeError:
        raise ManagedClinicalRealKvmDrillError('live Firecracker procfs cgroup evidence is not ASCII') from None
    if proc_cgroup_lines != [f'0::{relative_cgroup}']:
        raise ManagedClinicalRealKvmDrillError('live Firecracker procfs membership differs from its exact cgroup')
    pid_file_metadata = _read_live_pid_file(
        Path(cast(str, running.firecracker_pid_file_path)),
        pid=pid,
        expected_device=cast(int, running.firecracker_pid_file_device_id),
        expected_inode=cast(int, running.firecracker_pid_file_inode),
        expected_owner_uid=cast(int, running.firecracker_pid_file_owner_uid),
        expected_mode=cast(int, running.firecracker_pid_file_mode),
    )
    kvm = Path('/dev/kvm').lstat()
    if stat.S_ISLNK(kvm.st_mode) or not stat.S_ISCHR(kvm.st_mode):
        raise ManagedClinicalRealKvmDrillError('/dev/kvm changed type during live observation')
    matches: list[tuple[int, int]] = []
    fd_root = Path('/proc') / str(pid) / 'fd'
    count = 0
    with os.scandir(fd_root) as entries:
        for entry in entries:
            count += 1
            if count > spec.limits.open_files:
                raise ManagedClinicalRealKvmDrillError('live Firecracker descriptor inventory exceeds its fixed limit')
            if not entry.name.isascii() or not entry.name.isdigit():
                raise ManagedClinicalRealKvmDrillError('live Firecracker descriptor inventory is noncanonical')
            descriptor_number = int(entry.name)
            try:
                metadata = os.stat(entry.path, follow_symlinks=True)
            except FileNotFoundError:
                continue
            if stat.S_ISCHR(metadata.st_mode) and metadata.st_rdev == kvm.st_rdev:
                matches.append((descriptor_number, metadata.st_rdev))
    if len(matches) != 1:
        raise ManagedClinicalRealKvmDrillError(
            'live Firecracker does not expose exactly one descriptor for the pinned KVM device'
        )
    observed_at = datetime.now(UTC)
    identity_after = read_linux_process_identity(pid)
    latest = ledger.latest(running.run_id)
    kvm_after = Path('/dev/kvm').lstat()
    cgroup_after = cgroup_path.lstat()
    if (
        (
            identity_after.pid,
            identity_after.process_group_id,
            identity_after.session_id,
            identity_after.start_time_ticks,
        )
        != (
            identity_before.pid,
            identity_before.process_group_id,
            identity_before.session_id,
            identity_before.start_time_ticks,
        )
        or identity_after.process_state == 'Z'
        or latest != running_envelope
        or (kvm_after.st_dev, kvm_after.st_ino, kvm_after.st_rdev) != (kvm.st_dev, kvm.st_ino, kvm.st_rdev)
        or (cgroup_after.st_dev, cgroup_after.st_ino) != (cgroup_metadata.st_dev, cgroup_metadata.st_ino)
    ):
        raise ManagedClinicalRealKvmDrillError('live Firecracker identity changed during the bracketed observation')
    return ManagedClinicalRealKvmProcessObservation(
        run_id=running.run_id,
        ownership_envelope_sha256=authenticated_managed_clinical_ownership_sha256(running_envelope),
        firecracker_pid=pid,
        firecracker_start_time_ticks=identity_before.start_time_ticks,
        firecracker_process_group_id=identity_before.process_group_id,
        firecracker_session_id=identity_before.session_id,
        firecracker_executable_sha256=executable_sha256,
        kvm_device_id=kvm.st_dev,
        kvm_device_inode=kvm.st_ino,
        kvm_device_rdev=kvm.st_rdev,
        firecracker_kvm_fd=matches[0][0],
        firecracker_kvm_fd_rdev=matches[0][1],
        proc_cgroup_path=relative_cgroup,
        cgroup_path=str(cgroup_path),
        cgroup_device_id=cgroup_metadata.st_dev,
        cgroup_inode=cgroup_metadata.st_ino,
        firecracker_pid_file_path=cast(str, running.firecracker_pid_file_path),
        firecracker_pid_file_device_id=pid_file_metadata.st_dev,
        firecracker_pid_file_inode=pid_file_metadata.st_ino,
        firecracker_pid_file_owner_uid=pid_file_metadata.st_uid,
        firecracker_pid_file_mode=stat.S_IMODE(pid_file_metadata.st_mode),
        observed_at=observed_at,
    )


def _wait_for_live_observation(
    running: RunningManagedInvocation,
    *,
    ledger: DurableManagedClinicalOwnershipLedger,
    spec: FirecrackerWorkerSpec,
    timeout_seconds: float,
) -> ManagedClinicalRealKvmProcessObservation:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run_ids = ledger.run_ids()
        if len(run_ids) > 1:
            raise ManagedClinicalRealKvmDrillError('managed task created more than one ownership namespace')
        if run_ids:
            chain = ledger.chain(run_ids[0])
            latest = chain[-1]
            if latest.record.state == 'running':
                if len(chain) != 4:
                    raise ManagedClinicalRealKvmDrillError(
                        'managed task reached running through a noncanonical ownership chain'
                    )
                return _observe_live_firecracker(
                    ledger=ledger,
                    running_envelope=latest,
                    spec=spec,
                )
            if latest.record.state in {'capability_revoked', 'cleaned'}:
                raise ManagedClinicalRealKvmDrillError(
                    'managed task completed before its live process could be observed'
                )
        return_code = running.process.poll()
        if return_code is not None:
            raise ManagedClinicalRealKvmDrillError('managed task exited before its live process could be observed')
        time.sleep(0.005)
    raise ManagedClinicalRealKvmDrillError(
        'managed task did not publish a running ownership record before the observation deadline'
    )


def _persist_observation_and_release_gate(
    paths: DrillPaths,
    *,
    authorization: DrillAuthorization,
    observation: ManagedClinicalRealKvmProcessObservation,
) -> ManagedClinicalRealKvmObservationGateRelease:
    observation_path = paths.root / 'live-process-observation.json'
    observation_bytes = canonical_json_bytes(observation)
    _write_create_once(observation_path, observation_bytes)
    loaded_observation = ManagedClinicalRealKvmProcessObservation.model_validate_json(
        _read_pinned_file(
            observation_path,
            expected_sha256=_sha256_bytes(observation_bytes),
            maximum_bytes=1024 * 1024,
            require_root_owner=True,
        )
    )
    if loaded_observation != observation:
        raise ManagedClinicalRealKvmDrillError('persisted live observation differs before gate release')
    release = ManagedClinicalRealKvmObservationGateRelease(
        drill_id=authorization.drill_id,
        challenge_nonce_hex=authorization.challenge_nonce_hex,
        challenge_sha256=authorization.challenge_sha256,
        run_id=observation.run_id,
        ownership_envelope_sha256=observation.ownership_envelope_sha256,
        live_process_observation_sha256=_sha256_model(observation),
        gate_binding_token_hex=(authorization.observation_gate_binding_token.hex()),
        observed_at=observation.observed_at,
        released_at=datetime.now(UTC),
        persisted_path=str(authorization.observation_gate_path),
    )
    _write_create_once(
        authorization.observation_gate_path,
        canonical_json_bytes(release),
    )
    loaded = ManagedClinicalRealKvmObservationGateRelease.model_validate_json(
        _read_pinned_file(
            authorization.observation_gate_path,
            expected_sha256=_sha256_model(release),
            maximum_bytes=64 * 1024,
            require_root_owner=True,
        )
    )
    if loaded != release:
        raise ManagedClinicalRealKvmDrillError('persisted observation-gate release differs from its exact bytes')
    return loaded


def _load_startup_receipt_inventory(
    paths: DrillPaths,
) -> tuple[AuthenticatedManagedClinicalStartupCleanup, ...]:
    root = _private_directory(paths.startup_receipt_root, create=False)
    try:
        observed_entries = []
        with os.scandir(root) as scanned:
            for entry in scanned:
                if len(observed_entries) >= 3:
                    raise ManagedClinicalRealKvmDrillError('startup cleanup receipt inventory exceeds three entries')
                observed_entries.append(entry)
        entries = tuple(sorted(observed_entries, key=lambda item: item.name))
    except OSError:
        raise ManagedClinicalRealKvmDrillError('startup cleanup receipt inventory is unavailable') from None
    if len(entries) > 3 or any(
        not entry.name.endswith('.json')
        or _SHA256_RE.fullmatch(entry.name[:-5]) is None
        or entry.is_symlink()
        or not entry.is_file(follow_symlinks=False)
        for entry in entries
    ):
        raise ManagedClinicalRealKvmDrillError('startup cleanup receipt inventory is noncanonical or oversized')
    return tuple(
        load_authenticated_managed_cleanup(
            Path(entry.path),
            expected_root=root,
        )
        for entry in entries
    )


def _snapshot_evidence_tree(root: Path) -> tuple[tuple[object, ...], ...]:
    resolved = _private_directory(root, create=False)
    values: list[tuple[object, ...]] = []
    total_bytes = 0
    pending: list[tuple[Path, int]] = [(resolved, 0)]
    discovered_entries = 1
    while pending:
        path, depth = pending.pop()
        if depth > 64 or discovered_entries > MAX_EVIDENCE_SNAPSHOT_ENTRIES:
            raise ManagedClinicalRealKvmDrillError('production evidence tree exceeds its depth or entry bound')
        metadata = path.lstat()
        relative = path.relative_to(resolved).as_posix() or '.'
        if stat.S_ISDIR(metadata.st_mode):
            values.append(
                (
                    relative,
                    'directory',
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_uid,
                    metadata.st_gid,
                    stat.S_IMODE(metadata.st_mode),
                )
            )
            children: list[Path] = []
            with os.scandir(path) as entries:
                for entry in entries:
                    discovered_entries += 1
                    if discovered_entries > MAX_EVIDENCE_SNAPSHOT_ENTRIES:
                        raise ManagedClinicalRealKvmDrillError('production evidence tree exceeds its entry bound')
                    children.append(Path(entry.path))
            pending.extend((child, depth + 1) for child in children)
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ManagedClinicalRealKvmDrillError(
                'production evidence tree contains a symlink, hardlink, or special file'
            )
        total_bytes += metadata.st_size
        if total_bytes > MAX_EVIDENCE_SNAPSHOT_BYTES:
            raise ManagedClinicalRealKvmDrillError('production evidence tree exceeds its aggregate byte bound')
        values.append(
            (
                relative,
                'regular_file',
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_uid,
                metadata.st_gid,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_size,
                _sha256_file(path),
            )
        )
    return tuple(sorted(values, key=lambda item: cast(str, item[0])))


def _load_terminal_task_record(
    final: FinalComposition,
) -> ClinicalProductionTaskRecord:
    config = final.base.registry_config
    registry = SqliteClinicalProductionRegistry(
        Path(config.database_path),
        authority_id=config.registry_authority_id,
    )
    reservation_sha256 = final.reservation_context.reservation_sha256
    context = registry.reservation_context(reservation_sha256)
    records = registry.task_records(reservation_sha256)
    if (
        registry.reservation_hashes() != (reservation_sha256,)
        or context != final.reservation_context
        or len(records) != 1
        or records[0].episode_id != final.manifest.episode_id
        or records[0].state != 'succeeded'
    ):
        raise ManagedClinicalRealKvmDrillError(
            'authoritative registry does not contain exactly one successful reserved task'
        )
    return records[0]


def _select_registry_audits(
    final: FinalComposition,
    *,
    managed_key: bytes,
    task_record: ClinicalProductionTaskRecord,
) -> tuple[AuthenticatedManagedClinicalRegistryAudit, AuthenticatedManagedClinicalRegistryAudit]:
    config = final.base.registry_config
    if task_record.launch is None:
        raise ManagedClinicalRealKvmDrillError('successful task has no launch to audit')
    chain = load_authenticated_managed_registry_audit_chain(
        Path(config.protocol_audit_root),
        key=managed_key,
        expected_key_id=config.startup_cleanup_receipt_key_id,
        expected_config_sha256=managed_clinical_registry_config_sha256(config),
    )
    record_candidates: list[AuthenticatedManagedClinicalRegistryAudit] = []
    retry_candidates: list[AuthenticatedManagedClinicalRegistryAudit] = []
    for item in chain:
        if item.request.operation == 'record_run' and item.response.ok:
            request = ManagedRecordRunRequest.model_validate_json(canonical_json_bytes(item.request.payload))
            if (
                request.reservation_sha256 == final.reservation_context.reservation_sha256
                and request.episode_id == task_record.episode_id
            ):
                record_candidates.append(item)
        if item.request.operation == 'claim' and not item.response.ok and item.response.error_code == 'rejected':
            request = ManagedClaimRequest.model_validate_json(canonical_json_bytes(item.request.payload))
            if (
                request.reservation_sha256 == final.reservation_context.reservation_sha256
                and request.episode_id == task_record.episode_id
                and request.run_id != task_record.launch.run_id
            ):
                retry_candidates.append(item)
    if (
        len(record_candidates) != 1
        or len(retry_candidates) != 1
        or record_candidates[0].sequence >= retry_candidates[0].sequence
    ):
        raise ManagedClinicalRealKvmDrillError(
            'authenticated registry audit lacks one success followed by one retry denial'
        )
    return record_candidates[0], retry_candidates[0]


def _load_production_run(
    final: FinalComposition,
    *,
    inputs: PublicInputs,
    keys: DrillKeys,
    task_record: ClinicalProductionTaskRecord,
) -> LoadedClinicalProductionRunV02:
    redemption_sha256 = task_record.start_redemption_sha256
    launch = task_record.launch
    if redemption_sha256 is None or launch is None:
        raise ManagedClinicalRealKvmDrillError('successful registry task lacks launch/redemption evidence')
    manifest = final.manifest
    workspace = load_clinical_agentic_workspace(
        Path(manifest.workspace_root),
        expected_authenticated_receipt_sha256=(manifest.expected_authenticated_workspace_receipt_sha256),
        receipt_key=keys.workspace,
        expected_receipt_key_id=manifest.expected_workspace_receipt_key_id,
    )
    system = final.reservation_context.reservation.system
    root = Path(manifest.evidence_root) / launch.run_id
    loaded = load_clinical_production_run_v02(
        root,
        workspace=workspace,
        expected_authenticated_workspace_receipt_sha256=(manifest.expected_authenticated_workspace_receipt_sha256),
        workspace_receipt_key=keys.workspace,
        expected_workspace_receipt_key_id=manifest.expected_workspace_receipt_key_id,
        expected_run_id=launch.run_id,
        expected_attempt_reservation_sha256=redemption_sha256,
        policy=manifest.execution_policy,
        harness=manifest.harness,
        worker_spec=inputs.spec,
        worker_attestation_key=keys.worker,
        expected_worker_attestation_key_id=system.worker_attestation_key_id,
        gateway_receipt_key=keys.gateway,
        expected_gateway_receipt_key_id=system.gateway_receipt_key_id,
        expected_gateway_policy_sha256=authenticated_gateway_policy_sha256(manifest.gateway_policy),
        expected_gateway_route_sha256=gateway_model_route_sha256(manifest.gateway_route),
        guest_rpc_receipt_key=keys.guest_rpc,
        expected_guest_rpc_receipt_key_id=system.guest_rpc_receipt_key_id,
        expected_guest_rpc_policy_sha256=guest_rpc_policy_sha256(manifest.guest_rpc_policy),
        clinical_guest_bootstrap_receipt_key=keys.bootstrap_receipt,
        expected_clinical_guest_bootstrap_receipt_key_id=(system.guest_bootstrap_receipt_key_id),
        clinical_guest_bootstrap_trust_anchor=manifest.bootstrap_trust_anchor,
        receipt_key=keys.production,
        expected_receipt_key_id=system.production_receipt_key_id,
    )
    if loaded.authenticated_outer_receipt_sha256 != task_record.evidence_sha256:
        raise ManagedClinicalRealKvmDrillError(
            'reloaded production evidence differs from the authoritative registry digest'
        )
    return loaded


def _parse_successful_managed_output(
    completed: CompletedManagedInvocation,
    *,
    final: FinalComposition,
) -> dict[str, object]:
    terminal_failure = _parse_clean_managed_failure_output(
        completed,
        final=final,
    )
    if terminal_failure is not None:
        failure_code, terminal_code = terminal_failure
        raise ManagedClinicalRealKvmDrillError(
            f'first fixed managed entrypoint reported a terminal failure: {failure_code} / {terminal_code}'
        )
    if completed.return_code != 0 or completed.stderr or not completed.stdout.endswith(b'\n'):
        raise ManagedClinicalRealKvmDrillError('first fixed managed entrypoint did not report one clean success')
    try:
        value = json.loads(completed.stdout[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ManagedClinicalRealKvmDrillError('first fixed managed entrypoint output is invalid JSON') from None
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) + b'\n' != completed.stdout
        or set(value)
        != {
            'attempt_consumed',
            'episode_id',
            'evidence_sha256',
            'leaderboard_admitted',
            'live_deployment_qualification_claimed',
            'managed_one_host_authority',
            'reservation_sha256',
            'retry_permitted',
            'run_id',
            'status',
        }
        or value.get('status') != 'succeeded'
        or value.get('reservation_sha256') != final.reservation_context.reservation_sha256
        or value.get('episode_id') != final.manifest.episode_id
        or value.get('attempt_consumed') is not True
        or value.get('retry_permitted') is not False
        or value.get('managed_one_host_authority') is not True
        or value.get('live_deployment_qualification_claimed') is not False
        or value.get('leaderboard_admitted') is not False
    ):
        raise ManagedClinicalRealKvmDrillError('first fixed managed entrypoint output differs from its exact contract')
    run_id = value.get('run_id')
    evidence_sha256 = value.get('evidence_sha256')
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in '0123456789abcdef' for character in run_id)
        or not isinstance(evidence_sha256, str)
        or _SHA256_RE.fullmatch(evidence_sha256) is None
    ):
        raise ManagedClinicalRealKvmDrillError('first fixed managed entrypoint output has invalid result identifiers')
    return cast(dict[str, object], value)


def _persist_private_managed_invocation_output(
    paths: DrillPaths,
    *,
    label: Literal['first', 'retry'],
    completed: CompletedManagedInvocation,
) -> tuple[Path, Path]:
    """Retain bounded child output before interpreting its success/denial contract.

    These raw files stay inside the challenge's mode-0700 state root.  They are deliberately not
    reflected in public errors, but preserve the exact child diagnostics when parsing itself fails.
    """

    try:
        root_metadata = paths.root.lstat()
    except OSError:
        raise ManagedClinicalRealKvmDrillError(
            'managed invocation output root is unavailable before private persistence'
        ) from None
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalRealKvmDrillError('managed invocation output root is not one private owned directory')
    persisted: list[Path] = []
    for stream, payload in (('stdout', completed.stdout), ('stderr', completed.stderr)):
        path = paths.root / f'managed-entrypoint-{label}.{stream}'
        _write_create_once(path, payload)
        reloaded = _read_pinned_file(
            path,
            expected_sha256=_sha256_bytes(payload),
            maximum_bytes=MAX_MANAGED_OUTPUT_BYTES,
            require_root_owner=os.geteuid() == 0,
            allow_empty=True,
        )
        if not hmac.compare_digest(reloaded, payload):
            raise ManagedClinicalRealKvmDrillError(
                f'persisted {label} managed entrypoint {stream} changed after publication'
            )
        persisted.append(path)
    return cast(tuple[Path, Path], tuple(persisted))


def _parse_clean_managed_failure_output(
    completed: CompletedManagedInvocation,
    *,
    final: FinalComposition,
) -> tuple[str, str] | None:
    """Return only allowlisted terminal codes from one exact failure response.

    The fixed entrypoint deliberately reports stable launcher/registry failure
    codes without nested exception details.  Recognizing that exact contract
    keeps the collector's error useful while refusing to echo malformed or
    attacker-selected child output.
    """

    if completed.return_code != 1 or completed.stderr or not completed.stdout.endswith(b'\n'):
        return None
    try:
        value = json.loads(completed.stdout[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or canonical_json_bytes(value) + b'\n' != completed.stdout:
        return None
    failure_code = value.get('failure_code')
    terminal_code = value.get('terminal_code')
    run_id = value.get('run_id')
    if (
        set(value)
        != {
            'attempt_consumed',
            'episode_id',
            'failure_code',
            'leaderboard_admitted',
            'live_deployment_qualification_claimed',
            'managed_one_host_authority',
            'reservation_sha256',
            'retry_permitted',
            'run_id',
            'status',
            'terminal_code',
        }
        or value.get('status') != 'failed'
        or value.get('reservation_sha256') != final.reservation_context.reservation_sha256
        or value.get('episode_id') != final.manifest.episode_id
        or value.get('attempt_consumed') is not True
        or value.get('retry_permitted') is not False
        or value.get('managed_one_host_authority') is not True
        or value.get('live_deployment_qualification_claimed') is not False
        or value.get('leaderboard_admitted') is not False
        or not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in '0123456789abcdef' for character in run_id)
        or not isinstance(failure_code, str)
        or failure_code not in {item.value for item in ClinicalLauncherFailureCode}
        or not isinstance(terminal_code, str)
        or terminal_code not in {item.value for item in ClinicalProductionTerminalCode}
    ):
        return None
    return cast(tuple[str, str], (failure_code, terminal_code))


def _require_retry_denial_output(completed: CompletedManagedInvocation) -> None:
    if (
        completed.return_code != 70
        or completed.stdout
        or completed.stderr != b'managed clinical deployment rejected: bounded execution failed\n'
    ):
        raise ManagedClinicalRealKvmDrillError('retry process did not fail through the fixed managed denial surface')


def _one_new_cleanup(
    before: tuple[AuthenticatedManagedClinicalStartupCleanup, ...],
    after: tuple[AuthenticatedManagedClinicalStartupCleanup, ...],
) -> AuthenticatedManagedClinicalStartupCleanup:
    before_by_path = {item.persisted_path: item for item in before}
    after_by_path = {item.persisted_path: item for item in after}
    if (
        len(before_by_path) != len(before)
        or len(after_by_path) != len(after)
        or any(after_by_path.get(path) != item for path, item in before_by_path.items())
    ):
        raise ManagedClinicalRealKvmDrillError('startup cleanup receipt inventory rewrote a prior phase')
    new_paths = set(after_by_path) - set(before_by_path)
    if len(new_paths) != 1:
        raise ManagedClinicalRealKvmDrillError(
            'managed process did not publish exactly one new startup cleanup receipt'
        )
    return after_by_path[new_paths.pop()]


def _final_managed_state(
    final: FinalComposition,
    *,
    keys: DrillKeys,
    run_id: str,
) -> tuple[
    tuple[Any, ...],
    SqliteGatewayLedger,
    GatewayCapabilityRevocation,
    tuple[int, int, int, int, int, int],
]:
    ownership = DurableManagedClinicalOwnershipLedger(
        config=final.base.ownership_config,
        ownership_key=keys.managed,
    )
    chain = ownership.chain(run_id)
    if ownership.run_ids() != (run_id,) or tuple(item.record.state for item in chain) != (
        'preparing',
        'prepared',
        'start_bound',
        'running',
        'capability_revoked',
        'cleaned',
    ):
        raise ManagedClinicalRealKvmDrillError('final managed ownership chain is incomplete or contains another run')
    gateway = SqliteGatewayLedger(Path(final.manifest.gateway_ledger_path))
    record = _load_terminal_task_record(final)
    redemption = record.start_redemption
    if redemption is None:
        raise ManagedClinicalRealKvmDrillError('final successful task lacks its gateway capability redemption')
    revocation = gateway.capability_revocation(redemption.gateway_capability_id)
    if revocation is None:
        raise ManagedClinicalRealKvmDrillError('final gateway ledger lacks its durable capability tombstone')
    capabilities = RestartVisibleManagedGatewayCapabilityLedger(
        ownership=ownership,
        ownership_key=keys.managed,
        gateway_ledger=gateway,
        expected_model_route_sha256=gateway_model_route_sha256(final.manifest.gateway_route),
    )
    host = LinuxManagedClinicalHostAdapter(
        config=final.base.startup_config,
        ownership=ownership,
        ownership_key=keys.managed,
    )
    counts = (
        len(ownership.active()),
        len(capabilities.inventory()),
        len(host.scan_process_groups()),
        len(host.scan_cgroups()),
        len(host.scan_jail_roots()),
        len(host.scan_vsock_endpoints()),
    )
    if counts != (0, 0, 0, 0, 0, 0) or gateway.unrevoked_capability_bindings():
        raise ManagedClinicalRealKvmDrillError(
            'managed execution left active ownership, capability, process, or host artifacts'
        )
    return chain, gateway, revocation, counts


def _build_drill_evidence(
    arguments: argparse.Namespace,
    *,
    authorization: DrillAuthorization,
    inputs: PublicInputs,
    final: FinalComposition,
    task_record: ClinicalProductionTaskRecord,
    ownership_chain: tuple[Any, ...],
    observation: ManagedClinicalRealKvmProcessObservation,
    gate_release: ManagedClinicalRealKvmObservationGateRelease,
    record_audit: AuthenticatedManagedClinicalRegistryAudit,
    retry_audit: AuthenticatedManagedClinicalRegistryAudit,
    task_sha256_before_retry: str,
    task_sha256_after_retry: str,
    gateway: SqliteGatewayLedger,
    gateway_revocation: GatewayCapabilityRevocation,
    loaded: LoadedClinicalProductionRunV02,
    provisioning_cleanup: AuthenticatedManagedClinicalStartupCleanup,
    first_cleanup: AuthenticatedManagedClinicalStartupCleanup,
    retry_cleanup: AuthenticatedManagedClinicalStartupCleanup,
    first_stdout_path: Path,
    first_stdout: bytes,
    final_counts: tuple[int, int, int, int, int, int],
) -> ManagedClinicalRealKvmDrillEvidence:
    launch = task_record.launch
    if (
        launch is None
        or launch.run_id != observation.run_id
        or loaded.root != Path(final.manifest.evidence_root) / observation.run_id
    ):
        raise ManagedClinicalRealKvmDrillError('evidence components name different managed attempts')
    if final_counts != (0, 0, 0, 0, 0, 0):
        raise ManagedClinicalRealKvmDrillError('evidence construction received nonterminal managed host state')
    return ManagedClinicalRealKvmDrillEvidence(
        drill_id=authorization.drill_id,
        challenge_nonce_hex=authorization.challenge_nonce_hex,
        run_id=observation.run_id,
        worker_spec=inputs.spec,
        worker_spec_sha256=arguments.expected_worker_spec_sha256,
        disk_build_receipt_sha256=arguments.expected_disk_build_receipt_sha256,
        qualification_artifact_sha256=(arguments.expected_qualification_artifact_sha256),
        task_sha256=arguments.expected_task_sha256,
        deployment_sha256=_sha256_model(final.deployment),
        registry_config_sha256=managed_clinical_registry_config_sha256(final.base.registry_config),
        startup_config_sha256=managed_clinical_startup_config_sha256(final.base.startup_config),
        ownership_config_sha256=managed_clinical_ownership_config_sha256(final.base.ownership_config),
        operator_manifest_sha256=_sha256_model(final.manifest),
        provider_child_executable_sha256=(authorization.external_pins.provider_child_executable_sha256),
        provider_plan_sha256=authorization.external_pins.provider_plan_sha256,
        collector_entrypoint_sha256=arguments.expected_collector_entrypoint_sha256,
        collector_interpreter_sha256=(arguments.expected_collector_interpreter_sha256),
        collector_runtime_closure=inputs.collector_runtime_closure,
        launcher_process_executable_sha256=(arguments.expected_collector_interpreter_sha256),
        reservation_sha256=final.reservation_context.reservation_sha256,
        deployment=final.deployment,
        registry_config=final.base.registry_config,
        startup_config=final.base.startup_config,
        ownership_config=final.base.ownership_config,
        operator_manifest=final.manifest,
        reservation=final.reservation_context.reservation,
        task_record=task_record,
        ownership_chain=ownership_chain,
        live_process_observation=observation,
        observation_gate_release=gate_release,
        registry_observation=ManagedClinicalRegistryDrillObservation(
            record_run_audit=record_audit,
            retry_claim_audit=retry_audit,
            terminal_task_record_before_retry_sha256=(task_sha256_before_retry),
            terminal_task_record_after_retry_sha256=task_sha256_after_retry,
        ),
        gateway_ledger_identity=gateway.identity,
        gateway_revocation=gateway_revocation,
        production_run_root=str(loaded.root),
        production_run=loaded.authenticated_outer_receipt,
        bootstrap=loaded.clinical_guest_bootstrap,
        guest_rpc=loaded.guest_rpc_session,
        gateway_session=loaded.gateway_session,
        worker_attestation=loaded.worker_attestation,
        submission=loaded.submission,
        startup_cleanups=(
            ManagedClinicalStartupCleanupDrillObservation(
                phase='reservation_provisioning',
                authenticated_cleanup=provisioning_cleanup,
            ),
            ManagedClinicalStartupCleanupDrillObservation(
                phase='managed_operator_startup',
                authenticated_cleanup=first_cleanup,
            ),
            ManagedClinicalStartupCleanupDrillObservation(
                phase='retry_denial_reopen',
                authenticated_cleanup=retry_cleanup,
            ),
        ),
        managed_entrypoint_stdout_path=str(first_stdout_path),
        managed_entrypoint_stdout_sha256=_sha256_bytes(first_stdout),
        post_reconciliation_active_ownership_count=0,
        post_reconciliation_unrevoked_capability_count=0,
        post_reconciliation_process_group_count=0,
        post_reconciliation_cgroup_count=0,
        post_reconciliation_jail_count=0,
        post_reconciliation_vsock_count=0,
        collected_at=datetime.now(UTC),
    )


def _emergency_recover_failed_drill(
    paths: DrillPaths,
    *,
    final: FinalComposition,
    keys: DrillKeys,
) -> None:
    running = _start_managed_invocation(paths, label='recovery')
    completed = _finish_managed_invocation(running, timeout_seconds=120)
    if completed.return_code != 0 or completed.stderr or not completed.stdout.endswith(b'\n'):
        raise ManagedClinicalRealKvmDrillError('cleanup-only fixed recovery process did not complete cleanly')
    try:
        receipt = json.loads(completed.stdout[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ManagedClinicalRealKvmDrillError('cleanup-only fixed recovery process returned invalid JSON') from None
    if (
        not isinstance(receipt, dict)
        or canonical_json_bytes(receipt) + b'\n' != completed.stdout
        or receipt.get('orphan_cleanup_complete') is not True
        or receipt.get('consumed_attempts_terminalized') is not True
        or receipt.get('recovery_only_registry_mode') is not True
        or receipt.get('harness_or_model_execution_available') is not False
        or receipt.get('automatic_task_retry') is not False
    ):
        raise ManagedClinicalRealKvmDrillError('cleanup-only fixed recovery process returned the wrong safety receipt')
    ownership = DurableManagedClinicalOwnershipLedger(
        config=final.base.ownership_config,
        ownership_key=keys.managed,
    )
    host = LinuxManagedClinicalHostAdapter(
        config=final.base.startup_config,
        ownership=ownership,
        ownership_key=keys.managed,
    )
    remaining = (
        len(ownership.active()),
        len(host.scan_process_groups()),
        len(host.scan_cgroups()),
        len(host.scan_jail_roots()),
        len(host.scan_vsock_endpoints()),
    )
    if remaining != (0, 0, 0, 0, 0):
        raise ManagedClinicalRealKvmDrillError('cleanup-only recovery left managed Firecracker state behind')
    if paths.gateway_database.exists():
        gateway = SqliteGatewayLedger(paths.gateway_database)
        capabilities = RestartVisibleManagedGatewayCapabilityLedger(
            ownership=ownership,
            ownership_key=keys.managed,
            gateway_ledger=gateway,
            expected_model_route_sha256=gateway_model_route_sha256(final.manifest.gateway_route),
        )
        if capabilities.inventory() or gateway.unrevoked_capability_bindings():
            raise ManagedClinicalRealKvmDrillError('cleanup-only recovery left a managed gateway capability behind')


def _collect(
    arguments: argparse.Namespace,
    *,
    preverified_collector_runtime_closure: LoadedQualificationDriverRuntimeClosure,
) -> dict[str, object]:
    _require_linux_root_kvm()
    inputs = _load_public_inputs(
        arguments,
        preverified_collector_runtime_closure=(preverified_collector_runtime_closure),
    )
    _verify_loaded_module_runtime_binding(inputs.collector_runtime_closure)
    collector_private_key = _load_collector_private_key(arguments)
    gate_path, binding_token = _observation_gate_inputs(arguments)
    provider_plan = _canonical_provider_plan(
        inputs.task,
        observation_gate=(
            gate_path,
            binding_token,
            arguments.drill_id,
            arguments.challenge_nonce_hex,
        ),
    )
    provider_child = render_deterministic_provider_child(
        Path(inputs.collector_runtime_closure.manifest.interpreter_path)
    )
    authorization = _challenge_authorization(
        arguments,
        inputs=inputs,
        provider_plan=provider_plan,
        provider_child=provider_child,
        collector_private_key=collector_private_key,
        observation_gate_path=gate_path,
        observation_gate_binding_token=binding_token,
    )
    paths = DrillPaths.live(
        authorization.drill_id,
        authorization.challenge_sha256,
    )
    _initialize_live_paths(paths, authorization=authorization)
    keys = DrillKeys.generate()
    _persist_verifier_keys(
        paths,
        keys=keys,
        qualification_key=inputs.qualification_key,
    )
    persisted_keys, persisted_qualification_key = _load_persisted_verifier_keys(paths)
    if persisted_keys != keys or not hmac.compare_digest(
        persisted_qualification_key,
        inputs.qualification_key,
    ):
        raise ManagedClinicalRealKvmDrillError('challenge-private verifier keys differ immediately after persistence')
    bootstrap_seed = _load_bootstrap_seed(
        arguments.bootstrap_authorization_seed_file,
        guest_config=inputs.guest_config,
    )
    _write_provider_fixture(
        paths,
        plan=provider_plan,
        child=provider_child,
    )
    composition = _build_pre_reservation_composition(
        arguments,
        inputs=inputs,
        paths=paths,
        keys=keys,
        authorization=authorization,
        provider_plan=provider_plan,
        provider_child=provider_child,
    )
    reservation, provisioning_cleanup = _provision_reservation(
        composition=composition,
        keys=keys,
        inputs=inputs,
        authorization=authorization,
    )
    provisioned_receipts = _load_startup_receipt_inventory(paths)
    if provisioned_receipts != (provisioning_cleanup,):
        raise ManagedClinicalRealKvmDrillError('reservation provisioning did not retain exactly its cleanup receipt')
    final = _finalize_composition(
        composition=composition,
        reservation=reservation,
        paths=paths,
        authorization=authorization,
        keys=keys,
    )
    fixed_identity: tuple[int, int] | None = None
    active_invocation: RunningManagedInvocation | None = None
    try:
        fixed_identity = _write_fixed_deployment(
            final,
            paths=paths,
            keys=keys,
            inputs=inputs,
            bootstrap_seed=bootstrap_seed,
        )
        ownership = DurableManagedClinicalOwnershipLedger(
            config=final.base.ownership_config,
            ownership_key=keys.managed,
        )
        active_invocation = _start_managed_invocation(paths, label='first')
        observation = _wait_for_live_observation(
            active_invocation,
            ledger=ownership,
            spec=inputs.spec,
            timeout_seconds=float(OBSERVATION_GATE_TIMEOUT_SECONDS),
        )
        gate_release = _persist_observation_and_release_gate(
            paths,
            authorization=authorization,
            observation=observation,
        )
        first_completed = _finish_managed_invocation(
            active_invocation,
            timeout_seconds=float(inputs.spec.limits.wall_seconds) + 30.0,
        )
        active_invocation = None
        first_stdout_path, _first_stderr_path = _persist_private_managed_invocation_output(
            paths,
            label='first',
            completed=first_completed,
        )
        first_output = _parse_successful_managed_output(
            first_completed,
            final=final,
        )
        after_first_receipts = _load_startup_receipt_inventory(paths)
        first_cleanup = _one_new_cleanup(
            provisioned_receipts,
            after_first_receipts,
        )
        task_before_retry = _load_terminal_task_record(final)
        if (
            first_output['run_id'] != observation.run_id
            or first_output['evidence_sha256'] != task_before_retry.evidence_sha256
        ):
            raise ManagedClinicalRealKvmDrillError(
                'managed success output differs from the authoritative terminal task'
            )
        task_sha256_before_retry = _sha256_model(task_before_retry)
        first_chain, first_gateway, first_revocation, first_counts = _final_managed_state(
            final,
            keys=keys,
            run_id=observation.run_id,
        )
        if first_counts != (0, 0, 0, 0, 0, 0):
            raise ManagedClinicalRealKvmDrillError('first managed run did not reach a clean terminal state')
        evidence_tree_before_retry = _snapshot_evidence_tree(paths.evidence_root)

        active_invocation = _start_managed_invocation(paths, label='retry')
        retry_completed = _finish_managed_invocation(
            active_invocation,
            timeout_seconds=float(inputs.spec.limits.wall_seconds) + 30.0,
        )
        active_invocation = None
        _retry_stdout_path, _retry_stderr_path = _persist_private_managed_invocation_output(
            paths,
            label='retry',
            completed=retry_completed,
        )
        _require_retry_denial_output(retry_completed)
        after_retry_receipts = _load_startup_receipt_inventory(paths)
        retry_cleanup = _one_new_cleanup(
            after_first_receipts,
            after_retry_receipts,
        )
        task_after_retry = _load_terminal_task_record(final)
        task_sha256_after_retry = _sha256_model(task_after_retry)
        if (
            task_after_retry != task_before_retry
            or task_sha256_after_retry != task_sha256_before_retry
            or _snapshot_evidence_tree(paths.evidence_root) != evidence_tree_before_retry
        ):
            raise ManagedClinicalRealKvmDrillError(
                'retry denial changed terminal registry or production evidence state'
            )
        final_chain, final_gateway, final_revocation, final_counts = _final_managed_state(
            final,
            keys=keys,
            run_id=observation.run_id,
        )
        if (
            final_chain != first_chain
            or final_gateway.identity != first_gateway.identity
            or final_revocation != first_revocation
        ):
            raise ManagedClinicalRealKvmDrillError('retry denial changed ownership or gateway tombstone state')
        loaded = _load_production_run(
            final,
            inputs=inputs,
            keys=keys,
            task_record=task_after_retry,
        )
        record_audit, retry_audit = _select_registry_audits(
            final,
            managed_key=keys.managed,
            task_record=task_after_retry,
        )
        evidence = _build_drill_evidence(
            arguments,
            authorization=authorization,
            inputs=inputs,
            final=final,
            task_record=task_after_retry,
            ownership_chain=final_chain,
            observation=observation,
            gate_release=gate_release,
            record_audit=record_audit,
            retry_audit=retry_audit,
            task_sha256_before_retry=task_sha256_before_retry,
            task_sha256_after_retry=task_sha256_after_retry,
            gateway=final_gateway,
            gateway_revocation=final_revocation,
            loaded=loaded,
            provisioning_cleanup=provisioning_cleanup,
            first_cleanup=first_cleanup,
            retry_cleanup=retry_cleanup,
            first_stdout_path=first_stdout_path,
            first_stdout=first_completed.stdout,
            final_counts=final_counts,
        )
        persisted_keys, persisted_qualification_key = _load_persisted_verifier_keys(paths)
        verifier_keys = persisted_keys.verifier(qualification_key=persisted_qualification_key)
        verified = verify_managed_clinical_real_kvm_drill_from_persisted_state(
            evidence,
            external_pins=authorization.external_pins,
            keys=verifier_keys,
        )
        if verified != evidence:
            raise ManagedClinicalRealKvmDrillError('pre-signing independent verification changed drill evidence')
        authenticated = authenticate_managed_clinical_real_kvm_drill(
            evidence,
            private_key=collector_private_key,
        )
        independently_verify_authenticated_managed_clinical_real_kvm_drill(
            authenticated,
            expected_evidence_sha256=authenticated.evidence_sha256,
            expected_collector_public_key_hex=(arguments.expected_collector_public_key_hex),
            external_pins=authorization.external_pins,
            keys=verifier_keys,
        )
        _remove_fixed_deployment(paths, expected_identity=fixed_identity)
        fixed_identity = None
        authenticated_bytes = canonical_json_bytes(authenticated)
        authenticated_path = paths.root / EVIDENCE_FILE
        digest_path = paths.root / EVIDENCE_SHA256_FILE
        _write_create_once(authenticated_path, authenticated_bytes)
        _write_create_once(
            digest_path,
            (authenticated.evidence_sha256 + '\n').encode('ascii'),
        )
        reloaded = AuthenticatedManagedClinicalRealKvmDrill.model_validate_json(
            _read_pinned_file(
                authenticated_path,
                expected_sha256=_sha256_bytes(authenticated_bytes),
                require_root_owner=True,
            )
        )
        digest_bytes = _read_pinned_file(
            digest_path,
            expected_sha256=_sha256_bytes((authenticated.evidence_sha256 + '\n').encode('ascii')),
            maximum_bytes=65,
            require_root_owner=True,
        )
        if reloaded != authenticated or digest_bytes != (authenticated.evidence_sha256 + '\n').encode('ascii'):
            raise ManagedClinicalRealKvmDrillError('create-once signed drill evidence differs after reload')
        independently_verify_authenticated_managed_clinical_real_kvm_drill(
            reloaded,
            expected_evidence_sha256=authenticated.evidence_sha256,
            expected_collector_public_key_hex=(arguments.expected_collector_public_key_hex),
            external_pins=authorization.external_pins,
            keys=verifier_keys,
        )
        return {
            'status': 'succeeded',
            'drill_id': authorization.drill_id,
            'challenge_sha256': authorization.challenge_sha256,
            'release_pins_sha256': authorization.release_pins_sha256,
            'run_id': observation.run_id,
            'evidence_sha256': authenticated.evidence_sha256,
            'authenticated_evidence_path': str(authenticated_path),
            'evidence_digest_path': str(digest_path),
            'retained_state_root': str(paths.root),
            'retained_private_verifier_keys': True,
            'live_kvm_run_performed': True,
            'retry_denied_after_real_authority_restart': True,
            'external_provider_called': False,
            'development_only': True,
            'official_leaderboard_execution_qualified': False,
        }
    except BaseException:
        cleanup_errors: list[BaseException] = []
        if active_invocation is not None and _managed_process_group_exists(active_invocation.process_group_id):
            try:
                _kill_managed_process_group(
                    active_invocation.process,
                    process_group_id=active_invocation.process_group_id,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if fixed_identity is not None:
            try:
                _emergency_recover_failed_drill(
                    paths,
                    final=final,
                    keys=keys,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                _remove_fixed_deployment(
                    paths,
                    expected_identity=fixed_identity,
                )
                fixed_identity = None
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise ManagedClinicalRealKvmDrillError(
                'managed drill failed and its bounded emergency cleanup also failed'
            ) from cleanup_errors[0]
        raise


def main(
    *,
    preverified_collector_runtime_closure: (LoadedQualificationDriverRuntimeClosure | None) = None,
) -> None:
    arguments = _parser().parse_args()
    if arguments.command == 'dry-run-config':
        result = _dry_run_config(arguments)
    else:
        if preverified_collector_runtime_closure is None:
            raise ManagedClinicalRealKvmDrillError(
                'live preparation and collection require the isolated closure bootstrap'
            )
        if arguments.command == 'prepare-live':
            result = _prepare_live(
                arguments,
                preverified_collector_runtime_closure=(preverified_collector_runtime_closure),
            )
        elif arguments.command == 'collect':
            result = _collect(
                arguments,
                preverified_collector_runtime_closure=(preverified_collector_runtime_closure),
            )
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise ManagedClinicalRealKvmDrillError('unknown collector command')
    sys.stdout.buffer.write(canonical_json_bytes(result) + b'\n')
