"""Root-owned Unix-socket service for one Lane A attempt-registry authority.

The lower-level SQLite registry is transactional, but a caller can otherwise copy it or select a
different path.  This service owns one configured database path and exposes only bounded typed
operations over one private Unix socket.  Linux ``SO_PEERCRED`` authenticates the local service
account; launcher identity comes from service configuration and is never accepted from request
JSON.  This is one-host authority, not distributed consensus.
"""

from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import hmac
import os
import re
import secrets
import socket
import stat
import struct
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, Self, cast

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublicationError, rename_file_noreplace
from vaxreplay.agentic.clinical_execution_bridge import load_clinical_agentic_workspace
from vaxreplay.agentic.clinical_production_registry import (
    ClinicalProductionExplicitFailureCode,
    ClinicalProductionReservationContext,
    ClinicalProductionStartRedemption,
    ClinicalProductionSystemIdentity,
    ClinicalProductionTaskLaunch,
    ClinicalProductionTaskRecord,
    ClinicalProductionTerminalCode,
    ProductionRunReauthenticator,
    SqliteClinicalProductionRegistry,
    clinical_production_start_redemption_sha256,
    clinical_production_task_launch_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    AuthenticatedManagedClinicalStartupCleanup,
    ManagedClinicalAttemptInventoryRecord,
    ManagedClinicalStartupConfig,
    load_authenticated_managed_cleanup,
    managed_clinical_cleanup_key_id,
    managed_clinical_startup_config_sha256,
    verify_authenticated_managed_cleanup,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.execution_aggregation import ExecutionCohortManifest

MANAGED_CLINICAL_REGISTRY_CONFIG_SCHEMA_VERSION = 'vaxreplay.managed-clinical-registry-config.dev-v0.4'
MANAGED_CLINICAL_REGISTRY_REQUEST_SCHEMA_VERSION = 'vaxreplay.managed-clinical-registry-request.dev-v0.1'
MANAGED_CLINICAL_REGISTRY_RESPONSE_SCHEMA_VERSION = 'vaxreplay.managed-clinical-registry-response.dev-v0.1'
AUTHENTICATED_MANAGED_CLINICAL_REGISTRY_AUDIT_SCHEMA_VERSION = (
    'vaxreplay.authenticated-managed-clinical-registry-audit.dev-v0.1'
)

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_RUN_ID_PATTERN = r'^[0-9a-f]{32}$'
_ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
_MAX_FRAME_BYTES = 16 * 1024 * 1024
_MAX_FAILURE_BYTES = 8 * 1024 * 1024
_FRAME_HEADER = struct.Struct('!I')
_COPY_BUFFER_BYTES = 1024 * 1024
_MAX_AUDIT_BYTES = 40 * 1024 * 1024
MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES = 4_096
MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES = 64 * 1024 * 1024
LINUX_AF_UNIX_PATHNAME_MAX_BYTES = 107
_MAX_PROTOCOL_AUDIT_STAGING_ENTRIES = 1
_AUDIT_STAGING_NAME = re.compile(r'^\.audit-stage-[0-9a-f]{64}$')
_AUDIT_FILENAME = re.compile(r'^[0-9]{20}-[0-9a-f]{32}\.json$')
_AUDIT_HMAC_DOMAIN = b'vaxreplay.authenticated-managed-clinical-registry-audit.dev-v0.1\x00'
_AUDIT_ZERO_SHA256 = '0' * 64

type ManagedClinicalRegistryOperation = Literal[
    'begin_reconciliation',
    'finish_reconciliation',
    'reserve',
    'claim',
    'redeem',
    'status',
    'record_failure',
    'record_run',
]

_MUTATING_OPERATIONS = frozenset({'reserve', 'claim', 'redeem', 'record_failure', 'record_run'})


class ManagedClinicalRegistryError(RuntimeError):
    """The managed registry protocol or authority boundary rejected an operation."""


def _require_linux_af_unix_pathname(path: Path) -> None:
    encoded = os.fsencode(path)
    if not path.is_absolute() or not encoded or b'\x00' in encoded:
        raise ManagedClinicalRegistryError('managed registry socket must be one absolute pathname')
    if len(encoded) > LINUX_AF_UNIX_PATHNAME_MAX_BYTES:
        raise ManagedClinicalRegistryError('managed registry AF_UNIX pathname exceeds the Linux 107-byte limit')


class ManagedClinicalRegistryConfig(StrictModel):
    schema_version: Literal['vaxreplay.managed-clinical-registry-config.dev-v0.4'] = (
        MANAGED_CLINICAL_REGISTRY_CONFIG_SCHEMA_VERSION
    )
    service_id: str = Field(pattern=_ID_PATTERN)
    service_version: str = Field(min_length=1, max_length=200)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    database_path: str
    socket_path: str
    production_evidence_root: str
    protocol_audit_root: str
    allowed_launcher_uid: Literal[0] = 0
    allowed_launcher_gid: Literal[0] = 0
    canonical_launcher_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    launcher_process_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    service_process_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    startup_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    startup_cleanup_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    maximum_frame_bytes: Literal[16777216] = _MAX_FRAME_BYTES
    connection_timeout_seconds: float = Field(gt=0, le=30)
    database_path_selected_by_service_only: Literal[True] = True
    production_evidence_namespace_selected_by_service_only: Literal[True] = True
    launcher_identity_derived_from_peer_and_service_config: Literal[True] = True
    launcher_process_image_verified_from_procfs: Literal[True] = True
    service_process_image_verified_by_client: Literal[True] = True
    canonical_request_framing_required: Literal[True] = True
    database_inode_pinned_for_service_lifetime: Literal[True] = True
    exclusive_database_service_lock_required: Literal[True] = True
    stale_owned_socket_reaped_before_bind: Literal[True] = True
    root_owned_service_required: Literal[True] = True
    startup_quiescence_required_after_every_service_start: Literal[True] = True
    full_attempt_inventory_served_under_quiescence: Literal[True] = True
    signed_cleanup_receipt_required_to_release_quiescence: Literal[True] = True
    authenticated_protocol_audit_required_before_response: Literal[True] = True
    one_host_authority: Literal[True] = True
    cross_host_consensus_claimed: Literal[False] = False

    @field_validator(
        'database_path',
        'socket_path',
        'production_evidence_root',
        'protocol_audit_root',
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed registry paths must be normalized and absolute')
        return value

    @field_validator('socket_path')
    @classmethod
    def validate_socket_path(cls, value: str) -> str:
        encoded = os.fsencode(value)
        if b'\x00' in encoded:
            raise ValueError('managed registry socket path cannot contain NUL bytes')
        if len(encoded) > LINUX_AF_UNIX_PATHNAME_MAX_BYTES:
            raise ValueError('managed registry AF_UNIX pathname exceeds the Linux 107-byte limit')
        return value

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        paths = tuple(
            PurePosixPath(item)
            for item in (
                self.database_path,
                self.socket_path,
                self.production_evidence_root,
                self.protocol_audit_root,
            )
        )
        if len(set(paths)) != 4 or any(
            left in right.parents or right in left.parents
            for index, left in enumerate(paths)
            for right in paths[index + 1 :]
        ):
            raise ValueError('managed registry database, socket, evidence, and audit paths must be disjoint')
        return self


class ManagedClinicalPeerIdentity(StrictModel):
    pid: int = Field(gt=0, le=2**31 - 1)
    uid: int = Field(ge=0, le=2**31 - 1)
    gid: int = Field(ge=0, le=2**31 - 1)
    canonical_launcher_id: str = Field(pattern=_ID_PATTERN)
    canonical_launcher_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    derived_from_so_peercred_and_service_config: Literal[True] = True


class ManagedWorkspaceReference(StrictModel):
    root: str
    expected_authenticated_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('root')
    @classmethod
    def validate_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed workspace roots must be normalized absolute paths')
        return value


class ManagedReserveRequest(StrictModel):
    manifest: ExecutionCohortManifest
    workspaces: tuple[ManagedWorkspaceReference, ...] = Field(min_length=1, max_length=10_000)
    system: ClinicalProductionSystemIdentity
    registered_entry_id: str = Field(pattern=_ID_PATTERN)
    reserved_at: datetime

    @field_validator('reserved_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed reservation time')

    @model_validator(mode='after')
    def validate_workspace_order(self) -> Self:
        roots = tuple(item.root for item in self.workspaces)
        if roots != tuple(sorted(set(roots))):
            raise ValueError('managed workspace references require unique canonical root order')
        return self


class ManagedClaimRequest(StrictModel):
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    claimed_at: datetime

    @field_validator('claimed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed launch claim time')


class ManagedRedeemRequest(StrictModel):
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    launch_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_worker_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_rpc_session_id: str = Field(pattern=_RUN_ID_PATTERN)
    gateway_capability_id: str = Field(pattern=_SHA256_PATTERN)
    redeemed_at: datetime

    @field_validator('redeemed_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed start redemption time')


class ManagedStatusRequest(StrictModel):
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)


class ManagedBeginReconciliationRequest(StrictModel):
    startup_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    cleanup_receipt_key_id: str = Field(pattern=_SHA256_PATTERN)
    requested_at: datetime

    @field_validator('requested_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed reconciliation lease time')


class ManagedFinishReconciliationRequest(StrictModel):
    lease_token: str = Field(pattern=_SHA256_PATTERN)
    authenticated_cleanup: AuthenticatedManagedClinicalStartupCleanup


class ManagedRecordFailureRequest(StrictModel):
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    terminal_code: Literal[
        'scheduler_failure',
        'worker_launch_failure',
        'worker_terminal_failure',
        'worker_lost',
        'evidence_authentication_failed',
    ]
    failure_record_base64: str = Field(min_length=4, max_length=2 * _MAX_FAILURE_BYTES)
    terminal_at: datetime

    @field_validator('terminal_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed failure time')


class ManagedRecordRunRequest(StrictModel):
    reservation_sha256: str = Field(pattern=_SHA256_PATTERN)
    episode_id: str = Field(min_length=1, max_length=500)
    production_run_root: str
    terminal_at: datetime

    @field_validator('production_run_root')
    @classmethod
    def validate_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('production evidence root must be normalized and absolute')
        return value

    @field_validator('terminal_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed result time')


class ManagedClinicalRegistryRequest(StrictModel):
    schema_version: Literal['vaxreplay.managed-clinical-registry-request.dev-v0.1'] = (
        MANAGED_CLINICAL_REGISTRY_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_RUN_ID_PATTERN)
    operation: ManagedClinicalRegistryOperation
    payload: dict[str, Any]
    database_path_supplied: Literal[False] = False
    launcher_identity_supplied: Literal[False] = False


class ManagedClinicalRegistryResponse(StrictModel):
    schema_version: Literal['vaxreplay.managed-clinical-registry-response.dev-v0.1'] = (
        MANAGED_CLINICAL_REGISTRY_RESPONSE_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_RUN_ID_PATTERN)
    operation: ManagedClinicalRegistryOperation
    ok: bool
    result: dict[str, Any] | None = None
    error_code: Literal['invalid_request', 'unauthorized', 'rejected', 'internal'] | None = None
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    one_host_authority: Literal[True] = True
    cross_host_consensus_claimed: Literal[False] = False

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        if self.ok == (self.result is None) or self.ok == (self.error_code is not None):
            raise ValueError('managed registry response must contain exactly one result or error')
        return self


class ManagedClinicalRegistryAuditServerIdentity(StrictModel):
    """Service-side process and authoritative-file identity at one wire exchange."""

    service_pid: int = Field(gt=1, le=2**31 - 1)
    service_start_time_ticks: int = Field(gt=0, le=2**63 - 1)
    service_uid: int = Field(ge=0, le=2**31 - 1)
    service_gid: int = Field(ge=0, le=2**31 - 1)
    service_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    socket_path: str
    socket_device_id: int = Field(ge=0, le=2**63 - 1)
    socket_inode: int = Field(gt=0, le=2**63 - 1)
    database_path: str
    database_device_id: int = Field(ge=0, le=2**63 - 1)
    database_inode: int = Field(gt=0, le=2**63 - 1)

    @field_validator('socket_path', 'database_path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed registry audit paths must be normalized and absolute')
        return value


class AuthenticatedManagedClinicalRegistryAudit(StrictModel):
    """Create-once, HMAC-authenticated exact request/response wire evidence."""

    schema_version: Literal['vaxreplay.authenticated-managed-clinical-registry-audit.dev-v0.1'] = (
        AUTHENTICATED_MANAGED_CLINICAL_REGISTRY_AUDIT_SCHEMA_VERSION
    )
    registry_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    registry_authority_id: str = Field(pattern=_ID_PATTERN)
    sequence: int = Field(ge=0, le=2**63 - 1)
    predecessor_audit_sha256: str = Field(pattern=_SHA256_PATTERN)
    request: ManagedClinicalRegistryRequest
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    response: ManagedClinicalRegistryResponse
    response_sha256: str = Field(pattern=_SHA256_PATTERN)
    launcher_peer: ManagedClinicalPeerIdentity
    server: ManagedClinicalRegistryAuditServerIdentity
    audited_at: datetime
    audit_key_id: str = Field(pattern=_SHA256_PATTERN)
    audit_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    persisted_path: str
    persisted_create_once: Literal[True] = True
    file_fsync_complete: Literal[True] = True
    parent_directory_fsync_complete: Literal[True] = True
    response_not_sent_before_audit_persisted: Literal[True] = True
    request_and_response_content_free_of_provider_credentials: Literal[True] = True
    independent_host_attestation: Literal[False] = False

    @field_validator('audited_at')
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, 'managed registry audit time')

    @field_validator('persisted_path')
    @classmethod
    def validate_persisted_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or '..' in path.parts or str(path) != value:
            raise ValueError('managed registry audit path must be normalized and absolute')
        return value

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        if (
            self.request_sha256 != hashlib.sha256(canonical_json_bytes(self.request)).hexdigest()
            or self.response_sha256 != hashlib.sha256(canonical_json_bytes(self.response)).hexdigest()
            or self.response.request_id != self.request.request_id
            or self.response.operation != self.request.operation
            or self.response.registry_authority_id != self.registry_authority_id
            or self.sequence == 0
            and self.predecessor_audit_sha256 != _AUDIT_ZERO_SHA256
            or self.sequence > 0
            and self.predecessor_audit_sha256 == _AUDIT_ZERO_SHA256
        ):
            raise ValueError('managed registry audit has inconsistent exact bindings')
        return self


class ManagedRegistryBackend(Protocol):
    """Subset used by the service; the concrete implementation is the SQLite registry."""

    @property
    def authority_id(self) -> str: ...


class ManagedClinicalRegistryService:
    """Authenticated request dispatcher plus the root-only Unix-socket serving loop."""

    def __init__(
        self,
        *,
        config: ManagedClinicalRegistryConfig,
        workspace_receipt_keys_by_id: Mapping[str, bytes],
        evidence_reauthenticator: ProductionRunReauthenticator | None = None,
        registry: SqliteClinicalProductionRegistry | None = None,
        startup_config: ManagedClinicalStartupConfig | None = None,
        startup_cleanup_receipt_key: bytes | None = None,
        recovery_only: bool = False,
    ) -> None:
        if not isinstance(recovery_only, bool):
            raise TypeError('managed registry recovery-only mode must be a boolean')
        self.config = config
        self._workspace_keys = {key_id: bytes(key) for key_id, key in workspace_receipt_keys_by_id.items()}
        self._reauthenticate = evidence_reauthenticator
        self._registry = registry
        self._database_identity: tuple[int, int] | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._authority_lock_descriptor: int | None = None
        self._audit_sequence: int | None = None
        self._audit_predecessor_sha256: str | None = None
        self._audit_aggregate_bytes: int | None = None
        self._dispatch_lock = threading.RLock()
        self._reconciliation_required = True
        self._reconciliation_lease_token: str | None = None
        self._reconciliation_lease_peer_pid: int | None = None
        self._reconciliation_inventory_sha256: str | None = None
        self._startup_config = startup_config
        self._startup_cleanup_receipt_key = (
            None if startup_cleanup_receipt_key is None else bytes(startup_cleanup_receipt_key)
        )
        self._recovery_only = recovery_only
        if (startup_config is None) != (startup_cleanup_receipt_key is None):
            raise ValueError('managed registry startup config and cleanup key must be provisioned together')
        if startup_config is not None and (
            startup_config.registry_authority_id != config.registry_authority_id
            or managed_clinical_startup_config_sha256(startup_config) != config.startup_config_sha256
            or managed_clinical_cleanup_key_id(startup_cleanup_receipt_key or b'')
            != config.startup_cleanup_receipt_key_id
            or startup_config.cleanup_receipt_key_id != config.startup_cleanup_receipt_key_id
        ):
            raise ValueError('managed registry startup admission differs from deployment pins')

    @property
    def registry(self) -> SqliteClinicalProductionRegistry:
        if self._registry is None:
            self._registry = SqliteClinicalProductionRegistry(
                Path(self.config.database_path),
                authority_id=self.config.registry_authority_id,
            )
        if self._registry.authority_id != self.config.registry_authority_id:
            raise ManagedClinicalRegistryError('managed backend belongs to a different authority')
        return self._registry

    def handle_authenticated(
        self,
        request: ManagedClinicalRegistryRequest,
        *,
        peer: ManagedClinicalPeerIdentity,
    ) -> ManagedClinicalRegistryResponse:
        """Handle one already-authenticated peer; useful for deterministic adapter tests."""

        request = ManagedClinicalRegistryRequest.model_validate_json(canonical_json_bytes(request))
        peer = ManagedClinicalPeerIdentity.model_validate_json(canonical_json_bytes(peer))
        if (
            peer.uid,
            peer.gid,
            peer.canonical_launcher_id,
            peer.canonical_launcher_executable_sha256,
        ) != (
            self.config.allowed_launcher_uid,
            self.config.allowed_launcher_gid,
            self.config.canonical_launcher_id,
            self.config.canonical_launcher_executable_sha256,
        ):
            return self._error(request, 'unauthorized')
        try:
            with self._dispatch_lock:
                self._require_pinned_database_identity_if_serving()
                result = self._dispatch(request, peer=peer)
        except (ValueError, OSError, ManagedClinicalRegistryError):
            return self._error(request, 'rejected')
        except Exception:
            return self._error(request, 'internal')
        return ManagedClinicalRegistryResponse(
            request_id=request.request_id,
            operation=request.operation,
            ok=True,
            result=result,
            registry_authority_id=self.config.registry_authority_id,
        )

    def serve_forever(self) -> None:
        """Serve one request per connection on the configured root-owned Unix socket."""

        self.serve_until()

    def serve_until(
        self,
        *,
        stop_event: threading.Event | None = None,
        ready_event: threading.Event | None = None,
    ) -> None:
        """Serve until an optional deployment-owned stop event is set.

        ``serve_forever`` remains the ordinary daemon API.  The bounded standalone managed
        composition uses this lifecycle-aware variant so it can stop and join the authority
        before its one-shot systemd unit exits.  Readiness is published only after the database,
        authority lock, socket ownership, and listener have all been established.  A supplied
        stop event never opens the registry or skips startup reconciliation.
        """

        socket_path = Path(self.config.socket_path)
        listener: socket.socket | None = None
        try:
            # Defend against unchecked ``model_copy(update=...)`` values before
            # database construction can create the authoritative SQLite file.
            _require_linux_af_unix_pathname(socket_path)
            self._require_root_host_boundary()
            self._prepare_and_pin_database()
            if socket_path.exists() or socket_path.is_symlink():
                self._remove_stale_owned_socket(socket_path)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(socket_path))
            os.chown(
                socket_path,
                self.config.allowed_launcher_uid,
                self.config.allowed_launcher_gid,
                follow_symlinks=False,
            )
            os.chmod(socket_path, 0o600, follow_symlinks=False)
            metadata = socket_path.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or (metadata.st_uid, metadata.st_gid)
                != (
                    self.config.allowed_launcher_uid,
                    self.config.allowed_launcher_gid,
                )
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ManagedClinicalRegistryError('managed registry socket ownership could not be pinned')
            self._socket_identity = (metadata.st_dev, metadata.st_ino)
            _fsync_directory(socket_path.parent)
            self._initialize_protocol_audit()
            listener.listen(16)
            if stop_event is not None:
                listener.settimeout(min(self.config.connection_timeout_seconds, 0.25))
            if ready_event is not None:
                ready_event.set()
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(self.config.connection_timeout_seconds)
                    self._serve_connection(connection)
        finally:
            if listener is not None:
                listener.close()
            try:
                metadata = socket_path.lstat()
                if (
                    stat.S_ISSOCK(metadata.st_mode)
                    and self._socket_identity == (metadata.st_dev, metadata.st_ino)
                    and metadata.st_uid == self.config.allowed_launcher_uid
                    and metadata.st_gid == self.config.allowed_launcher_gid
                ):
                    socket_path.unlink()
                    _fsync_directory(socket_path.parent)
            except (ManagedClinicalRegistryError, OSError):
                pass
            self._socket_identity = None
            self._audit_sequence = None
            self._audit_predecessor_sha256 = None
            self._audit_aggregate_bytes = None
            self._release_authority_lock()

    def _serve_connection(self, connection: socket.socket) -> None:
        deadline = time.monotonic() + self.config.connection_timeout_seconds
        try:
            peer = authenticate_managed_registry_peer(connection, config=self.config)
            request_bytes = _recv_frame(
                connection,
                self.config.maximum_frame_bytes,
                deadline_monotonic=deadline,
            )
            request = ManagedClinicalRegistryRequest.model_validate_json(request_bytes)
            if not hmac.compare_digest(request_bytes, canonical_json_bytes(request)):
                raise ManagedClinicalRegistryError('managed registry request must use exact canonical JSON')
            response = self.handle_authenticated(request, peer=peer)
            self._persist_protocol_audit(
                request=request,
                response=response,
                peer=peer,
            )
        except (ManagedClinicalRegistryError, OSError, ValueError):
            return
        try:
            connection.settimeout(max(deadline - time.monotonic(), 1e-6))
            _send_frame(connection, canonical_json_bytes(response), self.config.maximum_frame_bytes)
        except (ManagedClinicalRegistryError, OSError):
            return

    def _initialize_protocol_audit(self) -> None:
        key = self._startup_cleanup_receipt_key
        if key is None:
            raise ManagedClinicalRegistryError('managed wire service requires the provisioned protocol-audit key')
        required_uid = os.geteuid()
        root = _prepare_private_audit_root(
            Path(self.config.protocol_audit_root),
            required_uid=required_uid,
        )
        _reap_incomplete_protocol_audit_staging(root)
        chain = load_authenticated_managed_registry_audit_chain(
            root,
            key=key,
            expected_key_id=self.config.startup_cleanup_receipt_key_id,
            expected_config_sha256=managed_clinical_registry_config_sha256(self.config),
            required_uid=required_uid,
        )
        sequence = len(chain)
        aggregate_bytes = sum(len(canonical_json_bytes(item)) for item in chain)
        if (
            sequence >= MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES
            or aggregate_bytes >= MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES
        ):
            raise ManagedClinicalRegistryError(
                'managed registry protocol audit has no bounded capacity for another response'
            )
        self._audit_sequence = sequence
        self._audit_aggregate_bytes = aggregate_bytes
        self._audit_predecessor_sha256 = (
            _AUDIT_ZERO_SHA256 if not chain else authenticated_managed_clinical_registry_audit_sha256(chain[-1])
        )

    def _persist_protocol_audit(
        self,
        *,
        request: ManagedClinicalRegistryRequest,
        response: ManagedClinicalRegistryResponse,
        peer: ManagedClinicalPeerIdentity,
    ) -> AuthenticatedManagedClinicalRegistryAudit:
        key = self._startup_cleanup_receipt_key
        sequence = self._audit_sequence
        predecessor = self._audit_predecessor_sha256
        aggregate_bytes = self._audit_aggregate_bytes
        if key is None or sequence is None or predecessor is None or aggregate_bytes is None:
            raise ManagedClinicalRegistryError('managed registry protocol audit is not initialized')
        if sequence >= MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES:
            raise ManagedClinicalRegistryError('managed registry protocol audit entry-count limit is exhausted')
        server = self._protocol_audit_server_identity()
        root = Path(self.config.protocol_audit_root)
        path = root / f'{sequence:020d}-{request.request_id}.json'
        unsigned = AuthenticatedManagedClinicalRegistryAudit(
            registry_config_sha256=managed_clinical_registry_config_sha256(self.config),
            registry_authority_id=self.config.registry_authority_id,
            sequence=sequence,
            predecessor_audit_sha256=predecessor,
            request=request,
            request_sha256=hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
            response=response,
            response_sha256=hashlib.sha256(canonical_json_bytes(response)).hexdigest(),
            launcher_peer=peer,
            server=server,
            audited_at=datetime.now(UTC),
            audit_key_id=self.config.startup_cleanup_receipt_key_id,
            audit_hmac_sha256=_AUDIT_ZERO_SHA256,
            persisted_path=str(path),
        )
        artifact = unsigned.model_copy(
            update={
                'audit_hmac_sha256': managed_clinical_registry_audit_hmac(
                    unsigned,
                    key=key,
                )
            }
        )
        encoded_artifact = canonical_json_bytes(artifact)
        if aggregate_bytes + len(encoded_artifact) > MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES:
            raise ManagedClinicalRegistryError('managed registry protocol audit aggregate-byte limit is exhausted')
        _write_create_once_audit(path, encoded_artifact)
        persisted = load_authenticated_managed_clinical_registry_audit(
            path,
            expected_root=root,
        )
        verify_authenticated_managed_clinical_registry_audit(
            persisted,
            key=key,
            expected_key_id=self.config.startup_cleanup_receipt_key_id,
            expected_config_sha256=managed_clinical_registry_config_sha256(self.config),
            expected_sequence=sequence,
            expected_predecessor_sha256=predecessor,
        )
        if persisted != artifact:
            raise ManagedClinicalRegistryError(
                'persisted managed registry protocol audit differs from exact wire evidence'
            )
        self._audit_sequence = sequence + 1
        self._audit_aggregate_bytes = aggregate_bytes + len(encoded_artifact)
        self._audit_predecessor_sha256 = authenticated_managed_clinical_registry_audit_sha256(persisted)
        return persisted

    def _protocol_audit_server_identity(self) -> ManagedClinicalRegistryAuditServerIdentity:
        database_identity = self._database_identity
        socket_identity = self._socket_identity
        if database_identity is None or socket_identity is None:
            raise ManagedClinicalRegistryError(
                'managed registry authority identities are unavailable for protocol audit'
            )
        executable_sha256 = _sha256_file(Path('/proc/self/exe'))
        if not hmac.compare_digest(
            executable_sha256,
            self.config.service_process_executable_sha256,
        ):
            raise ManagedClinicalRegistryError('managed registry service image differs from its protocol-audit pin')
        return ManagedClinicalRegistryAuditServerIdentity(
            service_pid=os.getpid(),
            service_start_time_ticks=_linux_process_start_time_ticks(os.getpid()),
            service_uid=os.geteuid(),
            service_gid=os.getegid(),
            service_executable_sha256=executable_sha256,
            socket_path=self.config.socket_path,
            socket_device_id=socket_identity[0],
            socket_inode=socket_identity[1],
            database_path=self.config.database_path,
            database_device_id=database_identity[0],
            database_inode=database_identity[1],
        )

    def _dispatch(
        self,
        request: ManagedClinicalRegistryRequest,
        *,
        peer: ManagedClinicalPeerIdentity,
    ) -> dict[str, Any]:
        operation = request.operation
        if operation == 'begin_reconciliation':
            payload = ManagedBeginReconciliationRequest.model_validate_json(canonical_json_bytes(request.payload))
            return self._begin_reconciliation(payload, peer=peer)
        if operation == 'finish_reconciliation':
            payload = ManagedFinishReconciliationRequest.model_validate_json(canonical_json_bytes(request.payload))
            return self._finish_reconciliation(payload, peer=peer)
        if self._recovery_only and operation not in {'status', 'record_failure'}:
            raise ManagedClinicalRegistryError('recovery-only registry rejects work admission and success publication')
        if operation in _MUTATING_OPERATIONS and self._reconciliation_required:
            raise ManagedClinicalRegistryError(
                'managed registry is quiesced pending authenticated startup reconciliation'
            )
        if operation == 'reserve':
            payload = ManagedReserveRequest.model_validate_json(canonical_json_bytes(request.payload))
            self._require_system_launcher(payload.system)
            workspaces = []
            for reference in payload.workspaces:
                key = self._workspace_keys.get(reference.expected_receipt_key_id)
                if key is None:
                    raise ManagedClinicalRegistryError('workspace key is not provisioned by the service')
                workspaces.append(
                    load_clinical_agentic_workspace(
                        Path(reference.root),
                        expected_authenticated_receipt_sha256=(reference.expected_authenticated_receipt_sha256),
                        receipt_key=key,
                        expected_receipt_key_id=reference.expected_receipt_key_id,
                    )
                )
            context = self.registry.reserve_cohort(
                manifest=payload.manifest,
                workspaces=workspaces,
                workspace_receipt_keys_by_id=self._workspace_keys,
                system=payload.system,
                registered_entry_id=payload.registered_entry_id,
                reserved_at=payload.reserved_at,
            )
            return {
                'reservation': context.reservation.model_dump(mode='json'),
                'reservation_sha256': context.reservation_sha256,
            }
        if operation == 'claim':
            payload = ManagedClaimRequest.model_validate_json(canonical_json_bytes(request.payload))
            context = self.registry.reservation_context(payload.reservation_sha256)
            self._require_service_launcher(context)
            result = self.registry.claim_task_launch(**payload.model_dump(mode='python'))
            return {'launch': result.model_dump(mode='json')}
        if operation == 'redeem':
            payload = ManagedRedeemRequest.model_validate_json(canonical_json_bytes(request.payload))
            context = self.registry.reservation_context(payload.reservation_sha256)
            self._require_service_launcher(context)
            result = self.registry.redeem_task_start(
                **payload.model_dump(mode='python'),
                canonical_launcher_id=self.config.canonical_launcher_id,
                canonical_launcher_executable_sha256=(self.config.canonical_launcher_executable_sha256),
            )
            return {'start_redemption': result.model_dump(mode='json')}
        if operation == 'status':
            payload = ManagedStatusRequest.model_validate_json(canonical_json_bytes(request.payload))
            context = self.registry.reservation_context(payload.reservation_sha256)
            self._require_service_launcher(context)
            records = self.registry.task_records(payload.reservation_sha256)
            return {
                'reservation': context.reservation.model_dump(mode='json'),
                'reservation_sha256': context.reservation_sha256,
                'task_records': [item.model_dump(mode='json') for item in records],
            }
        if operation == 'record_failure':
            payload = ManagedRecordFailureRequest.model_validate_json(canonical_json_bytes(request.payload))
            context = self.registry.reservation_context(payload.reservation_sha256)
            self._require_service_launcher(context)
            try:
                failure = base64.b64decode(payload.failure_record_base64, validate=True)
            except ValueError:
                raise ManagedClinicalRegistryError('failure record is not strict base64') from None
            if not 0 < len(failure) <= _MAX_FAILURE_BYTES:
                raise ManagedClinicalRegistryError('failure record is outside the service limit')
            terminal_code = cast(
                ClinicalProductionExplicitFailureCode,
                ClinicalProductionTerminalCode(payload.terminal_code),
            )
            result = self.registry.record_explicit_failure(
                reservation_sha256=payload.reservation_sha256,
                episode_id=payload.episode_id,
                terminal_code=terminal_code,
                failure_record=failure,
                terminal_at=payload.terminal_at,
            )
            return {'task_record': result.model_dump(mode='json')}
        if operation == 'record_run':
            payload = ManagedRecordRunRequest.model_validate_json(canonical_json_bytes(request.payload))
            context = self.registry.reservation_context(payload.reservation_sha256)
            self._require_service_launcher(context)
            if self._reauthenticate is None:
                raise ManagedClinicalRegistryError('service has no pinned evidence reauthenticator')
            production_run_root = self._require_production_run_root(payload)
            result = self.registry.record_production_run(
                reservation_sha256=payload.reservation_sha256,
                episode_id=payload.episode_id,
                production_run_root=production_run_root,
                reauthenticate=self._reauthenticate,
                terminal_at=payload.terminal_at,
            )
            return {'task_record': result.model_dump(mode='json')}
        raise ManagedClinicalRegistryError('unsupported managed registry operation')

    def _begin_reconciliation(
        self,
        payload: ManagedBeginReconciliationRequest,
        *,
        peer: ManagedClinicalPeerIdentity,
    ) -> dict[str, Any]:
        if (
            payload.startup_config_sha256 != self.config.startup_config_sha256
            or payload.cleanup_receipt_key_id != self.config.startup_cleanup_receipt_key_id
        ):
            raise ManagedClinicalRegistryError('reconciliation request differs from the service startup pins')
        if not self._reconciliation_required:
            # A caller cannot suspend an already-live authority.  Only a service restart closes
            # it again, avoiding an unprivileged denial-of-service primitive on the shared socket.
            raise ManagedClinicalRegistryError('managed registry startup reconciliation is already complete')
        inventory = self._authoritative_attempt_inventory()
        inventory_sha256 = _managed_attempt_inventory_sha256(inventory)
        if self._reconciliation_lease_token is None:
            self._reconciliation_lease_token = os.urandom(32).hex()
            self._reconciliation_lease_peer_pid = peer.pid
            self._reconciliation_inventory_sha256 = inventory_sha256
        elif self._reconciliation_lease_peer_pid != peer.pid:
            raise ManagedClinicalRegistryError('startup reconciliation lease is held by another launcher process')
        elif not hmac.compare_digest(inventory_sha256, self._reconciliation_inventory_sha256 or ''):
            raise ManagedClinicalRegistryError('authoritative attempt inventory changed during startup quiescence')
        return {
            'lease_token': self._reconciliation_lease_token,
            'attempt_inventory_sha256': inventory_sha256,
            'attempts': [item.model_dump(mode='json') for item in inventory],
            'mutations_quiesced': True,
            'service_restart_recloses_authority': True,
        }

    def _finish_reconciliation(
        self,
        payload: ManagedFinishReconciliationRequest,
        *,
        peer: ManagedClinicalPeerIdentity,
    ) -> dict[str, Any]:
        key = self._startup_cleanup_receipt_key
        startup_config = self._startup_config
        if key is None or startup_config is None:
            raise ManagedClinicalRegistryError('service has no provisioned startup cleanup verifier')
        if (
            not self._reconciliation_required
            or self._reconciliation_lease_token is None
            or self._reconciliation_lease_peer_pid != peer.pid
            or self._reconciliation_inventory_sha256 is None
            or not hmac.compare_digest(
                payload.lease_token,
                self._reconciliation_lease_token,
            )
        ):
            raise ManagedClinicalRegistryError('reconciliation lease is absent or stale')
        artifact = payload.authenticated_cleanup
        persisted = load_authenticated_managed_cleanup(
            Path(artifact.persisted_path),
            expected_root=Path(startup_config.receipt_root),
        )
        if persisted != artifact:
            raise ManagedClinicalRegistryError('cleanup receipt differs from its create-once persisted artifact')
        receipt = verify_authenticated_managed_cleanup(
            persisted,
            key=key,
            expected_key_id=self.config.startup_cleanup_receipt_key_id,
            expected_config_sha256=self.config.startup_config_sha256,
            expected_request_sha256=artifact.request_sha256,
        )
        inventory = self._authoritative_attempt_inventory()
        inventory_sha256 = _managed_attempt_inventory_sha256(inventory)
        expected_inventory_sha256 = self._reconciliation_inventory_sha256
        if (
            not hmac.compare_digest(inventory_sha256, expected_inventory_sha256)
            or not hmac.compare_digest(
                artifact.attempt_inventory_sha256,
                expected_inventory_sha256,
            )
            or not hmac.compare_digest(
                receipt.attempt_registry_inventory_sha256,
                expected_inventory_sha256,
            )
            or receipt.reconciliation_request_sha256 != artifact.request_sha256
        ):
            raise ManagedClinicalRegistryError('cleanup receipt differs from the quiesced attempt snapshot')
        self._reconciliation_required = False
        self._reconciliation_lease_token = None
        self._reconciliation_lease_peer_pid = None
        self._reconciliation_inventory_sha256 = None
        return {
            'startup_reconciliation_admitted': True,
            'attempt_inventory_sha256': inventory_sha256,
            'mutations_quiesced': False,
        }

    def _authoritative_attempt_inventory(
        self,
    ) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        values: list[ManagedClinicalAttemptInventoryRecord] = []
        for reservation_sha256 in self.registry.reservation_hashes():
            context = self.registry.reservation_context(reservation_sha256)
            self._require_service_launcher(context)
            reservation = context.reservation
            for record in self.registry.task_records(reservation_sha256):
                launch = record.launch
                if launch is None:
                    continue
                if record.state == 'reserved':
                    raise ManagedClinicalRegistryError('attempt inventory has launch data in a reserved record')
                redemption = record.start_redemption
                values.append(
                    ManagedClinicalAttemptInventoryRecord(
                        registry_authority_id=reservation.registry_authority_id,
                        reservation_sha256=reservation_sha256,
                        launch_sha256=clinical_production_task_launch_sha256(launch),
                        start_redemption_sha256=(
                            None if redemption is None else clinical_production_start_redemption_sha256(redemption)
                        ),
                        run_id=launch.run_id,
                        episode_id=record.episode_id,
                        worker_spec_sha256=reservation.system.worker_spec_sha256,
                        state=record.state,
                    )
                )
        return tuple(sorted(values, key=lambda item: (item.run_id, item.reservation_sha256)))

    def _require_service_launcher(self, context: ClinicalProductionReservationContext) -> None:
        self._require_system_launcher(context.reservation.system)

    def _require_system_launcher(self, system: ClinicalProductionSystemIdentity) -> None:
        if (
            system.canonical_launcher_id,
            system.canonical_launcher_executable_sha256,
        ) != (
            self.config.canonical_launcher_id,
            self.config.canonical_launcher_executable_sha256,
        ):
            raise ManagedClinicalRegistryError('reservation belongs to a different launcher identity')

    def _require_production_run_root(self, payload: ManagedRecordRunRequest) -> Path:
        record = next(
            (
                item
                for item in self.registry.task_records(payload.reservation_sha256)
                if item.episode_id == payload.episode_id
            ),
            None,
        )
        if record is None or record.launch is None:
            raise ManagedClinicalRegistryError('production evidence has no exact consumed task launch')
        expected = Path(self.config.production_evidence_root) / record.launch.run_id
        candidate = Path(payload.production_run_root)
        if candidate != expected:
            raise ManagedClinicalRegistryError('production evidence root differs from the service-owned run namespace')
        try:
            namespace = Path(self.config.production_evidence_root).lstat()
            metadata = candidate.lstat()
        except OSError:
            raise ManagedClinicalRegistryError('production evidence root is unavailable') from None
        if (
            stat.S_ISLNK(namespace.st_mode)
            or not stat.S_ISDIR(namespace.st_mode)
            or namespace.st_uid != 0
            or stat.S_IMODE(namespace.st_mode) != 0o700
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ManagedClinicalRegistryError('production evidence namespace must be root-owned mode-0700 directories')
        return candidate

    def _prepare_and_pin_database(self) -> None:
        if self._authority_lock_descriptor is not None:
            raise ManagedClinicalRegistryError('managed registry service database authority is already locked')
        if self._registry is not None:
            backend_path = getattr(self._registry, 'path', None)
            if not isinstance(backend_path, Path):
                raise ManagedClinicalRegistryError(
                    'served managed registry backend must expose its configured database path'
                )
            try:
                expected = Path(self.config.database_path).resolve(strict=False)
                observed = backend_path.resolve(strict=True)
            except OSError:
                raise ManagedClinicalRegistryError('served managed registry backend path is unavailable') from None
            if observed != expected:
                raise ManagedClinicalRegistryError(
                    'served managed registry backend differs from the service-owned database path'
                )
        # Eager construction creates and validates the database before the authority socket exists.
        # That prevents a request from racing first-use initialization and lets us pin one inode.
        _ = self.registry
        database = Path(self.config.database_path)
        self._require_root_host_boundary()
        metadata = database.lstat()
        self._database_identity = (metadata.st_dev, metadata.st_ino)
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(database, flags)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != self._database_identity:
                raise ManagedClinicalRegistryError('managed registry database changed while acquiring its service lock')
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _fsync_directory(database.parent)
        except (OSError, ManagedClinicalRegistryError):
            if descriptor is not None:
                os.close(descriptor)
            raise ManagedClinicalRegistryError(
                'another managed registry service already owns this database authority'
            ) from None
        self._authority_lock_descriptor = descriptor

    def _release_authority_lock(self) -> None:
        descriptor = self._authority_lock_descriptor
        self._authority_lock_descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _remove_stale_owned_socket(self, socket_path: Path) -> None:
        try:
            before = socket_path.lstat()
        except OSError:
            raise ManagedClinicalRegistryError(
                'managed registry socket changed during stale-socket reconciliation'
            ) from None
        if (
            not stat.S_ISSOCK(before.st_mode)
            or before.st_uid != self.config.allowed_launcher_uid
            or before.st_gid != self.config.allowed_launcher_gid
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise ManagedClinicalRegistryError('managed registry refuses an unowned existing socket path')
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(min(self.config.connection_timeout_seconds, 1.0))
            try:
                probe.connect(str(socket_path))
            except OSError as error:
                if error.errno != errno.ECONNREFUSED:
                    raise ManagedClinicalRegistryError(
                        'managed registry could not prove the existing socket is stale'
                    ) from None
            else:
                raise ManagedClinicalRegistryError('another managed registry service is already listening')
        finally:
            probe.close()
        try:
            after = socket_path.lstat()
        except OSError:
            raise ManagedClinicalRegistryError(
                'managed registry socket changed during stale-socket reconciliation'
            ) from None
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise ManagedClinicalRegistryError('managed registry socket changed during stale-socket reconciliation')
        socket_path.unlink()
        _fsync_directory(socket_path.parent)

    def _require_pinned_database_identity_if_serving(self) -> None:
        expected = self._database_identity
        if expected is None:
            return
        try:
            metadata = Path(self.config.database_path).lstat()
        except OSError:
            raise ManagedClinicalRegistryError(
                'managed registry database disappeared while the service was running'
            ) from None
        if (
            (metadata.st_dev, metadata.st_ino) != expected
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ManagedClinicalRegistryError(
                'managed registry database identity changed while the service was running'
            )

    def _require_root_host_boundary(self) -> None:
        if os.geteuid() != 0:
            raise ManagedClinicalRegistryError('managed registry service requires effective UID 0')
        for path in (
            Path(self.config.database_path).parent,
            Path(self.config.socket_path).parent,
            Path(self.config.production_evidence_root),
            Path(self.config.protocol_audit_root),
        ):
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != 0
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ManagedClinicalRegistryError('managed registry parents must be root-owned mode-0700')
        database = Path(self.config.database_path)
        if database.exists():
            metadata = database.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != 0
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ManagedClinicalRegistryError('managed registry database must be root-owned mode-0600')

    def _error(
        self,
        request: ManagedClinicalRegistryRequest,
        code: Literal['invalid_request', 'unauthorized', 'rejected', 'internal'],
    ) -> ManagedClinicalRegistryResponse:
        return ManagedClinicalRegistryResponse(
            request_id=request.request_id,
            operation=request.operation,
            ok=False,
            error_code=code,
            registry_authority_id=self.config.registry_authority_id,
        )


class ManagedClinicalRegistryClient:
    """Socket client matching the launcher-facing registry operations.

    Production evidence is authoritatively verified inside the root-owned service, so a launcher
    cannot weaken that decision.  After the service reports success, the client also invokes the
    launcher's verifier exactly once.  That second, independent reload is part of the launcher
    boundary contract and supplies the authenticated object returned by ``CanonicalClinicalLauncher``.
    """

    def __init__(
        self,
        config: ManagedClinicalRegistryConfig,
        *,
        expected_config_sha256: str,
    ) -> None:
        observed = managed_clinical_registry_config_sha256(config)
        if not hmac.compare_digest(observed, expected_config_sha256):
            raise ManagedClinicalRegistryError('managed registry client config differs from its external SHA-256 pin')
        self.config = config
        self.config_sha256 = observed
        self.socket_path = Path(config.socket_path)
        self.authority_id = config.registry_authority_id
        self._reconciliation_lock = threading.RLock()
        self._reconciliation_lease_token: str | None = None
        self._reconciliation_inventory: tuple[ManagedClinicalAttemptInventoryRecord, ...] | None = None
        self._reconciliation_inventory_sha256: str | None = None

    def begin_reconciliation(
        self,
        *,
        requested_at: datetime | None = None,
    ) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        """Acquire the boot-time quiescence lease and its complete attempt snapshot."""

        with self._reconciliation_lock:
            if self._reconciliation_inventory is not None:
                return self._reconciliation_inventory
            result = self._call(
                'begin_reconciliation',
                ManagedBeginReconciliationRequest(
                    startup_config_sha256=self.config.startup_config_sha256,
                    cleanup_receipt_key_id=(self.config.startup_cleanup_receipt_key_id),
                    requested_at=requested_at or datetime.now(UTC),
                ),
            )
            inventory = tuple(ManagedClinicalAttemptInventoryRecord.model_validate(item) for item in result['attempts'])
            if inventory != tuple(sorted(inventory, key=lambda item: (item.run_id, item.reservation_sha256))):
                raise ManagedClinicalRegistryError('managed reconciliation inventory is not in canonical order')
            inventory_sha256 = _managed_attempt_inventory_sha256(inventory)
            if (
                result.get('mutations_quiesced') is not True
                or result.get('service_restart_recloses_authority') is not True
                or not hmac.compare_digest(
                    str(result['attempt_inventory_sha256']),
                    inventory_sha256,
                )
            ):
                raise ManagedClinicalRegistryError('managed reconciliation lease response is incomplete')
            self._reconciliation_lease_token = str(result['lease_token'])
            self._reconciliation_inventory = inventory
            self._reconciliation_inventory_sha256 = inventory_sha256
            return inventory

    def inventory(self) -> tuple[ManagedClinicalAttemptInventoryRecord, ...]:
        """Implement the startup attempt-inventory protocol under the service lease."""

        return self.begin_reconciliation()

    def finish_reconciliation(
        self,
        authenticated_cleanup: AuthenticatedManagedClinicalStartupCleanup,
    ) -> None:
        """Release quiescence only after the service verifies exact persisted cleanup evidence."""

        with self._reconciliation_lock:
            token = self._reconciliation_lease_token
            inventory_sha256 = self._reconciliation_inventory_sha256
            if token is None or inventory_sha256 is None:
                raise ManagedClinicalRegistryError('managed reconciliation was not begun by this client')
            result = self._call(
                'finish_reconciliation',
                ManagedFinishReconciliationRequest(
                    lease_token=token,
                    authenticated_cleanup=authenticated_cleanup,
                ),
            )
            if (
                result.get('startup_reconciliation_admitted') is not True
                or result.get('mutations_quiesced') is not False
                or not hmac.compare_digest(
                    str(result['attempt_inventory_sha256']),
                    inventory_sha256,
                )
            ):
                raise ManagedClinicalRegistryError('managed registry returned incomplete startup admission')
            self._reconciliation_lease_token = None
            self._reconciliation_inventory = None
            self._reconciliation_inventory_sha256 = None

    def reservation_context(self, reservation_sha256: str) -> ClinicalProductionReservationContext:
        result = self._call('status', ManagedStatusRequest(reservation_sha256=reservation_sha256))
        return ClinicalProductionReservationContext(
            reservation=_reservation(result['reservation']),
            reservation_sha256=str(result['reservation_sha256']),
        )

    def reserve_managed(
        self,
        *,
        manifest: ExecutionCohortManifest,
        workspaces: tuple[ManagedWorkspaceReference, ...],
        system: ClinicalProductionSystemIdentity,
        registered_entry_id: str,
        reserved_at: datetime,
    ) -> ClinicalProductionReservationContext:
        """Reserve through service-provisioned workspace keys; no key bytes cross the socket."""

        result = self._call(
            'reserve',
            ManagedReserveRequest(
                manifest=manifest,
                workspaces=workspaces,
                system=system,
                registered_entry_id=registered_entry_id,
                reserved_at=reserved_at,
            ),
        )
        return ClinicalProductionReservationContext(
            reservation=_reservation(result['reservation']),
            reservation_sha256=str(result['reservation_sha256']),
        )

    def task_records(self, reservation_sha256: str) -> tuple[ClinicalProductionTaskRecord, ...]:
        result = self._call('status', ManagedStatusRequest(reservation_sha256=reservation_sha256))
        return tuple(
            ClinicalProductionTaskRecord.model_validate_json(canonical_json_bytes(item))
            for item in result['task_records']
        )

    def claim_task_launch(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        run_id: str,
        claimed_at: datetime,
    ) -> ClinicalProductionTaskLaunch:
        result = self._call(
            'claim',
            ManagedClaimRequest(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                run_id=run_id,
                claimed_at=claimed_at,
            ),
        )
        return ClinicalProductionTaskLaunch.model_validate_json(canonical_json_bytes(result['launch']))

    def redeem_task_start(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        launch_sha256: str,
        canonical_launcher_id: str,
        canonical_launcher_executable_sha256: str,
        prepared_worker_sha256: str,
        guest_rpc_session_id: str,
        gateway_capability_id: str,
        redeemed_at: datetime,
    ) -> ClinicalProductionStartRedemption:
        # Identity parameters are compatibility inputs only.  They are not sent; the service uses
        # its authenticated peer and fixed deployment configuration.
        del canonical_launcher_id, canonical_launcher_executable_sha256
        result = self._call(
            'redeem',
            ManagedRedeemRequest(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                launch_sha256=launch_sha256,
                prepared_worker_sha256=prepared_worker_sha256,
                guest_rpc_session_id=guest_rpc_session_id,
                gateway_capability_id=gateway_capability_id,
                redeemed_at=redeemed_at,
            ),
        )
        return ClinicalProductionStartRedemption.model_validate_json(canonical_json_bytes(result['start_redemption']))

    def record_explicit_failure(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        terminal_code: ClinicalProductionExplicitFailureCode,
        failure_record: bytes,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord:
        result = self._call(
            'record_failure',
            ManagedRecordFailureRequest(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                terminal_code=terminal_code.value,
                failure_record_base64=base64.b64encode(failure_record).decode('ascii'),
                terminal_at=terminal_at,
            ),
        )
        return ClinicalProductionTaskRecord.model_validate_json(canonical_json_bytes(result['task_record']))

    def record_production_run(
        self,
        *,
        reservation_sha256: str,
        episode_id: str,
        production_run_root: Path,
        reauthenticate: ProductionRunReauthenticator,
        terminal_at: datetime,
    ) -> ClinicalProductionTaskRecord:
        result = self._call(
            'record_run',
            ManagedRecordRunRequest(
                reservation_sha256=reservation_sha256,
                episode_id=episode_id,
                production_run_root=str(production_run_root),
                terminal_at=terminal_at,
            ),
        )
        record = ClinicalProductionTaskRecord.model_validate_json(canonical_json_bytes(result['task_record']))
        if record.state == 'succeeded':
            # The service has already made and persisted the authoritative verification decision.
            # This callback cannot turn a service rejection into success; it only preserves the
            # launcher-facing promise that a successful record performs one independent reload.
            # Use the hash from the validated terminal record rather than a caller-provided value.
            start_redemption_sha256 = record.start_redemption_sha256
            if start_redemption_sha256 is None:  # Defensive despite the task-record model invariant.
                raise ManagedClinicalRegistryError(
                    'managed registry returned success without a start-redemption commitment'
                )
            reloaded = reauthenticate(production_run_root, start_redemption_sha256)
            reloaded_evidence_sha256 = getattr(reloaded, 'authenticated_outer_receipt_sha256', None)
            if (
                record.evidence_sha256 is None
                or not isinstance(reloaded_evidence_sha256, str)
                or not hmac.compare_digest(reloaded_evidence_sha256, record.evidence_sha256)
            ):
                raise ManagedClinicalRegistryError(
                    'launcher evidence reload differs from the authoritative managed-registry digest'
                )
        return record

    def _call(
        self,
        operation: ManagedClinicalRegistryOperation,
        payload: StrictModel,
    ) -> dict[str, Any]:
        request = ManagedClinicalRegistryRequest(
            request_id=os.urandom(16).hex(),
            operation=operation,
            payload=payload.model_dump(mode='json'),
        )
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + self.config.connection_timeout_seconds
        try:
            socket_identity = _require_managed_socket_metadata(self.socket_path, config=self.config)
            connection.settimeout(self.config.connection_timeout_seconds)
            connection.connect(str(self.socket_path))
            _require_managed_socket_metadata(
                self.socket_path,
                config=self.config,
                expected_identity=socket_identity,
            )
            authenticate_managed_registry_server(connection, config=self.config)
            connection.settimeout(max(deadline - time.monotonic(), 1e-6))
            _send_frame(
                connection,
                canonical_json_bytes(request),
                self.config.maximum_frame_bytes,
            )
            response_bytes = _recv_frame(
                connection,
                self.config.maximum_frame_bytes,
                deadline_monotonic=deadline,
            )
            response = ManagedClinicalRegistryResponse.model_validate_json(response_bytes)
            if not hmac.compare_digest(response_bytes, canonical_json_bytes(response)):
                raise ManagedClinicalRegistryError('managed registry response must use exact canonical JSON')
        except (OSError, ValueError):
            raise ManagedClinicalRegistryError('managed registry request failed') from None
        finally:
            connection.close()
        if (
            response.request_id != request.request_id
            or response.operation != request.operation
            or response.registry_authority_id != self.authority_id
            or not response.ok
            or response.result is None
        ):
            raise ManagedClinicalRegistryError('managed registry rejected or mismatched the request')
        return response.result


def authenticate_managed_registry_peer(
    connection: socket.socket,
    *,
    config: ManagedClinicalRegistryConfig,
) -> ManagedClinicalPeerIdentity:
    """Read Linux kernel peer credentials; no identity field is accepted from the frame."""

    if not hasattr(socket, 'SO_PEERCRED'):
        raise ManagedClinicalRegistryError('managed registry peer authentication requires Linux SO_PEERCRED')
    try:
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
        pid, uid, gid = struct.unpack('3i', raw)
    except (OSError, struct.error):
        raise ManagedClinicalRegistryError('managed registry peer credentials are unavailable') from None
    if (uid, gid) != (config.allowed_launcher_uid, config.allowed_launcher_gid):
        raise ManagedClinicalRegistryError('managed registry peer is not the configured launcher account')
    _require_linux_process_image(
        pid,
        expected_sha256=config.launcher_process_executable_sha256,
        label='launcher',
    )
    return ManagedClinicalPeerIdentity(
        pid=pid,
        uid=uid,
        gid=gid,
        canonical_launcher_id=config.canonical_launcher_id,
        canonical_launcher_executable_sha256=config.canonical_launcher_executable_sha256,
    )


def authenticate_managed_registry_server(
    connection: socket.socket,
    *,
    config: ManagedClinicalRegistryConfig,
) -> ManagedClinicalPeerIdentity:
    """Authenticate the service process from the client side using the same kernel boundary."""

    if not hasattr(socket, 'SO_PEERCRED'):
        raise ManagedClinicalRegistryError('managed registry server authentication requires Linux SO_PEERCRED')
    try:
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
        pid, uid, gid = struct.unpack('3i', raw)
    except (OSError, struct.error):
        raise ManagedClinicalRegistryError('managed registry server credentials are unavailable') from None
    if (uid, gid) != (0, 0):
        raise ManagedClinicalRegistryError('managed registry server is not root-owned')
    _require_linux_process_image(
        pid,
        expected_sha256=config.service_process_executable_sha256,
        label='service',
    )
    return ManagedClinicalPeerIdentity(
        pid=pid,
        uid=uid,
        gid=gid,
        canonical_launcher_id=config.canonical_launcher_id,
        canonical_launcher_executable_sha256=config.canonical_launcher_executable_sha256,
    )


def managed_clinical_registry_config_sha256(config: ManagedClinicalRegistryConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def authenticated_managed_clinical_registry_audit_sha256(
    artifact: AuthenticatedManagedClinicalRegistryAudit,
) -> str:
    canonical = AuthenticatedManagedClinicalRegistryAudit.model_validate_json(canonical_json_bytes(artifact))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def verify_authenticated_managed_clinical_registry_audit(
    artifact: AuthenticatedManagedClinicalRegistryAudit,
    *,
    key: bytes,
    expected_key_id: str,
    expected_config_sha256: str,
    expected_sequence: int,
    expected_predecessor_sha256: str,
) -> AuthenticatedManagedClinicalRegistryAudit:
    canonical = AuthenticatedManagedClinicalRegistryAudit.model_validate_json(canonical_json_bytes(artifact))
    if (
        len(key) < 32
        or canonical.audit_key_id != expected_key_id
        or managed_clinical_cleanup_key_id(key) != expected_key_id
        or canonical.registry_config_sha256 != expected_config_sha256
        or canonical.sequence != expected_sequence
        or canonical.predecessor_audit_sha256 != expected_predecessor_sha256
    ):
        raise ManagedClinicalRegistryError('managed registry protocol audit differs from deployment pins')
    expected_hmac = managed_clinical_registry_audit_hmac(canonical, key=key)
    if not hmac.compare_digest(expected_hmac, canonical.audit_hmac_sha256):
        raise ManagedClinicalRegistryError('managed registry protocol audit authentication failed')
    return canonical


def managed_clinical_registry_audit_hmac(
    artifact: AuthenticatedManagedClinicalRegistryAudit,
    *,
    key: bytes,
) -> str:
    if len(key) < 32:
        raise ValueError('managed registry protocol audit key must contain at least 32 bytes')
    canonical = AuthenticatedManagedClinicalRegistryAudit.model_validate_json(canonical_json_bytes(artifact))
    unsigned = canonical.model_copy(update={'audit_hmac_sha256': _AUDIT_ZERO_SHA256})
    return hmac.new(
        key,
        _AUDIT_HMAC_DOMAIN + canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()


def load_authenticated_managed_clinical_registry_audit(
    path: Path,
    *,
    expected_root: Path,
    required_uid: int = 0,
) -> AuthenticatedManagedClinicalRegistryAudit:
    try:
        root = expected_root.resolve(strict=True)
        supplied = path.resolve(strict=True)
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit is unavailable') from None
    if supplied.parent != root or supplied.name != path.name:
        raise ManagedClinicalRegistryError('managed registry protocol audit escaped its configured root')
    metadata = supplied.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != required_uid
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= _MAX_AUDIT_BYTES
    ):
        raise ManagedClinicalRegistryError('managed registry protocol audit has unsafe metadata')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(supplied, flags)
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit could not be opened safely') from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ManagedClinicalRegistryError('managed registry protocol audit changed while opening')
        content = bytearray()
        while len(content) <= _MAX_AUDIT_BYTES:
            block = os.read(
                descriptor,
                min(1024 * 1024, _MAX_AUDIT_BYTES + 1 - len(content)),
            )
            if not block:
                break
            content.extend(block)
        opened_after = os.fstat(descriptor)
        stable_fields = (
            'st_dev',
            'st_ino',
            'st_mode',
            'st_uid',
            'st_gid',
            'st_nlink',
            'st_size',
            'st_mtime_ns',
            'st_ctime_ns',
        )
        if any(getattr(opened, field) != getattr(opened_after, field) for field in stable_fields):
            raise ManagedClinicalRegistryError('managed registry protocol audit changed while reading')
    finally:
        os.close(descriptor)
    try:
        metadata_after = supplied.lstat()
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit disappeared after reading') from None
    if any(getattr(opened_after, field) != getattr(metadata_after, field) for field in stable_fields):
        raise ManagedClinicalRegistryError('managed registry protocol audit changed after reading')
    try:
        artifact = AuthenticatedManagedClinicalRegistryAudit.model_validate_json(bytes(content))
    except ValueError:
        raise ManagedClinicalRegistryError('managed registry protocol audit has an invalid strict schema') from None
    if canonical_json_bytes(artifact) != bytes(content) or artifact.persisted_path != str(supplied):
        raise ManagedClinicalRegistryError('managed registry protocol audit is non-canonical or names another path')
    return artifact


def load_authenticated_managed_registry_audit_chain(
    root: Path,
    *,
    key: bytes,
    expected_key_id: str,
    expected_config_sha256: str,
    required_uid: int = 0,
) -> tuple[AuthenticatedManagedClinicalRegistryAudit, ...]:
    resolved = _prepare_private_audit_root(root, required_uid=required_uid)
    entries, inventory_aggregate_bytes = _bounded_protocol_audit_inventory(
        resolved,
        required_uid=required_uid,
        allow_staging=False,
    )
    chain: list[AuthenticatedManagedClinicalRegistryAudit] = []
    loaded_aggregate_bytes = 0
    predecessor = _AUDIT_ZERO_SHA256
    for sequence, path in enumerate(entries):
        expected_prefix = f'{sequence:020d}-'
        if (
            not path.name.startswith(expected_prefix)
            or not path.name.endswith('.json')
            or len(path.name) != len(expected_prefix) + 32 + 5
        ):
            raise ManagedClinicalRegistryError('managed registry protocol audit inventory has a gap or unexpected file')
        artifact = load_authenticated_managed_clinical_registry_audit(
            path,
            expected_root=resolved,
            required_uid=required_uid,
        )
        loaded_aggregate_bytes += len(canonical_json_bytes(artifact))
        if loaded_aggregate_bytes > MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES:
            raise ManagedClinicalRegistryError('managed registry protocol audit exceeds its aggregate-byte limit')
        verify_authenticated_managed_clinical_registry_audit(
            artifact,
            key=key,
            expected_key_id=expected_key_id,
            expected_config_sha256=expected_config_sha256,
            expected_sequence=sequence,
            expected_predecessor_sha256=predecessor,
        )
        if path.name != f'{sequence:020d}-{artifact.request.request_id}.json':
            raise ManagedClinicalRegistryError(
                'managed registry protocol audit filename differs from its request identity'
            )
        chain.append(artifact)
        predecessor = authenticated_managed_clinical_registry_audit_sha256(artifact)
    if loaded_aggregate_bytes != inventory_aggregate_bytes:
        raise ManagedClinicalRegistryError('managed registry protocol audit changed during bounded reload')
    return tuple(chain)


def _managed_attempt_inventory_sha256(
    values: tuple[ManagedClinicalAttemptInventoryRecord, ...],
) -> str:
    return hashlib.sha256(canonical_json_bytes([item.model_dump(mode='json') for item in values])).hexdigest()


def _require_linux_process_image(
    pid: int,
    *,
    expected_sha256: str,
    label: str,
    proc_root: Path = Path('/proc'),
) -> None:
    if pid <= 0:
        raise ManagedClinicalRegistryError(f'managed registry {label} PID is invalid')
    path = proc_root / str(pid) / 'exe'
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ManagedClinicalRegistryError(f'managed registry {label} process image is unavailable') from None
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManagedClinicalRegistryError(f'managed registry {label} process image is not a regular file')
        while True:
            block = os.read(descriptor, _COPY_BUFFER_BYTES)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
        raise ManagedClinicalRegistryError(f'managed registry {label} process image differs from its deployment pin')


def _require_managed_socket_metadata(
    path: Path,
    *,
    config: ManagedClinicalRegistryConfig,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError:
        raise ManagedClinicalRegistryError('managed registry socket is unavailable') from None
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != config.allowed_launcher_uid
        or metadata.st_gid != config.allowed_launcher_gid
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise ManagedClinicalRegistryError(
            'managed registry socket identity or ownership differs from its deployment pin'
        )
    return identity


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ManagedClinicalRegistryError(
            'managed registry parent directory could not be opened for durability'
        ) from None
    try:
        os.fsync(descriptor)
    except OSError:
        raise ManagedClinicalRegistryError(
            'managed registry parent directory could not be durably synchronized'
        ) from None
    finally:
        os.close(descriptor)


def _prepare_private_audit_root(path: Path, *, required_uid: int = 0) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_absolute():
        raise ManagedClinicalRegistryError('managed registry protocol audit root must be an absolute non-symlink path')
    created = False
    try:
        supplied.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit root could not be created') from None
    try:
        resolved = supplied.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit root is unavailable') from None
    if (
        resolved != supplied
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != required_uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ManagedClinicalRegistryError('managed registry protocol audit root must be root-owned mode-0700')
    if created:
        _fsync_directory(resolved.parent)
    return resolved


def _bounded_protocol_audit_inventory(
    root: Path,
    *,
    required_uid: int,
    allow_staging: bool,
) -> tuple[tuple[Path, ...], int]:
    """Inventory a private audit root without first materializing untrusted directory size."""

    published: list[Path] = []
    published_bytes = 0
    staging: list[Path] = []
    try:
        with os.scandir(root) as iterator:
            for entry in iterator:
                name = entry.name
                is_published = _AUDIT_FILENAME.fullmatch(name) is not None
                is_staging = _AUDIT_STAGING_NAME.fullmatch(name) is not None
                if not is_published and not (allow_staging and is_staging):
                    raise ManagedClinicalRegistryError(
                        'managed registry protocol audit inventory has an unexpected file'
                    )
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    raise ManagedClinicalRegistryError(
                        'managed registry protocol audit inventory metadata is unavailable'
                    ) from None
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != required_uid
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise ManagedClinicalRegistryError('managed registry protocol audit inventory has unsafe metadata')
                path = root / name
                if is_published:
                    if not 0 < metadata.st_size <= _MAX_AUDIT_BYTES:
                        raise ManagedClinicalRegistryError(
                            'managed registry protocol audit inventory has unsafe metadata'
                        )
                    published_bytes += metadata.st_size
                    if len(published) >= MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES:
                        raise ManagedClinicalRegistryError(
                            'managed registry protocol audit exceeds its entry-count limit'
                        )
                    if published_bytes > MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES:
                        raise ManagedClinicalRegistryError(
                            'managed registry protocol audit exceeds its aggregate-byte limit'
                        )
                    published.append(path)
                    continue
                if not 0 <= metadata.st_size <= _MAX_AUDIT_BYTES:
                    raise ManagedClinicalRegistryError(
                        'managed registry protocol audit staging file has unsafe metadata'
                    )
                if len(staging) >= _MAX_PROTOCOL_AUDIT_STAGING_ENTRIES:
                    raise ManagedClinicalRegistryError(
                        'managed registry protocol audit has too many incomplete staging files'
                    )
                staging.append(path)
    except ManagedClinicalRegistryError:
        raise
    except OSError:
        raise ManagedClinicalRegistryError('managed registry protocol audit inventory is unavailable') from None
    return tuple(sorted((*published, *staging), key=lambda item: item.name)), published_bytes


def _write_create_once_audit(path: Path, payload: bytes) -> None:
    if not 0 < len(payload) <= _MAX_AUDIT_BYTES:
        raise ManagedClinicalRegistryError('managed registry protocol audit has invalid size')
    required_uid = os.geteuid()
    root = _prepare_private_audit_root(path.parent, required_uid=required_uid)
    if path.parent != root or _AUDIT_FILENAME.fullmatch(path.name) is None:
        raise ManagedClinicalRegistryError('managed registry protocol audit output escaped its configured root')
    entries, aggregate_bytes = _bounded_protocol_audit_inventory(
        root,
        required_uid=required_uid,
        allow_staging=False,
    )
    if len(entries) >= MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES:
        raise ManagedClinicalRegistryError('managed registry protocol audit entry-count limit is exhausted')
    if aggregate_bytes + len(payload) > MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES:
        raise ManagedClinicalRegistryError('managed registry protocol audit aggregate-byte limit is exhausted')
    staging = root / f'.audit-stage-{secrets.token_hex(32)}'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor: int | None = None
    created = False
    staging_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(staging, flags, 0o600)
        created = True
        opened = os.fstat(descriptor)
        staging_identity = (opened.st_dev, opened.st_ino)
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(errno.EIO, 'protocol-audit staging write made no progress')
            offset += written
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_uid != required_uid
            or final.st_size != len(payload)
            or (final.st_dev, final.st_ino) != staging_identity
        ):
            raise ManagedClinicalRegistryError('managed registry protocol audit staging file changed while writing')
        os.close(descriptor)
        descriptor = None
        rename_file_noreplace(staging, path)
        created = False
    except (AtomicDirectoryPublicationError, ManagedClinicalRegistryError, OSError):
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created and staging_identity is not None:
            try:
                current = staging.lstat()
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == staging_identity:
                    staging.unlink()
                    _fsync_directory(root)
            except OSError:
                pass
        raise ManagedClinicalRegistryError(
            'managed registry protocol audit could not be persisted create-once'
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_directory(root)


def _reap_incomplete_protocol_audit_staging(root: Path) -> None:
    """Remove only unpublished operation-owned audit stages after process restart."""

    required_uid = os.geteuid()
    resolved = _prepare_private_audit_root(root, required_uid=required_uid)
    entries, _aggregate_bytes = _bounded_protocol_audit_inventory(
        resolved,
        required_uid=required_uid,
        allow_staging=True,
    )
    removed = False
    for path in entries:
        if _AUDIT_FILENAME.fullmatch(path.name) is not None:
            continue
        try:
            before = path.lstat()
        except OSError:
            raise ManagedClinicalRegistryError('managed registry protocol audit staging file is unavailable') from None
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != required_uid
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _MAX_AUDIT_BYTES
        ):
            raise ManagedClinicalRegistryError('managed registry protocol audit staging file has unsafe metadata')
        try:
            after = path.lstat()
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise ManagedClinicalRegistryError(
                    'managed registry protocol audit staging file changed before cleanup'
                )
            path.unlink()
        except OSError:
            raise ManagedClinicalRegistryError('managed registry protocol audit staging cleanup failed') from None
        removed = True
    if removed:
        _fsync_directory(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError:
        raise ManagedClinicalRegistryError('managed registry process image could not be hashed') from None
    return digest.hexdigest()


def _linux_process_start_time_ticks(pid: int) -> int:
    try:
        content = (Path('/proc') / str(pid) / 'stat').read_text(encoding='ascii')
        close = content.rfind(')')
        if close <= 0:
            raise ValueError
        fields_after_command = content[close + 2 :].split()
        value = int(fields_after_command[19])
    except (OSError, ValueError, IndexError):
        raise ManagedClinicalRegistryError('managed registry process start time is unavailable') from None
    if value <= 0:
        raise ManagedClinicalRegistryError('managed registry process start time is invalid')
    return value


def _reservation(value: object):
    from vaxreplay.agentic.clinical_production_registry import ClinicalProductionReservation

    return ClinicalProductionReservation.model_validate_json(canonical_json_bytes(value))


def _send_frame(connection: socket.socket, payload: bytes, maximum: int) -> None:
    if not payload or len(payload) > maximum:
        raise ManagedClinicalRegistryError('managed registry frame has invalid size')
    connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)


def _recv_frame(
    connection: socket.socket,
    maximum: int,
    *,
    deadline_monotonic: float | None = None,
) -> bytes:
    header = _recv_exact(
        connection,
        _FRAME_HEADER.size,
        deadline_monotonic=deadline_monotonic,
    )
    (size,) = _FRAME_HEADER.unpack(header)
    if size <= 0 or size > maximum:
        raise ManagedClinicalRegistryError('managed registry frame has invalid size')
    return _recv_exact(connection, size, deadline_monotonic=deadline_monotonic)


def _recv_exact(
    connection: socket.socket,
    size: int,
    *,
    deadline_monotonic: float | None,
) -> bytes:
    content = bytearray()
    while len(content) < size:
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise ManagedClinicalRegistryError('managed registry connection exceeded its total deadline')
            connection.settimeout(remaining)
        chunk = connection.recv(size - len(content))
        if not chunk:
            raise ManagedClinicalRegistryError('managed registry frame ended early')
        content.extend(chunk)
    return bytes(content)


def _aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{label} must include a UTC offset')
    return value.astimezone(UTC)


__all__ = [
    'AuthenticatedManagedClinicalRegistryAudit',
    'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_AGGREGATE_BYTES',
    'MAX_MANAGED_CLINICAL_PROTOCOL_AUDIT_ENTRIES',
    'ManagedBeginReconciliationRequest',
    'ManagedClaimRequest',
    'ManagedClinicalPeerIdentity',
    'ManagedClinicalRegistryAuditServerIdentity',
    'ManagedClinicalRegistryClient',
    'ManagedClinicalRegistryConfig',
    'ManagedClinicalRegistryError',
    'ManagedClinicalRegistryRequest',
    'ManagedClinicalRegistryResponse',
    'ManagedClinicalRegistryService',
    'ManagedFinishReconciliationRequest',
    'ManagedRecordFailureRequest',
    'ManagedRecordRunRequest',
    'ManagedRedeemRequest',
    'ManagedReserveRequest',
    'ManagedStatusRequest',
    'ManagedWorkspaceReference',
    'authenticate_managed_registry_peer',
    'authenticate_managed_registry_server',
    'authenticated_managed_clinical_registry_audit_sha256',
    'load_authenticated_managed_clinical_registry_audit',
    'load_authenticated_managed_registry_audit_chain',
    'managed_clinical_registry_audit_hmac',
    'managed_clinical_registry_config_sha256',
    'verify_authenticated_managed_clinical_registry_audit',
]
