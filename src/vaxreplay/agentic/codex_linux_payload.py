"""Deterministic, content-addressed Linux Codex harness payload packaging.

The package is an input to a future harness-disk build.  It binds one exact Codex executable,
one exact VaxReplay adapter executable, their declared support-file closure, the generic headless
adapter config, and an out-of-payload base-runtime manifest.  Package verification is offline and
requires independent expected hashes; it never treats a self-described vendor name as provenance.

The implementation intentionally does not claim that the declared dependency inventory is
complete, that an official Codex Linux release was obtained, that a disk was built or booted, or
that Linux/KVM qualification passed.  A testing-only opaque-entrypoint mode exercises the package
machinery without manufacturing those claims.  The non-test path performs bounded ELF64 structural
inspection and rejects an interpreter-bearing entrypoint, but even that is not release provenance
or execution evidence.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay._atomic import AtomicDirectoryPublication, AtomicDirectoryPublicationError
from vaxreplay.agentic.codex_guest_adapter import (
    CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
    CODEX_VENDOR_EXECUTABLE_PATH,
    codex_vendor_argv_template,
)
from vaxreplay.agentic.headless_guest_adapter import (
    HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
    HeadlessGuestAdapterConfig,
    headless_guest_adapter_config_sha256,
)
from vaxreplay.agentic.submitted_harness import HarnessFamily
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

CODEX_LINUX_DEPENDENCY_CLOSURE_SCHEMA_VERSION = 'vaxreplay.codex-linux-dependency-closure.dev-v0.1'
CODEX_LINUX_PAYLOAD_RECEIPT_SCHEMA_VERSION = 'vaxreplay.codex-linux-payload-receipt.dev-v0.1'
CODEX_LINUX_PAYLOAD_PROFILE = 'codex_cli_0.144.3_linux_guest_payload_v0.1'
CODEX_LINUX_PAYLOAD_RECEIPT_FILE = 'payload-receipt.json'
CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY = 'chunks'
CODEX_LINUX_PAYLOAD_CHUNK_BYTES = 4 * 1024 * 1024
CODEX_LINUX_PAYLOAD_CLOSURE_TARGET = '/opt/vaxreplay/etc/codex-dependency-closure.json'
CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET = '/opt/vaxreplay/etc/headless-guest-adapter.json'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_CLOSURE_FILES = 4_096
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_LOGICAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_UNIQUE_CHUNKS = 131_072
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_ELF64_HEADER = struct.Struct('<16sHHIQQQIHHHHHH')
_ELF64_PROGRAM_HEADER = struct.Struct('<IIQQQQQQ')
_ELF_MACHINE = {'x86_64': 62, 'aarch64': 183}
_PT_LOAD = 1
_PT_INTERP = 3


class CodexLinuxPayloadError(RuntimeError):
    """The payload source, package, external pin, or offline verification failed closed."""


class CodexLinuxPayloadRole(str, enum.Enum):
    VENDOR_EXECUTABLE = 'vendor_executable'
    ADAPTER_EXECUTABLE = 'adapter_executable'
    VENDOR_SUPPORT_FILE = 'vendor_support_file'
    ADAPTER_SUPPORT_FILE = 'adapter_support_file'
    DEPENDENCY_CLOSURE_MANIFEST = 'dependency_closure_manifest'
    HEADLESS_ADAPTER_CONFIG = 'headless_adapter_config'


class CodexLinuxEntrypointFormat(str, enum.Enum):
    ELF64_LITTLE_ENDIAN_NO_INTERPRETER = 'elf64_little_endian_no_interpreter'
    TEST_OPAQUE = 'test_opaque'


class CodexLinuxPackagingBoundary(str, enum.Enum):
    OFFLINE_ELF64_INSPECTION = 'offline_elf64_inspection'
    TEST_FIXTURE = 'test_fixture'


class CodexLinuxClosureEntry(StrictModel):
    """One exact file in the externally asserted payload dependency inventory."""

    target_path: str = Field(min_length=2, max_length=4_096)
    role: Literal[
        CodexLinuxPayloadRole.VENDOR_EXECUTABLE,
        CodexLinuxPayloadRole.ADAPTER_EXECUTABLE,
        CodexLinuxPayloadRole.VENDOR_SUPPORT_FILE,
        CodexLinuxPayloadRole.ADAPTER_SUPPORT_FILE,
    ]
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=_MAX_FILE_BYTES)
    mode: Literal[292, 365]

    @field_validator('target_path')
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        _require_payload_target(value)
        if value in {
            CODEX_LINUX_PAYLOAD_CLOSURE_TARGET,
            CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET,
        }:
            raise ValueError('generated payload metadata paths cannot be dependency entries')
        return value

    @model_validator(mode='after')
    def validate_role_path_and_mode(self) -> Self:
        if self.role == CodexLinuxPayloadRole.VENDOR_EXECUTABLE:
            if self.target_path != CODEX_VENDOR_EXECUTABLE_PATH or self.mode != 0o555:
                raise ValueError('the vendor executable entry must use its fixed path and mode 0555')
        elif self.role == CodexLinuxPayloadRole.ADAPTER_EXECUTABLE:
            if self.target_path != HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH or self.mode != 0o555:
                raise ValueError('the adapter executable entry must use its fixed path and mode 0555')
        elif self.target_path in {
            CODEX_VENDOR_EXECUTABLE_PATH,
            HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        }:
            raise ValueError('a support-file role cannot occupy a primary executable path')
        return self


class CodexLinuxDependencyClosureManifest(StrictModel):
    """Externally pinned declaration of the exact Codex/adapter payload closure.

    The completeness field is an assertion supplied by the artifact producer.  The offline
    verifier proves that the package equals this inventory; it cannot prove that an omitted runtime
    dependency does not exist.
    """

    schema_version: Literal['vaxreplay.codex-linux-dependency-closure.dev-v0.1'] = (
        CODEX_LINUX_DEPENDENCY_CLOSURE_SCHEMA_VERSION
    )
    profile: Literal['codex_cli_0.144.3_linux_guest_payload_v0.1'] = CODEX_LINUX_PAYLOAD_PROFILE
    target_system: Literal['Linux'] = 'Linux'
    target_machine: Literal['x86_64', 'aarch64']
    codex_reported_version: Literal['codex-cli 0.144.3'] = CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION
    vendor_entrypoint_format: CodexLinuxEntrypointFormat
    adapter_entrypoint_format: CodexLinuxEntrypointFormat
    base_rootfs_runtime_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    entries: tuple[CodexLinuxClosureEntry, ...] = Field(
        min_length=2,
        max_length=_MAX_CLOSURE_FILES,
    )
    complete_transitive_user_space_dependency_inventory_attested: Literal[True] = True
    base_runtime_is_supplied_outside_payload_and_digest_pinned: Literal[True] = True
    primary_entrypoints_have_no_elf_interpreter_attested: bool
    mutable_or_ambient_dependency_resolution_allowed: Literal[False] = False
    provider_credentials_are_dependency_entries: Literal[False] = False
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_inventory(self) -> Self:
        paths = tuple(item.target_path for item in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('closure entries must be unique and canonically sorted by target path')
        _reject_target_file_directory_collisions(paths)
        vendor = tuple(item for item in self.entries if item.role == CodexLinuxPayloadRole.VENDOR_EXECUTABLE)
        adapter = tuple(item for item in self.entries if item.role == CodexLinuxPayloadRole.ADAPTER_EXECUTABLE)
        if len(vendor) != 1 or len(adapter) != 1:
            raise ValueError('closure inventory requires exactly one vendor and one adapter executable')
        formats = (self.vendor_entrypoint_format, self.adapter_entrypoint_format)
        test_format = formats == (
            CodexLinuxEntrypointFormat.TEST_OPAQUE,
            CodexLinuxEntrypointFormat.TEST_OPAQUE,
        )
        elf_format = formats == (
            CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER,
            CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER,
        )
        if not (test_format or elf_format):
            raise ValueError('vendor and adapter entrypoints must use one common admitted format')
        if self.primary_entrypoints_have_no_elf_interpreter_attested != elf_format:
            raise ValueError('ELF-interpreter attestation differs from the declared entrypoint formats')
        return self


def codex_linux_dependency_closure_sha256(
    manifest: CodexLinuxDependencyClosureManifest,
) -> str:
    canonical = CodexLinuxDependencyClosureManifest.model_validate_json(canonical_json_bytes(manifest))
    return _sha256(canonical_json_bytes(canonical))


class CodexLinuxPayloadChunkBinding(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=CODEX_LINUX_PAYLOAD_CHUNK_BYTES)


class CodexLinuxPayloadFileBinding(StrictModel):
    """One future absolute target reconstructed from ordered content-addressed chunks."""

    target_path: str = Field(min_length=2, max_length=4_096)
    role: CodexLinuxPayloadRole
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=_MAX_FILE_BYTES)
    mode: Literal[256, 292, 365]
    chunks: tuple[CodexLinuxPayloadChunkBinding, ...] = Field(
        min_length=1,
        max_length=(_MAX_FILE_BYTES // CODEX_LINUX_PAYLOAD_CHUNK_BYTES) + 1,
    )

    @field_validator('target_path')
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        _require_payload_target(value)
        return value

    @model_validator(mode='after')
    def validate_chunks(self) -> Self:
        if sum(item.byte_count for item in self.chunks) != self.byte_count:
            raise ValueError('payload file chunks do not cover the exact file byte count')
        return self


class CodexLinuxPayloadReceipt(StrictModel):
    """Canonical package index and intentionally narrow evidence claims."""

    schema_version: Literal['vaxreplay.codex-linux-payload-receipt.dev-v0.1'] = (
        CODEX_LINUX_PAYLOAD_RECEIPT_SCHEMA_VERSION
    )
    profile: Literal['codex_cli_0.144.3_linux_guest_payload_v0.1'] = CODEX_LINUX_PAYLOAD_PROFILE
    packaging_boundary: CodexLinuxPackagingBoundary
    closure_manifest: CodexLinuxDependencyClosureManifest
    closure_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    headless_adapter_config: HeadlessGuestAdapterConfig
    headless_adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    files: tuple[CodexLinuxPayloadFileBinding, ...] = Field(
        min_length=4,
        max_length=_MAX_CLOSURE_FILES + 2,
    )
    file_count: int = Field(ge=4, le=_MAX_CLOSURE_FILES + 2)
    logical_payload_byte_count: int = Field(gt=0, le=_MAX_TOTAL_LOGICAL_BYTES)
    unique_chunk_count: int = Field(gt=0, le=_MAX_UNIQUE_CHUNKS)
    unique_chunk_byte_count: int = Field(gt=0, le=_MAX_TOTAL_LOGICAL_BYTES)
    payload_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    builder_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_files_read_as_stable_regular_nonsymlinks: Literal[True] = True
    exact_package_file_set_verified_before_and_after_publication: Literal[True] = True
    content_addressed_chunks_and_canonical_receipt_are_deterministic_identity: Literal[True] = True
    filesystem_timestamps_are_not_package_identity: Literal[True] = True
    linux_elf_structure_and_no_interpreter_verified: bool
    structurally_eligible_as_guest_disk_builder_input: bool
    dependency_inventory_completeness_was_producer_attested: Literal[True] = True
    dependency_inventory_completeness_independently_proven: Literal[False] = False
    official_codex_release_provenance_verified: Literal[False] = False
    credential_absence_proven_by_packaging: Literal[False] = False
    actual_linux_codex_executed: Literal[False] = False
    guest_disk_built: Literal[False] = False
    linux_kvm_qualified: Literal[False] = False
    official_leaderboard_admitted: Literal[False] = False
    development_only: Literal[True] = True

    @model_validator(mode='after')
    def validate_receipt(self) -> Self:
        closure_sha256 = codex_linux_dependency_closure_sha256(self.closure_manifest)
        headless_sha256 = headless_guest_adapter_config_sha256(self.headless_adapter_config)
        if self.closure_manifest_sha256 != closure_sha256:
            raise ValueError('payload receipt closure-manifest digest differs from its exact manifest')
        if self.headless_adapter_config_sha256 != headless_sha256:
            raise ValueError('payload receipt headless-config digest differs from its exact config')
        _require_headless_closure_binding(self.closure_manifest, self.headless_adapter_config)
        paths = tuple(item.target_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError('payload file bindings must be unique and canonically sorted')
        _reject_target_file_directory_collisions(paths)
        if self.file_count != len(self.files):
            raise ValueError('payload file_count differs from the complete file inventory')
        if self.logical_payload_byte_count != sum(item.byte_count for item in self.files):
            raise ValueError('payload logical byte count differs from its complete inventory')
        if self.payload_inventory_sha256 != _payload_inventory_sha256(self.files):
            raise ValueError('payload inventory digest differs from the exact file bindings')
        _require_receipt_file_inventory(self)
        unique_chunks: dict[str, int] = {}
        for file_binding in self.files:
            for chunk in file_binding.chunks:
                prior = unique_chunks.setdefault(chunk.sha256, chunk.byte_count)
                if prior != chunk.byte_count:
                    raise ValueError('one chunk digest is associated with conflicting byte counts')
        if self.unique_chunk_count != len(unique_chunks):
            raise ValueError('payload unique chunk count differs from its exact inventory')
        if self.unique_chunk_byte_count != sum(unique_chunks.values()):
            raise ValueError('payload unique chunk byte count differs from its exact inventory')
        real_boundary = self.packaging_boundary == CodexLinuxPackagingBoundary.OFFLINE_ELF64_INSPECTION
        manifest_uses_elf = (
            self.closure_manifest.vendor_entrypoint_format
            == CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER
            and self.closure_manifest.adapter_entrypoint_format
            == CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER
        )
        if (
            manifest_uses_elf != real_boundary
            or self.linux_elf_structure_and_no_interpreter_verified != real_boundary
            or self.structurally_eligible_as_guest_disk_builder_input != real_boundary
        ):
            raise ValueError('payload structural eligibility differs from its packaging boundary')
        return self


def codex_linux_payload_receipt_sha256(receipt: CodexLinuxPayloadReceipt) -> str:
    canonical = CodexLinuxPayloadReceipt.model_validate_json(canonical_json_bytes(receipt))
    return _sha256(canonical_json_bytes(canonical))


@dataclass(frozen=True)
class WrittenCodexLinuxPayload:
    root: Path
    receipt_path: Path
    receipt: CodexLinuxPayloadReceipt
    receipt_sha256: str


@dataclass(frozen=True)
class VerifiedCodexLinuxPayload:
    """Fresh offline verification result, not a serializable authorization capability."""

    root: Path
    receipt: CodexLinuxPayloadReceipt
    receipt_sha256: str


@dataclass(frozen=True)
class _MeasuredPayloadFile:
    binding: CodexLinuxPayloadFileBinding
    source_path: Path | None
    generated_bytes: bytes | None


def build_codex_linux_payload(
    source_files: Mapping[str, Path],
    output_dir: Path,
    *,
    closure_manifest: CodexLinuxDependencyClosureManifest,
    expected_closure_manifest_sha256: str,
    headless_adapter_config: HeadlessGuestAdapterConfig,
    expected_headless_adapter_config_sha256: str,
    testing_only: bool = False,
) -> WrittenCodexLinuxPayload:
    """Build and self-verify one create-once deterministic payload package.

    ``source_files`` must contain exactly the closure-manifest target paths.  The mapping and local
    source paths are deliberately absent from the receipt; only exact future target identities are
    package identity.  Non-test builds require both primary entrypoints to pass bounded ELF64
    little-endian/no-``PT_INTERP`` inspection for the declared target architecture.
    """

    try:
        closure = CodexLinuxDependencyClosureManifest.model_validate_json(canonical_json_bytes(closure_manifest))
        headless = HeadlessGuestAdapterConfig.model_validate_json(canonical_json_bytes(headless_adapter_config))
        _require_expected_sha256(expected_closure_manifest_sha256, 'closure manifest')
        _require_expected_sha256(expected_headless_adapter_config_sha256, 'headless adapter config')
        closure_sha256 = codex_linux_dependency_closure_sha256(closure)
        headless_sha256 = headless_guest_adapter_config_sha256(headless)
        if not hmac.compare_digest(closure_sha256, expected_closure_manifest_sha256):
            raise CodexLinuxPayloadError('closure manifest differs from its independent expected digest')
        if not hmac.compare_digest(headless_sha256, expected_headless_adapter_config_sha256):
            raise CodexLinuxPayloadError('headless adapter config differs from its independent expected digest')
        _require_headless_closure_binding(closure, headless)
        expected_paths = {item.target_path for item in closure.entries}
        if not isinstance(source_files, Mapping) or set(source_files) != expected_paths:
            raise CodexLinuxPayloadError('source mapping must equal the complete closure inventory')

        test_format = (
            closure.vendor_entrypoint_format == CodexLinuxEntrypointFormat.TEST_OPAQUE
            and closure.adapter_entrypoint_format == CodexLinuxEntrypointFormat.TEST_OPAQUE
        )
        if testing_only != test_format:
            raise CodexLinuxPayloadError('opaque fixture entrypoints require explicit testing-only packaging')
        boundary = (
            CodexLinuxPackagingBoundary.TEST_FIXTURE
            if testing_only
            else CodexLinuxPackagingBoundary.OFFLINE_ELF64_INSPECTION
        )

        measured: list[_MeasuredPayloadFile] = []
        for entry in closure.entries:
            source = source_files[entry.target_path]
            if not isinstance(source, Path):
                raise CodexLinuxPayloadError('payload source mappings must contain Path values')
            primary = entry.role in {
                CodexLinuxPayloadRole.VENDOR_EXECUTABLE,
                CodexLinuxPayloadRole.ADAPTER_EXECUTABLE,
            }
            measured.append(
                _measure_source_file(
                    entry,
                    source,
                    target_machine=closure.target_machine,
                    inspect_elf=primary and not testing_only,
                )
            )

        closure_bytes = canonical_json_bytes(closure)
        headless_bytes = canonical_json_bytes(headless)
        measured.extend(
            (
                _measure_generated_file(
                    CODEX_LINUX_PAYLOAD_CLOSURE_TARGET,
                    CodexLinuxPayloadRole.DEPENDENCY_CLOSURE_MANIFEST,
                    closure_bytes,
                ),
                _measure_generated_file(
                    CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET,
                    CodexLinuxPayloadRole.HEADLESS_ADAPTER_CONFIG,
                    headless_bytes,
                ),
            )
        )
        measured_files = tuple(sorted(measured, key=lambda item: item.binding.target_path))
        bindings = tuple(item.binding for item in measured_files)
        unique_chunks = _unique_chunks(bindings)
        receipt = CodexLinuxPayloadReceipt(
            packaging_boundary=boundary,
            closure_manifest=closure,
            closure_manifest_sha256=closure_sha256,
            headless_adapter_config=headless,
            headless_adapter_config_sha256=headless_sha256,
            files=bindings,
            file_count=len(bindings),
            logical_payload_byte_count=sum(item.byte_count for item in bindings),
            unique_chunk_count=len(unique_chunks),
            unique_chunk_byte_count=sum(unique_chunks.values()),
            payload_inventory_sha256=_payload_inventory_sha256(bindings),
            builder_source_sha256=_stable_file_sha256(Path(__file__)),
            build_contract_sha256=_build_contract_sha256(),
            linux_elf_structure_and_no_interpreter_verified=not testing_only,
            structurally_eligible_as_guest_disk_builder_input=not testing_only,
        )
        receipt_bytes = canonical_json_bytes(receipt)
        if len(receipt_bytes) > _MAX_RECEIPT_BYTES:
            raise CodexLinuxPayloadError('payload receipt exceeds its fixed byte limit')
        receipt_sha256 = _sha256(receipt_bytes)
    except CodexLinuxPayloadError:
        raise
    except (OSError, TypeError, ValueError):
        raise CodexLinuxPayloadError('payload inputs are invalid') from None

    try:
        with AtomicDirectoryPublication.create(output_dir) as publication:
            publication.make_directory(CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY, mode=0o555)
            written_chunks: set[str] = set()
            for item in measured_files:
                _publish_measured_file_chunks(publication, item, written_chunks)
            if written_chunks != set(unique_chunks):
                raise CodexLinuxPayloadError('published chunk set differs from the measured package')
            publication.write_bytes(CODEX_LINUX_PAYLOAD_RECEIPT_FILE, receipt_bytes, mode=0o444)
            target = publication.publish(root_mode=0o755)
            verified = verify_codex_linux_payload(
                target,
                expected_receipt_sha256=receipt_sha256,
                expected_closure_manifest_sha256=closure_sha256,
                expected_headless_adapter_config_sha256=headless_sha256,
            )
            if verified.receipt != receipt:
                raise CodexLinuxPayloadError('published package differs from its in-process build')
            publication.commit()
    except CodexLinuxPayloadError:
        raise
    except AtomicDirectoryPublicationError as error:
        raise CodexLinuxPayloadError(f'atomic payload publication failed: {error}') from error
    return WrittenCodexLinuxPayload(
        root=target,
        receipt_path=target / CODEX_LINUX_PAYLOAD_RECEIPT_FILE,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
    )


def _require_payload_target(value: str) -> None:
    path = PurePosixPath(value)
    if (
        '\x00' in value
        or not path.is_absolute()
        or '..' in path.parts
        or '.' in path.parts
        or path.as_posix() != value
        or not value.startswith('/opt/vaxreplay/')
    ):
        raise ValueError('payload target must be normalized beneath /opt/vaxreplay')


def _reject_target_file_directory_collisions(paths: tuple[str, ...]) -> None:
    path_set = set(paths)
    for value in paths:
        path = PurePosixPath(value)
        for parent in path.parents:
            if parent.as_posix() in path_set:
                raise ValueError('payload targets contain a file/directory collision')


def _require_headless_closure_binding(
    closure: CodexLinuxDependencyClosureManifest,
    headless: HeadlessGuestAdapterConfig,
) -> None:
    entries = {item.role: item for item in closure.entries}
    vendor = entries.get(CodexLinuxPayloadRole.VENDOR_EXECUTABLE)
    adapter = entries.get(CodexLinuxPayloadRole.ADAPTER_EXECUTABLE)
    closure_sha256 = codex_linux_dependency_closure_sha256(closure)
    expected = (
        HarnessFamily.CODEX,
        True,
        True,
        True,
        CODEX_VENDOR_EXECUTABLE_PATH,
        vendor.sha256 if vendor is not None else None,
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        adapter.sha256 if adapter is not None else None,
        CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
        codex_vendor_argv_template(),
        closure_sha256,
    )
    actual = (
        headless.family,
        headless.adapter_implementation_checked_in,
        headless.provider_shim_implementation_checked_in,
        headless.workspace_materialization_bridge_implementation_checked_in,
        headless.vendor_executable_path,
        headless.vendor_executable_sha256,
        headless.adapter_executable_path,
        headless.adapter_executable_sha256,
        headless.vendor_reported_version,
        headless.vendor_argv_template,
        headless.complete_dependency_closure_sha256,
    )
    if actual != expected:
        raise ValueError('headless adapter config differs from the exact Linux closure manifest')


def _require_receipt_file_inventory(receipt: CodexLinuxPayloadReceipt) -> None:
    by_path = {item.target_path: item for item in receipt.files}
    expected_paths = {item.target_path for item in receipt.closure_manifest.entries} | {
        CODEX_LINUX_PAYLOAD_CLOSURE_TARGET,
        CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET,
    }
    if set(by_path) != expected_paths:
        raise ValueError('payload file inventory differs from the closure plus generated metadata')
    for entry in receipt.closure_manifest.entries:
        binding = by_path[entry.target_path]
        if (
            binding.role != entry.role
            or binding.sha256 != entry.sha256
            or binding.byte_count != entry.byte_count
            or binding.mode != entry.mode
        ):
            raise ValueError('payload file binding differs from its closure entry')
    generated = (
        (
            CODEX_LINUX_PAYLOAD_CLOSURE_TARGET,
            CodexLinuxPayloadRole.DEPENDENCY_CLOSURE_MANIFEST,
            canonical_json_bytes(receipt.closure_manifest),
        ),
        (
            CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET,
            CodexLinuxPayloadRole.HEADLESS_ADAPTER_CONFIG,
            canonical_json_bytes(receipt.headless_adapter_config),
        ),
    )
    for target, role, body in generated:
        binding = by_path[target]
        if (
            binding.role != role
            or binding.sha256 != _sha256(body)
            or binding.byte_count != len(body)
            or binding.mode != 0o400
        ):
            raise ValueError('generated payload metadata binding differs from its canonical bytes')


def _payload_inventory_sha256(files: tuple[CodexLinuxPayloadFileBinding, ...]) -> str:
    return _sha256(canonical_json_bytes([item.model_dump(mode='json') for item in files]))


def _unique_chunks(files: tuple[CodexLinuxPayloadFileBinding, ...]) -> dict[str, int]:
    unique: dict[str, int] = {}
    for file_binding in files:
        for chunk in file_binding.chunks:
            prior = unique.setdefault(chunk.sha256, chunk.byte_count)
            if prior != chunk.byte_count:
                raise CodexLinuxPayloadError('one chunk digest has conflicting byte counts')
    if not 0 < len(unique) <= _MAX_UNIQUE_CHUNKS:
        raise CodexLinuxPayloadError('payload unique chunk count exceeds its fixed limit')
    return unique


def _build_contract_sha256() -> str:
    contract = {
        'schema_version': 'vaxreplay.codex-linux-payload-build-contract.dev-v0.1',
        'profile': CODEX_LINUX_PAYLOAD_PROFILE,
        'chunk_algorithm': 'fixed-size-sha256-v0.1',
        'chunk_bytes': CODEX_LINUX_PAYLOAD_CHUNK_BYTES,
        'receipt_file': CODEX_LINUX_PAYLOAD_RECEIPT_FILE,
        'chunk_directory': CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY,
        'closure_target': CODEX_LINUX_PAYLOAD_CLOSURE_TARGET,
        'headless_config_target': CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET,
        'source_types': 'stable-owned-regular-single-link-nonsymlink-only',
        'publication': 'descriptor-relative-atomic-create-once-directory',
        'package_root_mode': 0o755,
        'chunk_directory_mode': 0o555,
        'filesystem_timestamps_are_identity': False,
    }
    return _sha256(canonical_json_bytes(contract))


def _measure_source_file(
    entry: CodexLinuxClosureEntry,
    source_path: Path,
    *,
    target_machine: Literal['x86_64', 'aarch64'],
    inspect_elf: bool,
) -> _MeasuredPayloadFile:
    if not source_path.is_absolute():
        raise CodexLinuxPayloadError('payload source paths must be absolute')
    descriptor = _open_stable_source(source_path)
    try:
        before = os.fstat(descriptor)
        _require_source_metadata(before)
        if before.st_size != entry.byte_count:
            raise CodexLinuxPayloadError('payload source byte count differs from its closure entry')
        if inspect_elf:
            _inspect_elf64_no_interpreter(
                descriptor,
                byte_count=before.st_size,
                target_machine=target_machine,
            )
        digest = hashlib.sha256()
        chunks: list[CodexLinuxPayloadChunkBinding] = []
        while True:
            chunk = _read_up_to(descriptor, CODEX_LINUX_PAYLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(
                CodexLinuxPayloadChunkBinding(
                    sha256=_sha256(chunk),
                    byte_count=len(chunk),
                )
            )
        after = os.fstat(descriptor)
        _require_unchanged_metadata(before, after, label='payload source')
    finally:
        os.close(descriptor)
    if not chunks or not hmac.compare_digest(digest.hexdigest(), entry.sha256):
        raise CodexLinuxPayloadError('payload source bytes differ from their closure entry')
    binding = CodexLinuxPayloadFileBinding(
        target_path=entry.target_path,
        role=entry.role,
        sha256=entry.sha256,
        byte_count=entry.byte_count,
        mode=entry.mode,
        chunks=tuple(chunks),
    )
    return _MeasuredPayloadFile(
        binding=binding,
        source_path=source_path,
        generated_bytes=None,
    )


def _measure_generated_file(
    target_path: str,
    role: Literal[
        CodexLinuxPayloadRole.DEPENDENCY_CLOSURE_MANIFEST,
        CodexLinuxPayloadRole.HEADLESS_ADAPTER_CONFIG,
    ],
    body: bytes,
) -> _MeasuredPayloadFile:
    if not body or len(body) > _MAX_FILE_BYTES:
        raise CodexLinuxPayloadError('generated payload metadata is empty or oversized')
    chunks = tuple(
        CodexLinuxPayloadChunkBinding(
            sha256=_sha256(body[offset : offset + CODEX_LINUX_PAYLOAD_CHUNK_BYTES]),
            byte_count=len(body[offset : offset + CODEX_LINUX_PAYLOAD_CHUNK_BYTES]),
        )
        for offset in range(0, len(body), CODEX_LINUX_PAYLOAD_CHUNK_BYTES)
    )
    return _MeasuredPayloadFile(
        binding=CodexLinuxPayloadFileBinding(
            target_path=target_path,
            role=role,
            sha256=_sha256(body),
            byte_count=len(body),
            mode=0o400,
            chunks=chunks,
        ),
        source_path=None,
        generated_bytes=body,
    )


def _publish_measured_file_chunks(
    publication: AtomicDirectoryPublication,
    measured: _MeasuredPayloadFile,
    written_chunks: set[str],
) -> None:
    if measured.source_path is not None:
        _publish_source_chunks(publication, measured, written_chunks)
        return
    body = measured.generated_bytes
    if body is None:
        raise CodexLinuxPayloadError('measured payload file has no source bytes')
    observed: list[CodexLinuxPayloadChunkBinding] = []
    for offset in range(0, len(body), CODEX_LINUX_PAYLOAD_CHUNK_BYTES):
        chunk = body[offset : offset + CODEX_LINUX_PAYLOAD_CHUNK_BYTES]
        binding = CodexLinuxPayloadChunkBinding(sha256=_sha256(chunk), byte_count=len(chunk))
        observed.append(binding)
        _publish_chunk_once(publication, chunk, binding, written_chunks)
    if tuple(observed) != measured.binding.chunks or _sha256(body) != measured.binding.sha256:
        raise CodexLinuxPayloadError('generated payload bytes changed after measurement')


def _publish_source_chunks(
    publication: AtomicDirectoryPublication,
    measured: _MeasuredPayloadFile,
    written_chunks: set[str],
) -> None:
    source_path = measured.source_path
    if source_path is None:
        raise CodexLinuxPayloadError('source-backed payload file has no source path')
    descriptor = _open_stable_source(source_path)
    digest = hashlib.sha256()
    observed: list[CodexLinuxPayloadChunkBinding] = []
    try:
        before = os.fstat(descriptor)
        _require_source_metadata(before)
        for expected in measured.binding.chunks:
            chunk = _read_up_to(descriptor, CODEX_LINUX_PAYLOAD_CHUNK_BYTES)
            if not chunk:
                raise CodexLinuxPayloadError('payload source ended before its measured chunk inventory')
            binding = CodexLinuxPayloadChunkBinding(sha256=_sha256(chunk), byte_count=len(chunk))
            if binding != expected:
                raise CodexLinuxPayloadError('payload source chunk changed after measurement')
            digest.update(chunk)
            observed.append(binding)
            _publish_chunk_once(publication, chunk, binding, written_chunks)
        if _read_up_to(descriptor, 1):
            raise CodexLinuxPayloadError('payload source grew after measurement')
        after = os.fstat(descriptor)
        _require_unchanged_metadata(before, after, label='payload source publication')
    finally:
        os.close(descriptor)
    if (
        tuple(observed) != measured.binding.chunks
        or not hmac.compare_digest(digest.hexdigest(), measured.binding.sha256)
        or sum(item.byte_count for item in observed) != measured.binding.byte_count
    ):
        raise CodexLinuxPayloadError('published source chunks differ from their measured file')


def _publish_chunk_once(
    publication: AtomicDirectoryPublication,
    chunk: bytes,
    binding: CodexLinuxPayloadChunkBinding,
    written_chunks: set[str],
) -> None:
    if binding.sha256 in written_chunks:
        return
    publication.write_bytes(
        f'{CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY}/{binding.sha256}',
        chunk,
        mode=0o444,
    )
    written_chunks.add(binding.sha256)


def _open_stable_source(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexLinuxPayloadError('O_NOFOLLOW is required for payload packaging')
    try:
        return os.open(path, flags | no_follow)
    except OSError:
        raise CodexLinuxPayloadError('payload source cannot be opened safely') from None


def _require_source_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid not in {0, os.geteuid()}
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not 0 < metadata.st_size <= _MAX_FILE_BYTES
    ):
        raise CodexLinuxPayloadError('payload source is not an admitted stable regular file')


def _require_unchanged_metadata(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise CodexLinuxPayloadError(f'{label} changed while it was read')


def _read_up_to(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining:
        try:
            chunk = os.read(descriptor, remaining)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _inspect_elf64_no_interpreter(
    descriptor: int,
    *,
    byte_count: int,
    target_machine: Literal['x86_64', 'aarch64'],
) -> None:
    header = _pread_exact(descriptor, _ELF64_HEADER.size, 0)
    if len(header) != _ELF64_HEADER.size:
        raise CodexLinuxPayloadError('Linux entrypoint has a truncated ELF64 header')
    (
        identifier,
        elf_type,
        machine,
        version,
        _entry,
        program_offset,
        _section_offset,
        _flags,
        header_size,
        program_entry_size,
        program_count,
        _section_entry_size,
        _section_count,
        _section_name_index,
    ) = _ELF64_HEADER.unpack(header)
    if (
        identifier[:7] != b'\x7fELF\x02\x01\x01'
        or identifier[7] not in {0, 3}
        or elf_type not in {2, 3}
        or machine != _ELF_MACHINE[target_machine]
        or version != 1
        or header_size != _ELF64_HEADER.size
        or program_entry_size < _ELF64_PROGRAM_HEADER.size
        or not 0 < program_count <= 4_096
        or program_offset < _ELF64_HEADER.size
        or program_offset + program_entry_size * program_count > byte_count
    ):
        raise CodexLinuxPayloadError('Linux entrypoint ELF64 identity is unsupported')
    load_count = 0
    for index in range(program_count):
        offset = program_offset + index * program_entry_size
        raw = _pread_exact(descriptor, _ELF64_PROGRAM_HEADER.size, offset)
        if len(raw) != _ELF64_PROGRAM_HEADER.size:
            raise CodexLinuxPayloadError('Linux entrypoint has a truncated program header')
        program_type = _ELF64_PROGRAM_HEADER.unpack(raw)[0]
        if program_type == _PT_INTERP:
            raise CodexLinuxPayloadError('Linux payload entrypoints may not depend on an ELF interpreter')
        if program_type == _PT_LOAD:
            load_count += 1
    if load_count == 0:
        raise CodexLinuxPayloadError('Linux entrypoint has no loadable ELF segment')


def _pread_exact(descriptor: int, byte_count: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    cursor = offset
    remaining = byte_count
    while remaining:
        try:
            chunk = os.pread(descriptor, remaining, cursor)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def verify_codex_linux_payload(
    package_root: Path,
    *,
    expected_receipt_sha256: str,
    expected_closure_manifest_sha256: str,
    expected_headless_adapter_config_sha256: str,
) -> VerifiedCodexLinuxPayload:
    """Offline-verify exact package bytes from three independent expected digests.

    The verifier opens one non-symlink package directory and performs descriptor-relative reads.
    It requires the exact top-level file set, exact content-addressed chunk set, canonical receipt,
    complete reconstruction of every future target file, and agreement with the out-of-band
    receipt/closure/config pins.  Success proves package consistency only.
    """

    _require_expected_sha256(expected_receipt_sha256, 'payload receipt')
    _require_expected_sha256(expected_closure_manifest_sha256, 'closure manifest')
    _require_expected_sha256(expected_headless_adapter_config_sha256, 'headless adapter config')
    normalized_root = Path(os.path.abspath(os.fspath(package_root.expanduser())))
    root_descriptor = _open_package_directory(normalized_root)
    chunks_descriptor = -1
    try:
        root_before = os.fstat(root_descriptor)
        _require_package_directory_metadata(
            root_before,
            required_mode=0o755,
            label='payload package root',
        )
        try:
            root_names = set(os.listdir(root_descriptor))
        except OSError:
            raise CodexLinuxPayloadError('payload package root cannot be enumerated safely') from None
        if root_names != {CODEX_LINUX_PAYLOAD_RECEIPT_FILE, CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY}:
            raise CodexLinuxPayloadError('payload package root has a missing or unexpected entry')
        receipt_bytes = _read_regular_file_at(
            root_descriptor,
            CODEX_LINUX_PAYLOAD_RECEIPT_FILE,
            maximum_bytes=_MAX_RECEIPT_BYTES,
            required_mode=0o444,
            label='payload receipt',
        )
        receipt_sha256 = _sha256(receipt_bytes)
        if not hmac.compare_digest(receipt_sha256, expected_receipt_sha256):
            raise CodexLinuxPayloadError('payload receipt differs from its independent expected digest')
        try:
            receipt = CodexLinuxPayloadReceipt.model_validate_json(receipt_bytes)
        except (TypeError, ValueError):
            raise CodexLinuxPayloadError('payload receipt schema is invalid') from None
        if not hmac.compare_digest(receipt_bytes, canonical_json_bytes(receipt)):
            raise CodexLinuxPayloadError('payload receipt is not exact canonical JSON')
        if not hmac.compare_digest(
            receipt.closure_manifest_sha256,
            expected_closure_manifest_sha256,
        ):
            raise CodexLinuxPayloadError('payload closure differs from its independent expected digest')
        if not hmac.compare_digest(
            receipt.headless_adapter_config_sha256,
            expected_headless_adapter_config_sha256,
        ):
            raise CodexLinuxPayloadError('payload headless config differs from its independent expected digest')
        if not hmac.compare_digest(receipt.build_contract_sha256, _build_contract_sha256()):
            raise CodexLinuxPayloadError('payload receipt names a different build contract')

        chunks_descriptor = _open_directory_at(
            root_descriptor,
            CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY,
            label='payload chunks directory',
        )
        chunks_before = os.fstat(chunks_descriptor)
        _require_package_directory_metadata(
            chunks_before,
            required_mode=0o555,
            label='payload chunks directory',
        )
        expected_chunks = _unique_chunks(receipt.files)
        try:
            observed_chunk_names = set(os.listdir(chunks_descriptor))
        except OSError:
            raise CodexLinuxPayloadError('payload chunks cannot be enumerated safely') from None
        if observed_chunk_names != set(expected_chunks):
            raise CodexLinuxPayloadError('payload chunk directory differs from the exact receipt inventory')
        for digest, byte_count in expected_chunks.items():
            body = _read_regular_file_at(
                chunks_descriptor,
                digest,
                maximum_bytes=CODEX_LINUX_PAYLOAD_CHUNK_BYTES,
                required_mode=0o444,
                label='payload chunk',
            )
            if len(body) != byte_count or not hmac.compare_digest(_sha256(body), digest):
                raise CodexLinuxPayloadError('payload chunk differs from its content-addressed identity')

        generated_bodies = {
            CODEX_LINUX_PAYLOAD_CLOSURE_TARGET: canonical_json_bytes(receipt.closure_manifest),
            CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET: canonical_json_bytes(receipt.headless_adapter_config),
        }
        for file_binding in receipt.files:
            digest = hashlib.sha256()
            observed_bytes = 0
            generated_parts: list[bytes] | None = [] if file_binding.target_path in generated_bodies else None
            for chunk in file_binding.chunks:
                body = _read_regular_file_at(
                    chunks_descriptor,
                    chunk.sha256,
                    maximum_bytes=CODEX_LINUX_PAYLOAD_CHUNK_BYTES,
                    required_mode=0o444,
                    label='payload reconstruction chunk',
                )
                if len(body) != chunk.byte_count:
                    raise CodexLinuxPayloadError('payload reconstruction chunk has the wrong size')
                digest.update(body)
                observed_bytes += len(body)
                if generated_parts is not None:
                    generated_parts.append(body)
            if observed_bytes != file_binding.byte_count or not hmac.compare_digest(
                digest.hexdigest(), file_binding.sha256
            ):
                raise CodexLinuxPayloadError('payload target reconstruction differs from its receipt')
            if generated_parts is not None and b''.join(generated_parts) != generated_bodies[file_binding.target_path]:
                raise CodexLinuxPayloadError('generated payload metadata bytes are not canonical')

        chunks_after = os.fstat(chunks_descriptor)
        _require_unchanged_directory(chunks_before, chunks_after, label='payload chunks directory')
        root_after = os.fstat(root_descriptor)
        _require_unchanged_directory(root_before, root_after, label='payload package root')
    except CodexLinuxPayloadError:
        raise
    except (OSError, TypeError, ValueError):
        raise CodexLinuxPayloadError('payload package verification failed') from None
    finally:
        if chunks_descriptor >= 0:
            os.close(chunks_descriptor)
        os.close(root_descriptor)
    return VerifiedCodexLinuxPayload(
        root=normalized_root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
    )


def _open_package_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexLinuxPayloadError('O_NOFOLLOW is required for payload verification')
    try:
        return os.open(path, flags | no_follow)
    except OSError:
        raise CodexLinuxPayloadError('payload package root cannot be opened safely') from None


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexLinuxPayloadError('O_NOFOLLOW is required for payload verification')
    try:
        return os.open(name, flags | no_follow, dir_fd=parent_descriptor)
    except OSError:
        raise CodexLinuxPayloadError(f'{label} cannot be opened safely') from None


def _read_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    required_mode: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise CodexLinuxPayloadError('O_NOFOLLOW is required for payload verification')
    descriptor = -1
    try:
        descriptor = os.open(name, flags | no_follow, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(before.st_mode) != required_mode
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise CodexLinuxPayloadError(f'{label} metadata is not admitted')
        body = _read_up_to(descriptor, maximum_bytes + 1)
        after = os.fstat(descriptor)
        _require_unchanged_metadata(before, after, label=label)
        if len(body) != before.st_size or len(body) > maximum_bytes:
            raise CodexLinuxPayloadError(f'{label} byte count is invalid')
        return body
    except CodexLinuxPayloadError:
        raise
    except OSError:
        raise CodexLinuxPayloadError(f'{label} cannot be read safely') from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_package_directory_metadata(
    metadata: os.stat_result,
    *,
    required_mode: int,
    label: str,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in {0, os.geteuid()}
        or stat.S_IMODE(metadata.st_mode) != required_mode
    ):
        raise CodexLinuxPayloadError(f'{label} metadata is not admitted')


def _require_unchanged_directory(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise CodexLinuxPayloadError(f'{label} changed during offline verification')


def _stable_file_sha256(path: Path) -> str:
    descriptor = _open_stable_source(path)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require_source_metadata(before)
        while True:
            chunk = _read_up_to(descriptor, CODEX_LINUX_PAYLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        _require_unchanged_metadata(before, after, label='payload builder source')
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_expected_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise CodexLinuxPayloadError(f'{label} expected digest is invalid')
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        raise CodexLinuxPayloadError(f'{label} expected digest is invalid') from None
    if len(decoded) != 32 or value != value.lower():
        raise CodexLinuxPayloadError(f'{label} expected digest is invalid')


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


__all__ = [
    'CODEX_LINUX_DEPENDENCY_CLOSURE_SCHEMA_VERSION',
    'CODEX_LINUX_PAYLOAD_CHUNK_BYTES',
    'CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY',
    'CODEX_LINUX_PAYLOAD_CLOSURE_TARGET',
    'CODEX_LINUX_PAYLOAD_HEADLESS_CONFIG_TARGET',
    'CODEX_LINUX_PAYLOAD_PROFILE',
    'CODEX_LINUX_PAYLOAD_RECEIPT_FILE',
    'CODEX_LINUX_PAYLOAD_RECEIPT_SCHEMA_VERSION',
    'CodexLinuxClosureEntry',
    'CodexLinuxDependencyClosureManifest',
    'CodexLinuxEntrypointFormat',
    'CodexLinuxPackagingBoundary',
    'CodexLinuxPayloadChunkBinding',
    'CodexLinuxPayloadError',
    'CodexLinuxPayloadFileBinding',
    'CodexLinuxPayloadReceipt',
    'CodexLinuxPayloadRole',
    'VerifiedCodexLinuxPayload',
    'WrittenCodexLinuxPayload',
    'build_codex_linux_payload',
    'codex_linux_dependency_closure_sha256',
    'codex_linux_payload_receipt_sha256',
    'verify_codex_linux_payload',
]
