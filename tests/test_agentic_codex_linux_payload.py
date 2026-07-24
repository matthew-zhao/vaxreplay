from __future__ import annotations

import hashlib
import os
import stat
import struct
from pathlib import Path

import pytest

from vaxreplay.agentic.codex_guest_adapter import (
    CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
    CODEX_VENDOR_EXECUTABLE_PATH,
    codex_vendor_argv_template,
)
from vaxreplay.agentic.codex_linux_payload import (
    CODEX_LINUX_PAYLOAD_CHUNK_BYTES,
    CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY,
    CODEX_LINUX_PAYLOAD_RECEIPT_FILE,
    CodexLinuxClosureEntry,
    CodexLinuxDependencyClosureManifest,
    CodexLinuxEntrypointFormat,
    CodexLinuxPackagingBoundary,
    CodexLinuxPayloadError,
    CodexLinuxPayloadRole,
    build_codex_linux_payload,
    codex_linux_dependency_closure_sha256,
    verify_codex_linux_payload,
)
from vaxreplay.agentic.headless_guest_adapter import (
    HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
)
from vaxreplay.agentic.submitted_harness import HarnessFamily


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write_source(path: Path, body: bytes, *, executable: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(0o500 if executable else 0o400)
    return path


def _payload_materials(
    root: Path,
    *,
    vendor_body: bytes = b'fake-codex-entrypoint',
    adapter_body: bytes = b'fake-adapter-entrypoint',
    entrypoint_format: CodexLinuxEntrypointFormat = CodexLinuxEntrypointFormat.TEST_OPAQUE,
) -> tuple[
    dict[str, Path],
    CodexLinuxDependencyClosureManifest,
    HeadlessGuestAdapterConfig,
]:
    source_files = {
        CODEX_VENDOR_EXECUTABLE_PATH: _write_source(root / 'sources' / 'codex', vendor_body, executable=True),
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH: _write_source(
            root / 'sources' / 'adapter', adapter_body, executable=True
        ),
        '/opt/vaxreplay/lib/adapter/runtime.dat': _write_source(
            root / 'sources' / 'runtime.dat', b'pinned adapter support bytes', executable=False
        ),
    }
    roles = {
        CODEX_VENDOR_EXECUTABLE_PATH: CodexLinuxPayloadRole.VENDOR_EXECUTABLE,
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH: CodexLinuxPayloadRole.ADAPTER_EXECUTABLE,
        '/opt/vaxreplay/lib/adapter/runtime.dat': CodexLinuxPayloadRole.ADAPTER_SUPPORT_FILE,
    }
    modes = {
        CODEX_VENDOR_EXECUTABLE_PATH: 0o555,
        HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH: 0o555,
        '/opt/vaxreplay/lib/adapter/runtime.dat': 0o444,
    }
    entries = tuple(
        CodexLinuxClosureEntry(
            target_path=target,
            role=roles[target],
            sha256=_sha256(source_files[target].read_bytes()),
            byte_count=source_files[target].stat().st_size,
            mode=modes[target],
        )
        for target in sorted(source_files)
    )
    elf_format = entrypoint_format == CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER
    closure = CodexLinuxDependencyClosureManifest(
        target_machine='aarch64',
        vendor_entrypoint_format=entrypoint_format,
        adapter_entrypoint_format=entrypoint_format,
        base_rootfs_runtime_manifest_sha256='b' * 64,
        entries=entries,
        primary_entrypoints_have_no_elf_interpreter_attested=elf_format,
    )
    closure_sha256 = codex_linux_dependency_closure_sha256(closure)
    headless = HeadlessGuestAdapterConfig(
        family=HarnessFamily.CODEX,
        invocation_protocol=HeadlessInvocationProtocol.CODEX_EXEC,
        adapter_executable_sha256=_sha256(adapter_body),
        vendor_executable_path=CODEX_VENDOR_EXECUTABLE_PATH,
        vendor_executable_sha256=_sha256(vendor_body),
        complete_dependency_closure_sha256=closure_sha256,
        vendor_reported_version=CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
        vendor_version_output_sha256='c' * 64,
        vendor_config_template_sha256='d' * 64,
        vendor_argv_template=codex_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
        local_shell_enabled=True,
        adapter_implementation_checked_in=True,
        provider_shim_implementation_checked_in=True,
        workspace_materialization_bridge_implementation_checked_in=True,
    )
    return source_files, closure, headless


def _build_fixture(root: Path, output_name: str = 'package'):
    sources, closure, headless = _payload_materials(root)
    return build_codex_linux_payload(
        sources,
        root / output_name,
        closure_manifest=closure,
        expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(closure),
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(headless),
        testing_only=True,
    )


def _package_snapshot(root: Path) -> tuple[tuple[str, int, bytes], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            stat.S_IMODE(path.stat().st_mode),
            path.read_bytes(),
        )
        for path in sorted(root.rglob('*'))
        if path.is_file()
    )


def test_testing_payload_is_deterministic_deduplicated_and_offline_verifiable(tmp_path: Path) -> None:
    common_entrypoint = b'the same opaque fixture bytes'
    sources, closure, headless = _payload_materials(
        tmp_path,
        vendor_body=common_entrypoint,
        adapter_body=common_entrypoint,
    )
    closure_sha256 = codex_linux_dependency_closure_sha256(closure)
    headless_sha256 = headless_guest_adapter_config_sha256(headless)

    first = build_codex_linux_payload(
        sources,
        tmp_path / 'package-a',
        closure_manifest=closure,
        expected_closure_manifest_sha256=closure_sha256,
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_sha256,
        testing_only=True,
    )
    second = build_codex_linux_payload(
        sources,
        tmp_path / 'package-b',
        closure_manifest=closure,
        expected_closure_manifest_sha256=closure_sha256,
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_sha256,
        testing_only=True,
    )

    assert first.receipt == second.receipt
    assert first.receipt_sha256 == second.receipt_sha256
    assert _package_snapshot(first.root) == _package_snapshot(second.root)
    assert first.receipt.packaging_boundary == CodexLinuxPackagingBoundary.TEST_FIXTURE
    assert not first.receipt.linux_elf_structure_and_no_interpreter_verified
    assert not first.receipt.structurally_eligible_as_guest_disk_builder_input
    assert not first.receipt.actual_linux_codex_executed
    assert not first.receipt.linux_kvm_qualified
    assert first.receipt.unique_chunk_count < sum(len(item.chunks) for item in first.receipt.files)
    assert str(tmp_path).encode() not in first.receipt_path.read_bytes()

    verified = verify_codex_linux_payload(
        first.root,
        expected_receipt_sha256=first.receipt_sha256,
        expected_closure_manifest_sha256=closure_sha256,
        expected_headless_adapter_config_sha256=headless_sha256,
    )
    assert verified.receipt == first.receipt


def test_offline_verifier_rejects_chunk_mutation_extra_file_and_wrong_pin(tmp_path: Path) -> None:
    written = _build_fixture(tmp_path)
    closure_sha256 = written.receipt.closure_manifest_sha256
    headless_sha256 = written.receipt.headless_adapter_config_sha256
    chunk = next((written.root / CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY).iterdir())
    body = chunk.read_bytes()
    chunk.chmod(0o644)
    chunk.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    chunk.chmod(0o444)

    with pytest.raises(CodexLinuxPayloadError):
        verify_codex_linux_payload(
            written.root,
            expected_receipt_sha256=written.receipt_sha256,
            expected_closure_manifest_sha256=closure_sha256,
            expected_headless_adapter_config_sha256=headless_sha256,
        )

    second = _build_fixture(tmp_path / 'extra')
    chunk_dir = second.root / CODEX_LINUX_PAYLOAD_CHUNK_DIRECTORY
    chunk_dir.chmod(0o755)
    extra = chunk_dir / ('f' * 64)
    extra.write_bytes(b'extra')
    extra.chmod(0o444)
    chunk_dir.chmod(0o555)
    with pytest.raises(CodexLinuxPayloadError):
        verify_codex_linux_payload(
            second.root,
            expected_receipt_sha256=second.receipt_sha256,
            expected_closure_manifest_sha256=second.receipt.closure_manifest_sha256,
            expected_headless_adapter_config_sha256=second.receipt.headless_adapter_config_sha256,
        )

    third = _build_fixture(tmp_path / 'pin')
    with pytest.raises(CodexLinuxPayloadError):
        verify_codex_linux_payload(
            third.root,
            expected_receipt_sha256='0' * 64,
            expected_closure_manifest_sha256=third.receipt.closure_manifest_sha256,
            expected_headless_adapter_config_sha256=third.receipt.headless_adapter_config_sha256,
        )


def test_builder_rejects_incomplete_changed_symlink_and_non_test_opaque_inputs(tmp_path: Path) -> None:
    sources, closure, headless = _payload_materials(tmp_path)
    closure_sha256 = codex_linux_dependency_closure_sha256(closure)
    headless_sha256 = headless_guest_adapter_config_sha256(headless)

    missing = dict(sources)
    missing.pop(CODEX_VENDOR_EXECUTABLE_PATH)
    with pytest.raises(CodexLinuxPayloadError):
        build_codex_linux_payload(
            missing,
            tmp_path / 'missing',
            closure_manifest=closure,
            expected_closure_manifest_sha256=closure_sha256,
            headless_adapter_config=headless,
            expected_headless_adapter_config_sha256=headless_sha256,
            testing_only=True,
        )

    sources[CODEX_VENDOR_EXECUTABLE_PATH].chmod(0o700)
    sources[CODEX_VENDOR_EXECUTABLE_PATH].write_bytes(b'changed')
    sources[CODEX_VENDOR_EXECUTABLE_PATH].chmod(0o500)
    with pytest.raises(CodexLinuxPayloadError):
        build_codex_linux_payload(
            sources,
            tmp_path / 'changed',
            closure_manifest=closure,
            expected_closure_manifest_sha256=closure_sha256,
            headless_adapter_config=headless,
            expected_headless_adapter_config_sha256=headless_sha256,
            testing_only=True,
        )

    fresh_sources, fresh_closure, fresh_headless = _payload_materials(tmp_path / 'fresh')
    target = fresh_sources[CODEX_VENDOR_EXECUTABLE_PATH]
    target.unlink()
    target.symlink_to(fresh_sources[HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH])
    with pytest.raises(CodexLinuxPayloadError):
        build_codex_linux_payload(
            fresh_sources,
            tmp_path / 'symlink',
            closure_manifest=fresh_closure,
            expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(fresh_closure),
            headless_adapter_config=fresh_headless,
            expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(fresh_headless),
            testing_only=True,
        )

    opaque_sources, opaque_closure, opaque_headless = _payload_materials(tmp_path / 'opaque')
    with pytest.raises(CodexLinuxPayloadError):
        build_codex_linux_payload(
            opaque_sources,
            tmp_path / 'not-testing',
            closure_manifest=opaque_closure,
            expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(opaque_closure),
            headless_adapter_config=opaque_headless,
            expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(opaque_headless),
        )


def _minimal_elf64(*, machine: int = 183, interpreter: bool = False) -> bytes:
    program_types = (1, 3) if interpreter else (1,)
    program_count = len(program_types)
    byte_count = 64 + 56 * program_count
    identifier = b'\x7fELF' + bytes((2, 1, 1, 0)) + b'\x00' * 8
    header = struct.pack(
        '<16sHHIQQQIHHHHHH',
        identifier,
        2,
        machine,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        program_count,
        0,
        0,
        0,
    )
    programs = b''.join(
        struct.pack('<IIQQQQQQ', program_type, 5, 0, 0, 0, byte_count, byte_count, 4096)
        for program_type in program_types
    )
    return header + programs


def test_non_test_path_structurally_inspects_elf_without_claiming_release_or_execution(tmp_path: Path) -> None:
    elf = _minimal_elf64()
    sources, closure, headless = _payload_materials(
        tmp_path,
        vendor_body=elf,
        adapter_body=elf,
        entrypoint_format=CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER,
    )
    written = build_codex_linux_payload(
        sources,
        tmp_path / 'elf-package',
        closure_manifest=closure,
        expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(closure),
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(headless),
    )

    assert written.receipt.packaging_boundary == CodexLinuxPackagingBoundary.OFFLINE_ELF64_INSPECTION
    assert written.receipt.linux_elf_structure_and_no_interpreter_verified
    assert written.receipt.structurally_eligible_as_guest_disk_builder_input
    assert not written.receipt.official_codex_release_provenance_verified
    assert not written.receipt.actual_linux_codex_executed
    assert not written.receipt.guest_disk_built
    assert not written.receipt.linux_kvm_qualified

    bad_sources, bad_closure, bad_headless = _payload_materials(
        tmp_path / 'bad',
        vendor_body=elf,
        adapter_body=_minimal_elf64(interpreter=True),
        entrypoint_format=CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER,
    )
    with pytest.raises(CodexLinuxPayloadError, match='ELF interpreter'):
        build_codex_linux_payload(
            bad_sources,
            tmp_path / 'bad-package',
            closure_manifest=bad_closure,
            expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(bad_closure),
            headless_adapter_config=bad_headless,
            expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(bad_headless),
        )


def test_fixed_size_chunking_and_declared_architecture_are_enforced(tmp_path: Path) -> None:
    large_vendor = b'x' * (CODEX_LINUX_PAYLOAD_CHUNK_BYTES + 17)
    sources, closure, headless = _payload_materials(tmp_path, vendor_body=large_vendor)
    written = build_codex_linux_payload(
        sources,
        tmp_path / 'chunked-package',
        closure_manifest=closure,
        expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(closure),
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(headless),
        testing_only=True,
    )
    vendor = next(item for item in written.receipt.files if item.role == CodexLinuxPayloadRole.VENDOR_EXECUTABLE)
    assert tuple(item.byte_count for item in vendor.chunks) == (
        CODEX_LINUX_PAYLOAD_CHUNK_BYTES,
        17,
    )

    wrong_machine_sources, wrong_machine_closure, wrong_machine_headless = _payload_materials(
        tmp_path / 'wrong-machine',
        vendor_body=_minimal_elf64(machine=62),
        adapter_body=_minimal_elf64(),
        entrypoint_format=CodexLinuxEntrypointFormat.ELF64_LITTLE_ENDIAN_NO_INTERPRETER,
    )
    with pytest.raises(CodexLinuxPayloadError, match='ELF64 identity'):
        build_codex_linux_payload(
            wrong_machine_sources,
            tmp_path / 'wrong-machine-package',
            closure_manifest=wrong_machine_closure,
            expected_closure_manifest_sha256=codex_linux_dependency_closure_sha256(wrong_machine_closure),
            headless_adapter_config=wrong_machine_headless,
            expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(wrong_machine_headless),
        )


def test_payload_publication_is_create_once_and_headless_binding_is_exact(tmp_path: Path) -> None:
    sources, closure, headless = _payload_materials(tmp_path)
    closure_sha256 = codex_linux_dependency_closure_sha256(closure)
    headless_sha256 = headless_guest_adapter_config_sha256(headless)
    build_codex_linux_payload(
        sources,
        tmp_path / 'create-once',
        closure_manifest=closure,
        expected_closure_manifest_sha256=closure_sha256,
        headless_adapter_config=headless,
        expected_headless_adapter_config_sha256=headless_sha256,
        testing_only=True,
    )
    with pytest.raises(CodexLinuxPayloadError, match='already exists'):
        build_codex_linux_payload(
            sources,
            tmp_path / 'create-once',
            closure_manifest=closure,
            expected_closure_manifest_sha256=closure_sha256,
            headless_adapter_config=headless,
            expected_headless_adapter_config_sha256=headless_sha256,
            testing_only=True,
        )

    wrong_headless = headless.model_copy(update={'complete_dependency_closure_sha256': 'f' * 64})
    with pytest.raises(CodexLinuxPayloadError):
        build_codex_linux_payload(
            sources,
            tmp_path / 'wrong-binding',
            closure_manifest=closure,
            expected_closure_manifest_sha256=closure_sha256,
            headless_adapter_config=wrong_headless,
            expected_headless_adapter_config_sha256=headless_guest_adapter_config_sha256(wrong_headless),
            testing_only=True,
        )


def test_receipt_file_is_read_only_canonical_json(tmp_path: Path) -> None:
    written = _build_fixture(tmp_path)
    receipt_path = written.root / CODEX_LINUX_PAYLOAD_RECEIPT_FILE
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert receipt_path.read_bytes().startswith(b'{')
    assert not receipt_path.read_bytes().endswith(os.linesep.encode())
