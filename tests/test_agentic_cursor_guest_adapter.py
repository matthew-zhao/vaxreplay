from __future__ import annotations

import base64
import hashlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_scoring import _case, _submission
from vaxreplay.agentic.cursor_guest_adapter import (
    CURSOR_SUPPORTED_VENDOR_VERSION,
    CURSOR_VENDOR_EXECUTABLE_PATH,
    CursorProtocolFixtureConfig,
    CursorProtocolFixtureError,
    CursorProtocolFixtureFailureCode,
    capture_cursor_vendor_identity,
    cursor_vendor_argv_template,
    run_cursor_protocol_fixture,
)
from vaxreplay.agentic.guest_rpc import (
    ListWorkspaceResult,
    LogicalFileResult,
    ReadWorkspaceResult,
    SubmitResult,
)
from vaxreplay.agentic.headless_guest_adapter import (
    HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
    HeadlessGuestAdapterConfig,
    HeadlessInvocationProtocol,
    HeadlessResponseChannel,
    headless_guest_adapter_config_sha256,
)
from vaxreplay.agentic.submitted_harness import (
    HarnessExecutionMode,
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
    submitted_harness_manifest_sha256,
)
from vaxreplay.agentic.task_protocol import AgenticRuntimeSubmission, AgenticTaskInvocation
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission
from vaxreplay.runner._process import BoundedProcessResult

_MODEL_SELECTOR = 'organizer-model-route-1'
_SESSION_ID = '00000000-0000-4000-8000-000000000001'
_MODEL_CALL_ID = '00000000-0000-4000-8000-000000000002'
_REQUEST_ID = '00000000-0000-4000-8000-000000000003'


class _FakeClient:
    def __init__(self, *, files: dict[str, bytes]) -> None:
        self.files = files
        self.list_calls: list[tuple[int, int]] = []
        self.read_calls: list[tuple[str, int, int]] = []
        self.submissions: list[AgenticRuntimeSubmission] = []

    def list_workspace(self, *, cursor: int = 0, limit: int = 100) -> ListWorkspaceResult:
        self.list_calls.append((cursor, limit))
        names = sorted(self.files)
        page = names[cursor : cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(names) else None
        return ListWorkspaceResult(
            files=tuple(
                LogicalFileResult(
                    path=name,
                    media_type='text/plain',
                    sha256=hashlib.sha256(self.files[name]).hexdigest(),
                    byte_count=len(self.files[name]),
                )
                for name in page
            ),
            next_cursor=next_cursor,
        )

    def read_workspace(self, path: str, *, offset: int = 0, limit: int) -> ReadWorkspaceResult:
        self.read_calls.append((path, offset, limit))
        selected = self.files[path][offset : offset + limit]
        return ReadWorkspaceResult(
            content_base64=base64.b64encode(selected).decode('ascii'),
            offset=offset,
            byte_count=len(selected),
            eof=offset + len(selected) == len(self.files[path]),
        )

    def submit(self, submission: AgenticRuntimeSubmission) -> SubmitResult:
        self.submissions.append(submission)
        body = canonical_json_bytes(submission)
        return SubmitResult(
            submission_sha256=hashlib.sha256(body).hexdigest(),
            submission_bytes=len(body),
        )


def _invocation_and_submission() -> tuple[AgenticTaskInvocation, ExecutionSubmission]:
    task, _gold = _case(with_fact=False)
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256='a' * 64)
    return invocation, _submission(task)


def _materials(
    root: Path,
) -> tuple[
    CursorProtocolFixtureConfig,
    HeadlessGuestAdapterConfig,
    SubmittedHarnessManifest,
]:
    vendor_body = b'#!/bin/sh\nexit 97\n'
    vendor_path = root.joinpath(*Path(CURSOR_VENDOR_EXECUTABLE_PATH).parts[1:])
    vendor_path.parent.mkdir(parents=True)
    vendor_path.write_bytes(vendor_body)
    vendor_path.chmod(0o500)
    vendor_sha = hashlib.sha256(vendor_body).hexdigest()
    headless = HeadlessGuestAdapterConfig(
        family=HarnessFamily.CURSOR,
        invocation_protocol=HeadlessInvocationProtocol.CURSOR_PRINT,
        adapter_executable_sha256='1' * 64,
        vendor_executable_path=CURSOR_VENDOR_EXECUTABLE_PATH,
        vendor_executable_sha256=vendor_sha,
        complete_dependency_closure_sha256='3' * 64,
        vendor_reported_version=CURSOR_SUPPORTED_VENDOR_VERSION,
        vendor_version_output_sha256='4' * 64,
        vendor_config_template_sha256='5' * 64,
        vendor_argv_template=cursor_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
        local_shell_enabled=False,
    )
    headless_sha = headless_guest_adapter_config_sha256(headless)
    manifest = SubmittedHarnessManifest(
        harness_id='cursor-protocol-fixture',
        harness_version='dev-v0.1',
        family=HarnessFamily.CURSOR,
        execution_mode=HarnessExecutionMode.SUBMITTED_GUEST_AGENT,
        runtime_support=HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED,
        harness_image_sha256='6' * 64,
        harness_image_byte_count=4096,
        normalized_runtime_tree_sha256='7' * 64,
        guest_executable_path=HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        guest_executable_sha256=headless.adapter_executable_sha256,
        guest_argv=(
            HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
            '--expected-config-sha256',
            headless_sha,
        ),
        baked_config_sha256=headless_sha,
        dependency_closure_sha256=headless.complete_dependency_closure_sha256,
        reproducible_build_receipt_sha256='8' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=True,
            guest_local_shell_allowed=False,
        ),
        display_name='Cursor protocol fixture (not a runnable adapter)',
        submitter='fixture',
    )
    runtime = CursorProtocolFixtureConfig(
        headless_adapter_config_sha256=headless_sha,
        submitted_harness_manifest_sha256=submitted_harness_manifest_sha256(manifest),
        vendor_executable_sha256=vendor_sha,
        vendor_executable_byte_count=len(vendor_body),
    )
    return runtime, headless, manifest


def _cursor_events(
    *,
    prompt: bytes,
    submission: ExecutionSubmission,
    workspace_path: str,
    tool_kind: str = 'readToolCall',
) -> list[dict[str, object]]:
    submission_text = canonical_json_bytes(submission).decode('utf-8')
    tool_started = {
        tool_kind: {
            'args': {'path': 'TASK.md'},
        },
        'hookAdditionalContexts': [],
        'toolCallId': 'read-1',
        'startedAtMs': '1',
    }
    tool_completed = {
        tool_kind: {
            'args': {'path': 'TASK.md'},
            'result': {'content': 'brokered task bytes'},
        },
        'hookAdditionalContexts': [],
        'toolCallId': 'read-1',
        'startedAtMs': '1',
        'completedAtMs': '2',
    }
    return [
        {
            'type': 'system',
            'subtype': 'init',
            'apiKeySource': 'protocol-fixture',
            'cwd': workspace_path,
            'session_id': _SESSION_ID,
            'model': _MODEL_SELECTOR,
            'permissionMode': 'default',
        },
        {
            'type': 'user',
            'message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': prompt.decode('utf-8')}],
            },
            'session_id': _SESSION_ID,
        },
        {
            'type': 'tool_call',
            'subtype': 'started',
            'call_id': 'read-1',
            'tool_call': tool_started,
            'model_call_id': _MODEL_CALL_ID,
            'session_id': _SESSION_ID,
            'timestamp_ms': 1,
        },
        {
            'type': 'tool_call',
            'subtype': 'completed',
            'call_id': 'read-1',
            'tool_call': tool_completed,
            'model_call_id': _MODEL_CALL_ID,
            'session_id': _SESSION_ID,
            'timestamp_ms': 2,
        },
        {
            'type': 'assistant',
            'message': {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': submission_text}],
            },
            'model_call_id': _MODEL_CALL_ID,
            'session_id': _SESSION_ID,
            'timestamp_ms': 3,
        },
        {
            'type': 'result',
            'subtype': 'success',
            'duration_ms': 3,
            'duration_api_ms': 2,
            'is_error': False,
            'result': submission_text,
            'session_id': _SESSION_ID,
            'request_id': _REQUEST_ID,
            'usage': {'inputTokens': 1, 'outputTokens': 1},
        },
    ]


def _transport(
    submission: ExecutionSubmission,
    *,
    transform: Callable[[list[dict[str, object]]], list[dict[str, object]]] | None = None,
    raw_stdout: bytes | None = None,
) -> Callable[..., BoundedProcessResult]:
    def run(
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        wall_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        on_abort: Callable[[], None],
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> BoundedProcessResult:
        del wall_seconds, max_stdout_bytes, max_stderr_bytes, on_abort
        assert env is not None
        assert cwd is not None
        assert '--print' in argv
        assert argv[argv.index('--output-format') + 1] == 'stream-json'
        assert argv[argv.index('--mode') + 1] == 'ask'
        assert argv[argv.index('--sandbox') + 1] == 'enabled'
        assert argv[argv.index('--model') + 1] == _MODEL_SELECTOR
        assert [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == '--allowed-tools'] == [
            'read_tool_call',
            'ls_tool_call',
            'grep_tool_call',
            'glob_tool_call',
        ]
        assert Path(cwd).joinpath('TASK.md').read_bytes()
        forbidden_parts = ('API_KEY', 'AUTH_TOKEN', 'PASSWORD', 'PROXY', 'SECRET')
        assert not any(any(part in key for part in forbidden_parts) for key in env)
        if raw_stdout is not None:
            stdout = raw_stdout
        else:
            events = _cursor_events(
                prompt=input_bytes,
                submission=submission,
                workspace_path=str(cwd),
            )
            if transform is not None:
                events = transform(events)
            stdout = b''.join(canonical_json_bytes(event) + b'\n' for event in events)
        return BoundedProcessResult(
            exit_code=0,
            duration_ms=3,
            stdout=stdout,
            stderr=b'',
            termination='exited',
            stdout_truncated=False,
            stderr_truncated=False,
        )

    return run


def test_cursor_protocol_fixture_materializes_workspace_and_validates_tool_lifecycle(
    tmp_path: Path,
) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path)
    files = {
        'TASK.md': b'Use the frozen brokered sources.\n',
        'sources/evidence.txt': b'pre-cutoff evidence\n',
    }
    client = _FakeClient(files=files)
    os.environ['CURSOR_API_KEY'] = 'must-not-be-inherited'
    os.environ['HTTPS_PROXY'] = 'http://must-not-be-inherited.invalid'
    try:
        receipt = run_cursor_protocol_fixture(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            protocol_transport=_transport(submission),
            _guest_root=tmp_path,
        )
    finally:
        os.environ.pop('CURSOR_API_KEY')
        os.environ.pop('HTTPS_PROXY')

    assert client.submissions == [submission]
    assert receipt.workspace_file_count == len(files)
    assert receipt.workspace_byte_count == sum(map(len, files.values()))
    assert receipt.completed_read_only_tool_calls == 1
    assert [item.event_type for item in receipt.protocol_events] == [
        'system',
        'user',
        'tool_call',
        'tool_call',
        'assistant',
        'result',
    ]
    assert receipt.transport_was_caller_supplied
    assert receipt.transport_provenance_is_not_attested
    assert receipt.no_builtin_live_transport
    assert receipt.cursor_provider_gateway_bridge_implemented is False
    assert receipt.actual_cursor_binary_end_to_end_validated is False
    assert receipt.actual_external_model_call_claimed is False
    assert receipt.development_adapter_integrated is False
    assert receipt.linux_kvm_qualified is False
    assert manifest.runtime_support == HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED
    assert (tmp_path / 'run/vaxreplay/workspace/TASK.md').stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize(
    'transform',
    (
        lambda events: [
            *events[:2],
            {'type': 'retry', 'subtype': 'starting'},
            *events[2:],
        ],
        lambda events: [
            *events[:2],
            {
                **events[2],
                'tool_call': {'editToolCall': {'args': {'path': 'TASK.md'}}},
            },
            *events[3:],
        ],
        lambda events: [
            *events[:-2],
            {
                **events[-2],
                'session_id': 'different-session',
            },
            events[-1],
        ],
        lambda events: [
            *events[:-2],
            {
                **events[-2],
                'message': {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': '{}'}],
                },
            },
            {**events[-1], 'result': '{}'},
        ],
    ),
)
def test_cursor_protocol_fixture_rejects_unknown_write_mismatched_and_unbound_streams(
    tmp_path: Path,
    transform: Callable[[list[dict[str, object]]], list[dict[str, object]]],
) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path)
    client = _FakeClient(files={'TASK.md': b'x'})

    with pytest.raises(CursorProtocolFixtureError) as captured:
        run_cursor_protocol_fixture(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            protocol_transport=_transport(submission, transform=transform),
            _guest_root=tmp_path,
        )

    assert captured.value.code == CursorProtocolFixtureFailureCode.OUTPUT_REJECTED
    assert not client.submissions


def test_cursor_protocol_fixture_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path)
    client = _FakeClient(files={'TASK.md': b'x'})
    duplicate = b'{"type":"system","type":"system"}\n{}\n{}\n{}\n'

    with pytest.raises(CursorProtocolFixtureError) as captured:
        run_cursor_protocol_fixture(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            protocol_transport=_transport(submission, raw_stdout=duplicate),
            _guest_root=tmp_path,
        )

    assert captured.value.code == CursorProtocolFixtureFailureCode.OUTPUT_REJECTED
    assert not client.submissions


def test_cursor_workspace_escape_and_binding_tamper_fail_before_transport(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path)
    escaping = _FakeClient(files={'../future.txt': b'leak'})
    transport_called = False

    def forbidden_transport(*args: object, **kwargs: object) -> BoundedProcessResult:
        nonlocal transport_called
        transport_called = True
        raise AssertionError((args, kwargs))

    with pytest.raises(CursorProtocolFixtureError) as captured:
        run_cursor_protocol_fixture(
            escaping,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            protocol_transport=forbidden_transport,
            _guest_root=tmp_path,
        )
    assert captured.value.code == CursorProtocolFixtureFailureCode.WORKSPACE_REJECTED
    assert not transport_called

    fresh = tmp_path / 'fresh'
    runtime, headless, manifest = _materials(fresh)
    wrong = runtime.model_copy(update={'headless_adapter_config_sha256': 'f' * 64})
    client = _FakeClient(files={'TASK.md': b'x'})
    with pytest.raises(CursorProtocolFixtureError) as captured:
        run_cursor_protocol_fixture(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=wrong,
            headless_config=headless,
            submitted_manifest=manifest,
            protocol_transport=_transport(submission),
            _guest_root=fresh,
        )
    assert captured.value.code == CursorProtocolFixtureFailureCode.BINDING_REJECTED
    assert not client.list_calls


def test_cursor_cannot_promote_fixture_to_development_adapter() -> None:
    runtime, headless, manifest = _materials_for_validation_without_files()
    del runtime

    headless_payload = headless.model_dump(mode='python')
    headless_payload.update(
        adapter_implementation_checked_in=True,
        provider_shim_implementation_checked_in=True,
        workspace_materialization_bridge_implementation_checked_in=True,
    )
    with pytest.raises(ValidationError, match='no checked-in development adapter|has no checked-in'):
        HeadlessGuestAdapterConfig.model_validate(headless_payload)

    manifest_payload = manifest.model_dump(mode='python')
    manifest_payload['runtime_support'] = HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED
    with pytest.raises(ValidationError, match='checked-in development adapter'):
        SubmittedHarnessManifest.model_validate(manifest_payload)


def _materials_for_validation_without_files() -> tuple[
    CursorProtocolFixtureConfig,
    HeadlessGuestAdapterConfig,
    SubmittedHarnessManifest,
]:
    # Keep this helper pure so the promotion test does not need to mutate a shared /tmp path.
    vendor_sha = hashlib.sha256(b'fixture').hexdigest()
    headless = HeadlessGuestAdapterConfig(
        family=HarnessFamily.CURSOR,
        invocation_protocol=HeadlessInvocationProtocol.CURSOR_PRINT,
        adapter_executable_sha256='1' * 64,
        vendor_executable_path=CURSOR_VENDOR_EXECUTABLE_PATH,
        vendor_executable_sha256=vendor_sha,
        complete_dependency_closure_sha256='3' * 64,
        vendor_reported_version=CURSOR_SUPPORTED_VENDOR_VERSION,
        vendor_version_output_sha256='4' * 64,
        vendor_config_template_sha256='5' * 64,
        vendor_argv_template=cursor_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_JSONL_STDOUT,
        local_shell_enabled=False,
    )
    headless_sha = headless_guest_adapter_config_sha256(headless)
    manifest = SubmittedHarnessManifest(
        harness_id='cursor-protocol-fixture',
        harness_version='dev-v0.1',
        family=HarnessFamily.CURSOR,
        execution_mode=HarnessExecutionMode.SUBMITTED_GUEST_AGENT,
        runtime_support=HarnessRuntimeSupport.CONTRACT_ONLY_ADAPTER_REQUIRED,
        harness_image_sha256='6' * 64,
        harness_image_byte_count=4096,
        normalized_runtime_tree_sha256='7' * 64,
        guest_executable_path=HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
        guest_executable_sha256=headless.adapter_executable_sha256,
        guest_argv=(
            HEADLESS_GUEST_ADAPTER_EXECUTABLE_PATH,
            '--expected-config-sha256',
            headless_sha,
        ),
        baked_config_sha256=headless_sha,
        dependency_closure_sha256=headless.complete_dependency_closure_sha256,
        reproducible_build_receipt_sha256='8' * 64,
        interface=SubmittedHarnessInterface(
            guest_local_subprocesses_allowed=True,
            guest_local_shell_allowed=False,
        ),
        display_name='Cursor protocol fixture',
        submitter='fixture',
    )
    return (
        CursorProtocolFixtureConfig(
            headless_adapter_config_sha256=headless_sha,
            submitted_harness_manifest_sha256=submitted_harness_manifest_sha256(manifest),
            vendor_executable_sha256=vendor_sha,
            vendor_executable_byte_count=len(b'fixture'),
        ),
        headless,
        manifest,
    )


def test_cursor_vendor_identity_capture_is_exact_and_rejects_symlink(tmp_path: Path) -> None:
    executable = tmp_path / 'cursor-agent-real-file'
    executable.write_text(
        f"""#!{sys.executable}
import os
import sys
if sys.argv[1:] != ["--version"]:
    raise SystemExit(2)
if "CURSOR_API_KEY" in os.environ or "CURSOR_AUTH_TOKEN" in os.environ:
    raise SystemExit(3)
print("2026.07.09-a3815c0")
""",
        encoding='utf-8',
    )
    executable.chmod(0o500)

    evidence = capture_cursor_vendor_identity(executable)

    assert evidence.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert evidence.executable_byte_count == executable.stat().st_size
    assert evidence.reported_version == CURSOR_SUPPORTED_VENDOR_VERSION
    assert evidence.version_environment_was_exact_and_credential_free
    assert evidence.evidence_is_only_outer_wrapper_identity
    assert evidence.evidence_is_not_dependency_closure
    assert evidence.evidence_is_not_provider_bridge_validation
    assert evidence.evidence_is_not_linux_kvm_qualification

    symlink = tmp_path / 'cursor-agent-symlink'
    symlink.symlink_to(executable)
    with pytest.raises(CursorProtocolFixtureError) as captured:
        capture_cursor_vendor_identity(symlink)
    assert captured.value.code == CursorProtocolFixtureFailureCode.VENDOR_EXECUTABLE_REJECTED


def test_exact_installed_cursor_wrapper_identity_is_measurable_without_provider_call() -> None:
    executable = Path.home() / '.local/share/cursor-agent/versions/2026.07.09-a3815c0/cursor-agent'
    if not executable.exists():
        pytest.skip('the pinned local Cursor Agent wrapper is not installed')

    evidence = capture_cursor_vendor_identity(executable)

    assert evidence.executable_sha256 == ('eed61c5224668c9236334c4c68936a16aecc37374b592f59e31eb50433817831')
    assert evidence.executable_byte_count == 1_074
    assert evidence.reported_version == '2026.07.09-a3815c0'
