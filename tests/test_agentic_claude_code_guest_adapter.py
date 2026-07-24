from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path
from typing import Literal, cast

import pytest

from tests.test_clinicaltrials_execution_scoring import _case, _submission
from vaxreplay.agentic.claude_code_guest_adapter import (
    CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL,
    CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
    CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
    ClaudeCodeGuestAdapterConfig,
    ClaudeCodeGuestAdapterError,
    ClaudeCodeGuestAdapterFailureCode,
    capture_claude_code_vendor_identity,
    claude_code_vendor_argv_template,
    run_claude_code_guest_adapter,
)
from vaxreplay.agentic.gateway import AgenticGatewayUsage, AgenticModelMessage, AgenticModelResponse
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

_RUN_ID = '1' * 32
_MODEL_SELECTOR = 'organizer-model-route-1'
_RESOLVED_MODEL = 'resolved-organizer-model'


class _FakeClient:
    def __init__(
        self,
        *,
        files: dict[str, bytes],
        model_outputs: tuple[str, ...],
        stop_reasons: tuple[Literal['completed', 'max_output_tokens', 'refusal', 'provider_error'], ...] | None = None,
    ) -> None:
        self.files = files
        self.model_outputs = model_outputs
        self.stop_reasons = stop_reasons or cast(
            tuple[Literal['completed', 'max_output_tokens', 'refusal', 'provider_error'], ...],
            ('completed',) * len(model_outputs),
        )
        self.list_calls: list[tuple[int, int]] = []
        self.model_calls: list[tuple[tuple[AgenticModelMessage, ...], int, str | None]] = []
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
        selected = self.files[path][offset : offset + limit]
        return ReadWorkspaceResult(
            content_base64=base64.b64encode(selected).decode('ascii'),
            offset=offset,
            byte_count=len(selected),
            eof=offset + len(selected) == len(self.files[path]),
        )

    def model_generate(
        self,
        *,
        messages: tuple[AgenticModelMessage, ...],
        max_output_tokens: int,
        response_schema_sha256: str | None = None,
    ) -> AgenticModelResponse:
        index = len(self.model_calls)
        self.model_calls.append((messages, max_output_tokens, response_schema_sha256))
        return AgenticModelResponse(
            run_id=_RUN_ID,
            call_index=index,
            resolved_model_id=_RESOLVED_MODEL,
            content=self.model_outputs[index],
            stop_reason=self.stop_reasons[index],
            usage=AgenticGatewayUsage(input_tokens=37, output_tokens=11, reasoning_tokens=3),
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


def _assistant_decision(submission: ExecutionSubmission) -> str:
    return canonical_json_bytes(
        {
            'kind': 'assistant_text',
            'text': canonical_json_bytes(submission).decode('utf-8'),
        }
    ).decode('utf-8')


def _fake_vendor_body(
    *,
    agentic_loop: bool = False,
    wrong_api_key: bool = False,
    wrong_model: bool = False,
    unknown_tool: bool = False,
    permission_denial: bool = False,
    multiple_models: bool = False,
) -> bytes:
    script = f"""#!{sys.executable}
import json
import os
import sys
import urllib.request

args = sys.argv[1:]
required = ("--bare", "--safe-mode", "-p", "--no-session-persistence", "--no-chrome",
            "--disable-slash-commands", "--strict-mcp-config")
if any(item not in args for item in required):
    raise SystemExit(80)
if args[args.index("--tools") + 1] != "Read" or args[args.index("--allowedTools") + 1] != "Read":
    raise SystemExit(81)
if args[args.index("--mcp-config") + 1] != '{{"mcpServers":{{}}}}':
    raise SystemExit(82)
if os.environ.get("ANTHROPIC_API_KEY") != {CLAUDE_CODE_LOOPBACK_API_KEY_SENTINEL!r}:
    raise SystemExit(83)
if os.environ.get("ANTHROPIC_BASE_URL") != "http://127.0.0.1:43124":
    raise SystemExit(84)
for forbidden in ("ANTHROPIC_AUTH_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    if forbidden in os.environ:
        raise SystemExit(85)
managed = os.environ["CLAUDE_CODE_MANAGED_SETTINGS_PATH"]
if open(managed, "rb").read() != b"{{}}\\n":
    raise SystemExit(86)
model = args[args.index("--model") + 1]
if {wrong_model!r}:
    model = "wrong-route"
schema = json.loads(args[args.index("--json-schema") + 1])
prompt = sys.stdin.buffer.read().decode("utf-8")
tool = {{
    "name": ("FutureTool" if {unknown_tool!r} else "Read"),
    "description": "Read a file from the sealed workspace",
    "input_schema": {{
        "type": "object",
        "properties": {{"file_path": {{"type": "string"}}}},
        "required": ["file_path"],
        "additionalProperties": False,
    }},
}}
request = {{
    "model": model,
    "max_tokens": 4096,
    "messages": [{{"role": "user", "content": [{{"type": "text", "text": prompt}}]}}],
    "system": [{{"type": "text", "text": "fake pinned Claude transport"}}],
    "tools": [tool],
    "tool_choice": {{"type": "auto", "disable_parallel_tool_use": True}},
    "metadata": {{"user_id": "sealed-fixture"}},
    "stream": True,
    "output_config": {{"format": {{"type": "json_schema", "schema": schema}}}},
}}
headers = {{
    "Content-Type": "application/json",
    "x-api-key": ("wrong" if {wrong_api_key!r} else os.environ["ANTHROPIC_API_KEY"]),
    "anthropic-version": "2023-06-01",
}}

def exchange(payload):
    http_request = urllib.request.Request(
        os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=5) as response:
        stream = response.read().decode("utf-8")
    block = None
    model_id = None
    for line in stream.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "message_start":
            model_id = event["message"]["model"]
        elif event["type"] == "content_block_start":
            block = event["content_block"]
        elif event["type"] == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            else:
                block["input"] = json.loads(delta["partial_json"])
    return block, model_id

block, resolved_model = exchange(request)
turns = 1
if {agentic_loop!r}:
    if block["type"] != "tool_use" or block["name"] != "Read":
        raise SystemExit(87)
    if block["input"] != {{"file_path": "TASK.md"}}:
        raise SystemExit(88)
    tool_output = open("TASK.md", encoding="utf-8").read()
    request["messages"].extend((
        {{"role": "assistant", "content": [block]}},
        {{"role": "user", "content": [{{
            "type": "tool_result", "tool_use_id": block["id"], "content": tool_output,
        }}]}},
    ))
    block, second_model = exchange(request)
    if second_model != resolved_model:
        raise SystemExit(89)
    turns = 2
if block["type"] != "text":
    raise SystemExit(90)
submission = json.loads(block["text"])
model_usage = {{resolved_model: {{"inputTokens": 1, "outputTokens": 1}}}}
if {multiple_models!r}:
    model_usage["fallback-model"] = {{"inputTokens": 1}}
wrapper = {{
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 1,
    "duration_api_ms": 1,
    "num_turns": turns,
    "result": block["text"],
    "structured_output": submission,
    "stop_reason": "end_turn",
    "session_id": "00000000-0000-4000-8000-000000000001",
    "total_cost_usd": 0,
    "usage": {{"input_tokens": 1, "output_tokens": 1}},
    "modelUsage": model_usage,
    "permission_denials": ([{{"tool_name": "Read"}}] if {permission_denial!r} else []),
    "terminal_reason": "success",
    "fast_mode_state": "off",
    "uuid": "00000000-0000-4000-8000-000000000002",
    "errors": None,
    "model": resolved_model,
}}
sys.stdout.write(json.dumps(wrapper, separators=(",", ":")))
"""
    return script.encode('utf-8')


def _materials(
    root: Path,
    *,
    vendor_body: bytes,
) -> tuple[
    ClaudeCodeGuestAdapterConfig,
    HeadlessGuestAdapterConfig,
    SubmittedHarnessManifest,
]:
    vendor_path = root.joinpath(*Path(CLAUDE_CODE_VENDOR_EXECUTABLE_PATH).parts[1:])
    vendor_path.parent.mkdir(parents=True)
    vendor_path.write_bytes(vendor_body)
    vendor_path.chmod(0o500)
    vendor_sha = hashlib.sha256(vendor_body).hexdigest()
    headless = HeadlessGuestAdapterConfig(
        family=HarnessFamily.CLAUDE_CODE,
        invocation_protocol=HeadlessInvocationProtocol.CLAUDE_PRINT,
        adapter_executable_sha256='1' * 64,
        vendor_executable_path=CLAUDE_CODE_VENDOR_EXECUTABLE_PATH,
        vendor_executable_sha256=vendor_sha,
        complete_dependency_closure_sha256='3' * 64,
        vendor_reported_version=CLAUDE_CODE_SUPPORTED_VENDOR_VERSION,
        vendor_version_output_sha256='4' * 64,
        vendor_config_template_sha256='5' * 64,
        vendor_argv_template=claude_code_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_JSON_STDOUT,
        local_shell_enabled=False,
        adapter_implementation_checked_in=True,
        provider_shim_implementation_checked_in=True,
        workspace_materialization_bridge_implementation_checked_in=True,
    )
    headless_sha = headless_guest_adapter_config_sha256(headless)
    manifest = SubmittedHarnessManifest(
        harness_id='claude-code-sealed-development-adapter',
        harness_version='dev-v0.1',
        family=HarnessFamily.CLAUDE_CODE,
        execution_mode=HarnessExecutionMode.SUBMITTED_GUEST_AGENT,
        runtime_support=HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
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
        display_name='Claude Code sealed development adapter',
        submitter='fixture',
    )
    runtime = ClaudeCodeGuestAdapterConfig(
        headless_adapter_config_sha256=headless_sha,
        submitted_harness_manifest_sha256=submitted_harness_manifest_sha256(manifest),
        vendor_executable_sha256=vendor_sha,
        vendor_executable_byte_count=len(vendor_body),
    )
    return runtime, headless, manifest


def test_fake_claude_process_runs_through_messages_shim_and_submits_once(
    tmp_path: Path,
) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    files = {
        'TASK.md': b'Use the frozen sources.\n',
        'sources/a.txt': b'cutoff evidence alpha\n',
    }
    client = _FakeClient(files=files, model_outputs=(_assistant_decision(submission),))
    os.environ['ANTHROPIC_AUTH_TOKEN'] = 'must-not-be-inherited'
    os.environ['HTTPS_PROXY'] = 'http://must-not-be-inherited.invalid'
    try:
        receipt = run_claude_code_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    finally:
        os.environ.pop('ANTHROPIC_AUTH_TOKEN')
        os.environ.pop('HTTPS_PROXY')

    assert client.submissions == [submission]
    assert len(client.model_calls) == 1
    assert client.model_calls[0][2] is None
    assert receipt.workspace_file_count == len(files)
    assert receipt.workspace_byte_count == sum(map(len, files.values()))
    assert [item.response_content_type for item in receipt.shim_exchanges] == ['text']
    assert receipt.submit_result.submission_sha256 == receipt.submission_sha256
    assert receipt.local_vendor_tool_surface_was_exactly_read_only
    assert receipt.actual_pinned_macos_claude_end_to_end_validated is False
    assert receipt.actual_pinned_linux_claude_end_to_end_validated is False
    assert receipt.linux_kvm_qualified is False
    assert (tmp_path / 'run/vaxreplay/workspace/sources/a.txt').stat().st_mode & 0o777 == 0o400


def test_fake_claude_read_tool_round_trips_file_before_submission(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(
        tmp_path,
        vendor_body=_fake_vendor_body(agentic_loop=True),
    )
    task_body = b'Only pre-cutoff broker evidence is available.\n'
    client = _FakeClient(
        files={'TASK.md': task_body},
        model_outputs=(
            canonical_json_bytes(
                {'kind': 'tool_call', 'tool_name': 'Read', 'payload': {'file_path': 'TASK.md'}}
            ).decode('utf-8'),
            _assistant_decision(submission),
        ),
    )

    receipt = run_claude_code_guest_adapter(
        client,
        task_invocation=invocation,
        organizer_model_selector=_MODEL_SELECTOR,
        config=runtime,
        headless_config=headless,
        submitted_manifest=manifest,
        _guest_root=tmp_path,
    )

    assert [item.response_content_type for item in receipt.shim_exchanges] == [
        'tool_use',
        'text',
    ]
    assert receipt.shim_exchanges[1].prior_tool_use_count == 1
    assert receipt.shim_exchanges[1].prior_tool_result_count == 1
    second_request = __import__('json').loads(client.model_calls[1][0][1].content)
    assert second_request['messages'][-1]['content'][0]['content'] == task_body.decode('utf-8')
    assert client.submissions == [submission]


@pytest.mark.parametrize(
    ('vendor_body', 'expected_code', 'expected_model_calls'),
    (
        (
            _fake_vendor_body(wrong_api_key=True),
            ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED,
            0,
        ),
        (
            _fake_vendor_body(wrong_model=True),
            ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED,
            0,
        ),
        (
            _fake_vendor_body(unknown_tool=True),
            ClaudeCodeGuestAdapterFailureCode.SHIM_REJECTED,
            0,
        ),
        (
            _fake_vendor_body(permission_denial=True),
            ClaudeCodeGuestAdapterFailureCode.OUTPUT_REJECTED,
            1,
        ),
        (
            _fake_vendor_body(multiple_models=True),
            ClaudeCodeGuestAdapterFailureCode.OUTPUT_REJECTED,
            1,
        ),
    ),
)
def test_adversarial_transport_and_result_variants_fail_closed(
    tmp_path: Path,
    vendor_body: bytes,
    expected_code: ClaudeCodeGuestAdapterFailureCode,
    expected_model_calls: int,
) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=vendor_body)
    client = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )

    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        run_claude_code_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == expected_code
    assert len(client.model_calls) == expected_model_calls
    assert not client.submissions


def test_workspace_escape_and_vendor_substitution_fail_before_model_call(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    escaping = _FakeClient(
        files={'../future.txt': b'leak'},
        model_outputs=(_assistant_decision(submission),),
    )
    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        run_claude_code_guest_adapter(
            escaping,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    assert captured.value.code == ClaudeCodeGuestAdapterFailureCode.WORKSPACE_REJECTED
    assert not escaping.model_calls

    fresh_root = tmp_path / 'fresh'
    runtime, headless, manifest = _materials(fresh_root, vendor_body=_fake_vendor_body())
    vendor_path = fresh_root.joinpath(*Path(CLAUDE_CODE_VENDOR_EXECUTABLE_PATH).parts[1:])
    vendor_path.chmod(0o700)
    vendor_path.write_bytes(b'changed')
    vendor_path.chmod(0o500)
    substituted = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )
    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        run_claude_code_guest_adapter(
            substituted,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=fresh_root,
        )
    assert captured.value.code == ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED
    assert not substituted.list_calls


def test_binding_tamper_and_shell_like_model_selector_fail_closed(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    client = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )
    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        run_claude_code_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector='model;touch-pwned',
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    assert captured.value.code == ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED

    wrong_runtime = runtime.model_copy(update={'headless_adapter_config_sha256': 'f' * 64})
    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        run_claude_code_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=wrong_runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    assert captured.value.code == ClaudeCodeGuestAdapterFailureCode.BINDING_REJECTED
    assert not client.model_calls
    assert not client.submissions


def test_vendor_identity_capture_is_exact_and_rejects_symlink(tmp_path: Path) -> None:
    executable = tmp_path / 'claude-real-file'
    executable.write_text(
        f"""#!{sys.executable}
import os
import sys
if sys.argv[1:] != ["--version"]:
    raise SystemExit(2)
if "ANTHROPIC_API_KEY" in os.environ or "ANTHROPIC_AUTH_TOKEN" in os.environ:
    raise SystemExit(3)
print("2.1.195 (Claude Code)")
""",
        encoding='utf-8',
    )
    executable.chmod(0o500)

    evidence = capture_claude_code_vendor_identity(executable)

    assert evidence.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert evidence.executable_byte_count == executable.stat().st_size
    assert evidence.reported_version == CLAUDE_CODE_SUPPORTED_VENDOR_VERSION
    assert evidence.version_environment_was_exact_and_credential_free
    assert evidence.evidence_is_not_dependency_closure
    assert evidence.evidence_is_not_linux_kvm_qualification

    symlink = tmp_path / 'claude-symlink'
    symlink.symlink_to(executable)
    with pytest.raises(ClaudeCodeGuestAdapterError) as captured:
        capture_claude_code_vendor_identity(symlink)
    assert captured.value.code == ClaudeCodeGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED


def test_exact_installed_claude_identity_is_measurable_without_provider_call() -> None:
    executable = Path('/opt/homebrew/Caskroom/claude-code/2.1.195/claude')
    if not executable.exists():
        pytest.skip('the pinned local Claude Code binary is not installed')

    evidence = capture_claude_code_vendor_identity(executable)

    assert evidence.executable_sha256 == ('8b45adad93f336ab95f33e714494b19fd3377a494eb05c122c8677bc895876ad')
    assert evidence.executable_byte_count == 224_682_640
    assert evidence.reported_version == '2.1.195 (Claude Code)'
