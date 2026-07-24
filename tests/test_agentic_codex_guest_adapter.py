from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import pytest

from tests.test_clinicaltrials_execution_scoring import _case, _submission
from vaxreplay.agentic.codex_guest_adapter import (
    CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
    CODEX_VENDOR_EXECUTABLE_PATH,
    CodexGuestAdapterConfig,
    CodexGuestAdapterError,
    CodexGuestAdapterFailureCode,
    capture_codex_vendor_identity,
    codex_vendor_argv_template,
    run_codex_guest_adapter,
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
    SubmittedHarnessError,
    SubmittedHarnessInterface,
    SubmittedHarnessManifest,
    require_submitted_harness_binding,
    submitted_harness_manifest_sha256,
)
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_task import ExecutionSubmission

_RUN_ID = '1' * 32
_MODEL_SELECTOR = 'organizer-model-route-1'


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
        self.stop_reasons = stop_reasons or ('completed',) * len(model_outputs)
        self.list_calls: list[tuple[int, int]] = []
        self.read_calls: list[tuple[str, int, int]] = []
        self.model_calls: list[tuple[tuple[AgenticModelMessage, ...], int, str | None]] = []
        self.submissions: list[ExecutionSubmission] = []

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
            resolved_model_id='resolved-organizer-model',
            content=self.model_outputs[index],
            stop_reason=self.stop_reasons[index],
            usage=AgenticGatewayUsage(input_tokens=37, output_tokens=11, reasoning_tokens=3),
        )

    def submit(self, submission: ExecutionSubmission) -> SubmitResult:
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


def _fake_vendor_body(
    *,
    authorization: bool = False,
    agentic_loop: bool = False,
    host_header: str | None = None,
    unknown_tool: bool = False,
) -> bytes:
    script = f"""#!{sys.executable}
import json
import os
import sys
import urllib.request

for name in os.environ:
    upper = name.upper()
    forbidden = ("API_KEY", "AUTH", "BEARER", "CREDENTIAL", "PASSWORD", "PROXY", "SECRET", "TOKEN")
    if any(part in upper for part in forbidden):
        raise SystemExit(83)
args = sys.argv[1:]
if args[0] != "exec" or "--ephemeral" not in args or "--ignore-user-config" not in args:
    raise SystemExit(84)
model = args[args.index("--model") + 1]
output = args[args.index("--output-last-message") + 1]
prompt = sys.stdin.buffer.read()
request = {{
    "model": model,
    "instructions": "fake pinned Codex transport",
    "input": [{{
        "type": "message",
        "role": "user",
        "content": [{{"type": "input_text", "text": prompt.decode("utf-8")}}],
    }}],
    "tools": ([{{
        "type": "function",
        "name": "exec_command",
        "description": "read the sealed workspace",
        "strict": False,
        "parameters": {{
            "type": "object",
            "properties": {{"cmd": {{"type": "string"}}}},
            "required": ["cmd"],
            "additionalProperties": False,
        }},
    }}] if {agentic_loop!r} else []) + ([{{
        "type": "function",
        "name": "future_unsealed_tool",
        "description": "must fail closed",
        "strict": False,
        "parameters": {{"type": "object", "properties": {{}}}},
    }}] if {unknown_tool!r} else []),
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "reasoning": None,
    "store": False,
    "stream": True,
    "stream_options": None,
    "include": [],
    "service_tier": None,
    "prompt_cache_key": None,
    "text": None,
    "client_metadata": None,
}}
headers = {{"Content-Type": "application/json"}}
if {authorization!r}:
    headers["Authorization"] = "Bearer forbidden"
if {host_header!r} is not None:
    headers["Host"] = {host_header!r}
def exchange(payload):
    http_request = urllib.request.Request(
        "http://127.0.0.1:43123/v1/responses",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=5) as response:
        stream = response.read().decode("utf-8")
    for line in stream.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "response.output_item.done":
                return event["item"]
    raise SystemExit(86)

item = exchange(request)
if {agentic_loop!r}:
    if item["type"] != "function_call" or item["name"] != "exec_command":
        raise SystemExit(87)
    arguments = json.loads(item["arguments"])
    if arguments != {{"cmd": "cat TASK.md"}}:
        raise SystemExit(88)
    workspace = args[args.index("--cd") + 1]
    tool_output = open(os.path.join(workspace, "TASK.md"), encoding="utf-8").read()
    request["input"].extend((
        item,
        {{"type": "function_call_output", "call_id": item["call_id"], "output": tool_output}},
    ))
    item = exchange(request)
text = item["content"][0]["text"] if item["type"] == "message" else None
if not text:
    raise SystemExit(85)
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(descriptor, text.encode("utf-8"))
finally:
    os.close(descriptor)
"""
    return script.encode('utf-8')


def _materials(
    root: Path,
    *,
    vendor_body: bytes,
) -> tuple[CodexGuestAdapterConfig, HeadlessGuestAdapterConfig, SubmittedHarnessManifest]:
    vendor_path = root.joinpath(*Path(CODEX_VENDOR_EXECUTABLE_PATH).parts[1:])
    vendor_path.parent.mkdir(parents=True)
    vendor_path.write_bytes(vendor_body)
    vendor_path.chmod(0o500)
    vendor_sha = hashlib.sha256(vendor_body).hexdigest()
    headless = HeadlessGuestAdapterConfig(
        family=HarnessFamily.CODEX,
        invocation_protocol=HeadlessInvocationProtocol.CODEX_EXEC,
        adapter_executable_sha256='1' * 64,
        vendor_executable_path=CODEX_VENDOR_EXECUTABLE_PATH,
        vendor_executable_sha256=vendor_sha,
        complete_dependency_closure_sha256='3' * 64,
        vendor_reported_version=CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION,
        vendor_version_output_sha256='4' * 64,
        vendor_config_template_sha256='5' * 64,
        vendor_argv_template=codex_vendor_argv_template(),
        response_channel=HeadlessResponseChannel.BOUNDED_OUTPUT_FILE,
        local_shell_enabled=True,
        adapter_implementation_checked_in=True,
        provider_shim_implementation_checked_in=True,
        workspace_materialization_bridge_implementation_checked_in=True,
    )
    headless_sha = headless_guest_adapter_config_sha256(headless)
    manifest = SubmittedHarnessManifest(
        harness_id='codex-sealed-development-adapter',
        harness_version='dev-v0.1',
        family=HarnessFamily.CODEX,
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
            guest_local_shell_allowed=True,
        ),
        display_name='Codex sealed development adapter',
        submitter='fixture',
    )
    runtime = CodexGuestAdapterConfig(
        headless_adapter_config_sha256=headless_sha,
        submitted_harness_manifest_sha256=submitted_harness_manifest_sha256(manifest),
        vendor_executable_sha256=vendor_sha,
        vendor_executable_byte_count=len(vendor_body),
    )
    return runtime, headless, manifest


def _assistant_decision(submission: ExecutionSubmission) -> str:
    return canonical_json_bytes(
        {
            'kind': 'assistant_text',
            'text': canonical_json_bytes(submission).decode('utf-8'),
        }
    ).decode('utf-8')


def test_fake_codex_process_runs_through_loopback_shim_and_brokered_snapshot(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    files = {
        'TASK.md': b'Use the frozen sources.\n',
        'sources/a.txt': b'cutoff evidence alpha\n',
        'sources/b.txt': b'cutoff evidence beta\n',
    }
    client = _FakeClient(files=files, model_outputs=(_assistant_decision(submission),))
    os.environ['OPENAI_API_KEY'] = 'must-not-be-inherited'
    try:
        receipt = run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    finally:
        os.environ.pop('OPENAI_API_KEY')

    assert client.submissions == [submission]
    assert len(client.model_calls) == 1
    assert client.model_calls[0][2] is None
    assert receipt.workspace_file_count == len(files)
    assert receipt.workspace_byte_count == sum(map(len, files.values()))
    assert len(receipt.shim_exchanges) == 1
    assert receipt.shim_exchanges[0].response_item_type == 'message'
    assert receipt.submit_result.submission_sha256 == receipt.submission_sha256
    assert receipt.actual_pinned_linux_codex_end_to_end_validated is False
    assert receipt.linux_kvm_qualified is False
    workspace = tmp_path / 'run/vaxreplay/workspace'
    assert (workspace / 'sources/a.txt').read_bytes() == files['sources/a.txt']
    assert stat_mode(workspace / 'sources/a.txt') == 0o400


def test_fake_agentic_vendor_round_trips_local_tool_result_before_submission(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(
        tmp_path,
        vendor_body=_fake_vendor_body(agentic_loop=True),
    )
    task_body = b'Only pre-cutoff broker evidence is available.\n'
    first_decision = canonical_json_bytes(
        {'kind': 'tool_call', 'tool_name': 'exec_command', 'payload': {'cmd': 'cat TASK.md'}}
    ).decode('utf-8')
    client = _FakeClient(
        files={'TASK.md': task_body},
        model_outputs=(first_decision, _assistant_decision(submission)),
    )

    receipt = run_codex_guest_adapter(
        client,
        task_invocation=invocation,
        organizer_model_selector=_MODEL_SELECTOR,
        config=runtime,
        headless_config=headless,
        submitted_manifest=manifest,
        _guest_root=tmp_path,
    )

    assert [item.response_item_type for item in receipt.shim_exchanges] == [
        'function_call',
        'message',
    ]
    assert receipt.shim_exchanges[0].prior_vendor_tool_output_item_count == 0
    assert receipt.shim_exchanges[1].prior_vendor_tool_call_item_count == 1
    assert receipt.shim_exchanges[1].prior_vendor_tool_output_item_count == 1
    second_forwarded_request = json.loads(client.model_calls[1][0][1].content)
    assert second_forwarded_request['input'][-1] == {
        'type': 'function_call_output',
        'call_id': 'call_vaxreplay_00000000',
        'output': task_body.decode('utf-8'),
    }
    assert client.submissions == [submission]


def test_official_runtime_admission_still_rejects_development_codex_adapter(tmp_path: Path) -> None:
    _runtime, _headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    from vaxreplay.agentic.submitted_harness import make_agentic_harness_identity

    identity = make_agentic_harness_identity(
        manifest=manifest,
        requested_model_id='logical-model',
        adapter_id='direct-openai',
    )
    with pytest.raises(SubmittedHarnessError, match='not runtime-integrated or qualified'):
        require_submitted_harness_binding(
            manifest=manifest,
            identity=identity,
            worker_harness_sha256=manifest.harness_image_sha256,
            worker_harness_byte_count=manifest.harness_image_byte_count,
            logical_model_id='logical-model',
            adapter_id='direct-openai',
        )


def test_workspace_path_escape_fails_before_vendor_launch(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    client = _FakeClient(files={'../future.txt': b'leak'}, model_outputs=(_assistant_decision(submission),))

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == CodexGuestAdapterFailureCode.WORKSPACE_REJECTED
    assert not client.model_calls
    assert not client.submissions


def test_vendor_hash_substitution_fails_before_workspace_rpc(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    vendor_path = tmp_path.joinpath(*Path(CODEX_VENDOR_EXECUTABLE_PATH).parts[1:])
    vendor_path.chmod(0o700)
    vendor_path.write_bytes(b'changed')
    vendor_path.chmod(0o500)
    client = _FakeClient(files={'TASK.md': b'x'}, model_outputs=(_assistant_decision(submission),))

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED
    assert not client.list_calls


def test_loopback_shim_rejects_authorization_header_and_never_submits(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(
        tmp_path,
        vendor_body=_fake_vendor_body(authorization=True),
    )
    client = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == CodexGuestAdapterFailureCode.SHIM_REJECTED
    assert not client.model_calls
    assert not client.submissions


def test_loopback_shim_rejects_noncanonical_host_and_never_submits(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(
        tmp_path,
        vendor_body=_fake_vendor_body(host_header='localhost:43123'),
    )
    client = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == CodexGuestAdapterFailureCode.SHIM_REJECTED
    assert not client.model_calls
    assert not client.submissions


def test_loopback_shim_rejects_unknown_vendor_tool_before_model_call(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(
        tmp_path,
        vendor_body=_fake_vendor_body(unknown_tool=True),
    )
    client = _FakeClient(
        files={'TASK.md': b'x'},
        model_outputs=(_assistant_decision(submission),),
    )

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )

    assert captured.value.code == CodexGuestAdapterFailureCode.SHIM_REJECTED
    assert not client.model_calls
    assert not client.submissions


def test_tampered_cross_binding_and_shell_like_model_selector_fail_closed(tmp_path: Path) -> None:
    invocation, submission = _invocation_and_submission()
    runtime, headless, manifest = _materials(tmp_path, vendor_body=_fake_vendor_body())
    client = _FakeClient(files={'TASK.md': b'x'}, model_outputs=(_assistant_decision(submission),))

    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector='model;touch-pwned',
            config=runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    assert captured.value.code == CodexGuestAdapterFailureCode.BINDING_REJECTED

    wrong_runtime = runtime.model_copy(update={'headless_adapter_config_sha256': 'f' * 64})
    with pytest.raises(CodexGuestAdapterError) as captured:
        run_codex_guest_adapter(
            client,
            task_invocation=invocation,
            organizer_model_selector=_MODEL_SELECTOR,
            config=wrong_runtime,
            headless_config=headless,
            submitted_manifest=manifest,
            _guest_root=tmp_path,
        )
    assert captured.value.code == CodexGuestAdapterFailureCode.BINDING_REJECTED


def test_vendor_identity_capture_uses_regular_file_and_credential_free_version_process(
    tmp_path: Path,
) -> None:
    executable = tmp_path / 'codex-real-file'
    executable.write_text(
        f"""#!{sys.executable}
import os
import sys
if sys.argv[1:] != ["--version"]:
    raise SystemExit(2)
if any("API_KEY" in name or "TOKEN" in name or "PROXY" in name for name in os.environ):
    raise SystemExit(3)
print("codex-cli 0.144.3")
""",
        encoding='utf-8',
    )
    executable.chmod(0o500)

    evidence = capture_codex_vendor_identity(executable)

    assert evidence.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert evidence.executable_byte_count == executable.stat().st_size
    assert evidence.reported_version == CODEX_GUEST_ADAPTER_SUPPORTED_VENDOR_VERSION
    assert evidence.version_stderr_bytes == 0
    assert evidence.evidence_is_not_dependency_closure
    assert evidence.evidence_is_not_linux_kvm_qualification

    symlink = tmp_path / 'codex-symlink'
    symlink.symlink_to(executable)
    with pytest.raises(CodexGuestAdapterError) as captured:
        capture_codex_vendor_identity(symlink)
    assert captured.value.code == CodexGuestAdapterFailureCode.VENDOR_EXECUTABLE_REJECTED


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
