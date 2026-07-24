"""Development-only execution of one verified challenge through a local CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Submission
from vaxreplay.reference_harness.commands import (
    HarnessCommand,
    build_harness_command,
    make_openai_strict_json_schema,
)
from vaxreplay.reference_harness.schema import (
    CursorEventKindObservation,
    CursorParseConsistencyFlags,
    CursorParseFailureInventory,
    RawResponseSource,
    ReferenceHarnessFailure,
    ReferenceHarnessFailureCode,
    ReferenceHarnessLimits,
    ReferenceHarnessName,
    ReferenceHarnessReceipt,
    ReferenceHarnessRuntimeIdentity,
)
from vaxreplay.runner._process import BoundedProcessResult, run_bounded_process
from vaxreplay.runner.challenge import challenge_envelope_sha256
from vaxreplay.runner.schema import ChallengeEnvelope

_MAX_ENVELOPE_BYTES = 256 * 1024 * 1024
_VERSION_WALL_SECONDS = 10
_VERSION_OUTPUT_BYTES = 4_096
_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()
_PROMPT_PREAMBLE = 'VAXREPLAY CLI TRANSCRIPT v0.1\n'
_CURSOR_OUTPUT_PREAMBLE = (
    'BEGIN CURSOR OUTPUT CONTRACT\n'
    'Return exactly one JSON object that validates against the JSON Schema below. '
    'Do not use Markdown fences or add commentary. Do not call tools.\n'
)
_CURSOR_EVENT_SUBTYPES: dict[str, frozenset[str | None]] = {
    'system': frozenset({'init'}),
    'user': frozenset({None}),
    'thinking': frozenset({'delta', 'completed'}),
    'assistant': frozenset({None}),
    'tool_call': frozenset({'started', 'completed'}),
    'result': frozenset({'success'}),
}
_CURSOR_EVENT_TOKEN_PATTERN = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
_CURSOR_RUNTIME_TREE_SCHEMA_VERSION = 'vaxreplay.cursor-runtime-regular-file-tree.v1'
_CURSOR_RUNTIME_CHUNK_SCHEMA_VERSION = 'vaxreplay.cursor-runtime-chunk-manifest.v1'
_MAX_CURSOR_RUNTIME_FILES = 20_000
_MAX_CURSOR_RUNTIME_BYTES = 2 * 1024 * 1024 * 1024


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        wall_seconds: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        on_abort: Callable[[], None],
        env: Mapping[str, str] | None = None,
    ) -> BoundedProcessResult: ...


class ReferenceHarnessInputError(ValueError):
    """Raised when an envelope is not canonical or does not match its expected hash."""


class HarnessVersionError(RuntimeError):
    """Raised when the selected CLI cannot report a bounded, usable version."""


@dataclass(frozen=True)
class VerifiedChallengeEnvelope:
    envelope: ChallengeEnvelope
    envelope_sha256: str


@dataclass(frozen=True)
class _ParsedResponse:
    submission: Submission | None
    failure: ReferenceHarnessFailure | None
    resolved_model: str | None = None
    additional_reported_models: tuple[str, ...] = ()
    cursor_parse_failure_inventory: CursorParseFailureInventory | None = None


@dataclass(frozen=True)
class _ExecutableIdentity:
    invocation_path: str
    sha256: str | None
    runtime_identity: ReferenceHarnessRuntimeIdentity | None = None


@dataclass(frozen=True)
class _RuntimeFileIdentity:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _CursorParsedEvent:
    line: int
    event: dict[str, object]


@dataclass(frozen=True)
class _CursorStreamMetadata:
    resolved_model: str | None


class _CursorStreamValidationError(ValueError):
    """Raised only with fixed, content-free Cursor stream validation details."""


def load_verified_challenge_envelope(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> VerifiedChallengeEnvelope:
    """Load one canonical envelope and optionally bind it to a challenge-manifest hash."""

    content = _read_regular_file(path, _MAX_ENVELOPE_BYTES)
    try:
        envelope = ChallengeEnvelope.model_validate_json(content)
    except ValueError as error:
        raise ReferenceHarnessInputError(f'invalid challenge envelope: {error}') from error
    canonical = canonical_json_bytes(envelope)
    if content != canonical:
        raise ReferenceHarnessInputError('challenge envelope must use canonical JSON encoding')
    return verify_challenge_envelope(envelope, expected_sha256=expected_sha256)


def verify_challenge_envelope(
    envelope: ChallengeEnvelope,
    *,
    expected_sha256: str | None = None,
) -> VerifiedChallengeEnvelope:
    envelope_sha256 = challenge_envelope_sha256(envelope)
    if expected_sha256 is not None and envelope_sha256 != expected_sha256:
        raise ReferenceHarnessInputError('challenge envelope hash does not match the expected hash')
    return VerifiedChallengeEnvelope(envelope=envelope, envelope_sha256=envelope_sha256)


def render_challenge_prompt(envelope: ChallengeEnvelope) -> bytes:
    """Losslessly render the ordered system/user pair for single-prompt CLI interfaces."""

    system_message, user_message = envelope.messages
    rendered = (
        f'{_PROMPT_PREAMBLE}'
        f'BEGIN SYSTEM MESSAGE\n{system_message.content}\nEND SYSTEM MESSAGE\n'
        f'BEGIN USER MESSAGE\n{user_message.content}\nEND USER MESSAGE\n'
    )
    return rendered.encode('utf-8')


def render_cursor_challenge_prompt(envelope: ChallengeEnvelope, submission_schema_json: str) -> bytes:
    """Render the canonical transcript plus Cursor's prompt-level output contract.

    Cursor Agent has no response-schema CLI option.  This prompt is not a substitute for provider
    structured output; the terminal result is still validated locally and fails closed.
    """

    rendered = render_challenge_prompt(envelope)
    contract = f'{_CURSOR_OUTPUT_PREAMBLE}{submission_schema_json}\nEND CURSOR OUTPUT CONTRACT\n'
    return rendered + contract.encode('utf-8')


def resolve_harness_version(
    executable: str,
    *,
    process_runner: ProcessRunner = run_bounded_process,
    env: Mapping[str, str] | None = None,
) -> str:
    try:
        result = process_runner(
            (executable, '--version'),
            input_bytes=b'',
            wall_seconds=_VERSION_WALL_SECONDS,
            max_stdout_bytes=_VERSION_OUTPUT_BYTES,
            max_stderr_bytes=_VERSION_OUTPUT_BYTES,
            on_abort=lambda: None,
            env=env,
        )
    except OSError as error:
        raise HarnessVersionError('could not launch the harness version command') from error
    if result.termination != 'exited' or result.exit_code != 0:
        raise HarnessVersionError('harness version command did not exit successfully')
    version_bytes = result.stdout.strip() or result.stderr.strip()
    try:
        version = version_bytes.decode('utf-8').splitlines()[0].strip()
    except (UnicodeDecodeError, IndexError) as error:
        raise HarnessVersionError('harness version output is not valid UTF-8') from error
    if not version or len(version) > 256 or '\x00' in version:
        raise HarnessVersionError('harness version output is missing or too long')
    return version


def run_reference_harness(
    verified: VerifiedChallengeEnvelope,
    *,
    harness: ReferenceHarnessName,
    requested_model: str | None,
    executable: str | None = None,
    limits: ReferenceHarnessLimits | None = None,
    claude_max_budget_usd: str = '1.00',
    env: Mapping[str, str] | None = None,
    process_runner: ProcessRunner = run_bounded_process,
    version_resolver: Callable[[str], str] | None = None,
) -> ReferenceHarnessReceipt:
    """Run one local CLI diagnostic and return a canonicalizable receipt.

    This function intentionally makes no sealed-execution claim.  It records the
    command in a redacted form while sending all challenge text through stdin.
    """

    selected_limits = limits or ReferenceHarnessLimits()
    default_executables = {
        ReferenceHarnessName.CODEX: 'codex',
        ReferenceHarnessName.CLAUDE: 'claude',
        ReferenceHarnessName.CURSOR: 'cursor-agent',
    }
    selected_executable = executable or default_executables[harness]
    _validate_identifier(selected_executable, 'executable')
    if requested_model is not None:
        _validate_identifier(requested_model, 'requested_model')
    budget = _normalize_budget(claude_max_budget_usd)
    submission_schema = Submission.model_json_schema()
    if harness == ReferenceHarnessName.CODEX:
        submission_schema = make_openai_strict_json_schema(submission_schema)
    submission_schema_bytes = canonical_json_bytes(submission_schema)
    submission_schema_json = submission_schema_bytes.decode('utf-8')
    prompt_bytes = (
        render_cursor_challenge_prompt(verified.envelope, submission_schema_json)
        if harness == ReferenceHarnessName.CURSOR
        else render_challenge_prompt(verified.envelope)
    )
    executable_identity = _resolve_executable_identity(selected_executable, harness=harness)
    selected_executable = executable_identity.invocation_path

    with tempfile.TemporaryDirectory(prefix='vaxreplay-reference-') as temporary_directory:
        temporary_root = Path(temporary_directory)
        work_dir = temporary_root / 'workspace'
        control_dir = temporary_root / 'control'
        work_dir.mkdir(mode=0o700)
        control_dir.mkdir(mode=0o700)
        schema_path = control_dir / 'submission.schema.json'
        response_path = control_dir / 'last-message.json'
        schema_path.write_bytes(submission_schema_bytes)
        command = build_harness_command(
            harness,
            executable=selected_executable,
            requested_model=requested_model,
            work_dir=work_dir,
            schema_path=schema_path,
            response_path=response_path,
            submission_schema_json=submission_schema_json,
            claude_max_budget_usd=budget,
        )
        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        try:
            harness_version = (
                version_resolver(selected_executable)
                if version_resolver is not None
                else resolve_harness_version(selected_executable, process_runner=process_runner, env=env)
            )
        except (HarnessVersionError, OSError, RuntimeError, ValueError):
            return _receipt(
                verified,
                harness=harness,
                harness_version='unresolved',
                harness_executable_sha256=executable_identity.sha256,
                harness_runtime_identity=executable_identity.runtime_identity,
                requested_model=requested_model,
                command=command,
                prompt_bytes=prompt_bytes,
                started_at=started_at,
                started_monotonic=started_monotonic,
                process_result=None,
                raw_response=b'',
                raw_response_source=RawResponseSource.NONE,
                raw_response_truncated=False,
                parsed=_failure(
                    ReferenceHarnessFailureCode.VERSION_UNAVAILABLE,
                    'The local CLI version could not be resolved; the model invocation was not started.',
                ),
            )

        try:
            process_result = process_runner(
                command.argv,
                input_bytes=prompt_bytes,
                wall_seconds=selected_limits.wall_seconds,
                max_stdout_bytes=(
                    selected_limits.max_cli_stdout_bytes
                    if harness == ReferenceHarnessName.CODEX
                    else selected_limits.max_response_bytes
                ),
                max_stderr_bytes=selected_limits.max_cli_stderr_bytes,
                on_abort=lambda: None,
                env=env,
            )
        except OSError:
            return _receipt(
                verified,
                harness=harness,
                harness_version=harness_version,
                harness_executable_sha256=executable_identity.sha256,
                harness_runtime_identity=executable_identity.runtime_identity,
                requested_model=requested_model,
                command=command,
                prompt_bytes=prompt_bytes,
                started_at=started_at,
                started_monotonic=started_monotonic,
                process_result=None,
                raw_response=b'',
                raw_response_source=RawResponseSource.NONE,
                raw_response_truncated=False,
                parsed=_failure(
                    ReferenceHarnessFailureCode.LAUNCH_ERROR,
                    'The local harness process could not be launched.',
                ),
            )
        except RuntimeError:
            return _receipt(
                verified,
                harness=harness,
                harness_version=harness_version,
                harness_executable_sha256=executable_identity.sha256,
                harness_runtime_identity=executable_identity.runtime_identity,
                requested_model=requested_model,
                command=command,
                prompt_bytes=prompt_bytes,
                started_at=started_at,
                started_monotonic=started_monotonic,
                process_result=None,
                raw_response=b'',
                raw_response_source=RawResponseSource.NONE,
                raw_response_truncated=False,
                parsed=_failure(
                    ReferenceHarnessFailureCode.EXECUTION_ERROR,
                    'The bounded local harness supervisor failed before it could return a process result.',
                ),
            )

        raw_response, raw_source, raw_truncated, response_failure = _capture_raw_response(
            harness=harness,
            command=command,
            process_result=process_result,
            maximum_bytes=selected_limits.max_response_bytes,
        )
        parsed = _process_or_response_failure(
            harness=harness,
            process_result=process_result,
            raw_response=raw_response,
            response_failure=response_failure,
            cursor_expected_prompt=prompt_bytes if harness == ReferenceHarnessName.CURSOR else None,
            cursor_expected_workspace=work_dir if harness == ReferenceHarnessName.CURSOR else None,
        )
        return _receipt(
            verified,
            harness=harness,
            harness_version=harness_version,
            harness_executable_sha256=executable_identity.sha256,
            harness_runtime_identity=executable_identity.runtime_identity,
            requested_model=requested_model,
            command=command,
            prompt_bytes=prompt_bytes,
            started_at=started_at,
            started_monotonic=started_monotonic,
            process_result=process_result,
            raw_response=raw_response,
            raw_response_source=raw_source,
            raw_response_truncated=raw_truncated,
            parsed=parsed,
        )


def canonical_receipt_bytes(receipt: ReferenceHarnessReceipt) -> bytes:
    return canonical_json_bytes(receipt)


def _capture_raw_response(
    *,
    harness: ReferenceHarnessName,
    command: HarnessCommand,
    process_result: BoundedProcessResult,
    maximum_bytes: int,
) -> tuple[bytes, RawResponseSource, bool, ReferenceHarnessFailure | None]:
    if harness in (ReferenceHarnessName.CLAUDE, ReferenceHarnessName.CURSOR):
        if not process_result.stdout:
            return b'', RawResponseSource.NONE, process_result.stdout_truncated, None
        return (
            process_result.stdout,
            command.raw_response_source,
            process_result.stdout_truncated,
            None,
        )
    assert command.response_path is not None
    try:
        metadata = command.response_path.lstat()
    except FileNotFoundError:
        return b'', RawResponseSource.NONE, False, None
    if not stat.S_ISREG(metadata.st_mode) or command.response_path.is_symlink():
        return (
            b'',
            RawResponseSource.NONE,
            False,
            ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.MISSING_RESPONSE,
                detail='Codex did not produce a regular output-last-message file.',
            ),
        )
    if metadata.st_size > maximum_bytes:
        return (
            b'',
            RawResponseSource.NONE,
            True,
            ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.RESPONSE_LIMIT,
                detail='The Codex output-last-message file exceeded the configured response limit.',
            ),
        )
    try:
        content = command.response_path.read_bytes()
    except OSError:
        return (
            b'',
            RawResponseSource.NONE,
            False,
            ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.MISSING_RESPONSE,
                detail='The Codex output-last-message file could not be read.',
            ),
        )
    if not content:
        return b'', RawResponseSource.NONE, False, None
    return content, RawResponseSource.CODEX_LAST_MESSAGE, False, None


def _process_or_response_failure(
    *,
    harness: ReferenceHarnessName,
    process_result: BoundedProcessResult,
    raw_response: bytes,
    response_failure: ReferenceHarnessFailure | None,
    cursor_expected_prompt: bytes | None,
    cursor_expected_workspace: Path | None,
) -> _ParsedResponse:
    if process_result.termination == 'timed_out':
        return _failure(ReferenceHarnessFailureCode.TIMED_OUT, 'The local harness exceeded its wall-time limit.')
    if process_result.termination == 'response_limit':
        code = (
            ReferenceHarnessFailureCode.CLI_STDOUT_LIMIT
            if harness == ReferenceHarnessName.CODEX
            else ReferenceHarnessFailureCode.RESPONSE_LIMIT
        )
        return _failure(code, 'The local harness exceeded its bounded stdout limit.')
    if process_result.termination == 'log_limit':
        return _failure(
            ReferenceHarnessFailureCode.CLI_STDERR_LIMIT,
            'The local harness exceeded its bounded stderr limit.',
        )
    if process_result.exit_code != 0:
        return _failure(
            ReferenceHarnessFailureCode.NONZERO_EXIT,
            'The local harness exited with a nonzero status.',
        )
    if response_failure is not None:
        return _ParsedResponse(submission=None, failure=response_failure)
    if not raw_response:
        return _failure(
            ReferenceHarnessFailureCode.MISSING_RESPONSE,
            'The local harness exited without a model response.',
        )
    if harness == ReferenceHarnessName.CODEX:
        return _parse_codex_response(raw_response)
    if harness == ReferenceHarnessName.CLAUDE:
        return _parse_claude_response(raw_response)
    if cursor_expected_prompt is None or cursor_expected_workspace is None:
        return _failure(
            ReferenceHarnessFailureCode.EXECUTION_ERROR,
            'The Cursor wrapper was missing its locally pinned invocation context.',
        )
    return _parse_cursor_response(
        raw_response,
        expected_prompt=cursor_expected_prompt,
        expected_workspace=cursor_expected_workspace,
    )


def _parse_codex_response(raw_response: bytes) -> _ParsedResponse:
    try:
        decoded = raw_response.decode('utf-8')
    except UnicodeDecodeError:
        return _failure(ReferenceHarnessFailureCode.INVALID_UTF8, 'The Codex response was not valid UTF-8.')
    try:
        submission = Submission.model_validate_json(decoded)
    except ValidationError as error:
        return _failure(
            ReferenceHarnessFailureCode.INVALID_SUBMISSION,
            'The Codex final message did not validate as a VaxReplay Submission.',
            validation_error_count=error.error_count(),
        )
    return _ParsedResponse(submission=submission, failure=None)


def _parse_claude_response(raw_response: bytes) -> _ParsedResponse:
    try:
        decoded = raw_response.decode('utf-8')
    except UnicodeDecodeError:
        return _failure(ReferenceHarnessFailureCode.INVALID_UTF8, 'The Claude response was not valid UTF-8.')
    try:
        wrapper = json.loads(decoded)
    except json.JSONDecodeError:
        return _failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            'Claude stdout did not contain one valid JSON result wrapper.',
        )
    if not isinstance(wrapper, dict):
        return _failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            'Claude stdout JSON was not a result object.',
        )
    resolved_model, additional_models = _reported_claude_models(wrapper)
    if wrapper.get('is_error') is True:
        return _ParsedResponse(
            submission=None,
            failure=ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.PROVIDER_ERROR,
                detail='Claude reported a provider or harness error.',
            ),
            resolved_model=resolved_model,
            additional_reported_models=additional_models,
        )
    candidate: object
    if 'structured_output' in wrapper:
        candidate = wrapper['structured_output']
    elif isinstance(wrapper.get('result'), str):
        try:
            candidate = json.loads(wrapper['result'])
        except json.JSONDecodeError:
            return _ParsedResponse(
                submission=None,
                failure=ReferenceHarnessFailure(
                    code=ReferenceHarnessFailureCode.INVALID_SUBMISSION,
                    detail='Claude result text did not contain a JSON VaxReplay Submission.',
                ),
                resolved_model=resolved_model,
                additional_reported_models=additional_models,
            )
    else:
        candidate = wrapper
    try:
        submission = Submission.model_validate_json(json.dumps(candidate))
    except TypeError:
        return _ParsedResponse(
            submission=None,
            failure=ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.INVALID_SUBMISSION,
                detail='Claude structured output was not JSON-serializable.',
            ),
            resolved_model=resolved_model,
            additional_reported_models=additional_models,
        )
    except ValidationError as error:
        return _ParsedResponse(
            submission=None,
            failure=ReferenceHarnessFailure(
                code=ReferenceHarnessFailureCode.INVALID_SUBMISSION,
                detail='Claude structured output did not validate as a VaxReplay Submission.',
                validation_error_count=error.error_count(),
            ),
            resolved_model=resolved_model,
            additional_reported_models=additional_models,
        )
    return _ParsedResponse(
        submission=submission,
        failure=None,
        resolved_model=resolved_model,
        additional_reported_models=additional_models,
    )


def _reported_claude_models(wrapper: dict[str, object]) -> tuple[str | None, tuple[str, ...]]:
    models: list[str] = []
    top_level_model = wrapper.get('model')
    if isinstance(top_level_model, str) and top_level_model:
        models.append(top_level_model)
    model_usage = wrapper.get('modelUsage')
    if isinstance(model_usage, dict):
        models.extend(model for model in model_usage if isinstance(model, str) and model)
    unique_models = tuple(dict.fromkeys(models))
    if not unique_models:
        return None, ()
    return unique_models[0], unique_models[1:]


def _parse_cursor_response(
    raw_response: bytes,
    *,
    expected_prompt: bytes,
    expected_workspace: Path,
) -> _ParsedResponse:
    try:
        decoded = raw_response.decode('utf-8')
    except UnicodeDecodeError:
        return _failure(ReferenceHarnessFailureCode.INVALID_UTF8, 'The Cursor response was not valid UTF-8.')

    physical_lines = decoded.splitlines()
    nonempty_lines = [(line_number, line) for line_number, line in enumerate(physical_lines, start=1) if line.strip()]
    if not nonempty_lines:
        return _failure(ReferenceHarnessFailureCode.MISSING_RESPONSE, 'Cursor stdout had no JSON events.')

    parsed_events: list[_CursorParsedEvent] = []
    for line_number, line in nonempty_lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line, object_pairs_hook=_cursor_json_object)
        except json.JSONDecodeError:
            inventory = _build_cursor_parse_failure_inventory(
                parsed_events,
                total_lines=len(physical_lines),
                nonempty_lines=len(nonempty_lines),
                first_unparseable_line=line_number,
                duplicate_json_key_observed=False,
                expected_prompt=expected_prompt,
                expected_workspace=expected_workspace,
            )
            return _cursor_failure(
                ReferenceHarnessFailureCode.INVALID_WRAPPER,
                'Cursor stdout did not contain valid JSON or NDJSON events.',
                inventory=inventory,
            )
        except _CursorStreamValidationError:
            inventory = _build_cursor_parse_failure_inventory(
                parsed_events,
                total_lines=len(physical_lines),
                nonempty_lines=len(nonempty_lines),
                first_unparseable_line=line_number,
                duplicate_json_key_observed=True,
                expected_prompt=expected_prompt,
                expected_workspace=expected_workspace,
            )
            return _cursor_failure(
                ReferenceHarnessFailureCode.INVALID_WRAPPER,
                'Cursor stdout contained a duplicate JSON object key.',
                inventory=inventory,
            )
        if not isinstance(event, dict):
            inventory = _build_cursor_parse_failure_inventory(
                parsed_events,
                total_lines=len(physical_lines),
                nonempty_lines=len(nonempty_lines),
                first_unparseable_line=line_number,
                duplicate_json_key_observed=False,
                expected_prompt=expected_prompt,
                expected_workspace=expected_workspace,
            )
            return _cursor_failure(
                ReferenceHarnessFailureCode.INVALID_WRAPPER,
                'A Cursor stdout event was not a JSON object.',
                inventory=inventory,
            )
        parsed_events.append(_CursorParsedEvent(line=line_number, event=event))

    events = [item.event for item in parsed_events]
    failure_inventory = _build_cursor_parse_failure_inventory(
        parsed_events,
        total_lines=len(physical_lines),
        nonempty_lines=len(nonempty_lines),
        first_unparseable_line=None,
        duplicate_json_key_observed=False,
        expected_prompt=expected_prompt,
        expected_workspace=expected_workspace,
    )

    try:
        stream_metadata = _validate_cursor_stream(
            events,
            expected_prompt=expected_prompt,
            expected_workspace=expected_workspace,
        )
    except _CursorStreamValidationError as error:
        return _cursor_failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            f'Cursor stdout violated the pinned stream contract: {error}.',
            inventory=failure_inventory,
        )

    if any(event.get('type') == 'tool_call' for event in events):
        return _cursor_failure(
            ReferenceHarnessFailureCode.UNEXPECTED_TOOL_CALL,
            'Cursor emitted a tool call despite the no-tools output contract.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
        )

    terminal_events = [event for event in events if event.get('type') == 'result']
    if len(terminal_events) != 1 or events[-1] is not terminal_events[0]:
        return _cursor_failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            'Cursor stdout did not end with exactly one terminal result event.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
        )
    terminal = terminal_events[0]
    if not isinstance(terminal.get('is_error'), bool):
        return _cursor_failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            'The Cursor terminal result did not contain a boolean is_error field.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
        )
    if terminal['is_error'] is True:
        return _cursor_failure(
            ReferenceHarnessFailureCode.PROVIDER_ERROR,
            'Cursor reported a provider or harness error.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
        )
    result = terminal.get('result')
    if not isinstance(result, str):
        return _cursor_failure(
            ReferenceHarnessFailureCode.INVALID_WRAPPER,
            'The Cursor terminal result did not contain assistant text.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
        )
    try:
        submission = Submission.model_validate_json(result)
    except ValidationError as error:
        return _cursor_failure(
            ReferenceHarnessFailureCode.INVALID_SUBMISSION,
            'Cursor terminal result text did not validate as a VaxReplay Submission.',
            inventory=failure_inventory,
            resolved_model=stream_metadata.resolved_model,
            validation_error_count=error.error_count(),
        )
    return _ParsedResponse(
        submission=submission,
        failure=None,
        resolved_model=stream_metadata.resolved_model,
    )


def _cursor_failure(
    code: ReferenceHarnessFailureCode,
    detail: str,
    *,
    inventory: CursorParseFailureInventory,
    resolved_model: str | None = None,
    validation_error_count: int | None = None,
) -> _ParsedResponse:
    return _ParsedResponse(
        submission=None,
        failure=ReferenceHarnessFailure(
            code=code,
            detail=detail,
            validation_error_count=validation_error_count,
        ),
        resolved_model=resolved_model,
        cursor_parse_failure_inventory=inventory,
    )


def _build_cursor_parse_failure_inventory(
    parsed_events: Sequence[_CursorParsedEvent],
    *,
    total_lines: int,
    nonempty_lines: int,
    first_unparseable_line: int | None,
    duplicate_json_key_observed: bool,
    expected_prompt: bytes,
    expected_workspace: Path,
) -> CursorParseFailureInventory:
    kinds: dict[tuple[str, str | None], list[int]] = {}
    events = [item.event for item in parsed_events]
    for item in parsed_events:
        event_type = _sanitize_cursor_event_token(item.event.get('type'), allow_none=False)
        assert event_type is not None
        event_subtype = _sanitize_cursor_event_token(item.event.get('subtype'), allow_none=True)
        kinds.setdefault((event_type, event_subtype), []).append(item.line)

    observations = tuple(
        CursorEventKindObservation(
            event_type=event_type,
            event_subtype=event_subtype,
            count=len(lines),
            first_line=lines[0],
            last_line=lines[-1],
        )
        for (event_type, event_subtype), lines in sorted(
            kinds.items(),
            key=lambda item: (item[1][0], item[0][0], item[0][1] or ''),
        )
    )
    consistency = _cursor_consistency_flags(
        events,
        expected_prompt=expected_prompt,
        expected_workspace=expected_workspace,
    )
    return CursorParseFailureInventory(
        total_lines=total_lines,
        nonempty_lines=nonempty_lines,
        parsed_event_lines=len(parsed_events),
        first_unparseable_line=first_unparseable_line,
        duplicate_json_key_observed=duplicate_json_key_observed,
        event_kinds=observations,
        tool_event_count=sum(event.get('type') == 'tool_call' for event in events),
        terminal_event_count=sum(event.get('type') == 'result' for event in events),
        consistency=consistency,
    )


def _sanitize_cursor_event_token(value: object, *, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str) and _CURSOR_EVENT_TOKEN_PATTERN.fullmatch(value) is not None:
        return value
    return 'invalid'


def _cursor_consistency_flags(
    events: Sequence[dict[str, object]],
    *,
    expected_prompt: bytes,
    expected_workspace: Path,
) -> CursorParseConsistencyFlags:
    event_contract_valid = True
    for event in events:
        event_type = event.get('type')
        subtype = event.get('subtype')
        if (
            not isinstance(event_type, str)
            or event_type not in _CURSOR_EVENT_SUBTYPES
            or (subtype is not None and not isinstance(subtype, str))
            or subtype not in _CURSOR_EVENT_SUBTYPES.get(event_type, frozenset())
        ):
            event_contract_valid = False
            break

    first_init_valid = bool(events and events[0].get('type') == 'system' and events[0].get('subtype') == 'init')
    single_system_init = sum(event.get('type') == 'system' for event in events) == 1
    user_indexes = [index for index, event in enumerate(events) if event.get('type') == 'user']
    user_position_valid = user_indexes == [1]
    terminal_indexes = [index for index, event in enumerate(events) if event.get('type') == 'result']
    terminal_position_valid = terminal_indexes == [len(events) - 1]

    session_consistent = _cursor_metadata_is_consistent(events, 'session_id', maximum_length=512)
    model_consistent = _cursor_metadata_is_consistent(events, 'model', maximum_length=512)

    workspace_consistent = False
    if first_init_valid:
        initial_cwd = events[0].get('cwd')
        if initial_cwd is None:
            workspace_consistent = True
        elif isinstance(initial_cwd, str) and initial_cwd and '\x00' not in initial_cwd:
            workspace_consistent = Path(initial_cwd).resolve() == expected_workspace.resolve()

    user_transcript_consistent = False
    if user_position_valid:
        try:
            reported_text = _cursor_message_text(events[1], expected_role='user')
            expected_text = expected_prompt.decode('utf-8')
            user_transcript_consistent = reported_text in (expected_text, expected_text.rstrip('\n'))
        except (UnicodeDecodeError, _CursorStreamValidationError):
            user_transcript_consistent = False

    return CursorParseConsistencyFlags(
        event_type_subtype_contract_valid=event_contract_valid,
        first_system_init_valid=first_init_valid,
        single_system_init=single_system_init,
        user_event_position_valid=user_position_valid,
        terminal_event_position_valid=terminal_position_valid,
        session_metadata_consistent=session_consistent,
        model_metadata_consistent=model_consistent,
        workspace_metadata_consistent=workspace_consistent,
        user_transcript_consistent=user_transcript_consistent,
    )


def _cursor_metadata_is_consistent(
    events: Sequence[dict[str, object]],
    key: str,
    *,
    maximum_length: int,
) -> bool:
    try:
        _consistent_cursor_metadata(events, key, maximum_length=maximum_length)
    except _CursorStreamValidationError:
        return False
    return True


def _validate_cursor_stream(
    events: Sequence[dict[str, object]],
    *,
    expected_prompt: bytes,
    expected_workspace: Path,
) -> _CursorStreamMetadata:
    """Validate Cursor's current documented one-shot stream shape and echoed metadata.

    Unknown future event types deliberately fail closed.  Unknown object fields remain allowed so
    backward-compatible metadata additions cannot themselves break the adapter.
    """

    for event in events:
        event_type = event.get('type')
        if not isinstance(event_type, str) or event_type not in _CURSOR_EVENT_SUBTYPES:
            raise _CursorStreamValidationError('unknown event type')
        subtype = event.get('subtype')
        if subtype is not None and not isinstance(subtype, str):
            raise _CursorStreamValidationError('invalid event subtype')
        if subtype not in _CURSOR_EVENT_SUBTYPES[event_type]:
            raise _CursorStreamValidationError('unknown event subtype')

    _validate_cursor_thinking_sequence(events)

    first = events[0]
    if first.get('type') != 'system' or first.get('subtype') != 'init':
        raise _CursorStreamValidationError('missing first system init event')
    if sum(event.get('type') == 'system' for event in events) != 1:
        raise _CursorStreamValidationError('duplicate system init event')

    terminal_indexes = [index for index, event in enumerate(events) if event.get('type') == 'result']
    if terminal_indexes != [len(events) - 1]:
        raise _CursorStreamValidationError('invalid terminal result position')

    session_id = _consistent_cursor_metadata(events, 'session_id', maximum_length=512)
    resolved_model = _consistent_cursor_metadata(events, 'model', maximum_length=512)
    del session_id  # The value is validated for consistency but intentionally not persisted.

    initial_cwd = first.get('cwd')
    if initial_cwd is not None:
        if not isinstance(initial_cwd, str) or not initial_cwd or '\x00' in initial_cwd:
            raise _CursorStreamValidationError('invalid init workspace metadata')
        if Path(initial_cwd).resolve() != expected_workspace.resolve():
            raise _CursorStreamValidationError('init workspace mismatch')

    user_indexes = [index for index, event in enumerate(events) if event.get('type') == 'user']
    if user_indexes != [1]:
        raise _CursorStreamValidationError('missing or misplaced user event')
    expected_text = expected_prompt.decode('utf-8')
    reported_text = _cursor_message_text(events[1], expected_role='user')
    if reported_text not in (expected_text, expected_text.rstrip('\n')):
        raise _CursorStreamValidationError('user transcript mismatch')

    for event in events[2:-1]:
        if event.get('type') == 'assistant':
            _cursor_message_text(event, expected_role='assistant')

    return _CursorStreamMetadata(resolved_model=resolved_model)


def _validate_cursor_thinking_sequence(events: Sequence[dict[str, object]]) -> None:
    """Admit one bounded reasoning-event block without reading or retaining its payload."""

    thinking_indexes = [index for index, event in enumerate(events) if event.get('type') == 'thinking']
    if not thinking_indexes:
        return
    if thinking_indexes[0] != 2 or thinking_indexes[-1] >= len(events) - 1:
        raise _CursorStreamValidationError('invalid thinking event position')
    if thinking_indexes != list(range(thinking_indexes[0], thinking_indexes[-1] + 1)):
        raise _CursorStreamValidationError('noncontiguous thinking event block')
    assistant_indexes = [index for index, event in enumerate(events) if event.get('type') == 'assistant']
    if assistant_indexes and thinking_indexes[-1] > assistant_indexes[0]:
        raise _CursorStreamValidationError('thinking event follows assistant output')

    subtypes = [events[index].get('subtype') for index in thinking_indexes]
    if len(subtypes) < 2 or subtypes[-1] != 'completed' or any(subtype != 'delta' for subtype in subtypes[:-1]):
        raise _CursorStreamValidationError('invalid thinking event sequence')


def _cursor_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _CursorStreamValidationError('duplicate JSON object key')
        result[key] = value
    return result


def _consistent_cursor_metadata(
    events: Sequence[dict[str, object]],
    key: str,
    *,
    maximum_length: int,
) -> str | None:
    observed: str | None = None
    for event in events:
        if key not in event:
            continue
        value = event[key]
        if not isinstance(value, str) or not value or len(value) > maximum_length or '\x00' in value:
            raise _CursorStreamValidationError('invalid transcript metadata')
        if observed is None:
            observed = value
        elif observed != value:
            raise _CursorStreamValidationError('inconsistent transcript metadata')
    return observed


def _cursor_message_text(event: dict[str, object], *, expected_role: str) -> str:
    message = event.get('message')
    if not isinstance(message, dict) or message.get('role') != expected_role:
        raise _CursorStreamValidationError('invalid message role')
    content = message.get('content')
    if not isinstance(content, list) or not content:
        raise _CursorStreamValidationError('invalid message content')
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get('type') != 'text':
            raise _CursorStreamValidationError('invalid message content')
        text = item.get('text')
        if not isinstance(text, str):
            raise _CursorStreamValidationError('invalid message content')
        text_parts.append(text)
    return ''.join(text_parts)


def _failure(
    code: ReferenceHarnessFailureCode,
    detail: str,
    *,
    validation_error_count: int | None = None,
) -> _ParsedResponse:
    return _ParsedResponse(
        submission=None,
        failure=ReferenceHarnessFailure(
            code=code,
            detail=detail,
            validation_error_count=validation_error_count,
        ),
    )


def _receipt(
    verified: VerifiedChallengeEnvelope,
    *,
    harness: ReferenceHarnessName,
    harness_version: str,
    harness_executable_sha256: str | None,
    harness_runtime_identity: ReferenceHarnessRuntimeIdentity | None,
    requested_model: str | None,
    command: HarnessCommand,
    prompt_bytes: bytes,
    started_at: datetime,
    started_monotonic: float,
    process_result: BoundedProcessResult | None,
    raw_response: bytes,
    raw_response_source: RawResponseSource,
    raw_response_truncated: bool,
    parsed: _ParsedResponse,
) -> ReferenceHarnessReceipt:
    finished_at = datetime.now(UTC)
    stdout = process_result.stdout if process_result is not None else b''
    stderr = process_result.stderr if process_result is not None else b''
    envelope = verified.envelope
    return ReferenceHarnessReceipt(
        challenge_id=envelope.challenge_id,
        suite_id=envelope.suite_id,
        suite_manifest_sha256=envelope.suite_manifest_sha256,
        envelope_sha256=verified.envelope_sha256,
        ordinal=envelope.ordinal,
        episode_id=envelope.binding.episode_id,
        episode_manifest_sha256=envelope.binding.manifest_sha256,
        prompt_variant=envelope.prompt_variant,
        harness_name=harness,
        harness_version=harness_version,
        harness_executable_sha256=harness_executable_sha256,
        harness_executable_sha256_scope=(
            'invoked_file_bytes' if harness_executable_sha256 is not None else 'unavailable'
        ),
        harness_runtime_identity=harness_runtime_identity,
        requested_model=requested_model,
        resolved_model=parsed.resolved_model,
        additional_reported_models=parsed.additional_reported_models,
        command_argv=command.receipt_argv,
        rendered_prompt_sha256=_sha256(prompt_bytes),
        rendered_prompt_bytes=len(prompt_bytes),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=max(0, round((time.monotonic() - started_monotonic) * 1000)),
        process_termination=process_result.termination if process_result is not None else 'not_started',
        exit_code=process_result.exit_code if process_result is not None else None,
        raw_response_source=raw_response_source,
        raw_response_sha256=_sha256(raw_response),
        raw_response_bytes=len(raw_response),
        raw_response_truncated=raw_response_truncated,
        cli_stdout_sha256=_sha256(stdout),
        cli_stdout_bytes=len(stdout),
        cli_stdout_truncated=process_result.stdout_truncated if process_result is not None else False,
        cli_stderr_sha256=_sha256(stderr),
        cli_stderr_bytes=len(stderr),
        cli_stderr_truncated=process_result.stderr_truncated if process_result is not None else False,
        submission=parsed.submission,
        failure=parsed.failure,
        cursor_parse_failure_inventory=parsed.cursor_parse_failure_inventory,
    )


def _normalize_budget(value: str) -> str:
    try:
        budget = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('claude_max_budget_usd must be a decimal number') from error
    if not budget.is_finite() or budget <= 0 or budget > 100:
        raise ValueError('claude_max_budget_usd must be greater than zero and at most 100')
    return format(budget, 'f')


def _resolve_executable_identity(
    executable: str,
    *,
    harness: ReferenceHarnessName,
) -> _ExecutableIdentity:
    """Resolve and hash the exact launcher file when it is locally readable.

    This is evidence about local bytes, not a remote attestation.  On any lookup/read failure the
    original executable remains usable and the receipt explicitly records that the hash was unavailable.
    """

    located = shutil.which(executable)
    if located is None:
        return _ExecutableIdentity(invocation_path=executable, sha256=None)
    try:
        path = Path(located).resolve(strict=True)
        _, executable_sha256 = _hash_local_regular_file(path)
    except (OSError, ValueError):
        return _ExecutableIdentity(invocation_path=executable, sha256=None)
    runtime_identity = _resolve_cursor_runtime_identity(path) if harness == ReferenceHarnessName.CURSOR else None
    return _ExecutableIdentity(
        invocation_path=str(path),
        sha256=executable_sha256,
        runtime_identity=runtime_identity,
    )


def _resolve_cursor_runtime_identity(executable_path: Path) -> ReferenceHarnessRuntimeIdentity | None:
    """Hash a recognized Cursor version directory without persisting local paths.

    Discovery is deliberately narrow: an unrecognized layout remains ``None`` rather than making
    a partial identity claim.  Symlinks and non-regular entries also make the identity unavailable.
    """

    root = executable_path.parent
    package_path = root / 'package.json'
    entrypoint_path = root / 'index.js'
    try:
        package_content = _read_local_regular_file(package_path, 64 * 1024)
        package = json.loads(package_content)
        if not isinstance(package, dict) or package.get('name') != '@anysphere/agent-cli-runtime':
            return None
        _hash_local_regular_file(entrypoint_path)

        identities: list[_RuntimeFileIdentity] = []
        total_bytes = 0
        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            directory_names.sort()
            file_names.sort()
            for directory_name in directory_names:
                if (directory_path / directory_name).is_symlink():
                    raise ValueError('Cursor runtime directory contains a symlink')
            for file_name in file_names:
                file_path = directory_path / file_name
                if file_path.is_symlink():
                    raise ValueError('Cursor runtime directory contains a symlink')
                size, file_sha256 = _hash_local_regular_file(file_path)
                total_bytes += size
                if len(identities) >= _MAX_CURSOR_RUNTIME_FILES or total_bytes > _MAX_CURSOR_RUNTIME_BYTES:
                    raise ValueError('Cursor runtime directory exceeds identity bounds')
                identities.append(
                    _RuntimeFileIdentity(
                        relative_path=file_path.relative_to(root).as_posix(),
                        size=size,
                        sha256=file_sha256,
                    )
                )
    except (json.JSONDecodeError, OSError, ValueError):
        return None

    identities.sort(key=lambda item: item.relative_path)
    by_path = {item.relative_path: item for item in identities}
    if 'index.js' not in by_path or 'package.json' not in by_path or executable_path.name not in by_path:
        return None
    chunks = [item for item in identities if item.relative_path.endswith('.index.js')]
    if not chunks:
        return None
    return ReferenceHarnessRuntimeIdentity(
        tree_sha256=_runtime_file_manifest_sha256(_CURSOR_RUNTIME_TREE_SCHEMA_VERSION, identities),
        regular_file_count=len(identities),
        total_file_bytes=total_bytes,
        chunk_manifest_sha256=_runtime_file_manifest_sha256(_CURSOR_RUNTIME_CHUNK_SCHEMA_VERSION, chunks),
        chunk_file_count=len(chunks),
        entrypoint_sha256=by_path['index.js'].sha256,
        package_manifest_sha256=by_path['package.json'].sha256,
    )


def _runtime_file_manifest_sha256(schema_version: str, files: Sequence[_RuntimeFileIdentity]) -> str:
    manifest = {
        'schema_version': schema_version,
        'files': [{'path': item.relative_path, 'size': item.size, 'sha256': item.sha256} for item in files],
    }
    return _sha256(canonical_json_bytes(manifest))


def _hash_local_regular_file(path: Path) -> tuple[int, str]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('identity target is not a regular file')
        digest = hashlib.sha256()
        observed_size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            observed_size += len(chunk)
        after = os.fstat(descriptor)
        if (
            observed_size != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
        ):
            raise ValueError('identity target changed while hashing')
        return observed_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_local_regular_file(path: Path, maximum_bytes: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
            raise ValueError('local identity metadata is not a bounded regular file')
        content = bytearray()
        while chunk := os.read(descriptor, min(65_536, maximum_bytes - len(content) + 1)):
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ValueError('local identity metadata exceeds its bound')
        return bytes(content)
    finally:
        os.close(descriptor)


def _validate_identifier(value: str, field: str) -> None:
    if not value or '\x00' in value:
        raise ValueError(f'{field} must be non-empty and NUL-free')


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReferenceHarnessInputError(f'cannot open challenge envelope: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReferenceHarnessInputError('challenge envelope must be a regular file')
        if metadata.st_size > maximum_bytes:
            raise ReferenceHarnessInputError('challenge envelope exceeds its size limit')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ReferenceHarnessInputError('challenge envelope exceeds its size limit')
    except OSError as error:
        raise ReferenceHarnessInputError(f'cannot read challenge envelope: {error}') from error
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    if not value:
        return _EMPTY_SHA256
    return hashlib.sha256(value).hexdigest()
