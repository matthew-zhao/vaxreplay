"""Pure command construction for supported local reference harnesses."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vaxreplay.reference_harness.schema import RawResponseSource, ReferenceHarnessName


@dataclass(frozen=True)
class HarnessCommand:
    """An executable argv plus the content-free form safe to persist in a receipt."""

    argv: tuple[str, ...]
    receipt_argv: tuple[str, ...]
    raw_response_source: RawResponseSource
    response_path: Path | None


def make_openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Require every object property while preserving nullable optional fields.

    OpenAI structured outputs require ``required`` to enumerate every key in each object. Pydantic
    omits nullable fields with defaults from that array, so recursively tighten a copy for Codex
    without changing the provider-neutral validation model used after inference.
    """

    tightened = copy.deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop('default', None)
            properties = value.get('properties')
            if isinstance(properties, dict):
                value['required'] = list(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(tightened)
    return tightened


def build_harness_command(
    harness: ReferenceHarnessName,
    *,
    executable: str,
    requested_model: str | None,
    work_dir: Path,
    schema_path: Path,
    response_path: Path,
    submission_schema_json: str,
    claude_max_budget_usd: str,
) -> HarnessCommand:
    if harness == ReferenceHarnessName.CODEX:
        return build_codex_command(
            executable=executable,
            requested_model=requested_model,
            work_dir=work_dir,
            schema_path=schema_path,
            response_path=response_path,
        )
    if harness == ReferenceHarnessName.CLAUDE:
        return build_claude_command(
            executable=executable,
            requested_model=requested_model,
            submission_schema_json=submission_schema_json,
            claude_max_budget_usd=claude_max_budget_usd,
        )
    if harness == ReferenceHarnessName.CURSOR:
        return build_cursor_command(
            executable=executable,
            requested_model=requested_model,
            work_dir=work_dir,
        )
    raise ValueError(f'unsupported reference harness: {harness}')


def build_codex_command(
    *,
    executable: str,
    requested_model: str | None,
    work_dir: Path,
    schema_path: Path,
    response_path: Path,
) -> HarnessCommand:
    """Build a one-turn Codex exec command; the prompt is supplied only on stdin."""

    prefix = (
        executable,
        'exec',
        '--ephemeral',
        '--ignore-user-config',
        '--ignore-rules',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        '--cd',
        str(work_dir),
        '--config',
        'web_search="disabled"',
        '--color',
        'never',
        '--output-schema',
        str(schema_path),
        '--output-last-message',
        str(response_path),
    )
    receipt_prefix = (
        Path(executable).name,
        'exec',
        '--ephemeral',
        '--ignore-user-config',
        '--ignore-rules',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        '--cd',
        '<EMPTY_WORK_DIR>',
        '--config',
        'web_search="disabled"',
        '--color',
        'never',
        '--output-schema',
        '<SUBMISSION_SCHEMA_FILE>',
        '--output-last-message',
        '<LAST_MESSAGE_FILE>',
    )
    model_arguments = ('--model', requested_model) if requested_model is not None else ()
    return HarnessCommand(
        argv=(*prefix, *model_arguments, '-'),
        receipt_argv=(*receipt_prefix, *model_arguments, '-'),
        raw_response_source=RawResponseSource.CODEX_LAST_MESSAGE,
        response_path=response_path,
    )


def build_claude_command(
    *,
    executable: str,
    requested_model: str | None,
    submission_schema_json: str,
    claude_max_budget_usd: str,
) -> HarnessCommand:
    """Build a no-tools Claude print command; stdin is used when no prompt arg is given."""

    prefix = (
        executable,
        '--print',
        '--safe-mode',
        '--no-session-persistence',
        '--no-chrome',
        '--disable-slash-commands',
        '--tools=',
        '--permission-mode',
        'dontAsk',
        '--strict-mcp-config',
        '--mcp-config',
        '{"mcpServers":{}}',
        '--output-format',
        'json',
        '--json-schema',
        submission_schema_json,
        '--max-budget-usd',
        claude_max_budget_usd,
    )
    receipt_prefix = (
        Path(executable).name,
        '--print',
        '--safe-mode',
        '--no-session-persistence',
        '--no-chrome',
        '--disable-slash-commands',
        '--tools=<NO_TOOLS>',
        '--permission-mode',
        'dontAsk',
        '--strict-mcp-config',
        '--mcp-config',
        '<EMPTY_MCP_CONFIG>',
        '--output-format',
        'json',
        '--json-schema',
        '<SUBMISSION_JSON_SCHEMA>',
        '--max-budget-usd',
        claude_max_budget_usd,
    )
    model_arguments = ('--model', requested_model) if requested_model is not None else ()
    return HarnessCommand(
        argv=(*prefix, *model_arguments),
        receipt_argv=(*receipt_prefix, *model_arguments),
        raw_response_source=RawResponseSource.CLAUDE_STDOUT,
        response_path=None,
    )


def build_cursor_command(
    *,
    executable: str,
    requested_model: str | None,
    work_dir: Path,
) -> HarnessCommand:
    """Build a read-only Cursor Agent run with an auditable NDJSON event stream.

    Cursor does not currently expose a CLI flag for a response JSON Schema or a strict empty MCP
    configuration.  Ask mode, its sandbox, an empty workspace, and the absence of any auto-approval
    flag are the strongest controls exposed by the CLI.  The runner rejects any observed tool call.
    """

    prefix = (
        executable,
        '--print',
        '--output-format',
        'stream-json',
        '--mode',
        'ask',
        '--sandbox',
        'enabled',
        '--trust',
        '--workspace',
        str(work_dir),
    )
    receipt_prefix = (
        Path(executable).name,
        '--print',
        '--output-format',
        'stream-json',
        '--mode',
        'ask',
        '--sandbox',
        'enabled',
        '--trust',
        '--workspace',
        '<EMPTY_WORK_DIR>',
    )
    model_arguments = ('--model', requested_model) if requested_model is not None else ()
    return HarnessCommand(
        argv=(*prefix, *model_arguments),
        receipt_argv=(*receipt_prefix, *model_arguments),
        raw_response_source=RawResponseSource.CURSOR_STDOUT,
        response_path=None,
    )
