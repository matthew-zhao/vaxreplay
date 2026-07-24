from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateForecast, Submission
from vaxreplay.reference_harness.commands import make_openai_strict_json_schema
from vaxreplay.reference_harness.runner import (
    ReferenceHarnessInputError,
    canonical_receipt_bytes,
    load_verified_challenge_envelope,
    render_challenge_prompt,
    render_cursor_challenge_prompt,
    run_reference_harness,
    verify_challenge_envelope,
)
from vaxreplay.reference_harness.schema import (
    ReferenceHarnessFailureCode,
    ReferenceHarnessName,
)
from vaxreplay.runner._process import BoundedProcessResult
from vaxreplay.runner.challenge import build_challenge_bundle, challenge_envelope_sha256


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _envelope():  # type annotation would repeat the core schema solely for a test helper
    with tempfile.TemporaryDirectory() as temporary_directory:
        challenge = build_challenge_bundle(
            Path(temporary_directory) / 'challenge',
            challenge_id='reference-challenge',
            suite_id='reference-suite',
            episode_dirs=[_fixture()],
        )
        return challenge.envelopes[0]


def _submission(envelope) -> Submission:
    return Submission(
        episode_id=envelope.binding.episode_id,
        manifest_sha256=envelope.binding.manifest_sha256,
        ranking=['candidate-1'],
        forecasts=[
            CandidateForecast(
                candidate_id='candidate-1',
                target_id='target-1',
                horizon_days=1,
                probability=0.5,
            )
        ],
    )


def _cursor_stream(
    submission: Submission,
    *,
    prompt: bytes,
    workspace: Path,
    between: tuple[dict[str, object], ...] = (),
    init_updates: Mapping[str, object] | None = None,
    user_text: str | None = None,
    result_updates: Mapping[str, object] | None = None,
) -> bytes:
    init: dict[str, object] = {
        'type': 'system',
        'subtype': 'init',
        'model': 'Cursor Model Display',
        'session_id': 'fictional-session',
        'cwd': str(workspace),
    }
    if init_updates is not None:
        init.update(init_updates)
    user: dict[str, object] = {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [{'type': 'text', 'text': prompt.decode('utf-8') if user_text is None else user_text}],
        },
        'session_id': 'fictional-session',
    }
    result: dict[str, object] = {
        'type': 'result',
        'subtype': 'success',
        'is_error': False,
        'result': canonical_json_bytes(submission).decode('utf-8'),
        'session_id': 'fictional-session',
    }
    if result_updates is not None:
        result.update(result_updates)
    return b'\n'.join(canonical_json_bytes(event) for event in (init, user, *between, result))


class ReferenceHarnessRunnerTest(unittest.TestCase):
    def test_openai_schema_requires_nullable_fields_recursively(self) -> None:
        original = {
            'type': 'object',
            'properties': {
                'required_value': {'type': 'string'},
                'optional_value': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None},
                'nested': {
                    'type': 'object',
                    'properties': {'optional_nested': {'type': ['string', 'null'], 'default': None}},
                },
            },
            'required': ['required_value'],
        }

        tightened = make_openai_strict_json_schema(original)

        self.assertEqual(tightened['required'], ['required_value', 'optional_value', 'nested'])
        nested = tightened['properties']['nested']
        self.assertEqual(nested['required'], ['optional_nested'])
        self.assertNotIn('default', tightened['properties']['optional_value'])
        self.assertNotIn('default', nested['properties']['optional_nested'])
        self.assertEqual(original['required'], ['required_value'])

    def test_loader_requires_canonical_bytes_and_expected_hash(self) -> None:
        envelope = _envelope()
        expected_hash = challenge_envelope_sha256(envelope)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'envelope.json'
            path.write_bytes(canonical_json_bytes(envelope))
            verified = load_verified_challenge_envelope(path, expected_sha256=expected_hash)
            self.assertEqual(verified.envelope_sha256, expected_hash)

            path.write_text(json.dumps(envelope.model_dump(mode='json'), indent=2), encoding='utf-8')
            with self.assertRaisesRegex(ReferenceHarnessInputError, 'canonical'):
                load_verified_challenge_envelope(path, expected_sha256=expected_hash)

        with self.assertRaisesRegex(ReferenceHarnessInputError, 'expected hash'):
            verify_challenge_envelope(envelope, expected_sha256='0' * 64)

    def test_render_preserves_ordered_system_and_user_messages(self) -> None:
        envelope = _envelope()
        rendered = render_challenge_prompt(envelope).decode('utf-8')

        self.assertLess(rendered.index('BEGIN SYSTEM MESSAGE'), rendered.index('BEGIN USER MESSAGE'))
        self.assertIn(envelope.messages[0].content, rendered)
        self.assertIn(envelope.messages[1].content, rendered)

    def test_cursor_prompt_adds_schema_contract_without_changing_transcript(self) -> None:
        envelope = _envelope()
        schema_json = canonical_json_bytes(Submission.model_json_schema()).decode('utf-8')
        rendered = render_cursor_challenge_prompt(envelope, schema_json)

        self.assertTrue(rendered.startswith(render_challenge_prompt(envelope)))
        self.assertIn(b'BEGIN CURSOR OUTPUT CONTRACT', rendered)
        self.assertIn(schema_json.encode('utf-8'), rendered)

    def test_codex_success_emits_canonical_development_receipt(self) -> None:
        envelope = _envelope()
        submission_bytes = canonical_json_bytes(_submission(envelope))

        def fake_process(argv, **_kwargs):
            response_path = Path(argv[argv.index('--output-last-message') + 1])
            response_path.write_bytes(submission_bytes)
            return BoundedProcessResult(
                exit_code=0,
                duration_ms=10,
                stdout=b'codex diagnostic output',
                stderr=b'',
                termination='exited',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        receipt = run_reference_harness(
            verify_challenge_envelope(envelope),
            harness=ReferenceHarnessName.CODEX,
            requested_model='gpt-test',
            executable='/opt/bin/codex',
            process_runner=fake_process,
            version_resolver=lambda _executable: 'codex-cli test-version',
        )

        self.assertIsNotNone(receipt.submission)
        self.assertIsNone(receipt.failure)
        self.assertTrue(receipt.development_only)
        self.assertFalse(receipt.network_isolation)
        self.assertTrue(receipt.model_weight_contamination_uncontrolled)
        self.assertEqual(receipt.raw_response_bytes, len(submission_bytes))
        self.assertEqual(receipt.command_argv[0], 'codex')
        self.assertNotIn(envelope.messages[1].content, ' '.join(receipt.command_argv))
        self.assertEqual(
            canonical_receipt_bytes(receipt),
            canonical_json_bytes(json.loads(canonical_receipt_bytes(receipt))),
        )

    def test_claude_wrapper_extracts_submission_and_reported_model(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)
        wrapper = canonical_json_bytes(
            {
                'type': 'result',
                'is_error': False,
                'structured_output': submission.model_dump(mode='json'),
                'modelUsage': {'claude-resolved-1': {'inputTokens': 12}},
            }
        )

        def fake_process(argv, **_kwargs):
            self.assertIn('{"mcpServers":{}}', argv)
            return BoundedProcessResult(
                exit_code=0,
                duration_ms=10,
                stdout=wrapper,
                stderr=b'',
                termination='exited',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        receipt = run_reference_harness(
            verify_challenge_envelope(envelope),
            harness=ReferenceHarnessName.CLAUDE,
            requested_model='sonnet',
            executable='/opt/bin/claude',
            process_runner=fake_process,
            version_resolver=lambda _executable: 'test (Claude Code)',
        )

        self.assertEqual(receipt.resolved_model, 'claude-resolved-1')
        self.assertEqual(receipt.submission, submission)
        self.assertIn('--tools=<NO_TOOLS>', receipt.command_argv)
        self.assertNotIn('structured_output', ' '.join(receipt.command_argv))

    def test_cursor_stream_extracts_submission_and_reported_model(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)

        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / 'cursor-agent'
            executable.write_bytes(b'fictional cursor launcher bytes')
            executable.chmod(0o700)

            def fake_process(argv, **kwargs):
                self.assertEqual(argv[0], str(executable.resolve()))
                self.assertEqual(argv[argv.index('--mode') + 1], 'ask')
                self.assertEqual(argv[argv.index('--sandbox') + 1], 'enabled')
                workspace = Path(argv[argv.index('--workspace') + 1])
                self.assertEqual(list(workspace.iterdir()), [])
                self.assertIn(b'BEGIN CURSOR OUTPUT CONTRACT', kwargs['input_bytes'])
                self.assertEqual(kwargs['max_stdout_bytes'], 1_048_576)
                stream = _cursor_stream(
                    submission,
                    prompt=kwargs['input_bytes'],
                    workspace=workspace,
                    between=(
                        {
                            'type': 'assistant',
                            'message': {
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': 'ignored delta'}],
                            },
                            'session_id': 'fictional-session',
                            'model': 'Cursor Model Display',
                        },
                    ),
                    result_updates={'model': 'Cursor Model Display'},
                )
                return BoundedProcessResult(
                    exit_code=0,
                    duration_ms=10,
                    stdout=stream,
                    stderr=b'',
                    termination='exited',
                    stdout_truncated=False,
                    stderr_truncated=False,
                )

            receipt = run_reference_harness(
                verify_challenge_envelope(envelope),
                harness=ReferenceHarnessName.CURSOR,
                requested_model='cursor-model-test',
                executable=str(executable),
                process_runner=fake_process,
                version_resolver=lambda _executable: '2026.test-version',
            )

        self.assertEqual(receipt.schema_version, 'vaxreplay.reference-harness-receipt.v0.4')
        self.assertEqual(receipt.resolved_model, 'Cursor Model Display')
        self.assertEqual(receipt.submission, submission)
        self.assertEqual(
            receipt.harness_executable_sha256,
            hashlib.sha256(b'fictional cursor launcher bytes').hexdigest(),
        )
        self.assertEqual(receipt.harness_executable_sha256_scope, 'invoked_file_bytes')
        self.assertFalse(receipt.harness_executable_identity_attested)
        self.assertFalse(receipt.provider_route_attested)
        self.assertFalse(receipt.sealed_execution)
        self.assertIsNone(receipt.cursor_parse_failure_inventory)
        self.assertIsNone(receipt.harness_runtime_identity)
        self.assertIn('<EMPTY_WORK_DIR>', receipt.command_argv)
        self.assertNotIn(envelope.messages[1].content, ' '.join(receipt.command_argv))

    def test_cursor_rejects_tool_call_even_if_terminal_submission_is_valid(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)

        def fake_process(argv, **kwargs):
            workspace = Path(argv[argv.index('--workspace') + 1])
            stream = _cursor_stream(
                submission,
                prompt=kwargs['input_bytes'],
                workspace=workspace,
                between=(
                    {
                        'type': 'thinking',
                        'subtype': 'delta',
                        'text': 'ignored private reasoning',
                        'session_id': 'fictional-session',
                    },
                    {
                        'type': 'thinking',
                        'subtype': 'completed',
                        'session_id': 'fictional-session',
                    },
                    {
                        'type': 'tool_call',
                        'subtype': 'started',
                        'tool_call': {'webSearchToolCall': {'args': {'query': 'forbidden'}}},
                        'session_id': 'fictional-session',
                    },
                ),
            )
            return BoundedProcessResult(
                exit_code=0,
                duration_ms=10,
                stdout=stream,
                stderr=b'',
                termination='exited',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        receipt = run_reference_harness(
            verify_challenge_envelope(envelope),
            harness=ReferenceHarnessName.CURSOR,
            requested_model='cursor-model-test',
            executable='/opt/bin/cursor-agent',
            process_runner=fake_process,
            version_resolver=lambda _executable: '2026.test-version',
        )

        assert receipt.failure is not None
        self.assertEqual(receipt.failure.code, ReferenceHarnessFailureCode.UNEXPECTED_TOOL_CALL)
        self.assertIsNone(receipt.submission)

    def test_cursor_malformed_and_provider_error_wrappers_fail_closed(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)
        cases = (
            ('malformed', ReferenceHarnessFailureCode.INVALID_WRAPPER),
            ('provider_error', ReferenceHarnessFailureCode.PROVIDER_ERROR),
        )
        for case, expected_code in cases:
            with self.subTest(expected_code=expected_code):

                def fake_process(argv, **kwargs):
                    workspace = Path(argv[argv.index('--workspace') + 1])
                    output = (
                        b'{not-json}\n'
                        if case == 'malformed'
                        else _cursor_stream(
                            submission,
                            prompt=kwargs['input_bytes'],
                            workspace=workspace,
                            result_updates={'is_error': True, 'result': 'provider failed'},
                        )
                    )
                    return BoundedProcessResult(
                        exit_code=0,
                        duration_ms=10,
                        stdout=output,
                        stderr=b'',
                        termination='exited',
                        stdout_truncated=False,
                        stderr_truncated=False,
                    )

                receipt = run_reference_harness(
                    verify_challenge_envelope(envelope),
                    harness=ReferenceHarnessName.CURSOR,
                    requested_model='cursor-model-test',
                    executable='/opt/bin/cursor-agent',
                    process_runner=fake_process,
                    version_resolver=lambda _executable: '2026.test-version',
                )

                assert receipt.failure is not None
                self.assertEqual(receipt.failure.code, expected_code)
                self.assertIsNone(receipt.submission)

    def test_cursor_unknown_events_and_spoofed_transcript_metadata_fail_closed(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)

        def build_case(case: str, prompt: bytes, workspace: Path) -> bytes:
            if case == 'unknown_event':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    between=({'type': 'future_event', 'session_id': 'fictional-session'},),
                )
            if case == 'unknown_subtype':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    between=(
                        {
                            'type': 'assistant',
                            'subtype': 'future_delta',
                            'message': {
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': 'delta'}],
                            },
                            'session_id': 'fictional-session',
                        },
                    ),
                )
            if case == 'session_splice':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    result_updates={'session_id': 'different-session'},
                )
            if case == 'model_splice':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    result_updates={'model': 'Different Model'},
                )
            if case == 'workspace_spoof':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    init_updates={'cwd': str(workspace.parent / 'different-workspace')},
                )
            if case == 'user_spoof':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    user_text='different fictional transcript',
                )
            if case == 'duplicate_init':
                return _cursor_stream(
                    submission,
                    prompt=prompt,
                    workspace=workspace,
                    between=(
                        {
                            'type': 'system',
                            'subtype': 'init',
                            'model': 'Cursor Model Display',
                            'session_id': 'fictional-session',
                            'cwd': str(workspace),
                        },
                    ),
                )
            if case == 'duplicate_json_key':
                valid = _cursor_stream(submission, prompt=prompt, workspace=workspace).splitlines()
                duplicate = (
                    b'{"type":"assistant","type":"assistant","message":'
                    b'{"role":"assistant","content":[{"type":"text","text":"delta"}]},'
                    b'"session_id":"fictional-session"}'
                )
                return b'\n'.join((valid[0], valid[1], duplicate, valid[2]))
            valid = _cursor_stream(submission, prompt=prompt, workspace=workspace).splitlines()
            return b'\n'.join(valid[1:])

        cases = (
            'unknown_event',
            'unknown_subtype',
            'session_splice',
            'model_splice',
            'workspace_spoof',
            'user_spoof',
            'duplicate_init',
            'duplicate_json_key',
            'missing_init',
        )
        for case in cases:
            with self.subTest(case=case):

                def fake_process(argv, **kwargs):
                    workspace = Path(argv[argv.index('--workspace') + 1])
                    return BoundedProcessResult(
                        exit_code=0,
                        duration_ms=10,
                        stdout=build_case(case, kwargs['input_bytes'], workspace),
                        stderr=b'',
                        termination='exited',
                        stdout_truncated=False,
                        stderr_truncated=False,
                    )

                receipt = run_reference_harness(
                    verify_challenge_envelope(envelope),
                    harness=ReferenceHarnessName.CURSOR,
                    requested_model='cursor-model-test',
                    executable='/opt/bin/cursor-agent',
                    process_runner=fake_process,
                    version_resolver=lambda _executable: '2026.test-version',
                )

                assert receipt.failure is not None
                self.assertEqual(receipt.failure.code, ReferenceHarnessFailureCode.INVALID_WRAPPER)
                self.assertIsNone(receipt.submission)
                inventory = receipt.cursor_parse_failure_inventory
                assert inventory is not None
                self.assertGreaterEqual(inventory.nonempty_lines, inventory.parsed_event_lines)
                self.assertEqual(inventory.tool_event_count, 0)
                if case == 'duplicate_json_key':
                    self.assertTrue(inventory.duplicate_json_key_observed)
                    self.assertIsNotNone(inventory.first_unparseable_line)

    def test_cursor_thinking_events_are_ignored_and_submission_remains_valid(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)
        private_reasoning = 'PRIVATE REASONING PAYLOAD MUST NEVER ENTER THE RECEIPT'

        def fake_process(argv, **kwargs):
            workspace = Path(argv[argv.index('--workspace') + 1])
            thinking_deltas = tuple(
                {
                    'type': 'thinking',
                    'subtype': 'delta',
                    'text': f'{private_reasoning} delta {index}',
                    'session_id': 'fictional-session',
                    'timestamp_ms': index + 1,
                }
                for index in range(72)
            )
            stream = _cursor_stream(
                submission,
                prompt=kwargs['input_bytes'],
                workspace=workspace,
                between=(
                    *thinking_deltas,
                    {
                        'type': 'thinking',
                        'subtype': 'completed',
                        'session_id': 'fictional-session',
                        'timestamp_ms': 73,
                    },
                    {
                        'type': 'assistant',
                        'message': {
                            'role': 'assistant',
                            'content': [{'type': 'text', 'text': 'ignored assistant transcript'}],
                        },
                        'session_id': 'fictional-session',
                    },
                ),
            )
            return BoundedProcessResult(
                exit_code=0,
                duration_ms=10,
                stdout=stream,
                stderr=b'',
                termination='exited',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        receipt = run_reference_harness(
            verify_challenge_envelope(envelope),
            harness=ReferenceHarnessName.CURSOR,
            requested_model='cursor-reasoning-model-test',
            executable='/opt/bin/cursor-agent',
            process_runner=fake_process,
            version_resolver=lambda _executable: '2026.test-version',
        )

        self.assertEqual(receipt.submission, submission)
        self.assertIsNone(receipt.failure)
        self.assertIsNone(receipt.cursor_parse_failure_inventory)
        self.assertNotIn(private_reasoning.encode('utf-8'), canonical_receipt_bytes(receipt))

    def test_cursor_invalid_thinking_sequences_fail_closed(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)
        delta = {
            'type': 'thinking',
            'subtype': 'delta',
            'text': 'private reasoning',
            'session_id': 'fictional-session',
        }
        completed = {
            'type': 'thinking',
            'subtype': 'completed',
            'session_id': 'fictional-session',
        }
        assistant = {
            'type': 'assistant',
            'message': {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'ignored assistant transcript'}],
            },
            'session_id': 'fictional-session',
        }
        cases = {
            'missing_subtype': ({'type': 'thinking', 'session_id': 'fictional-session'},),
            'unknown_subtype': ({'type': 'thinking', 'subtype': 'future', 'session_id': 'fictional-session'},),
            'completed_before_delta': (completed, delta),
            'missing_completed': (delta,),
            'duplicate_completed': (delta, completed, completed),
            'delta_after_completed': (delta, completed, delta),
            'thinking_after_assistant': (assistant, delta, completed),
            'inconsistent_session': (
                {**delta, 'session_id': 'different-session'},
                completed,
            ),
            'invalid_model_metadata': (
                {**delta, 'model': 7},
                completed,
            ),
        }

        for case, between in cases.items():
            with self.subTest(case=case):

                def fake_process(argv, **kwargs):
                    workspace = Path(argv[argv.index('--workspace') + 1])
                    return BoundedProcessResult(
                        exit_code=0,
                        duration_ms=10,
                        stdout=_cursor_stream(
                            submission,
                            prompt=kwargs['input_bytes'],
                            workspace=workspace,
                            between=between,
                        ),
                        stderr=b'',
                        termination='exited',
                        stdout_truncated=False,
                        stderr_truncated=False,
                    )

                receipt = run_reference_harness(
                    verify_challenge_envelope(envelope),
                    harness=ReferenceHarnessName.CURSOR,
                    requested_model='cursor-reasoning-model-test',
                    executable='/opt/bin/cursor-agent',
                    process_runner=fake_process,
                    version_resolver=lambda _executable: '2026.test-version',
                )

                assert receipt.failure is not None
                self.assertEqual(receipt.failure.code, ReferenceHarnessFailureCode.INVALID_WRAPPER)
                self.assertIsNone(receipt.submission)

    def test_cursor_runtime_tree_and_chunks_are_bound_when_discoverable(self) -> None:
        envelope = _envelope()
        submission = _submission(envelope)
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory) / 'cursor-runtime'
            runtime.mkdir()
            executable = runtime / 'cursor-agent'
            executable.write_bytes(b'fictional cursor launcher bytes')
            executable.chmod(0o700)
            (runtime / 'index.js').write_bytes(b'fictional cursor entrypoint')
            chunk = runtime / '1683.index.js'
            chunk.write_bytes(b'fictional cursor chunk v1')
            (runtime / 'package.json').write_text(
                '{"name":"@anysphere/agent-cli-runtime","private":true}\n',
                encoding='utf-8',
            )

            def run_once():
                def fake_process(argv, **kwargs):
                    workspace = Path(argv[argv.index('--workspace') + 1])
                    return BoundedProcessResult(
                        exit_code=0,
                        duration_ms=10,
                        stdout=_cursor_stream(
                            submission,
                            prompt=kwargs['input_bytes'],
                            workspace=workspace,
                        ),
                        stderr=b'',
                        termination='exited',
                        stdout_truncated=False,
                        stderr_truncated=False,
                    )

                return run_reference_harness(
                    verify_challenge_envelope(envelope),
                    harness=ReferenceHarnessName.CURSOR,
                    requested_model='cursor-model-test',
                    executable=str(executable),
                    process_runner=fake_process,
                    version_resolver=lambda _executable: '2026.test-version',
                )

            first = run_once()
            chunk.write_bytes(b'fictional cursor chunk v2')
            second = run_once()

        first_runtime = first.harness_runtime_identity
        second_runtime = second.harness_runtime_identity
        assert first_runtime is not None and second_runtime is not None
        self.assertEqual(first_runtime.regular_file_count, 4)
        self.assertEqual(first_runtime.chunk_file_count, 1)
        self.assertEqual(
            first_runtime.entrypoint_sha256,
            hashlib.sha256(b'fictional cursor entrypoint').hexdigest(),
        )
        self.assertFalse(first_runtime.identity_attested)
        self.assertNotEqual(first_runtime.tree_sha256, second_runtime.tree_sha256)
        self.assertNotEqual(first_runtime.chunk_manifest_sha256, second_runtime.chunk_manifest_sha256)
        self.assertEqual(first.harness_executable_sha256, second.harness_executable_sha256)

    def test_timeout_is_a_structured_failure_without_raw_log_text(self) -> None:
        envelope = _envelope()

        def fake_process(argv, **_kwargs):
            del argv
            return BoundedProcessResult(
                exit_code=-9,
                duration_ms=10,
                stdout=b'partial secret-looking response',
                stderr=b'sensitive provider error',
                termination='timed_out',
                stdout_truncated=False,
                stderr_truncated=False,
            )

        receipt = run_reference_harness(
            verify_challenge_envelope(envelope),
            harness=ReferenceHarnessName.CLAUDE,
            requested_model='sonnet',
            process_runner=fake_process,
            version_resolver=lambda _executable: 'test (Claude Code)',
        )

        assert receipt.failure is not None
        self.assertEqual(receipt.failure.code, ReferenceHarnessFailureCode.TIMED_OUT)
        serialized = canonical_receipt_bytes(receipt)
        self.assertNotIn(b'sensitive provider error', serialized)
        self.assertNotIn(b'partial secret-looking response', serialized)
        self.assertGreater(receipt.raw_response_bytes, 0)


if __name__ == '__main__':
    unittest.main()
