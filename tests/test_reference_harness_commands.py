from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vaxreplay.reference_harness.commands import build_claude_command, build_codex_command, build_cursor_command


class ReferenceHarnessCommandTest(unittest.TestCase):
    def test_codex_command_is_ephemeral_read_only_and_prompt_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            command = build_codex_command(
                executable='/opt/bin/codex',
                requested_model='gpt-test',
                work_dir=root,
                schema_path=root / 'schema.json',
                response_path=root / 'response.json',
            )

        self.assertEqual(command.argv[-1], '-')
        self.assertIn('--ephemeral', command.argv)
        self.assertIn('--ignore-user-config', command.argv)
        self.assertIn('--skip-git-repo-check', command.argv)
        self.assertEqual(command.argv[command.argv.index('--sandbox') + 1], 'read-only')
        self.assertEqual(command.argv[command.argv.index('--config') + 1], 'web_search="disabled"')
        self.assertEqual(command.argv[command.argv.index('--model') + 1], 'gpt-test')
        self.assertEqual(command.receipt_argv[0], 'codex')
        self.assertIn('<SUBMISSION_SCHEMA_FILE>', command.receipt_argv)
        self.assertNotIn(temporary_directory, ' '.join(command.receipt_argv))

    def test_claude_command_disables_tools_and_persistence(self) -> None:
        command = build_claude_command(
            executable='/opt/bin/claude',
            requested_model='claude-test',
            submission_schema_json='{"private":"schema-content"}',
            claude_max_budget_usd='1.00',
        )

        self.assertIn('--print', command.argv)
        self.assertIn('--safe-mode', command.argv)
        self.assertIn('--no-session-persistence', command.argv)
        self.assertIn('--tools=', command.argv)
        self.assertEqual(command.argv[command.argv.index('--output-format') + 1], 'json')
        self.assertEqual(command.argv[command.argv.index('--model') + 1], 'claude-test')
        self.assertNotIn('{"private":"schema-content"}', command.receipt_argv)
        self.assertIn('<SUBMISSION_JSON_SCHEMA>', command.receipt_argv)

    def test_cursor_command_uses_ask_mode_sandbox_and_empty_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            command = build_cursor_command(
                executable='/opt/bin/cursor-agent',
                requested_model='cursor-model-test',
                work_dir=Path(temporary_directory),
            )

        self.assertIn('--print', command.argv)
        self.assertEqual(command.argv[command.argv.index('--output-format') + 1], 'stream-json')
        self.assertEqual(command.argv[command.argv.index('--mode') + 1], 'ask')
        self.assertEqual(command.argv[command.argv.index('--sandbox') + 1], 'enabled')
        self.assertIn('--trust', command.argv)
        self.assertEqual(command.argv[command.argv.index('--model') + 1], 'cursor-model-test')
        self.assertNotIn('--force', command.argv)
        self.assertNotIn('--approve-mcps', command.argv)
        self.assertNotIn('--add-dir', command.argv)
        self.assertEqual(command.receipt_argv[0], 'cursor-agent')
        self.assertIn('<EMPTY_WORK_DIR>', command.receipt_argv)
        self.assertNotIn(temporary_directory, ' '.join(command.receipt_argv))


if __name__ == '__main__':
    unittest.main()
