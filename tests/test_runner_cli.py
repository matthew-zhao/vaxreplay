from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.runner.cli import main
from vaxreplay.runner.orchestrator import receipt_key_id
from vaxreplay.runner.schema import IsolationTier, RunnerPolicy, SystemSubmissionManifest


class RunnerCliTest(unittest.TestCase):
    def test_hash_commands_canonicalize_system_and_policy(self) -> None:
        system = SystemSubmissionManifest(
            submission_id='system-1',
            image_ref='sha256:' + 'a' * 64,
            entrypoint=('/opt/vaxreplay/run',),
            model_id='model-1',
            harness_id='harness-1',
        )
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            system_path = root / 'system.json'
            policy_path = root / 'policy.json'
            system_path.write_text(system.model_dump_json(indent=2), encoding='utf-8')
            policy_path.write_text(policy.model_dump_json(indent=2), encoding='utf-8')

            for argv, key, value in (
                (
                    ['vaxreplay-runner', 'system-hash', '--system-manifest', str(system_path)],
                    'system_manifest_sha256',
                    system,
                ),
                (
                    ['vaxreplay-runner', 'policy-hash', '--policy', str(policy_path)],
                    'policy_sha256',
                    policy,
                ),
            ):
                with self.subTest(command=argv[1]):
                    output = io.StringIO()
                    with patch.object(sys, 'argv', argv), redirect_stdout(output):
                        main()
                    self.assertEqual(
                        json.loads(output.getvalue())[key],
                        hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
                    )

    def test_receipt_key_id_command(self) -> None:
        key = bytes(range(32))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'receipt-key.hex'
            path.write_text(key.hex() + '\n', encoding='ascii')
            output = io.StringIO()
            with (
                patch.object(
                    sys,
                    'argv',
                    ['vaxreplay-runner', 'receipt-key-id', '--receipt-key', str(path)],
                ),
                redirect_stdout(output),
            ):
                main()

            self.assertEqual(json.loads(output.getvalue())['receipt_key_id'], receipt_key_id(key))


if __name__ == '__main__':
    unittest.main()
