from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from vaxreplay.runner._process import run_bounded_process


class RunnerProcessTest(unittest.TestCase):
    def test_captures_stdin_stdout_and_stderr_without_a_shell(self) -> None:
        result = run_bounded_process(
            (
                sys.executable,
                '-c',
                'import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data); sys.stderr.write("log")',
            ),
            input_bytes=b'challenge',
            wall_seconds=5,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            on_abort=lambda: None,
        )

        self.assertEqual(result.termination, 'exited')
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, b'challenge')
        self.assertEqual(result.stderr, b'log')

    def test_response_flood_is_bounded_and_aborted(self) -> None:
        aborted: list[bool] = []
        result = run_bounded_process(
            (sys.executable, '-c', 'import os, time; os.write(1, b"x" * 1000000); time.sleep(5)'),
            input_bytes=b'',
            wall_seconds=5,
            max_stdout_bytes=4096,
            max_stderr_bytes=1024,
            on_abort=lambda: aborted.append(True),
        )

        self.assertEqual(result.termination, 'response_limit')
        self.assertEqual(len(result.stdout), 4096)
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(aborted, [True])

    def test_limit_plus_one_is_detected_before_worker_exit(self) -> None:
        result = run_bounded_process(
            (
                sys.executable,
                '-c',
                'import os, time; os.write(1, b"x" * 4097); time.sleep(5)',
            ),
            input_bytes=b'',
            wall_seconds=5,
            max_stdout_bytes=4096,
            max_stderr_bytes=1024,
            on_abort=lambda: None,
        )

        self.assertEqual(result.termination, 'response_limit')
        self.assertTrue(result.stdout_truncated)
        self.assertLess(result.duration_ms, 2000)

    def test_exact_limit_is_allowed(self) -> None:
        result = run_bounded_process(
            (sys.executable, '-c', 'import os; os.write(1, b"x" * 4096)'),
            input_bytes=b'',
            wall_seconds=5,
            max_stdout_bytes=4096,
            max_stderr_bytes=1024,
            on_abort=lambda: None,
        )

        self.assertEqual(result.termination, 'exited')
        self.assertFalse(result.stdout_truncated)
        self.assertEqual(len(result.stdout), 4096)

    def test_wall_timeout_aborts_process_group(self) -> None:
        aborted: list[bool] = []
        result = run_bounded_process(
            (sys.executable, '-c', 'import time; time.sleep(5)'),
            input_bytes=b'',
            wall_seconds=1,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            on_abort=lambda: aborted.append(True),
        )

        self.assertEqual(result.termination, 'timed_out')
        self.assertEqual(aborted, [True])
        self.assertLess(result.duration_ms, 4000)

    def test_descendants_are_killed_after_the_supervised_parent_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sentinel = Path(temporary_directory) / 'escaped-child'
            child_code = f'import time, pathlib; time.sleep(0.5); pathlib.Path({str(sentinel)!r}).write_text("leak")'
            parent_code = (
                'import subprocess, sys; '
                f'subprocess.Popen([sys.executable, "-c", {child_code!r}], '
                'stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)'
            )
            result = run_bounded_process(
                (sys.executable, '-c', parent_code),
                input_bytes=b'',
                wall_seconds=5,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                on_abort=lambda: None,
            )
            time.sleep(0.8)

            self.assertEqual(result.termination, 'exited')
            self.assertFalse(sentinel.exists())

    def test_passes_only_explicit_inherited_descriptors(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        try:
            os.write(write_descriptor, b'brokered-secret')
            os.close(write_descriptor)
            write_descriptor = -1
            result = run_bounded_process(
                (
                    sys.executable,
                    '-c',
                    f'import os; os.write(1, os.read({read_descriptor}, 64))',
                ),
                input_bytes=b'',
                wall_seconds=5,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                on_abort=lambda: None,
                pass_fds=(read_descriptor,),
            )
        finally:
            os.close(read_descriptor)
            if write_descriptor >= 0:
                os.close(write_descriptor)

        self.assertEqual(result.termination, 'exited')
        self.assertEqual(result.stdout, b'brokered-secret')

    def test_rejects_ambiguous_pass_fd_collections_before_launch(self) -> None:
        for value in ([3], (3, 3), (-1,), (True,)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'pass_fds'):
                run_bounded_process(
                    (sys.executable, '-c', 'raise AssertionError("must not launch")'),
                    input_bytes=b'',
                    wall_seconds=5,
                    max_stdout_bytes=1024,
                    max_stderr_bytes=1024,
                    on_abort=lambda: None,
                    pass_fds=value,  # type: ignore[arg-type]
                )


if __name__ == '__main__':
    unittest.main()
