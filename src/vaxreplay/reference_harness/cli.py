"""Command-line entry point for development-only local reference runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaxreplay.reference_harness.runner import (
    canonical_receipt_bytes,
    load_verified_challenge_envelope,
    run_reference_harness,
)
from vaxreplay.reference_harness.schema import (
    ReferenceHarnessLimits,
    ReferenceHarnessName,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one VaxReplay envelope through a development-only local Codex, Claude, or Cursor CLI',
    )
    parser.add_argument('--envelope', required=True, type=Path)
    parser.add_argument('--expected-envelope-sha256', required=True)
    parser.add_argument('--harness', required=True, choices=[item.value for item in ReferenceHarnessName])
    parser.add_argument('--model', required=True)
    parser.add_argument('--executable')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--wall-seconds', type=int, default=600)
    parser.add_argument('--max-response-bytes', type=int, default=1_048_576)
    parser.add_argument('--max-cli-stdout-bytes', type=int, default=1_048_576)
    parser.add_argument('--max-cli-stderr-bytes', type=int, default=1_048_576)
    parser.add_argument('--claude-max-budget-usd', default='1.00')
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    verified = load_verified_challenge_envelope(
        arguments.envelope,
        expected_sha256=arguments.expected_envelope_sha256,
    )
    receipt = run_reference_harness(
        verified,
        harness=ReferenceHarnessName(arguments.harness),
        requested_model=arguments.model,
        executable=arguments.executable,
        limits=ReferenceHarnessLimits(
            wall_seconds=arguments.wall_seconds,
            max_response_bytes=arguments.max_response_bytes,
            max_cli_stdout_bytes=arguments.max_cli_stdout_bytes,
            max_cli_stderr_bytes=arguments.max_cli_stderr_bytes,
        ),
        claude_max_budget_usd=arguments.claude_max_budget_usd,
    )
    output = canonical_receipt_bytes(receipt) + b'\n'
    if arguments.receipt is None:
        sys.stdout.buffer.write(output)
    else:
        with arguments.receipt.open('xb') as destination:
            destination.write(output)
    if receipt.failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
