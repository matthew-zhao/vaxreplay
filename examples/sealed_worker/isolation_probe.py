#!/usr/bin/env python3
"""Development canary for the concrete Docker controls used by the example worker."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import worker


def main() -> None:
    failures: list[str] = []
    if os.getuid() == 0 or os.getgid() == 0:
        failures.append('worker-is-root')
    for forbidden_path in (
        Path('/var/run/docker.sock'),
        Path('/workspace'),
        Path('/Users'),
        Path('/host'),
    ):
        if forbidden_path.exists():
            failures.append(f'host-path-visible:{forbidden_path}')
    try:
        Path('/etc/vaxreplay-write-probe').write_text('unexpected', encoding='utf-8')
    except OSError:
        pass
    else:
        failures.append('root-filesystem-writable')
    scratch_probe = Path('/tmp/vaxreplay-scratch-probe')
    try:
        scratch_probe.write_text('expected', encoding='utf-8')
        scratch_probe.unlink()
    except OSError:
        failures.append('scratch-not-writable')
    for family, address in (
        (socket.AF_INET, ('1.1.1.1', 53)),
        (socket.AF_INET6, ('2001:4860:4860::8888', 53)),
    ):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.25)
                if connection.connect_ex(address) == 0:
                    failures.append(f'network-reachable:{address[0]}')
        except OSError:
            pass
    if failures:
        sys.stderr.write(json.dumps({'isolation_probe': 'failed', 'failures': failures}) + '\n')
        raise SystemExit(70)
    sys.stderr.write(json.dumps({'isolation_probe': 'passed'}) + '\n')
    worker.main()


if __name__ == '__main__':
    main()
