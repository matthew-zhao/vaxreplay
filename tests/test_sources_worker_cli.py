from __future__ import annotations

import io
import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from vaxreplay.sources import worker_cli
from vaxreplay.sources.clinicaltrials import adapt_ctgov_study_candidates, verify_ctgov_source
from vaxreplay.sources.iedb import (
    adapt_tier_a_iedb_antigen_targets,
    verify_tier_a_iedb_source,
)
from vaxreplay.sources.immport import (
    adapt_tier_a_immport_arms,
    verify_tier_a_immport_source,
)


@pytest.mark.parametrize(
    ('entrypoint', 'callback'),
    (
        (worker_cli.run_iedb_source_verifier, verify_tier_a_iedb_source),
        (worker_cli.run_ctgov_source_verifier, verify_ctgov_source),
        (worker_cli.run_immport_source_verifier, verify_tier_a_immport_source),
    ),
)
def test_source_entrypoints_use_generic_hermetic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[[bytes], bytes],
    callback: Callable[..., object],
) -> None:
    observed: list[tuple[bytes, Callable[..., object]]] = []

    def fake(request_bytes: bytes, worker: Callable[..., object]) -> bytes:
        observed.append((request_bytes, worker))
        return b'canonical-response'

    monkeypatch.setattr(worker_cli, 'run_source_verifier_worker', fake)
    assert entrypoint(b'canonical-request') == b'canonical-response'
    assert observed == [(b'canonical-request', callback)]


@pytest.mark.parametrize(
    ('entrypoint', 'callback'),
    (
        (worker_cli.run_iedb_antigen_adapter, adapt_tier_a_iedb_antigen_targets),
        (worker_cli.run_ctgov_study_adapter, adapt_ctgov_study_candidates),
        (worker_cli.run_immport_arm_adapter, adapt_tier_a_immport_arms),
    ),
)
def test_adapter_entrypoints_use_generic_hermetic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[[bytes], bytes],
    callback: Callable[..., object],
) -> None:
    observed: list[tuple[bytes, Callable[..., object]]] = []

    def fake(request_bytes: bytes, worker: Callable[..., object]) -> bytes:
        observed.append((request_bytes, worker))
        return b'canonical-response'

    monkeypatch.setattr(worker_cli, 'run_adapter_worker', fake)
    assert entrypoint(b'canonical-request') == b'canonical-response'
    assert observed == [(b'canonical-request', callback)]


def test_dispatch_rejects_unknown_or_empty_requests() -> None:
    with pytest.raises(ValueError, match='unknown'):
        worker_cli.dispatch('unknown-worker', b'request')
    with pytest.raises(ValueError, match='nonempty'):
        worker_cli.dispatch('iedb-source-verifier', b'')


def test_worker_main_does_not_echo_source_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = 'SOURCE-CONTROLLED-SECRET-CANARY'

    def rejected_source(_worker_name: str, _request_bytes: bytes) -> bytes:
        raise ValueError(f'duplicate source field {secret!r}')

    monkeypatch.setattr(worker_cli, 'dispatch', rejected_source)
    monkeypatch.setattr(
        sys,
        'stdin',
        SimpleNamespace(buffer=io.BytesIO(f'captured body containing {secret}'.encode())),
    )

    assert worker_cli.main(['immport-source-verifier']) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'production source worker rejected input\n'
    assert secret not in captured.out
    assert secret not in captured.err
