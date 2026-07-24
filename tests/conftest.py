"""Explicit test-only fixtures shared by lifecycle contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest


@pytest.fixture
def synthetic_official_replay_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy synthetic lifecycle fixtures reloadable as explicitly official.

    Production official admission is promotion-backed.  The lifecycle fixtures predate
    that factory and exercise downstream proof composition, so only modules which opt
    into this fixture receive the historical force-official reconstruction shim.  The
    pytest monkeypatch is scoped to one test and is restored even when a tamper case
    fails mid-load.
    """

    import vaxreplay.operations.prospective_campaign_archive as archive_module
    import vaxreplay.prospective_release as release_module

    real_builder = release_module.build_verified_prospective_admission

    def force_official(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        rebuilt = real_builder(*args, **kwargs)
        return replace(
            rebuilt,
            admission=rebuilt.admission.model_copy(update={'purpose': 'official_benchmark'}),
        )

    monkeypatch.setattr(
        release_module,
        'build_verified_prospective_admission',
        force_official,
    )
    monkeypatch.setattr(
        archive_module,
        '_derive_official_release_sources',
        lambda _release: ('immport',),
    )
