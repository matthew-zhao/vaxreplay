from __future__ import annotations

import pytest

from vaxreplay.qa.attack_catalog import (
    AttackCatalog,
    AttackFamily,
    attack_catalog_sha256,
    default_attack_catalog,
)


def test_default_catalog_covers_every_required_attack_family() -> None:
    catalog = default_attack_catalog()

    assert {attack.family for attack in catalog.attacks} == set(AttackFamily)
    assert tuple(attack.attack_id for attack in catalog.attacks) == tuple(
        sorted(attack.attack_id for attack in catalog.attacks)
    )
    assert len(attack_catalog_sha256(catalog)) == 64


def test_attack_catalog_rejects_duplicate_or_unsorted_ids() -> None:
    catalog = default_attack_catalog()

    with pytest.raises(ValueError, match='sorted'):
        AttackCatalog(catalog_id='unsorted', attacks=tuple(reversed(catalog.attacks)))
    with pytest.raises(ValueError, match='unique'):
        AttackCatalog(catalog_id='duplicate', attacks=(*catalog.attacks, catalog.attacks[-1]))


def test_attack_catalog_rejects_missing_families() -> None:
    catalog = default_attack_catalog()

    with pytest.raises(ValueError, match='omits required families'):
        AttackCatalog(catalog_id='incomplete', attacks=catalog.attacks[:1])
