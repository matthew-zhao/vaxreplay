from __future__ import annotations

from pathlib import Path

import pytest

import vaxreplay.operations._immutable_tree as immutable_tree_module
from vaxreplay.operations._immutable_tree import ImmutableTreeError, snapshot_immutable_tree


def _snapshot(root: Path, **overrides):
    limits = {
        'max_files': 10,
        'max_directories': 10,
        'max_file_bytes': 1024,
        'max_total_bytes': 4096,
        'max_path_characters': 128,
    }
    limits.update(overrides)
    return snapshot_immutable_tree(root, **limits)


def test_immutable_tree_snapshot_reads_exact_nested_inventory(tmp_path: Path) -> None:
    root = tmp_path / 'artifact'
    (root / 'nested').mkdir(parents=True)
    (root / 'manifest.json').write_bytes(b'{}')
    (root / 'nested' / 'payload.bin').write_bytes(b'payload')

    snapshot = _snapshot(
        root,
        per_path_byte_limits={'manifest.json': 16},
        aggregate_exempt_paths=frozenset({'manifest.json'}),
    )

    assert snapshot.root == root.resolve()
    assert dict(snapshot.files) == {
        'manifest.json': b'{}',
        'nested/payload.bin': b'payload',
    }
    snapshot.require_exact_files({'manifest.json', 'nested/payload.bin'})


def test_immutable_tree_snapshot_rejects_empty_or_unexpected_directories(tmp_path: Path) -> None:
    root = tmp_path / 'artifact'
    (root / 'expected').mkdir(parents=True)
    (root / 'empty').mkdir()
    (root / 'expected' / 'payload.bin').write_bytes(b'payload')

    snapshot = _snapshot(root)

    with pytest.raises(ImmutableTreeError, match='empty, missing, or unexpected'):
        snapshot.require_exact_files({'expected/payload.bin'})


@pytest.mark.parametrize(
    ('entries', 'overrides', 'message'),
    (
        ((('one.bin', b'1'), ('two.bin', b'2')), {'max_files': 1}, 'file count'),
        ((('one/payload.bin', b'1'), ('two/payload.bin', b'2')), {'max_directories': 1}, 'directory count'),
        ((('payload.bin', b'12'),), {'max_file_bytes': 1}, 'byte limit'),
        (
            (('one.bin', b'12'), ('two.bin', b'34')),
            {'max_total_bytes': 3},
            'aggregate byte limit',
        ),
    ),
)
def test_immutable_tree_snapshot_enforces_shape_and_size_bounds(
    tmp_path: Path,
    entries: tuple[tuple[str, bytes], ...],
    overrides: dict[str, int],
    message: str,
) -> None:
    root = tmp_path / 'artifact'
    root.mkdir()
    for relative, payload in entries:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    with pytest.raises(ImmutableTreeError, match=message):
        _snapshot(root, **overrides)


def test_immutable_tree_snapshot_rejects_intermediate_symlink(tmp_path: Path) -> None:
    root = tmp_path / 'artifact'
    outside = tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (outside / 'payload.bin').write_bytes(b'payload')
    (root / 'linked').symlink_to(outside, target_is_directory=True)

    with pytest.raises(ImmutableTreeError, match='symbolic links'):
        _snapshot(root)


def test_immutable_tree_snapshot_rejects_symlink_in_root_parent_chain(tmp_path: Path) -> None:
    real_parent = tmp_path / 'real-parent'
    root = real_parent / 'artifact'
    root.mkdir(parents=True)
    (root / 'payload.bin').write_bytes(b'payload')
    link_parent = tmp_path / 'link-parent'
    link_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ImmutableTreeError, match='without following links'):
        _snapshot(link_parent / 'artifact')


def test_immutable_tree_snapshot_rejects_directory_swap_before_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'artifact'
    stable_parent = root / 'stable-parent'
    declared = stable_parent / 'declared'
    replacement = tmp_path / 'replacement'
    declared.mkdir(parents=True)
    replacement.mkdir()
    (declared / 'payload.bin').write_bytes(b'original')
    (replacement / 'payload.bin').write_bytes(b'replacement')
    original_read = immutable_tree_module._read_regular_file_at
    swapped = False

    def swap_before_open(root_descriptor, relative, max_bytes, inventory):
        nonlocal swapped
        if not swapped:
            swapped = True
            declared.rename(stable_parent / 'displaced')
            replacement.rename(declared)
        return original_read(root_descriptor, relative, max_bytes, inventory)

    monkeypatch.setattr(immutable_tree_module, '_read_regular_file_at', swap_before_open)

    with pytest.raises(ImmutableTreeError, match='directory changed'):
        _snapshot(root)


def test_immutable_tree_snapshot_rejects_file_added_after_initial_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'artifact'
    root.mkdir()
    (root / 'payload.bin').write_bytes(b'payload')
    original_inventory = immutable_tree_module._inventory_tree
    inventory_calls = 0

    def add_file_before_final_scan(root_descriptor, **limits):
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 2:
            (root / 'unmanifested.bin').write_bytes(b'late')
        return original_inventory(root_descriptor, **limits)

    monkeypatch.setattr(immutable_tree_module, '_inventory_tree', add_file_before_final_scan)

    with pytest.raises(ImmutableTreeError, match='changed while being read'):
        _snapshot(root)


def test_immutable_tree_snapshot_rejects_root_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'artifact'
    root.mkdir()
    (root / 'payload.bin').write_bytes(b'payload')
    original_identity_check = immutable_tree_module._require_root_path_identity

    def replace_before_identity_check(resolved, expected):
        resolved.rename(tmp_path / 'displaced')
        resolved.mkdir()
        (resolved / 'payload.bin').write_bytes(b'payload')
        return original_identity_check(resolved, expected)

    monkeypatch.setattr(
        immutable_tree_module,
        '_require_root_path_identity',
        replace_before_identity_check,
    )

    with pytest.raises(ImmutableTreeError, match='root path changed'):
        _snapshot(root)
