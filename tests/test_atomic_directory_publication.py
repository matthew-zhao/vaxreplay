from __future__ import annotations

import os
from pathlib import Path

import pytest

import vaxreplay._atomic as atomic_module
from vaxreplay._atomic import (
    AtomicDirectoryPublication,
    AtomicDirectoryPublicationError,
)


def test_descriptor_relative_publication_roundtrip_is_create_once(tmp_path: Path) -> None:
    target = tmp_path / 'artifact'
    with AtomicDirectoryPublication.create(target) as publication:
        publication.make_directory('nested')
        publication.write_bytes('nested/payload.bin', b'bound payload')
        installed = publication.publish()
        assert (installed / 'nested' / 'payload.bin').read_bytes() == b'bound payload'
        publication.commit()

    assert tuple(path.name for path in tmp_path.iterdir()) == ('artifact',)
    with pytest.raises(AtomicDirectoryPublicationError, match='already exists'):
        AtomicDirectoryPublication.create(target)


def test_missing_parent_is_never_created_implicitly(tmp_path: Path) -> None:
    missing_parent = tmp_path / 'not-preauthenticated'

    with pytest.raises(AtomicDirectoryPublicationError, match='parent must already exist'):
        AtomicDirectoryPublication.create(missing_parent / 'artifact')

    assert not missing_parent.exists()


def test_writable_shared_parent_is_rejected(tmp_path: Path) -> None:
    original_mode = tmp_path.stat().st_mode & 0o777
    tmp_path.chmod(original_mode | 0o020)
    try:
        with pytest.raises(AtomicDirectoryPublicationError, match='not group/world writable'):
            AtomicDirectoryPublication.create(tmp_path / 'artifact')
    finally:
        tmp_path.chmod(original_mode)


def test_constructor_failure_removes_exact_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'artifact'
    original_directory_identity = atomic_module._directory_identity

    def fail_after_opening_tree(metadata, label):  # noqa: ANN001
        if label == 'opened private publication tree':
            raise AtomicDirectoryPublicationError('injected constructor failure')
        return original_directory_identity(metadata, label)

    monkeypatch.setattr(atomic_module, '_directory_identity', fail_after_opening_tree)
    with pytest.raises(AtomicDirectoryPublicationError, match='injected constructor failure'):
        AtomicDirectoryPublication.create(target)

    assert not target.exists()
    assert not tuple(tmp_path.glob('.artifact.vaxreplay-private-*'))


def test_replaced_public_staging_name_cannot_redirect_privileged_writes(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'artifact'
    outside = tmp_path / 'outside'
    outside.mkdir()
    displaced = tmp_path / 'displaced-private-container'

    with pytest.raises(AtomicDirectoryPublicationError, match='container was replaced'):
        with AtomicDirectoryPublication.create(target) as publication:
            public_container = publication.parent / publication._container_name
            os.rename(public_container, displaced)
            os.symlink(outside, public_container, target_is_directory=True)
            publication.write_bytes('target.json', b'operation-owned bytes')
            publication.publish()
            publication.commit()

    assert list(outside.iterdir()) == []
    assert (target / 'target.json').read_bytes() == b'operation-owned bytes'
    assert displaced.is_dir()


def test_target_race_is_never_overwritten_and_owned_staging_is_removed(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'artifact'
    with pytest.raises(AtomicDirectoryPublicationError, match='already exists'):
        with AtomicDirectoryPublication.create(target) as publication:
            publication.write_bytes('payload.bin', b'owned')
            target.mkdir()
            (target / 'unrelated.bin').write_bytes(b'unrelated')
            publication.publish()

    assert (target / 'unrelated.bin').read_bytes() == b'unrelated'
    assert not tuple(tmp_path.glob('.artifact.vaxreplay-private-*'))


def test_post_publish_replacement_is_left_untouched_and_cleanup_fails_closed(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'artifact'
    displaced = tmp_path / 'displaced-owned-artifact'
    marker = b'unrelated replacement'

    with pytest.raises(AtomicDirectoryPublicationError, match='replacement left untouched'):
        with AtomicDirectoryPublication.create(target) as publication:
            publication.write_bytes('payload.bin', b'owned')
            publication.publish()
            os.rename(target, displaced)
            target.mkdir()
            (target / 'marker.bin').write_bytes(marker)
            raise RuntimeError('force post-publication cleanup')

    assert (target / 'marker.bin').read_bytes() == marker
    assert (displaced / 'payload.bin').read_bytes() == b'owned'


def test_source_swap_during_publish_is_restored_without_installing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'artifact'
    original_rename = atomic_module._rename_directory_noreplace_at
    raced = False

    def racing_rename(source_descriptor, source_name, target_descriptor, target_name):  # noqa: ANN001
        nonlocal raced
        if not raced and source_name == 'owned-tree' and target_name == target.name:
            raced = True
            os.rename(
                'owned-tree',
                'displaced-owned-tree',
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            os.mkdir('owned-tree', mode=0o700, dir_fd=source_descriptor)
            replacement_descriptor = os.open(
                'owned-tree/replacement.bin',
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_descriptor,
            )
            try:
                os.write(replacement_descriptor, b'unrelated replacement')
            finally:
                os.close(replacement_descriptor)
        return original_rename(source_descriptor, source_name, target_descriptor, target_name)

    monkeypatch.setattr(atomic_module, '_rename_directory_noreplace_at', racing_rename)
    with pytest.raises(AtomicDirectoryPublicationError, match='cleanup failed closed'):
        with AtomicDirectoryPublication.create(target) as publication:
            publication.write_bytes('payload.bin', b'owned')
            publication.publish()

    assert not target.exists()
    private_containers = tuple(tmp_path.glob('.artifact.vaxreplay-private-*'))
    assert len(private_containers) == 1
    assert (private_containers[0] / 'owned-tree' / 'replacement.bin').read_bytes() == b'unrelated replacement'
    assert (private_containers[0] / 'displaced-owned-tree' / 'payload.bin').read_bytes() == b'owned'


def test_source_swap_during_cleanup_restores_unrelated_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'artifact'
    displaced = tmp_path / 'displaced-owned-artifact'
    original_rename = os.rename
    raced = False

    def racing_rename(source, destination, *args, **kwargs):  # noqa: ANN001
        nonlocal raced
        if not raced and source == target.name and destination == 'failed-tree':
            raced = True
            source_descriptor = kwargs['src_dir_fd']
            original_rename(
                source,
                displaced.name,
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            os.mkdir(source, mode=0o700, dir_fd=source_descriptor)
            replacement_descriptor = os.open(
                f'{source}/replacement.bin',
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_descriptor,
            )
            try:
                os.write(replacement_descriptor, b'unrelated replacement')
            finally:
                os.close(replacement_descriptor)
        return original_rename(source, destination, *args, **kwargs)

    with pytest.raises(AtomicDirectoryPublicationError, match='replacement was restored'):
        with AtomicDirectoryPublication.create(target) as publication:
            publication.write_bytes('payload.bin', b'owned')
            publication.publish()
            monkeypatch.setattr(os, 'rename', racing_rename)
            raise RuntimeError('force cleanup')

    assert (target / 'replacement.bin').read_bytes() == b'unrelated replacement'
    assert (displaced / 'payload.bin').read_bytes() == b'owned'


def test_container_swap_during_cleanup_restores_unrelated_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'artifact'
    original_rename = atomic_module._rename_directory_noreplace_at
    raced = False
    public_container_name = ''

    def racing_rename(source_descriptor, source_name, target_descriptor, target_name):  # noqa: ANN001
        nonlocal raced
        if (
            not raced
            and source_name == public_container_name
            and target_name.startswith(f'{public_container_name}.cleanup-')
        ):
            raced = True
            os.rename(
                source_name,
                'displaced-private-container',
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            os.mkdir(source_name, mode=0o700, dir_fd=source_descriptor)
            replacement_descriptor = os.open(
                f'{source_name}/replacement.bin',
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_descriptor,
            )
            try:
                os.write(replacement_descriptor, b'unrelated replacement')
            finally:
                os.close(replacement_descriptor)
        return original_rename(source_descriptor, source_name, target_descriptor, target_name)

    with pytest.raises(AtomicDirectoryPublicationError, match='replacement was restored'):
        with AtomicDirectoryPublication.create(target) as publication:
            public_container_name = publication._container_name
            publication.write_bytes('payload.bin', b'owned')
            publication.publish()
            publication.commit()
            monkeypatch.setattr(atomic_module, '_rename_directory_noreplace_at', racing_rename)

    assert (target / 'payload.bin').read_bytes() == b'owned'
    assert (tmp_path / public_container_name / 'replacement.bin').read_bytes() == b'unrelated replacement'
    assert (tmp_path / 'displaced-private-container').is_dir()
