from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vaxreplay.public_preview import (
    PublicPreviewError,
    PublicPreviewPolicy,
    build_public_preview,
)


def _policy() -> PublicPreviewPolicy:
    approved_paths = (
        'README.md',
        'pyproject.toml',
        'src/vaxreplay/__init__.py',
    )
    approved_paths_sha256 = hashlib.sha256(''.join(f'{path}\n' for path in approved_paths).encode('utf-8')).hexdigest()
    return PublicPreviewPolicy.model_validate(
        {
            'schema_version': 2,
            'release_name': 'v0.1.0-alpha.1',
            'approved_static_path_count': len(approved_paths),
            'approved_static_paths_sha256': approved_paths_sha256,
            'mapped_files': [{'source': 'template.md', 'destination': 'README.md'}],
            'include_files': ['pyproject.toml'],
            'optional_files': ['LICENSE'],
            'include_trees': [
                {
                    'source': 'src/vaxreplay',
                    'destination': 'src/vaxreplay',
                    'exclude': ['excluded.py'],
                }
            ],
            'excluded_components': ['__pycache__'],
            'excluded_globs': ['**/*.pyc'],
            'forbidden_prefixes': ['private/'],
            'forbidden_suffixes': ['.key'],
            'forbidden_text_patterns': [{'name': 'real identity', 'pattern': 'FORBIDDEN-REAL-ID'}],
            'max_file_bytes': 1024,
            'final_required_files': ['LICENSE'],
        }
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / 'source'
    (source / 'src' / 'vaxreplay').mkdir(parents=True)
    (source / 'template.md').write_text('# Sanitized preview\n', encoding='utf-8')
    (source / 'pyproject.toml').write_text('[project]\nname = "preview"\n', encoding='utf-8')
    (source / 'src' / 'vaxreplay' / '__init__.py').write_text('VALUE = 1\n', encoding='utf-8')
    (source / 'src' / 'vaxreplay' / 'excluded.py').write_text('VALUE = 2\n', encoding='utf-8')
    (source / 'src' / 'vaxreplay' / '__pycache__').mkdir()
    (source / 'src' / 'vaxreplay' / '__pycache__' / 'module.pyc').write_bytes(b'cache')
    return source


def test_draft_build_is_allowlisted_marked_and_manifested(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / 'output'

    build = build_public_preview(
        source_root=source,
        output_root=output,
        policy=_policy(),
        draft=True,
        source_revision='a' * 40,
        source_dirty=True,
    )

    assert build.draft
    assert (output / 'README.md').read_text(encoding='utf-8') == '# Sanitized preview\n'
    assert (output / 'src' / 'vaxreplay' / '__init__.py').is_file()
    assert not (output / 'src' / 'vaxreplay' / 'excluded.py').exists()
    assert not (output / 'src' / 'vaxreplay' / '__pycache__').exists()
    assert (output / 'DRAFT-NOT-FOR-DISTRIBUTION.md').is_file()
    manifest = (output / 'MANIFEST.sha256').read_text(encoding='utf-8')
    assert 'README.md' in manifest
    assert 'DRAFT-NOT-FOR-DISTRIBUTION.md' in manifest
    build_info = json.loads((output / 'BUILD-INFO.json').read_text(encoding='utf-8'))
    assert build.static_export_path_count == 3
    assert build.static_export_paths_sha256 == build_info['static_export_paths_sha256']
    assert build.private_export_policy_canonical_sha256 == build_info['private_export_policy_canonical_sha256']


def test_final_build_requires_clean_source_and_release_metadata(tmp_path: Path) -> None:
    source = _source(tmp_path)

    with pytest.raises(PublicPreviewError, match='dirty source tree'):
        build_public_preview(
            source_root=source,
            output_root=tmp_path / 'dirty-output',
            policy=_policy(),
            draft=False,
            source_revision='a' * 40,
            source_dirty=True,
        )
    assert not (tmp_path / 'dirty-output').exists()

    with pytest.raises(PublicPreviewError, match='missing required files'):
        build_public_preview(
            source_root=source,
            output_root=tmp_path / 'unlicensed-output',
            policy=_policy(),
            draft=False,
            source_revision='a' * 40,
            source_dirty=False,
        )
    assert not (tmp_path / 'unlicensed-output').exists()


def test_build_rejects_forbidden_content_and_existing_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    module = source / 'src' / 'vaxreplay' / '__init__.py'
    module.write_text("TRIAL = 'FORBIDDEN-REAL-ID'\n", encoding='utf-8')

    with pytest.raises(PublicPreviewError, match='real identity'):
        build_public_preview(
            source_root=source,
            output_root=tmp_path / 'forbidden-output',
            policy=_policy(),
            draft=True,
            source_revision='a' * 40,
            source_dirty=True,
        )
    assert not (tmp_path / 'forbidden-output').exists()

    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(PublicPreviewError, match='output already exists'):
        build_public_preview(
            source_root=source,
            output_root=existing,
            policy=_policy(),
            draft=True,
            source_revision='a' * 40,
            source_dirty=True,
        )


def test_build_rejects_unreviewed_file_in_included_tree(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / 'src' / 'vaxreplay' / 'new_module.py').write_text('VALUE = 3\n', encoding='utf-8')
    output = tmp_path / 'path-drift-output'

    with pytest.raises(PublicPreviewError, match='static path inventory differs') as error:
        build_public_preview(
            source_root=source,
            output_root=output,
            policy=_policy(),
            draft=True,
            source_revision='a' * 40,
            source_dirty=True,
        )

    assert 'actual count=4' in str(error.value)
    assert not output.exists()


def test_build_rejects_symlinked_included_tree(tmp_path: Path) -> None:
    source = _source(tmp_path)
    tree = source / 'src' / 'vaxreplay'
    external_tree = tmp_path / 'external-vaxreplay'
    tree.rename(external_tree)
    tree.symlink_to(external_tree, target_is_directory=True)
    output = tmp_path / 'symlink-output'

    with pytest.raises(PublicPreviewError, match='symlinks are not allowed'):
        build_public_preview(
            source_root=source,
            output_root=output,
            policy=_policy(),
            draft=True,
            source_revision='a' * 40,
            source_dirty=True,
        )

    assert not output.exists()
