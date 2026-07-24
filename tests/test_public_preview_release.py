from __future__ import annotations

from pathlib import Path

import pytest

from vaxreplay.public_preview import (
    PublicPreviewError,
    PublicPreviewPolicy,
    build_public_preview,
)


def _policy() -> PublicPreviewPolicy:
    return PublicPreviewPolicy.model_validate(
        {
            'schema_version': 1,
            'release_name': 'v0.1.0-alpha.1',
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
