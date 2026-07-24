from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vaxreplay.operations.prospective_campaign_archive as archive_module
from tests.test_prospective_release import (
    _build,
    _case_verifier,
    _decision_verifier,
    _source_capture_verifier,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.prospective_campaign_archive import (
    CAMPAIGN_ARCHIVE_FILE_NAME,
    CAMPAIGN_ARCHIVE_INDEX_FILE_NAME,
    MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES,
    BuiltProspectiveCampaignArchive,
    ProspectiveCampaignArchiveError,
    build_prospective_campaign_archive,
    build_verified_prospective_campaign_archive,
    verify_and_materialize_prospective_campaign_archive,
    verify_and_materialize_prospective_campaign_archive_files,
    write_prospective_campaign_archive,
)
from vaxreplay.operations.release_readiness import TierAReleaseScope
from vaxreplay.prospective_release import LoadedProspectiveCohortRelease
from vaxreplay.temporal_schema import TemporalArtifactReceipt

_RELEASE_SCOPE = TierAReleaseScope(
    sources=('immport',),
    tasks=('preclinical_candidate_advancement',),
    includes_model_leaderboard=False,
)


def _source_release(tmp_path: Path) -> LoadedProspectiveCohortRelease:
    fixture_root = tmp_path / 'fixture'
    fixture_root.mkdir()
    return _build(fixture_root)


def _built_archive(
    tmp_path: Path,
) -> tuple[LoadedProspectiveCohortRelease, BuiltProspectiveCampaignArchive]:
    release = _source_release(tmp_path)
    built = build_verified_prospective_campaign_archive(
        release.root,
        expected_release_sha256=release.release_sha256,
        expected_release_id=release.manifest.release_id,
        expected_purpose=release.manifest.purpose,
        expected_release_scope=_RELEASE_SCOPE,
        decision_receipt_verifier=_decision_verifier,
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
    )
    return release, built


def _index_data(built: BuiltProspectiveCampaignArchive) -> dict[str, Any]:
    return cast(dict[str, Any], built.index.model_dump(mode='json'))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_bytes(
    tmp_path: Path,
    *,
    release: LoadedProspectiveCohortRelease,
    built: BuiltProspectiveCampaignArchive,
    archive_bytes: bytes | None = None,
    index_bytes: bytes | None = None,
    expected_archive_sha256: str | None = None,
    expected_index_sha256: str | None = None,
    expected_release_id: str | None = None,
    expected_purpose: str | None = None,
    output_name: str = 'materialized',
    decision_verifier=_decision_verifier,
):
    archive = built.archive_bytes if archive_bytes is None else archive_bytes
    index = built.index_bytes if index_bytes is None else index_bytes
    return verify_and_materialize_prospective_campaign_archive(
        archive_bytes=archive,
        index_bytes=index,
        output_dir=tmp_path / output_name,
        expected_archive_sha256=(
            built.index.archive_sha256 if expected_archive_sha256 is None else expected_archive_sha256
        ),
        expected_index_sha256=built.index_sha256 if expected_index_sha256 is None else expected_index_sha256,
        expected_release_id=release.manifest.release_id if expected_release_id is None else expected_release_id,
        expected_purpose=cast(
            Any,
            release.manifest.purpose if expected_purpose is None else expected_purpose,
        ),
        expected_release_scope=_RELEASE_SCOPE,
        decision_receipt_verifier=decision_verifier,
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
    )


def _write_kwargs(release: LoadedProspectiveCohortRelease) -> dict[str, Any]:
    return {
        'expected_release_sha256': release.release_sha256,
        'expected_release_id': release.manifest.release_id,
        'expected_purpose': release.manifest.purpose,
        'expected_release_scope': _RELEASE_SCOPE,
        'decision_receipt_verifier': _decision_verifier,
        'case_universe_seal_verifier': _case_verifier,
        'source_capture_verifier': _source_capture_verifier,
    }


def test_deterministic_archive_roundtrip_reloads_exact_release(tmp_path: Path) -> None:
    release, first = _built_archive(tmp_path)
    second = build_verified_prospective_campaign_archive(
        release.root,
        **_write_kwargs(release),
    )

    assert second == first
    assert first.index_bytes == canonical_json_bytes(first.index)
    assert first.index_sha256 == _sha256(first.index_bytes)
    assert first.index.archive_sha256 == _sha256(first.archive_bytes)
    assert first.index.archive_byte_count == len(first.archive_bytes)
    assert first.index.release_json_sha256 == release.release_sha256
    assert first.index.prospective_release_manifest_sha256 == release.release_sha256
    assert first.index.release_purpose == release.manifest.purpose
    assert first.index.release_scope == _RELEASE_SCOPE
    assert first.index.release_scope_sha256 == _sha256(canonical_json_bytes(_RELEASE_SCOPE))
    assert tuple(binding.path for binding in first.index.files) == tuple(
        sorted(binding.path for binding in first.index.files)
    )

    cursor = 0
    for binding in first.index.files:
        assert binding.offset == cursor
        original = release.root.joinpath(*PurePosixPath(binding.path).parts).read_bytes()
        assert first.archive_bytes[binding.offset : binding.offset + binding.byte_count] == original
        assert binding.sha256 == _sha256(original)
        cursor += binding.byte_count
    assert cursor == len(first.archive_bytes)

    verified = _verify_bytes(tmp_path, release=release, built=first)
    assert verified.archive_sha256 == first.index.archive_sha256
    assert verified.index_sha256 == first.index_sha256
    assert verified.index == first.index
    assert verified.release.release_sha256 == release.release_sha256
    assert verified.release.verified_admission.admission == release.verified_admission.admission
    assert verified.release.verified_admission.suite == release.verified_admission.suite
    assert verified.release.verified_admission.split_inventory == release.verified_admission.split_inventory
    assert verified.release.verified_admission.case_universe == release.verified_admission.case_universe
    assert tuple(package.manifest_sha256 for package in verified.release.verified_admission.packages) == tuple(
        package.manifest_sha256 for package in release.verified_admission.packages
    )
    assert tuple(seal.manifest_sha256 for seal in verified.release.verified_admission.seals) == tuple(
        seal.manifest_sha256 for seal in release.verified_admission.seals
    )
    assert verified.release.root == (tmp_path / 'materialized').resolve(strict=True)


def test_production_builder_reauthenticates_source_release(tmp_path: Path) -> None:
    release = _source_release(tmp_path)
    with pytest.raises(ProspectiveCampaignArchiveError, match='authenticated prospective cohort release'):
        build_verified_prospective_campaign_archive(
            release.root,
            expected_release_sha256=release.release_sha256,
            expected_release_id=release.manifest.release_id,
            expected_purpose=release.manifest.purpose,
            expected_release_scope=_RELEASE_SCOPE,
            decision_receipt_verifier=lambda _receipt, _proof: False,
            case_universe_seal_verifier=_case_verifier,
            source_capture_verifier=_source_capture_verifier,
        )


def test_official_scope_is_exact_union_of_authenticated_promotion_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"canonical":"promotion-handoff"}'
    descriptor = SimpleNamespace(
        promotion_id='promotion-1',
        capture_index=SimpleNamespace(
            source_verifications=(
                SimpleNamespace(
                    source_id='iedb:epitope-export',
                    result=SimpleNamespace(
                        verifier=SimpleNamespace(verifier_id=archive_module.IEDB_SOURCE_VERIFIER_ID)
                    ),
                ),
                SimpleNamespace(
                    source_id='immport:study-panel',
                    result=SimpleNamespace(
                        verifier=SimpleNamespace(verifier_id=archive_module.IMMPORT_SOURCE_VERIFIER_ID)
                    ),
                ),
                SimpleNamespace(
                    source_id='clinicaltrials-gov:api-v2',
                    result=SimpleNamespace(
                        verifier=SimpleNamespace(verifier_id=archive_module.CTGOV_SOURCE_VERIFIER_ID)
                    ),
                ),
            )
        ),
    )
    real_canonical_json_bytes = archive_module.canonical_json_bytes
    monkeypatch.setattr(
        archive_module,
        'PromotionHandoffDescriptor',
        SimpleNamespace(model_validate_json=lambda candidate: descriptor),
    )
    monkeypatch.setattr(
        archive_module,
        'canonical_json_bytes',
        lambda candidate: payload if candidate is descriptor else real_canonical_json_bytes(candidate),
    )
    release = SimpleNamespace(
        manifest=SimpleNamespace(purpose='official_benchmark'),
        verified_admission=SimpleNamespace(
            admission=SimpleNamespace(purpose='official_benchmark'),
            suite=SimpleNamespace(task_type='preclinical_candidate_advancement'),
            packages=(
                SimpleNamespace(
                    manifest=SimpleNamespace(source_captures=(SimpleNamespace(source_id='promotion:promotion-1'),)),
                    source_capture_artifacts={'promotion:promotion-1': payload},
                ),
            ),
        ),
    )
    exact_scope = TierAReleaseScope(
        sources=('clinicaltrials.gov', 'iedb', 'immport'),
        tasks=('preclinical_candidate_advancement',),
        includes_model_leaderboard=True,
    )

    assert archive_module._derive_official_release_sources(cast(Any, release)) == exact_scope.sources
    archive_module._verify_release_scope(cast(Any, release), exact_scope)
    with pytest.raises(ProspectiveCampaignArchiveError, match='promotion source union'):
        archive_module._verify_release_scope(
            cast(Any, release),
            exact_scope.model_copy(update={'sources': ('iedb', 'immport')}),
        )


@pytest.mark.parametrize(
    ('source_id', 'verifier_id', 'pattern'),
    [
        ('immport:study-panel', 'unregistered-verifier', 'unknown production source verifier'),
        ('immport:study-panel', archive_module.IEDB_SOURCE_VERIFIER_ID, 'differs from its source namespace'),
        ('iedb:', archive_module.IEDB_SOURCE_VERIFIER_ID, 'differs from its source namespace'),
    ],
)
def test_official_source_derivation_rejects_unknown_or_cross_namespace_verifiers(
    source_id: str,
    verifier_id: str,
    pattern: str,
) -> None:
    with pytest.raises(ProspectiveCampaignArchiveError, match=pattern):
        archive_module._readiness_source_for_verifier(
            source_id=source_id,
            verifier_id=verifier_id,
        )


def test_atomic_writer_is_deterministic_and_create_once(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    output = tmp_path / 'published'
    written = write_prospective_campaign_archive(
        release.root,
        output,
        **_write_kwargs(release),
    )
    archive_before = written.archive_path.read_bytes()
    index_before = written.index_path.read_bytes()

    assert set(path.name for path in written.root.iterdir()) == {
        CAMPAIGN_ARCHIVE_FILE_NAME,
        CAMPAIGN_ARCHIVE_INDEX_FILE_NAME,
    }
    assert archive_before == built.archive_bytes
    assert index_before == built.index_bytes
    assert written.index_sha256 == built.index_sha256
    with pytest.raises(ProspectiveCampaignArchiveError, match='already exists'):
        write_prospective_campaign_archive(
            release.root,
            output,
            **_write_kwargs(release),
        )
    assert written.archive_path.read_bytes() == archive_before
    assert written.index_path.read_bytes() == index_before


@pytest.mark.parametrize('delta', [1, -1], ids=['gap', 'overlap'])
def test_rejects_gap_or_overlap_even_with_expected_index_digest(tmp_path: Path, delta: int) -> None:
    release, built = _built_archive(tmp_path)
    data = _index_data(built)
    files = cast(list[dict[str, Any]], data['files'])
    files[1]['offset'] += delta
    index_bytes = canonical_json_bytes(data)

    with pytest.raises(ProspectiveCampaignArchiveError, match='contiguous, ordered, and non-overlapping'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=index_bytes,
            expected_index_sha256=_sha256(index_bytes),
        )
    assert not (tmp_path / 'materialized').exists()


def test_rejects_traversal_path_before_materialization(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    data = _index_data(built)
    files = cast(list[dict[str, Any]], data['files'])
    files[0]['path'] = '../escape'
    index_bytes = canonical_json_bytes(data)

    with pytest.raises(ProspectiveCampaignArchiveError, match='file path is unsafe'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=index_bytes,
            expected_index_sha256=_sha256(index_bytes),
        )
    assert not (tmp_path / 'escape').exists()
    assert not (tmp_path / 'materialized').exists()


def test_rejects_trailing_archive_bytes_even_when_top_level_digest_is_rebound(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    archive_bytes = built.archive_bytes + b'trailing-byte'
    data = _index_data(built)
    data['archive_sha256'] = _sha256(archive_bytes)
    data['archive_byte_count'] = len(archive_bytes)
    index_bytes = canonical_json_bytes(data)

    with pytest.raises(ProspectiveCampaignArchiveError, match='without trailing bytes'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            archive_bytes=archive_bytes,
            index_bytes=index_bytes,
            expected_archive_sha256=_sha256(archive_bytes),
            expected_index_sha256=_sha256(index_bytes),
        )


def test_rejects_tampered_file_range_even_when_archive_and_index_hashes_are_expected(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    binding = next(item for item in built.index.files if item.byte_count > 0)
    tampered = bytearray(built.archive_bytes)
    tampered[binding.offset] ^= 1
    archive_bytes = bytes(tampered)
    data = _index_data(built)
    data['archive_sha256'] = _sha256(archive_bytes)
    index_bytes = canonical_json_bytes(data)

    with pytest.raises(ProspectiveCampaignArchiveError, match='file range differs'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            archive_bytes=archive_bytes,
            index_bytes=index_bytes,
            expected_archive_sha256=_sha256(archive_bytes),
            expected_index_sha256=_sha256(index_bytes),
        )


@pytest.mark.parametrize('which', ['archive', 'index'])
def test_requires_independent_expected_digests(tmp_path: Path, which: str) -> None:
    release, built = _built_archive(tmp_path)
    with pytest.raises(ProspectiveCampaignArchiveError, match='out-of-band expected digest'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            expected_archive_sha256=('0' * 64 if which == 'archive' else built.index.archive_sha256),
            expected_index_sha256='0' * 64 if which == 'index' else built.index_sha256,
        )


def test_rejects_wrong_expected_release_id(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    with pytest.raises(ProspectiveCampaignArchiveError, match='different expected release'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            expected_release_id='another-release',
        )
    assert not (tmp_path / 'materialized').exists()


def test_rejects_index_release_id_that_disagrees_with_release_json(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    data = _index_data(built)
    data['release_id'] = 'another-release'
    index_bytes = canonical_json_bytes(data)
    with pytest.raises(ProspectiveCampaignArchiveError, match='identify different releases'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=index_bytes,
            expected_index_sha256=_sha256(index_bytes),
            expected_release_id='another-release',
        )


def test_tier_a_purpose_mismatch_fails_before_leaving_output(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    data = _index_data(built)
    data['release_purpose'] = 'official_benchmark'
    index_bytes = canonical_json_bytes(data)

    with pytest.raises(ProspectiveCampaignArchiveError, match='different release purposes'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=index_bytes,
            expected_index_sha256=_sha256(index_bytes),
            expected_purpose='official_benchmark',
        )
    assert release.manifest.purpose == 'prospective_research'
    assert not (tmp_path / 'materialized').exists()


def test_rejects_wrong_tree_digest_and_noncanonical_index(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    wrong_tree = _index_data(built)
    wrong_tree['tree_sha256'] = '0' * 64
    wrong_tree_bytes = canonical_json_bytes(wrong_tree)
    with pytest.raises(ProspectiveCampaignArchiveError, match='tree digest differs'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=wrong_tree_bytes,
            expected_index_sha256=_sha256(wrong_tree_bytes),
        )

    noncanonical = json.dumps(_index_data(built), indent=2, sort_keys=True).encode()
    with pytest.raises(ProspectiveCampaignArchiveError, match='canonical JSON'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            index_bytes=noncanonical,
            expected_index_sha256=_sha256(noncanonical),
            output_name='noncanonical-output',
        )


def test_rejects_source_symlink_empty_directory_and_oversized_file(tmp_path: Path) -> None:
    for kind in ('symlink', 'empty-directory', 'oversized-file'):
        case_root = tmp_path / kind
        case_root.mkdir()
        release = _build(case_root)
        if kind == 'symlink':
            os.symlink('release.json', release.root / 'uncommitted-link')
            pattern = 'symbolic links'
        elif kind == 'empty-directory':
            (release.root / 'uncommitted-directory').mkdir()
            pattern = 'unexpected directory'
        else:
            with (release.root / 'oversized.bin').open('wb') as output:
                output.truncate(MAX_PROSPECTIVE_CAMPAIGN_ARCHIVE_BYTES + 1)
            pattern = 'byte limit'
        with pytest.raises(ProspectiveCampaignArchiveError, match=pattern):
            build_prospective_campaign_archive(release.root, release_scope=_RELEASE_SCOPE)


def test_file_verifier_rejects_symlink_input(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    archive_path = tmp_path / 'archive.bin'
    index_path = tmp_path / 'index.json'
    archive_path.write_bytes(built.archive_bytes)
    index_path.write_bytes(built.index_bytes)
    archive_link = tmp_path / 'archive-link.bin'
    os.symlink(archive_path.name, archive_link)

    with pytest.raises(ProspectiveCampaignArchiveError, match='without following links'):
        verify_and_materialize_prospective_campaign_archive_files(
            archive_path=archive_link,
            index_path=index_path,
            output_dir=tmp_path / 'materialized',
            expected_archive_sha256=built.index.archive_sha256,
            expected_index_sha256=built.index_sha256,
            expected_release_id=release.manifest.release_id,
            expected_purpose=release.manifest.purpose,
            expected_release_scope=_RELEASE_SCOPE,
            decision_receipt_verifier=_decision_verifier,
            case_universe_seal_verifier=_case_verifier,
            source_capture_verifier=_source_capture_verifier,
        )


def test_materializer_never_overwrites_existing_directory_or_symlink(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    first = _verify_bytes(tmp_path, release=release, built=built)
    release_json_before = (first.release.root / 'release.json').read_bytes()
    with pytest.raises(ProspectiveCampaignArchiveError, match='already exists'):
        _verify_bytes(tmp_path, release=release, built=built)
    assert (first.release.root / 'release.json').read_bytes() == release_json_before

    outside = tmp_path / 'outside'
    outside.mkdir()
    symlink_output = tmp_path / 'symlink-output'
    os.symlink(outside, symlink_output, target_is_directory=True)
    with pytest.raises(ProspectiveCampaignArchiveError, match='already exists'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            output_name='symlink-output',
        )
    assert list(outside.iterdir()) == []


def test_consumer_semantic_reverification_failure_cleans_staging(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    with pytest.raises(ProspectiveCampaignArchiveError, match='valid prospective cohort release'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            decision_verifier=lambda _receipt, _proof: False,
        )
    assert not (tmp_path / 'materialized').exists()
    assert not tuple(tmp_path.glob('.materialized.staging-*'))


def test_post_rename_semantic_reload_failure_removes_exact_installed_tree(tmp_path: Path) -> None:
    release, built = _built_archive(tmp_path)
    verifier_calls = 0

    def fail_only_post_rename(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
        nonlocal verifier_calls
        verifier_calls += 1
        return not (tmp_path / 'materialized').exists() and _decision_verifier(receipt, proof)

    with pytest.raises(ProspectiveCampaignArchiveError, match='valid prospective cohort release'):
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            decision_verifier=fail_only_post_rename,
        )

    assert verifier_calls >= 2
    assert not (tmp_path / 'materialized').exists()
    assert not tuple(tmp_path.glob('.materialized.staging-*'))
    assert not tuple(tmp_path.glob('.vaxreplay-cleanup-*'))


def test_post_rename_root_replacement_is_left_untouched_and_cleanup_fails_closed(
    tmp_path: Path,
) -> None:
    release, built = _built_archive(tmp_path)
    verifier_calls = 0
    target = tmp_path / 'materialized'
    displaced = tmp_path / 'displaced-owned-release'
    marker = b'unrelated replacement content'

    def replace_root_then_reject(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
        nonlocal verifier_calls
        verifier_calls += 1
        if not target.exists():
            return _decision_verifier(receipt, proof)
        os.rename(target, displaced)
        target.mkdir()
        (target / 'do-not-delete.txt').write_bytes(marker)
        return False

    with pytest.raises(ProspectiveCampaignArchiveError, match='cleanup failed closed') as captured:
        _verify_bytes(
            tmp_path,
            release=release,
            built=built,
            decision_verifier=replace_root_then_reject,
        )

    assert 'replacement left untouched' in str(captured.value)
    assert verifier_calls >= 2
    assert (target / 'do-not-delete.txt').read_bytes() == marker
    assert (displaced / 'release.json').is_file()
    assert not tuple(tmp_path.glob('.vaxreplay-cleanup-*'))
