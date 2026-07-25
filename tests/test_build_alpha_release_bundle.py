from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from scripts import build_alpha_release_bundle as release


def _record_row(name: str, payload: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode('ascii')
    return [name, f'sha256={digest}', str(len(payload))]


def _wheel(path: Path, *, bad_digest: bool = False) -> None:
    package_path = 'example/__init__.py'
    metadata_path = 'example-1.0.dist-info/METADATA'
    record_path = 'example-1.0.dist-info/RECORD'
    files = {
        package_path: b'VALUE = 1\n',
        metadata_path: b'Metadata-Version: 2.4\nName: example\nVersion: 1.0\n',
    }
    rows = [_record_row(name, payload) for name, payload in files.items()]
    if bad_digest:
        rows[0][1] = f'sha256={"a" * 43}'
    rows.append([record_path, '', ''])
    stream = io.StringIO()
    csv.writer(stream, lineterminator='\n').writerows(rows)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(record_path, stream.getvalue().encode())


def _core_metadata() -> bytes:
    return (
        'Metadata-Version: 2.4\n'
        'Name: vaxreplay\n'
        f'Version: {release.PROJECT_VERSION}\n'
        'License-Expression: Apache-2.0\n'
        'Requires-Python: >=3.12\n'
        'Requires-Dist: cryptography>=43\n'
        'Requires-Dist: pydantic<3,>=2.10\n'
        'Requires-Dist: pytest>=8; extra == "dev"\n'
        '\n'
    ).encode()


def _entry_points() -> bytes:
    lines = ['[console_scripts]']
    lines.extend(f'{name} = {target}' for name, target in sorted(release._EXPECTED_ENTRY_POINTS.items()))
    return ('\n'.join(lines) + '\n').encode()


def _release_wheel(path: Path, source: Path) -> None:
    dist_info = f'vaxreplay-{release.PROJECT_VERSION}.dist-info'
    record_path = f'{dist_info}/RECORD'
    files = {
        'vaxreplay/__init__.py': b'__version__ = "0.1.0a1"\n',
        f'{dist_info}/METADATA': _core_metadata(),
        f'{dist_info}/WHEEL': (
            b'Wheel-Version: 1.0\nGenerator: focused-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n'
        ),
        f'{dist_info}/entry_points.txt': _entry_points(),
        f'{dist_info}/licenses/LICENSE': (source / 'LICENSE').read_bytes(),
        f'{dist_info}/licenses/NOTICE': (source / 'NOTICE').read_bytes(),
    }
    rows = [_record_row(name, payload) for name, payload in sorted(files.items())]
    rows.append([record_path, '', ''])
    stream = io.StringIO()
    csv.writer(stream, lineterminator='\n').writerows(rows)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(record_path, stream.getvalue().encode())


def _pyproject() -> bytes:
    scripts = '\n'.join(
        f'{json.dumps(name)} = {json.dumps(target)}' for name, target in sorted(release._EXPECTED_ENTRY_POINTS.items())
    )
    return (
        '[project]\n'
        'name = "vaxreplay"\n'
        f'version = "{release.PROJECT_VERSION}"\n'
        'dependencies = ["cryptography>=43", "pydantic>=2.10,<3"]\n'
        '\n'
        '[project.scripts]\n'
        f'{scripts}\n'
    ).encode()


def _release_sdist(
    path: Path,
    source: Path,
    *,
    reverse: bool,
    mtime: int,
    extra_file: tuple[str, bytes] | None = None,
) -> None:
    root = f'vaxreplay-{release.PROJECT_VERSION}'
    files = {
        f'{root}/LICENSE': (source / 'LICENSE').read_bytes(),
        f'{root}/NOTICE': (source / 'NOTICE').read_bytes(),
        f'{root}/PKG-INFO': _core_metadata(),
        f'{root}/pyproject.toml': _pyproject(),
        f'{root}/src/vaxreplay/__init__.py': b'__version__ = "0.1.0a1"\n',
        f'{root}/src/vaxreplay.egg-info/entry_points.txt': _entry_points(),
    }
    if extra_file is not None:
        name, payload = extra_file
        files[f'{root}/{name}'] = payload
    directories = {
        root,
        f'{root}/src',
        f'{root}/src/vaxreplay',
        f'{root}/src/vaxreplay.egg-info',
    }
    for name in files:
        parent = PurePosixPath(name).parent
        while parent.as_posix() != '.':
            directories.add(parent.as_posix())
            parent = parent.parent
    members = [(name, None) for name in directories] + list(files.items())
    members.sort(key=lambda item: item[0], reverse=reverse)
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.uid = 501
            member.gid = 20
            member.mtime = mtime
            if payload is None:
                member.type = tarfile.DIRTYPE
                member.mode = 0o700
                archive.addfile(member)
            else:
                member.mode = 0o600
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ('git', *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _minimal_public_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / 'repo'
    root.mkdir()
    _git(root, 'init', '-q')
    _git(root, 'config', 'user.name', 'Release Test')
    _git(root, 'config', 'user.email', 'release-test@example.invalid')
    build_info = {
        'schema_version': 2,
        'release_name': release.RELEASE,
        'draft': False,
        'source_revision': 'b' * 40,
        'source_dirty': False,
        'file_count_before_generated_metadata': 1,
        'private_export_policy_canonical_sha256': 'c' * 64,
        'static_export_path_count': 1,
        'static_export_paths_sha256': hashlib.sha256(b'README.md\n').hexdigest(),
    }
    (root / 'README.md').write_text('# Public export\n', encoding='utf-8')
    (root / 'BUILD-INFO.json').write_text(
        json.dumps(build_info, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    build_info_digest = hashlib.sha256((root / 'BUILD-INFO.json').read_bytes()).hexdigest()
    readme_digest = hashlib.sha256((root / 'README.md').read_bytes()).hexdigest()
    (root / 'MANIFEST.sha256').write_text(
        f'{build_info_digest}  BUILD-INFO.json\n{readme_digest}  README.md\n',
        encoding='utf-8',
    )
    _git(root, 'add', 'BUILD-INFO.json', 'MANIFEST.sha256', 'README.md')
    _git(root, 'commit', '-q', '-m', 'public export')
    return root, _git(root, 'rev-parse', 'HEAD')


@pytest.mark.parametrize(
    'member',
    (
        '/absolute',
        '../escape',
        'one/../../escape',
        'one//two',
        r'one\\two',
        'C:/drive',
        'bad\u0085name',
    ),
)
def test_archive_member_path_rejects_unsafe_names(member: str) -> None:
    with pytest.raises(release.ReleaseBundleError, match='unsafe archive member'):
        release._safe_member_path(member)


def test_wheel_record_is_fully_verified(tmp_path: Path) -> None:
    valid = tmp_path / 'valid.whl'
    invalid = tmp_path / 'invalid.whl'
    _wheel(valid)
    _wheel(invalid, bad_digest=True)

    release._verify_wheel_record(valid)
    assert {entry['path'] for entry in release._inspect_zip(valid)} == {
        'example/__init__.py',
        'example-1.0.dist-info/METADATA',
        'example-1.0.dist-info/RECORD',
    }
    with pytest.raises(release.ReleaseBundleError, match='RECORD mismatch'):
        release._verify_wheel_record(invalid)


def test_tar_inventory_rejects_links(tmp_path: Path) -> None:
    archive_path = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        link = tarfile.TarInfo('root/link')
        link.type = tarfile.SYMTYPE
        link.linkname = '../../outside'
        archive.addfile(link)

    with pytest.raises(release.ReleaseBundleError, match='unsupported tar member type'):
        release._inspect_tar(archive_path)


def test_zip_inventory_rejects_directory_symlinks_and_control_characters(tmp_path: Path) -> None:
    symlink = tmp_path / 'symlink.zip'
    with zipfile.ZipFile(symlink, 'w') as archive:
        member = zipfile.ZipInfo('linked-directory/')
        member.external_attr = (0o120777 << 16) | 0x10
        archive.writestr(member, b'')
    with pytest.raises(release.ReleaseBundleError, match='unsupported zip member type'):
        release._inspect_zip(symlink)

    control = tmp_path / 'control.zip'
    with zipfile.ZipFile(control, 'w') as archive:
        archive.writestr('bad\nname', b'unsafe')
    with pytest.raises(release.ReleaseBundleError, match='unsafe archive member'):
        release._inspect_zip(control)


def test_exact_source_verification_rejects_dirty_or_symbolic_input(tmp_path: Path) -> None:
    root, commit = _minimal_public_repo(tmp_path)

    state = release._verify_source(root, commit, require_tag=False)

    assert state.commit == commit
    assert state.manifest_entries == 2
    with pytest.raises(release.ReleaseBundleError, match='full 40-character'):
        release._verify_source(root, 'HEAD', require_tag=False)
    (root / 'untracked.txt').write_text('dirty\n', encoding='utf-8')
    with pytest.raises(release.ReleaseBundleError, match='dirty'):
        release._verify_source(root, commit, require_tag=False)


def test_source_archive_exactly_matches_manifest_and_round_trips_gzip(tmp_path: Path) -> None:
    root, commit = _minimal_public_repo(tmp_path)
    state = release._verify_source(root, commit, require_tag=False)
    destination = tmp_path / release.SOURCE_ARCHIVE
    temporary = tmp_path / 'temporary'
    temporary.mkdir()

    raw = release._create_source_archive(root, state, destination, temporary)

    assert gzip.decompress(destination.read_bytes()) == raw
    members = release._inspect_tar(destination)
    release._require_archive_root(
        members,
        root=f'vaxreplay-{release.RELEASE}',
        archive_name=destination.name,
    )
    assert {item['path'] for item in members if item['type'] == 'file'} == {
        f'vaxreplay-{release.RELEASE}/BUILD-INFO.json',
        f'vaxreplay-{release.RELEASE}/MANIFEST.sha256',
        f'vaxreplay-{release.RELEASE}/README.md',
    }

    corrupt_state = replace(
        state,
        manifest_digests={**state.manifest_digests, 'README.md': '0' * 64},
    )
    second_temporary = tmp_path / 'temporary-corrupt'
    second_temporary.mkdir()
    with pytest.raises(release.ReleaseBundleError, match='digest_mismatch'):
        release._create_source_archive(
            root,
            corrupt_state,
            tmp_path / 'corrupt.tar.gz',
            second_temporary,
        )


def test_canonical_sdists_and_realistic_package_contract_happy_path(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'LICENSE').write_bytes(b'Apache License 2.0\n')
    (source / 'NOTICE').write_bytes(b'VaxReplay notice\n')
    wheel = tmp_path / f'vaxreplay-{release.PROJECT_VERSION}-py3-none-any.whl'
    first = tmp_path / f'vaxreplay-{release.PROJECT_VERSION}.tar.gz'
    second = tmp_path / 'second.tar.gz'
    _release_wheel(wheel, source)
    _release_sdist(first, source, reverse=False, mtime=100)
    _release_sdist(second, source, reverse=True, mtime=999)

    release._canonicalize_sdist(first, source_date_epoch=1234)
    release._canonicalize_sdist(second, source_date_epoch=1234)

    assert first.read_bytes() == second.read_bytes()
    release._validate_package_boundaries(wheel=wheel, sdist=first, public_source=source)
    members = release._inspect_tar(first)
    release._require_archive_root(
        members,
        root=f'vaxreplay-{release.PROJECT_VERSION}',
        archive_name=first.name,
    )
    with tarfile.open(first, 'r:gz') as archive:
        assert all((member.uid, member.gid, member.mtime) == (0, 0, 1234) for member in archive)
        assert all((member.uname, member.gname) == ('', '') for member in archive)


def test_package_contract_rejects_sdist_test_leakage(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'LICENSE').write_bytes(b'Apache License 2.0\n')
    (source / 'NOTICE').write_bytes(b'VaxReplay notice\n')
    wheel = tmp_path / f'vaxreplay-{release.PROJECT_VERSION}-py3-none-any.whl'
    sdist = tmp_path / f'vaxreplay-{release.PROJECT_VERSION}.tar.gz'
    _release_wheel(wheel, source)
    _release_sdist(
        sdist,
        source,
        reverse=False,
        mtime=100,
        extra_file=('tests/test_private.py', b'assert True\n'),
    )
    release._canonicalize_sdist(sdist, source_date_epoch=1234)

    with pytest.raises(release.ReleaseBundleError, match='package boundary'):
        release._validate_package_boundaries(wheel=wheel, sdist=sdist, public_source=source)


def test_reviewed_dependency_policy_exactly_matches_runtime_lock() -> None:
    root = Path(__file__).parents[1]
    state = release.SourceState(
        commit='a' * 40,
        tree='b' * 40,
        source_date_epoch=0,
        created='1970-01-01T00:00:00Z',
        tag_present=False,
        build_info={},
        manifest_sha256='c' * 64,
        manifest_entries=1,
        manifest_digests={},
    )

    inventory, dependencies = release._load_runtime_inventory(
        root / 'uv.lock',
        root / 'release' / 'dependency-license-policy.json',
        state=state,
    )

    names = {item['name'] for item in inventory['dependencies']}
    assert names == {
        'annotated-types',
        'cffi',
        'cryptography',
        'pycparser',
        'pydantic',
        'pydantic-core',
        'typing-extensions',
        'typing-inspection',
    }
    assert dependencies['cffi']['license_expression'] == 'MIT-0'
    assert dependencies['vaxreplay']['depends_on'] == ['cryptography', 'pydantic']


def test_spdx_is_structurally_valid_and_bound_to_release_wheel(tmp_path: Path) -> None:
    state = release.SourceState(
        commit='a' * 40,
        tree='b' * 40,
        source_date_epoch=1234,
        created='1970-01-01T00:20:34Z',
        tag_present=False,
        build_info={},
        manifest_sha256='c' * 64,
        manifest_entries=1,
        manifest_digests={},
    )
    wheel = tmp_path / f'vaxreplay-{release.PROJECT_VERSION}-py3-none-any.whl'
    wheel.write_bytes(b'release wheel bytes')
    dependencies = {
        'vaxreplay': {
            'name': 'vaxreplay',
            'version': release.PROJECT_VERSION,
            'depends_on': ['pydantic'],
            'license_expression': 'Apache-2.0',
            'purl': f'pkg:pypi/vaxreplay@{release.PROJECT_VERSION}',
            'graph': {'vaxreplay': ['pydantic'], 'pydantic': []},
        },
        'pydantic': {
            'name': 'pydantic',
            'version': '2.13.4',
            'depends_on': [],
            'license_expression': 'MIT',
            'purl': 'pkg:pypi/pydantic@2.13.4',
        },
    }

    sbom = release._build_sbom(dependencies, state=state, wheel=wheel)

    release._validate_spdx(sbom, wheel=wheel, state=state)
    root = next(package for package in sbom['packages'] if package['name'] == 'vaxreplay')
    assert root['downloadLocation'].endswith(f'/download/{release.RELEASE}/{wheel.name}')
    assert root['checksums'] == [
        {'algorithm': 'SHA256', 'checksumValue': hashlib.sha256(wheel.read_bytes()).hexdigest()}
    ]
    assert sbom['creationInfo']['comment'].startswith('The creation timestamp is the public commit')

    broken = json.loads(json.dumps(sbom))
    broken['relationships'][1]['relatedSpdxElement'] = 'SPDXRef-Package-absent'
    with pytest.raises(release.ReleaseBundleError, match='unknown endpoint'):
        release._validate_spdx(broken, wheel=wheel, state=state)


def test_builder_requires_exact_python_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert release._verify_python() == '3.12.13'
    monkeypatch.setattr(release.sys, 'version_info', (3, 12, 12))
    with pytest.raises(release.ReleaseBundleError, match='Python 3.12.13 is required'):
        release._verify_python()


def test_checksums_are_sorted_and_generated_metadata_rejects_local_paths(tmp_path: Path) -> None:
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'z.txt').write_text('z\n', encoding='utf-8')
    (bundle / 'a.txt').write_text('a\n', encoding='utf-8')

    release._write_checksums(bundle)

    lines = (bundle / release.CHECKSUMS_NAME).read_text(encoding='utf-8').splitlines()
    assert [line.split('  ', 1)[1] for line in lines] == ['a.txt', 'z.txt']
    assert all(re.fullmatch(r'[0-9a-f]{64}  [a-z]\.txt', line) for line in lines)

    generated = bundle / 'generated.md'
    generated.write_text(f'local: {tmp_path}/secret\n', encoding='utf-8')
    with pytest.raises(release.ReleaseBundleError, match='local absolute path'):
        release._audit_generated_text((generated,), forbidden_roots=(tmp_path,))


def test_release_workflow_and_verification_contract_are_pinned() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8')
    verify = (root / 'release' / f'alpha-{release.RELEASE}' / 'VERIFY.md').read_text(encoding='utf-8')
    notes = (root / 'release' / f'alpha-{release.RELEASE}' / 'RELEASE-NOTES.md').read_text(encoding='utf-8')

    for pin in (
        'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1',
        'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97',
        'astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9',
        'actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6',
        'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a',
    ):
        assert pin in workflow
    assert 'version: 0.11.7' in workflow
    assert 'test "$RELEASE_COMMIT" = "$GITHUB_SHA"' in workflow
    assert 'subject-checksums:' in workflow
    assert 'sbom-path:' in workflow
    assert '${{ steps.provenance.outputs.bundle-path }}' in workflow
    assert '${{ steps.sbom.outputs.bundle-path }}' in workflow
    assert 'gh release' not in workflow and 'git tag' not in workflow

    assert 'gh attestation verify SHA256SUMS' not in verify
    assert 'vaxreplay-v0.1.0-alpha.1-source.tar.gz' in verify
    assert '--predicate-type https://spdx.dev/Document/v2.3' in verify
    assert '--source-digest %VAXREPLAY_PUBLIC_COMMIT%' in verify
    assert '`workflow_dispatch` run is a pre-tag candidate only' in verify
    assert '`STATUS.md`' not in notes
    assert '`docs/alpha_scope.md`' in notes
