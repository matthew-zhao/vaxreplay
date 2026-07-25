#!/usr/bin/env python3
"""Build the VaxReplay v0.1.0-alpha.1 candidate with reproducible core artifacts.

The script intentionally uses only the Python standard library. It shells out to
Git, gzip, and the exactly pinned uv executable for source and package builds.
It creates checksummed payloads, but never creates or imitates a signature.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

RELEASE = 'v0.1.0-alpha.1'
PROJECT_VERSION = '0.1.0a1'
PUBLIC_REPOSITORY = 'matthew-zhao/vaxreplay'
UV_VERSION = '0.11.7'
PYTHON_VERSION = (3, 12, 13)
SOURCE_ARCHIVE = f'vaxreplay-{RELEASE}-source.tar.gz'
SBOM_NAME = f'vaxreplay-{RELEASE}.spdx.json'
DEPENDENCIES_NAME = f'vaxreplay-{RELEASE}-dependencies.json'
DEPENDENCY_LICENSES_NAME = f'vaxreplay-{RELEASE}-dependency-licenses.md'
ARCHIVE_INVENTORY_NAME = f'vaxreplay-{RELEASE}-archive-inventory.json'
BUILD_RECEIPT_NAME = f'vaxreplay-{RELEASE}-build-receipt.json'
RELEASE_BINDING_NAME = f'vaxreplay-{RELEASE}-release-binding.json'
EXPORT_BUILD_INFO_NAME = f'vaxreplay-{RELEASE}-export-build-info.json'
PUBLIC_MANIFEST_NAME = f'vaxreplay-{RELEASE}-public-tree-manifest.sha256'
RELEASE_REVIEW_NAME = f'vaxreplay-{RELEASE}-release-review.md'
RELEASE_NOTES_NAME = f'vaxreplay-{RELEASE}-RELEASE-NOTES.md'
VERIFY_NAME = f'vaxreplay-{RELEASE}-VERIFY.md'
CHECKSUMS_NAME = 'SHA256SUMS'

_FULL_COMMIT = re.compile(r'[0-9a-f]{40}')
_UNRESOLVED_TOKEN = re.compile(r'%VAXREPLAY_[A-Z0-9_]+%')
_SECRET_PATTERNS = (
    re.compile(r'-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----'),
    re.compile(r'(?<![A-Za-z0-9])sk-(?!test-)(?:proj-)?[A-Za-z0-9_-]{20,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
)
_LOCAL_PATH = re.compile(r'(?<![A-Za-z0-9])/(?:Users|home|private/tmp|tmp)/[^\s"\'`]+')
_EXPECTED_RUNTIME_REQUIREMENTS = ('cryptography>=43', 'pydantic>=2.10,<3')
_EXPECTED_ENTRY_POINTS = {
    'vaxreplay': 'vaxreplay.cli:main',
    'vaxreplay-feasibility': 'vaxreplay.feasibility.cli:main',
    'vaxreplay-iedb': 'vaxreplay.iedb.cli:main',
    'vaxreplay-ops': 'vaxreplay.operations.cli:main',
    'vaxreplay-prospective': 'vaxreplay.prospective_cli:main',
    'vaxreplay-release-readiness': 'vaxreplay.operations.release_readiness_cli:main',
    'vaxreplay-runner': 'vaxreplay.runner.cli:main',
}


class ReleaseBundleError(RuntimeError):
    """Raised when a release candidate cannot be built safely."""


@dataclass(frozen=True)
class SourceState:
    commit: str
    tree: str
    source_date_epoch: int
    created: str
    tag_present: bool
    build_info: Mapping[str, Any]
    manifest_sha256: str
    manifest_entries: int
    manifest_digests: Mapping[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            tuple(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=stdin,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as error:
        raise ReleaseBundleError(f'required executable was not found: {command[0]}') from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode('utf-8', errors='replace').strip()
        raise ReleaseBundleError(f'command failed ({command[0]}): {detail}') from error


def _command_text(command: Sequence[str], *, cwd: Path) -> str:
    return _run(command, cwd=cwd).stdout.decode('utf-8').strip()


def _git_text(source: Path, *args: str) -> str:
    return _command_text(('git', *args), cwd=source)


def _safe_member_path(raw: str, *, directory: bool = False) -> str:
    if '\\' in raw or any(unicodedata.category(character) == 'Cc' for character in raw):
        raise ReleaseBundleError(f'unsafe archive member path: {raw!r}')
    candidate = raw[:-1] if directory and raw.endswith('/') else raw
    if not candidate or candidate.startswith('/') or candidate.endswith('/'):
        raise ReleaseBundleError(f'unsafe archive member path: {raw!r}')
    raw_parts = candidate.split('/')
    if any(not part or part in {'.', '..'} or ':' in part for part in raw_parts):
        raise ReleaseBundleError(f'unsafe archive member path: {raw!r}')
    path = PurePosixPath(candidate)
    if path.is_absolute() or path.parts != tuple(raw_parts):
        raise ReleaseBundleError(f'unsafe archive member path: {raw!r}')
    return path.as_posix()


def _verify_public_manifest(source: Path, commit: str) -> tuple[str, int, dict[str, str]]:
    manifest_path = source / 'MANIFEST.sha256'
    try:
        raw = manifest_path.read_text(encoding='utf-8')
    except OSError as error:
        raise ReleaseBundleError(f'cannot read MANIFEST.sha256: {error}') from error
    if not raw.endswith('\n'):
        raise ReleaseBundleError('MANIFEST.sha256 must end with a newline')

    listed: list[str] = []
    digests: dict[str, str] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if match is None:
            raise ReleaseBundleError(f'invalid MANIFEST.sha256 line {line_number}')
        expected, member = match.groups()
        normalized = _safe_member_path(member)
        if normalized != member:
            raise ReleaseBundleError(f'non-canonical manifest path: {member!r}')
        if member in listed:
            raise ReleaseBundleError(f'duplicate manifest path: {member}')
        path = source / member
        if not path.is_file() or path.is_symlink():
            raise ReleaseBundleError(f'manifest path is not a regular file: {member}')
        if _sha256_file(path) != expected:
            raise ReleaseBundleError(f'manifest digest mismatch: {member}')
        listed.append(member)
        digests[member] = expected

    if listed != sorted(listed):
        raise ReleaseBundleError('MANIFEST.sha256 paths are not sorted')
    tracked = _git_text(source, 'ls-tree', '-r', '--name-only', commit).splitlines()
    expected_tracked = sorted((*listed, 'MANIFEST.sha256'))
    if sorted(tracked) != expected_tracked:
        missing = sorted(set(tracked) - set(expected_tracked))
        extra = sorted(set(expected_tracked) - set(tracked))
        raise ReleaseBundleError(
            f'public manifest does not exactly cover the commit (unlisted={missing}, absent={extra})'
        )
    if 'BUILD-INFO.json' not in listed:
        raise ReleaseBundleError('MANIFEST.sha256 must include BUILD-INFO.json')
    return _sha256_bytes(raw.encode('utf-8')), len(listed), digests


def _verify_source(source: Path, commit: str, *, require_tag: bool) -> SourceState:
    if _FULL_COMMIT.fullmatch(commit) is None:
        raise ReleaseBundleError('--commit must be the full 40-character lowercase public commit')
    resolved = _git_text(source, 'rev-parse', '--verify', f'{commit}^{{commit}}')
    if resolved != commit:
        raise ReleaseBundleError(f'commit resolved to a different object: {resolved}')
    head = _git_text(source, 'rev-parse', 'HEAD')
    if head != commit:
        raise ReleaseBundleError(f'checkout HEAD {head} does not equal requested commit {commit}')
    if _git_text(source, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise ReleaseBundleError('release input checkout is dirty')

    try:
        build_info = json.loads((source / 'BUILD-INFO.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseBundleError(f'cannot load BUILD-INFO.json: {error}') from error
    if (
        build_info.get('schema_version') != 2
        or build_info.get('release_name') != RELEASE
        or build_info.get('draft') is not False
        or build_info.get('source_dirty') is not False
    ):
        raise ReleaseBundleError('BUILD-INFO.json is not a clean final alpha export')
    source_revision = build_info.get('source_revision')
    if not isinstance(source_revision, str) or _FULL_COMMIT.fullmatch(source_revision) is None:
        raise ReleaseBundleError('BUILD-INFO.json source_revision must be a full lowercase commit')

    manifest_sha256, manifest_entries, manifest_digests = _verify_public_manifest(source, commit)
    static_paths = sorted(set(manifest_digests) - {'BUILD-INFO.json'})
    static_paths_sha256 = _sha256_bytes(''.join(f'{path}\n' for path in static_paths).encode('utf-8'))
    static_count = build_info.get('static_export_path_count')
    if (
        not isinstance(static_count, int)
        or isinstance(static_count, bool)
        or static_count <= 0
        or static_count != len(static_paths)
        or build_info.get('file_count_before_generated_metadata') != static_count
        or manifest_entries != static_count + 1
        or build_info.get('static_export_paths_sha256') != static_paths_sha256
    ):
        raise ReleaseBundleError('BUILD-INFO.json static export inventory does not match MANIFEST.sha256')
    for field in (
        'private_export_policy_canonical_sha256',
        'static_export_paths_sha256',
    ):
        value = build_info.get(field)
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
            raise ReleaseBundleError(f'BUILD-INFO.json {field} must be a lowercase SHA-256')
    tag_ref = f'refs/tags/{RELEASE}'
    tag_present = False
    try:
        tag_present = _git_text(source, 'rev-parse', '--verify', f'{tag_ref}^{{commit}}') == commit
    except ReleaseBundleError:
        tag_present = False
    if require_tag and not tag_present:
        raise ReleaseBundleError(f'{tag_ref} does not resolve to requested commit {commit}')

    epoch_text = _git_text(source, 'show', '-s', '--format=%ct', commit)
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise ReleaseBundleError(f'invalid Git commit timestamp: {epoch_text!r}') from error
    created = datetime.fromtimestamp(epoch, tz=UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    tree = _git_text(source, 'rev-parse', f'{commit}^{{tree}}')
    return SourceState(
        commit=commit,
        tree=tree,
        source_date_epoch=epoch,
        created=created,
        tag_present=tag_present,
        build_info=build_info,
        manifest_sha256=manifest_sha256,
        manifest_entries=manifest_entries,
        manifest_digests=manifest_digests,
    )


def _inspect_tar(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    try:
        archive = tarfile.open(path, mode='r:*')
    except (OSError, tarfile.TarError) as error:
        raise ReleaseBundleError(f'cannot read tar archive {path.name}: {error}') from error
    with archive:
        for member in archive.getmembers():
            if not (member.isdir() or member.isreg()):
                raise ReleaseBundleError(f'unsupported tar member type in {path.name}: {member.name}')
            normalized = _safe_member_path(member.name, directory=member.isdir())
            if normalized in seen:
                raise ReleaseBundleError(f'duplicate tar member in {path.name}: {normalized}')
            seen.add(normalized)
            entry: dict[str, object] = {
                'path': normalized,
                'type': 'directory' if member.isdir() else 'file',
                'size': member.size,
                'mode': f'{member.mode & 0o7777:04o}',
            }
            if member.isreg():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ReleaseBundleError(f'cannot read tar member in {path.name}: {normalized}')
                entry['sha256'] = _sha256_bytes(extracted.read())
            result.append(entry)
    return sorted(result, key=lambda item: str(item['path']))


def _inspect_zip(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseBundleError(f'cannot read zip archive {path.name}: {error}') from error
    with archive:
        for member in archive.infolist():
            normalized = _safe_member_path(member.filename, directory=member.is_dir())
            if normalized in seen:
                raise ReleaseBundleError(f'duplicate zip member in {path.name}: {normalized}')
            seen.add(normalized)
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            allowed_types = {0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG}
            if file_type not in allowed_types:
                raise ReleaseBundleError(f'unsupported zip member type in {path.name}: {normalized}')
            entry: dict[str, object] = {
                'path': normalized,
                'type': 'directory' if member.is_dir() else 'file',
                'size': member.file_size,
                'mode': f'{mode & 0o7777:04o}',
            }
            if not member.is_dir():
                entry['sha256'] = _sha256_bytes(archive.read(member))
            result.append(entry)
    return sorted(result, key=lambda item: str(item['path']))


def _require_archive_root(
    members: Sequence[Mapping[str, object]],
    *,
    root: str,
    archive_name: str,
) -> None:
    roots = [member for member in members if member['path'] == root]
    if len(roots) != 1 or roots[0]['type'] != 'directory':
        raise ReleaseBundleError(f'{archive_name} must contain exactly one root directory {root!r}')
    prefix = f'{root}/'
    if any(member['path'] != root and not str(member['path']).startswith(prefix) for member in members):
        raise ReleaseBundleError(f'{archive_name} has a member outside {root!r}')


def _verify_wheel_record(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        files = {member.filename: member for member in archive.infolist() if not member.is_dir()}
        record_paths = [name for name in files if name.endswith('.dist-info/RECORD')]
        if len(record_paths) != 1:
            raise ReleaseBundleError(f'wheel must contain exactly one RECORD: {path.name}')
        record_path = record_paths[0]
        try:
            rows = tuple(csv.reader(io.StringIO(archive.read(record_path).decode('utf-8'))))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ReleaseBundleError(f'invalid wheel RECORD in {path.name}: {error}') from error

        recorded: set[str] = set()
        for row in rows:
            if len(row) != 3:
                raise ReleaseBundleError(f'invalid wheel RECORD row in {path.name}')
            name, encoded_digest, size_text = row
            if _safe_member_path(name) != name or name in recorded:
                raise ReleaseBundleError(f'invalid or duplicate wheel RECORD path: {name!r}')
            if name not in files:
                raise ReleaseBundleError(f'wheel RECORD names an absent file: {name}')
            recorded.add(name)
            if name == record_path:
                if encoded_digest or size_text:
                    raise ReleaseBundleError('the wheel RECORD row for RECORD must omit hash and size')
                continue
            if not encoded_digest.startswith('sha256=') or not size_text.isdigit():
                raise ReleaseBundleError(f'invalid wheel RECORD hash or size: {name}')
            payload = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode('ascii')
            if encoded_digest != f'sha256={expected}' or int(size_text) != len(payload):
                raise ReleaseBundleError(f'wheel RECORD mismatch: {name}')
        if recorded != set(files):
            raise ReleaseBundleError(f'wheel RECORD does not cover every file: {sorted(set(files) - recorded)}')


def _headers(payload: bytes, *, context: str) -> Message:
    try:
        return BytesParser(policy=compat32).parsebytes(payload)
    except (UnicodeDecodeError, ValueError) as error:
        raise ReleaseBundleError(f'invalid package metadata in {context}: {error}') from error


def _metadata(payload: bytes, *, context: str) -> Message:
    message = _headers(payload, context=context)
    if (
        message.get('Name', '').lower() != 'vaxreplay'
        or message.get('Version') != PROJECT_VERSION
        or message.get('License-Expression') != 'Apache-2.0'
        or message.get('Requires-Python') != '>=3.12'
    ):
        raise ReleaseBundleError(f'unexpected name/version/license/Python metadata in {context}')
    requirements: set[tuple[str, tuple[str, ...]]] = set()
    for raw in message.get_all('Requires-Dist', []):
        requirement, separator, marker = raw.partition(';')
        if separator and re.search(r'\bextra\s*==', marker):
            continue
        match = re.fullmatch(r'\s*([A-Za-z0-9_.-]+)\s*(.*)', requirement)
        if match is None or (separator and marker.strip()):
            raise ReleaseBundleError(f'unsupported runtime requirement in {context}: {raw!r}')
        name, specifier = match.groups()
        normalized_name = re.sub(r'[-_.]+', '-', name).lower()
        clauses = tuple(sorted(part for part in specifier.replace(' ', '').split(',') if part))
        requirements.add((normalized_name, clauses))

    expected: set[tuple[str, tuple[str, ...]]] = set()
    for raw in _EXPECTED_RUNTIME_REQUIREMENTS:
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(.*)', raw)
        assert match is not None
        name, specifier = match.groups()
        expected.add(
            (
                re.sub(r'[-_.]+', '-', name).lower(),
                tuple(sorted(part for part in specifier.split(',') if part)),
            )
        )
    if requirements != expected:
        raise ReleaseBundleError(
            f'runtime dependencies in {context} do not match the alpha contract: {sorted(requirements)}'
        )
    return message


def _entry_points(payload: bytes, *, context: str) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string(payload.decode('utf-8'))
    except (UnicodeDecodeError, configparser.Error) as error:
        raise ReleaseBundleError(f'invalid entry_points.txt in {context}: {error}') from error
    if parser.sections() != ['console_scripts']:
        raise ReleaseBundleError(f'unexpected entry-point groups in {context}: {parser.sections()}')
    actual = {name: target.strip() for name, target in parser.items('console_scripts')}
    if actual != _EXPECTED_ENTRY_POINTS:
        raise ReleaseBundleError(f'console entry points in {context} do not match the alpha contract')


def _tar_payloads(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode='r:*') as archive:
        for member in archive.getmembers():
            if not member.isreg():
                continue
            normalized = _safe_member_path(member.name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ReleaseBundleError(f'cannot read tar member in {path.name}: {normalized}')
            payloads[normalized] = stream.read()
    return payloads


def _validate_package_boundaries(*, wheel: Path, sdist: Path, public_source: Path) -> None:
    """Validate package identity, metadata, licenses, entry points, and intended contents."""

    _verify_wheel_record(wheel)
    wheel_members = _inspect_zip(wheel)
    wheel_paths = {str(member['path']) for member in wheel_members if member['type'] == 'file'}
    if any('tests' in PurePosixPath(path).parts for path in wheel_paths):
        raise ReleaseBundleError('wheel must not contain tests')
    expected_dist_info = f'vaxreplay-{PROJECT_VERSION}.dist-info'
    dist_info_roots = {
        PurePosixPath(path).parts[0] for path in wheel_paths if PurePosixPath(path).parts[0].endswith('.dist-info')
    }
    if dist_info_roots != {expected_dist_info}:
        raise ReleaseBundleError(f'wheel has unexpected dist-info roots: {sorted(dist_info_roots)}')
    required_wheel_paths = {
        'vaxreplay/__init__.py',
        f'{expected_dist_info}/METADATA',
        f'{expected_dist_info}/WHEEL',
        f'{expected_dist_info}/entry_points.txt',
        f'{expected_dist_info}/RECORD',
        f'{expected_dist_info}/licenses/LICENSE',
        f'{expected_dist_info}/licenses/NOTICE',
    }
    missing_wheel = sorted(required_wheel_paths - wheel_paths)
    if missing_wheel:
        raise ReleaseBundleError(f'wheel is missing required package files: {missing_wheel}')
    with zipfile.ZipFile(wheel) as archive:
        _metadata(archive.read(f'{expected_dist_info}/METADATA'), context='wheel METADATA')
        wheel_metadata = _headers(
            archive.read(f'{expected_dist_info}/WHEEL'),
            context='wheel WHEEL',
        )
        # WHEEL is not core metadata, so validate its fields directly after a permissive parse.
        if wheel_metadata.get('Root-Is-Purelib', '').lower() != 'true' or wheel_metadata.get_all('Tag', []) != [
            'py3-none-any'
        ]:
            raise ReleaseBundleError('wheel is not the expected pure-Python py3-none-any artifact')
        _entry_points(
            archive.read(f'{expected_dist_info}/entry_points.txt'),
            context='wheel',
        )
        for name in ('LICENSE', 'NOTICE'):
            if archive.read(f'{expected_dist_info}/licenses/{name}') != (public_source / name).read_bytes():
                raise ReleaseBundleError(f'wheel {name} is not an exact copy of the public source file')

    sdist_members = _inspect_tar(sdist)
    root = f'vaxreplay-{PROJECT_VERSION}'
    _require_archive_root(sdist_members, root=root, archive_name=sdist.name)
    sdist_payloads = _tar_payloads(sdist)
    relative_files = {path[len(root) + 1 :] for path in sdist_payloads}
    forbidden_roots = {
        '.github',
        'benchmarks',
        'deploy',
        'docs',
        'examples',
        'private',
        'release',
        'scripts',
        'tests',
    }
    leaked = sorted(
        path for path in relative_files if PurePosixPath(path).parts and PurePosixPath(path).parts[0] in forbidden_roots
    )
    if leaked:
        raise ReleaseBundleError(f'sdist crosses the intended package boundary: {leaked}')
    required_sdist = {
        'LICENSE',
        'NOTICE',
        'PKG-INFO',
        'pyproject.toml',
        'src/vaxreplay/__init__.py',
        'src/vaxreplay.egg-info/entry_points.txt',
    }
    missing_sdist = sorted(required_sdist - relative_files)
    if missing_sdist:
        raise ReleaseBundleError(f'sdist is missing required package files: {missing_sdist}')
    for name in ('LICENSE', 'NOTICE'):
        if sdist_payloads[f'{root}/{name}'] != (public_source / name).read_bytes():
            raise ReleaseBundleError(f'sdist {name} is not an exact copy of the public source file')
    _metadata(sdist_payloads[f'{root}/PKG-INFO'], context='sdist PKG-INFO')
    _entry_points(
        sdist_payloads[f'{root}/src/vaxreplay.egg-info/entry_points.txt'],
        context='sdist',
    )
    try:
        pyproject = tomllib.loads(sdist_payloads[f'{root}/pyproject.toml'].decode('utf-8'))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseBundleError(f'invalid sdist pyproject.toml: {error}') from error
    project = pyproject.get('project')
    if not isinstance(project, dict):
        raise ReleaseBundleError('sdist pyproject.toml has no project table')
    if (
        project.get('name') != 'vaxreplay'
        or project.get('version') != PROJECT_VERSION
        or tuple(project.get('dependencies', ())) != _EXPECTED_RUNTIME_REQUIREMENTS
        or project.get('scripts') != _EXPECTED_ENTRY_POINTS
    ):
        raise ReleaseBundleError('sdist pyproject.toml does not match the alpha package contract')


def _create_source_archive(source: Path, state: SourceState, destination: Path, temporary: Path) -> bytes:
    root = f'vaxreplay-{RELEASE}'
    prefix = f'{root}/'
    raw_tar = temporary / 'public-source.tar'
    with raw_tar.open('wb') as output:
        _run(
            ('git', 'archive', '--format=tar', f'--prefix={prefix}', state.commit),
            cwd=source,
            stdout=output,
        )
    raw_members = _inspect_tar(raw_tar)
    _require_archive_root(raw_members, root=root, archive_name=raw_tar.name)
    with raw_tar.open('rb') as source_stream, destination.open('wb') as output:
        _run(('gzip', '-n', '-9', '-c'), cwd=source, stdin=source_stream, stdout=output)
    members = _inspect_tar(destination)
    _require_archive_root(members, root=root, archive_name=destination.name)
    raw_payload = raw_tar.read_bytes()
    try:
        decompressed = gzip.decompress(destination.read_bytes())
    except (OSError, EOFError) as error:
        raise ReleaseBundleError(f'cannot decompress deterministic source archive: {error}') from error
    if decompressed != raw_payload:
        raise ReleaseBundleError('compressed source archive does not decode to the inspected Git archive')

    expected_digests = dict(state.manifest_digests)
    expected_digests['MANIFEST.sha256'] = state.manifest_sha256
    archived_files = {
        str(member['path'])[len(prefix) :]: str(member['sha256']) for member in members if member['type'] == 'file'
    }
    if archived_files != expected_digests:
        missing = sorted(set(expected_digests) - set(archived_files))
        extra = sorted(set(archived_files) - set(expected_digests))
        mismatched = sorted(
            name
            for name in set(expected_digests) & set(archived_files)
            if expected_digests[name] != archived_files[name]
        )
        raise ReleaseBundleError(
            'source archive does not exactly match the public manifest '
            f'(missing={missing}, extra={extra}, digest_mismatch={mismatched})'
        )
    return raw_payload


def _extract_source(raw_tar: bytes, destination: Path) -> Path:
    destination.mkdir()
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode='r:') as archive:
            for member in archive.getmembers():
                if not (member.isdir() or member.isreg()):
                    raise ReleaseBundleError(f'unsafe source member type: {member.name}')
                _safe_member_path(member.name, directory=member.isdir())
            archive.extractall(destination, filter='data')
    except (OSError, tarfile.TarError) as error:
        raise ReleaseBundleError(f'cannot extract public source archive: {error}') from error
    extracted = destination / f'vaxreplay-{RELEASE}'
    if not extracted.is_dir():
        raise ReleaseBundleError('source archive extraction did not create expected root')
    return extracted


def _verify_uv(uv: str, source: Path) -> str:
    version = _command_text((uv, '--version'), cwd=source).splitlines()[0]
    match = re.fullmatch(r'uv ([0-9]+\.[0-9]+\.[0-9]+)(?: .*)?', version)
    if match is None or match.group(1) != UV_VERSION:
        raise ReleaseBundleError(f'uv {UV_VERSION} is required, got {version!r}')
    return version


def _verify_python() -> str:
    actual = sys.version_info[:3]
    if actual != PYTHON_VERSION:
        expected_text = '.'.join(str(part) for part in PYTHON_VERSION)
        actual_text = '.'.join(str(part) for part in actual)
        raise ReleaseBundleError(f'Python {expected_text} is required, got {actual_text}')
    return '.'.join(str(part) for part in actual)


def _one_fresh_package_build(
    raw_source_tar: bytes,
    *,
    uv: str,
    environment: Mapping[str, str],
    root: Path,
    build_number: int,
) -> tuple[Path, Path]:
    extraction = root / f'build-{build_number}'
    source = _extract_source(raw_source_tar, extraction)
    output = root / f'artifacts-{build_number}'
    output.mkdir()
    _run((uv, 'build', '--out-dir', str(output)), cwd=source, env=environment)
    wheels = sorted(output.glob('*.whl'))
    sdists = sorted(output.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseBundleError(f'package build {build_number} produced {len(wheels)} wheels and {len(sdists)} sdists')
    return wheels[0], sdists[0]


def _canonicalize_sdist(path: Path, *, source_date_epoch: int) -> None:
    """Repack one validated sdist with deterministic tar and gzip metadata."""

    inspected = _inspect_tar(path)
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode='r:gz') as archive:
        by_name = {_safe_member_path(member.name, directory=member.isdir()): member for member in archive.getmembers()}
        for item in inspected:
            if item['type'] != 'file':
                continue
            member = by_name[str(item['path'])]
            stream = archive.extractfile(member)
            if stream is None:
                raise ReleaseBundleError(f'cannot read sdist member during canonicalization: {member.name}')
            payloads[str(item['path'])] = stream.read()

    canonical = path.with_name(f'.{path.name}.canonical')
    try:
        with canonical.open('wb') as compressed:
            with gzip.GzipFile(
                filename='',
                mode='wb',
                compresslevel=9,
                fileobj=compressed,
                mtime=0,
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode='w|',
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for item in inspected:
                        name = str(item['path'])
                        member = tarfile.TarInfo(name)
                        member.uid = 0
                        member.gid = 0
                        member.uname = ''
                        member.gname = ''
                        member.mtime = source_date_epoch
                        member.pax_headers = {}
                        if item['type'] == 'directory':
                            member.type = tarfile.DIRTYPE
                            member.mode = 0o755
                            member.size = 0
                            archive.addfile(member)
                        else:
                            payload = payloads[name]
                            member.type = tarfile.REGTYPE
                            member.mode = 0o755 if int(str(item['mode']), 8) & 0o111 else 0o644
                            member.size = len(payload)
                            archive.addfile(member, io.BytesIO(payload))
        canonical.replace(path)
    finally:
        canonical.unlink(missing_ok=True)
    _inspect_tar(path)


def _build_packages(
    raw_source_tar: bytes,
    *,
    source: Path,
    state: SourceState,
    uv: str,
    temporary: Path,
    bundle: Path,
) -> tuple[Path, Path, str]:
    uv_version = _verify_uv(uv, source)
    environment = dict(os.environ)
    environment.update(
        {
            'SOURCE_DATE_EPOCH': str(state.source_date_epoch),
            'PYTHONHASHSEED': '0',
            'TZ': 'UTC',
            'LC_ALL': 'C.UTF-8',
        }
    )
    first_wheel, first_sdist = _one_fresh_package_build(
        raw_source_tar,
        uv=uv,
        environment=environment,
        root=temporary,
        build_number=1,
    )
    second_wheel, second_sdist = _one_fresh_package_build(
        raw_source_tar,
        uv=uv,
        environment=environment,
        root=temporary,
        build_number=2,
    )
    _canonicalize_sdist(first_sdist, source_date_epoch=state.source_date_epoch)
    _canonicalize_sdist(second_sdist, source_date_epoch=state.source_date_epoch)
    for first, second, kind in (
        (first_wheel, second_wheel, 'wheel'),
        (first_sdist, second_sdist, 'sdist'),
    ):
        if first.name != second.name or first.read_bytes() != second.read_bytes():
            raise ReleaseBundleError(f'the two fresh {kind} builds are not byte-for-byte reproducible')
    expected_wheel = f'vaxreplay-{PROJECT_VERSION}-py3-none-any.whl'
    expected_sdist = f'vaxreplay-{PROJECT_VERSION}.tar.gz'
    if first_wheel.name != expected_wheel or first_sdist.name != expected_sdist:
        raise ReleaseBundleError(f'unexpected package names: {first_wheel.name!r}, {first_sdist.name!r}')
    wheel = bundle / first_wheel.name
    sdist = bundle / first_sdist.name
    shutil.copyfile(first_wheel, wheel)
    shutil.copyfile(first_sdist, sdist)
    _inspect_zip(wheel)
    _verify_wheel_record(wheel)
    sdist_members = _inspect_tar(sdist)
    _require_archive_root(
        sdist_members,
        root=f'vaxreplay-{PROJECT_VERSION}',
        archive_name=sdist.name,
    )
    _validate_package_boundaries(wheel=wheel, sdist=sdist, public_source=source)
    return wheel, sdist, uv_version


def _load_runtime_inventory(
    lock_path: Path,
    policy_path: Path,
    *,
    state: SourceState,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    try:
        lock = tomllib.loads(lock_path.read_text(encoding='utf-8'))
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise ReleaseBundleError(f'cannot load dependency inputs: {error}') from error
    if policy.get('schema_version') != 1 or policy.get('release_name') != RELEASE:
        raise ReleaseBundleError('dependency policy has the wrong schema or release')
    review = policy.get('review')
    if not isinstance(review, dict) or review.get('status') != 'reviewed-for-alpha-inventory':
        raise ReleaseBundleError('dependency policy is not marked reviewed for this alpha inventory')

    packages_raw = lock.get('package')
    if not isinstance(packages_raw, list):
        raise ReleaseBundleError('uv.lock has no package table')
    locked: dict[str, dict[str, object]] = {}
    for raw in packages_raw:
        if not isinstance(raw, dict) or not isinstance(raw.get('name'), str):
            raise ReleaseBundleError('uv.lock contains an invalid package record')
        name = raw['name']
        if name in locked:
            raise ReleaseBundleError(f'uv.lock contains ambiguous duplicate package name: {name}')
        locked[name] = raw
    root = locked.get('vaxreplay')
    if root is None or root.get('version') != PROJECT_VERSION:
        raise ReleaseBundleError('uv.lock does not contain the expected VaxReplay version')

    closure: set[str] = set()
    pending = ['vaxreplay']
    edges: dict[str, list[str]] = {}
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        package = locked.get(name)
        if package is None:
            raise ReleaseBundleError(f'uv.lock dependency is missing: {name}')
        closure.add(name)
        dependencies = package.get('dependencies', [])
        if not isinstance(dependencies, list):
            raise ReleaseBundleError(f'invalid dependency list for {name}')
        child_names: list[str] = []
        for item in dependencies:
            if not isinstance(item, dict) or not isinstance(item.get('name'), str):
                raise ReleaseBundleError(f'invalid dependency edge for {name}')
            child_names.append(item['name'])
            pending.append(item['name'])
        edges[name] = sorted(set(child_names))

    policy_packages_raw = policy.get('packages')
    if not isinstance(policy_packages_raw, list):
        raise ReleaseBundleError('dependency policy has no packages')
    policy_packages: dict[str, dict[str, object]] = {}
    for package in policy_packages_raw:
        if not isinstance(package, dict) or not isinstance(package.get('name'), str):
            raise ReleaseBundleError('dependency policy contains an invalid package')
        name = package['name']
        if name in policy_packages:
            raise ReleaseBundleError(f'duplicate dependency policy package: {name}')
        policy_packages[name] = package
    expected_policy_names = closure - {'vaxreplay'}
    if set(policy_packages) != expected_policy_names:
        raise ReleaseBundleError(
            'dependency policy does not exactly match runtime closure '
            f'(missing={sorted(expected_policy_names - set(policy_packages))}, '
            f'extra={sorted(set(policy_packages) - expected_policy_names)})'
        )

    direct = set(edges['vaxreplay'])
    records: list[dict[str, object]] = []
    for name in sorted(expected_policy_names):
        locked_package = locked[name]
        package_policy = policy_packages[name]
        version = locked_package.get('version')
        required = ('version', 'license_expression', 'homepage', 'purl', 'review_note')
        if any(not isinstance(package_policy.get(field), str) or not package_policy[field] for field in required):
            raise ReleaseBundleError(f'incomplete dependency policy package: {name}')
        if package_policy['version'] != version:
            raise ReleaseBundleError(
                f'dependency policy version for {name} is {package_policy["version"]}, lock has {version}'
            )
        records.append(
            {
                'name': name,
                'version': version,
                'direct': name in direct,
                'depends_on': edges[name],
                'license_expression': package_policy['license_expression'],
                'homepage': package_policy['homepage'],
                'purl': package_policy['purl'],
                'review_note': package_policy['review_note'],
            }
        )

    inventory: dict[str, object] = {
        'schema_version': 1,
        'release_name': RELEASE,
        'public_commit': state.commit,
        'source_timestamp': state.created,
        'timestamp_basis': 'public_commit_timestamp',
        'source': 'uv.lock',
        'scope': 'default runtime dependency closure; optional extras and build/dev tools excluded',
        'policy_review': review,
        'dependencies': records,
    }
    graph = {name: edges[name] for name in sorted(closure)}
    return inventory, {record['name']: record for record in records} | {
        'vaxreplay': {
            'name': 'vaxreplay',
            'version': PROJECT_VERSION,
            'depends_on': edges['vaxreplay'],
            'license_expression': policy['project']['license_expression'],
            'purl': policy['project']['purl'],
            'graph': graph,
        }
    }


def _dependency_markdown(inventory: Mapping[str, object]) -> str:
    review = inventory['policy_review']
    assert isinstance(review, dict)
    dependencies = inventory['dependencies']
    assert isinstance(dependencies, list)
    lines = [
        f'# VaxReplay {RELEASE} runtime dependency licenses',
        '',
        f'Public commit: `{inventory["public_commit"]}`',
        '',
        f'Policy status: `{review["status"]}` as of `{review["as_of"]}`.',
        '',
        'This version-specific inventory covers the default runtime closure in `uv.lock`. It excludes',
        'optional extras, build tools, and development tools. It is informational and not legal advice.',
        '',
        '| Package | Version | Direct | SPDX expression | Upstream | Review note |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for raw in dependencies:
        assert isinstance(raw, dict)
        values = {
            key: str(raw[key]).replace('|', r'\|')
            for key in ('name', 'version', 'license_expression', 'homepage', 'review_note')
        }
        lines.append(
            f'| `{values["name"]}` | `{values["version"]}` | '
            f'{"yes" if raw["direct"] else "no"} | `{values["license_expression"]}` | '
            f'[link]({values["homepage"]}) | {values["review_note"]} |'
        )
    lines.extend(
        [
            '',
            'The machine-readable companion records dependency edges and Package URLs.',
            '',
        ]
    )
    return '\n'.join(lines)


def _spdx_id(name: str) -> str:
    return 'SPDXRef-Package-' + re.sub(r'[^A-Za-z0-9.-]', '-', name)


def _build_sbom(
    dependency_records: Mapping[str, Mapping[str, object]],
    *,
    state: SourceState,
    wheel: Path,
) -> dict[str, object]:
    packages: list[dict[str, object]] = []
    for name in sorted(dependency_records):
        record = dependency_records[name]
        package: dict[str, object] = {
            'SPDXID': _spdx_id(name),
            'name': name,
            'versionInfo': record['version'],
            'downloadLocation': 'NOASSERTION',
            'filesAnalyzed': False,
            'licenseConcluded': record['license_expression'],
            'licenseDeclared': record['license_expression'],
            'copyrightText': 'NOASSERTION',
            'externalRefs': [
                {
                    'referenceCategory': 'PACKAGE-MANAGER',
                    'referenceType': 'purl',
                    'referenceLocator': record['purl'],
                }
            ],
        }
        if name == 'vaxreplay':
            package['downloadLocation'] = (
                f'https://github.com/{PUBLIC_REPOSITORY}/releases/download/{RELEASE}/{wheel.name}'
            )
            package['checksums'] = [{'algorithm': 'SHA256', 'checksumValue': _sha256_file(wheel)}]
        packages.append(package)

    root_record = dependency_records['vaxreplay']
    graph = root_record['graph']
    assert isinstance(graph, dict)
    relationships: list[dict[str, str]] = [
        {
            'spdxElementId': 'SPDXRef-DOCUMENT',
            'relationshipType': 'DESCRIBES',
            'relatedSpdxElement': _spdx_id('vaxreplay'),
        }
    ]
    for parent in sorted(graph):
        children = graph[parent]
        assert isinstance(children, list)
        for child in children:
            relationships.append(
                {
                    'spdxElementId': _spdx_id(parent),
                    'relationshipType': 'DEPENDS_ON',
                    'relatedSpdxElement': _spdx_id(child),
                }
            )
    return {
        'spdxVersion': 'SPDX-2.3',
        'dataLicense': 'CC0-1.0',
        'SPDXID': 'SPDXRef-DOCUMENT',
        'name': f'VaxReplay {RELEASE} runtime SBOM',
        'documentNamespace': (f'https://github.com/{PUBLIC_REPOSITORY}/releases/{RELEASE}/spdx/{state.commit}'),
        'creationInfo': {
            'created': state.created,
            'creators': ['Tool: scripts/build_alpha_release_bundle.py'],
            'licenseListVersion': '3.27',
            'comment': 'The creation timestamp is the public commit timestamp for deterministic metadata.',
        },
        'documentComment': 'Timestamp basis: public_commit_timestamp.',
        'documentDescribes': [_spdx_id('vaxreplay')],
        'packages': packages,
        'relationships': relationships,
    }


def _validate_spdx(sbom: Mapping[str, object], *, wheel: Path, state: SourceState) -> None:
    if (
        sbom.get('spdxVersion') != 'SPDX-2.3'
        or sbom.get('dataLicense') != 'CC0-1.0'
        or sbom.get('SPDXID') != 'SPDXRef-DOCUMENT'
    ):
        raise ReleaseBundleError('SBOM is not a valid SPDX 2.3 document header')
    creation = sbom.get('creationInfo')
    if not isinstance(creation, dict) or creation.get('created') != state.created:
        raise ReleaseBundleError('SBOM creation timestamp is not bound to the public commit timestamp')
    namespace = sbom.get('documentNamespace')
    if not isinstance(namespace, str) or state.commit not in namespace:
        raise ReleaseBundleError('SBOM namespace is not bound to the public commit')

    packages = sbom.get('packages')
    if not isinstance(packages, list) or not packages:
        raise ReleaseBundleError('SBOM contains no packages')
    package_ids: set[str] = set()
    by_name: dict[str, Mapping[str, object]] = {}
    for raw in packages:
        if not isinstance(raw, dict):
            raise ReleaseBundleError('SBOM package is not an object')
        package_id = raw.get('SPDXID')
        name = raw.get('name')
        if (
            not isinstance(package_id, str)
            or not package_id.startswith('SPDXRef-Package-')
            or package_id in package_ids
            or not isinstance(name, str)
            or name in by_name
            or raw.get('filesAnalyzed') is not False
            or not isinstance(raw.get('licenseConcluded'), str)
            or not isinstance(raw.get('licenseDeclared'), str)
            or raw.get('licenseConcluded') != raw.get('licenseDeclared')
        ):
            raise ReleaseBundleError('SBOM contains an invalid or duplicate package')
        external_refs = raw.get('externalRefs')
        if (
            not isinstance(external_refs, list)
            or len(external_refs) != 1
            or not isinstance(external_refs[0], dict)
            or external_refs[0].get('referenceType') != 'purl'
            or not str(external_refs[0].get('referenceLocator', '')).startswith('pkg:pypi/')
        ):
            raise ReleaseBundleError(f'SBOM package {name} lacks exactly one Package URL')
        package_ids.add(package_id)
        by_name[name] = raw

    root = by_name.get('vaxreplay')
    expected_download = f'https://github.com/{PUBLIC_REPOSITORY}/releases/download/{RELEASE}/{wheel.name}'
    if (
        root is None
        or root.get('versionInfo') != PROJECT_VERSION
        or root.get('downloadLocation') != expected_download
        or root.get('checksums') != [{'algorithm': 'SHA256', 'checksumValue': _sha256_file(wheel)}]
    ):
        raise ReleaseBundleError('SBOM root package is not bound to the release wheel')

    described = sbom.get('documentDescribes')
    if described != [_spdx_id('vaxreplay')]:
        raise ReleaseBundleError('SBOM documentDescribes does not identify VaxReplay')
    relationships = sbom.get('relationships')
    if not isinstance(relationships, list) or not relationships:
        raise ReleaseBundleError('SBOM contains no relationships')
    valid_ids = package_ids | {'SPDXRef-DOCUMENT'}
    describes = 0
    edges: set[tuple[str, str, str]] = set()
    for raw in relationships:
        if not isinstance(raw, dict):
            raise ReleaseBundleError('SBOM relationship is not an object')
        parent = raw.get('spdxElementId')
        kind = raw.get('relationshipType')
        child = raw.get('relatedSpdxElement')
        if parent not in valid_ids or child not in package_ids or kind not in {'DESCRIBES', 'DEPENDS_ON'}:
            raise ReleaseBundleError('SBOM relationship has an unknown endpoint or type')
        edge = (str(parent), str(kind), str(child))
        if edge in edges:
            raise ReleaseBundleError('SBOM contains a duplicate relationship')
        edges.add(edge)
        if kind == 'DESCRIBES':
            describes += 1
            if edge != ('SPDXRef-DOCUMENT', 'DESCRIBES', _spdx_id('vaxreplay')):
                raise ReleaseBundleError('SBOM contains an invalid DESCRIBES relationship')
    if describes != 1:
        raise ReleaseBundleError('SBOM must contain exactly one DESCRIBES relationship')
    depended_on = {child for _, kind, child in edges if kind == 'DEPENDS_ON'}
    if depended_on != package_ids - {_spdx_id('vaxreplay')}:
        raise ReleaseBundleError('SBOM dependency relationships do not cover the runtime closure')


def _artifact(path: Path, media_type: str) -> dict[str, object]:
    return {
        'filename': path.name,
        'sha256': _sha256_file(path),
        'bytes': path.stat().st_size,
        'media_type': media_type,
    }


def _archive_inventory(paths: Iterable[Path], *, state: SourceState) -> dict[str, object]:
    archives: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: item.name):
        if path.suffix == '.whl':
            archive_format = 'zip'
            members = _inspect_zip(path)
        else:
            archive_format = 'tar+gzip'
            members = _inspect_tar(path)
        archives.append(
            {
                **_artifact(
                    path,
                    'application/zip' if archive_format == 'zip' else 'application/gzip',
                ),
                'format': archive_format,
                'member_count': len(members),
                'members': members,
            }
        )
    return {
        'schema_version': 1,
        'release_name': RELEASE,
        'public_commit': state.commit,
        'source_timestamp': state.created,
        'timestamp_basis': 'public_commit_timestamp',
        'archives': archives,
    }


def _copy_exact(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    destination.write_bytes(payload)
    if destination.read_bytes() != payload:
        raise ReleaseBundleError(f'exact-copy verification failed: {destination.name}')


def _render_template(source: Path, destination: Path, replacements: Mapping[str, str]) -> None:
    text = source.read_text(encoding='utf-8')
    for token, value in replacements.items():
        text = text.replace(token, value)
    unresolved = _UNRESOLVED_TOKEN.search(text)
    if unresolved is not None:
        raise ReleaseBundleError(f'unresolved release template token: {unresolved.group(0)}')
    destination.write_text(text, encoding='utf-8')


def _media_type(path: Path) -> str:
    if path.suffix == '.whl':
        return 'application/zip'
    if path.name.endswith('.tar.gz'):
        return 'application/gzip'
    if path.suffix == '.json':
        return 'application/json'
    if path.suffix in {'.md', '.sha256'} or path.name == CHECKSUMS_NAME:
        return 'text/plain'
    return 'application/octet-stream'


def _payloads(bundle: Path) -> list[Path]:
    return sorted(
        (path for path in bundle.iterdir() if path.is_file() and path.name != CHECKSUMS_NAME),
        key=lambda item: item.name,
    )


def _write_checksums(bundle: Path) -> None:
    payloads = _payloads(bundle)
    text = ''.join(f'{_sha256_file(path)}  {path.name}\n' for path in payloads)
    (bundle / CHECKSUMS_NAME).write_text(text, encoding='utf-8')


def _audit_generated_text(paths: Iterable[Path], *, forbidden_roots: Iterable[Path]) -> None:
    forbidden = tuple(str(path.resolve()) for path in forbidden_roots)
    for path in paths:
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        if any(root in text for root in forbidden) or _LOCAL_PATH.search(text) or 'file://' in text:
            raise ReleaseBundleError(f'generated release metadata contains a local absolute path: {path.name}')
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise ReleaseBundleError(f'generated release metadata contains secret-like text: {path.name}')


def _tool_version(command: Sequence[str], *, source: Path) -> str:
    completed = _run(command, cwd=source)
    output = completed.stdout or completed.stderr
    lines = output.decode('utf-8', errors='replace').splitlines()
    if not lines:
        raise ReleaseBundleError(f'{command[0]} did not report a tool version')
    return lines[0].strip()


def build_bundle(
    *,
    source: Path,
    commit: str,
    output: Path,
    uv: str,
    require_tag: bool = False,
) -> dict[str, object]:
    """Build one fresh candidate directory and atomically publish it to ``output``."""

    python_version = _verify_python()
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ReleaseBundleError(f'output already exists: {output}')
    state = _verify_source(source, commit, require_tag=require_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.staging-', dir=output.parent))
    temporary_root = Path(tempfile.mkdtemp(prefix='vaxreplay-alpha-build-'))
    try:
        source_archive = staging / SOURCE_ARCHIVE
        raw_source_tar = _create_source_archive(source, state, source_archive, temporary_root)
        wheel, sdist, uv_version = _build_packages(
            raw_source_tar,
            source=source,
            state=state,
            uv=uv,
            temporary=temporary_root,
            bundle=staging,
        )

        export_build_info = staging / EXPORT_BUILD_INFO_NAME
        public_manifest = staging / PUBLIC_MANIFEST_NAME
        _copy_exact(source / 'BUILD-INFO.json', export_build_info)
        _copy_exact(source / 'MANIFEST.sha256', public_manifest)

        policy_path = source / 'release' / 'dependency-license-policy.json'
        inventory, dependency_records = _load_runtime_inventory(
            source / 'uv.lock',
            policy_path,
            state=state,
        )
        dependencies_path = staging / DEPENDENCIES_NAME
        dependency_licenses_path = staging / DEPENDENCY_LICENSES_NAME
        _write_json(dependencies_path, inventory)
        dependency_licenses_path.write_text(_dependency_markdown(inventory), encoding='utf-8')

        sbom_path = staging / SBOM_NAME
        sbom = _build_sbom(dependency_records, state=state, wheel=wheel)
        _validate_spdx(sbom, wheel=wheel, state=state)
        _write_json(sbom_path, sbom)
        archive_inventory_path = staging / ARCHIVE_INVENTORY_NAME
        _write_json(
            archive_inventory_path,
            _archive_inventory((source_archive, wheel, sdist), state=state),
        )

        template_root = source / 'release' / f'alpha-{RELEASE}'
        release_notes = staging / RELEASE_NOTES_NAME
        release_review = staging / RELEASE_REVIEW_NAME
        verify = staging / VERIFY_NAME
        _render_template(template_root / 'RELEASE-NOTES.md', release_notes, {})
        _render_template(template_root / 'RELEASE-REVIEW.md', release_review, {})
        _render_template(
            template_root / 'VERIFY.md',
            verify,
            {'%VAXREPLAY_PUBLIC_COMMIT%': state.commit},
        )

        artifacts_before_receipt = [_artifact(path, _media_type(path)) for path in _payloads(staging)]
        receipt = {
            'schema_version': 1,
            'release_name': RELEASE,
            'public_repository': PUBLIC_REPOSITORY,
            'public_commit': state.commit,
            'public_tree': state.tree,
            'tag': RELEASE,
            'tag_present_during_build': state.tag_present,
            'source_timestamp': state.created,
            'timestamp_basis': 'public_commit_timestamp',
            'source_date_epoch': state.source_date_epoch,
            'tools': {
                'python': python_version,
                'uv': uv_version,
                'git': _tool_version(('git', '--version'), source=source),
                'gzip': _tool_version(('gzip', '--version'), source=source),
            },
            'commands': [
                f'git archive --format=tar --prefix=vaxreplay-{RELEASE}/ <PUBLIC_COMMIT>',
                'gzip -n -9 -c',
                f'uv {UV_VERSION} build --out-dir <FRESH_OUTPUT> (two fresh source extractions)',
                'canonicalize sdist tar/gzip metadata, then byte-compare both wheels and both sdists',
            ],
            'environment': {
                'SOURCE_DATE_EPOCH': str(state.source_date_epoch),
                'PYTHONHASHSEED': '0',
                'TZ': 'UTC',
                'LC_ALL': 'C.UTF-8',
            },
            'export_publisher_assertions': {
                'build_info_schema_version': state.build_info.get('schema_version'),
                'private_source_revision': state.build_info.get('source_revision'),
                'private_export_policy_canonical_sha256': state.build_info.get(
                    'private_export_policy_canonical_sha256'
                ),
                'static_export_path_count': state.build_info.get('static_export_path_count'),
                'static_export_paths_sha256': state.build_info.get('static_export_paths_sha256'),
            },
            'claims': {
                'same_run_two_fresh_wheels_byte_identical': True,
                'same_run_two_fresh_canonical_sdists_byte_identical': True,
                'deterministic_git_archive_gzip_n': True,
                'wheel_record_validated': True,
                'safe_archive_members_validated': True,
                'whole_bundle_reproducibility_claimed': False,
                'hermetic': False,
                'network_isolation_enforced': False,
            },
            'artifacts_at_receipt_time': artifacts_before_receipt,
        }
        receipt_path = staging / BUILD_RECEIPT_NAME
        _write_json(receipt_path, receipt)

        bound_artifacts = [_artifact(path, _media_type(path)) for path in _payloads(staging)]
        release_binding = {
            'schema_version': 1,
            'release_name': RELEASE,
            'tag': RELEASE,
            'public_repository': PUBLIC_REPOSITORY,
            'public_commit': state.commit,
            'public_tree': state.tree,
            'source_timestamp': state.created,
            'timestamp_basis': 'public_commit_timestamp',
            'export': {
                'build_info_sha256': _sha256_file(export_build_info),
                'public_tree_manifest_sha256': state.manifest_sha256,
                'public_tree_manifest_entries': state.manifest_entries,
                'private_source_revision': state.build_info.get('source_revision'),
                'private_source_revision_claim': (
                    'publisher assertion copied exactly from the sanitized export BUILD-INFO.json'
                ),
                'private_export_policy_canonical_sha256': state.build_info.get(
                    'private_export_policy_canonical_sha256'
                ),
                'static_export_path_count': state.build_info.get('static_export_path_count'),
                'static_export_paths_sha256': state.build_info.get('static_export_paths_sha256'),
                'binding_fields_claim': (
                    'publisher assertions copied exactly from the sanitized export BUILD-INFO.json'
                ),
            },
            'artifacts': bound_artifacts,
            'attestation': {
                'created_by_builder': False,
                'expected_workflow': '.github/workflows/release.yml',
                'note': (
                    'A manual workflow may attest a pre-approval candidate; only a fresh bundle '
                    'attested from the exact release tag is eligible for final publication.'
                ),
            },
        }
        binding_path = staging / RELEASE_BINDING_NAME
        _write_json(binding_path, release_binding)
        _write_checksums(staging)

        generated_text = [
            dependencies_path,
            dependency_licenses_path,
            sbom_path,
            archive_inventory_path,
            release_notes,
            release_review,
            verify,
            receipt_path,
            binding_path,
            staging / CHECKSUMS_NAME,
        ]
        _audit_generated_text(
            generated_text,
            forbidden_roots=(source, output, staging, temporary_root, Path.home()),
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    return {
        'release_name': RELEASE,
        'public_commit': state.commit,
        'public_tree': state.tree,
        'output': output.name,
        'payload_count': len(tuple(output.iterdir())),
        'sha256sums_sha256': _sha256_file(output / CHECKSUMS_NAME),
        'signature_created': False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--commit', required=True, help='exact full public commit checked out at HEAD')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uv', default='uv', help=f'path to uv {UV_VERSION}')
    parser.add_argument(
        '--require-tag',
        action='store_true',
        help=f'require refs/tags/{RELEASE} to resolve to --commit',
    )
    args = parser.parse_args()
    try:
        result = build_bundle(
            source=args.source,
            commit=args.commit,
            output=args.output,
            uv=args.uv,
            require_tag=args.require_tag,
        )
    except (OSError, ReleaseBundleError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
