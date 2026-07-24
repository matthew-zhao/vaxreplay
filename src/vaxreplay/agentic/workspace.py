"""Atomic construction and exact verification of Agentic Replay workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vaxreplay.agentic.schema import (
    AGENTIC_LOGICAL_WORKSPACE_CONTRACT_SHA256,
    AGENTIC_LOGICAL_WORKSPACE_CONTRACT_VERSION,
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticBuildPolicy,
    AgenticDiscoveryDisposition,
    AgenticDiscoveryManifest,
    AgenticMediaType,
    AgenticSourceCatalogEntry,
    AgenticTaskEnvelope,
    AgenticTransformationReceipt,
    AgenticWorkspaceEntry,
    AgenticWorkspaceManifest,
    AgenticWorkspaceSource,
    agentic_model_sha256,
    normalized_relative_path,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import EpisodeManifest

_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
_MANIFEST_PATH = 'private/workspace-manifest.json'
_SOURCE_INVENTORY_PATH = 'private/source-inventory.json'
_TRANSFORM_INVENTORY_PATH = 'private/transformation-inventory.json'
_MODEL_SURFACE_PATH = 'private/model-visible-surface.json'
_BUILD_POLICY_PATH = 'private/build-policy.json'
_DISCOVERY_MANIFEST_PATH = 'private/discovery-manifest.json'
_EPISODE_MANIFEST_PATH = 'private/episode-manifest.json'
_PRIVATE_FILES = {
    _BUILD_POLICY_PATH,
    _DISCOVERY_MANIFEST_PATH,
    _EPISODE_MANIFEST_PATH,
    _MANIFEST_PATH,
    _SOURCE_INVENTORY_PATH,
    _TRANSFORM_INVENTORY_PATH,
    _MODEL_SURFACE_PATH,
}
_MODEL_SURFACE_MAGIC = b'VAXREPLAY-AGENTIC-MODEL-SURFACE-V0.2\n'
_MODEL_SURFACE_END = b'\nEND-FILE\n'


def _normalize_broker_path(path: str) -> str:
    try:
        return normalized_relative_path(path)
    except ValueError as error:
        raise AgenticWorkspaceError(f'invalid logical workspace path: {path}') from error


class AgenticWorkspaceError(ValueError):
    """Raised when a workspace contains an unbound or unsafe model-visible artifact."""


@dataclass(frozen=True)
class AgenticLogicalFile:
    """The only per-file metadata exposed by the logical worker workspace."""

    path: str
    media_type: AgenticMediaType
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class AgenticLogicalSearchHit:
    path: str
    start_byte: int
    end_byte: int


class AgenticLogicalWorkspaceBroker:
    """Metadata-minimal LIST/READ/SEARCH view over the committed workspace bytes.

    This in-process object defines and exercises the broker contract. It is not itself a hostile-code
    security boundary: a production executor must place participant code outside the organizer process
    and expose only these operations over an authenticated IPC channel.
    """

    contract_version = AGENTIC_LOGICAL_WORKSPACE_CONTRACT_VERSION
    contract_sha256 = AGENTIC_LOGICAL_WORKSPACE_CONTRACT_SHA256
    raw_host_filesystem_exposure_sealed = False

    def __init__(self, *, entries: tuple[AgenticWorkspaceEntry, ...], surface: bytes) -> None:
        files = parse_model_visible_surface_bytes(surface)
        by_path = {entry.path: entry for entry in entries}
        if set(files) != set(by_path):
            raise AgenticWorkspaceError('logical workspace broker inventory does not match its manifest')
        self._files = files
        self._metadata = tuple(
            AgenticLogicalFile(
                path=path,
                media_type=by_path[path].media_type,
                sha256=by_path[path].sha256,
                byte_count=by_path[path].byte_count,
            )
            for path in sorted(files)
        )

    def list_files(self) -> tuple[AgenticLogicalFile, ...]:
        return self._metadata

    def read(self, path: str, *, offset: int = 0, limit: int | None = None) -> bytes:
        normalized_relative_path = _normalize_broker_path(path)
        content = self._files.get(normalized_relative_path)
        if content is None:
            raise AgenticWorkspaceError(f'logical workspace path is not in the committed surface: {path}')
        if offset < 0 or offset > len(content):
            raise AgenticWorkspaceError('logical workspace read offset is outside the file')
        if limit is not None and limit <= 0:
            raise AgenticWorkspaceError('logical workspace read limit must be positive')
        return content[offset:] if limit is None else content[offset : offset + limit]

    def search(
        self,
        needle: str,
        *,
        paths: tuple[str, ...] | None = None,
        max_results: int = 100,
    ) -> tuple[AgenticLogicalSearchHit, ...]:
        if not needle or len(needle.encode('utf-8')) > 4096:
            raise AgenticWorkspaceError('logical workspace search needle must contain 1 to 4096 UTF-8 bytes')
        if max_results <= 0 or max_results > 1000:
            raise AgenticWorkspaceError('logical workspace max_results must be between 1 and 1000')
        selected_paths = (
            tuple(sorted(self._files)) if paths is None else tuple(_normalize_broker_path(path) for path in paths)
        )
        if len(selected_paths) != len(set(selected_paths)) or any(path not in self._files for path in selected_paths):
            raise AgenticWorkspaceError('logical workspace search paths must be unique committed files')
        encoded = needle.encode('utf-8')
        hits: list[AgenticLogicalSearchHit] = []
        for path in selected_paths:
            content = self._files[path]
            start = 0
            while len(hits) < max_results:
                match = content.find(encoded, start)
                if match < 0:
                    break
                hits.append(AgenticLogicalSearchHit(path=path, start_byte=match, end_byte=match + len(encoded)))
                start = match + max(1, len(encoded))
            if len(hits) >= max_results:
                break
        return tuple(hits)


@dataclass(frozen=True)
class LoadedAgenticWorkspace:
    root: Path
    input_root: Path
    task: AgenticTaskEnvelope
    episode_manifest: EpisodeManifest
    build_policy: AgenticBuildPolicy
    discovery_manifest: AgenticDiscoveryManifest
    manifest: AgenticWorkspaceManifest
    manifest_sha256: str
    sources: tuple[AgenticWorkspaceSource, ...]
    transformations: tuple[AgenticTransformationReceipt, ...]
    catalog: tuple[AgenticSourceCatalogEntry, ...]
    model_visible_surface: bytes

    @property
    def source_by_id(self) -> dict[str, AgenticWorkspaceSource]:
        return {source.source_id: source for source in self.sources}

    def read_source(self, source_id: str) -> bytes:
        source = self.source_by_id.get(source_id)
        if source is None:
            raise AgenticWorkspaceError(f'unknown workspace source_id: {source_id}')
        return _read_regular_file(self.input_root / source.path, source.byte_count, expected_mode=0o444)

    def brokered_surface(self) -> AgenticLogicalWorkspaceBroker:
        """Return the committed logical surface; do not give worker code ``input_root``."""

        return AgenticLogicalWorkspaceBroker(entries=self.manifest.entries, surface=self.model_visible_surface)


def build_agentic_workspace(
    *,
    workspace_id: str,
    task: AgenticTaskEnvelope,
    episode_manifest: EpisodeManifest,
    build_policy: AgenticBuildPolicy,
    discovery_manifest: AgenticDiscoveryManifest,
    assurance_profile: AgenticAssuranceProfile,
    sources: tuple[AgenticWorkspaceSource, ...],
    transformations: tuple[AgenticTransformationReceipt, ...],
    source_bytes: dict[str, bytes],
    output_root: Path,
) -> LoadedAgenticWorkspace:
    """Build only the public task surface plus organizer-private binding manifests.

    Temporal proof verification and contamination admission are deliberately separate trusted
    phases. This compiler proves exact inventory and byte binding; it does not upgrade provenance.
    """

    _validate_build_inputs(
        task,
        episode_manifest,
        build_policy,
        discovery_manifest,
        sources,
        transformations,
        source_bytes,
    )
    target = output_root.expanduser().resolve()
    if target.exists():
        raise AgenticWorkspaceError(f'workspace output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        input_root = staging / 'input'
        private_root = staging / 'private'
        source_root = input_root / 'sources'
        source_root.mkdir(parents=True)
        private_root.mkdir()

        task_bytes = canonical_json_bytes(task)
        episode_manifest_bytes = canonical_json_bytes(episode_manifest)
        build_policy_bytes = canonical_json_bytes(build_policy)
        discovery_manifest_bytes = canonical_json_bytes(discovery_manifest)
        task_markdown = _render_task_markdown(task)
        catalog = tuple(
            AgenticSourceCatalogEntry(
                source_id=source.source_id,
                path=source.path,
                title=source.display_title,
                media_type=source.media_type,
            )
            for source in sources
        )
        catalog_bytes = canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.agentic-source-catalog.v0.1',
                'sources': [entry.model_dump(mode='json') for entry in catalog],
            }
        )
        visible_files: dict[str, bytes] = {
            'TASK.json': task_bytes,
            'TASK.md': task_markdown,
            'source-catalog.json': catalog_bytes,
        }
        for source in sources:
            content = source_bytes[source.source_id]
            _validate_visible_content(source.path, source.media_type, content)
            visible_files[source.path] = content

        entries = _workspace_entries(visible_files, sources)
        source_inventory_bytes = canonical_json_bytes([source.model_dump(mode='json') for source in sources])
        transform_inventory_bytes = canonical_json_bytes(
            [receipt.model_dump(mode='json') for receipt in transformations]
        )
        tree_sha256 = _workspace_tree_sha256(visible_files)
        model_surface = model_visible_surface_bytes(visible_files)
        manifest = AgenticWorkspaceManifest(
            workspace_id=workspace_id,
            task_id=task.task_id,
            episode_id=task.episode_id,
            episode_manifest_sha256=task.episode_manifest_sha256,
            decision_at=task.decision_at,
            assurance_profile=assurance_profile,
            historically_preregistered=task.historically_preregistered,
            build_policy_sha256=hashlib.sha256(build_policy_bytes).hexdigest(),
            discovery_manifest_sha256=hashlib.sha256(discovery_manifest_bytes).hexdigest(),
            alias_seed_commitment_sha256=build_policy.alias_seed_commitment_sha256,
            alias_permutation_receipt_sha256=agentic_model_sha256(discovery_manifest.alias_permutation_receipt),
            task_sha256=hashlib.sha256(task_bytes).hexdigest(),
            source_inventory_sha256=hashlib.sha256(source_inventory_bytes).hexdigest(),
            transformation_inventory_sha256=hashlib.sha256(transform_inventory_bytes).hexdigest(),
            workspace_tree_sha256=tree_sha256,
            model_visible_surface_sha256=hashlib.sha256(model_surface).hexdigest(),
            episode_synthetic=episode_manifest.synthetic,
            episode_split=episode_manifest.split,
            episode_label_commitment_scheme=episode_manifest.label_commitment_scheme,
            episode_reward_version=episode_manifest.reward_version,
            prospective_input_structurally_eligible=_prospective_input_structurally_eligible(
                task=task,
                episode_manifest=episode_manifest,
                assurance_profile=assurance_profile,
            ),
            entries=entries,
        )

        for relative_path, content in visible_files.items():
            destination = input_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            destination.chmod(0o444)
        (private_root / 'workspace-manifest.json').write_bytes(canonical_json_bytes(manifest))
        (private_root / 'episode-manifest.json').write_bytes(episode_manifest_bytes)
        (private_root / 'build-policy.json').write_bytes(build_policy_bytes)
        (private_root / 'discovery-manifest.json').write_bytes(discovery_manifest_bytes)
        (private_root / 'source-inventory.json').write_bytes(source_inventory_bytes)
        (private_root / 'transformation-inventory.json').write_bytes(transform_inventory_bytes)
        (private_root / 'model-visible-surface.json').write_bytes(model_surface)
        for path in private_root.iterdir():
            path.chmod(0o600)
        private_root.chmod(0o700)
        for directory in sorted(
            (path for path in input_root.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        input_root.chmod(0o555)
        os.replace(staging, target)
    except BaseException:
        _force_remove(staging)
        raise
    return load_agentic_workspace(target)


def load_agentic_workspace(root: Path) -> LoadedAgenticWorkspace:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise AgenticWorkspaceError('workspace package root cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise AgenticWorkspaceError(f'workspace package does not exist: {resolved}')
    input_root = resolved / 'input'
    _validate_package_topology(resolved)

    manifest_bytes = _read_regular_file(resolved / _MANIFEST_PATH, _MAX_FILE_BYTES, expected_mode=0o600)
    build_policy_bytes = _read_regular_file(
        resolved / _BUILD_POLICY_PATH,
        _MAX_FILE_BYTES,
        expected_mode=0o600,
    )
    episode_manifest_bytes = _read_regular_file(
        resolved / _EPISODE_MANIFEST_PATH,
        _MAX_FILE_BYTES,
        expected_mode=0o600,
    )
    discovery_manifest_bytes = _read_regular_file(
        resolved / _DISCOVERY_MANIFEST_PATH,
        _MAX_WORKSPACE_BYTES,
        expected_mode=0o600,
    )
    source_inventory_bytes = _read_regular_file(
        resolved / _SOURCE_INVENTORY_PATH,
        _MAX_WORKSPACE_BYTES,
        expected_mode=0o600,
    )
    transform_inventory_bytes = _read_regular_file(
        resolved / _TRANSFORM_INVENTORY_PATH,
        _MAX_WORKSPACE_BYTES,
        expected_mode=0o600,
    )
    surface_bytes = _read_regular_file(
        resolved / _MODEL_SURFACE_PATH,
        _MAX_WORKSPACE_BYTES * 2,
        expected_mode=0o600,
    )
    try:
        manifest = AgenticWorkspaceManifest.model_validate_json(manifest_bytes)
        episode_manifest = EpisodeManifest.model_validate_json(episode_manifest_bytes)
        build_policy = AgenticBuildPolicy.model_validate_json(build_policy_bytes)
        discovery_manifest = AgenticDiscoveryManifest.model_validate_json(discovery_manifest_bytes)
        source_values = json.loads(source_inventory_bytes)
        transform_values = json.loads(transform_inventory_bytes)
        sources = tuple(
            AgenticWorkspaceSource.model_validate_json(canonical_json_bytes(value)) for value in source_values
        )
        transformations = tuple(
            AgenticTransformationReceipt.model_validate_json(canonical_json_bytes(value)) for value in transform_values
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AgenticWorkspaceError(f'invalid organizer-private workspace metadata: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise AgenticWorkspaceError('workspace manifest must use canonical JSON')
    if build_policy_bytes != canonical_json_bytes(build_policy):
        raise AgenticWorkspaceError('workspace build policy must use canonical JSON')
    if episode_manifest_bytes != canonical_json_bytes(episode_manifest):
        raise AgenticWorkspaceError('episode manifest must use canonical JSON')
    if hashlib.sha256(episode_manifest_bytes).hexdigest() != manifest.episode_manifest_sha256:
        raise AgenticWorkspaceError('episode manifest does not match the workspace manifest commitment')
    if discovery_manifest_bytes != canonical_json_bytes(discovery_manifest):
        raise AgenticWorkspaceError('workspace discovery manifest must use canonical JSON')
    if (
        hashlib.sha256(build_policy_bytes).hexdigest() != manifest.build_policy_sha256
        or hashlib.sha256(discovery_manifest_bytes).hexdigest() != manifest.discovery_manifest_sha256
        or build_policy.alias_seed_commitment_sha256 != manifest.alias_seed_commitment_sha256
        or agentic_model_sha256(discovery_manifest.alias_permutation_receipt)
        != manifest.alias_permutation_receipt_sha256
    ):
        raise AgenticWorkspaceError('workspace selection commitments do not match the manifest')
    if source_inventory_bytes != canonical_json_bytes([source.model_dump(mode='json') for source in sources]):
        raise AgenticWorkspaceError('source inventory must use canonical JSON')
    if transform_inventory_bytes != canonical_json_bytes(
        [receipt.model_dump(mode='json') for receipt in transformations]
    ):
        raise AgenticWorkspaceError('transformation inventory must use canonical JSON')
    if hashlib.sha256(source_inventory_bytes).hexdigest() != manifest.source_inventory_sha256:
        raise AgenticWorkspaceError('source inventory does not match the workspace manifest')
    if hashlib.sha256(transform_inventory_bytes).hexdigest() != manifest.transformation_inventory_sha256:
        raise AgenticWorkspaceError('transformation inventory does not match the workspace manifest')

    _validate_source_inventory(sources, transformations)
    visible_files = _read_exact_visible_inventory(input_root, manifest.entries)
    if _workspace_tree_sha256(visible_files) != manifest.workspace_tree_sha256:
        raise AgenticWorkspaceError('workspace tree does not match its manifest')
    expected_surface = model_visible_surface_bytes(visible_files)
    if surface_bytes != expected_surface or hashlib.sha256(surface_bytes).hexdigest() != (
        manifest.model_visible_surface_sha256
    ):
        raise AgenticWorkspaceError('model-visible surface does not match the exact workspace tree')

    try:
        task = AgenticTaskEnvelope.model_validate_json(visible_files['TASK.json'])
        catalog_value = json.loads(visible_files['source-catalog.json'])
        if not isinstance(catalog_value, dict) or catalog_value.get('schema_version') != (
            'vaxreplay.agentic-source-catalog.v0.1'
        ):
            raise ValueError('invalid source catalog schema_version')
        raw_catalog = catalog_value.get('sources')
        if not isinstance(raw_catalog, list):
            raise ValueError('source catalog sources must be a list')
        catalog = tuple(
            AgenticSourceCatalogEntry.model_validate_json(canonical_json_bytes(value)) for value in raw_catalog
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AgenticWorkspaceError(f'invalid public task or source catalog: {error}') from error
    if visible_files['TASK.json'] != canonical_json_bytes(task):
        raise AgenticWorkspaceError('TASK.json must use canonical JSON')
    if visible_files['TASK.md'] != _render_task_markdown(task):
        raise AgenticWorkspaceError('TASK.md is not the deterministic rendering of TASK.json')
    if hashlib.sha256(visible_files['TASK.json']).hexdigest() != manifest.task_sha256:
        raise AgenticWorkspaceError('TASK.json does not match the workspace manifest')
    if (
        task.task_id,
        task.episode_id,
        task.episode_manifest_sha256,
        task.decision_at,
        task.historically_preregistered,
    ) != (
        manifest.task_id,
        manifest.episode_id,
        manifest.episode_manifest_sha256,
        manifest.decision_at,
        manifest.historically_preregistered,
    ):
        raise AgenticWorkspaceError('public task identity does not match the workspace manifest')
    _validate_selection_contract(task, build_policy, discovery_manifest, sources, transformations)
    _validate_episode_manifest(task, episode_manifest)
    if (
        manifest.episode_synthetic,
        manifest.episode_split,
        manifest.episode_label_commitment_scheme,
        manifest.episode_reward_version,
        manifest.prospective_input_structurally_eligible,
    ) != (
        episode_manifest.synthetic,
        episode_manifest.split,
        episode_manifest.label_commitment_scheme,
        episode_manifest.reward_version,
        _prospective_input_structurally_eligible(
            task=task,
            episode_manifest=episode_manifest,
            assurance_profile=manifest.assurance_profile,
        ),
    ):
        raise AgenticWorkspaceError('workspace manifest does not bind the exact episode release properties')
    expected_catalog = tuple(
        AgenticSourceCatalogEntry(
            source_id=source.source_id,
            path=source.path,
            title=source.display_title,
            media_type=source.media_type,
        )
        for source in sources
    )
    if catalog != expected_catalog:
        raise AgenticWorkspaceError('public source catalog does not match the private source inventory')
    for source in sources:
        content = visible_files.get(source.path)
        if content is None or (hashlib.sha256(content).hexdigest(), len(content)) != (
            source.sha256,
            source.byte_count,
        ):
            raise AgenticWorkspaceError(f'workspace source bytes do not match inventory: {source.source_id}')
        _validate_visible_content(source.path, source.media_type, content)

    return LoadedAgenticWorkspace(
        root=resolved,
        input_root=input_root,
        task=task,
        episode_manifest=episode_manifest,
        build_policy=build_policy,
        discovery_manifest=discovery_manifest,
        manifest=manifest,
        manifest_sha256=agentic_model_sha256(manifest),
        sources=sources,
        transformations=transformations,
        catalog=catalog,
        model_visible_surface=surface_bytes,
    )


def model_visible_surface_bytes(visible_files: dict[str, bytes]) -> bytes:
    """Unescaped UTF-8 framing that preserves every raw path/content n-gram for auditing."""

    framed = bytearray(_MODEL_SURFACE_MAGIC)
    for path in sorted(visible_files):
        path_bytes = path.encode('utf-8')
        content = visible_files[path]
        try:
            content.decode('utf-8')
        except UnicodeDecodeError as error:
            raise AgenticWorkspaceError(f'model-visible file is not UTF-8: {path}') from error
        framed.extend(f'FILE {len(path_bytes)} {len(content)}\n'.encode('ascii'))
        framed.extend(path_bytes)
        framed.extend(b'\n')
        framed.extend(content)
        framed.extend(_MODEL_SURFACE_END)
    return bytes(framed)


def parse_model_visible_surface_bytes(payload: bytes) -> dict[str, bytes]:
    """Parse and canonicalize the length-framed exact audit surface."""

    if not payload.startswith(_MODEL_SURFACE_MAGIC):
        raise AgenticWorkspaceError('model-visible surface has an unsupported framing version')
    offset = len(_MODEL_SURFACE_MAGIC)
    files: dict[str, bytes] = {}
    previous_path: str | None = None
    while offset < len(payload):
        header_end = payload.find(b'\n', offset)
        if header_end < 0:
            raise AgenticWorkspaceError('model-visible surface has a truncated file header')
        header = payload[offset:header_end].split(b' ')
        if len(header) != 3 or header[0] != b'FILE':
            raise AgenticWorkspaceError('model-visible surface has an invalid file header')
        try:
            path_bytes_length = int(header[1])
            content_length = int(header[2])
        except ValueError as error:
            raise AgenticWorkspaceError('model-visible surface has a nonnumeric frame length') from error
        if path_bytes_length <= 0 or content_length <= 0:
            raise AgenticWorkspaceError('model-visible surface frame lengths must be positive')
        path_start = header_end + 1
        path_end = path_start + path_bytes_length
        content_start = path_end + 1
        content_end = content_start + content_length
        frame_end = content_end + len(_MODEL_SURFACE_END)
        if (
            frame_end > len(payload)
            or payload[path_end:content_start] != b'\n'
            or (payload[content_end:frame_end] != _MODEL_SURFACE_END)
        ):
            raise AgenticWorkspaceError('model-visible surface frame length or delimiter is invalid')
        try:
            path = payload[path_start:path_end].decode('utf-8')
            payload[content_start:content_end].decode('utf-8')
        except UnicodeDecodeError as error:
            raise AgenticWorkspaceError('model-visible surface paths and contents must be UTF-8') from error
        if previous_path is not None and path <= previous_path:
            raise AgenticWorkspaceError('model-visible surface paths must be unique and sorted')
        files[path] = payload[content_start:content_end]
        previous_path = path
        offset = frame_end
    if not files or model_visible_surface_bytes(files) != payload:
        raise AgenticWorkspaceError('model-visible surface is not canonical')
    return files


def _validate_build_inputs(
    task: AgenticTaskEnvelope,
    episode_manifest: EpisodeManifest,
    build_policy: AgenticBuildPolicy,
    discovery_manifest: AgenticDiscoveryManifest,
    sources: tuple[AgenticWorkspaceSource, ...],
    transformations: tuple[AgenticTransformationReceipt, ...],
    source_bytes: dict[str, bytes],
) -> None:
    if not sources:
        raise AgenticWorkspaceError('Agentic workspace requires at least one source')
    source_ids = tuple(source.source_id for source in sources)
    source_paths = tuple(source.path for source in sources)
    if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
        raise AgenticWorkspaceError('workspace sources must use unique source IDs in sorted order')
    if source_paths != tuple(sorted(source_paths)) or len(source_paths) != len(set(source_paths)):
        raise AgenticWorkspaceError('workspace source paths must be unique and sorted')
    folded_paths = tuple(path.casefold() for path in source_paths)
    if len(folded_paths) != len(set(folded_paths)):
        raise AgenticWorkspaceError('workspace source paths cannot collide under Unicode case folding')
    if set(source_bytes) != set(source_ids):
        raise AgenticWorkspaceError('source_bytes must cover every and only declared workspace source')
    total_bytes = 0
    for source in sources:
        content = source_bytes[source.source_id]
        total_bytes += len(content)
        if total_bytes > _MAX_WORKSPACE_BYTES:
            raise AgenticWorkspaceError('workspace source bytes exceed the aggregate limit')
        if (hashlib.sha256(content).hexdigest(), len(content)) != (source.sha256, source.byte_count):
            raise AgenticWorkspaceError(f'source byte binding mismatch: {source.source_id}')
    _validate_source_inventory(sources, transformations)
    _validate_episode_manifest(task, episode_manifest)
    _validate_selection_contract(task, build_policy, discovery_manifest, sources, transformations)
    if task.episode_id == '' or task.episode_manifest_sha256 == '':  # unreachable under strict schema
        raise AgenticWorkspaceError('task identity cannot be empty')


def _validate_episode_manifest(task: AgenticTaskEnvelope, episode: EpisodeManifest) -> None:
    if agentic_model_sha256(episode) != task.episode_manifest_sha256:
        raise AgenticWorkspaceError('public task does not bind the exact episode manifest')
    if (
        episode.episode_id,
        episode.decision_at,
        episode.task_type,
        tuple(episode.candidate_ids),
        episode.portfolio_size,
        episode.closed_book,
        episode.network_allowed,
    ) != (
        task.episode_id,
        task.decision_at,
        task.task_type,
        task.candidate_ids,
        task.portfolio_size,
        True,
        False,
    ):
        raise AgenticWorkspaceError('episode manifest semantics do not match the public Agentic task')


def _prospective_input_structurally_eligible(
    *,
    task: AgenticTaskEnvelope,
    episode_manifest: EpisodeManifest,
    assurance_profile: AgenticAssuranceProfile,
) -> bool:
    """Structural precondition only; temporal admission and a release seal remain mandatory."""

    return (
        not episode_manifest.synthetic
        and episode_manifest.split.value == 'test'
        and episode_manifest.label_commitment_scheme.value == 'hmac-sha256'
        and assurance_profile == AgenticAssuranceProfile.PROSPECTIVE_EXACT
        and task.historically_preregistered
    )


def _validate_source_inventory(
    sources: tuple[AgenticWorkspaceSource, ...],
    transformations: tuple[AgenticTransformationReceipt, ...],
) -> None:
    source_by_id = {source.source_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise AgenticWorkspaceError('workspace source IDs must be unique')
    receipt_ids = tuple(receipt.receipt_id for receipt in transformations)
    if receipt_ids != tuple(sorted(receipt_ids)) or len(receipt_ids) != len(set(receipt_ids)):
        raise AgenticWorkspaceError('transformation receipts must use unique IDs in sorted order')
    output_ids = tuple(receipt.output_source_id for receipt in transformations)
    if len(output_ids) != len(set(output_ids)):
        raise AgenticWorkspaceError('transformation receipts must use unique output source IDs')
    receipt_by_id = {receipt.receipt_id: receipt for receipt in transformations}
    for source in sources:
        if source.artifact_kind.value == 'raw':
            continue
        receipt = receipt_by_id.get(source.transformation_receipt_id or '')
        if receipt is None:
            raise AgenticWorkspaceError(f'derived source lacks its transformation receipt: {source.source_id}')
        if receipt.output_source_id != source.source_id or (receipt.output_sha256, receipt.output_bytes) != (
            source.sha256,
            source.byte_count,
        ):
            raise AgenticWorkspaceError('transformation output binding does not match its derived source')
        if receipt.input_source_ids != source.parent_source_ids:
            raise AgenticWorkspaceError('transformation inputs do not match derived source parents')
        if any(parent not in source_by_id for parent in source.parent_source_ids):
            raise AgenticWorkspaceError('derived source references an unknown parent')
    derived_ids = {source.source_id for source in sources if source.artifact_kind.value == 'derived'}
    if set(output_ids) != derived_ids:
        raise AgenticWorkspaceError('transformation receipts must cover every and only derived source')
    _assert_acyclic_sources(source_by_id)


def _validate_selection_contract(
    task: AgenticTaskEnvelope,
    build_policy: AgenticBuildPolicy,
    discovery_manifest: AgenticDiscoveryManifest,
    sources: tuple[AgenticWorkspaceSource, ...],
    transformations: tuple[AgenticTransformationReceipt, ...],
) -> None:
    if (build_policy.task_id, build_policy.decision_at) != (task.task_id, task.decision_at):
        raise AgenticWorkspaceError('build policy does not match the public task identity and cutoff')
    if discovery_manifest.task_id != task.task_id or discovery_manifest.build_policy_sha256 != agentic_model_sha256(
        build_policy
    ):
        raise AgenticWorkspaceError('discovery manifest does not match the pinned build policy and task')

    alias_receipt = discovery_manifest.alias_permutation_receipt
    if (
        alias_receipt.task_id != task.task_id
        or alias_receipt.alias_scheme_sha256 != build_policy.alias_scheme_sha256
        or alias_receipt.alias_seed_commitment_sha256 != build_policy.alias_seed_commitment_sha256
        or alias_receipt.permutation_algorithm_id != build_policy.alias_permutation_algorithm_id
        or alias_receipt.generator_id != build_policy.alias_generator_id
        or alias_receipt.generator_version != build_policy.alias_generator_version
        or alias_receipt.generator_executable_sha256 != build_policy.alias_generator_executable_sha256
        or alias_receipt.generator_config_sha256 != build_policy.alias_generator_config_sha256
    ):
        raise AgenticWorkspaceError('alias permutation receipt does not match the pinned build policy and task')
    if not (build_policy.created_at <= alias_receipt.generated_at <= discovery_manifest.created_at):
        raise AgenticWorkspaceError(
            'alias generation must follow its build-policy precommit and precede discovery finalization'
        )

    expected_candidate_ids = tuple(f'candidate-{index:03d}' for index in range(1, len(task.candidate_ids) + 1))
    if task.candidate_ids != expected_candidate_ids:
        raise AgenticWorkspaceError('public candidates must use contiguous neutral aliases in presentation order')
    receipt_candidate_ids = tuple(item.public_candidate_id for item in alias_receipt.candidate_assignments)
    if receipt_candidate_ids != task.candidate_ids:
        raise AgenticWorkspaceError('public candidate order does not match the alias permutation receipt')

    expected_source_presentations = tuple(
        (
            f'source-{index:03d}',
            f'sources/source-{index:03d}{Path(source.path).suffix}',
            f'Source {index:03d}',
            source.sha256,
            source.byte_count,
        )
        for index, source in enumerate(sources, start=1)
    )
    actual_source_presentations = tuple(
        (source.source_id, source.path, source.display_title, source.sha256, source.byte_count) for source in sources
    )
    if actual_source_presentations != expected_source_presentations:
        raise AgenticWorkspaceError(
            'workspace sources must use contiguous neutral IDs, paths, titles, and presentation order'
        )
    receipt_source_presentations = tuple(
        (
            item.public_source_id,
            item.public_path,
            item.public_title,
            item.artifact_sha256,
            item.artifact_bytes,
        )
        for item in alias_receipt.source_assignments
    )
    if receipt_source_presentations != actual_source_presentations:
        raise AgenticWorkspaceError('public source order does not match the alias permutation receipt')

    raw_by_id = {source.source_id: source for source in sources if source.artifact_kind == AgenticArtifactKind.RAW}
    included_sources = {
        record.workspace_source_id: record
        for record in discovery_manifest.sources
        if record.disposition == AgenticDiscoveryDisposition.INCLUDED
    }
    if set(included_sources) != set(raw_by_id):
        raise AgenticWorkspaceError('workspace raw sources do not exactly match included discovery records')
    for source_id, source in raw_by_id.items():
        record = included_sources[source_id]
        if (
            record.artifact_sha256,
            record.artifact_bytes,
            record.selected_temporal_proof_id,
            record.effective_available_at_upper,
        ) != (
            source.sha256,
            source.byte_count,
            source.selected_proof_id,
            source.effective_available_at_upper,
        ):
            raise AgenticWorkspaceError('included discovery record does not bind its exact workspace source')

    included_candidates = {
        candidate.public_candidate_id
        for candidate in discovery_manifest.candidates
        if candidate.disposition == AgenticDiscoveryDisposition.INCLUDED
    }
    if included_candidates != set(task.candidate_ids):
        raise AgenticWorkspaceError('public candidate aliases do not exactly match included discovery candidates')
    candidate_keys_by_alias = {
        candidate.public_candidate_id: candidate.candidate_key_commitment_sha256
        for candidate in discovery_manifest.candidates
        if candidate.disposition == AgenticDiscoveryDisposition.INCLUDED
    }
    receipt_candidate_keys_by_alias = {
        assignment.public_candidate_id: assignment.candidate_key_commitment_sha256
        for assignment in alias_receipt.candidate_assignments
    }
    if receipt_candidate_keys_by_alias != candidate_keys_by_alias:
        raise AgenticWorkspaceError('alias permutation receipt does not bind the included private candidate keys')

    allowed_transforms = {
        (
            item.transform_id,
            item.transform_version,
            item.executable_sha256,
            item.config_sha256,
        )
        for item in build_policy.allowed_transforms
    }
    actual_transforms = {
        (
            receipt.transform_id,
            receipt.transform_version,
            receipt.executable_sha256,
            receipt.config_sha256,
        )
        for receipt in transformations
    }
    if actual_transforms != allowed_transforms:
        raise AgenticWorkspaceError('workspace transformations do not exactly match the build-policy allowlist')


def _assert_acyclic_sources(source_by_id: dict[str, AgenticWorkspaceSource]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(source_id: str) -> None:
        if source_id in visiting:
            raise AgenticWorkspaceError('workspace provenance graph contains a cycle')
        if source_id in visited:
            return
        visiting.add(source_id)
        for parent in source_by_id[source_id].parent_source_ids:
            visit(parent)
        visiting.remove(source_id)
        visited.add(source_id)

    for source_id in source_by_id:
        visit(source_id)


def _workspace_entries(
    visible_files: dict[str, bytes],
    sources: tuple[AgenticWorkspaceSource, ...],
) -> tuple[AgenticWorkspaceEntry, ...]:
    source_by_path = {source.path: source for source in sources}
    system_metadata = {
        'TASK.json': (AgenticMediaType.JSON, 'task-envelope'),
        'TASK.md': (AgenticMediaType.MARKDOWN, 'task-instructions'),
        'source-catalog.json': (AgenticMediaType.JSON, 'source-catalog'),
    }
    entries: list[AgenticWorkspaceEntry] = []
    for path in sorted(visible_files):
        content = visible_files[path]
        if path in system_metadata:
            media_type, provenance_node = system_metadata[path]
        else:
            source = source_by_path[path]
            media_type, provenance_node = source.media_type, f'source:{source.source_id}'
        entries.append(
            AgenticWorkspaceEntry(
                path=path,
                sha256=hashlib.sha256(content).hexdigest(),
                byte_count=len(content),
                media_type=media_type,
                provenance_node_id=provenance_node,
            )
        )
    return tuple(entries)


def _workspace_tree_sha256(visible_files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path in sorted(visible_files):
        path_bytes = path.encode('utf-8')
        content = visible_files[path]
        digest.update(len(path_bytes).to_bytes(8, 'big'))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()


def _render_task_markdown(task: AgenticTaskEnvelope) -> bytes:
    text = (
        '# Agentic Replay task\n\n'
        f'Task ID: `{task.task_id}`\n\n'
        f'Historical cutoff: `{task.decision_at.isoformat()}`\n\n'
        f'{task.instructions.rstrip()}\n\n'
        'Use only files under this read-only input workspace. Write exactly one final '
        '`submission.json` using the response protocol declared in `TASK.json`.\n'
    )
    return text.encode('utf-8')


def _validate_visible_content(path: str, media_type: AgenticMediaType, content: bytes) -> None:
    if not content or len(content) > _MAX_FILE_BYTES:
        raise AgenticWorkspaceError(f'model-visible file is empty or exceeds its size limit: {path}')
    if b'\x00' in content or b'\r' in content:
        raise AgenticWorkspaceError(f'model-visible files cannot contain NUL or noncanonical CR newlines: {path}')
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError as error:
        raise AgenticWorkspaceError(f'model-visible file is not UTF-8: {path}') from error
    if media_type == AgenticMediaType.JSON:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise AgenticWorkspaceError(f'invalid JSON workspace source: {path}') from error
        if content != canonical_json_bytes(value):
            raise AgenticWorkspaceError(f'JSON workspace source must use canonical encoding: {path}')
    elif media_type == AgenticMediaType.JSONL:
        lines = content.splitlines(keepends=True)
        if not lines or any(not line.endswith(b'\n') for line in lines):
            raise AgenticWorkspaceError(f'JSONL workspace source must end every record with LF: {path}')
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AgenticWorkspaceError(f'invalid JSONL workspace source: {path}') from error
            if line != canonical_json_bytes(value) + b'\n':
                raise AgenticWorkspaceError(f'JSONL workspace records must use canonical encoding: {path}')


def _read_exact_visible_inventory(
    input_root: Path,
    entries: tuple[AgenticWorkspaceEntry, ...],
) -> dict[str, bytes]:
    expected = {entry.path: entry for entry in entries}
    expected_directories = {'sources'}
    for entry in entries:
        parts = Path(entry.path).parts[:-1]
        for length in range(1, len(parts) + 1):
            expected_directories.add(Path(*parts[:length]).as_posix())
    actual: dict[str, Path] = {}
    actual_directories: set[str] = set()
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(input_root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink() or name.startswith('.'):
                raise AgenticWorkspaceError('workspace directories cannot be symlinks or hidden')
            relative = path.relative_to(input_root).as_posix()
            _validate_directory(path, expected_mode=0o555)
            actual_directories.add(relative)
        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(input_root).as_posix()
            if name.startswith('.') or path.is_symlink():
                raise AgenticWorkspaceError('workspace files cannot be symlinks or hidden')
            actual[relative] = path
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        raise AgenticWorkspaceError(f'workspace directory inventory mismatch; missing={missing}, extra={extra}')
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AgenticWorkspaceError(f'workspace exact inventory mismatch; missing={missing}, extra={extra}')
    visible: dict[str, bytes] = {}
    for path, entry in expected.items():
        content = _read_regular_file(actual[path], entry.byte_count, expected_mode=0o444)
        total_bytes += len(content)
        if total_bytes > _MAX_WORKSPACE_BYTES:
            raise AgenticWorkspaceError('workspace exceeds the aggregate byte limit')
        if (hashlib.sha256(content).hexdigest(), len(content)) != (entry.sha256, entry.byte_count):
            raise AgenticWorkspaceError(f'workspace file binding mismatch: {path}')
        _validate_visible_content(path, entry.media_type, content)
        visible[path] = content
    return visible


def _validate_package_topology(root: Path) -> None:
    names: set[str] = set()
    for entry in os.scandir(root):
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise AgenticWorkspaceError('workspace package root can contain only input/ and private/ directories')
        names.add(entry.name)
    if names != {'input', 'private'}:
        raise AgenticWorkspaceError('workspace package root must contain exactly input/ and private/')
    _validate_directory(root / 'input', expected_mode=0o555)
    _validate_directory(root / 'private', expected_mode=0o700)
    private_files: set[str] = set()
    private_invalid: list[str] = []
    for entry in os.scandir(root / 'private'):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            private_invalid.append(entry.name)
        else:
            private_files.add(f'private/{entry.name}')
    if private_files != _PRIVATE_FILES or private_invalid:
        raise AgenticWorkspaceError('workspace private metadata has an unexpected inventory')


def _validate_directory(path: Path, *, expected_mode: int) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise AgenticWorkspaceError(f'cannot inspect workspace directory {path}: {error}') from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise AgenticWorkspaceError(
            f'workspace directory must have mode {expected_mode:04o} and cannot be another file type: {path}'
        )
    if hasattr(os, 'listxattr') and os.listxattr(path, follow_symlinks=False):
        raise AgenticWorkspaceError(f'workspace directories cannot carry extended attributes: {path}')


def _read_regular_file(path: Path, maximum_bytes: int, *, expected_mode: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AgenticWorkspaceError(f'cannot open workspace file {path}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AgenticWorkspaceError(f'workspace artifact must be one regular, unlinked file: {path}')
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise AgenticWorkspaceError(f'workspace file must have mode {expected_mode:04o}: {path}')
        if metadata.st_size > maximum_bytes:
            raise AgenticWorkspaceError(f'workspace file exceeds its declared or global size limit: {path}')
        if hasattr(os, 'listxattr') and os.listxattr(path, follow_symlinks=False):
            raise AgenticWorkspaceError(f'workspace files cannot carry extended attributes: {path}')
        content = bytearray()
        while True:
            remaining = maximum_bytes - len(content)
            chunk = os.read(descriptor, min(65_536, remaining + 1))
            if not chunk:
                return bytes(content)
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise AgenticWorkspaceError(f'workspace file exceeds its size limit: {path}')
    except OSError as error:
        raise AgenticWorkspaceError(f'cannot read workspace file {path}: {error}') from error
    finally:
        os.close(descriptor)


def _force_remove(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob('*'):
        try:
            child.chmod(0o700)
        except OSError:
            pass
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)
