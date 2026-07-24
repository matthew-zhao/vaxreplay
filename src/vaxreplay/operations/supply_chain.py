"""Offline source-worker OCI supply-chain verification.

The verifier in this module deliberately does not contact a registry, package index,
transparency service, or build service.  A caller must retain the exact source archive,
wheel, runtime lock, recipe, SBOM, provenance statement, OCI manifest/config, every
layer, and every runtime distribution.  The verified result is a cryptographic binding
of those bytes; it is not a claim that a builder or registry is independently operated.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.hermetic_execution import IMPLEMENTATION_LABEL, WORKER_PROTOCOL
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_OCI_DIGEST_PATTERN = r'^sha256:[0-9a-f]{64}$'
_DIGEST_REF_PATTERN = re.compile(r'^.+@sha256:([0-9a-f]{64})$')
_PACKAGE_NAME_PATTERN = r'^[a-z0-9]+(?:[._-][a-z0-9]+)*$'
_MAX_JSON_BYTES = 16 * 1024 * 1024
_LAYER_IO_CHUNK_BYTES = 1024 * 1024
_MAX_UNCOMPRESSED_LAYER_BYTES = 8 * 1024 * 1024 * 1024
_OCI_UNCOMPRESSED_LAYER_MEDIA_TYPE = 'application/vnd.oci.image.layer.v1.tar'
_OCI_GZIP_LAYER_MEDIA_TYPES = {
    'application/vnd.docker.image.rootfs.diff.tar.gzip',
    'application/vnd.oci.image.layer.v1.tar+gzip',
}
_GZIP_MAGIC = b'\x1f\x8b\x08'
_ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'

SOURCE_WORKER_SBOM_SCHEMA_VERSION = 'vaxreplay.source-worker-sbom.v0.1'
SOURCE_WORKER_PROVENANCE_SCHEMA_VERSION = 'vaxreplay.source-worker-provenance.v0.1'
SOURCE_WORKER_VERIFICATION_SCHEMA_VERSION = 'vaxreplay.source-worker-supply-chain-report.v0.1'

SourceWorkerName = Literal[
    'ctgov-source-verifier',
    'ctgov-study-adapter',
    'iedb-antigen-adapter',
    'iedb-source-verifier',
    'immport-arm-adapter',
    'immport-authenticated-producer',
    'immport-source-verifier',
]
OciPlatform = Literal['linux/amd64', 'linux/arm64']


class SourceWorkerSupplyChainError(ValueError):
    """A retained build material or cross-artifact binding failed closed."""


class RetainedMaterial(StrictModel):
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(ge=0)


class SourceWorkerComponent(StrictModel):
    """One installed Python distribution retained alongside the image."""

    name: str = Field(pattern=_PACKAGE_NAME_PATTERN, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    purl: str = Field(pattern=r'^pkg:pypi/[a-z0-9._-]+@[A-Za-z0-9][A-Za-z0-9._+!-]{0,199}$', max_length=512)
    distribution_filename: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}$')
    distribution: RetainedMaterial
    licenses: tuple[str, ...] = Field(min_length=1, max_length=32)

    @field_validator('licenses')
    @classmethod
    def validate_licenses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError('component licenses must be sorted and unique')
        if any(not item or len(item) > 200 or item.strip() != item or '\x00' in item for item in value):
            raise ValueError('component licenses must be nonempty, trimmed, bounded strings')
        return value

    @model_validator(mode='after')
    def validate_purl_identity(self) -> Self:
        expected = f'pkg:pypi/{self.name}@{self.version}'
        if self.purl != expected:
            raise ValueError('component purl must exactly bind its normalized name and version')
        return self


class SourceWorkerSbom(StrictModel):
    """Canonical compact SBOM whose subjects are the retained build inputs and image."""

    schema_version: Literal['vaxreplay.source-worker-sbom.v0.1'] = SOURCE_WORKER_SBOM_SCHEMA_VERSION
    sbom_id: str = Field(pattern=SAFE_ID_PATTERN)
    generated_at: datetime
    primary_package_name: Literal['vaxreplay'] = 'vaxreplay'
    source_archive: RetainedMaterial
    primary_package: RetainedMaterial
    runtime_lock: RetainedMaterial
    build_recipe: RetainedMaterial
    image_manifest: RetainedMaterial
    components: tuple[SourceWorkerComponent, ...] = Field(min_length=2, max_length=10_000)

    @field_validator('generated_at')
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return aware_utc(value, 'SBOM generated_at')

    @model_validator(mode='after')
    def validate_components(self) -> Self:
        keys = tuple((item.name, item.version, item.distribution.sha256) for item in self.components)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError('SBOM components must use unique canonical sort order')
        primary = tuple(item for item in self.components if item.name == self.primary_package_name)
        if len(primary) != 1 or primary[0].distribution != self.primary_package:
            raise ValueError('SBOM must contain exactly one vaxreplay component bound to the primary package')
        return self


class SourceWorkerBuildProvenance(StrictModel):
    """Canonical build statement binding all retained materials to one OCI target."""

    schema_version: Literal['vaxreplay.source-worker-provenance.v0.1'] = SOURCE_WORKER_PROVENANCE_SCHEMA_VERSION
    provenance_id: str = Field(pattern=SAFE_ID_PATTERN)
    builder_id: str = Field(pattern=SAFE_ID_PATTERN)
    build_started_at: datetime
    build_finished_at: datetime
    worker_name: SourceWorkerName
    package_version: str = Field(min_length=1, max_length=200)
    platform: OciPlatform
    base_image_ref: str = Field(min_length=1, max_length=2048)
    image_ref: str = Field(min_length=1, max_length=2048)
    resolved_image_id: str = Field(pattern=_OCI_DIGEST_PATTERN)
    source_archive: RetainedMaterial
    primary_package: RetainedMaterial
    runtime_lock: RetainedMaterial
    build_recipe: RetainedMaterial
    sbom: RetainedMaterial
    image_manifest: RetainedMaterial
    image_config: RetainedMaterial
    implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    entrypoint: tuple[str, ...] = Field(min_length=4, max_length=16)
    runtime_user: Literal['65532:65532'] = '65532:65532'
    labels: dict[str, str] = Field(min_length=9, max_length=32)
    build_network_disabled: Literal[True] = True
    dependency_resolution_disabled: Literal[True] = True
    digest_pinned_base_image: Literal[True] = True
    source_date_epoch: Literal[0] = 0

    @field_validator('build_started_at', 'build_finished_at')
    @classmethod
    def validate_build_timestamp(cls, value: datetime) -> datetime:
        return aware_utc(value, 'build timestamp')

    @field_validator('base_image_ref', 'image_ref')
    @classmethod
    def validate_digest_reference(cls, value: str) -> str:
        if value != value.strip() or _DIGEST_REF_PATTERN.fullmatch(value) is None:
            raise ValueError('OCI references must be named sha256 digest references')
        return value

    @field_validator('labels')
    @classmethod
    def validate_label_syntax(cls, value: dict[str, str]) -> dict[str, str]:
        for name, setting in value.items():
            if (
                not re.fullmatch(r'[a-z0-9]+(?:[._-][a-z0-9]+)*', name)
                or len(name) > 200
                or not setting
                or len(setting) > 2048
                or '\x00' in setting
            ):
                raise ValueError('OCI labels must use bounded portable names and nonempty values')
        return value

    @model_validator(mode='after')
    def validate_build_bindings(self) -> Self:
        if self.build_finished_at < self.build_started_at:
            raise ValueError('build_finished_at cannot predate build_started_at')
        manifest_match = _DIGEST_REF_PATTERN.fullmatch(self.image_ref)
        if manifest_match is None or manifest_match.group(1) != self.image_manifest.sha256:
            raise ValueError('image_ref must identify the exact OCI manifest digest')
        if self.resolved_image_id != f'sha256:{self.image_config.sha256}':
            raise ValueError('resolved_image_id must identify the exact OCI config digest')
        if self.worker_name == 'immport-authenticated-producer':
            expected_entrypoint = (
                '/usr/local/bin/python',
                '-I',
                '-m',
                'vaxreplay.operations.immport_producer_cli',
            )
        else:
            expected_entrypoint = (
                '/usr/local/bin/python',
                '-I',
                '-m',
                'vaxreplay.sources.worker_cli',
                self.worker_name,
            )
        if self.entrypoint != expected_entrypoint:
            raise ValueError('source-worker entrypoint must select exactly one reviewed worker')
        expected_labels = {
            IMPLEMENTATION_LABEL: self.implementation_sha256,
            'org.opencontainers.image.title': (
                'vaxreplay-immport-producer'
                if self.worker_name == 'immport-authenticated-producer'
                else 'vaxreplay-source-worker'
            ),
            'org.opencontainers.image.version': self.package_version,
            'org.vaxreplay.package-sha256': self.primary_package.sha256,
            'org.vaxreplay.recipe-sha256': self.build_recipe.sha256,
            'org.vaxreplay.runtime-lock-sha256': self.runtime_lock.sha256,
            'org.vaxreplay.source-archive-sha256': self.source_archive.sha256,
            'org.vaxreplay.worker': self.worker_name,
        }
        if self.worker_name == 'immport-authenticated-producer':
            expected_labels['org.vaxreplay.credential-fd'] = '3'
            expected_labels['org.vaxreplay.producer-protocol'] = 'vaxreplay.immport-producer-request.v0.1'
        else:
            expected_labels['org.vaxreplay.worker-protocol'] = WORKER_PROTOCOL
        if self.labels != expected_labels:
            raise ValueError('provenance labels must be the exact source-worker label set')
        return self


class SourceWorkerSupplyChainReport(StrictModel):
    schema_version: Literal['vaxreplay.source-worker-supply-chain-report.v0.1'] = (
        SOURCE_WORKER_VERIFICATION_SCHEMA_VERSION
    )
    provenance_id: str = Field(pattern=SAFE_ID_PATTERN)
    worker_name: SourceWorkerName
    platform: OciPlatform
    image_ref: str = Field(min_length=1, max_length=2048)
    resolved_image_id: str = Field(pattern=_OCI_DIGEST_PATTERN)
    source_archive_sha256: str = Field(pattern=_SHA256_PATTERN)
    primary_package_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    build_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    sbom_sha256: str = Field(pattern=_SHA256_PATTERN)
    provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    component_count: int = Field(ge=2)
    layer_count: int = Field(ge=1)
    every_component_distribution_verified: Literal[True] = True
    every_oci_layer_verified: Literal[True] = True
    every_rootfs_diff_id_verified: Literal[True] = True
    oci_config_policy_verified: Literal[True] = True
    external_builder_identity_verified: Literal[False] = False


def material_binding(payload: bytes) -> RetainedMaterial:
    if not isinstance(payload, bytes):
        raise TypeError('retained material must be exact bytes')
    return RetainedMaterial(sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload))


def verify_source_worker_supply_chain(
    *,
    source_archive_bytes: bytes,
    primary_package_bytes: bytes,
    runtime_lock_bytes: bytes,
    build_recipe_bytes: bytes,
    sbom_bytes: bytes,
    provenance_bytes: bytes,
    oci_manifest_bytes: bytes,
    oci_config_bytes: bytes,
    component_distribution_bytes: Mapping[str, bytes],
    oci_layer_bytes: Mapping[str, bytes],
) -> SourceWorkerSupplyChainReport:
    """Verify an entire retained source-worker build without network access.

    Distribution and layer mappings are keyed by lowercase SHA-256 digest without a
    ``sha256:`` prefix.  Extra entries are rejected so an archived verification bundle
    has one unambiguous inventory.
    """

    for label, payload in (
        ('source archive', source_archive_bytes),
        ('primary package', primary_package_bytes),
        ('runtime lock', runtime_lock_bytes),
        ('build recipe', build_recipe_bytes),
        ('SBOM', sbom_bytes),
        ('provenance', provenance_bytes),
        ('OCI manifest', oci_manifest_bytes),
        ('OCI config', oci_config_bytes),
    ):
        if not isinstance(payload, bytes) or not payload:
            raise SourceWorkerSupplyChainError(f'{label} must be nonempty exact bytes')

    sbom = _parse_canonical_model(sbom_bytes, SourceWorkerSbom, 'SBOM')
    provenance = _parse_canonical_model(provenance_bytes, SourceWorkerBuildProvenance, 'provenance')
    supplied_materials = {
        'source archive': (sbom.source_archive, source_archive_bytes),
        'primary package': (sbom.primary_package, primary_package_bytes),
        'runtime lock': (sbom.runtime_lock, runtime_lock_bytes),
        'build recipe': (sbom.build_recipe, build_recipe_bytes),
        'image manifest': (sbom.image_manifest, oci_manifest_bytes),
    }
    for label, (binding, payload) in supplied_materials.items():
        _require_material(binding, payload, label)
    _require_material(provenance.source_archive, source_archive_bytes, 'provenance source archive')
    _require_material(provenance.primary_package, primary_package_bytes, 'provenance primary package')
    _require_material(provenance.runtime_lock, runtime_lock_bytes, 'provenance runtime lock')
    _require_material(provenance.build_recipe, build_recipe_bytes, 'provenance build recipe')
    _require_material(provenance.sbom, sbom_bytes, 'provenance SBOM')
    _require_material(provenance.image_manifest, oci_manifest_bytes, 'provenance image manifest')
    _require_material(provenance.image_config, oci_config_bytes, 'provenance image config')
    if (
        provenance.source_archive != sbom.source_archive
        or provenance.primary_package != sbom.primary_package
        or provenance.runtime_lock != sbom.runtime_lock
        or provenance.build_recipe != sbom.build_recipe
        or provenance.image_manifest != sbom.image_manifest
    ):
        raise SourceWorkerSupplyChainError('SBOM and provenance bind different retained subjects')
    if sbom.generated_at < provenance.build_finished_at:
        raise SourceWorkerSupplyChainError('SBOM generation predates the completed image build')

    expected_distributions = {component.distribution.sha256: component.distribution for component in sbom.components}
    if set(component_distribution_bytes) != set(expected_distributions):
        raise SourceWorkerSupplyChainError('runtime distribution inventory differs from the exact SBOM components')
    for digest, binding in expected_distributions.items():
        _require_digest_key(digest, 'runtime distribution')
        _require_material(binding, component_distribution_bytes[digest], f'runtime distribution {digest}')
    if primary_package_bytes != component_distribution_bytes[sbom.primary_package.sha256]:
        raise SourceWorkerSupplyChainError('primary package bytes differ from their SBOM component bytes')

    manifest = _parse_json_object(oci_manifest_bytes, 'OCI manifest')
    config = _parse_json_object(oci_config_bytes, 'OCI config')
    layer_descriptors = _verify_oci_manifest(manifest, oci_config_bytes)
    expected_layers = {digest.removeprefix('sha256:'): size for digest, size, _media_type in layer_descriptors}
    if set(oci_layer_bytes) != set(expected_layers):
        raise SourceWorkerSupplyChainError('OCI layer inventory differs from the exact manifest')
    for digest, byte_count in expected_layers.items():
        _require_digest_key(digest, 'OCI layer')
        _require_material(
            RetainedMaterial(sha256=digest, byte_count=byte_count),
            oci_layer_bytes[digest],
            f'OCI layer {digest}',
        )
    diff_ids = _verify_oci_config(
        config,
        provenance,
        expected_layer_count=len(layer_descriptors),
    )
    _verify_oci_layer_diff_ids(layer_descriptors, oci_layer_bytes, diff_ids)

    return SourceWorkerSupplyChainReport(
        provenance_id=provenance.provenance_id,
        worker_name=provenance.worker_name,
        platform=provenance.platform,
        image_ref=provenance.image_ref,
        resolved_image_id=provenance.resolved_image_id,
        source_archive_sha256=provenance.source_archive.sha256,
        primary_package_sha256=provenance.primary_package.sha256,
        runtime_lock_sha256=provenance.runtime_lock.sha256,
        build_recipe_sha256=provenance.build_recipe.sha256,
        sbom_sha256=provenance.sbom.sha256,
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        component_count=len(sbom.components),
        layer_count=len(layer_descriptors),
    )


def _verify_oci_manifest(
    manifest: dict[str, Any],
    config_bytes: bytes,
) -> tuple[tuple[str, int, str], ...]:
    if manifest.get('schemaVersion') != 2:
        raise SourceWorkerSupplyChainError('OCI manifest must use schemaVersion 2')
    if manifest.get('mediaType') not in {
        'application/vnd.oci.image.manifest.v1+json',
        'application/vnd.docker.distribution.manifest.v2+json',
    }:
        raise SourceWorkerSupplyChainError('OCI manifest mediaType is not a supported image manifest')
    config_value = manifest.get('config')
    config = _parse_descriptor(config_value, 'OCI config descriptor')
    if not isinstance(config_value, dict) or config_value.get('mediaType') not in {
        'application/vnd.oci.image.config.v1+json',
        'application/vnd.docker.container.image.v1+json',
    }:
        raise SourceWorkerSupplyChainError('OCI config descriptor has an unsupported mediaType')
    if config[0] != f'sha256:{hashlib.sha256(config_bytes).hexdigest()}' or config[1] != len(config_bytes):
        raise SourceWorkerSupplyChainError('OCI config descriptor differs from exact config bytes')
    layers = manifest.get('layers')
    if not isinstance(layers, list) or not layers or len(layers) > 10_000:
        raise SourceWorkerSupplyChainError('OCI manifest must contain a bounded nonempty layer list')
    parsed = tuple(_parse_layer_descriptor(item) for item in layers)
    if len({digest for digest, _size, _media_type in parsed}) != len(parsed):
        raise SourceWorkerSupplyChainError('OCI manifest contains duplicate layer digests')
    return parsed


def _parse_layer_descriptor(value: object) -> tuple[str, int, str]:
    digest, size = _parse_descriptor(value, 'OCI layer descriptor')
    if not isinstance(value, dict):  # _parse_descriptor already rejects this; narrows the type.
        raise SourceWorkerSupplyChainError('OCI layer descriptor must be an object')
    media_type = value.get('mediaType')
    if not isinstance(media_type, str) or media_type not in {
        _OCI_UNCOMPRESSED_LAYER_MEDIA_TYPE,
        *_OCI_GZIP_LAYER_MEDIA_TYPES,
    }:
        raise SourceWorkerSupplyChainError('OCI layer descriptor has an unsupported mediaType')
    if size < 1:
        raise SourceWorkerSupplyChainError('OCI layer descriptor must bind a nonempty layer')
    return digest, size, media_type


def _parse_descriptor(value: object, label: str) -> tuple[str, int]:
    if not isinstance(value, dict) or set(value) - {'annotations', 'digest', 'mediaType', 'size', 'urls'}:
        raise SourceWorkerSupplyChainError(f'{label} contains unsupported fields')
    digest = value.get('digest')
    size = value.get('size')
    media_type = value.get('mediaType')
    if not isinstance(digest, str) or not re.fullmatch(_OCI_DIGEST_PATTERN, digest):
        raise SourceWorkerSupplyChainError(f'{label} has an invalid sha256 digest')
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SourceWorkerSupplyChainError(f'{label} has an invalid size')
    if not isinstance(media_type, str) or not media_type or len(media_type) > 200:
        raise SourceWorkerSupplyChainError(f'{label} has an invalid mediaType')
    if 'urls' in value:
        raise SourceWorkerSupplyChainError(f'{label} cannot depend on external blob URLs')
    annotations = value.get('annotations')
    if annotations is not None and not isinstance(annotations, dict):
        raise SourceWorkerSupplyChainError(f'{label} annotations must be an object')
    return digest, size


def _verify_oci_config(
    config: dict[str, Any],
    provenance: SourceWorkerBuildProvenance,
    *,
    expected_layer_count: int,
) -> tuple[str, ...]:
    expected_os, expected_architecture = provenance.platform.split('/', maxsplit=1)
    if config.get('os') != expected_os or config.get('architecture') != expected_architecture:
        raise SourceWorkerSupplyChainError('OCI config platform differs from provenance')
    image_config = config.get('config')
    if not isinstance(image_config, dict):
        raise SourceWorkerSupplyChainError('OCI config omits its runtime config object')
    allowed_runtime_fields = {'Cmd', 'Entrypoint', 'Env', 'Labels', 'User', 'Volumes', 'WorkingDir'}
    if set(image_config) - allowed_runtime_fields:
        raise SourceWorkerSupplyChainError('OCI runtime config contains unsupported ambient fields')
    if image_config.get('Entrypoint') != list(provenance.entrypoint):
        raise SourceWorkerSupplyChainError('OCI entrypoint differs from provenance')
    if image_config.get('User') != provenance.runtime_user:
        raise SourceWorkerSupplyChainError('OCI image must select the fixed non-root runtime user')
    if image_config.get('WorkingDir') not in (None, '/'):
        raise SourceWorkerSupplyChainError('OCI image working directory must be the filesystem root')
    if image_config.get('Cmd') not in (None, []):
        raise SourceWorkerSupplyChainError('OCI image cannot declare an implicit command')
    if image_config.get('Env') not in (None, []):
        raise SourceWorkerSupplyChainError('OCI image cannot declare ambient environment variables')
    if image_config.get('Volumes') not in (None, {}):
        raise SourceWorkerSupplyChainError('OCI image cannot declare volumes')
    if image_config.get('Labels') != provenance.labels:
        raise SourceWorkerSupplyChainError('OCI labels differ from the exact provenance labels')
    rootfs = config.get('rootfs')
    if not isinstance(rootfs, dict) or rootfs.get('type') != 'layers':
        raise SourceWorkerSupplyChainError('OCI config must declare a layered rootfs')
    diff_ids = rootfs.get('diff_ids')
    if (
        not isinstance(diff_ids, list)
        or not diff_ids
        or any(not isinstance(item, str) or re.fullmatch(_OCI_DIGEST_PATTERN, item) is None for item in diff_ids)
    ):
        raise SourceWorkerSupplyChainError('OCI config has an invalid rootfs diff-id inventory')
    if len(diff_ids) != expected_layer_count:
        raise SourceWorkerSupplyChainError('OCI config rootfs inventory differs from manifest layer count')
    return tuple(diff_ids)


def _verify_oci_layer_diff_ids(
    layer_descriptors: tuple[tuple[str, int, str], ...],
    layer_bytes: Mapping[str, bytes],
    expected_diff_ids: tuple[str, ...],
) -> None:
    for (compressed_digest, _size, media_type), expected_diff_id in zip(
        layer_descriptors,
        expected_diff_ids,
        strict=True,
    ):
        digest_key = compressed_digest.removeprefix('sha256:')
        actual_uncompressed_sha256 = _uncompressed_layer_sha256(
            layer_bytes[digest_key],
            media_type,
        )
        if expected_diff_id != f'sha256:{actual_uncompressed_sha256}':
            raise SourceWorkerSupplyChainError(
                'OCI config rootfs diff-id differs from the ordered uncompressed layer bytes'
            )


def _uncompressed_layer_sha256(payload: bytes, media_type: str) -> str:
    if media_type == _OCI_UNCOMPRESSED_LAYER_MEDIA_TYPE:
        if payload.startswith(_GZIP_MAGIC) or payload.startswith(_ZSTD_MAGIC):
            raise SourceWorkerSupplyChainError('uncompressed OCI layer mediaType contains a compressed payload')
        if len(payload) > _MAX_UNCOMPRESSED_LAYER_BYTES:
            raise SourceWorkerSupplyChainError('uncompressed OCI layer exceeds the verification limit')
        return hashlib.sha256(payload).hexdigest()
    if media_type in _OCI_GZIP_LAYER_MEDIA_TYPES:
        return _gzip_uncompressed_sha256(payload)
    raise SourceWorkerSupplyChainError('OCI layer descriptor has an unsupported mediaType')


def _gzip_uncompressed_sha256(payload: bytes) -> str:
    if not payload.startswith(_GZIP_MAGIC):
        raise SourceWorkerSupplyChainError('gzip OCI layer mediaType does not contain a gzip stream')
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    digest = hashlib.sha256()
    uncompressed_size = 0
    cursor = 0
    try:
        while cursor < len(payload):
            if decompressor.eof:
                raise SourceWorkerSupplyChainError('gzip OCI layer contains trailing data or another member')
            chunk = payload[cursor : cursor + _LAYER_IO_CHUNK_BYTES]
            cursor += len(chunk)
            pending = chunk
            while pending:
                output = decompressor.decompress(pending, _LAYER_IO_CHUNK_BYTES)
                uncompressed_size += len(output)
                if uncompressed_size > _MAX_UNCOMPRESSED_LAYER_BYTES:
                    raise SourceWorkerSupplyChainError('gzip OCI layer exceeds the uncompressed verification limit')
                digest.update(output)
                if decompressor.eof:
                    if decompressor.unused_data or cursor != len(payload):
                        raise SourceWorkerSupplyChainError('gzip OCI layer contains trailing data or another member')
                    pending = b''
                    break
                unconsumed = decompressor.unconsumed_tail
                if unconsumed and not output and len(unconsumed) >= len(pending):
                    raise SourceWorkerSupplyChainError('gzip OCI layer decompression made no progress')
                pending = unconsumed
    except zlib.error as error:
        raise SourceWorkerSupplyChainError('gzip OCI layer is malformed or has an invalid checksum') from error
    if not decompressor.eof:
        raise SourceWorkerSupplyChainError('gzip OCI layer is truncated')
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise SourceWorkerSupplyChainError('gzip OCI layer contains trailing data or another member')
    return digest.hexdigest()


def _require_material(binding: RetainedMaterial, payload: bytes, label: str) -> None:
    if not isinstance(payload, bytes):
        raise SourceWorkerSupplyChainError(f'{label} must be exact bytes')
    if len(payload) != binding.byte_count or hashlib.sha256(payload).hexdigest() != binding.sha256:
        raise SourceWorkerSupplyChainError(f'{label} differs from its retained material binding')


def _require_digest_key(value: str, label: str) -> None:
    if re.fullmatch(_SHA256_PATTERN, value) is None:
        raise SourceWorkerSupplyChainError(f'{label} mapping key must be lowercase SHA-256 hex')


def _parse_canonical_model[ModelT: StrictModel](payload: bytes, model: type[ModelT], label: str) -> ModelT:
    _parse_json(payload, label)
    try:
        parsed = model.model_validate_json(payload)
    except ValueError as error:
        raise SourceWorkerSupplyChainError(f'{label} does not match its strict schema') from error
    if payload != canonical_json_bytes(parsed):
        raise SourceWorkerSupplyChainError(f'{label} must use exact canonical JSON bytes')
    return parsed


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = _parse_json(payload, label)
    if not isinstance(value, dict):
        raise SourceWorkerSupplyChainError(f'{label} must be a JSON object')
    return value


def _parse_json(payload: bytes, label: str) -> Any:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_JSON_BYTES:
        raise SourceWorkerSupplyChainError(f'{label} must be nonempty and at most {_MAX_JSON_BYTES} bytes')
    if payload.startswith(b'\xef\xbb\xbf') or b'\x00' in payload:
        raise SourceWorkerSupplyChainError(f'{label} contains a forbidden encoding marker')

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise SourceWorkerSupplyChainError(f'{label} contains a duplicate JSON key')
            result[name] = value
        return result

    def reject_constant(_value: str) -> None:
        raise SourceWorkerSupplyChainError(f'{label} contains a non-finite number')

    try:
        return json.loads(
            payload.decode('utf-8', errors='strict'),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceWorkerSupplyChainError(f'{label} is not strict UTF-8 JSON') from error
