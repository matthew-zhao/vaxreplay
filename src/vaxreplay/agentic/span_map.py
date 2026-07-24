"""Exact-byte span maps for deterministic identity-masking transformations.

The span map is organizer-private.  It binds every declared public-output span to an exact byte
range inside a named field and row of a named member of a content-addressed source artifact.  The
generic verifier deliberately understands only regular files and ZIP members; semantic validation
of a source-specific table remains the transformation verifier's responsibility.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import io
import stat
import zipfile
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.schema import AgenticDerivationKind, AgenticTransformationReceipt, normalized_relative_path
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

AGENTIC_SPAN_MAP_SCHEMA_VERSION = 'vaxreplay.agentic-span-map.v0.2'
AGENTIC_NEUTRAL_ALIAS_POLICY_SCHEMA_VERSION = 'vaxreplay.agentic-neutral-alias-policy.v0.1'
AGENTIC_IDENTITY_MASK_TRANSFORM_ID = 'identity-mask'
AGENTIC_IDENTITY_MASK_TRANSFORM_VERSION = '1'
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_MAX_SPAN_MAP_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_ARTIFACTS = 100_000
_MAX_SOURCE_MEMBERS = 100_000
_MAX_MAPPINGS = 1_000_000
_MAX_MEMBER_BYTES = 256 * 1024 * 1024


class AgenticSpanMapError(ValueError):
    """Raised when a private span map or one of its exact-byte bindings is invalid."""


class AgenticSourceContainerKind(str, enum.Enum):
    FILE = 'file'
    ZIP = 'zip'


class AgenticSpanMappingKind(str, enum.Enum):
    COPIED = 'copied'
    MASKED_REPLACEMENT = 'masked_replacement'


class AgenticNeutralAliasNamespace(str, enum.Enum):
    """Closed vocabulary for aliases that cannot carry task outcomes or scientific claims."""

    ARM = 'arm'
    CANDIDATE = 'candidate'
    CONDITION = 'condition'
    DOCUMENT = 'document'
    ENDPOINT = 'endpoint'
    ENTITY = 'entity'
    GROUP = 'group'
    INTERVENTION = 'intervention'
    MEASURE = 'measure'
    ORGANIZATION = 'organization'
    PRODUCT = 'product'
    REGIMEN = 'regimen'
    SOURCE = 'source'
    SPONSOR = 'sponsor'
    STUDY = 'study'
    TRIAL = 'trial'


class AgenticNeutralAliasBinding(StrictModel):
    """Private one-to-one binding from an exact source identity to an opaque alias token."""

    source_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    namespace: AgenticNeutralAliasNamespace
    ordinal: int = Field(ge=1, le=999_999)
    alias_token: str = Field(pattern=r'^[a-z]+-[0-9]{3,6}$')

    @model_validator(mode='after')
    def validate_alias_token(self) -> Self:
        expected = f'{self.namespace.value}-{self.ordinal:03d}'
        if self.alias_token != expected:
            raise ValueError('neutral alias token must be the canonical namespace/ordinal rendering')
        return self


class AgenticNeutralAliasPolicy(StrictModel):
    """Exact private identity-to-alias configuration used by one identity-mask execution."""

    schema_version: Literal['vaxreplay.agentic-neutral-alias-policy.v0.1'] = AGENTIC_NEUTRAL_ALIAS_POLICY_SCHEMA_VERSION
    policy_id: str = Field(min_length=1, max_length=256)
    scheme: Literal['closed-namespace-dense-ordinal-v1'] = 'closed-namespace-dense-ordinal-v1'
    bindings: tuple[AgenticNeutralAliasBinding, ...] = Field(min_length=1, max_length=_MAX_MAPPINGS)

    @model_validator(mode='after')
    def validate_bindings(self) -> Self:
        keys = tuple(
            (binding.source_identity_sha256, binding.namespace.value, binding.ordinal) for binding in self.bindings
        )
        if keys != tuple(sorted(keys)):
            raise ValueError('neutral alias bindings must use canonical source-hash order')
        source_hashes = tuple(binding.source_identity_sha256 for binding in self.bindings)
        aliases = tuple(binding.alias_token for binding in self.bindings)
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError('one source identity cannot receive multiple neutral aliases')
        if len(aliases) != len(set(aliases)):
            raise ValueError('one neutral alias cannot represent multiple source identities')
        for namespace in AgenticNeutralAliasNamespace:
            namespace_bindings = sorted(
                (binding for binding in self.bindings if binding.namespace == namespace),
                key=lambda binding: binding.source_identity_sha256,
            )
            ordinals = [binding.ordinal for binding in namespace_bindings]
            if ordinals and ordinals != list(range(1, len(ordinals) + 1)):
                raise ValueError(
                    'neutral aliases must use dense ordinals assigned by source-identity hash within each namespace'
                )
        return self


class AgenticSpanMapSourceArtifact(StrictModel):
    source_artifact_id: str = Field(min_length=1, max_length=256)
    container_kind: AgenticSourceContainerKind
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)


class AgenticSpanMapSourceMember(StrictModel):
    source_member_id: str = Field(min_length=1, max_length=256)
    source_artifact_id: str = Field(min_length=1, max_length=256)
    member_path: str = Field(min_length=1, max_length=4_096)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0, le=_MAX_MEMBER_BYTES)

    @field_validator('member_path')
    @classmethod
    def validate_member_path(cls, value: str) -> str:
        return normalized_relative_path(value)


class AgenticIdentityMaskSpan(StrictModel):
    """One unambiguous public-output to private-source byte mapping."""

    mapping_id: str = Field(min_length=1, max_length=256)
    kind: AgenticSpanMappingKind
    output_start_byte: int = Field(ge=0)
    output_end_byte: int = Field(gt=0)
    output_span_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_member_id: str = Field(min_length=1, max_length=256)
    source_row_id: str = Field(min_length=1, max_length=4_096)
    source_row_start_byte: int = Field(ge=0)
    source_row_end_byte: int = Field(gt=0)
    source_row_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_row_id_start_byte: int = Field(ge=0)
    source_row_id_end_byte: int = Field(gt=0)
    source_field_name: str = Field(min_length=1, max_length=4_096)
    source_field_start_byte: int = Field(ge=0)
    source_field_end_byte: int = Field(gt=0)
    source_field_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_start_byte: int = Field(ge=0)
    source_end_byte: int = Field(gt=0)
    source_span_sha256: str = Field(pattern=_SHA256_PATTERN)
    neutral_alias_token: str | None = Field(default=None, pattern=r'^[a-z]+-[0-9]{3,6}$')

    @model_validator(mode='after')
    def validate_nested_spans(self) -> Self:
        if self.output_end_byte <= self.output_start_byte:
            raise ValueError('output span must be non-empty and ordered')
        if self.source_row_end_byte <= self.source_row_start_byte:
            raise ValueError('source row span must be non-empty and ordered')
        if not (
            self.source_row_start_byte
            <= self.source_row_id_start_byte
            < self.source_row_id_end_byte
            <= self.source_row_end_byte
        ):
            raise ValueError('source row-ID span must be non-empty and contained in its row')
        if not (
            self.source_row_start_byte
            <= self.source_field_start_byte
            < self.source_field_end_byte
            <= self.source_row_end_byte
        ):
            raise ValueError('source field span must be non-empty and contained in its row')
        if not (
            self.source_field_start_byte <= self.source_start_byte < self.source_end_byte <= self.source_field_end_byte
        ):
            raise ValueError('source span must be non-empty and contained in its field')
        if self.kind == AgenticSpanMappingKind.COPIED:
            if self.neutral_alias_token is not None:
                raise ValueError('copied spans cannot declare a neutral alias')
        else:
            if self.neutral_alias_token is None:
                raise ValueError('masked replacements require an explicit neutral alias token')
            if (self.source_start_byte, self.source_end_byte) != (
                self.source_field_start_byte,
                self.source_field_end_byte,
            ):
                raise ValueError('masked replacements must bind the complete source identity field')
        return self


class AgenticIdentityMaskSpanMap(StrictModel):
    schema_version: Literal['vaxreplay.agentic-span-map.v0.2'] = AGENTIC_SPAN_MAP_SCHEMA_VERSION
    span_map_id: str = Field(min_length=1, max_length=256)
    transformation_receipt_id: str = Field(min_length=1, max_length=256)
    output_source_id: str = Field(min_length=1, max_length=256)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_bytes: int = Field(gt=0)
    complete_output_coverage: bool
    neutral_alias_policy: AgenticNeutralAliasPolicy | None = None
    source_artifacts: tuple[AgenticSpanMapSourceArtifact, ...] = Field(
        min_length=1,
        max_length=_MAX_SOURCE_ARTIFACTS,
    )
    source_members: tuple[AgenticSpanMapSourceMember, ...] = Field(
        min_length=1,
        max_length=_MAX_SOURCE_MEMBERS,
    )
    mappings: tuple[AgenticIdentityMaskSpan, ...] = Field(min_length=1, max_length=_MAX_MAPPINGS)

    @model_validator(mode='after')
    def validate_inventory_and_coverage(self) -> Self:
        artifact_ids = tuple(binding.source_artifact_id for binding in self.source_artifacts)
        if artifact_ids != tuple(sorted(artifact_ids)) or len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('source artifacts must use unique IDs in sorted order')

        member_keys = tuple(
            (binding.source_artifact_id, binding.member_path, binding.source_member_id)
            for binding in self.source_members
        )
        if member_keys != tuple(sorted(member_keys)):
            raise ValueError('source members must use canonical artifact/path/ID order')
        member_ids = tuple(binding.source_member_id for binding in self.source_members)
        if len(member_ids) != len(set(member_ids)):
            raise ValueError('source members must use unique IDs')
        if unknown := {binding.source_artifact_id for binding in self.source_members} - set(artifact_ids):
            raise ValueError(f'source members reference unknown artifacts: {sorted(unknown)}')

        mapping_order = tuple(
            (mapping.output_start_byte, mapping.output_end_byte, mapping.mapping_id) for mapping in self.mappings
        )
        if mapping_order != tuple(sorted(mapping_order)):
            raise ValueError('span mappings must use canonical output-span order')
        mapping_ids = tuple(mapping.mapping_id for mapping in self.mappings)
        if len(mapping_ids) != len(set(mapping_ids)):
            raise ValueError('span mappings must use unique IDs')
        if unknown := {mapping.source_member_id for mapping in self.mappings} - set(member_ids):
            raise ValueError(f'span mappings reference unknown source members: {sorted(unknown)}')

        masked_mappings = tuple(
            mapping for mapping in self.mappings if mapping.kind == AgenticSpanMappingKind.MASKED_REPLACEMENT
        )
        if bool(masked_mappings) != (self.neutral_alias_policy is not None):
            raise ValueError(
                'masked replacements require one exact neutral-alias policy and copied-only maps forbid one'
            )
        if self.neutral_alias_policy is not None:
            binding_by_source_hash = {
                binding.source_identity_sha256: binding for binding in self.neutral_alias_policy.bindings
            }
            used_source_hashes = {mapping.source_span_sha256 for mapping in masked_mappings}
            if used_source_hashes != set(binding_by_source_hash):
                raise ValueError('neutral-alias policy must cover every and only masked source identity')
            for mapping in masked_mappings:
                binding = binding_by_source_hash[mapping.source_span_sha256]
                if mapping.neutral_alias_token != binding.alias_token:
                    raise ValueError('masked replacement token does not match its neutral-alias policy binding')

        member_by_id = {binding.source_member_id: binding for binding in self.source_members}
        previous_end = 0
        for index, mapping in enumerate(self.mappings):
            if mapping.output_end_byte > self.output_bytes:
                raise ValueError('span mapping falls outside the declared output artifact')
            if index and mapping.output_start_byte < previous_end:
                raise ValueError('span mappings overlap and make output provenance ambiguous')
            if self.complete_output_coverage and mapping.output_start_byte != previous_end:
                raise ValueError('complete span map has an unmapped output gap')
            member = member_by_id[mapping.source_member_id]
            if mapping.source_row_end_byte > member.byte_count:
                raise ValueError('span mapping falls outside the declared source member')
            previous_end = mapping.output_end_byte
        if self.complete_output_coverage and previous_end != self.output_bytes:
            raise ValueError('complete span map does not cover the output suffix')

        used_members = {mapping.source_member_id for mapping in self.mappings}
        if used_members != set(member_ids):
            raise ValueError('source-member inventory must contain every and only mapped members')
        used_artifacts = {member_by_id[member_id].source_artifact_id for member_id in used_members}
        if used_artifacts != set(artifact_ids):
            raise ValueError('source-artifact inventory must contain every and only mapped artifacts')

        row_locations: dict[tuple[str, str], tuple[int, int, int, int, str]] = {}
        field_locations: dict[tuple[str, int, int, str], tuple[int, int, str]] = {}
        for mapping in self.mappings:
            row_key = (mapping.source_member_id, mapping.source_row_id)
            row_value = (
                mapping.source_row_start_byte,
                mapping.source_row_end_byte,
                mapping.source_row_id_start_byte,
                mapping.source_row_id_end_byte,
                mapping.source_row_sha256,
            )
            if row_key in row_locations and row_locations[row_key] != row_value:
                raise ValueError('one source row ID is ambiguously bound to multiple row spans')
            row_locations[row_key] = row_value

            field_key = (
                mapping.source_member_id,
                mapping.source_row_start_byte,
                mapping.source_row_end_byte,
                mapping.source_field_name,
            )
            field_value = (
                mapping.source_field_start_byte,
                mapping.source_field_end_byte,
                mapping.source_field_sha256,
            )
            if field_key in field_locations and field_locations[field_key] != field_value:
                raise ValueError('one source field is ambiguously bound to multiple field spans')
            field_locations[field_key] = field_value
        return self


class VerifiedAgenticIdentityMaskSpanMap(StrictModel):
    span_map_id: str = Field(min_length=1)
    transformation_receipt_id: str = Field(min_length=1)
    output_source_id: str = Field(min_length=1)
    span_map_sha256: str = Field(pattern=_SHA256_PATTERN)
    span_map_bytes: int = Field(gt=0)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_bytes: int = Field(gt=0)
    source_artifact_count: int = Field(gt=0)
    source_member_count: int = Field(gt=0)
    mapping_count: int = Field(gt=0)
    complete_output_coverage: bool
    neutral_alias_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    neutral_alias_binding_count: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_alias_verification(self) -> Self:
        if (self.neutral_alias_policy_sha256 is None) != (self.neutral_alias_binding_count == 0):
            raise ValueError('verified neutral-alias policy hash and binding count must be present together')
        return self


def verify_agentic_identity_mask_span_map(
    span_map_artifact_bytes: bytes,
    *,
    receipt: AgenticTransformationReceipt,
    output_bytes: bytes,
    source_artifacts: Mapping[str, bytes],
) -> VerifiedAgenticIdentityMaskSpanMap:
    """Authenticate and fully validate one deterministic identity-mask span map.

    ``source_artifacts`` is an exact inventory keyed by the receipt's input source IDs.  A ``file``
    artifact is its own sole member.  A ``zip`` artifact is parsed here, and every selected member
    must occur exactly once as a non-directory, non-encrypted, non-symlink entry.
    """

    if not isinstance(span_map_artifact_bytes, bytes):
        raise AgenticSpanMapError('span-map artifact must be bytes')
    if not span_map_artifact_bytes or len(span_map_artifact_bytes) > _MAX_SPAN_MAP_BYTES:
        raise AgenticSpanMapError('span-map artifact has an invalid byte count')
    if receipt.span_map_sha256 is None or receipt.span_map_bytes is None:
        raise AgenticSpanMapError('transformation receipt does not bind a span-map artifact')
    if (
        receipt.kind != AgenticDerivationKind.DETERMINISTIC
        or receipt.transform_id != AGENTIC_IDENTITY_MASK_TRANSFORM_ID
        or receipt.transform_version != AGENTIC_IDENTITY_MASK_TRANSFORM_VERSION
        or receipt.semantic_rewrite
        or receipt.network_allowed
        or receipt.outcome_namespace_mounted
        or not receipt.label_blind
    ):
        raise AgenticSpanMapError('receipt is not an admissible deterministic identity-mask execution')
    if (len(span_map_artifact_bytes), _sha256(span_map_artifact_bytes)) != (
        receipt.span_map_bytes,
        receipt.span_map_sha256,
    ):
        raise AgenticSpanMapError('span-map artifact does not match its transformation receipt')

    try:
        span_map = AgenticIdentityMaskSpanMap.model_validate_json(span_map_artifact_bytes)
    except ValueError as error:
        raise AgenticSpanMapError(f'invalid span-map artifact: {error}') from error
    if canonical_json_bytes(span_map) != span_map_artifact_bytes:
        raise AgenticSpanMapError('span-map artifact must use canonical JSON encoding')
    if (
        span_map.transformation_receipt_id,
        span_map.output_source_id,
        span_map.output_sha256,
        span_map.output_bytes,
        span_map.complete_output_coverage,
    ) != (
        receipt.receipt_id,
        receipt.output_source_id,
        receipt.output_sha256,
        receipt.output_bytes,
        receipt.source_span_mapping_complete,
    ):
        raise AgenticSpanMapError('span-map artifact does not match its transformation receipt claims')
    if span_map.neutral_alias_policy is not None:
        policy_bytes = canonical_json_bytes(span_map.neutral_alias_policy)
        if not hmac.compare_digest(_sha256(policy_bytes), receipt.config_sha256):
            raise AgenticSpanMapError('neutral-alias policy does not match the receipt-bound transform config')

    if not isinstance(output_bytes, bytes):
        raise AgenticSpanMapError('public output artifact must be bytes')
    _require_exact_binding(output_bytes, span_map.output_sha256, span_map.output_bytes, 'public output artifact')
    output_boundaries = _utf8_boundaries(output_bytes, 'public output artifact')

    if not isinstance(source_artifacts, Mapping):
        raise AgenticSpanMapError('source artifacts must be a mapping')
    expected_artifact_ids = tuple(binding.source_artifact_id for binding in span_map.source_artifacts)
    if set(source_artifacts) != set(expected_artifact_ids) or set(source_artifacts) != set(receipt.input_source_ids):
        raise AgenticSpanMapError('source artifacts must cover every and only transformation inputs')
    artifact_by_id = {binding.source_artifact_id: binding for binding in span_map.source_artifacts}
    verified_artifact_bytes: dict[str, bytes] = {}
    for artifact_id in expected_artifact_ids:
        payload = source_artifacts[artifact_id]
        if not isinstance(payload, bytes):
            raise AgenticSpanMapError(f'source artifact {artifact_id} must be bytes')
        binding = artifact_by_id[artifact_id]
        _require_exact_binding(payload, binding.sha256, binding.byte_count, f'source artifact {artifact_id}')
        verified_artifact_bytes[artifact_id] = payload

    members_by_artifact: dict[str, list[AgenticSpanMapSourceMember]] = {}
    for member in span_map.source_members:
        members_by_artifact.setdefault(member.source_artifact_id, []).append(member)
    verified_members: dict[str, bytes] = {}
    for artifact_id, bindings in members_by_artifact.items():
        artifact_binding = artifact_by_id[artifact_id]
        artifact_bytes = verified_artifact_bytes[artifact_id]
        if artifact_binding.container_kind == AgenticSourceContainerKind.FILE:
            if len(bindings) != 1:
                raise AgenticSpanMapError('regular-file source artifacts must bind exactly one member')
            member = bindings[0]
            _require_exact_binding(
                artifact_bytes, member.sha256, member.byte_count, f'source member {member.member_path}'
            )
            verified_members[member.source_member_id] = artifact_bytes
        else:
            verified_members.update(_verify_zip_members(artifact_id, artifact_bytes, bindings))

    member_boundaries = {
        member_id: _utf8_boundaries(payload, f'source member {member_id}')
        for member_id, payload in verified_members.items()
    }
    for mapping in span_map.mappings:
        source = verified_members[mapping.source_member_id]
        boundaries = member_boundaries[mapping.source_member_id]
        _require_boundaries(
            boundaries,
            (
                mapping.source_row_start_byte,
                mapping.source_row_end_byte,
                mapping.source_row_id_start_byte,
                mapping.source_row_id_end_byte,
                mapping.source_field_start_byte,
                mapping.source_field_end_byte,
                mapping.source_start_byte,
                mapping.source_end_byte,
            ),
            f'source mapping {mapping.mapping_id}',
        )
        _require_boundaries(
            output_boundaries,
            (mapping.output_start_byte, mapping.output_end_byte),
            f'output mapping {mapping.mapping_id}',
        )
        row = source[mapping.source_row_start_byte : mapping.source_row_end_byte]
        row_id = source[mapping.source_row_id_start_byte : mapping.source_row_id_end_byte]
        field = source[mapping.source_field_start_byte : mapping.source_field_end_byte]
        source_span = source[mapping.source_start_byte : mapping.source_end_byte]
        output_span = output_bytes[mapping.output_start_byte : mapping.output_end_byte]
        _require_span_hash(row, mapping.source_row_sha256, f'source row for {mapping.mapping_id}')
        _require_span_hash(field, mapping.source_field_sha256, f'source field for {mapping.mapping_id}')
        _require_span_hash(source_span, mapping.source_span_sha256, f'source span for {mapping.mapping_id}')
        _require_span_hash(output_span, mapping.output_span_sha256, f'output span for {mapping.mapping_id}')
        if row_id.decode('utf-8') != mapping.source_row_id:
            raise AgenticSpanMapError(f'source row ID bytes do not match mapping {mapping.mapping_id}')
        if mapping.kind == AgenticSpanMappingKind.COPIED and output_span != source_span:
            raise AgenticSpanMapError(f'copied mapping {mapping.mapping_id} changes source bytes')
        if mapping.kind == AgenticSpanMappingKind.MASKED_REPLACEMENT:
            alias_token = mapping.neutral_alias_token
            if alias_token is None:
                raise AgenticSpanMapError(f'masked replacement {mapping.mapping_id} has no neutral alias')
            if output_span != alias_token.encode('ascii'):
                raise AgenticSpanMapError(
                    f'masked replacement {mapping.mapping_id} is not its receipt-bound neutral alias'
                )
            if output_span == source_span:
                raise AgenticSpanMapError(f'masked replacement {mapping.mapping_id} does not replace source bytes')

    neutral_alias_policy_sha256 = (
        _sha256(canonical_json_bytes(span_map.neutral_alias_policy))
        if span_map.neutral_alias_policy is not None
        else None
    )
    neutral_alias_binding_count = (
        len(span_map.neutral_alias_policy.bindings) if span_map.neutral_alias_policy is not None else 0
    )

    return VerifiedAgenticIdentityMaskSpanMap(
        span_map_id=span_map.span_map_id,
        transformation_receipt_id=span_map.transformation_receipt_id,
        output_source_id=span_map.output_source_id,
        span_map_sha256=receipt.span_map_sha256,
        span_map_bytes=receipt.span_map_bytes,
        output_sha256=span_map.output_sha256,
        output_bytes=span_map.output_bytes,
        source_artifact_count=len(span_map.source_artifacts),
        source_member_count=len(span_map.source_members),
        mapping_count=len(span_map.mappings),
        complete_output_coverage=span_map.complete_output_coverage,
        neutral_alias_policy_sha256=neutral_alias_policy_sha256,
        neutral_alias_binding_count=neutral_alias_binding_count,
    )


def _verify_zip_members(
    artifact_id: str,
    artifact_bytes: bytes,
    bindings: list[AgenticSpanMapSourceMember],
) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(artifact_bytes)) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            verified: dict[str, bytes] = {}
            for binding in bindings:
                matches = infos_by_name.get(binding.member_path, [])
                if len(matches) != 1:
                    raise AgenticSpanMapError(
                        f'ZIP source artifact {artifact_id} must contain exactly one {binding.member_path}'
                    )
                info = matches[0]
                unix_mode = info.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or (file_type and file_type != stat.S_IFREG)
                    or info.file_size != binding.byte_count
                    or info.file_size > _MAX_MEMBER_BYTES
                ):
                    raise AgenticSpanMapError(
                        f'ZIP source member {artifact_id}/{binding.member_path} is not an admissible regular file'
                    )
                payload = archive.read(info)
                _require_exact_binding(
                    payload,
                    binding.sha256,
                    binding.byte_count,
                    f'source member {artifact_id}/{binding.member_path}',
                )
                verified[binding.source_member_id] = payload
            return verified
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as error:
        raise AgenticSpanMapError(f'cannot verify ZIP source artifact {artifact_id}: {error}') from error


def _utf8_boundaries(payload: bytes, label: str) -> frozenset[int]:
    try:
        text = payload.decode('utf-8')
    except UnicodeDecodeError as error:
        raise AgenticSpanMapError(f'{label} is not valid UTF-8') from error
    boundaries = {0}
    offset = 0
    for character in text:
        offset += len(character.encode('utf-8'))
        boundaries.add(offset)
    return frozenset(boundaries)


def _require_boundaries(boundaries: frozenset[int], offsets: tuple[int, ...], label: str) -> None:
    if any(offset not in boundaries for offset in offsets):
        raise AgenticSpanMapError(f'{label} splits a UTF-8 code point or falls outside its artifact')


def _require_exact_binding(payload: bytes, expected_sha256: str, expected_bytes: int, label: str) -> None:
    if len(payload) != expected_bytes or not hmac.compare_digest(_sha256(payload), expected_sha256):
        raise AgenticSpanMapError(f'{label} does not match its exact-byte binding')


def _require_span_hash(payload: bytes, expected_sha256: str, label: str) -> None:
    if not hmac.compare_digest(_sha256(payload), expected_sha256):
        raise AgenticSpanMapError(f'{label} does not match its span hash')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
