from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vaxreplay.agentic.schema import AgenticDerivationKind, AgenticTransformationReceipt
from vaxreplay.agentic.span_map import (
    AgenticIdentityMaskSpan,
    AgenticIdentityMaskSpanMap,
    AgenticNeutralAliasBinding,
    AgenticNeutralAliasNamespace,
    AgenticNeutralAliasPolicy,
    AgenticSourceContainerKind,
    AgenticSpanMapError,
    AgenticSpanMappingKind,
    AgenticSpanMapSourceArtifact,
    AgenticSpanMapSourceMember,
    verify_agentic_identity_mask_span_map,
)
from vaxreplay.bundle import canonical_json_bytes


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip(member: bytes, *, duplicate: bool = False) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('records.txt', member)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr('records.txt', member)
    return target.getvalue()


def _mapping(
    *,
    mapping_id: str,
    kind: AgenticSpanMappingKind,
    output: bytes,
    output_start: int,
    output_end: int,
    source: bytes,
    source_start: int,
    source_end: int,
    field_name: str,
    field_start: int,
    field_end: int,
    neutral_alias_token: str | None = None,
) -> AgenticIdentityMaskSpan:
    return AgenticIdentityMaskSpan(
        mapping_id=mapping_id,
        kind=kind,
        output_start_byte=output_start,
        output_end_byte=output_end,
        output_span_sha256=_sha256(output[output_start:output_end]),
        source_member_id='member-001',
        source_row_id='row-7',
        source_row_start_byte=0,
        source_row_end_byte=len(source),
        source_row_sha256=_sha256(source),
        source_row_id_start_byte=0,
        source_row_id_end_byte=5,
        source_field_name=field_name,
        source_field_start_byte=field_start,
        source_field_end_byte=field_end,
        source_field_sha256=_sha256(source[field_start:field_end]),
        source_start_byte=source_start,
        source_end_byte=source_end,
        source_span_sha256=_sha256(source[source_start:source_end]),
        neutral_alias_token=neutral_alias_token,
    )


def _case(
    *,
    container_kind: AgenticSourceContainerKind = AgenticSourceContainerKind.FILE,
    duplicate_zip_member: bool = False,
) -> tuple[
    AgenticIdentityMaskSpanMap,
    bytes,
    AgenticTransformationReceipt,
    bytes,
    dict[str, bytes],
]:
    source = b'row-7|Product-X|120\n'
    output = b'row-7|candidate-001|120\n'
    artifact = (
        source if container_kind == AgenticSourceContainerKind.FILE else _zip(source, duplicate=duplicate_zip_member)
    )
    alias_policy = AgenticNeutralAliasPolicy(
        policy_id='mask-policy-001',
        bindings=(
            AgenticNeutralAliasBinding(
                source_identity_sha256=_sha256(source[6:15]),
                namespace=AgenticNeutralAliasNamespace.CANDIDATE,
                ordinal=1,
                alias_token='candidate-001',
            ),
        ),
    )
    span_map = AgenticIdentityMaskSpanMap(
        span_map_id='mask-map-001',
        transformation_receipt_id='transform-001',
        output_source_id='source-002',
        output_sha256=_sha256(output),
        output_bytes=len(output),
        complete_output_coverage=True,
        neutral_alias_policy=alias_policy,
        source_artifacts=(
            AgenticSpanMapSourceArtifact(
                source_artifact_id='source-001',
                container_kind=container_kind,
                sha256=_sha256(artifact),
                byte_count=len(artifact),
            ),
        ),
        source_members=(
            AgenticSpanMapSourceMember(
                source_member_id='member-001',
                source_artifact_id='source-001',
                member_path='records.txt',
                sha256=_sha256(source),
                byte_count=len(source),
            ),
        ),
        mappings=(
            _mapping(
                mapping_id='map-001',
                kind=AgenticSpanMappingKind.COPIED,
                output=output,
                output_start=0,
                output_end=6,
                source=source,
                source_start=0,
                source_end=6,
                field_name='row-prefix',
                field_start=0,
                field_end=6,
            ),
            _mapping(
                mapping_id='map-002',
                kind=AgenticSpanMappingKind.MASKED_REPLACEMENT,
                output=output,
                output_start=6,
                output_end=19,
                source=source,
                source_start=6,
                source_end=15,
                field_name='product',
                field_start=6,
                field_end=15,
                neutral_alias_token='candidate-001',
            ),
            _mapping(
                mapping_id='map-003',
                kind=AgenticSpanMappingKind.COPIED,
                output=output,
                output_start=19,
                output_end=24,
                source=source,
                source_start=15,
                source_end=20,
                field_name='dose-suffix',
                field_start=15,
                field_end=20,
            ),
        ),
    )
    span_map_bytes = canonical_json_bytes(span_map)
    receipt = AgenticTransformationReceipt(
        receipt_id='transform-001',
        kind=AgenticDerivationKind.DETERMINISTIC,
        input_source_ids=('source-001',),
        output_source_id='source-002',
        output_sha256=_sha256(output),
        output_bytes=len(output),
        transform_id='identity-mask',
        transform_version='1',
        executable_sha256='1' * 64,
        config_sha256=_sha256(canonical_json_bytes(alias_policy)),
        execution_receipt_sha256='3' * 64,
        execution_receipt_bytes=10,
        executed_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_rewrite=False,
        source_span_mapping_complete=True,
        span_map_sha256=_sha256(span_map_bytes),
        span_map_bytes=len(span_map_bytes),
    )
    return span_map, span_map_bytes, receipt, output, {'source-001': artifact}


def _receipt_for_map(receipt: AgenticTransformationReceipt, span_map_bytes: bytes) -> AgenticTransformationReceipt:
    return receipt.model_copy(
        update={
            'span_map_sha256': _sha256(span_map_bytes),
            'span_map_bytes': len(span_map_bytes),
        }
    )


def test_verifies_complete_identity_mask_map_for_file_and_zip_sources() -> None:
    for container_kind in (AgenticSourceContainerKind.FILE, AgenticSourceContainerKind.ZIP):
        _, span_map_bytes, receipt, output, sources = _case(container_kind=container_kind)
        verified = verify_agentic_identity_mask_span_map(
            span_map_bytes,
            receipt=receipt,
            output_bytes=output,
            source_artifacts=sources,
        )
        assert verified.span_map_sha256 == receipt.span_map_sha256
        assert verified.mapping_count == 3
        assert verified.source_artifact_count == 1
        assert verified.source_member_count == 1
        assert verified.complete_output_coverage is True
        assert verified.neutral_alias_policy_sha256 == receipt.config_sha256
        assert verified.neutral_alias_binding_count == 1


def test_rejects_arbitrary_future_content_disguised_as_a_masked_replacement() -> None:
    span_map, _, receipt, _, sources = _case()
    future_output = b'row-7|future-success|120\n'
    value = span_map.model_dump(mode='json')
    value.update(
        {
            'output_sha256': _sha256(future_output),
            'output_bytes': len(future_output),
        }
    )
    value['mappings'][1].update(
        {
            'output_end_byte': 20,
            'output_span_sha256': _sha256(b'future-success'),
        }
    )
    value['mappings'][2].update(
        {
            'output_start_byte': 20,
            'output_end_byte': len(future_output),
            'output_span_sha256': _sha256(b'|120\n'),
        }
    )
    adversarial_map_bytes = canonical_json_bytes(value)
    adversarial_receipt = receipt.model_copy(
        update={
            'output_sha256': _sha256(future_output),
            'output_bytes': len(future_output),
            'span_map_sha256': _sha256(adversarial_map_bytes),
            'span_map_bytes': len(adversarial_map_bytes),
        }
    )

    with pytest.raises(AgenticSpanMapError, match='receipt-bound neutral alias'):
        verify_agentic_identity_mask_span_map(
            adversarial_map_bytes,
            receipt=adversarial_receipt,
            output_bytes=future_output,
            source_artifacts=sources,
        )


def test_rejects_alias_policy_not_bound_as_the_transform_config() -> None:
    _, span_map_bytes, receipt, output, sources = _case()
    with pytest.raises(AgenticSpanMapError, match='receipt-bound transform config'):
        verify_agentic_identity_mask_span_map(
            span_map_bytes,
            receipt=receipt.model_copy(update={'config_sha256': 'f' * 64}),
            output_bytes=output,
            source_artifacts=sources,
        )


def test_neutral_alias_policy_rejects_outcome_namespaces() -> None:
    with pytest.raises(ValidationError, match='Input should be'):
        AgenticNeutralAliasBinding.model_validate(
            {
                'source_identity_sha256': 'a' * 64,
                'namespace': 'outcome',
                'ordinal': 1,
                'alias_token': 'outcome-001',
            }
        )


def test_neutral_alias_policy_rejects_outcome_conditioned_alias_permutations() -> None:
    with pytest.raises(ValidationError, match='assigned by source-identity hash'):
        AgenticNeutralAliasPolicy(
            policy_id='outcome-conditioned-policy',
            bindings=(
                AgenticNeutralAliasBinding(
                    source_identity_sha256='a' * 64,
                    namespace=AgenticNeutralAliasNamespace.CANDIDATE,
                    ordinal=2,
                    alias_token='candidate-002',
                ),
                AgenticNeutralAliasBinding(
                    source_identity_sha256='b' * 64,
                    namespace=AgenticNeutralAliasNamespace.CANDIDATE,
                    ordinal=1,
                    alias_token='candidate-001',
                ),
            ),
        )


def test_complete_receipt_requires_exact_span_map_binding() -> None:
    _, _, receipt, _, _ = _case()
    with pytest.raises(ValidationError, match='exact private span-map binding'):
        AgenticTransformationReceipt.model_validate(
            {
                **receipt.model_dump(),
                'span_map_sha256': None,
                'span_map_bytes': None,
            }
        )


def test_rejects_overlapping_ambiguous_output_mappings() -> None:
    span_map, _, receipt, output, sources = _case()
    value = span_map.model_dump(mode='json')
    value['mappings'][1]['output_start_byte'] = 5
    malformed = canonical_json_bytes(value)
    with pytest.raises(AgenticSpanMapError, match='overlap.*ambiguous'):
        verify_agentic_identity_mask_span_map(
            malformed,
            receipt=_receipt_for_map(receipt, malformed),
            output_bytes=output,
            source_artifacts=sources,
        )


def test_rejects_complete_map_with_output_gap() -> None:
    span_map, _, receipt, output, sources = _case()
    value = span_map.model_dump(mode='json')
    value['mappings'][1]['output_start_byte'] = 7
    malformed = canonical_json_bytes(value)
    with pytest.raises(AgenticSpanMapError, match='unmapped output gap'):
        verify_agentic_identity_mask_span_map(
            malformed,
            receipt=_receipt_for_map(receipt, malformed),
            output_bytes=output,
            source_artifacts=sources,
        )


def test_rejects_source_mapping_outside_declared_member() -> None:
    span_map, _, receipt, output, sources = _case()
    value = span_map.model_dump(mode='json')
    for mapping in value['mappings']:
        mapping['source_row_end_byte'] = 21
    malformed = canonical_json_bytes(value)
    with pytest.raises(AgenticSpanMapError, match='outside the declared source member'):
        verify_agentic_identity_mask_span_map(
            malformed,
            receipt=_receipt_for_map(receipt, malformed),
            output_bytes=output,
            source_artifacts=sources,
        )


def test_rejects_ambiguous_row_and_field_bindings() -> None:
    span_map, _, receipt, output, sources = _case()
    value = span_map.model_dump(mode='json')
    value['mappings'][1].update(
        {
            'source_row_start_byte': 1,
            'source_row_id_start_byte': 1,
            'source_row_id_end_byte': 5,
        }
    )
    malformed = canonical_json_bytes(value)
    with pytest.raises(AgenticSpanMapError, match='row ID is ambiguously bound'):
        verify_agentic_identity_mask_span_map(
            malformed,
            receipt=_receipt_for_map(receipt, malformed),
            output_bytes=output,
            source_artifacts=sources,
        )


def test_rejects_mutated_exact_source_artifact() -> None:
    _, span_map_bytes, receipt, output, sources = _case()
    mutated = bytearray(sources['source-001'])
    mutated[-1] ^= 1
    with pytest.raises(AgenticSpanMapError, match='source artifact source-001.*exact-byte binding'):
        verify_agentic_identity_mask_span_map(
            span_map_bytes,
            receipt=receipt,
            output_bytes=output,
            source_artifacts={'source-001': bytes(mutated)},
        )


def test_rejects_duplicate_zip_member_as_ambiguous() -> None:
    _, span_map_bytes, receipt, output, sources = _case(
        container_kind=AgenticSourceContainerKind.ZIP,
        duplicate_zip_member=True,
    )
    with pytest.raises(AgenticSpanMapError, match='exactly one records.txt'):
        verify_agentic_identity_mask_span_map(
            span_map_bytes,
            receipt=receipt,
            output_bytes=output,
            source_artifacts=sources,
        )


def test_rejects_noncanonical_map_even_when_receipt_hash_matches() -> None:
    span_map, _, receipt, output, sources = _case()
    pretty = json.dumps(span_map.model_dump(mode='json'), indent=2).encode('utf-8')
    with pytest.raises(AgenticSpanMapError, match='canonical JSON'):
        verify_agentic_identity_mask_span_map(
            pretty,
            receipt=_receipt_for_map(receipt, pretty),
            output_bytes=output,
            source_artifacts=sources,
        )
