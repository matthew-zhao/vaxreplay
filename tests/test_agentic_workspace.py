from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.agentic_helpers import bind_episode_manifest, selection_contract
from vaxreplay.agentic.schema import (
    AgenticArtifactKind,
    AgenticAssuranceProfile,
    AgenticDerivationKind,
    AgenticFactQuery,
    AgenticMediaType,
    AgenticTaskEnvelope,
    AgenticTransformationReceipt,
    AgenticValueType,
    AgenticWorkspaceSource,
    ArtifactTemporalProof,
    AvailabilityInterval,
    TemporalProofKind,
    agentic_model_sha256,
)
from vaxreplay.agentic.workspace import (
    AgenticWorkspaceError,
    build_agentic_workspace,
    load_agentic_workspace,
    parse_model_visible_surface_bytes,
)
from vaxreplay.bundle import canonical_json_bytes


def _task() -> AgenticTaskEnvelope:
    return AgenticTaskEnvelope(
        task_id='agentic-task-1',
        episode_id='episode-1',
        episode_manifest_sha256='e' * 64,
        decision_at=datetime(2020, 2, 1, 23, 59, 59, tzinfo=UTC),
        task_type='early_clinical_arm_prioritization',
        candidate_ids=('candidate-001', 'candidate-002'),
        portfolio_size=1,
        instructions='Inspect the frozen sources, extract the requested fact, and rank both candidates.',
        fact_queries=(
            AgenticFactQuery(
                query_id='dose-a',
                description='Dose assigned to candidate A',
                value_type=AgenticValueType.NUMBER,
                unit='microgram',
                candidate_id='candidate-001',
            ),
        ),
        historically_preregistered=False,
    )


def _raw_source(
    content: bytes,
    *,
    source_id: str = 'source-001',
    path: str = 'sources/source-001.txt',
    media_type: AgenticMediaType = AgenticMediaType.TEXT,
) -> AgenticWorkspaceSource:
    digest = hashlib.sha256(content).hexdigest()
    witnessed = AvailabilityInterval(
        lower_at=datetime(2020, 2, 1, tzinfo=UTC),
        upper_at=datetime(2020, 2, 1, 23, 59, 59, tzinfo=UTC),
        precision='day',
        timezone_basis='UTC upper-bound convention',
    )
    proof = ArtifactTemporalProof(
        proof_id=f'proof-{source_id}',
        kind=TemporalProofKind.SOURCE_ATTESTED_SNAPSHOT,
        artifact_sha256=digest,
        artifact_bytes=len(content),
        witnessed=witnessed,
        authority_id='source-archive-label',
        proof_sha256='a' * 64,
        proof_bytes=10,
        verification_uri='https://example.test/archive',
    )
    return AgenticWorkspaceSource(
        source_id=source_id,
        path=path,
        display_title=f'Source {source_id.removeprefix("source-")}',
        artifact_kind=AgenticArtifactKind.RAW,
        media_type=media_type,
        sha256=digest,
        byte_count=len(content),
        source_url='https://example.test/source',
        license_id='fixture-license',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        temporal_proofs=(proof,),
        selected_proof_id=proof.proof_id,
        effective_available_at_upper=witnessed.upper_at,
    )


def _build(tmp_path: Path, content: bytes = b'candidate|dose\ncandidate-001|120\n'):
    source = _raw_source(content)
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (source,))
    return build_agentic_workspace(
        workspace_id='workspace-1',
        task=task,
        episode_manifest=episode_manifest,
        build_policy=policy,
        discovery_manifest=discovery,
        assurance_profile=AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT,
        sources=(source,),
        transformations=(),
        source_bytes={source.source_id: content},
        output_root=tmp_path / 'workspace',
    )


def test_build_and_load_binds_every_visible_path_and_byte(tmp_path: Path) -> None:
    workspace = _build(tmp_path)

    assert workspace.manifest.assurance_profile == AgenticAssuranceProfile.SOURCE_ATTESTED_BEST_EFFORT
    assert workspace.read_source('source-001').startswith(b'candidate|dose')
    assert workspace.manifest_sha256
    surface = parse_model_visible_surface_bytes(workspace.model_visible_surface)
    paths = list(surface)
    assert paths == ['TASK.json', 'TASK.md', 'source-catalog.json', 'sources/source-001.txt']
    assert 'candidate-001|120' in workspace.model_visible_surface.decode()

    reloaded = load_agentic_workspace(workspace.root)
    assert reloaded.manifest == workspace.manifest
    assert reloaded.model_visible_surface == workspace.model_visible_surface


def test_workspace_rejects_extra_file_and_source_mutation(tmp_path: Path) -> None:
    workspace = _build(tmp_path)
    workspace.input_root.chmod(0o755)
    (workspace.input_root / 'extra.txt').write_text('future result', encoding='utf-8')
    workspace.input_root.chmod(0o555)
    with pytest.raises(AgenticWorkspaceError, match='exact inventory mismatch'):
        load_agentic_workspace(workspace.root)

    workspace.input_root.chmod(0o755)
    (workspace.input_root / 'extra.txt').unlink()
    workspace.input_root.chmod(0o555)
    source_path = workspace.input_root / 'sources/source-001.txt'
    source_path.chmod(0o644)
    source_path.write_text('candidate|dose\ncandidate-001|999\n', encoding='utf-8')
    source_path.chmod(0o444)
    with pytest.raises(AgenticWorkspaceError, match='binding mismatch'):
        load_agentic_workspace(workspace.root)


def test_workspace_rejects_symlink_and_hidden_file(tmp_path: Path) -> None:
    workspace = _build(tmp_path)
    workspace.input_root.chmod(0o755)
    (workspace.input_root / '.hidden').write_text('leak', encoding='utf-8')
    workspace.input_root.chmod(0o555)
    with pytest.raises(AgenticWorkspaceError, match='hidden'):
        load_agentic_workspace(workspace.root)

    workspace.input_root.chmod(0o755)
    (workspace.input_root / '.hidden').unlink()
    (workspace.input_root / 'link.txt').symlink_to(workspace.input_root / 'TASK.md')
    workspace.input_root.chmod(0o555)
    with pytest.raises(AgenticWorkspaceError, match='symlinks'):
        load_agentic_workspace(workspace.root)


def test_workspace_rejects_empty_directory_and_permission_metadata_channels(tmp_path: Path) -> None:
    workspace = _build(tmp_path)
    workspace.input_root.chmod(0o755)
    (workspace.input_root / 'phase3-success').mkdir()
    (workspace.input_root / 'phase3-success').chmod(0o555)
    workspace.input_root.chmod(0o555)
    with pytest.raises(AgenticWorkspaceError, match='directory inventory mismatch'):
        load_agentic_workspace(workspace.root)

    workspace.input_root.chmod(0o755)
    (workspace.input_root / 'phase3-success').chmod(0o755)
    (workspace.input_root / 'phase3-success').rmdir()
    workspace.input_root.chmod(0o555)
    source_path = workspace.input_root / 'sources/source-001.txt'
    source_path.chmod(0o666)
    with pytest.raises(AgenticWorkspaceError, match='mode 0444'):
        load_agentic_workspace(workspace.root)

    source_path.chmod(0o444)
    workspace.input_root.chmod(0o755)
    with pytest.raises(AgenticWorkspaceError, match='mode 0555'):
        load_agentic_workspace(workspace.root)


def test_noncanonical_json_source_fails_before_materialization(tmp_path: Path) -> None:
    content = b'{"b": 2, "a": 1}\n'
    source = _raw_source(
        content,
        path='sources/source-001.json',
        media_type=AgenticMediaType.JSON,
    )
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (source,))
    with pytest.raises(AgenticWorkspaceError, match='canonical encoding'):
        build_agentic_workspace(
            workspace_id='workspace-1',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(source,),
            transformations=(),
            source_bytes={source.source_id: content},
            output_root=tmp_path / 'workspace',
        )

    canonical = canonical_json_bytes({'a': 1, 'b': 2})
    assert _raw_source(
        canonical,
        path='sources/source-001.json',
        media_type=AgenticMediaType.JSON,
    ).byte_count == len(canonical)


def test_source_byte_binding_and_output_reuse_fail_closed(tmp_path: Path) -> None:
    source = _raw_source(b'expected\n')
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (source,))
    with pytest.raises(AgenticWorkspaceError, match='byte binding mismatch'):
        build_agentic_workspace(
            workspace_id='workspace-1',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(source,),
            transformations=(),
            source_bytes={source.source_id: b'different\n'},
            output_root=tmp_path / 'workspace',
        )

    _build(tmp_path)
    with pytest.raises(AgenticWorkspaceError, match='already exists'):
        _build(tmp_path)


def test_paths_reject_hidden_and_unicode_case_collisions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='hidden'):
        _raw_source(b'x\n', path='sources/.future.txt')

    upper = _raw_source(b'a\n', source_id='source-001', path='sources/source-001.TXT')
    lower = _raw_source(b'b\n', source_id='source-002', path='sources/source-001.txt')
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (upper, lower))
    with pytest.raises(AgenticWorkspaceError, match='case folding'):
        build_agentic_workspace(
            workspace_id='workspace-1',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(upper, lower),
            transformations=(),
            source_bytes={'source-001': b'a\n', 'source-002': b'b\n'},
            output_root=tmp_path / 'workspace',
        )


def test_selection_contract_rejects_semantic_public_source_presentation() -> None:
    semantic = _raw_source(b'evidence\n', path='sources/high-dose-winner.txt')
    task, _episode_manifest = bind_episode_manifest(_task())

    with pytest.raises(ValidationError, match='neutral source-NNN'):
        selection_contract(task, (semantic,))


def test_workspace_cannot_omit_a_source_included_by_discovery_capture(tmp_path: Path) -> None:
    source_a = _raw_source(b'a evidence\n', source_id='source-001', path='sources/source-001.txt')
    source_b = _raw_source(b'b evidence\n', source_id='source-002', path='sources/source-002.txt')
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (source_a, source_b))

    with pytest.raises(AgenticWorkspaceError, match='source order'):
        build_agentic_workspace(
            workspace_id='workspace-omitted-source',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(source_a,),
            transformations=(),
            source_bytes={'source-001': b'a evidence\n'},
            output_root=tmp_path / 'workspace',
        )


def test_duplicate_transformation_outputs_cannot_shadow_declared_receipt(tmp_path: Path) -> None:
    raw_bytes = b'raw evidence\n'
    derived_bytes = b'extracted evidence\n'
    raw = _raw_source(raw_bytes, source_id='source-001', path='sources/source-001.txt')

    def receipt(receipt_id: str, kind: AgenticDerivationKind) -> AgenticTransformationReceipt:
        execution = f'execution-{receipt_id}'.encode()
        return AgenticTransformationReceipt(
            receipt_id=receipt_id,
            kind=kind,
            input_source_ids=('source-001',),
            output_source_id='source-002',
            output_sha256=hashlib.sha256(derived_bytes).hexdigest(),
            output_bytes=len(derived_bytes),
            transform_id=f'transform-{receipt_id}',
            transform_version='1',
            executable_sha256='1' * 64,
            config_sha256='2' * 64,
            execution_receipt_sha256=hashlib.sha256(execution).hexdigest(),
            execution_receipt_bytes=len(execution),
            executed_at=datetime(2026, 1, 1, tzinfo=UTC),
            semantic_rewrite=kind == AgenticDerivationKind.LLM,
            source_span_mapping_complete=kind == AgenticDerivationKind.DETERMINISTIC,
            span_map_sha256='3' * 64 if kind == AgenticDerivationKind.DETERMINISTIC else None,
            span_map_bytes=1 if kind == AgenticDerivationKind.DETERMINISTIC else None,
        )

    declared = receipt('a-declared-llm', AgenticDerivationKind.LLM)
    shadow = receipt('z-shadow-deterministic', AgenticDerivationKind.DETERMINISTIC)
    derived = AgenticWorkspaceSource(
        source_id='source-002',
        path='sources/source-002.txt',
        display_title='Source 002',
        artifact_kind=AgenticArtifactKind.DERIVED,
        media_type=AgenticMediaType.TEXT,
        sha256=hashlib.sha256(derived_bytes).hexdigest(),
        byte_count=len(derived_bytes),
        source_url='derived://fixture',
        license_id='inherits-parent',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        effective_available_at_upper=raw.effective_available_at_upper,
        parent_source_ids=('source-001',),
        transformation_receipt_id=declared.receipt_id,
    )
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (raw, derived), (declared, shadow))
    with pytest.raises(AgenticWorkspaceError, match='unique output source IDs'):
        build_agentic_workspace(
            workspace_id='workspace-shadow',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(raw, derived),
            transformations=(declared, shadow),
            source_bytes={'source-001': raw_bytes, 'source-002': derived_bytes},
            output_root=tmp_path / 'workspace',
        )


def test_alias_permutation_receipt_binds_private_candidate_mapping(tmp_path: Path) -> None:
    source = _raw_source(b'neutral evidence\n')
    task, episode_manifest = bind_episode_manifest(_task())
    policy, discovery = selection_contract(task, (source,))
    assignments = discovery.alias_permutation_receipt.candidate_assignments
    swapped_assignments = (
        assignments[0].model_copy(
            update={'candidate_key_commitment_sha256': assignments[1].candidate_key_commitment_sha256}
        ),
        assignments[1].model_copy(
            update={'candidate_key_commitment_sha256': assignments[0].candidate_key_commitment_sha256}
        ),
    )
    altered_receipt = discovery.alias_permutation_receipt.model_copy(
        update={'candidate_assignments': swapped_assignments}
    )
    altered_discovery = discovery.model_copy(
        update={
            'alias_permutation_receipt': altered_receipt,
            'build_policy_sha256': agentic_model_sha256(policy),
        }
    )

    with pytest.raises(AgenticWorkspaceError, match='private candidate keys'):
        build_agentic_workspace(
            workspace_id='workspace-swapped-aliases',
            task=task,
            episode_manifest=episode_manifest,
            build_policy=policy,
            discovery_manifest=altered_discovery,
            assurance_profile=AgenticAssuranceProfile.FIXTURE,
            sources=(source,),
            transformations=(),
            source_bytes={'source-001': b'neutral evidence\n'},
            output_root=tmp_path / 'workspace',
        )


def test_logical_broker_exposes_only_committed_content_metadata(tmp_path: Path) -> None:
    workspace = _build(tmp_path)
    broker = workspace.brokered_surface()

    assert broker.raw_host_filesystem_exposure_sealed is False
    assert workspace.manifest.raw_host_filesystem_exposure_sealed is False
    assert workspace.manifest.official_release_ready is False
    assert workspace.manifest.prospective_input_structurally_eligible is False
    assert workspace.manifest.episode_synthetic is True
    assert workspace.manifest.episode_split.value == 'dev'
    assert not hasattr(broker, 'root')
    assert [file.path for file in broker.list_files()] == [
        'TASK.json',
        'TASK.md',
        'source-catalog.json',
        'sources/source-001.txt',
    ]
    source_metadata = broker.list_files()[-1]
    assert set(vars(source_metadata)) == {'path', 'media_type', 'sha256', 'byte_count'}
    assert broker.read('sources/source-001.txt', offset=0, limit=9) == b'candidate'
    hits = broker.search('candidate-001', paths=('sources/source-001.txt',))
    assert hits and {hit.path for hit in hits} == {'sources/source-001.txt'}
    with pytest.raises(AgenticWorkspaceError, match='invalid logical workspace path'):
        broker.read('../private/workspace-manifest.json')
