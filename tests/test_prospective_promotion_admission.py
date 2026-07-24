from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.plan_selection import (
    PlanSelectionMaterialSpec,
    PlanSelectionPolicyBinding,
)
from vaxreplay.operations.promotion import AdapterSpec
from vaxreplay.prospective import ProspectiveFileBinding, ProspectiveSourceCaptureBinding, SourceCaptureArtifact
from vaxreplay.prospective_admission import (
    PromotionArchiveAdmissionPolicy,
    PromotionArchiveAdmissionVerifier,
    PromotionArchivePolicyEntry,
    PromotionArchiveVerificationSpec,
    ProspectiveAdmissionError,
    build_verified_prospective_admission,
    make_promotion_archive_admission_verifier,
    promotion_archive_policy_bytes,
    promotion_archive_policy_sha256,
)

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _fixture(tmp_path: Path):
    root = tmp_path / 'promotion'
    root.mkdir()
    descriptor_bytes = b'canonical promotion handoff descriptor'
    index_bytes = b'canonical capture index'
    scope_bytes = b'canonical scope policy'
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    scope_sha256 = hashlib.sha256(scope_bytes).hexdigest()
    precommit_sha256 = '5' * 64
    manifest_sha256 = '1' * 64
    descriptor_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
    source_id = 'promotion:promotion-1'
    selection_policy_bytes = b'fixture first-write-wins registry policy'
    selection_trust_bytes = b'fixture registry trust policy'
    selection_verifier_bytes = b'fixture plan-selection verifier'
    selection_policy = PlanSelectionPolicyBinding(
        campaign_id='pandemic-campaign-2027',
        selection_key='antigen-prioritization-plan',
        registry_id='independent-plan-registry',
        authority_id='benchmark-authority',
        policy_id='first-write-wins-v1',
        policy_sha256=hashlib.sha256(selection_policy_bytes).hexdigest(),
        trust_policy_id='registry-trust-v1',
        trust_policy_sha256=hashlib.sha256(selection_trust_bytes).hexdigest(),
        verifier_id='plan-selection-verifier-v1',
        verifier_implementation_sha256=hashlib.sha256(selection_verifier_bytes).hexdigest(),
    )
    selection_policy_sha256 = hashlib.sha256(canonical_json_bytes(selection_policy)).hexdigest()
    selection_manifest_sha256 = '6' * 64
    selection_materials = PlanSelectionMaterialSpec(
        policy=selection_policy,
        policy_bytes=selection_policy_bytes,
        trust_policy_bytes=selection_trust_bytes,
        verifier_implementation_bytes=selection_verifier_bytes,
        verifier=lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected verifier call')),
    )
    fake_scope = object()
    fake_index = SimpleNamespace(
        scope_policy=SimpleNamespace(sha256=scope_sha256),
        scope_precommit=SimpleNamespace(archive_sha256=precommit_sha256),
        campaign_id=selection_policy.campaign_id,
        selection_key=selection_policy.selection_key,
        selection_policy_sha256=selection_policy_sha256,
        selection_policy_artifact_sha256=selection_policy.policy_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
    )
    fake_descriptor = SimpleNamespace(
        promotion_id='promotion-1',
        promotion_manifest_sha256=manifest_sha256,
        capture_index=fake_index,
        maximum_source_release_at=_T0,
        maximum_captured_at=_T0 + timedelta(seconds=1),
        witnessed_at=_T0 + timedelta(seconds=2),
        promotion_created_at=_T0 + timedelta(seconds=3),
        campaign_id=selection_policy.campaign_id,
        selection_key=selection_policy.selection_key,
        selection_policy_sha256=selection_policy_sha256,
        selection_policy_artifact_sha256=selection_policy.policy_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
    )
    binding = ProspectiveSourceCaptureBinding(
        source_id=source_id,
        source_release_at=fake_descriptor.maximum_source_release_at,
        captured_at=fake_descriptor.maximum_captured_at,
        witnessed_at=fake_descriptor.witnessed_at,
        file=ProspectiveFileBinding(
            path='source-captures/000000.json',
            sha256=descriptor_sha256,
            byte_count=len(descriptor_bytes),
        ),
    )
    loaded_source = SourceCaptureArtifact(
        source_id=source_id,
        source_release_at=binding.source_release_at,
        captured_at=binding.captured_at,
        witnessed_at=binding.witnessed_at,
        manifest_bytes=descriptor_bytes,
    )
    loaded = SimpleNamespace(
        root=root.resolve(),
        manifest_sha256=manifest_sha256,
        index=fake_index,
        index_bytes=index_bytes,
        handoff_descriptor=fake_descriptor,
        handoff_descriptor_bytes=descriptor_bytes,
        source_captures=(loaded_source,),
        candidates=(),
        evidence=(),
    )
    policy = PromotionArchiveAdmissionPolicy(
        policy_id='official-promotion-archives-v1',
        archives=(
            PromotionArchivePolicyEntry(
                promotion_id='promotion-1',
                source_id=source_id,
                promotion_manifest_sha256=manifest_sha256,
                capture_index_sha256=index_sha256,
                handoff_descriptor_sha256=descriptor_sha256,
                scope_policy_sha256=scope_sha256,
                scope_precommit_sha256=precommit_sha256,
                campaign_id=selection_policy.campaign_id,
                selection_key=selection_policy.selection_key,
                selection_policy_sha256=selection_policy_sha256,
                selection_policy_artifact_sha256=selection_policy.policy_sha256,
                selection_manifest_sha256=selection_manifest_sha256,
            ),
        ),
    )
    spec = PromotionArchiveVerificationSpec(
        promotion_root=root,
        expected_promotion_sha256=manifest_sha256,
        expected_scope_policy=fake_scope,  # type: ignore[arg-type]
        scope_precommit_witness_materials=object(),  # type: ignore[arg-type]
        witness_materials=object(),  # type: ignore[arg-type]
        source_verifiers={},
        adapter=AdapterSpec(
            adapter_id='generic-fixture-adapter',
            adapter_version='v1',
            implementation_bytes=b'fixture adapter implementation',
            policy_bytes=b'fixture adapter policy',
            execution_environment_bytes=b'fixture adapter environment',
        ),
        verified_at=_T0 + timedelta(days=1),
        expected_scope_precommit_sha256=precommit_sha256,
        expected_campaign_id=selection_policy.campaign_id,
        expected_selection_key=selection_policy.selection_key,
        expected_selection_policy_sha256=selection_policy_sha256,
        expected_selection_policy_artifact_sha256=selection_policy.policy_sha256,
        expected_selection_manifest_sha256=selection_manifest_sha256,
        selection_materials=selection_materials,
    )
    return {
        'root': root,
        'source_id': source_id,
        'descriptor_bytes': descriptor_bytes,
        'index_bytes': index_bytes,
        'scope_bytes': scope_bytes,
        'fake_scope': fake_scope,
        'fake_index': fake_index,
        'fake_descriptor': fake_descriptor,
        'binding': binding,
        'loaded': loaded,
        'policy': policy,
        'spec': spec,
        'selection_policy': selection_policy,
        'selection_materials': selection_materials,
    }


def test_archive_verifier_reloads_and_rejects_swapped_root_digest_index_and_times(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        if value is fixture['fake_descriptor'] or hasattr(value, 'promotion_manifest_sha256'):
            return fixture['descriptor_bytes']
        if value is fixture['fake_index']:
            return fixture['index_bytes']
        return original_canonical_json_bytes(value)

    with patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes):
        verifier = make_promotion_archive_admission_verifier(
            policy=fixture['policy'],
            archives={fixture['source_id']: fixture['spec']},
        )

    policy_bytes = promotion_archive_policy_bytes(fixture['policy'])
    loader_target = 'vaxreplay.prospective_admission.load_capture_promotion'
    parser_target = 'vaxreplay.prospective_admission.PromotionHandoffDescriptor.model_validate_json'
    index_hash_target = 'vaxreplay.prospective_admission.capture_index_sha256'
    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        patch(parser_target, return_value=fixture['fake_descriptor']),
        patch(index_hash_target, return_value=hashlib.sha256(fixture['index_bytes']).hexdigest()),
        patch(loader_target, return_value=fixture['loaded']) as loader,
    ):
        assert verifier(fixture['binding'], fixture['descriptor_bytes'], policy_bytes)
        loader.assert_called_once()
        assert loader.call_args.kwargs['selection_materials'] is fixture['selection_materials']
        assert (
            loader.call_args.kwargs['expected_selection_manifest_sha256']
            == fixture['policy'].archives[0].selection_manifest_sha256
        )

        swapped_root = SimpleNamespace(**vars(fixture['loaded']))
        swapped_root.root = (tmp_path / 'different-promotion').resolve()
        loader.return_value = swapped_root
        assert not verifier(fixture['binding'], fixture['descriptor_bytes'], policy_bytes)

        loader.return_value = fixture['loaded']
        swapped_digest = SimpleNamespace(**vars(fixture['fake_descriptor']))
        swapped_digest.promotion_manifest_sha256 = 'f' * 64
        with patch(parser_target, return_value=swapped_digest):
            assert not verifier(fixture['binding'], fixture['descriptor_bytes'], policy_bytes)

        with patch(index_hash_target, return_value='0' * 64):
            assert not verifier(fixture['binding'], fixture['descriptor_bytes'], policy_bytes)

        swapped_times = fixture['binding'].model_copy(
            update={'captured_at': fixture['binding'].captured_at + timedelta(microseconds=1)}
        )
        assert not verifier(swapped_times, fixture['descriptor_bytes'], policy_bytes)


def test_policy_hash_commits_exact_archive_digest_allowlist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = fixture['policy']
    stale = original.model_dump()
    stale['schema_version'] = 'vaxreplay.promotion-archive-admission-policy.v0.1'
    with pytest.raises(ValueError, match='schema_version'):
        PromotionArchiveAdmissionPolicy.model_validate(stale)
    changed_entry = original.archives[0].model_copy(update={'promotion_manifest_sha256': 'f' * 64})
    changed = original.model_copy(update={'archives': (changed_entry,)})

    assert b'"promotion_manifest_sha256":"' + b'1' * 64 + b'"' in promotion_archive_policy_bytes(original)
    assert promotion_archive_policy_sha256(original) != promotion_archive_policy_sha256(changed)
    changed_selection = original.model_copy(
        update={'archives': (original.archives[0].model_copy(update={'selection_manifest_sha256': 'e' * 64}),)}
    )
    assert promotion_archive_policy_sha256(original) != promotion_archive_policy_sha256(changed_selection)


def test_admission_rejects_direct_verifier_with_policy_bytes_split_brain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    forged_policy_bytes = b'benign policy bytes committed by the admission'
    forged = PromotionArchiveAdmissionVerifier(
        policy=fixture['policy'],
        policy_bytes=forged_policy_bytes,
        archives={fixture['source_id']: fixture['spec']},
        require_hermetic_execution=True,
    )

    assert not forged(fixture['binding'], fixture['descriptor_bytes'], forged_policy_bytes)
    with pytest.raises(ProspectiveAdmissionError, match='policy_bytes differ from its canonical policy'):
        build_verified_prospective_admission(
            release_id='forged-release',
            suite_id='forged-suite',
            packages=(),
            seals=(),
            split_inventory=None,  # type: ignore[arg-type]
            case_universe=None,  # type: ignore[arg-type]
            case_universe_proof=b'not reached',
            eligibility_protocol=b'eligibility',
            verifier_policy=b'verifier',
            source_capture_policy=forged_policy_bytes,
            attempt_policy=b'attempt',
            run_deadline_at=_T0,
            receipt_verifier=lambda *_args: True,  # type: ignore[arg-type]
            case_universe_seal_verifier=lambda *_args: True,
            source_capture_verifier=forged,
        )


@pytest.mark.parametrize(
    ('spec_updates', 'message'),
    (
        ({'expected_selection_key': 'different-plan-key'}, 'plan selection differs'),
        ({'expected_selection_policy_sha256': 'd' * 64}, 'plan selection differs'),
        ({'expected_selection_policy_artifact_sha256': 'c' * 64}, 'plan selection differs'),
        ({'expected_selection_manifest_sha256': 'b' * 64}, 'plan selection differs'),
    ),
)
def test_archive_factory_rejects_selection_spec_mismatch(
    tmp_path: Path,
    spec_updates: dict[str, object],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    mismatched = replace(fixture['spec'], **spec_updates)
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        return original_canonical_json_bytes(value)

    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        pytest.raises(ProspectiveAdmissionError, match=message),
    ):
        make_promotion_archive_admission_verifier(
            policy=fixture['policy'],
            archives={fixture['source_id']: mismatched},
        )


def test_archive_factory_rejects_selection_material_policy_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    materials = fixture['selection_materials']
    different_policy = materials.policy.model_copy(update={'selection_key': 'different-plan-key'})
    mismatched = replace(
        fixture['spec'],
        selection_materials=replace(materials, policy=different_policy),
    )
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        return original_canonical_json_bytes(value)

    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        pytest.raises(ProspectiveAdmissionError, match='material policy differs'),
    ):
        make_promotion_archive_admission_verifier(
            policy=fixture['policy'],
            archives={fixture['source_id']: mismatched},
        )


def test_tier_a_archive_factory_rejects_legacy_in_process_callbacks(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        return original_canonical_json_bytes(value)

    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        pytest.raises(ProspectiveAdmissionError, match='requires hermetic source-verifier and adapter specs'),
    ):
        make_promotion_archive_admission_verifier(
            policy=fixture['policy'],
            archives={fixture['source_id']: fixture['spec']},
            require_hermetic_execution=True,
        )


def test_archive_factory_rejects_symlink_in_promotion_root_parent_chain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    link_parent = tmp_path / 'link-parent'
    link_parent.symlink_to(tmp_path, target_is_directory=True)
    linked_spec = replace(
        fixture['spec'],
        promotion_root=link_parent / fixture['root'].name,
    )
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        return original_canonical_json_bytes(value)

    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        pytest.raises(ProspectiveAdmissionError, match='root is unsafe'),
    ):
        make_promotion_archive_admission_verifier(
            policy=fixture['policy'],
            archives={fixture['source_id']: linked_spec},
        )


def test_archive_factory_detects_reused_root_by_device_and_inode(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first_entry = fixture['policy'].archives[0]
    second_entry = first_entry.model_copy(
        update={
            'promotion_id': 'promotion-2',
            'source_id': 'promotion:promotion-2',
            'promotion_manifest_sha256': '2' * 64,
        }
    )
    policy = fixture['policy'].model_copy(update={'archives': (first_entry, second_entry)})
    second_spec = replace(
        fixture['spec'],
        promotion_root=tmp_path / 'different-path',
        expected_promotion_sha256='2' * 64,
    )
    original_canonical_json_bytes = __import__(
        'vaxreplay.prospective_admission',
        fromlist=['canonical_json_bytes'],
    ).canonical_json_bytes

    def exact_bytes(value):
        if value is fixture['fake_scope']:
            return fixture['scope_bytes']
        return original_canonical_json_bytes(value)

    with (
        patch('vaxreplay.prospective_admission.canonical_json_bytes', side_effect=exact_bytes),
        patch(
            'vaxreplay.prospective_admission.immutable_root_identity',
            side_effect=(
                (fixture['root'], (123, 456)),
                (tmp_path / 'different-path', (123, 456)),
            ),
        ),
        pytest.raises(ProspectiveAdmissionError, match='cannot reuse one root'),
    ):
        make_promotion_archive_admission_verifier(
            policy=policy,
            archives={
                first_entry.source_id: fixture['spec'],
                second_entry.source_id: second_spec,
            },
        )
