from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_clinicaltrials_execution_public_release import _release
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_identity_inventory import (
    AuthenticatedExecutionIdentityFingerprintInventory,
    ExecutionIdentityFingerprintError,
    ExecutionIdentityFingerprintInventory,
    authenticate_execution_identity_fingerprint_inventory,
    build_execution_identity_fingerprint_inventory,
    execution_identity_fingerprint_key_id,
    load_authenticated_execution_identity_fingerprint_inventory,
    load_frozen_decision_catalog,
    write_authenticated_execution_identity_fingerprint_inventory,
)

_AUDIT_KEY = bytes(range(64, 96))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root: Path):
    workspace, _workspace_pin, release = _release(root)
    catalogs = tuple(
        load_frozen_decision_catalog(
            path,
            expected_build_receipt_sha256=_sha256(path / 'BUILD-RECEIPT.json'),
            allow_synthetic_test_only=True,
        )
        for path in (root / 'source-2018', root / 'source-2020')
    )
    return workspace, release, catalogs


def test_local_inventory_binds_exact_surfaces_without_exporting_identifiers(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    inventory = build_execution_identity_fingerprint_inventory(
        inventory_id='test-local-identity-inventory',
        release=release,
        workspace=workspace,
        catalogs=catalogs,
    )

    assert inventory.aggregate.case_count == len(inventory.cases) == 2
    assert inventory.aggregate.exact_visible_profile_unique_count == 2
    assert inventory.aggregate.coarsened_registry_core_near_unique_count == 2
    assert inventory.exact_model_facing_surfaces_bound
    assert inventory.complete_release_case_universe_covered
    assert inventory.external_model_or_provider_calls_made is False
    assert inventory.formal_contamination_strata_assigned is False
    assert inventory.workspace_future_leakage_audit_complete is False
    assert inventory.model_weight_contamination_eliminated is False
    payload = canonical_json_bytes(inventory)
    assert b'NCT' not in payload.upper()
    assert b'organizer_private_nct_id' not in payload

    by_episode = {item.context.episode_id: item for item in release.tasks}
    for case in inventory.cases:
        assert case.surface_binding.task_context_sha256 == by_episode[case.surface_binding.episode_id].context_sha256
        assert case.true_target_present_in_every_candidate_set
        assert not case.organizer_private_identity_copied_to_output


def test_catalog_loader_requires_external_pin_exact_inventory_and_real_mode(tmp_path: Path) -> None:
    _workspace, _release_build, _catalogs = _inputs(tmp_path)
    source = tmp_path / 'source-2018'
    pin = _sha256(source / 'BUILD-RECEIPT.json')

    with pytest.raises(ExecutionIdentityFingerprintError, match='external pin'):
        load_frozen_decision_catalog(
            source,
            expected_build_receipt_sha256='0' * 64,
            allow_synthetic_test_only=True,
        )
    with pytest.raises(ExecutionIdentityFingerprintError, match='synthetic'):
        load_frozen_decision_catalog(source, expected_build_receipt_sha256=pin)

    extra = source / 'extra.txt'
    extra.write_text('uncommitted', encoding='utf-8')
    with pytest.raises(ExecutionIdentityFingerprintError, match='exact artifact inventory'):
        load_frozen_decision_catalog(
            source,
            expected_build_receipt_sha256=pin,
            allow_synthetic_test_only=True,
        )


def test_missing_anchor_catalog_fails_before_case_output(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    with pytest.raises(ExecutionIdentityFingerprintError, match='same-anchor'):
        build_execution_identity_fingerprint_inventory(
            inventory_id='missing-anchor',
            release=release,
            workspace=workspace,
            catalogs=catalogs[:1],
        )


def test_surface_mutation_after_release_verification_fails_closed(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    task_path = release.root / release.receipt.artifacts[0].relative_path
    original = task_path.read_bytes()
    task_path.chmod(0o644)
    task_path.write_bytes(original + b'\nchanged-after-verification\n')
    task_path.chmod(0o444)

    with pytest.raises(ExecutionIdentityFingerprintError, match='unsafe or oversized|changed after verification'):
        build_execution_identity_fingerprint_inventory(
            inventory_id='post-verification-mutation',
            release=release,
            workspace=workspace,
            catalogs=catalogs,
        )


def test_authenticated_artifact_is_atomic_private_pinned_and_hmac_verified(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    inventory = build_execution_identity_fingerprint_inventory(
        inventory_id='authenticated-local-identity-inventory',
        release=release,
        workspace=workspace,
        catalogs=catalogs,
    )
    key_id = execution_identity_fingerprint_key_id(_AUDIT_KEY)
    authenticated = authenticate_execution_identity_fingerprint_inventory(
        inventory,
        key=_AUDIT_KEY,
        expected_key_id=key_id,
    )
    output = tmp_path / 'identity-inventory'
    artifact_sha256 = write_authenticated_execution_identity_fingerprint_inventory(
        authenticated,
        output_root=output,
    )
    loaded = load_authenticated_execution_identity_fingerprint_inventory(
        output,
        expected_artifact_sha256=artifact_sha256,
        key=_AUDIT_KEY,
        expected_key_id=key_id,
    )

    assert loaded == authenticated
    assert (output.stat().st_mode & 0o777) == 0o500
    artifact = next(output.iterdir())
    assert (artifact.stat().st_mode & 0o777) == 0o400
    assert b'NCT' not in artifact.read_bytes().upper()
    with pytest.raises(ExecutionIdentityFingerprintError, match='external pin'):
        load_authenticated_execution_identity_fingerprint_inventory(
            output,
            expected_artifact_sha256='0' * 64,
            key=_AUDIT_KEY,
            expected_key_id=key_id,
        )
    with pytest.raises(ExecutionIdentityFingerprintError, match='key'):
        load_authenticated_execution_identity_fingerprint_inventory(
            output,
            expected_artifact_sha256=artifact_sha256,
            key=b'x' * 32,
            expected_key_id=key_id,
        )


def test_inventory_schema_rejects_forged_aggregate_and_weight_claims(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    inventory = build_execution_identity_fingerprint_inventory(
        inventory_id='schema-negative-test',
        release=release,
        workspace=workspace,
        catalogs=catalogs,
    )
    payload = inventory.model_dump(mode='json')
    forged_aggregate = dict(payload['aggregate'])
    forged_aggregate['exact_registry_core_unique_count'] = 0
    with pytest.raises(ValidationError, match='does not reconstruct'):
        ExecutionIdentityFingerprintInventory.model_validate_json(
            canonical_json_bytes({**payload, 'aggregate': forged_aggregate})
        )
    with pytest.raises(ValidationError):
        ExecutionIdentityFingerprintInventory.model_validate_json(
            canonical_json_bytes({**payload, 'model_weight_contamination_eliminated': True})
        )


def test_authenticated_envelope_rejects_identifier_injection_and_hash_forgery(tmp_path: Path) -> None:
    workspace, release, catalogs = _inputs(tmp_path)
    inventory = build_execution_identity_fingerprint_inventory(
        inventory_id='envelope-negative-test',
        release=release,
        workspace=workspace,
        catalogs=catalogs,
    )
    authenticated = authenticate_execution_identity_fingerprint_inventory(
        inventory,
        key=_AUDIT_KEY,
        expected_key_id=execution_identity_fingerprint_key_id(_AUDIT_KEY),
    )
    payload = json.loads(canonical_json_bytes(authenticated))
    payload['inventory_sha256'] = '0' * 64
    with pytest.raises(ValidationError, match='wrong inventory hash'):
        AuthenticatedExecutionIdentityFingerprintInventory.model_validate_json(canonical_json_bytes(payload))

    case = inventory.cases[0].model_dump(mode='json')
    case['surface_binding']['episode_id'] = 'NCT00000001'
    inventory_payload = inventory.model_dump(mode='json')
    inventory_payload['cases'][0] = case
    with pytest.raises(ValidationError):
        ExecutionIdentityFingerprintInventory.model_validate_json(canonical_json_bytes(inventory_payload))
