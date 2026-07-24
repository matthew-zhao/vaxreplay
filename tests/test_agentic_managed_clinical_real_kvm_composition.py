from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import vaxreplay.agentic.managed_clinical_real_kvm_collector as collector
from tests.test_agentic_clinical_operator import _manifest, _receipt_bound_manifest
from tests.test_clinicaltrials_execution_aggregation import _case
from vaxreplay.agentic.clinical_guest_executable import LaneAClinicalGuestConfig
from vaxreplay.agentic.firecracker import firecracker_model_sha256
from vaxreplay.agentic.guest_disk_build import lane_a_guest_disk_build_receipt_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.execution_aggregation import ExecutionCohortManifest


def test_execution_cohort_derives_a_schema_valid_public_lineage() -> None:
    challenge_sha256 = 'c' * 64

    cohort = collector._execution_cohort(
        _case(1).task,
        challenge_sha256=challenge_sha256,
    )

    assert cohort.tasks[0].public_lineage_id == f'lineage-{challenge_sha256[:20]}'
    assert ExecutionCohortManifest.model_validate_json(canonical_json_bytes(cohort)) == cohort


def test_pre_reservation_composition_validates_all_generated_id_surfaces(tmp_path: Path) -> None:
    manifest, materials = _manifest(tmp_path / 'operator-subject')
    bound_manifest, disk_receipt = _receipt_bound_manifest(manifest, materials)
    challenge_sha256 = 'c' * 64
    task = _case(1).task
    state_root = tmp_path / 'managed-state'
    state_root.mkdir()
    paths = cast(
        collector.DrillPaths,
        SimpleNamespace(
            workspace_root=state_root / 'workspace',
            provider_child=state_root / 'provider-fixture' / 'provider-child',
            provider_plan=state_root / 'provider-fixture' / 'provider-plan.json',
            registry_database=state_root / 'registry' / 'attempts.sqlite3',
            registry_socket=Path('/run/vrk/composition-test/attempts.sock'),
            evidence_root=state_root / 'production-evidence',
            protocol_audit_root=state_root / 'registry-audit',
            startup_receipt_root=state_root / 'startup-receipts',
            ownership_root=state_root / 'ownership',
            gateway_database=state_root / 'gateway' / 'gateway.sqlite3',
            config_root=state_root / 'fixed-config',
            operator_secret_root=state_root / 'fixed-config' / 'operator-secrets',
            managed_secret_root=state_root / 'fixed-config' / 'managed-secrets',
        ),
    )
    arguments = cast(
        Namespace,
        SimpleNamespace(
            expected_disk_build_receipt_sha256=lane_a_guest_disk_build_receipt_sha256(disk_receipt),
            expected_worker_spec_sha256=firecracker_model_sha256(materials.spec),
            worker_spec=tmp_path / 'worker.json',
            disk_build_receipt=tmp_path / 'disk-build-receipt.json',
            qualification_root=tmp_path / 'qualification',
            expected_qualification_artifact_sha256='1' * 64,
            expected_qualification_key_id='2' * 64,
            expected_qualification_collector_evidence_sha256='3' * 64,
            expected_qualification_probe_manifest_sha256='4' * 64,
            expected_qualification_runtime_closure_manifest_sha256='5' * 64,
            expected_qualification_runtime_closure_receipt_sha256='6' * 64,
            expected_qualification_runtime_closure_sha256='7' * 64,
            expected_qualification_collector_public_key_hex='8' * 64,
            expected_qualification_collector_key_id='9' * 64,
            expected_qualification_verifier_source_sha256='a' * 64,
            expected_collector_interpreter_sha256='b' * 64,
        ),
    )
    inputs = cast(
        collector.PublicInputs,
        SimpleNamespace(
            spec=materials.spec,
            policy=materials.policy,
            rpc_policy=materials.guest.policy,
            guest_config=LaneAClinicalGuestConfig(
                trust_anchor=bound_manifest.bootstrap_trust_anchor,
                guest_rpc_port=materials.spec.guest_rpc_port,
            ),
            disk_receipt=disk_receipt,
            task=task,
        ),
    )
    authorization = cast(
        collector.DrillAuthorization,
        SimpleNamespace(
            challenge_sha256=challenge_sha256,
            registry_authority_id=collector.managed_clinical_real_kvm_authority_id(challenge_sha256=challenge_sha256),
        ),
    )

    composition = collector._build_pre_reservation_composition(
        arguments,
        inputs=inputs,
        paths=paths,
        keys=collector.DrillKeys.generate(),
        authorization=authorization,
        provider_plan=b'{}',
        provider_child=b'#!/bin/false\n',
    )

    assert composition.cohort.tasks[0].public_lineage_id == f'lineage-{challenge_sha256[:20]}'
    assert composition.cohort.cohort_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.runtime_config.runtime_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.runtime_config.bootstrap_connection_timeout_seconds == min(
        30.0,
        float(materials.spec.limits.wall_seconds),
    )
    assert composition.runtime_config.bootstrap_connection_timeout_seconds > 5.0
    assert composition.gateway_policy.gateway_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.gateway_route.route_id == f'managed-real-kvm-public-{challenge_sha256[:20]}'
    assert composition.startup_config.reconciler_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.ownership_config.ledger_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.registry_config.service_id == f'managed-real-kvm-{challenge_sha256[:32]}'
    assert composition.provisional_manifest.episode_id == task.context.episode_id
