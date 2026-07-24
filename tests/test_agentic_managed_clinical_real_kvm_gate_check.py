from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import vaxreplay.agentic.managed_clinical_real_kvm_gate_check as gate_check
from vaxreplay.agentic.provider_subprocess import ProviderSubprocessRequest
from vaxreplay.bundle import canonical_json_bytes


def test_gate_check_builds_one_exact_canonical_blocked_call_fixture(tmp_path: Path) -> None:
    interpreter = Path(sys.executable).resolve(strict=True)

    fixture = gate_check._build_fixture(output_root=tmp_path, interpreter=interpreter)

    plan = json.loads(fixture.plan)
    request = ProviderSubprocessRequest.model_validate_json(fixture.request)
    assert canonical_json_bytes(plan) == fixture.plan
    assert canonical_json_bytes(request) == fixture.request
    assert fixture.child.startswith(f'#!{interpreter} -ISB\n'.encode())
    assert plan['turns'] == [
        {
            'call_index': 0,
            'content': fixture.expected_content,
            'input_tokens': 1,
            'output_tokens': 1,
            'reasoning_tokens': 0,
        }
    ]
    assert plan['observation_gate'] == {
        'binding_token_sha256': gate_check._sha256(gate_check._GATE_BINDING_TOKEN),
        'challenge_nonce_hex': gate_check._CHALLENGE_NONCE_HEX,
        'drill_id': gate_check._DRILL_ID,
        'path': str(tmp_path / 'observation-gate.json'),
        'provider_call_index': 0,
        'timeout_seconds': 10,
    }
    assert request.request.run_id == fixture.release.run_id
    assert request.request.call_index == fixture.release.provider_call_index == 0
    assert fixture.release.persisted_path == str(tmp_path / 'observation-gate.json')
    assert fixture.release.live_process_observation_sha256 == gate_check._sha256(
        b'vaxreplay-gate-check-no-live-process-observation-v1'
    )


def test_gate_check_receipt_cannot_claim_live_or_authenticated_evidence() -> None:
    digest = '0' * 64
    receipt = gate_check.ManagedClinicalRealKvmGateCheckReceipt(
        runtime_closure_id='test-closure',
        runtime_closure_manifest_sha256=digest,
        runtime_closure_receipt_sha256=digest,
        runtime_closure_sha256=digest,
        runtime_closure_entry_count=1,
        interpreter_path='/opt/runtime/bin/python3',
        checker_module_path='/opt/runtime/checker.py',
        checker_module_sha256=digest,
        collector_module_path='/opt/runtime/collector.py',
        collector_module_sha256=digest,
        output_root='/root/check',
        provider_plan_path='/root/check/plan.json',
        provider_plan_sha256=digest,
        provider_child_path='/root/check/child',
        provider_child_sha256=digest,
        provider_request_path='/root/check/request.json',
        provider_request_sha256=digest,
        observation_gate_path='/root/check/gate.json',
        observation_gate_sha256=digest,
        provider_response_path='/root/check/response.json',
        provider_response_sha256=digest,
    )

    assert receipt.passed is True
    assert receipt.runtime_closure_verified is True
    assert receipt.reproducible_build_claimed is False
    assert receipt.self_contained_executable_claimed is False
    assert receipt.native_operating_system_libraries_pinned is False
    assert receipt.external_provider_called is False
    assert receipt.real_kvm_run_performed is False
    assert receipt.firecracker_process_started is False
    assert receipt.fixed_deployment_written is False
    assert receipt.receipt_authenticated is False
    assert receipt.live_qualification_claimed is False
    assert canonical_json_bytes(receipt) == canonical_json_bytes(
        gate_check.ManagedClinicalRealKvmGateCheckReceipt.model_validate_json(canonical_json_bytes(receipt))
    )


def test_gate_check_requires_linux_root_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate_check.sys, 'platform', 'darwin')

    with pytest.raises(gate_check.ManagedClinicalRealKvmGateCheckError, match='UID 0 on Linux'):
        gate_check._require_execution_environment()


def test_gate_check_rejects_non_absolute_or_existing_output(tmp_path: Path) -> None:
    with pytest.raises(gate_check.ManagedClinicalRealKvmGateCheckError, match='normalized and absolute'):
        gate_check._normalized_fresh_output_root(Path('relative'))
    with pytest.raises(gate_check.ManagedClinicalRealKvmGateCheckError, match='must not already exist'):
        gate_check._normalized_fresh_output_root(tmp_path)


def test_gate_check_cli_requires_exact_runtime_closure_pins(tmp_path: Path) -> None:
    parser = gate_check._parser()
    arguments = parser.parse_args(
        [
            '--output-root',
            str(tmp_path / 'output'),
            '--runtime-closure-root',
            str(tmp_path / 'closure'),
            '--expected-runtime-closure-manifest-sha256',
            '1' * 64,
            '--expected-runtime-closure-receipt-sha256',
            '2' * 64,
            '--expected-runtime-closure-sha256',
            '3' * 64,
        ]
    )

    assert arguments.output_root == tmp_path / 'output'
    assert arguments.runtime_closure_root == tmp_path / 'closure'
