from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import cast

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.hermetic_execution import HermeticCallbackExecutor, HermeticSandboxPolicy
from vaxreplay.operations.plan_selection import PlanSelectionPolicyBinding
from vaxreplay.operations.promotion import AdapterSpec, HermeticExecutionSpec, SourceVerifierSpec
from vaxreplay.operations.promotion_schema import PromotionScopePolicy, PromotionSourceScope
from vaxreplay.operations.scope_precommit import derive_pre_capture_plan
from vaxreplay.operations.witness import ExternalWitnessMethod, WitnessPolicyBinding

_T0 = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
_PUBLIC_KEY = bytes(range(32))
_SECCOMP_SHA256 = hashlib.sha256(b'pinned seccomp profile').hexdigest()


def _sandbox_policy(*, wall_seconds: int = 30) -> bytes:
    policy = HermeticSandboxPolicy(
        policy_id='tier-a-hermetic-policy-v1',
        authority_id='independent-hermetic-runner',
        signing_key_id='runner-key-2026-07',
        signing_public_key_sha256=hashlib.sha256(_PUBLIC_KEY).hexdigest(),
        seccomp_profile_sha256=_SECCOMP_SHA256,
        wall_seconds=wall_seconds,
        memory_mib=256,
        milli_cpus=500,
        pids=32,
        scratch_mib=32,
        open_files=64,
        max_input_bytes=1024 * 1024,
        max_callback_policy_bytes=1024 * 1024,
        max_output_bytes=1024 * 1024,
        max_worker_response_bytes=2 * 1024 * 1024,
        max_log_bytes=64 * 1024,
    )
    return canonical_json_bytes(policy)


def _hermetic_execution(*, wall_seconds: int = 30) -> HermeticExecutionSpec:
    return HermeticExecutionSpec(
        sandbox_policy_bytes=_sandbox_policy(wall_seconds=wall_seconds),
        seccomp_profile_bytes=b'pinned seccomp profile',
        trusted_public_key_bytes=_PUBLIC_KEY,
        executor=cast(HermeticCallbackExecutor, object()),
    )


def _scope() -> PromotionScopePolicy:
    return PromotionScopePolicy(
        policy_id='tier-a-scope-v1',
        store_id='a' * 32,
        checkpoint_created_at_not_before=_T0 + timedelta(hours=1),
        checkpoint_created_at_not_after=_T0 + timedelta(hours=2),
        sources=(
            PromotionSourceScope(
                source_id='iedb:prospective-iq-api',
                job_spec_sha256s=('b' * 64,),
                scheduled_from=_T0 + timedelta(minutes=30),
                scheduled_through=_T0 + timedelta(minutes=30),
            ),
        ),
    )


def _selection_policy() -> PlanSelectionPolicyBinding:
    return PlanSelectionPolicyBinding(
        campaign_id='pandemic-campaign-2027',
        selection_key='antigen-universe-v1',
        registry_id='independent-selection-registry',
        authority_id='benchmark-authority',
        policy_id='first-write-wins-v1',
        policy_sha256='1' * 64,
        trust_policy_id='selection-trust-v1',
        trust_policy_sha256='2' * 64,
        verifier_id='selection-verifier-v1',
        verifier_implementation_sha256='3' * 64,
    )


def _witness_policy() -> WitnessPolicyBinding:
    return WitnessPolicyBinding(
        authority_id='independent-capture-witness',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id='capture-witness-v1',
        policy_sha256='4' * 64,
        trust_policy_id='capture-witness-trust-v1',
        trust_policy_sha256='5' * 64,
        verifier_id='capture-witness-verifier-v1',
        verifier_implementation_sha256='6' * 64,
    )


def _specs(*, wall_seconds: int = 30) -> tuple[dict[str, SourceVerifierSpec], AdapterSpec]:
    execution = _hermetic_execution(wall_seconds=wall_seconds)
    sources = {
        'iedb:prospective-iq-api': SourceVerifierSpec(
            verifier_id='iedb-source-verifier',
            verifier_version='v1',
            implementation_bytes=b'iedb verifier image implementation',
            policy_bytes=b'iedb source verifier policy',
            execution_environment_bytes=b'iedb verifier environment',
            hermetic_execution=execution,
        )
    }
    adapter = AdapterSpec(
        adapter_id='iedb-antigen-adapter',
        adapter_version='v1',
        implementation_bytes=b'iedb adapter image implementation',
        policy_bytes=b'iedb adapter policy',
        execution_environment_bytes=b'iedb adapter environment',
        hermetic_execution=execution,
        allowed_exclusion_reason_codes=('source_metadata_record',),
    )
    return sources, adapter


def test_pre_capture_plan_binds_hermetic_sandbox_key_and_seccomp() -> None:
    sources, adapter = _specs()
    plan = derive_pre_capture_plan(
        scope_policy=_scope(),
        selection_policy=_selection_policy(),
        capture_witness_policy=_witness_policy(),
        source_verifiers=sources,
        adapter=adapter,
    )

    expected_sandbox_sha256 = hashlib.sha256(_sandbox_policy()).hexdigest()
    expected_public_key_sha256 = hashlib.sha256(_PUBLIC_KEY).hexdigest()
    for binding in (plan.source_verifiers[0].hermetic_execution, plan.adapter.hermetic_execution):
        assert binding is not None
        assert binding.sandbox_policy_sha256 == expected_sandbox_sha256
        assert binding.trusted_public_key_sha256 == expected_public_key_sha256
        assert binding.seccomp_profile_sha256 == _SECCOMP_SHA256
        assert binding.authority_id == 'independent-hermetic-runner'
        assert binding.signing_key_id == 'runner-key-2026-07'


def test_changing_hermetic_policy_changes_pre_capture_plan_identity() -> None:
    sources, adapter = _specs(wall_seconds=30)
    original = derive_pre_capture_plan(
        scope_policy=_scope(),
        selection_policy=_selection_policy(),
        capture_witness_policy=_witness_policy(),
        source_verifiers=sources,
        adapter=adapter,
    )
    changed_sources, changed_adapter = _specs(wall_seconds=31)
    changed = derive_pre_capture_plan(
        scope_policy=_scope(),
        selection_policy=_selection_policy(),
        capture_witness_policy=_witness_policy(),
        source_verifiers=changed_sources,
        adapter=changed_adapter,
    )

    assert (
        hashlib.sha256(canonical_json_bytes(original)).hexdigest()
        != hashlib.sha256(canonical_json_bytes(changed)).hexdigest()
    )
