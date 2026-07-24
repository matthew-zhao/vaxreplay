from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_operations_release_decision import _decision
from tests.test_prospective_release import (
    _ATTEMPT_POLICY,
    _CASE_PROOF,
    _ELIGIBILITY,
    _SOURCE_CAPTURE_POLICY,
    _VERIFIER_POLICY,
    _admission,
    _build,
    _case_verifier,
    _decision_verifier,
    _source_capture_verifier,
)
from vaxreplay.operations.prospective_campaign_archive import build_prospective_campaign_archive
from vaxreplay.operations.prospective_release_approval import (
    TierAProspectiveReleaseApprovalError,
    TierAProspectiveReleaseApprovalReport,
    verify_and_materialize_tier_a_prospective_release,
)
from vaxreplay.operations.release_readiness import TierAReleaseScope
from vaxreplay.prospective_release import build_prospective_cohort_release

_OFFICIAL_SCOPE = TierAReleaseScope(
    sources=('immport',),
    tasks=('preclinical_candidate_advancement',),
    includes_model_leaderboard=True,
)


@contextmanager
def _official_archive(root: Path):
    """Test-only official artifact; promotion-factory invariants are tested separately."""

    import vaxreplay.prospective_release as release_module

    real_builder = release_module.build_verified_prospective_admission
    research = _admission(root / 'inputs')

    def force_official(*args, **kwargs):
        rebuilt = real_builder(*args, **kwargs)
        return replace(
            rebuilt,
            admission=rebuilt.admission.model_copy(update={'purpose': 'official_benchmark'}),
        )

    official = replace(
        research,
        admission=research.admission.model_copy(update={'purpose': 'official_benchmark'}),
    )
    with (
        patch(
            'vaxreplay.prospective_release.build_verified_prospective_admission',
            side_effect=force_official,
        ),
        patch(
            'vaxreplay.operations.prospective_campaign_archive._derive_official_release_sources',
            return_value=('immport',),
        ),
    ):
        release = build_prospective_cohort_release(
            root / 'release',
            challenge_id='prospective-challenge-1',
            verified_admission=official,
            case_universe_proof=_CASE_PROOF,
            eligibility_protocol=_ELIGIBILITY,
            verifier_policy=_VERIFIER_POLICY,
            source_capture_policy=_SOURCE_CAPTURE_POLICY,
            attempt_policy=_ATTEMPT_POLICY,
            decision_receipt_verifier=_decision_verifier,
            case_universe_seal_verifier=_case_verifier,
            source_capture_verifier=_source_capture_verifier,
        )
        yield (
            release,
            build_prospective_campaign_archive(
                release.root,
                release_scope=_OFFICIAL_SCOPE,
            ),
        )


def _approval_arguments(
    tmp_path: Path,
    release,
    archive,
    *,
    readiness_scope: TierAReleaseScope | None = None,
) -> dict[str, object]:
    return {
        **_decision(
            tmp_path / 'decision',
            campaign_release_id=release.manifest.release_id,
            release_archive_bytes=archive.archive_bytes,
            release_archive_index_bytes=archive.index_bytes,
            readiness_scope=readiness_scope or archive.index.release_scope,
        ),
        'materialized_release_dir': tmp_path / 'materialized-release',
        'decision_receipt_verifier': _decision_verifier,
        'case_universe_seal_verifier': _case_verifier,
        'source_capture_verifier': _source_capture_verifier,
    }


def test_authenticated_decision_materializes_the_exact_official_release(tmp_path: Path) -> None:
    with _official_archive(tmp_path / 'official') as (release, archive):
        approved = verify_and_materialize_tier_a_prospective_release(**_approval_arguments(tmp_path, release, archive))

    assert approved.archive.release.release_sha256 == release.release_sha256
    assert approved.archive.index == archive.index
    assert approved.report.release_id == release.manifest.release_id
    assert approved.report.release_purpose == 'official_benchmark'
    assert approved.report.prospective_release_sha256 == release.release_sha256
    assert approved.report.release_scope == _OFFICIAL_SCOPE
    assert approved.report.release_scope_sha256 == archive.index.release_scope_sha256
    assert approved.report.prospective_release_semantics_reverified
    assert approved.report.official_benchmark_purpose_verified
    assert not approved.report.deployment_facts_independently_observed_by_this_verifier
    assert not approved.report.organizational_independence_cryptographically_proven
    assert (tmp_path / 'materialized-release' / 'release.json').is_file()

    with pytest.raises(ValueError, match='schema_version'):
        TierAProspectiveReleaseApprovalReport.model_validate(
            {
                **approved.report.model_dump(mode='json'),
                'schema_version': 'vaxreplay.tier-a-prospective-release-approval-report.v0.1',
            }
        )


def test_opaque_campaign_archive_can_no_longer_complete_approval(tmp_path: Path) -> None:
    output = tmp_path / 'materialized-release'
    with pytest.raises(TierAProspectiveReleaseApprovalError, match='not one valid official'):
        verify_and_materialize_tier_a_prospective_release(
            **_decision(tmp_path / 'decision'),
            materialized_release_dir=output,
            decision_receipt_verifier=_decision_verifier,
            case_universe_seal_verifier=_case_verifier,
            source_capture_verifier=_source_capture_verifier,
        )
    assert not output.exists()


def test_research_release_is_rejected_before_materialization(tmp_path: Path) -> None:
    release = _build(tmp_path / 'research')
    archive = build_prospective_campaign_archive(
        release.root,
        release_scope=_OFFICIAL_SCOPE,
    )
    output = tmp_path / 'materialized-release'
    arguments = _approval_arguments(tmp_path, release, archive)
    arguments['materialized_release_dir'] = output

    with pytest.raises(TierAProspectiveReleaseApprovalError, match='not one valid official'):
        verify_and_materialize_tier_a_prospective_release(**arguments)
    assert not output.exists()


def test_semantic_verifier_rejection_leaves_no_release(tmp_path: Path) -> None:
    with _official_archive(tmp_path / 'official') as (release, archive):
        arguments = _approval_arguments(tmp_path, release, archive)
        arguments['decision_receipt_verifier'] = lambda *_args: False
        with pytest.raises(TierAProspectiveReleaseApprovalError, match='not one valid official'):
            verify_and_materialize_tier_a_prospective_release(**arguments)
    assert not (tmp_path / 'materialized-release').exists()


@pytest.mark.parametrize(
    'readiness_scope',
    (
        TierAReleaseScope(
            sources=('iedb',),
            tasks=('preclinical_candidate_advancement',),
            includes_model_leaderboard=True,
        ),
        TierAReleaseScope(
            sources=('immport',),
            tasks=('antigen_target_prioritization',),
            includes_model_leaderboard=True,
        ),
        TierAReleaseScope(
            sources=('immport',),
            tasks=('preclinical_candidate_advancement',),
            includes_model_leaderboard=False,
        ),
    ),
)
def test_readiness_cannot_underdeclare_the_campaign_release_scope(
    tmp_path: Path,
    readiness_scope: TierAReleaseScope,
) -> None:
    output = tmp_path / 'materialized-release'
    with _official_archive(tmp_path / 'official') as (release, archive):
        arguments = _approval_arguments(
            tmp_path,
            release,
            archive,
            readiness_scope=readiness_scope,
        )
        with pytest.raises(TierAProspectiveReleaseApprovalError, match='not one valid official'):
            verify_and_materialize_tier_a_prospective_release(**arguments)
    assert not output.exists()


def test_composite_does_not_accept_a_serialized_decision_report() -> None:
    import inspect

    parameters = inspect.signature(verify_and_materialize_tier_a_prospective_release).parameters
    assert 'decision_report' not in parameters
    assert 'readiness_report' not in parameters
    assert 'expected_campaign_trust_policy_sha256' in parameters
    assert 'expected_readiness_policy_sha256' in parameters
