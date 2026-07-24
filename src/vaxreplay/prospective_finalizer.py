"""Trusted post-outcome join for prospectively sealed benchmark episodes.

The public, pre-outcome side of a prospective episode binds both its model-visible
``decision_snapshot_sha256`` and the lineage-bearing ``decision_context_sha256``
derived from the exact source captures.  The deterministic evaluator, by
contrast, accepts a ``Submission`` bound to the final (and therefore
outcome-committing) episode manifest.  This module provides the deliberately
small join between those identities, but it does not expose a scoring entrypoint:
the returned schemas are ordinary caller-constructible values, not proof that
their receipt and policy inputs were verified.

No outcome-derived value is copied into a model response.  Finalization first
revalidates the complete Tier A temporal chain and then the adapter adds only
the final manifest hash required by the deterministic evaluator.  Official
scoring is available only through ``prospective_cohort_finalizer``, which reloads
and reverifies the complete persisted proof chain before returning a score.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseSelectionAudit,
    CaseSelectionDisposition,
    case_selection_audit_sha256,
)
from vaxreplay.case_schema import Submission
from vaxreplay.prospective_schema import (
    ProspectiveChallengeAdmission,
    ProspectiveEpisodeBinding,
    ProspectiveFinalizationBinding,
    ProspectiveSubmission,
    prospective_challenge_admission_sha256,
)
from vaxreplay.temporal_schema import (
    DecisionTimeConfig,
    TemporalAdmissionEnvelope,
    TemporalAdmissionError,
    TemporalReceiptVerifier,
    model_sha256,
    require_official_temporal_admission,
)

__all__ = [
    'ProspectiveFinalizationError',
    'adapt_prospective_submission',
    'finalize_prospective_episode',
]


class ProspectiveFinalizationError(ValueError):
    """Raised when a pre-outcome challenge cannot be joined to final labels."""


def finalize_prospective_episode(
    prospective_admission: ProspectiveChallengeAdmission,
    prospective_episode: ProspectiveEpisodeBinding,
    final_bundle: EpisodeBundle,
    temporal_admission: TemporalAdmissionEnvelope,
    *,
    receipt_artifacts: Mapping[str, bytes],
    receipt_verifier: TemporalReceiptVerifier,
    protocol_artifacts: Mapping[str, bytes],
    raw_outcome_source: bytes,
    label_derivation_audit: bytes,
    case_selection_audit: CaseSelectionAudit,
    finalized_at: datetime,
    case_selection_audit_commitment: str | None = None,
) -> ProspectiveFinalizationBinding:
    """Verify and bind one sealed decision to its later private scoring episode.

    ``case_selection_audit_commitment`` is optional because the returned
    finalization binding itself commits the canonical audit.  Callers that
    separately publish or seal the audit can pass its expected hash here to
    require an exact match as part of the join.
    """

    prospective_admission = _canonical_prospective_admission(prospective_admission)
    prospective_episode = _canonical_prospective_episode(prospective_episode)
    case_selection_audit = _canonical_case_selection_audit(case_selection_audit)

    admitted_episode = next(
        (episode for episode in prospective_admission.episodes if episode.episode_id == prospective_episode.episode_id),
        None,
    )
    if admitted_episode != prospective_episode:
        raise ProspectiveFinalizationError(
            'prospective episode is absent from, or differs from, the sealed challenge admission'
        )

    try:
        verified_temporal_admission = require_official_temporal_admission(
            temporal_admission,
            final_bundle,
            receipt_artifacts=receipt_artifacts,
            receipt_verifier=receipt_verifier,
            protocol_artifacts=protocol_artifacts,
            raw_outcome_source=raw_outcome_source,
            label_derivation_audit=label_derivation_audit,
            expected_decision_context_sha256=prospective_episode.decision_context_sha256,
            expected_decision_context_bytes=prospective_episode.decision_context_bytes,
        )
    except TemporalAdmissionError as error:
        raise ProspectiveFinalizationError(f'official temporal admission failed: {error}') from error

    final_decision = verified_temporal_admission.decision_snapshot
    final_decision_sha256 = model_sha256(final_decision)
    if (
        final_decision != prospective_episode.decision_snapshot
        or final_decision_sha256 != prospective_episode.decision_snapshot_sha256
    ):
        raise ProspectiveFinalizationError(
            'final decision snapshot changed the prospectively sealed decision or protocol'
        )

    first_label_available_at = verified_temporal_admission.outcome_snapshot.first_label_available_at
    if first_label_available_at <= prospective_admission.run_deadline_at:
        raise ProspectiveFinalizationError('first label availability must be after the prospective run deadline')
    if finalized_at.tzinfo is None or finalized_at.utcoffset() is None:
        raise ProspectiveFinalizationError('finalized_at must include a UTC offset')
    finalized_at = finalized_at.astimezone(timezone.utc)
    if finalized_at < verified_temporal_admission.admitted_at:
        raise ProspectiveFinalizationError('finalized_at cannot precede the verified temporal admission')

    audit_sha256 = case_selection_audit_sha256(case_selection_audit)
    if case_selection_audit_commitment is not None and case_selection_audit_commitment != audit_sha256:
        raise ProspectiveFinalizationError('case-selection audit does not match its supplied commitment')
    _require_case_selection_binding(
        case_selection_audit,
        prospective_admission=prospective_admission,
        prospective_episode=prospective_episode,
        final_bundle=final_bundle,
    )

    try:
        result = ProspectiveFinalizationBinding(
            release_id=prospective_admission.release_id,
            purpose=prospective_admission.purpose,
            episode_id=prospective_episode.episode_id,
            prospective_admission_sha256=prospective_challenge_admission_sha256(prospective_admission),
            decision_snapshot_sha256=prospective_episode.decision_snapshot_sha256,
            decision_context_sha256=prospective_episode.decision_context_sha256,
            temporal_admission_sha256=model_sha256(verified_temporal_admission),
            case_selection_audit_sha256=audit_sha256,
            finalized_at=finalized_at,
        )
        return result.require_episode(prospective_admission, prospective_episode)
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid prospective finalization: {error}') from error


def adapt_prospective_submission(
    submission: ProspectiveSubmission,
    *,
    prospective_admission: ProspectiveChallengeAdmission,
    prospective_episode: ProspectiveEpisodeBinding,
    final_bundle: EpisodeBundle,
    finalization: ProspectiveFinalizationBinding,
) -> Submission:
    """Add only the final manifest identity to a validated prospective response."""

    prospective_admission = _canonical_prospective_admission(prospective_admission)
    prospective_episode = _canonical_prospective_episode(prospective_episode)
    submission = _canonical_prospective_submission(submission)
    finalization = _canonical_finalization(finalization)

    try:
        finalization.require_episode(prospective_admission, prospective_episode)
        submission.require_episode(prospective_episode)
        final_bundle.validate_integrity()
    except ValueError as error:
        raise ProspectiveFinalizationError(f'prospective response cannot be adapted: {error}') from error

    # The full decision/protocol equality is enforced when ``finalization`` is
    # minted.  Rechecking the manifest-derived config here prevents accidentally
    # adapting the response against a different in-memory final bundle.
    if DecisionTimeConfig.from_manifest(final_bundle.manifest) != prospective_episode.decision_snapshot.config:
        raise ProspectiveFinalizationError(
            'final scoring bundle does not match the prospectively sealed decision config'
        )

    return Submission(
        episode_id=submission.episode_id,
        manifest_sha256=final_bundle.manifest_sha256,
        ranking=list(submission.ranking),
        forecasts=list(submission.forecasts),
        assessments=list(submission.assessments),
    )


def _require_case_selection_binding(
    audit: CaseSelectionAudit,
    *,
    prospective_admission: ProspectiveChallengeAdmission,
    prospective_episode: ProspectiveEpisodeBinding,
    final_bundle: EpisodeBundle,
) -> None:
    if audit.case_universe_sha256 != prospective_admission.case_universe_sha256:
        raise ProspectiveFinalizationError('case-selection audit does not bind the prospective case universe')

    records = tuple(record for record in audit.records if record.episode_id == prospective_episode.episode_id)
    if len(records) != 1:
        raise ProspectiveFinalizationError(
            'case-selection audit must contain exactly one admitted record for the episode'
        )
    record = records[0]
    if record.disposition != CaseSelectionDisposition.ADMITTED:
        raise ProspectiveFinalizationError('prospective episode is not admitted by the case-selection audit')
    if record.manifest_sha256 != final_bundle.manifest_sha256:
        raise ProspectiveFinalizationError('case-selection audit does not bind the final episode manifest')
    candidate_count = len(prospective_episode.decision_snapshot.config.candidate_ids)
    if record.panel_count != candidate_count or record.observed_count != candidate_count:
        raise ProspectiveFinalizationError('case-selection admitted panel does not match the sealed candidate panel')


def _canonical_prospective_admission(
    admission: ProspectiveChallengeAdmission,
) -> ProspectiveChallengeAdmission:
    try:
        return ProspectiveChallengeAdmission.model_validate_json(canonical_json_bytes(admission))
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid prospective challenge admission: {error}') from error


def _canonical_prospective_episode(
    episode: ProspectiveEpisodeBinding,
) -> ProspectiveEpisodeBinding:
    try:
        return ProspectiveEpisodeBinding.model_validate_json(canonical_json_bytes(episode))
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid prospective episode binding: {error}') from error


def _canonical_prospective_submission(submission: ProspectiveSubmission) -> ProspectiveSubmission:
    try:
        return ProspectiveSubmission.model_validate_json(canonical_json_bytes(submission))
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid prospective submission: {error}') from error


def _canonical_finalization(
    finalization: ProspectiveFinalizationBinding,
) -> ProspectiveFinalizationBinding:
    try:
        return ProspectiveFinalizationBinding.model_validate_json(canonical_json_bytes(finalization))
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid prospective finalization binding: {error}') from error


def _canonical_case_selection_audit(audit: CaseSelectionAudit) -> CaseSelectionAudit:
    try:
        return CaseSelectionAudit.model_validate_json(canonical_json_bytes(audit))
    except ValueError as error:
        raise ProspectiveFinalizationError(f'invalid case-selection audit: {error}') from error
