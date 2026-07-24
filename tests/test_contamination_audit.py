from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.contamination import (
    ArtifactContaminationAudit,
    AuditDisposition,
    AuditReasonCode,
    CalibrationPolicy,
    ContaminationAuditError,
    ContaminationAuditPolicy,
    ContaminationFinding,
    ExactByteSpan,
    ExactRetrievalConfig,
    FindingKind,
    FindingSeverity,
    IdentifierNeedle,
    JudgeCalibrationResult,
    JudgeVerdict,
    LlmJudgeOutput,
    PinnedLlmJudge,
    RetrievalKind,
    artifact_binding,
    audit_manifest_sha256,
    build_contamination_audit,
    make_audit_input,
    make_audit_manifest,
    make_llm_audit_run,
    model_sha256,
    retrieve_exact_candidates,
    validate_exact_byte_span,
    verify_contamination_audit,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _judge(judge_id: str) -> PinnedLlmJudge:
    return PinnedLlmJudge(
        judge_id=judge_id,
        provider=f'provider-{judge_id}',
        model_id=f'model-{judge_id}',
        model_revision='2026-07-01',
        system_fingerprint=f'fingerprint-{judge_id}',
        system_manifest_sha256=_sha(f'system-{judge_id}'.encode()),
        prompt_sha256=_sha(f'prompt-{judge_id}'.encode()),
        config_sha256=_sha(f'config-{judge_id}'.encode()),
    )


def _policy(*, maximum_candidates: int = 100) -> ContaminationAuditPolicy:
    return ContaminationAuditPolicy(
        policy_id='retrospective-screen-v1',
        retrieval=ExactRetrievalConfig(
            ngram_tokens=4,
            minimum_ngram_bytes=12,
            maximum_candidates=maximum_candidates,
        ),
        calibration=CalibrationPolicy(
            minimum_canary_count=4,
            minimum_negative_control_count=4,
            minimum_canary_recall=0.75,
            maximum_false_positive_rate=0.25,
        ),
        judges=(_judge('judge-alpha'), _judge('judge-beta')),
    )


def _calibration(*, detected: int = 4, false_positives: int = 0) -> JudgeCalibrationResult:
    return JudgeCalibrationResult(
        canary_count=4,
        canary_detected_count=detected,
        negative_control_count=4,
        false_positive_count=false_positives,
    )


def _span(artifact_id: str, payload: bytes, quote: str) -> ExactByteSpan:
    encoded = quote.encode('utf-8')
    start = payload.index(encoded)
    return ExactByteSpan(
        artifact_id=artifact_id,
        artifact_sha256=_sha(payload),
        start_byte=start,
        end_byte=start + len(encoded),
        quote=quote,
    )


def _input(case_id: str, public: bytes, comparisons: dict[str, bytes]):
    return make_audit_input(
        case_id=case_id,
        episode_id=f'episode-{case_id}',
        decision_package_sha256=_sha(f'decision-{case_id}'.encode()),
        episode_manifest_sha256=_sha(f'manifest-{case_id}'.encode()),
        public_artifact_id='public-envelope',
        public_payload=public,
        comparison_payloads=comparisons,
    )


def _runs(audit_input, policy, outputs: tuple[LlmJudgeOutput, LlmJudgeOutput]):
    return tuple(
        make_llm_audit_run(
            run_id=f'run-{judge.judge_id}',
            judge=judge,
            audit_input=audit_input,
            output=output,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=index + 1),
        )
        for index, (judge, output) in enumerate(zip(policy.judges, outputs, strict=True))
    )


def _clear_output(*, calibration: JudgeCalibrationResult | None = None) -> LlmJudgeOutput:
    return LlmJudgeOutput(
        verdict=JudgeVerdict.CLEAR,
        calibration=calibration or _calibration(),
    )


class ContaminationSchemaTest(unittest.TestCase):
    def test_requires_two_distinct_sorted_pinned_judges(self) -> None:
        with self.assertRaises(ValidationError):
            ContaminationAuditPolicy(policy_id='bad', judges=(_judge('only-one'),))
        with self.assertRaisesRegex(ValidationError, 'sorted order'):
            ContaminationAuditPolicy(
                policy_id='bad-order',
                judges=(_judge('judge-beta'), _judge('judge-alpha')),
            )

    def test_judge_output_and_run_are_hash_bound(self) -> None:
        public = b'Candidate A has ordinary pre-cutoff evidence.'
        comparisons = {'future': b'Candidate B later validated.'}
        audit_input = _input('case-hash', public, comparisons)
        policy = _policy()
        run = _runs(audit_input, policy, (_clear_output(), _clear_output()))[0]

        forged = run.model_dump(mode='python')
        forged['output_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValidationError, 'canonical hash'):
            type(run).model_validate(forged)

    def test_exact_spans_use_utf8_byte_offsets_and_fail_closed(self) -> None:
        payload = 'Evidence about β-cell recognition.'.encode()
        binding = artifact_binding('unicode-evidence', payload)
        span = _span('unicode-evidence', payload, 'β-cell')
        validate_exact_byte_span(payload, binding, span)

        with self.assertRaisesRegex(ContaminationAuditError, 'hash and byte-count'):
            validate_exact_byte_span(payload.replace(b'cell', b'CELL'), binding, span)


class DeterministicRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = b'Trial NCT12345678 reports the exact future validation signal strongly positive.'
        self.comparisons = {
            'future-outcome': (b'Outcome for NCT12345678: the exact future validation signal strongly positive.')
        }
        self.audit_input = _input('case-retrieval', self.public, self.comparisons)
        self.policy = _policy()
        self.identifiers = (
            IdentifierNeedle(
                identifier_id='trial-registration',
                identifier_type='clinical_trial_registration',
                value='NCT12345678',
                reference_artifact_id='future-outcome',
            ),
        )

    def test_retrieval_is_literal_deterministic_and_exact_span_grounded(self) -> None:
        first = retrieve_exact_candidates(
            self.audit_input,
            public_payload=self.public,
            comparison_payloads=self.comparisons,
            identifiers=self.identifiers,
            policy=self.policy,
        )
        second = retrieve_exact_candidates(
            self.audit_input,
            public_payload=self.public,
            comparison_payloads=self.comparisons,
            identifiers=reversed(self.identifiers),
            policy=self.policy,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {candidate.kind for candidate in first},
            {RetrievalKind.EXACT_NGRAM, RetrievalKind.IDENTIFIER},
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in first),
            tuple(sorted(candidate.candidate_id for candidate in first)),
        )
        for candidate in first:
            validate_exact_byte_span(self.public, self.audit_input.public_artifact, candidate.public_span)
            validate_exact_byte_span(
                self.comparisons['future-outcome'],
                self.audit_input.comparison_artifacts[0],
                candidate.reference_span,
            )

    def test_retrieval_refuses_tampered_inputs_and_signal_truncation(self) -> None:
        with self.assertRaisesRegex(ContaminationAuditError, 'hash and byte-count'):
            retrieve_exact_candidates(
                self.audit_input,
                public_payload=self.public + b'tamper',
                comparison_payloads=self.comparisons,
                identifiers=self.identifiers,
                policy=self.policy,
            )
        with self.assertRaisesRegex(ContaminationAuditError, 'refusing to truncate'):
            retrieve_exact_candidates(
                self.audit_input,
                public_payload=self.public,
                comparison_payloads=self.comparisons,
                identifiers=self.identifiers,
                policy=_policy(maximum_candidates=1),
            )


class FixedAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = b'Candidate A has ordinary pre-cutoff cellular evidence.'
        self.comparisons = {'future-outcome': b'Candidate Z later had a favorable result.'}
        self.audit_input = _input('case-clean', self.public, self.comparisons)
        self.policy = _policy()

    def _build(self, outputs, **kwargs):
        return build_contamination_audit(
            audit_id='audit-clean',
            audit_input=self.audit_input,
            policy=self.policy,
            public_payload=self.public,
            comparison_payloads=self.comparisons,
            judge_runs=_runs(self.audit_input, self.policy, outputs),
            screened_at=NOW,
            **kwargs,
        )

    def test_pass_means_only_no_signal_detected_and_retains_residual_risk(self) -> None:
        audit = self._build((_clear_output(), _clear_output()))

        self.assertEqual(audit.disposition, AuditDisposition.PASS)
        self.assertEqual(audit.reason_codes, (AuditReasonCode.NO_DETECTED_SIGNALS,))
        self.assertTrue(audit.residual_contamination_possible)
        self.assertFalse(audit.proves_absence_of_contamination)
        self.assertEqual(
            ArtifactContaminationAudit.model_validate_json(canonical_json_bytes(audit)),
            audit,
        )
        verify_contamination_audit(
            audit,
            policy=self.policy,
            public_payload=self.public,
            comparison_payloads=self.comparisons,
        )

    def test_suspicion_or_judge_disagreement_requires_manual_review(self) -> None:
        finding = ContaminationFinding(
            finding_id='finding-source-clue',
            kind=FindingKind.SOURCE_REIDENTIFICATION,
            severity=FindingSeverity.LOW,
            public_span=_span('public-envelope', self.public, 'ordinary pre-cutoff'),
            detail='Distinctive wording may permit source re-identification.',
        )
        suspicious = LlmJudgeOutput(
            verdict=JudgeVerdict.SUSPICIOUS,
            findings=(finding,),
            calibration=_calibration(),
        )
        audit = self._build((suspicious, _clear_output()))

        self.assertEqual(audit.disposition, AuditDisposition.MANUAL_REVIEW)
        self.assertIn(AuditReasonCode.JUDGE_DISAGREEMENT, audit.reason_codes)
        self.assertIn(AuditReasonCode.JUDGE_FINDING, audit.reason_codes)

    def test_failed_canary_calibration_cannot_silently_pass(self) -> None:
        weak = _clear_output(calibration=_calibration(detected=1))
        audit = self._build((weak, _clear_output()))

        self.assertEqual(audit.disposition, AuditDisposition.MANUAL_REVIEW)
        self.assertIn(AuditReasonCode.CALIBRATION_BELOW_THRESHOLD, audit.reason_codes)

    def test_high_severity_finding_is_quarantined(self) -> None:
        public_quote = 'ordinary pre-cutoff'
        reference_quote = 'favorable result'
        finding = ContaminationFinding(
            finding_id='finding-outcome',
            kind=FindingKind.OUTCOME_DISCLOSURE,
            severity=FindingSeverity.HIGH,
            public_span=_span('public-envelope', self.public, public_quote),
            reference_span=_span('future-outcome', self.comparisons['future-outcome'], reference_quote),
            detail='The public phrase is judged to disclose the later outcome.',
        )
        contaminated = LlmJudgeOutput(
            verdict=JudgeVerdict.CONTAMINATED,
            findings=(finding,),
            calibration=_calibration(),
        )
        audit = self._build((contaminated, _clear_output()))

        self.assertEqual(audit.disposition, AuditDisposition.QUARANTINE)
        self.assertIn(AuditReasonCode.HIGH_SEVERITY_FINDING, audit.reason_codes)

    def test_identifier_match_is_quarantined_even_when_judges_say_clear(self) -> None:
        public = b'Candidate from NCT12345678 has pre-cutoff evidence.'
        comparisons = {'future-outcome': b'NCT12345678 later posted results.'}
        audit_input = _input('case-identifier', public, comparisons)
        identifier = IdentifierNeedle(
            identifier_id='trial-id',
            identifier_type='clinical_trial_registration',
            value='NCT12345678',
            reference_artifact_id='future-outcome',
        )
        audit = build_contamination_audit(
            audit_id='audit-identifier',
            audit_input=audit_input,
            policy=self.policy,
            public_payload=public,
            comparison_payloads=comparisons,
            judge_runs=_runs(audit_input, self.policy, (_clear_output(), _clear_output())),
            screened_at=NOW,
            identifiers=(identifier,),
        )

        self.assertEqual(audit.disposition, AuditDisposition.QUARANTINE)
        self.assertIn(AuditReasonCode.IDENTIFIER_MATCH, audit.reason_codes)

    def test_missing_or_unpinned_judge_run_fails_closed(self) -> None:
        one_run = _runs(self.audit_input, self.policy, (_clear_output(), _clear_output()))[:1]
        with self.assertRaisesRegex(ContaminationAuditError, 'exactly one run'):
            build_contamination_audit(
                audit_id='audit-missing-run',
                audit_input=self.audit_input,
                policy=self.policy,
                public_payload=self.public,
                comparison_payloads=self.comparisons,
                judge_runs=one_run,
                screened_at=NOW,
            )

    def test_complete_manifest_is_sorted_and_hash_bound(self) -> None:
        audit_b = self._build((_clear_output(), _clear_output()))
        input_a = _input('case-a', self.public, self.comparisons)
        audit_a = build_contamination_audit(
            audit_id='audit-a',
            audit_input=input_a,
            policy=self.policy,
            public_payload=self.public,
            comparison_payloads=self.comparisons,
            judge_runs=_runs(input_a, self.policy, (_clear_output(), _clear_output())),
            screened_at=NOW,
        )
        manifest = make_audit_manifest(
            manifest_id='manifest-v1',
            case_universe_sha256='f' * 64,
            policy=self.policy,
            audits=(audit_b, audit_a),
        )

        self.assertEqual(tuple(audit.audit_input.case_id for audit in manifest.audits), ('case-a', 'case-clean'))
        self.assertEqual(manifest.policy_sha256, model_sha256(self.policy))
        self.assertEqual(len(audit_manifest_sha256(manifest)), 64)


if __name__ == '__main__':
    unittest.main()
