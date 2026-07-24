from __future__ import annotations

import hashlib
import json
import unittest

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.contamination import (
    CalibrationBooleanResult,
    FindingKind,
    FindingSeverity,
    JudgeReviewAdapterError,
    JudgeVerdict,
    LlmJudgeQuoteReview,
    QuoteFindingReview,
    adapt_quote_review,
    build_quote_review_prompt,
)

PUBLIC_ID = 'public-decision-package'
CANARY_IDS = ('ctl-a1f9c43e872b605d', 'ctl-74be12d980ac36f1')
NEGATIVE_IDS = ('ctl-c8350d14ef729ab6', 'ctl-19d7fa40b2e86c53')
EXTRA_CONTROL_ID = 'ctl-e6a1d9057cf2438b'


def _finding(
    *,
    kind: FindingKind = FindingKind.OUTCOME_DISCLOSURE,
    severity: FindingSeverity = FindingSeverity.HIGH,
    public_quote: str = 'β response later ranked first',
    reference_artifact_id: str | None = 'future-results',
    reference_quote: str | None = 'β response ranked first',
    detail: str = 'The public artifact states a protected post-cutoff ranking as fact.',
) -> QuoteFindingReview:
    return QuoteFindingReview(
        kind=kind,
        severity=severity,
        public_quote=public_quote,
        reference_artifact_id=reference_artifact_id,
        reference_quote=reference_quote,
        detail=detail,
    )


def _review(
    *,
    verdict: JudgeVerdict = JudgeVerdict.CONTAMINATED,
    findings: tuple[QuoteFindingReview, ...] | None = None,
    source_guess: str | None = 'Possible protected trial report',
) -> LlmJudgeQuoteReview:
    return LlmJudgeQuoteReview(
        verdict=verdict,
        findings=findings if findings is not None else (_finding(),),
        control_results=(
            CalibrationBooleanResult(control_id=CANARY_IDS[0], detected=True),
            CalibrationBooleanResult(control_id=NEGATIVE_IDS[0], detected=False),
            CalibrationBooleanResult(control_id=CANARY_IDS[1], detected=False),
            CalibrationBooleanResult(control_id=NEGATIVE_IDS[1], detected=True),
        ),
        source_guess=source_guess,
    )


def _adapt(
    review: LlmJudgeQuoteReview,
    *,
    public: bytes = 'Résumé: β response later ranked first.'.encode(),
    comparisons: dict[str, bytes] | None = None,
):
    return adapt_quote_review(
        review,
        public_artifact_id=PUBLIC_ID,
        public_payload=public,
        comparison_payloads=comparisons
        if comparisons is not None
        else {'future-results': 'The protected result says β response ranked first.'.encode()},
        canary_control_ids=CANARY_IDS,
        negative_control_ids=NEGATIVE_IDS,
    )


class QuoteReviewSchemaTest(unittest.TestCase):
    def test_practical_json_parses_strictly_and_rejects_extra_fields(self) -> None:
        payload = {
            'verdict': 'clear',
            'findings': [],
            'control_results': [
                {'control_id': CANARY_IDS[0], 'detected': True},
                {'control_id': CANARY_IDS[1], 'detected': True},
                {'control_id': NEGATIVE_IDS[0], 'detected': False},
                {'control_id': NEGATIVE_IDS[1], 'detected': False},
            ],
            'source_guess': None,
        }
        parsed = LlmJudgeQuoteReview.model_validate_json(json.dumps(payload))
        self.assertEqual(parsed.verdict, JudgeVerdict.CLEAR)

        payload['provider_reasoning'] = 'not retained'
        with self.assertRaisesRegex(ValidationError, 'Extra inputs are not permitted'):
            LlmJudgeQuoteReview.model_validate_json(json.dumps(payload))

    def test_finding_reference_fields_are_paired_and_required_for_future_facts(self) -> None:
        with self.assertRaisesRegex(ValidationError, 'provided together'):
            _finding(reference_quote=None)
        with self.assertRaisesRegex(ValidationError, 'require an exact reference quote'):
            _finding(reference_artifact_id=None, reference_quote=None)

        intrinsic = _finding(
            kind=FindingKind.SOURCE_REIDENTIFICATION,
            severity=FindingSeverity.MEDIUM,
            reference_artifact_id=None,
            reference_quote=None,
        )
        self.assertIsNone(intrinsic.reference_quote)

    def test_verdict_rules_and_calibration_ids_are_strict(self) -> None:
        with self.assertRaisesRegex(ValidationError, 'clear review cannot contain findings'):
            _review(verdict=JudgeVerdict.CLEAR)
        with self.assertRaisesRegex(ValidationError, 'non-clear review requires'):
            _review(verdict=JudgeVerdict.SUSPICIOUS, findings=())
        with self.assertRaisesRegex(ValidationError, 'requires a high or critical'):
            _review(
                verdict=JudgeVerdict.CONTAMINATED,
                findings=(_finding(severity=FindingSeverity.MEDIUM),),
            )

        with self.assertRaisesRegex(ValidationError, 'control result IDs must be unique'):
            LlmJudgeQuoteReview(
                verdict=JudgeVerdict.CLEAR,
                control_results=(
                    CalibrationBooleanResult(control_id=CANARY_IDS[0], detected=True),
                    CalibrationBooleanResult(control_id=CANARY_IDS[0], detected=False),
                ),
            )

        with self.assertRaisesRegex(ValidationError, 'String should match pattern'):
            CalibrationBooleanResult(control_id='canary-future-result', detected=True)


class QuoteReviewAdaptationTest(unittest.TestCase):
    def test_unicode_quotes_become_hash_bound_byte_spans_and_counts(self) -> None:
        public = 'Résumé: β response later ranked first.'.encode()
        reference = 'The protected result says β response ranked first.'.encode()
        output = _adapt(_review(), public=public, comparisons={'future-results': reference})

        self.assertEqual(output.verdict, JudgeVerdict.CONTAMINATED)
        finding = output.findings[0]
        public_quote = 'β response later ranked first'.encode()
        reference_quote = 'β response ranked first'.encode()
        self.assertEqual(finding.public_span.start_byte, public.index(public_quote))
        self.assertEqual(finding.public_span.end_byte, public.index(public_quote) + len(public_quote))
        self.assertEqual(finding.public_span.artifact_sha256, hashlib.sha256(public).hexdigest())
        assert finding.reference_span is not None
        self.assertEqual(finding.reference_span.start_byte, reference.index(reference_quote))
        self.assertEqual(finding.reference_span.artifact_sha256, hashlib.sha256(reference).hexdigest())
        self.assertEqual(output.calibration.canary_count, 2)
        self.assertEqual(output.calibration.canary_detected_count, 1)
        self.assertEqual(output.calibration.negative_control_count, 2)
        self.assertEqual(output.calibration.false_positive_count, 1)

    def test_source_guess_is_excluded_from_canonical_output(self) -> None:
        first = _adapt(_review(source_guess='Guess A'))
        second = _adapt(_review(source_guess='Completely different guess'))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertNotIn(b'source_guess', canonical_json_bytes(first))
        self.assertNotIn(b'Guess A', canonical_json_bytes(first))

    def test_findings_are_deterministically_identified_and_sorted(self) -> None:
        public = b'Unique protected identifier X. Unique future outcome Y.'
        references = {
            'future-results': b'Reference future outcome Y.',
            'source-map': b'Reference protected identifier X.',
        }
        future = _finding(
            public_quote='Unique future outcome Y',
            reference_quote='future outcome Y',
            detail='Future outcome disclosure.',
        )
        identifier = _finding(
            kind=FindingKind.SOURCE_IDENTIFIER,
            severity=FindingSeverity.HIGH,
            public_quote='protected identifier X',
            reference_artifact_id='source-map',
            reference_quote='protected identifier X',
            detail='Protected identifier disclosure.',
        )
        first = _adapt(_review(findings=(future, identifier)), public=public, comparisons=references)
        second = _adapt(_review(findings=(identifier, future)), public=public, comparisons=references)
        self.assertEqual(first, second)
        ids = tuple(finding.finding_id for finding in first.findings)
        self.assertEqual(ids, tuple(sorted(ids)))

    def test_absent_and_ambiguous_public_quotes_fail_closed(self) -> None:
        absent = _finding(public_quote='not in the public artifact')
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'quote is absent'):
            _adapt(_review(findings=(absent,)))

        ambiguous = _finding(public_quote='ana')
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'quote is ambiguous'):
            _adapt(_review(findings=(ambiguous,)), public=b'banana')

    def test_absent_ambiguous_and_unknown_reference_quotes_fail_closed(self) -> None:
        public = b'Only one public statement.'
        absent = _finding(public_quote='one public statement', reference_quote='missing result')
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'quote is absent'):
            _adapt(_review(findings=(absent,)), public=public)

        ambiguous = _finding(public_quote='one public statement', reference_quote='ana')
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'quote is ambiguous'):
            _adapt(
                _review(findings=(ambiguous,)),
                public=public,
                comparisons={'future-results': b'banana'},
            )

        unknown = _finding(
            public_quote='one public statement',
            reference_artifact_id='not-bound',
            reference_quote='some result',
        )
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'unknown comparison artifact'):
            _adapt(_review(findings=(unknown,)), public=public)

    def test_calibration_inventory_cannot_be_inflated_or_swapped(self) -> None:
        review = _review().model_copy(
            update={
                'control_results': (
                    CalibrationBooleanResult(control_id=CANARY_IDS[0], detected=True),
                    CalibrationBooleanResult(control_id=CANARY_IDS[1], detected=True),
                    CalibrationBooleanResult(control_id=NEGATIVE_IDS[0], detected=False),
                    CalibrationBooleanResult(control_id=EXTRA_CONTROL_ID, detected=False),
                )
            }
        )
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'control result inventory mismatch'):
            _adapt(review)

        duplicated = _review().model_copy(
            update={'control_results': (*_review().control_results, _review().control_results[0])}
        )
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'control result IDs must be unique'):
            _adapt(duplicated)

        with self.assertRaisesRegex(JudgeReviewAdapterError, 'must be disjoint'):
            adapt_quote_review(
                _review(),
                public_artifact_id=PUBLIC_ID,
                public_payload=b'public',
                comparison_payloads={'future-results': b'future'},
                canary_control_ids=('same',),
                negative_control_ids=('same',),
            )

    def test_all_bound_artifacts_must_be_nonempty_utf8_bytes(self) -> None:
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'not valid UTF-8'):
            _adapt(_review(), comparisons={'future-results': b'\xff'})
        with self.assertRaisesRegex(ValueError, 'non-empty bytes'):
            _adapt(_review(), comparisons={'future-results': b''})


class QuoteReviewPromptTest(unittest.TestCase):
    def test_prompt_is_deterministic_complete_and_rubric_specific(self) -> None:
        common = {
            'public_artifact_id': PUBLIC_ID,
            'public_payload': b'Pre-cutoff protocol with arm definitions.',
            'canary_controls': {
                CANARY_IDS[1]: b'future outcome',
                CANARY_IDS[0]: b'source identifier',
            },
            'negative_control_controls': {
                NEGATIVE_IDS[1]: b'endpoint definition',
                NEGATIVE_IDS[0]: b'plausible hypothesis',
            },
        }
        first = build_quote_review_prompt(
            comparison_payloads={'z-results': b'late result', 'a-results': b'early protected result'},
            **common,
        )
        second = build_quote_review_prompt(
            comparison_payloads={'a-results': b'early protected result', 'z-results': b'late result'},
            **common,
        )
        self.assertEqual(first, second)
        self.assertIn('Do not browse, call tools, perform outside lookup', first)
        self.assertIn('protocol structure is not contamination', first)
        self.assertIn('disclosed future facts', first)
        self.assertIn('source_reidentification', first)
        self.assertIn('Return one JSON object and no prose', first)
        self.assertIn('control_results', first)
        self.assertNotIn('control_type', first)
        self.assertNotIn('canary', first.casefold())
        self.assertNotIn('negative', first.casefold())

        material = json.loads(first.split('REVIEW_MATERIAL_JSON\n', maxsplit=1)[1])
        self.assertEqual(material['public_artifact']['content'], 'Pre-cutoff protocol with arm definitions.')
        self.assertEqual(
            [artifact['artifact_id'] for artifact in material['comparison_artifacts']],
            ['a-results', 'z-results'],
        )
        self.assertEqual(
            [control['control_id'] for control in material['calibration_controls']],
            sorted((*CANARY_IDS, *NEGATIVE_IDS)),
        )
        self.assertTrue(all(set(control) == {'control_id', 'content'} for control in material['calibration_controls']))

    def test_prompt_requires_complete_utf8_calibration_inventory(self) -> None:
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'at least one canary'):
            build_quote_review_prompt(
                public_artifact_id=PUBLIC_ID,
                public_payload=b'public',
                comparison_payloads={'future': b'future'},
                canary_controls={},
                negative_control_controls={NEGATIVE_IDS[0]: b'legitimate control'},
            )
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'must be disjoint'):
            build_quote_review_prompt(
                public_artifact_id=PUBLIC_ID,
                public_payload=b'public',
                comparison_payloads={'future': b'future'},
                canary_controls={CANARY_IDS[0]: b'future disclosure'},
                negative_control_controls={CANARY_IDS[0]: b'ordinary hypothesis'},
            )
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'not valid UTF-8'):
            build_quote_review_prompt(
                public_artifact_id=PUBLIC_ID,
                public_payload=b'public',
                comparison_payloads={'future': b'future'},
                canary_controls={CANARY_IDS[0]: b'future disclosure'},
                negative_control_controls={NEGATIVE_IDS[0]: b'\xff'},
            )

    def test_prompt_rejects_role_revealing_control_ids(self) -> None:
        with self.assertRaisesRegex(JudgeReviewAdapterError, 'must be opaque'):
            build_quote_review_prompt(
                public_artifact_id=PUBLIC_ID,
                public_payload=b'public',
                comparison_payloads={'future': b'future'},
                canary_controls={'canary-future-result': b'future disclosure'},
                negative_control_controls={NEGATIVE_IDS[0]: b'ordinary hypothesis'},
            )


if __name__ == '__main__':
    unittest.main()
