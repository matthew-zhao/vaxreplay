from __future__ import annotations

import json

import pytest

from vaxreplay.case_schema import Submission
from vaxreplay.qa.parsing import SubmissionParseError, SubmissionParseReason, parse_submission


def _valid_submission() -> dict[str, object]:
    return {
        'schema_version': 'vaxreplay.v0.1',
        'episode_id': 'episode-1',
        'manifest_sha256': '0' * 64,
        'ranking': ['candidate-1'],
        'forecasts': [
            {
                'candidate_id': 'candidate-1',
                'target_id': 'functional-validation',
                'horizon_days': 180,
                'probability': 0.5,
            }
        ],
        'assessments': [
            {
                'candidate_id': 'candidate-1',
                'dimension': 'immunogenicity',
                'conclusion': 'favorable',
                'citations': [
                    {
                        'evidence_id': 'evidence-1',
                        'stance': 'support',
                        'quote': 'A sufficiently long supporting evidence quote.',
                    }
                ],
            }
        ],
    }


def _json() -> str:
    return json.dumps(_valid_submission(), separators=(',', ':'))


def _assert_reason(payload: object, reason: SubmissionParseReason, **kwargs: int) -> SubmissionParseError:
    with pytest.raises(SubmissionParseError) as caught:
        parse_submission(payload, **kwargs)  # type: ignore[arg-type]
    assert caught.value.reason is reason
    return caught.value


def test_accepts_exactly_one_valid_document_as_str_or_bytes() -> None:
    expected = Submission.model_validate_json(_json())

    assert parse_submission(f' \n{_json()}\t') == expected
    assert parse_submission(_json().encode('utf-8')) == expected


def test_rejects_unsupported_input_types() -> None:
    _assert_reason(bytearray(_json(), 'utf-8'), SubmissionParseReason.INVALID_INPUT_TYPE)


@pytest.mark.parametrize(
    ('kwargs', 'reason'),
    [
        ({'max_bytes': 0}, SubmissionParseReason.INVALID_LIMIT),
        ({'max_characters': True}, SubmissionParseReason.INVALID_LIMIT),
        ({'max_bytes': 8}, SubmissionParseReason.INPUT_TOO_LARGE),
        ({'max_characters': 8}, SubmissionParseReason.INPUT_TOO_LARGE),
    ],
)
def test_enforces_typed_positive_size_limits(
    kwargs: dict[str, int],
    reason: SubmissionParseReason,
) -> None:
    _assert_reason(_json(), reason, **kwargs)


def test_enforces_encoded_byte_limit_for_multibyte_str() -> None:
    payload = _json().replace('episode-1', 'épisode-1')

    _assert_reason(
        payload,
        SubmissionParseReason.INPUT_TOO_LARGE,
        max_bytes=len(payload),
        max_characters=len(payload),
    )


@pytest.mark.parametrize('payload', [b'\xff', b'{"episode_id":"\x80"}', '\ud800'])
def test_rejects_invalid_utf8_or_unicode_scalars(payload: bytes | str) -> None:
    _assert_reason(payload, SubmissionParseReason.INVALID_UTF8)


@pytest.mark.parametrize(
    'payload',
    [
        '',
        '   \n',
        '{',
        'not-json',
        '{"unterminated":',
        '{"control":"\x01"}',
    ],
)
def test_rejects_invalid_json(payload: str) -> None:
    _assert_reason(payload, SubmissionParseReason.INVALID_JSON)


@pytest.mark.parametrize(
    'payload',
    [
        '{"episode_id":"first","episode_id":"second"}',
        '{"outer":{"candidate_id":"first","candidate_id":"second"}}',
        '{"outer":{"inner":{"evidence_id":"first","evidence_id":"second"}}}',
    ],
)
def test_rejects_duplicate_keys_at_every_nesting_level(payload: str) -> None:
    error = _assert_reason(payload, SubmissionParseReason.DUPLICATE_KEY)
    assert 'duplicate JSON object member' in error.detail


@pytest.mark.parametrize('suffix', ['{}', 'null', 'garbage', '/* comment */'])
def test_rejects_any_trailing_non_whitespace_data(suffix: str) -> None:
    _assert_reason(f'{_json()} {suffix}', SubmissionParseReason.TRAILING_DATA)


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_rejects_non_standard_numeric_constants(token: str) -> None:
    payload = _json().replace('0.5', token)

    _assert_reason(payload, SubmissionParseReason.NON_STANDARD_NUMBER)


@pytest.mark.parametrize('token', ['1e9999', '-1e9999', '1.0000000001', '-0.0000000001'])
def test_submission_validation_rejects_extreme_or_out_of_range_probabilities(token: str) -> None:
    payload = _json().replace('0.5', token)

    error = _assert_reason(payload, SubmissionParseReason.INVALID_SUBMISSION)
    assert error.validation_error_count is not None
    assert error.validation_error_count >= 1


def test_submission_validation_rejects_extra_fields_and_wrong_types() -> None:
    payload = _valid_submission()
    payload['unexpected'] = True
    payload['ranking'] = [1]

    error = _assert_reason(json.dumps(payload), SubmissionParseReason.INVALID_SUBMISSION)
    assert error.validation_error_count == 2
