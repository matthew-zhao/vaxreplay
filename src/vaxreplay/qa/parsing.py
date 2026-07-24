"""Fail-closed parsing for untrusted VaxReplay model submissions."""

from __future__ import annotations

import enum
import json
import re
from typing import Any, NoReturn

from pydantic import ValidationError

from vaxreplay.case_schema import Submission

DEFAULT_MAX_SUBMISSION_BYTES = 1_048_576
DEFAULT_MAX_SUBMISSION_CHARACTERS = 1_048_576

_JSON_WHITESPACE = re.compile(r'[ \t\r\n]*')


class SubmissionParseReason(str, enum.Enum):
    """Stable, machine-readable reasons for rejecting an untrusted response."""

    INVALID_INPUT_TYPE = 'invalid_input_type'
    INVALID_LIMIT = 'invalid_limit'
    INPUT_TOO_LARGE = 'input_too_large'
    INVALID_UTF8 = 'invalid_utf8'
    INVALID_JSON = 'invalid_json'
    DUPLICATE_KEY = 'duplicate_key'
    NON_STANDARD_NUMBER = 'non_standard_number'
    TRAILING_DATA = 'trailing_data'
    INVALID_SUBMISSION = 'invalid_submission'


class SubmissionParseError(ValueError):
    """A fail-closed response rejection with a stable reason code."""

    def __init__(
        self,
        reason: SubmissionParseReason,
        detail: str,
        *,
        validation_error_count: int | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.validation_error_count = validation_error_count
        super().__init__(f'{reason.value}: {detail}')


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class _NonStandardNumberError(ValueError):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(token)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_non_standard_number(token: str) -> NoReturn:
    raise _NonStandardNumberError(token)


def _validate_limit(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_LIMIT,
            f'{name} must be a positive integer',
        )


def _decode_payload(
    payload: bytes | str,
    *,
    max_bytes: int,
    max_characters: int,
) -> str:
    if isinstance(payload, bytes):
        if len(payload) > max_bytes:
            raise SubmissionParseError(
                SubmissionParseReason.INPUT_TOO_LARGE,
                f'encoded submission is {len(payload)} bytes; maximum is {max_bytes}',
            )
        try:
            text = payload.decode('utf-8', errors='strict')
        except UnicodeDecodeError as error:
            raise SubmissionParseError(
                SubmissionParseReason.INVALID_UTF8,
                f'UTF-8 decoding failed at byte offset {error.start}',
            ) from error
    elif isinstance(payload, str):
        text = payload
        if len(text) > max_characters:
            raise SubmissionParseError(
                SubmissionParseReason.INPUT_TOO_LARGE,
                f'submission is {len(text)} characters; maximum is {max_characters}',
            )
        try:
            encoded_length = len(text.encode('utf-8', errors='strict'))
        except UnicodeEncodeError as error:
            raise SubmissionParseError(
                SubmissionParseReason.INVALID_UTF8,
                f'submission contains an invalid Unicode scalar at character offset {error.start}',
            ) from error
        if encoded_length > max_bytes:
            raise SubmissionParseError(
                SubmissionParseReason.INPUT_TOO_LARGE,
                f'encoded submission is {encoded_length} bytes; maximum is {max_bytes}',
            )
    else:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_INPUT_TYPE,
            'submission must be bytes or str',
        )

    if len(text) > max_characters:
        raise SubmissionParseError(
            SubmissionParseReason.INPUT_TOO_LARGE,
            f'submission is {len(text)} characters; maximum is {max_characters}',
        )
    return text


def parse_submission(
    payload: bytes | str,
    *,
    max_bytes: int = DEFAULT_MAX_SUBMISSION_BYTES,
    max_characters: int = DEFAULT_MAX_SUBMISSION_CHARACTERS,
) -> Submission:
    """Parse one untrusted JSON document and validate it as a ``Submission``.

    This parser rejects ambiguous JSON before schema validation: duplicate object
    member names, non-standard numeric constants, invalid UTF-8, and any
    non-whitespace content following the first JSON value.
    """

    _validate_limit('max_bytes', max_bytes)
    _validate_limit('max_characters', max_characters)
    text = _decode_payload(payload, max_bytes=max_bytes, max_characters=max_characters)

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_standard_number,
        strict=True,
    )
    leading_whitespace = _JSON_WHITESPACE.match(text)
    if leading_whitespace is None:
        raise AssertionError('JSON whitespace matcher must always match')
    start = leading_whitespace.end()
    try:
        _, end = decoder.raw_decode(text, idx=start)
    except _DuplicateKeyError as error:
        raise SubmissionParseError(
            SubmissionParseReason.DUPLICATE_KEY,
            f'duplicate JSON object member {error.key!r}',
        ) from error
    except _NonStandardNumberError as error:
        raise SubmissionParseError(
            SubmissionParseReason.NON_STANDARD_NUMBER,
            f'non-standard JSON numeric token {error.token!r}',
        ) from error
    except json.JSONDecodeError as error:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_JSON,
            f'JSON decoding failed at line {error.lineno}, column {error.colno}',
        ) from error
    except (RecursionError, ValueError) as error:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_JSON,
            'JSON decoding failed',
        ) from error

    if _JSON_WHITESPACE.fullmatch(text, pos=end) is None:
        raise SubmissionParseError(
            SubmissionParseReason.TRAILING_DATA,
            'non-whitespace data follows the first JSON document',
        )

    try:
        # Validate the original document in Pydantic's JSON mode. Strict models
        # intentionally distinguish JSON enum strings from Python strings.
        return Submission.model_validate_json(text)
    except ValidationError as error:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_SUBMISSION,
            f'submission schema validation failed with {error.error_count()} error(s)',
            validation_error_count=error.error_count(),
        ) from error
    except RecursionError as error:
        raise SubmissionParseError(
            SubmissionParseReason.INVALID_SUBMISSION,
            'submission schema validation exceeded the nesting limit',
        ) from error
