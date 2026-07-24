"""Operator CLI for preparing label-free prospective decision packages."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import CandidateRecord, EvidenceRecord, StrictModel
from vaxreplay.prospective import (
    SourceCaptureArtifact,
    build_prospective_decision_package,
    load_prospective_decision_package,
)
from vaxreplay.temporal_schema import DecisionTimeConfig


class CaptureIndexEntry(StrictModel):
    source_id: str = Field(min_length=1)
    source_release_at: datetime
    captured_at: datetime
    witnessed_at: datetime
    manifest_path: str = Field(min_length=1)

    @field_validator('source_release_at', 'captured_at', 'witnessed_at')
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('capture index timestamps must include a UTC offset')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def validate_capture_order(self) -> Self:
        if self.captured_at < self.source_release_at:
            raise ValueError('captured_at cannot predate the source release')
        if self.witnessed_at < self.captured_at:
            raise ValueError('witnessed_at cannot predate capture completion')
        return self


class CaptureIndex(StrictModel):
    captures: tuple[CaptureIndexEntry, ...] = Field(min_length=1)

    @field_validator('captures')
    @classmethod
    def validate_captures(cls, value: tuple[CaptureIndexEntry, ...]) -> tuple[CaptureIndexEntry, ...]:
        source_ids = tuple(entry.source_id for entry in value)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError('capture index source IDs must be unique')
        return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare prospective VaxReplay decision packages')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build = subparsers.add_parser('build-package')
    build.add_argument('--config', required=True)
    build.add_argument('--candidates-jsonl', required=True)
    build.add_argument('--evidence-jsonl', required=True)
    build.add_argument('--candidate-set-definition', required=True)
    build.add_argument('--evidence-acquisition-spec', required=True)
    build.add_argument('--outcome-adjudication-spec', required=True)
    build.add_argument('--candidate-set-available-at', required=True)
    build.add_argument('--capture-index', required=True)
    build.add_argument('--output-dir', required=True)

    for name in ('inspect-package', 'receipt-requests'):
        command = subparsers.add_parser(name)
        command.add_argument('--package-dir', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == 'build-package':
        config = DecisionTimeConfig.model_validate_json(Path(args.config).read_bytes())
        candidates = _read_jsonl(Path(args.candidates_jsonl), CandidateRecord)
        evidence = _read_jsonl(Path(args.evidence_jsonl), EvidenceRecord)
        capture_index_path = Path(args.capture_index)
        capture_index = CaptureIndex.model_validate_json(capture_index_path.read_bytes())
        source_captures = tuple(
            SourceCaptureArtifact(
                source_id=entry.source_id,
                source_release_at=entry.source_release_at,
                captured_at=entry.captured_at,
                witnessed_at=entry.witnessed_at,
                manifest_bytes=_resolve_capture_path(capture_index_path, entry.manifest_path).read_bytes(),
            )
            for entry in capture_index.captures
        )
        package = build_prospective_decision_package(
            Path(args.output_dir),
            config=config,
            candidates=candidates,
            evidence=evidence,
            protocol_artifacts={
                'candidate_set_definition': Path(args.candidate_set_definition).read_bytes(),
                'evidence_acquisition_spec': Path(args.evidence_acquisition_spec).read_bytes(),
                'outcome_adjudication_spec': Path(args.outcome_adjudication_spec).read_bytes(),
            },
            candidate_set_available_at=TypeAdapter(datetime).validate_python(args.candidate_set_available_at),
            source_captures=source_captures,
        )
        _write_json(_package_summary(package))
        return

    package = load_prospective_decision_package(Path(args.package_dir))
    if args.command == 'inspect-package':
        _write_json(_package_summary(package))
    else:
        _write_json([request.model_dump(mode='json') for request in package.receipt_requests])


def _package_summary(package) -> dict[str, object]:
    return {
        'episode_id': package.manifest.episode.episode_id,
        'decision_at': package.manifest.episode.decision_at.isoformat(),
        'decision_package_sha256': package.manifest_sha256,
        'decision_snapshot_sha256': package.manifest.episode.decision_snapshot_sha256,
        'candidate_count': len(package.candidates),
        'visible_evidence_count': len(package.evidence),
        'source_capture_count': len(package.manifest.source_captures),
        'receipt_request_count': len(package.receipt_requests),
        'tier_a_eligible': False,
        'next_action': 'obtain and verify independent timestamp proofs for all receipt requests',
    }


def _read_jsonl(path: Path, model):
    payload = path.read_bytes()
    if not payload.endswith(b'\n'):
        raise ValueError(f'{path.name} must end with a newline')
    records = tuple(model.model_validate_json(line) for line in payload.splitlines())
    if not records:
        raise ValueError(f'{path.name} cannot be empty')
    return records


def _resolve_capture_path(index_path: Path, manifest_path: str) -> Path:
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = index_path.parent / path
    return path


def _write_json(value: object) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode('utf-8') + '\n')


if __name__ == '__main__':
    main()
