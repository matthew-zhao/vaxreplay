"""Operator CLI for signing and offline verification of Tier-A publications."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import stat
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, TypeVar

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.campaign_publication import (
    MAX_PUBLICATION_ARTIFACT_BYTES,
    CampaignPublicationError,
    CampaignPublicationManifest,
    PublicationReceiptStatement,
    SignedCampaignPublicationManifest,
    WorkerImageProvenance,
    sign_campaign_publication_manifest,
    sign_publication_receipt,
    sign_worker_image_provenance,
    verify_campaign_publication,
)
from vaxreplay.operations.operator_trust import (
    OperatorTrustError,
    add_signer_arguments,
    signer_and_clock_from_args,
)
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc

ARTIFACT_PATH_MAP_SCHEMA_VERSION = 'vaxreplay.publication-artifact-path-map.v0.1'
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_ARTIFACT_MAP_BYTES = 4 * 1024 * 1024
_MAX_CLI_ARTIFACT_BYTES = MAX_PUBLICATION_ARTIFACT_BYTES
ModelT = TypeVar('ModelT', bound=StrictModel)


class ArtifactPathEntry(StrictModel):
    artifact_id: str = Field(pattern=SAFE_ID_PATTERN)
    path: str = Field(min_length=1, max_length=4096)

    @field_validator('path')
    @classmethod
    def validate_path(cls, value: str) -> str:
        if '\x00' in value or '\r' in value or '\n' in value:
            raise ValueError('artifact path contains a forbidden character')
        return value


class PublicationArtifactPathMap(StrictModel):
    """Exact, sorted artifact-ID to file-path inventory supplied by the verifier."""

    schema_version: Literal['vaxreplay.publication-artifact-path-map.v0.1'] = ARTIFACT_PATH_MAP_SCHEMA_VERSION
    artifacts: tuple[ArtifactPathEntry, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode='after')
    def validate_entries(self) -> Self:
        ids = tuple(item.artifact_id for item in self.artifacts)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError('artifact path map must be sorted and unique by artifact_id')
        return self


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sign and independently offline-verify a canonical Tier-A campaign publication'
    )
    commands = parser.add_subparsers(dest='command', required=True)

    manifest = commands.add_parser('sign-manifest', help='sign one canonical campaign manifest')
    manifest.add_argument('--input', required=True)
    manifest.add_argument('--output', required=True)
    add_signer_arguments(manifest, dev_required=True)

    receipt = commands.add_parser('sign-receipt', help='sign one independent publication receipt')
    receipt.add_argument('--input', required=True)
    receipt.add_argument('--output', required=True)
    add_signer_arguments(receipt, dev_required=True)

    provenance = commands.add_parser('sign-provenance', help='sign normalized worker-image provenance')
    provenance.add_argument('--input', required=True)
    provenance.add_argument('--signing-key-id', required=True)
    provenance.add_argument('--output', required=True)
    add_signer_arguments(provenance, dev_required=True)

    verify = commands.add_parser(
        'verify',
        help='offline-verify exact artifacts and independent receipts against separate trust',
    )
    verify.add_argument('--signed-manifest', required=True)
    verify.add_argument('--trust-policy', required=True)
    verify.add_argument('--expected-trust-policy-sha256', required=True)
    verify.add_argument('--artifact-map', required=True)
    verify.add_argument('--receipt', action='append', required=True)
    verify.add_argument('--verified-at', required=True)
    verify.add_argument('--output')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == 'verify':
            _verify(args)
            return
        signer, gate = signer_and_clock_from_args(args)
        input_bytes = _read_regular(Path(args.input), _MAX_CONTROL_BYTES)
        if args.command == 'sign-manifest':
            statement = _canonical_model(input_bytes, CampaignPublicationManifest)
            if gate is not None:
                gate.require_synchronized(security_time=statement.created_at)
            payload = canonical_json_bytes(sign_campaign_publication_manifest(statement, signer=signer))
        elif args.command == 'sign-receipt':
            statement = _canonical_model(input_bytes, PublicationReceiptStatement)
            if gate is not None:
                gate.require_synchronized(security_time=statement.published_at)
            payload = canonical_json_bytes(sign_publication_receipt(statement, signer=signer))
        elif args.command == 'sign-provenance':
            statement = _canonical_model(input_bytes, WorkerImageProvenance)
            if gate is not None:
                gate.require_synchronized(security_time=statement.created_at)
            payload = canonical_json_bytes(
                sign_worker_image_provenance(
                    statement,
                    signing_key_id=args.signing_key_id,
                    signer=signer,
                )
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise CampaignPublicationError('unknown campaign publication command')
        output = _write_exclusive(Path(args.output), payload)
        _stdout(
            {
                'output': str(output),
                'sha256': hashlib.sha256(payload).hexdigest(),
                'signer_mode': 'external' if args.external_signer_process else 'development-local',
            }
        )
    except (OSError, ValueError, CampaignPublicationError, OperatorTrustError):
        _stderr({'error': 'campaign_publication_operation_failed', 'status': 'failed'})
        raise SystemExit(2) from None


def _verify(args: argparse.Namespace) -> None:
    signed_bytes = _read_regular(Path(args.signed_manifest), _MAX_CONTROL_BYTES)
    signed = _canonical_model(signed_bytes, SignedCampaignPublicationManifest)
    trust_bytes = _read_regular(Path(args.trust_policy), _MAX_CONTROL_BYTES)
    if not _matches_sha256(trust_bytes, args.expected_trust_policy_sha256):
        raise CampaignPublicationError('trust policy differs from the separately supplied out-of-band digest')
    map_path = Path(args.artifact_map)
    path_map = _canonical_model(
        _read_regular(map_path, _MAX_ARTIFACT_MAP_BYTES),
        PublicationArtifactPathMap,
    )
    expected = {item.artifact_id: item for item in signed.manifest.artifacts}
    mapped = {item.artifact_id: item.path for item in path_map.artifacts}
    if set(mapped) != set(expected):
        raise CampaignPublicationError('artifact path map differs from exact manifest inventory')
    base = map_path.expanduser().absolute().parent.resolve(strict=True)
    artifacts: dict[str, bytes] = {}
    for artifact_id in sorted(expected):
        binding = expected[artifact_id]
        if binding.byte_count > _MAX_CLI_ARTIFACT_BYTES:
            raise CampaignPublicationError('artifact exceeds the bounded CLI verifier limit')
        requested = Path(mapped[artifact_id])
        artifact_path = requested if requested.is_absolute() else base / requested
        artifacts[artifact_id] = _read_regular(artifact_path, binding.byte_count)
    receipt_paths = tuple(Path(path) for path in args.receipt)
    if len(receipt_paths) < 2 or len(receipt_paths) > 32:
        raise CampaignPublicationError('offline verification requires 2 to 32 receipt files')
    receipts = tuple(_read_regular(path, _MAX_RECEIPT_BYTES) for path in receipt_paths)
    verified_at = _parse_time(args.verified_at)
    report = verify_campaign_publication(
        signed_bytes,
        trust_policy_bytes=trust_bytes,
        expected_trust_policy_sha256=args.expected_trust_policy_sha256,
        artifacts=artifacts,
        publication_receipt_bytes=receipts,
        verified_at=verified_at,
    )
    payload = canonical_json_bytes(report)
    if args.output:
        output = _write_exclusive(Path(args.output), payload)
        _stdout({'output': str(output), 'sha256': hashlib.sha256(payload).hexdigest()})
    else:
        sys.stdout.buffer.write(payload + b'\n')


def _canonical_model(payload: bytes, model: type[ModelT]) -> ModelT:
    parsed = model.model_validate_json(payload)
    if payload != canonical_json_bytes(parsed):
        raise CampaignPublicationError('campaign publication input must use canonical JSON')
    return parsed


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise CampaignPublicationError('verified-at is invalid') from None
    return aware_utc(parsed, 'publication verification time')


def _matches_sha256(payload: bytes, expected: object) -> bool:
    return (
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in '0123456789abcdef' for character in expected)
        and hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected)
    )


def _read_regular(path: Path, maximum: int) -> bytes:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(requested, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CampaignPublicationError('campaign publication input must be a regular file')
        if before.st_size < 1 or before.st_size > maximum:
            raise CampaignPublicationError('campaign publication input size is invalid')
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CampaignPublicationError('campaign publication input changed while being read')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CampaignPublicationError('campaign publication input changed while being read')
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise CampaignPublicationError('campaign publication input changed while being read')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> Path:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError('short write')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _stdout(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b'\n')


def _stderr(value: object) -> None:
    sys.stderr.buffer.write(canonical_json_bytes(value) + b'\n')


if __name__ == '__main__':
    main()


__all__ = [
    'ARTIFACT_PATH_MAP_SCHEMA_VERSION',
    'ArtifactPathEntry',
    'PublicationArtifactPathMap',
    'main',
]
