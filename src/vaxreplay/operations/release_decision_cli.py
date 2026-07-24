"""Offline artifact-level publication/readiness composition.

The command cross-binds exact archive bytes but deliberately does not parse them as a
prospective release. Production approval must subsequently use the in-process semantic
composition in :mod:`vaxreplay.operations.prospective_release_approval`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.campaign_publication import (
    MAX_PUBLICATION_ARTIFACT_BYTES,
    SignedCampaignPublicationManifest,
)
from vaxreplay.operations.campaign_publication_cli import (
    PublicationArtifactPathMap,
    _parse_time,
    _read_regular,
    _write_exclusive,
)
from vaxreplay.operations.release_decision import verify_tier_a_release_decision
from vaxreplay.operations.release_readiness import TierAReleaseReadinessManifest
from vaxreplay.operations.release_readiness_cli import (
    _read_authority_keys,
    _read_evidence_root,
    _read_named_paths,
)

_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = MAX_PUBLICATION_ARTIFACT_BYTES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Offline-verify one artifact-level Tier A publication/readiness decision'
    )
    parser.add_argument('--signed-campaign-manifest', required=True)
    parser.add_argument('--campaign-trust-policy', required=True)
    parser.add_argument('--expected-campaign-trust-policy-sha256', required=True)
    parser.add_argument('--campaign-artifact-map', required=True)
    parser.add_argument('--publication-receipt', action='append', required=True)
    parser.add_argument('--readiness-policy', required=True)
    parser.add_argument('--expected-readiness-policy-sha256', required=True)
    parser.add_argument('--readiness-manifest', required=True)
    parser.add_argument('--readiness-subject', action='append', default=[], metavar='ROLE=PATH')
    parser.add_argument('--readiness-evidence-root', required=True)
    parser.add_argument('--readiness-authority-key', action='append', default=[], metavar='ID=PATH')
    parser.add_argument('--verification-time-evidence', required=True)
    parser.add_argument('--verification-time-public-key', required=True)
    parser.add_argument('--verified-at', required=True)
    parser.add_argument('--output')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        signed_bytes = _read_regular(
            Path(arguments.signed_campaign_manifest),
            _MAX_CONTROL_BYTES,
        )
        signed = SignedCampaignPublicationManifest.model_validate_json(signed_bytes)
        if signed_bytes != canonical_json_bytes(signed):
            raise ValueError('signed campaign manifest must be canonical JSON')
        campaign_trust_bytes = _read_regular(
            Path(arguments.campaign_trust_policy),
            _MAX_CONTROL_BYTES,
        )
        campaign_artifacts = _read_campaign_artifacts(
            Path(arguments.campaign_artifact_map),
            signed,
        )
        receipts = tuple(_read_regular(Path(path), _MAX_RECEIPT_BYTES) for path in arguments.publication_receipt)
        readiness_policy_bytes = _read_regular(
            Path(arguments.readiness_policy),
            _MAX_CONTROL_BYTES,
        )
        readiness_manifest_bytes = _read_regular(
            Path(arguments.readiness_manifest),
            _MAX_CONTROL_BYTES,
        )
        readiness_manifest = TierAReleaseReadinessManifest.model_validate_json(readiness_manifest_bytes)
        if readiness_manifest_bytes != canonical_json_bytes(readiness_manifest):
            raise ValueError('readiness manifest must be canonical JSON')
        subjects = _read_named_paths(arguments.readiness_subject, 'readiness subject')
        evidence = _read_evidence_root(
            Path(arguments.readiness_evidence_root),
            {item.statement.evidence_artifact.sha256 for item in readiness_manifest.evidence},
        )
        keys = _read_authority_keys(arguments.readiness_authority_key)
        verification_time_evidence = _read_regular(
            Path(arguments.verification_time_evidence),
            _MAX_CONTROL_BYTES,
        )
        verification_time_public_key = _read_regular(
            Path(arguments.verification_time_public_key),
            32,
        )
        report = verify_tier_a_release_decision(
            signed_campaign_manifest_bytes=signed_bytes,
            campaign_trust_policy_bytes=campaign_trust_bytes,
            expected_campaign_trust_policy_sha256=(arguments.expected_campaign_trust_policy_sha256),
            campaign_artifacts=campaign_artifacts,
            publication_receipt_bytes=receipts,
            readiness_policy_bytes=readiness_policy_bytes,
            expected_readiness_policy_sha256=(arguments.expected_readiness_policy_sha256),
            readiness_manifest_bytes=readiness_manifest_bytes,
            readiness_release_subject_bytes=subjects,
            readiness_evidence_artifact_bytes=evidence,
            readiness_authority_public_key_bytes=keys,
            verification_time_evidence_bytes=verification_time_evidence,
            verification_time_public_key_bytes=verification_time_public_key,
            verified_at=_parse_time(arguments.verified_at),
        )
        payload = canonical_json_bytes(report)
        if arguments.output:
            output = _write_exclusive(Path(arguments.output), payload)
            sys.stdout.buffer.write(
                canonical_json_bytes({'output': str(output), 'sha256': hashlib.sha256(payload).hexdigest()}) + b'\n'
            )
        else:
            sys.stdout.buffer.write(payload + b'\n')
        return 0
    except (OSError, TypeError, ValueError):
        sys.stderr.write(
            json.dumps(
                {'error': 'tier_a_release_decision_failed', 'status': 'failed'},
                sort_keys=True,
                separators=(',', ':'),
            )
            + '\n'
        )
        return 2


def _read_campaign_artifacts(
    map_path: Path,
    signed: SignedCampaignPublicationManifest,
) -> dict[str, bytes]:
    path_map_bytes = _read_regular(map_path, 4 * 1024 * 1024)
    path_map = PublicationArtifactPathMap.model_validate_json(path_map_bytes)
    if path_map_bytes != canonical_json_bytes(path_map):
        raise ValueError('campaign artifact map must be canonical JSON')
    mapped = {item.artifact_id: item.path for item in path_map.artifacts}
    expected = {item.artifact_id: item for item in signed.manifest.artifacts}
    if set(mapped) != set(expected):
        raise ValueError('campaign artifact map differs from signed manifest')
    base = map_path.expanduser().absolute().parent.resolve(strict=True)
    artifacts: dict[str, bytes] = {}
    for artifact_id, binding in sorted(expected.items()):
        if binding.byte_count > _MAX_ARTIFACT_BYTES:
            raise ValueError('campaign artifact exceeds verifier limit')
        requested = Path(mapped[artifact_id])
        path = requested if requested.is_absolute() else base / requested
        artifacts[artifact_id] = _read_regular(path, binding.byte_count)
    return artifacts


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())


__all__ = ['main']
