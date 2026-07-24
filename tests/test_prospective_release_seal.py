from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tests.test_operations_prospective_release_approval import (
    _approval_arguments,
    _official_archive,
)
from tests.test_prospective_release import (
    _ATTEMPT_POLICY,
    _CASE_PROOF,
    _ELIGIBILITY,
    _SOURCE_CAPTURE_POLICY,
    _VERIFIER_POLICY,
    _case_verifier,
    _decision_verifier,
    _source_capture_verifier,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.prospective_campaign_archive import build_prospective_campaign_archive
from vaxreplay.operations.prospective_release_approval import (
    VerifiedTierAProspectiveReleaseApproval,
    verify_and_materialize_tier_a_prospective_release,
)
from vaxreplay.operations.prospective_release_approval_identity import (
    TierAProspectiveReleaseApprovalReplay,
    reverify_tier_a_prospective_release_approval_identity,
)
from vaxreplay.prospective_release import (
    LoadedProspectiveCohortRelease,
    build_prospective_cohort_release,
)
from vaxreplay.runner.prospective_challenge import load_prospective_challenge_bundle
from vaxreplay.runner.prospective_release_seal import (
    ProspectiveReleaseSealIntegrityError,
    ProspectiveReleaseSealTarget,
    ProspectiveReleaseTimestampProof,
    build_prospective_release_seal,
    build_prospective_release_seal_target,
    load_prospective_release_seal,
)
from vaxreplay.temporal_schema import TemporalReceiptAuthority

pytestmark = pytest.mark.usefixtures('synthetic_official_replay_patch')

_PROOF_BYTES = b'fictional public transparency proof over the exact release target'
_APPROVAL_CONTEXTS: dict[
    Path,
    tuple[str, TierAProspectiveReleaseApprovalReplay],
] = {}
_APPROVAL_RESULTS: dict[Path, VerifiedTierAProspectiveReleaseApproval] = {}


class _ApprovalClock(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        value = cls(2024, 3, 1, 1, 0, tzinfo=timezone.utc)
        return value if tz is None else value.astimezone(tz)


def _build_approved_release(root: Path) -> LoadedProspectiveCohortRelease:
    with _official_archive(root / 'official-input') as (base_release, base_archive):
        extended_admission = replace(
            base_release.verified_admission,
            admission=base_release.verified_admission.admission.model_copy(
                update={
                    'run_deadline_at': base_release.verified_admission.admission.run_deadline_at + timedelta(days=30)
                }
            ),
        )
        release = build_prospective_cohort_release(
            root / 'extended-release',
            challenge_id='prospective-challenge-1',
            verified_admission=extended_admission,
            case_universe_proof=_CASE_PROOF,
            eligibility_protocol=_ELIGIBILITY,
            verifier_policy=_VERIFIER_POLICY,
            source_capture_policy=_SOURCE_CAPTURE_POLICY,
            attempt_policy=_ATTEMPT_POLICY,
            decision_receipt_verifier=_decision_verifier,
            case_universe_seal_verifier=_case_verifier,
            source_capture_verifier=_source_capture_verifier,
        )
        archive = build_prospective_campaign_archive(
            release.root,
            release_scope=base_archive.index.release_scope,
        )
        from tests import test_operations_campaign_publication as campaign_test_module
        from tests import test_operations_selection_registry as registry_test_module
        from vaxreplay.operations import witness_service as witness_service_module

        real_registry = campaign_test_module._registry
        real_registry_trust = registry_test_module._trust
        historical_registry_time = datetime(2024, 2, 28, tzinfo=timezone.utc)

        def build_historical_trust(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            kwargs.setdefault('valid_from', historical_registry_time - timedelta(days=1))
            return real_registry_trust(*args, **kwargs)

        def build_historical_registry(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            kwargs['clock'] = historical_registry_time
            with (
                patch.object(registry_test_module, '_T0', historical_registry_time),
                patch.object(registry_test_module, '_trust', build_historical_trust),
                patch.object(
                    witness_service_module,
                    '_security_time',
                    lambda: historical_registry_time,
                ),
            ):
                return real_registry(*args, **kwargs)

        with (
            patch.object(campaign_test_module, 'datetime', _ApprovalClock),
            patch.object(campaign_test_module, '_registry', build_historical_registry),
        ):
            arguments = _approval_arguments(root / 'approval', release, archive)
        approval = verify_and_materialize_tier_a_prospective_release(**arguments)
    replay_arguments = dict(arguments)
    replay_arguments.pop('materialized_release_dir')
    replay = TierAProspectiveReleaseApprovalReplay(
        **replay_arguments,  # type: ignore[arg-type]
        materialization_parent=root,
    )
    digest = hashlib.sha256(canonical_json_bytes(approval.report)).hexdigest()
    _APPROVAL_CONTEXTS[release.root] = (digest, replay)
    _APPROVAL_RESULTS[release.root] = approval
    return release


def _approval_kwargs(release: LoadedProspectiveCohortRelease) -> dict[str, object]:
    digest, replay = _APPROVAL_CONTEXTS[release.root]
    return {
        'expected_approval_report_sha256': digest,
        'approval_replay': replay,
    }


def _release_decision_verifier(_release: LoadedProspectiveCohortRelease):  # noqa: ANN201
    return _decision_verifier


def _timestamp_verifier(proof: ProspectiveReleaseTimestampProof, payload: bytes) -> bool:
    return proof.authority_id == 'fixture-release-log' and payload == _PROOF_BYTES


def _opening(release: LoadedProspectiveCohortRelease) -> datetime:
    latest_witness = max(
        release.verified_admission.case_universe.seal.witnessed_at,
        *(
            receipt.witnessed_at
            for decision_seal in release.verified_admission.seals
            for receipt in decision_seal.manifest.receipts
        ),
        *(
            capture.witnessed_at
            for package in release.verified_admission.packages
            for capture in package.manifest.source_captures
        ),
    )
    approval_verified_at = _APPROVAL_CONTEXTS[release.root][1].verified_at
    return max(latest_witness, approval_verified_at) + timedelta(hours=1)


def _target(release: LoadedProspectiveCohortRelease, opening: datetime):
    return build_prospective_release_seal_target(
        release,
        submissions_open_at=opening,
        decision_receipt_verifier=_release_decision_verifier(release),
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
        **_approval_kwargs(release),
    )


def _proof(target, *, witnessed_at=None, authority=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG):
    target_bytes = canonical_json_bytes(target)
    default_witnessed_at = (
        target.latest_prerequisite_witnessed_at
        + (target.submissions_open_at - target.latest_prerequisite_witnessed_at) / 2
    )
    return ProspectiveReleaseTimestampProof(
        receipt_id='release-receipt-1',
        authority_type=authority,
        authority_id='fixture-release-log',
        target_sha256=hashlib.sha256(target_bytes).hexdigest(),
        target_bytes=len(target_bytes),
        prospective_release_sha256=target.prospective_release_sha256,
        witnessed_at=witnessed_at or default_witnessed_at,
        proof_sha256=hashlib.sha256(_PROOF_BYTES).hexdigest(),
        proof_bytes=len(_PROOF_BYTES),
        verification_uri='https://log.invalid/releases/release-receipt-1',
    )


def _build_seal(root: Path):
    release = _build_approved_release(root)
    opening = _opening(release)
    target = _target(release, opening)
    seal = build_prospective_release_seal(
        root / 'release-seal',
        release=release,
        submissions_open_at=opening,
        timestamp_proof=_proof(target),
        proof_bytes=_PROOF_BYTES,
        decision_receipt_verifier=_release_decision_verifier(release),
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
        **_approval_kwargs(release),
        timestamp_verifier=_timestamp_verifier,
    )
    return release, opening, seal


class ProspectiveReleaseSealTest(unittest.TestCase):
    def test_binds_complete_fully_reverified_release_before_opening(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release, opening, seal = _build_seal(Path(temporary_directory))
            verifier_calls = 0

            def counting_verifier(proof: ProspectiveReleaseTimestampProof, payload: bytes) -> bool:
                nonlocal verifier_calls
                verifier_calls += 1
                return _timestamp_verifier(proof, payload)

            loaded = load_prospective_release_seal(
                seal.root,
                release=release,
                submissions_open_at=opening,
                decision_receipt_verifier=_release_decision_verifier(release),
                case_universe_seal_verifier=_case_verifier,
                source_capture_verifier=_source_capture_verifier,
                **_approval_kwargs(release),
                timestamp_verifier=counting_verifier,
            )

            self.assertEqual(verifier_calls, 1)
            self.assertEqual(loaded.target.prospective_release_sha256, release.release_sha256)
            expected_approval_sha256, _replay = _APPROVAL_CONTEXTS[release.root]
            self.assertEqual(
                loaded.target.tier_a_release_approval.approval_report_sha256,
                expected_approval_sha256,
            )
            self.assertEqual(
                loaded.target.tier_a_release_approval.verified_at,
                _replay.verified_at,
            )
            self.assertGreaterEqual(
                loaded.target.latest_prerequisite_witnessed_at,
                loaded.target.tier_a_release_approval.verified_at,
            )
            self.assertGreater(
                loaded.manifest.timestamp_proof.witnessed_at,
                loaded.target.tier_a_release_approval.verified_at,
            )
            self.assertTrue(loaded.target.tier_a_release_approval.release_scope.includes_model_leaderboard)
            self.assertEqual(
                loaded.target.schema_version,
                'vaxreplay.prospective-release-seal-target.v0.2',
            )
            self.assertEqual(loaded.target.release_id, release.manifest.release_id)
            self.assertEqual(loaded.target.challenge_id, release.challenge.manifest.challenge_id)
            self.assertEqual(loaded.target.prompt_variant, release.challenge.manifest.prompt_variant)
            self.assertEqual(loaded.target.challenge_sample_index, release.challenge.envelopes[0].sample_index)
            self.assertEqual(loaded.target.run_deadline_at, release.verified_admission.admission.run_deadline_at)
            self.assertGreater(loaded.target.release_tree_file_count, release.manifest.episode_count)
            self.assertGreater(loaded.target.release_tree_bytes, loaded.target.prospective_release_manifest_bytes)
            self.assertLess(loaded.manifest.timestamp_proof.witnessed_at, opening)
            self.assertLess(loaded.manifest.timestamp_proof.witnessed_at, loaded.target.run_deadline_at)
            self.assertEqual(
                {path.name for path in loaded.root.iterdir()},
                {'seal.json', 'target.json', 'timestamp-proof.bin'},
            )
            rendered = loaded.target.model_dump_json()
            for forbidden in ('labels_sha256', 'outcome', 'final_manifest'):
                self.assertNotIn(forbidden, rendered)

    def test_requires_fresh_original_input_replay_and_out_of_band_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            release = _build_approved_release(root)
            opening = _opening(release)
            digest, replay = _APPROVAL_CONTEXTS[release.root]

            with self.assertRaisesRegex(ValueError, 'out-of-band digest'):
                build_prospective_release_seal_target(
                    release,
                    submissions_open_at=opening,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    expected_approval_report_sha256='0' * 64,
                    approval_replay=replay,
                )

            artifacts = dict(replay.campaign_artifacts)
            artifact_id = next(iter(artifacts))
            artifacts[artifact_id] += b'tampered'
            changed_replay = replace(replay, campaign_artifacts=artifacts)
            with self.assertRaisesRegex(ValueError, 'approval replay failed'):
                build_prospective_release_seal_target(
                    release,
                    submissions_open_at=opening,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    expected_approval_report_sha256=digest,
                    approval_replay=changed_replay,
                )

            parameters = inspect.signature(build_prospective_release_seal_target).parameters
            self.assertNotIn('approval_report_bytes', parameters)
            self.assertNotIn('approval', parameters)
            self.assertNotIn('approval_verifier', parameters)
            self.assertIn('approval_replay', parameters)

    def test_replay_rejects_report_and_decision_verification_time_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build_approved_release(Path(temporary_directory))
            _digest, replay = _APPROVAL_CONTEXTS[release.root]
            approval = _APPROVAL_RESULTS[release.root]
            changed_report = approval.report.model_copy(
                update={'verified_at': approval.report.verified_at + timedelta(seconds=1)}
            )
            changed_approval = replace(approval, report=changed_report)
            changed_digest = hashlib.sha256(canonical_json_bytes(changed_report)).hexdigest()
            with (
                patch(
                    'vaxreplay.operations.prospective_release_approval_identity.'
                    'verify_and_materialize_tier_a_prospective_release',
                    return_value=changed_approval,
                ),
                self.assertRaisesRegex(ValueError, 'internally inconsistent release authorization'),
            ):
                reverify_tier_a_prospective_release_approval_identity(
                    expected_approval_report_sha256=changed_digest,
                    approval_replay=replay,
                )

    def test_target_model_rejects_nested_approval_cross_binding_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _release, _opening_at, seal = _build_seal(Path(temporary_directory))
            mismatches = {
                'release_id': 'different-release-id',
                'prospective_release_sha256': '0' * 64,
                'release_tree_sha256': '1' * 64,
                'challenge_bundle_sha256': '2' * 64,
                'episode_count': seal.target.episode_count + 1,
            }
            for field_name, changed_value in mismatches.items():
                with self.subTest(field_name=field_name):
                    payload = seal.target.model_dump(mode='python')
                    payload['tier_a_release_approval'] = seal.target.tier_a_release_approval.model_copy(
                        update={field_name: changed_value}
                    )
                    with self.assertRaisesRegex(ValidationError, 'direct fields differ'):
                        ProspectiveReleaseSealTarget.model_validate(payload)

    def test_rejects_approval_for_a_different_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            approved_release = _build_approved_release(root / 'approved')
            other_release = build_prospective_cohort_release(
                root / 'other',
                challenge_id='different-prospective-challenge',
                verified_admission=approved_release.verified_admission,
                case_universe_proof=approved_release.case_universe_proof,
                eligibility_protocol=approved_release.eligibility_protocol,
                verifier_policy=approved_release.verifier_policy,
                source_capture_policy=approved_release.source_capture_policy,
                attempt_policy=approved_release.attempt_policy,
                decision_receipt_verifier=_decision_verifier,
                case_universe_seal_verifier=_case_verifier,
                source_capture_verifier=_source_capture_verifier,
            )
            with self.assertRaisesRegex(ValueError, 'bound to different prospective release'):
                build_prospective_release_seal_target(
                    other_release,
                    submissions_open_at=_opening(approved_release),
                    decision_receipt_verifier=_decision_verifier,
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    **_approval_kwargs(approved_release),
                )

    def test_rejects_late_and_too_early_external_witnesses(self) -> None:
        for witness_kind in ('at_open', 'after_open', 'at_prerequisite', 'at_deadline'):
            with self.subTest(witness_kind=witness_kind), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                release = _build_approved_release(root)
                opening = _opening(release)
                target = _target(release, opening)
                if witness_kind == 'at_open':
                    witnessed_at = target.submissions_open_at
                    message = 'before submissions open'
                elif witness_kind == 'after_open':
                    witnessed_at = target.submissions_open_at + timedelta(seconds=1)
                    message = 'before submissions open'
                elif witness_kind == 'at_prerequisite':
                    witnessed_at = target.latest_prerequisite_witnessed_at
                    message = 'must follow every source'
                else:
                    witnessed_at = target.run_deadline_at
                    message = 'before submissions open'
                with self.assertRaisesRegex(ValueError, message):
                    build_prospective_release_seal(
                        root / 'release-seal',
                        release=release,
                        submissions_open_at=opening,
                        timestamp_proof=_proof(target, witnessed_at=witnessed_at),
                        proof_bytes=_PROOF_BYTES,
                        decision_receipt_verifier=_release_decision_verifier(release),
                        case_universe_seal_verifier=_case_verifier,
                        source_capture_verifier=_source_capture_verifier,
                        **_approval_kwargs(release),
                        timestamp_verifier=_timestamp_verifier,
                    )

    def test_rejects_non_timestamp_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build_approved_release(Path(temporary_directory))
            target = _target(release, _opening(release))
            with self.assertRaisesRegex(ValidationError, 'RFC 3161 or a public transparency log'):
                _proof(target, authority=TemporalReceiptAuthority.ORGANIZER_ATTESTATION)

    def test_loader_reruns_external_and_release_authority_verifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release, opening, seal = _build_seal(Path(temporary_directory))
            cases = (
                (
                    'release proof',
                    _release_decision_verifier(release),
                    _case_verifier,
                    lambda _proof, _payload: False,
                ),
                ('receipt verifier rejected', lambda _receipt, _proof: False, _case_verifier, _timestamp_verifier),
                (
                    'case-universe verifier rejected',
                    _release_decision_verifier(release),
                    lambda _seal, _proof: False,
                    _timestamp_verifier,
                ),
            )
            for message, decision_verifier, case_verifier, timestamp_verifier in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_prospective_release_seal(
                        seal.root,
                        release=release,
                        submissions_open_at=opening,
                        decision_receipt_verifier=decision_verifier,
                        case_universe_seal_verifier=case_verifier,
                        source_capture_verifier=_source_capture_verifier,
                        **_approval_kwargs(release),
                        timestamp_verifier=timestamp_verifier,
                    )

    def test_rejects_forged_or_structurally_loaded_release_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            release = _build_approved_release(root)
            opening = _opening(release)
            forged = replace(release, release_sha256='f' * 64)
            with self.assertRaisesRegex(ValueError, 'reverification failed'):
                _target(forged, opening)

            structural_challenge = load_prospective_challenge_bundle(release.challenge.root)
            structural_release = replace(release, challenge=structural_challenge)
            with self.assertRaisesRegex(ValueError, 'decision proofs were not reverified'):
                _target(structural_release, opening)

            with self.assertRaisesRegex(ValueError, 'fully loaded prospective cohort release'):
                build_prospective_release_seal_target(  # type: ignore[arg-type]
                    release.manifest,
                    submissions_open_at=opening,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    **_approval_kwargs(release),
                )

    def test_rejects_release_tree_tampering_after_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release, opening, seal = _build_seal(Path(temporary_directory))
            (release.root / 'verifier-policy.bin').write_bytes(b'tampered after sealing')
            with self.assertRaisesRegex(ValueError, 'cohort release reverification failed'):
                load_prospective_release_seal(
                    seal.root,
                    release=release,
                    submissions_open_at=opening,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    **_approval_kwargs(release),
                    timestamp_verifier=_timestamp_verifier,
                )

    def test_rejects_target_tampering_even_when_manifest_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release, opening, seal = _build_seal(Path(temporary_directory))
            target_path = seal.root / 'target.json'
            altered = seal.target.model_copy(update={'release_tree_sha256': '0' * 64})
            altered_bytes = canonical_json_bytes(altered)
            target_path.write_bytes(altered_bytes)
            manifest_path = seal.root / 'seal.json'
            proof = seal.manifest.timestamp_proof.model_copy(
                update={
                    'target_sha256': hashlib.sha256(altered_bytes).hexdigest(),
                    'target_bytes': len(altered_bytes),
                }
            )
            manifest = seal.manifest.model_copy(
                update={
                    'target_sha256': hashlib.sha256(altered_bytes).hexdigest(),
                    'target_bytes': len(altered_bytes),
                    'timestamp_proof': proof,
                }
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                ProspectiveReleaseSealIntegrityError,
                'direct fields differ from its Tier A approval identity',
            ):
                load_prospective_release_seal(
                    seal.root,
                    release=release,
                    submissions_open_at=opening,
                    decision_receipt_verifier=_release_decision_verifier(release),
                    case_universe_seal_verifier=_case_verifier,
                    source_capture_verifier=_source_capture_verifier,
                    **_approval_kwargs(release),
                    timestamp_verifier=lambda _proof, _payload: True,
                )

    def test_rejects_extra_files_noncanonical_json_and_symlinks(self) -> None:
        for tamper_kind in ('extra', 'noncanonical', 'symlink'):
            with self.subTest(tamper_kind=tamper_kind), tempfile.TemporaryDirectory() as temporary_directory:
                release, opening, seal = _build_seal(Path(temporary_directory))
                if tamper_kind == 'extra':
                    (seal.root / 'notes.txt').write_bytes(b'unbound')
                    message = 'allowlist mismatch'
                elif tamper_kind == 'noncanonical':
                    target_path = seal.root / 'target.json'
                    target_path.write_bytes(target_path.read_bytes() + b'\n')
                    message = 'canonical JSON'
                else:
                    proof_path = seal.root / 'timestamp-proof.bin'
                    proof_path.unlink()
                    os.symlink('target.json', proof_path)
                    message = 'cannot contain symlinks'
                with self.assertRaisesRegex(ProspectiveReleaseSealIntegrityError, message):
                    load_prospective_release_seal(
                        seal.root,
                        release=release,
                        submissions_open_at=opening,
                        decision_receipt_verifier=_release_decision_verifier(release),
                        case_universe_seal_verifier=_case_verifier,
                        source_capture_verifier=_source_capture_verifier,
                        **_approval_kwargs(release),
                        timestamp_verifier=_timestamp_verifier,
                    )


if __name__ == '__main__':
    unittest.main()
