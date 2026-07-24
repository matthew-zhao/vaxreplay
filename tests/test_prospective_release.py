from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_inventory import (
    CaseUniverseDisposition,
    CaseUniverseEntry,
    CaseUniverseManifest,
    CaseUniverseSeal,
    case_universe_content_sha256,
)
from vaxreplay.case_schema import Split
from vaxreplay.prospective import (
    SourceCaptureArtifact,
    build_prospective_decision_package,
    build_prospective_decision_seal,
)
from vaxreplay.prospective_admission import build_verified_prospective_admission
from vaxreplay.prospective_release import (
    ProspectiveCohortReleaseManifest,
    ProspectiveReleaseIntegrityError,
    build_prospective_cohort_release,
    load_prospective_cohort_release,
)
from vaxreplay.prospective_schema import ProspectiveAttemptPolicy, ProspectiveSplitInventory
from vaxreplay.temporal_schema import (
    DecisionTimeConfig,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalReceiptAuthority,
)

_CASE_PROOF = b'verified fictional prospective case-universe proof'
_ELIGIBILITY = b'fixed exhaustive eligibility query, inclusion, exclusion, and ordering protocol'
_VERIFIER_POLICY = b'fixed verifier authority allowlist, signature rules, and failure semantics'
_SOURCE_CAPTURE_POLICY = b'fixed source-capture schema, replay, provenance, and timestamp eligibility rules'
_ATTEMPT_POLICY = canonical_json_bytes(ProspectiveAttemptPolicy())


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _decision_verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return receipt.authority_id == 'fixture-public-log' and proof.startswith(b'decision-proof:')


def _case_verifier(seal: CaseUniverseSeal, proof: bytes) -> bool:
    return seal.authority_id == 'fixture-public-log' and proof == _CASE_PROOF


def _source_capture_verifier(binding, manifest_bytes: bytes, policy: bytes) -> bool:  # noqa: ANN001
    expected = canonical_json_bytes(
        {
            'schema_version': 'fictional.capture.v1',
            'source_id': 'fictional-source',
            'records_sha256': 'a' * 64,
        }
    )
    return policy == _SOURCE_CAPTURE_POLICY and binding.source_id == 'fictional-source' and manifest_bytes == expected


def _admission(root: Path):
    bundle = EpisodeBundle.load(_fixture())
    config = DecisionTimeConfig.from_manifest(
        bundle.manifest.model_copy(update={'synthetic': False, 'split': Split.TEST})
    )
    package = build_prospective_decision_package(
        root / 'package',
        config=config,
        candidates=bundle.candidates,
        evidence=bundle.visible_evidence,
        protocol_artifacts={
            'candidate_set_definition': b'complete panel rules fixed before cutoff',
            'evidence_acquisition_spec': b'source release and evidence availability rules',
            'outcome_adjudication_spec': b'endpoint, horizon, censoring, and derivation rules',
        },
        candidate_set_available_at=config.decision_at - timedelta(days=4),
        source_captures=(
            SourceCaptureArtifact(
                source_id='fictional-source',
                source_release_at=config.decision_at - timedelta(days=3),
                captured_at=config.decision_at - timedelta(days=2),
                witnessed_at=config.decision_at - timedelta(days=1),
                manifest_bytes=canonical_json_bytes(
                    {
                        'schema_version': 'fictional.capture.v1',
                        'source_id': 'fictional-source',
                        'records_sha256': 'a' * 64,
                    }
                ),
            ),
        ),
    )
    witnessed_at = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            package.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (package.manifest.episode.decision_snapshot.latest_visible_evidence_at),
        TemporalArtifactRole.DECISION_SNAPSHOT: config.decision_at,
    }
    receipts: list[TemporalArtifactReceipt] = []
    proofs: dict[str, bytes] = {}
    for ordinal, request in enumerate(package.receipt_requests):
        receipt_id = f'receipt-{ordinal}'
        proof = f'decision-proof:{ordinal}'.encode()
        proofs[receipt_id] = proof
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=request.role,
                artifact_schema_version=request.artifact_schema_version,
                artifact_sha256=request.artifact_sha256,
                artifact_bytes=request.artifact_bytes,
                witnessed_at=witnessed_at[request.role],
                authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
                authority_id='fixture-public-log',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://log.invalid/{receipt_id}',
            )
        )
    decision_seal = build_prospective_decision_seal(
        root / 'decision-seal',
        package=package,
        receipts=tuple(receipts),
        proof_artifacts=proofs,
        receipt_verifier=_decision_verifier,
        verified_at=config.decision_at + timedelta(seconds=1),
    )

    entry = CaseUniverseEntry(
        case_id='fictional-case-1',
        lineage_group_id=package.manifest.episode.lineage_group_id,
        disposition=CaseUniverseDisposition.PREELIGIBLE,
        decision_package_sha256=package.manifest_sha256,
    )
    eligibility_hash = hashlib.sha256(_ELIGIBILITY).hexdigest()
    universe_content_hash = case_universe_content_sha256(
        universe_id='prospective-universe-1',
        eligibility_protocol_sha256=eligibility_hash,
        entries=(entry,),
    )
    case_universe = CaseUniverseManifest(
        universe_id='prospective-universe-1',
        eligibility_protocol_sha256=eligibility_hash,
        entries=(entry,),
        universe_content_sha256=universe_content_hash,
        seal=CaseUniverseSeal(
            universe_content_sha256=universe_content_hash,
            witnessed_at=config.decision_at,
            authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
            authority_id='fixture-public-log',
            proof_sha256=hashlib.sha256(_CASE_PROOF).hexdigest(),
            proof_bytes=len(_CASE_PROOF),
            verification_uri='https://log.invalid/case-universe',
        ),
    )
    split_inventory = ProspectiveSplitInventory(
        inventory_id='prospective-split-1',
        episodes=(package.manifest.episode,),
    )
    verified = build_verified_prospective_admission(
        release_id='prospective-release-1',
        suite_id='prospective-suite-1',
        packages=(package,),
        seals=(decision_seal,),
        split_inventory=split_inventory,
        case_universe=case_universe,
        case_universe_proof=_CASE_PROOF,
        eligibility_protocol=_ELIGIBILITY,
        verifier_policy=_VERIFIER_POLICY,
        source_capture_policy=_SOURCE_CAPTURE_POLICY,
        attempt_policy=_ATTEMPT_POLICY,
        run_deadline_at=config.decision_at + timedelta(days=1),
        receipt_verifier=_decision_verifier,
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
    )
    return verified


def _build(root: Path):
    verified = _admission(root)
    release = build_prospective_cohort_release(
        root / 'release',
        challenge_id='prospective-challenge-1',
        verified_admission=verified,
        case_universe_proof=_CASE_PROOF,
        eligibility_protocol=_ELIGIBILITY,
        verifier_policy=_VERIFIER_POLICY,
        source_capture_policy=_SOURCE_CAPTURE_POLICY,
        attempt_policy=_ATTEMPT_POLICY,
        decision_receipt_verifier=_decision_verifier,
        case_universe_seal_verifier=_case_verifier,
        source_capture_verifier=_source_capture_verifier,
    )
    return release


def _load(
    root: Path,
    *,
    decision_verifier=_decision_verifier,
    case_verifier=_case_verifier,
    source_verifier=_source_capture_verifier,
):
    return load_prospective_cohort_release(
        root,
        decision_receipt_verifier=decision_verifier,
        case_universe_seal_verifier=case_verifier,
        source_capture_verifier=source_verifier,
    )


class ProspectiveCohortReleaseTest(unittest.TestCase):
    def test_packages_and_reconstructs_the_exact_verified_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build(Path(temporary_directory))
            loaded = _load(release.root)

            self.assertEqual(loaded.release_sha256, release.release_sha256)
            self.assertEqual(loaded.verified_admission, release.verified_admission)
            self.assertEqual(loaded.challenge.admission, release.verified_admission.admission)
            self.assertEqual(
                loaded.verified_admission.admission.schema_version,
                'vaxreplay.prospective-challenge-admission.v0.3',
            )
            self.assertEqual(loaded.verified_admission.admission.purpose, 'prospective_research')
            self.assertEqual(loaded.verified_admission.suite.schema_version, 'vaxreplay.prospective-suite.v0.2')
            self.assertEqual(
                loaded.verified_admission.split_inventory.schema_version,
                'vaxreplay.prospective-split-inventory.v0.2',
            )
            self.assertTrue(loaded.challenge.authority_proofs_reverified)
            self.assertEqual(loaded.eligibility_protocol, _ELIGIBILITY)
            self.assertEqual(loaded.verifier_policy, _VERIFIER_POLICY)
            self.assertEqual(loaded.source_capture_policy, _SOURCE_CAPTURE_POLICY)
            self.assertEqual(loaded.attempt_policy, _ATTEMPT_POLICY)
            self.assertEqual(loaded.case_universe_proof, _CASE_PROOF)

            manifest_json = loaded.manifest.model_dump_json()
            for forbidden in ('labels_sha256', 'outcome', 'final_manifest', 'manifest_sha256'):
                self.assertNotIn(forbidden, manifest_json)
            self.assertEqual(
                set(path.name for path in loaded.root.iterdir()),
                {
                    'release.json',
                    'challenge',
                    'split-inventory.json',
                    'case-universe.json',
                    'case-universe-proof.bin',
                    'eligibility-protocol.bin',
                    'verifier-policy.bin',
                    'source-capture-policy.bin',
                    'attempt-policy.bin',
                },
            )

    def test_rejects_tampered_bound_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build(Path(temporary_directory))
            (release.root / 'verifier-policy.bin').write_bytes(b'tampered')
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'artifact changed'):
                _load(release.root)

    def test_rejects_extra_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build(Path(temporary_directory))
            extra = release.root / 'uncommitted'
            extra.mkdir()
            (extra / 'notes.txt').write_bytes(b'uncommitted')
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'file allowlist mismatch'):
                _load(release.root)

    def test_rejects_file_and_directory_symlinks(self) -> None:
        for link_kind in ('file', 'directory'):
            with self.subTest(link_kind=link_kind), tempfile.TemporaryDirectory() as temporary_directory:
                release = _build(Path(temporary_directory))
                if link_kind == 'file':
                    os.symlink('verifier-policy.bin', release.root / 'uncommitted-link')
                else:
                    os.symlink('challenge', release.root / 'uncommitted-link')
                with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'symlink'):
                    _load(release.root)

    def test_rejects_wrong_policy_even_if_manifest_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build(Path(temporary_directory))
            policy_path = release.root / 'attempt-policy.bin'
            wrong_policy = b'a different attempt policy'
            policy_path.write_bytes(wrong_policy)
            manifest_path = release.root / 'release.json'
            manifest = ProspectiveCohortReleaseManifest.model_validate_json(manifest_path.read_bytes())
            bindings = tuple(
                binding.model_copy(
                    update={
                        'sha256': hashlib.sha256(wrong_policy).hexdigest(),
                        'byte_count': len(wrong_policy),
                    }
                )
                if binding.path == 'attempt-policy.bin'
                else binding
                for binding in manifest.files
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest.model_copy(update={'files': bindings})))
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'reconstructed prospective admission'):
                _load(release.root)

    def test_rejects_bad_decision_and_case_universe_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = _build(Path(temporary_directory))
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'receipt verifier rejected'):
                _load(release.root, decision_verifier=lambda _receipt, _proof: False)
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'case-universe verifier rejected'):
                _load(release.root, case_verifier=lambda _seal, _proof: False)
            with self.assertRaisesRegex(ProspectiveReleaseIntegrityError, 'source-capture verifier rejected'):
                _load(release.root, source_verifier=lambda _binding, _manifest, _policy: False)


if __name__ == '__main__':
    unittest.main()
