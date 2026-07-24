from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import vaxreplay.prospective as prospective_module
from vaxreplay.bundle import EpisodeBundle, canonical_json_bytes
from vaxreplay.case_schema import Split
from vaxreplay.prospective import (
    ProspectiveDecisionPackageManifest,
    ProspectiveIntegrityError,
    SourceCaptureArtifact,
    build_prospective_decision_package,
    build_prospective_decision_seal,
    load_prospective_decision_package,
    load_prospective_decision_seal,
)
from vaxreplay.prospective_cli import main as prospective_main
from vaxreplay.temporal_schema import (
    DecisionTimeConfig,
    TemporalArtifactReceipt,
    TemporalArtifactRole,
    TemporalReceiptAuthority,
)


def _fixture() -> Path:
    return Path(__file__).parent / 'fixtures' / 'synthetic_preclinical_v1'


def _inputs(*, capture_version: int = 1):
    bundle = EpisodeBundle.load(_fixture())
    manifest = bundle.manifest.model_copy(update={'synthetic': False, 'split': Split.TEST})
    config = DecisionTimeConfig.from_manifest(manifest)
    protocols = {
        'candidate_set_definition': b'fixed complete-panel query, inclusion, exclusion, and ordering rules',
        'evidence_acquisition_spec': b'fixed source builds, queries, pages, mappings, and availability rules',
        'outcome_adjudication_spec': b'fixed endpoint, horizon, censoring, grade, and derivation rules',
    }
    source_manifest = canonical_json_bytes(
        {
            'schema_version': 'fictional.capture.v1',
            'source_id': 'fictional-source',
            'version': capture_version,
            'records_sha256': 'a' * 64,
        }
    )
    capture = SourceCaptureArtifact(
        source_id='fictional-source',
        source_release_at=config.decision_at - timedelta(days=3),
        captured_at=config.decision_at - timedelta(days=2),
        witnessed_at=config.decision_at - timedelta(days=1),
        manifest_bytes=source_manifest,
    )
    return bundle, config, protocols, capture


def _build_package(root: Path, *, capture_version: int = 1):
    bundle, config, protocols, capture = _inputs(capture_version=capture_version)
    package = build_prospective_decision_package(
        root,
        config=config,
        candidates=bundle.candidates,
        evidence=bundle.visible_evidence,
        protocol_artifacts=protocols,
        candidate_set_available_at=config.decision_at - timedelta(days=4),
        source_captures=(capture,),
    )
    return bundle, package


def _receipt_materials(package):
    witnessed = {
        TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL: (
            package.manifest.episode.decision_snapshot.protocol_commitments.candidate_set_available_at
        ),
        TemporalArtifactRole.EVIDENCE_SNAPSHOT: (package.manifest.episode.decision_snapshot.latest_visible_evidence_at),
        TemporalArtifactRole.DECISION_SNAPSHOT: package.manifest.episode.decision_at,
    }
    receipts = []
    proofs = {}
    for ordinal, request in enumerate(package.receipt_requests):
        receipt_id = f'receipt-{ordinal}'
        proof = f'verified fictional transparency proof {ordinal}'.encode()
        proofs[receipt_id] = proof
        receipts.append(
            TemporalArtifactReceipt(
                receipt_id=receipt_id,
                role=request.role,
                artifact_schema_version=request.artifact_schema_version,
                artifact_sha256=request.artifact_sha256,
                artifact_bytes=request.artifact_bytes,
                witnessed_at=witnessed[request.role],
                authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
                authority_id='fixture-transparency-log',
                receipt_sha256=hashlib.sha256(proof).hexdigest(),
                receipt_bytes=len(proof),
                verification_uri=f'https://transparency.invalid/{receipt_id}',
            )
        )
    return tuple(receipts), proofs


def _verifier(receipt: TemporalArtifactReceipt, proof: bytes) -> bool:
    return receipt.authority_id == 'fixture-transparency-log' and proof.startswith(
        b'verified fictional transparency proof'
    )


class ProspectiveDecisionPackageTest(unittest.TestCase):
    def test_builds_label_free_package_and_receipt_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, package = _build_package(root / 'package')

            decision_bytes = (root / 'package' / 'decision.json').read_bytes()
            self.assertNotIn(b'labels_sha256', decision_bytes)
            self.assertNotIn(b'outcome_snapshot', decision_bytes)
            self.assertNotIn(b'manifest_sha256', decision_bytes)
            self.assertEqual(package.manifest.schema_version, 'vaxreplay.prospective-decision-package.v0.3')
            self.assertEqual(
                package.manifest.episode.schema_version,
                'vaxreplay.prospective-episode-binding.v0.2',
            )
            self.assertEqual(
                package.manifest.source_captures[0].witnessed_at,
                package.manifest.episode.decision_at - timedelta(days=1),
            )
            self.assertEqual(len(package.receipt_requests), 3)
            self.assertEqual(
                tuple(request.role for request in package.receipt_requests),
                (
                    TemporalArtifactRole.CANDIDATE_UNIVERSE_OR_PANEL,
                    TemporalArtifactRole.EVIDENCE_SNAPSHOT,
                    TemporalArtifactRole.DECISION_SNAPSHOT,
                ),
            )
            self.assertEqual(load_prospective_decision_package(root / 'package').manifest, package.manifest)

            poisoned = {
                **package.manifest.model_dump(mode='json'),
                'labels_sha256': 'f' * 64,
            }
            with self.assertRaises(ValidationError):
                ProspectiveDecisionPackageManifest.model_validate(poisoned)
            with self.assertRaises(ValidationError):
                ProspectiveDecisionPackageManifest.model_validate(
                    package.manifest.model_copy(
                        update={'schema_version': 'vaxreplay.prospective-decision-package.v0.2'}
                    ).model_dump()
                )

    def test_source_capture_changes_package_identity_but_not_decision_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, first = _build_package(root / 'first', capture_version=1)
            _bundle, second = _build_package(root / 'second', capture_version=2)

            self.assertEqual(
                first.manifest.episode.decision_snapshot_sha256,
                second.manifest.episode.decision_snapshot_sha256,
            )
            self.assertNotEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertEqual(first.receipt_requests[:2], second.receipt_requests[:2])
            self.assertNotEqual(first.receipt_requests[2], second.receipt_requests[2])
            self.assertEqual(
                first.receipt_requests[2].artifact_schema_version,
                'vaxreplay.prospective-decision-context.v0.1',
            )
            with self.assertRaisesRegex(ValidationError, 'exact source-capture decision context'):
                ProspectiveDecisionPackageManifest.model_validate(
                    {
                        **first.manifest.model_dump(),
                        'episode': first.manifest.episode.model_copy(update={'decision_context_sha256': 'f' * 64}),
                    }
                )

            receipts, proofs = _receipt_materials(first)
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'does not bind the requested artifact'):
                build_prospective_decision_seal(
                    root / 'reused-receipt-seal',
                    package=second,
                    receipts=receipts,
                    proof_artifacts=proofs,
                    receipt_verifier=_verifier,
                    verified_at=second.manifest.episode.decision_at + timedelta(seconds=1),
                )

    def test_promotion_namespace_rejects_unknown_schema_and_hybrid_lineage(self) -> None:
        bundle, config, protocols, legacy_capture = _inputs()
        unknown_promotion = SourceCaptureArtifact(
            source_id='promotion:fake',
            source_release_at=legacy_capture.source_release_at,
            captured_at=legacy_capture.captured_at,
            witnessed_at=legacy_capture.witnessed_at,
            manifest_bytes=canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.promotion-handoff.v9.9',
                    'promotion_id': 'fake',
                }
            ),
        )
        unknown_handoff_outside_namespace = SourceCaptureArtifact(
            source_id='legacy-handoff',
            source_release_at=legacy_capture.source_release_at,
            captured_at=legacy_capture.captured_at,
            witnessed_at=legacy_capture.witnessed_at,
            manifest_bytes=canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.promotion-handoff.v9.9',
                    'promotion_id': 'fake',
                }
            ),
        )
        unknown_capture_index_outside_namespace = SourceCaptureArtifact(
            source_id='legacy-index',
            source_release_at=legacy_capture.source_release_at,
            captured_at=legacy_capture.captured_at,
            witnessed_at=legacy_capture.witnessed_at,
            manifest_bytes=canonical_json_bytes(
                {
                    'schema_version': 'vaxreplay.capture-index.v9.9',
                    'promotion_id': 'fake',
                }
            ),
        )
        bridge_archive_schemas_outside_namespace = tuple(
            SourceCaptureArtifact(
                source_id=f'legacy-{case}',
                source_release_at=legacy_capture.source_release_at,
                captured_at=legacy_capture.captured_at,
                witnessed_at=legacy_capture.witnessed_at,
                manifest_bytes=canonical_json_bytes(
                    {
                        'schema_version': schema_version,
                        'promotion_id': 'fake',
                    }
                ),
            )
            for case, schema_version in (
                ('capture-promotion-current', 'vaxreplay.capture-promotion.v0.6'),
                ('capture-promotion-future', 'vaxreplay.capture-promotion.v9.9'),
                ('scope-precommit-current', 'vaxreplay.scope-precommit.v0.2'),
                ('scope-precommit-future', 'vaxreplay.scope-precommit.v9.9'),
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            common = {
                'config': config,
                'candidates': bundle.candidates,
                'evidence': bundle.visible_evidence,
                'protocol_artifacts': protocols,
                'candidate_set_available_at': config.decision_at - timedelta(days=4),
            }
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'unknown or invalid handoff schema'):
                build_prospective_decision_package(
                    root / 'unknown',
                    source_captures=(unknown_promotion,),
                    **common,
                )
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'cannot mix promotion and legacy'):
                build_prospective_decision_package(
                    root / 'hybrid',
                    source_captures=(legacy_capture, unknown_promotion),
                    **common,
                )
            for case, capture in (
                ('handoff', unknown_handoff_outside_namespace),
                ('capture-index', unknown_capture_index_outside_namespace),
                *(
                    (capture.source_id.removeprefix('legacy-'), capture)
                    for capture in bridge_archive_schemas_outside_namespace
                ),
            ):
                with (
                    self.subTest(case=case),
                    self.assertRaisesRegex(ProspectiveIntegrityError, 'unknown or invalid handoff schema'),
                ):
                    build_prospective_decision_package(
                        root / f'unknown-outside-{case}',
                        source_captures=(capture,),
                        **common,
                    )

    def test_rejects_future_evidence_instead_of_silently_filtering_it(self) -> None:
        bundle, config, protocols, capture = _inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, 'after decision_at'):
                build_prospective_decision_package(
                    Path(temporary_directory).resolve() / 'package',
                    config=config,
                    candidates=bundle.candidates,
                    evidence=bundle.evidence,
                    protocol_artifacts=protocols,
                    candidate_set_available_at=config.decision_at - timedelta(days=4),
                    source_captures=(capture,),
                )

    def test_requires_ordered_source_capture_witness_before_decision(self) -> None:
        bundle, config, protocols, capture = _inputs()
        with self.assertRaisesRegex(ValueError, 'cannot predate capture completion'):
            SourceCaptureArtifact(
                source_id=capture.source_id,
                source_release_at=capture.source_release_at,
                captured_at=capture.captured_at,
                witnessed_at=capture.captured_at - timedelta(seconds=1),
                manifest_bytes=capture.manifest_bytes,
            )

        late_witness = SourceCaptureArtifact(
            source_id=capture.source_id,
            source_release_at=capture.source_release_at,
            captured_at=capture.captured_at,
            witnessed_at=config.decision_at + timedelta(seconds=1),
            manifest_bytes=capture.manifest_bytes,
        )
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(
                ValueError,
                'externally witnessed at or before decision_at',
            ),
        ):
            build_prospective_decision_package(
                Path(temporary_directory).resolve() / 'package',
                config=config,
                candidates=bundle.candidates,
                evidence=bundle.visible_evidence,
                protocol_artifacts=protocols,
                candidate_set_available_at=config.decision_at - timedelta(days=4),
                source_captures=(late_witness,),
            )

    def test_loader_rejects_tampering_extra_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, _package = _build_package(root / 'original')
            for case in ('tamper', 'extra', 'symlink'):
                copied = root / case
                shutil.copytree(root / 'original', copied)
                if case == 'tamper':
                    (copied / 'evidence.jsonl').write_bytes(b'{}\n')
                    message = 'does not match'
                elif case == 'extra':
                    (copied / 'future-labels.json').write_text('{}', encoding='utf-8')
                    message = 'allowlist'
                else:
                    (copied / 'leak').symlink_to(_fixture() / 'private')
                    message = 'symlink'
                with self.subTest(case=case), self.assertRaisesRegex(ProspectiveIntegrityError, message):
                    load_prospective_decision_package(copied)

    def test_loader_rejects_declared_symlink_and_in_place_read_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, _package = _build_package(root / 'original')

            linked = root / 'linked'
            shutil.copytree(root / 'original', linked)
            (linked / 'evidence.jsonl').unlink()
            (linked / 'evidence.jsonl').symlink_to('candidates.jsonl')
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'symlink'):
                load_prospective_decision_package(linked)

            raced = root / 'raced'
            shutil.copytree(root / 'original', raced)
            evidence_path = raced / 'evidence.jsonl'
            evidence_identity = (evidence_path.stat().st_dev, evidence_path.stat().st_ino)
            original_read = prospective_module.os.read
            mutated = False

            def mutate_during_read(descriptor: int, byte_count: int) -> bytes:
                nonlocal mutated
                payload = original_read(descriptor, byte_count)
                metadata = prospective_module.os.fstat(descriptor)
                if not mutated and payload and (metadata.st_dev, metadata.st_ino) == evidence_identity:
                    with evidence_path.open('ab') as handle:
                        handle.write(b'x')
                    mutated = True
                return payload

            with patch.object(prospective_module.os, 'read', side_effect=mutate_during_read):
                with self.assertRaisesRegex(ProspectiveIntegrityError, 'changed while read'):
                    load_prospective_decision_package(raced)
            self.assertTrue(mutated)

    def test_loader_detects_root_replacement_after_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            package_path = root / 'package'
            displaced_path = root / 'displaced'
            _bundle, _package = _build_package(package_path)
            original_read = prospective_module._read_regular_file_snapshot_at
            replaced = False

            def replace_root_before_artifact_read(
                descriptor: int,
                relative_path: str,
                maximum_bytes: int,
            ):
                nonlocal replaced
                if relative_path == 'candidates.jsonl' and not replaced:
                    package_path.rename(displaced_path)
                    package_path.mkdir()
                    replaced = True
                return original_read(descriptor, relative_path, maximum_bytes)

            with patch.object(
                prospective_module,
                '_read_regular_file_snapshot_at',
                side_effect=replace_root_before_artifact_read,
            ):
                with self.assertRaisesRegex(ProspectiveIntegrityError, 'root changed while being read'):
                    load_prospective_decision_package(package_path)
            self.assertTrue(replaced)

    def test_loader_rejects_symlink_in_artifact_root_parent_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            real_parent = root / 'real-parent'
            real_parent.mkdir()
            _bundle, _package = _build_package(real_parent / 'package')
            link_parent = root / 'link-parent'
            link_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(ProspectiveIntegrityError, 'cannot open .* root'):
                load_prospective_decision_package(link_parent / 'package')

    def test_publication_exclusively_preserves_a_racing_target_and_foreign_lock(self) -> None:
        bundle, config, protocols, capture = _inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            target = root / 'package'
            original_rename = prospective_module._rename_directory_noreplace

            def create_target_at_install(source: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / 'owner.txt').write_bytes(b'created by racing publisher')
                original_rename(source, destination)

            with patch.object(
                prospective_module,
                '_rename_directory_noreplace',
                side_effect=create_target_at_install,
            ):
                with self.assertRaisesRegex(FileExistsError, 'already exists'):
                    build_prospective_decision_package(
                        target,
                        config=config,
                        candidates=bundle.candidates,
                        evidence=bundle.visible_evidence,
                        protocol_artifacts=protocols,
                        candidate_set_available_at=config.decision_at - timedelta(days=4),
                        source_captures=(capture,),
                    )
            self.assertEqual((target / 'owner.txt').read_bytes(), b'created by racing publisher')
            self.assertFalse((root / '.package.publish.lock').exists())
            self.assertFalse(any(path.name.startswith('.package.') for path in root.iterdir()))

            locked_target = root / 'locked-package'
            foreign_lock = root / '.locked-package.publish.lock'
            foreign_lock.write_bytes(b'foreign publisher')
            with self.assertRaisesRegex(ValueError, 'already locked'):
                build_prospective_decision_package(
                    locked_target,
                    config=config,
                    candidates=bundle.candidates,
                    evidence=bundle.visible_evidence,
                    protocol_artifacts=protocols,
                    candidate_set_available_at=config.decision_at - timedelta(days=4),
                    source_captures=(capture,),
                )
            self.assertEqual(foreign_lock.read_bytes(), b'foreign publisher')

    def test_seal_requires_and_reverifies_independent_timestamp_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, package = _build_package(root / 'package')
            receipts, proofs = _receipt_materials(package)
            seal = build_prospective_decision_seal(
                root / 'seal',
                package=package,
                receipts=receipts,
                proof_artifacts=proofs,
                receipt_verifier=_verifier,
                verified_at=package.manifest.episode.decision_at + timedelta(seconds=1),
            )
            self.assertEqual(seal.manifest.decision_package_sha256, package.manifest_sha256)
            self.assertEqual(
                load_prospective_decision_seal(
                    root / 'seal',
                    package=package,
                    receipt_verifier=_verifier,
                ).manifest,
                seal.manifest,
            )

            copied = root / 'tampered-seal'
            shutil.copytree(root / 'seal', copied)
            (copied / 'proofs' / '000001.bin').write_bytes(b'forged')
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'does not match'):
                load_prospective_decision_seal(copied, package=package, receipt_verifier=_verifier)

            linked = root / 'linked-seal'
            shutil.copytree(root / 'seal', linked)
            proof_path = linked / 'proofs' / '000001.bin'
            proof_path.unlink()
            proof_path.symlink_to('000000.bin')
            with self.assertRaisesRegex(ProspectiveIntegrityError, 'symlink'):
                load_prospective_decision_seal(linked, package=package, receipt_verifier=_verifier)

    def test_seal_rejects_self_attestation_and_late_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, package = _build_package(root / 'package')
            receipts, proofs = _receipt_materials(package)
            cases = (
                (
                    'self-attested',
                    tuple(
                        receipt.model_copy(update={'authority_type': TemporalReceiptAuthority.ORGANIZER_ATTESTATION})
                        if ordinal == 0
                        else receipt
                        for ordinal, receipt in enumerate(receipts)
                    ),
                    'external timestamp authority',
                ),
                (
                    'late',
                    tuple(
                        receipt.model_copy(
                            update={'witnessed_at': package.manifest.episode.decision_at + timedelta(seconds=1)}
                        )
                        if ordinal == 2
                        else receipt
                        for ordinal, receipt in enumerate(receipts)
                    ),
                    'by decision_at',
                ),
            )
            for case, bad_receipts, message in cases:
                with self.subTest(case=case), self.assertRaisesRegex(ProspectiveIntegrityError, message):
                    build_prospective_decision_seal(
                        root / f'seal-{case}',
                        package=package,
                        receipts=bad_receipts,
                        proof_artifacts=proofs,
                        receipt_verifier=_verifier,
                        verified_at=package.manifest.episode.decision_at + timedelta(seconds=2),
                    )

    def test_seal_publication_exclusively_preserves_a_racing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            _bundle, package = _build_package(root / 'package')
            receipts, proofs = _receipt_materials(package)
            target = root / 'seal'
            original_rename = prospective_module._rename_directory_noreplace

            def create_target_at_install(source: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / 'owner.txt').write_bytes(b'created by racing seal publisher')
                original_rename(source, destination)

            with patch.object(
                prospective_module,
                '_rename_directory_noreplace',
                side_effect=create_target_at_install,
            ):
                with self.assertRaisesRegex(FileExistsError, 'already exists'):
                    build_prospective_decision_seal(
                        target,
                        package=package,
                        receipts=receipts,
                        proof_artifacts=proofs,
                        receipt_verifier=_verifier,
                        verified_at=package.manifest.episode.decision_at + timedelta(seconds=1),
                    )
            self.assertEqual((target / 'owner.txt').read_bytes(), b'created by racing seal publisher')
            self.assertFalse((root / '.seal.publish.lock').exists())
            self.assertFalse(any(path.name.startswith('.seal.') for path in root.iterdir()))

    def test_operator_cli_builds_package_and_explicitly_refuses_tier_a_claim(self) -> None:
        bundle, config, protocols, capture = _inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            (root / 'config.json').write_bytes(canonical_json_bytes(config))
            (root / 'candidates.jsonl').write_bytes(
                b''.join(canonical_json_bytes(record) + b'\n' for record in bundle.candidates)
            )
            (root / 'evidence.jsonl').write_bytes(
                b''.join(canonical_json_bytes(record) + b'\n' for record in bundle.visible_evidence)
            )
            for name, payload in protocols.items():
                (root / f'{name}.bin').write_bytes(payload)
            (root / 'source-capture.json').write_bytes(capture.manifest_bytes)
            (root / 'capture-index.json').write_bytes(
                canonical_json_bytes(
                    {
                        'captures': [
                            {
                                'source_id': capture.source_id,
                                'source_release_at': capture.source_release_at.isoformat(),
                                'captured_at': capture.captured_at.isoformat(),
                                'witnessed_at': capture.witnessed_at.isoformat(),
                                'manifest_path': 'source-capture.json',
                            }
                        ]
                    }
                )
            )
            argv = [
                'vaxreplay-prospective',
                'build-package',
                '--config',
                str(root / 'config.json'),
                '--candidates-jsonl',
                str(root / 'candidates.jsonl'),
                '--evidence-jsonl',
                str(root / 'evidence.jsonl'),
                '--candidate-set-definition',
                str(root / 'candidate_set_definition.bin'),
                '--evidence-acquisition-spec',
                str(root / 'evidence_acquisition_spec.bin'),
                '--outcome-adjudication-spec',
                str(root / 'outcome_adjudication_spec.bin'),
                '--candidate-set-available-at',
                (config.decision_at - timedelta(days=4)).isoformat(),
                '--capture-index',
                str(root / 'capture-index.json'),
                '--output-dir',
                str(root / 'package'),
            ]
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(output):
                prospective_main()

            summary = json.loads(output.getvalue())
            self.assertFalse(summary['tier_a_eligible'])
            self.assertEqual(summary['receipt_request_count'], 3)
            self.assertTrue((root / 'package' / 'decision.json').is_file())


if __name__ == '__main__':
    unittest.main()
