from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pydantic import ValidationError

import vaxreplay.operations.witness as witness_module
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.schema import LedgerCheckpoint, checkpoint_bytes, checkpoint_sha256
from vaxreplay.operations.witness import (
    AuthenticatedExternalWitnessFacts,
    ExternalCheckpointWitnessProvider,
    ExternalWitnessClaim,
    ExternalWitnessMethod,
    TrustedCheckpointWitnessVerifier,
    WitnessPolicyBinding,
    WitnessVerificationError,
    broker_witness_checkpoint,
    load_witnessed_checkpoint,
)

_T0 = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
_PROOF = b'fictional external transparency inclusion proof over the checkpoint digest'


def _checkpoint() -> LedgerCheckpoint:
    return LedgerCheckpoint(
        store_id='a' * 32,
        created_at=_T0,
        through_sequence=7,
        through_event_sha256='b' * 64,
        object_count=3,
        object_inventory_sha256='c' * 64,
    )


def _policy() -> WitnessPolicyBinding:
    return WitnessPolicyBinding(
        authority_id='independent-log-operator',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id='checkpoint-witness-policy-v1',
        policy_sha256='d' * 64,
        trust_policy_id='checkpoint-trust-roots-2026-07',
        trust_policy_sha256='e' * 64,
        verifier_id='offline-rekor-verifier',
        verifier_implementation_sha256='f' * 64,
    )


def _claim() -> ExternalWitnessClaim:
    return ExternalWitnessClaim(
        verification_uri='https://log.invalid/entries/checkpoint-receipt-1',
    )


def _facts(
    *,
    witnessed_at: datetime | None = None,
    authority_id: str | None = None,
    checkpoint_digest: str | None = None,
) -> AuthenticatedExternalWitnessFacts:
    return AuthenticatedExternalWitnessFacts(
        receipt_id='checkpoint-receipt-1',
        authority_id=authority_id or _policy().authority_id,
        witness_id='independent-log-key-2026',
        method=ExternalWitnessMethod.PUBLIC_TRANSPARENCY_LOG,
        policy_id=_policy().policy_id,
        checkpoint_sha256=checkpoint_digest or checkpoint_sha256(_checkpoint()),
        witnessed_at=witnessed_at or _T0 + timedelta(seconds=1),
    )


class OperationsCheckpointWitnessTest(unittest.TestCase):
    def test_broker_submits_only_commitment_and_persists_offline_reverifiable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve() / 'witness'
            requests = []
            verifier_calls = []

            def provider(request):
                requests.append(request)
                return _claim(), _PROOF

            def verifier(target_bytes, proof_bytes, policy):
                verifier_calls.append((target_bytes, proof_bytes, policy))
                if target_bytes != checkpoint_bytes(_checkpoint()) or proof_bytes != _PROOF or policy != _policy():
                    raise ValueError('unexpected trusted-verifier input')
                return _facts()

            built = broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=provider,
                verifier=verifier,
                verified_at=_T0 + timedelta(seconds=2),
            )

            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertEqual(request.checkpoint_sha256, checkpoint_sha256(_checkpoint()))
            self.assertEqual(request.checkpoint_bytes, len(checkpoint_bytes(_checkpoint())))
            self.assertNotIn('checkpoint', request.model_dump(exclude={'checkpoint_sha256', 'checkpoint_bytes'}))
            self.assertEqual(
                {path.name for path in root.iterdir()}, {'checkpoint.json', 'external-proof.bin', 'witness.json'}
            )
            self.assertEqual(built.checkpoint_bytes, checkpoint_bytes(_checkpoint()))
            self.assertEqual(built.proof_bytes, _PROOF)
            self.assertTrue(built.proof_reverified)
            self.assertFalse(built.manifest.tier_a_eligibility_established)
            self.assertTrue(built.manifest.external_proof_verification_required)
            self.assertEqual(built.witnessed_at, _T0 + timedelta(seconds=1))
            self.assertEqual(len(verifier_calls), 2)  # pre-write verification plus load-back verification

            loaded = load_witnessed_checkpoint(
                root,
                verifier=verifier,
                expected_policy=_policy(),
                verified_at=_T0 + timedelta(seconds=3),
                expected_checkpoint_sha256=checkpoint_sha256(_checkpoint()),
            )
            self.assertEqual(loaded, built)
            self.assertEqual(len(verifier_calls), 3)

    def test_provider_and_verifier_are_mandatory_and_fail_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / 'witness'
            provider_called = False

            def provider(_request):
                nonlocal provider_called
                provider_called = True
                return _claim(), _PROOF

            with self.assertRaisesRegex(ValueError, 'provider is required'):
                broker_witness_checkpoint(
                    root,
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=cast(ExternalCheckpointWitnessProvider, None),
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                )
            with self.assertRaisesRegex(ValueError, 'trusted checkpoint witness verifier is required'):
                broker_witness_checkpoint(
                    root,
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=provider,
                    verifier=cast(TrustedCheckpointWitnessVerifier, None),
                )
            self.assertFalse(provider_called)
            self.assertFalse(root.exists())

            built = broker_witness_checkpoint(
                parent / 'valid',
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=provider,
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            with self.assertRaisesRegex(WitnessVerificationError, 'trusted checkpoint witness verifier is required'):
                load_witnessed_checkpoint(
                    built.root,
                    verifier=cast(TrustedCheckpointWitnessVerifier, None),
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )

    def test_trusted_verifier_rejection_or_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()

            def provider(_request):
                return _claim(), _PROOF

            with self.assertRaisesRegex(WitnessVerificationError, 'rejected'):
                broker_witness_checkpoint(
                    parent / 'rejected',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=provider,
                    verifier=lambda _target, _proof, _policy_binding: cast(AuthenticatedExternalWitnessFacts, False),
                    verified_at=_T0 + timedelta(seconds=2),
                )
            self.assertFalse((parent / 'rejected').exists())

            def exploding_verifier(_target, _proof, _policy_binding):
                raise RuntimeError('simulated verifier defect')

            with self.assertRaisesRegex(WitnessVerificationError, 'simulated verifier defect'):
                broker_witness_checkpoint(
                    parent / 'failed',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=provider,
                    verifier=exploding_verifier,
                    verified_at=_T0 + timedelta(seconds=2),
                )
            self.assertFalse((parent / 'failed').exists())

    def test_temporal_sanity_and_pinned_authority_are_enforced_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()

            with self.assertRaisesRegex(WitnessVerificationError, 'predates creation'):
                broker_witness_checkpoint(
                    parent / 'early',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(
                        witnessed_at=_T0 - timedelta(microseconds=1)
                    ),
                    verified_at=_T0 + timedelta(seconds=2),
                )
            with self.assertRaisesRegex(WitnessVerificationError, 'verification time predates'):
                broker_witness_checkpoint(
                    parent / 'future',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(witnessed_at=_T0 + timedelta(seconds=3)),
                    verified_at=_T0 + timedelta(seconds=2),
                )
            with self.assertRaisesRegex(WitnessVerificationError, 'pinned policy'):
                broker_witness_checkpoint(
                    parent / 'wrong-authority',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(authority_id='organizer-local-service'),
                    verified_at=_T0 + timedelta(seconds=2),
                )
            with self.assertRaisesRegex(WitnessVerificationError, 'different checkpoint imprint'):
                broker_witness_checkpoint(
                    parent / 'wrong-imprint',
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(checkpoint_digest='0' * 64),
                    verified_at=_T0 + timedelta(seconds=2),
                )
            self.assertFalse(any(parent.iterdir()))

    def test_publication_never_replaces_an_existing_witness_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / 'witness'
            first = broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            before = {path.name: path.read_bytes() for path in root.iterdir()}

            with self.assertRaisesRegex(ValueError, 'output already exists'):
                broker_witness_checkpoint(
                    root,
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), b'different proof'),
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    verified_at=_T0 + timedelta(seconds=2),
                )

            self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)
            self.assertFalse((parent / '.witness.publish.lock').exists())
            self.assertEqual(
                load_witnessed_checkpoint(
                    root,
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                ),
                first,
            )

    def test_publication_preserves_target_created_at_install_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / 'witness'
            original_rename = witness_module.rename_directory_noreplace

            def race_at_install(source: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / 'owner.txt').write_bytes(b'created by competing publisher')
                original_rename(source, destination)

            with (
                patch.object(
                    witness_module,
                    'rename_directory_noreplace',
                    side_effect=race_at_install,
                ),
                self.assertRaisesRegex(ValueError, 'output already exists'),
            ):
                broker_witness_checkpoint(
                    root,
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    verified_at=_T0 + timedelta(seconds=2),
                )

            self.assertEqual((root / 'owner.txt').read_bytes(), b'created by competing publisher')
            self.assertFalse((parent / '.witness.publish.lock').exists())
            self.assertFalse(any(path.name.startswith('.witness.') for path in parent.iterdir()))

    def test_publication_reload_rejects_root_replaced_by_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / 'witness'
            displaced = parent / 'displaced-witness'
            original_release = witness_module._release_publication_lock

            def replace_after_install(lock_path: Path, descriptor: int) -> None:
                original_release(lock_path, descriptor)
                root.rename(displaced)
                root.symlink_to(displaced, target_is_directory=True)

            with (
                patch.object(
                    witness_module,
                    '_release_publication_lock',
                    side_effect=replace_after_install,
                ),
                self.assertRaisesRegex(WitnessVerificationError, 'cannot open witnessed checkpoint root'),
            ):
                broker_witness_checkpoint(
                    root,
                    checkpoint=_checkpoint(),
                    policy=_policy(),
                    provider=lambda _request: (_claim(), _PROOF),
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    verified_at=_T0 + timedelta(seconds=2),
                )

            self.assertTrue(root.is_symlink())
            self.assertTrue(displaced.is_dir())

    def test_load_rejects_root_replaced_after_artifact_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory).resolve()
            root = parent / 'witness'
            broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            displaced = parent / 'displaced-witness'
            original_read = witness_module._read_regular_file_at
            replaced = False

            def replace_after_proof(root_descriptor: int, name: str, max_bytes: int) -> bytes:
                nonlocal replaced
                payload = original_read(root_descriptor, name, max_bytes)
                if name == 'external-proof.bin' and not replaced:
                    root.rename(displaced)
                    root.mkdir()
                    replaced = True
                return payload

            with (
                patch.object(witness_module, '_read_regular_file_at', side_effect=replace_after_proof),
                self.assertRaisesRegex(WitnessVerificationError, 'root changed while being read'),
            ):
                load_witnessed_checkpoint(
                    root,
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )

            self.assertTrue(replaced)

    def test_load_rejects_symlink_in_witness_root_parent_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory).resolve()
            real_parent = temporary / 'real-parent'
            real_parent.mkdir()
            root = real_parent / 'witness'
            broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            link_parent = temporary / 'link-parent'
            link_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(WitnessVerificationError, 'cannot open witnessed checkpoint root'):
                load_witnessed_checkpoint(
                    link_parent / 'witness',
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )

    def test_load_detects_exact_byte_tampering_before_trusted_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve() / 'witness'

            def provider(_request):
                return _claim(), _PROOF

            built = broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=provider,
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            (root / 'external-proof.bin').write_bytes(b'tampered proof')
            verifier_called = False

            def verifier(_target, _proof, _policy_binding):
                nonlocal verifier_called
                verifier_called = True
                return _facts()

            with self.assertRaisesRegex(WitnessVerificationError, 'hash and size binding'):
                load_witnessed_checkpoint(
                    built.root,
                    verifier=verifier,
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )
            self.assertFalse(verifier_called)

    def test_checkpoint_encoding_inventory_and_local_signature_method_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve() / 'witness'

            def verifier(_target, _proof, _policy_binding):
                return _facts()

            broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=verifier,
                verified_at=_T0 + timedelta(seconds=2),
            )

            checkpoint_path = root / 'checkpoint.json'
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b' ')
            with self.assertRaisesRegex(WitnessVerificationError, 'canonical JSON'):
                load_witnessed_checkpoint(
                    root,
                    verifier=verifier,
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )

    def test_load_rejects_a_receipt_time_not_authenticated_by_the_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve() / 'witness'
            built = broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            changed_receipt = built.manifest.receipt.model_copy(update={'witnessed_at': _T0})
            changed_manifest = built.manifest.model_copy(update={'receipt': changed_receipt})
            (root / 'witness.json').write_bytes(canonical_json_bytes(changed_manifest))

            with self.assertRaisesRegex(WitnessVerificationError, 'authenticated facts parsed from the proof'):
                load_witnessed_checkpoint(
                    root,
                    verifier=lambda _target, _proof, _policy_binding: _facts(),
                    expected_policy=_policy(),
                    verified_at=_T0 + timedelta(seconds=3),
                )

            with self.assertRaises(ValidationError):
                AuthenticatedExternalWitnessFacts.model_validate_json(
                    b'{"authority_id":"local","checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                    b'"method":"local_signature","policy_id":"policy","receipt_id":"local-1",'
                    b'"schema_version":"vaxreplay.operations-authenticated-external-witness-facts.v0.1",'
                    b'"witness_id":"local-key","witnessed_at":"2026-07-13T12:00:01Z"}'
                )

    def test_load_rejects_a_manifest_that_does_not_match_the_out_of_band_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve() / 'witness'
            broker_witness_checkpoint(
                root,
                checkpoint=_checkpoint(),
                policy=_policy(),
                provider=lambda _request: (_claim(), _PROOF),
                verifier=lambda _target, _proof, _policy_binding: _facts(),
                verified_at=_T0 + timedelta(seconds=2),
            )
            wrong_policy = _policy().model_copy(update={'trust_policy_sha256': '0' * 64})
            verifier_called = False

            def verifier(_target, _proof, _policy_binding):
                nonlocal verifier_called
                verifier_called = True
                return _facts()

            with self.assertRaisesRegex(WitnessVerificationError, 'out-of-band pinned policy'):
                load_witnessed_checkpoint(
                    root,
                    verifier=verifier,
                    expected_policy=wrong_policy,
                    verified_at=_T0 + timedelta(seconds=3),
                )
            self.assertFalse(verifier_called)


if __name__ == '__main__':
    unittest.main()
