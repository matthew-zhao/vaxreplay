from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.hermetic_execution import (
    IMPLEMENTATION_LABEL,
    Ed25519ReceiptSigner,
    HermeticCallbackMaterials,
    HermeticExecutionError,
    HermeticExecutionResponse,
    HermeticMaterialBinding,
    HermeticOciEnvironment,
    HermeticSandboxPolicy,
    OciHermeticCallbackExecutor,
    SignedHermeticExecutionReceipt,
    build_hermetic_docker_argv,
    build_hermetic_worker_response,
    parse_hermetic_worker_request,
    verify_hermetic_execution_bundle,
)
from vaxreplay.runner._process import BoundedProcessResult

_IMPLEMENTATION = b'fixture verifier implementation bytes'
_CALLBACK_POLICY = b'{"allow":"fixture"}'
_INPUT = b'{"capture":"opaque"}'
_SECCOMP = b'{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}'
_PRIVATE_KEY = bytes(range(32))


def _signer() -> Ed25519ReceiptSigner:
    return Ed25519ReceiptSigner(key_id='tier-a-runner-key-1', private_key_bytes=_PRIVATE_KEY)


def _environment() -> HermeticOciEnvironment:
    return HermeticOciEnvironment(
        environment_id='tier-a-callback-python-1',
        image_ref='registry.example/vaxreplay/callback@sha256:' + 'a' * 64,
        expected_image_id='sha256:' + 'b' * 64,
        platform='linux/amd64',
        entrypoint=('/opt/vaxreplay/worker', '--stdio'),
    )


def _policy() -> HermeticSandboxPolicy:
    signer = _signer()
    return HermeticSandboxPolicy(
        policy_id='tier-a-hermetic-1',
        authority_id='independent-runner-1',
        signing_key_id=signer.key_id,
        signing_public_key_sha256=hashlib.sha256(signer.public_key_bytes).hexdigest(),
        seccomp_profile_sha256=hashlib.sha256(_SECCOMP).hexdigest(),
        wall_seconds=30,
        memory_mib=256,
        milli_cpus=500,
        pids=32,
        scratch_mib=32,
        open_files=64,
        max_input_bytes=1024 * 1024,
        max_callback_policy_bytes=1024 * 1024,
        max_output_bytes=1024 * 1024,
        max_worker_response_bytes=2 * 1024 * 1024,
        max_log_bytes=64 * 1024,
    )


def _materials() -> HermeticCallbackMaterials:
    return HermeticCallbackMaterials(
        implementation_bytes=_IMPLEMENTATION,
        execution_environment_bytes=canonical_json_bytes(_environment()),
        callback_policy_bytes=_CALLBACK_POLICY,
    )


def _inspection(*, image_env: object = None, implementation_sha256: str | None = None) -> dict[str, object]:
    return {
        'Id': _environment().expected_image_id,
        'Os': 'linux',
        'Architecture': 'amd64',
        'RepoDigests': [_environment().image_ref],
        'Config': {
            'Volumes': None,
            'Cmd': None,
            'Env': image_env,
            'Labels': {
                IMPLEMENTATION_LABEL: implementation_sha256 or hashlib.sha256(_IMPLEMENTATION).hexdigest(),
            },
        },
    }


def _process_result_for_request(request_bytes: bytes) -> BoundedProcessResult:
    request = json.loads(request_bytes)
    output = b'{"verified":true}'
    response = HermeticExecutionResponse(
        invocation_id=request['invocation_id'],
        invocation_index=request['invocation_index'],
        purpose=request['purpose'],
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        output=HermeticMaterialBinding(
            sha256=hashlib.sha256(output).hexdigest(),
            byte_count=len(output),
        ),
        output_base64=base64.b64encode(output).decode('ascii'),
    )
    return BoundedProcessResult(
        exit_code=0,
        duration_ms=17,
        stdout=canonical_json_bytes(response),
        stderr=b'',
        termination='exited',
        stdout_truncated=False,
        stderr_truncated=False,
    )


class HermeticExecutionTest(unittest.TestCase):
    def _executor(self) -> OciHermeticCallbackExecutor:
        with patch(
            'vaxreplay.operations.hermetic_execution._resolve_docker',
            return_value='/usr/local/bin/docker',
        ):
            return OciHermeticCallbackExecutor(
                sandbox_policy=_policy(),
                seccomp_profile_bytes=_SECCOMP,
                signer=_signer(),
                clock=lambda: datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
            )

    def _execute(self):
        executor = self._executor()
        inspection_bytes = json.dumps(_inspection()).encode()
        query_results = (
            b'linux|29.2.0\n',
            inspection_bytes,
            (b'c' * 64) + b'\n',
            b'',
        )

        def run_worker(_argv, *, input_bytes, **_kwargs):
            return _process_result_for_request(input_bytes)

        with (
            patch.object(executor, '_query', side_effect=query_results),
            patch.object(executor, '_request_remove'),
            patch('vaxreplay.operations.hermetic_execution.run_bounded_process', side_effect=run_worker),
        ):
            bundle = executor.execute(
                purpose='source_verifier',
                invocation_id='iedb-source-verification-1',
                invocation_index=3,
                input_bytes=_INPUT,
                materials=_materials(),
            )
        return executor, bundle, canonical_json_bytes(_inspection())

    def test_docker_argv_has_no_network_or_mounts_and_enforces_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            seccomp = Path(temp_dir) / 'seccomp.json'
            seccomp.write_bytes(_SECCOMP)
            argv = build_hermetic_docker_argv(
                runtime='/usr/local/bin/docker',
                container_name='vaxreplay-hermetic-test',
                environment=_environment(),
                policy=_policy(),
                seccomp_profile_path=seccomp,
            )
        for option, expected in (
            ('--network', 'none'),
            ('--cap-drop', 'ALL'),
            ('--user', '65532:65532'),
            ('--pids-limit', '32'),
            ('--memory', '256m'),
            ('--memory-swap', '256m'),
            ('--cpus', '0.5'),
            ('--platform', 'linux/amd64'),
        ):
            index = argv.index(option)
            self.assertEqual(argv[index + 1], expected)
        self.assertIn('--read-only', argv)
        self.assertIn('no-new-privileges:true', argv)
        self.assertTrue(any(item.startswith('seccomp=') for item in argv))
        self.assertNotIn('--mount', argv)
        self.assertNotIn('--volume', argv)
        self.assertNotIn('-v', argv)
        self.assertNotIn('--privileged', argv)
        self.assertNotIn('/bin/sh', argv)
        self.assertEqual(
            argv[-3:],
            ('/opt/vaxreplay/worker', _environment().expected_image_id, '--stdio'),
        )

    def test_execution_returns_exact_signed_offline_verifiable_bytes(self) -> None:
        executor, bundle, inspection_bytes = self._execute()

        self.assertEqual(bundle.output_bytes, b'{"verified":true}')
        self.assertEqual(bundle.request_bytes, canonical_json_bytes(bundle.request))
        self.assertEqual(bundle.response_bytes, canonical_json_bytes(bundle.response))
        self.assertEqual(bundle.receipt_bytes, canonical_json_bytes(bundle.receipt))
        self.assertEqual(bundle.receipt.attestation.purpose, 'source_verifier')
        self.assertEqual(bundle.receipt.attestation.invocation_index, 3)
        self.assertTrue(bundle.receipt.attestation.network_disabled)
        self.assertTrue(bundle.receipt.attestation.cleanup_verified)

        loaded = verify_hermetic_execution_bundle(
            request_bytes=bundle.request_bytes,
            response_bytes=bundle.response_bytes,
            receipt_bytes=bundle.receipt_bytes,
            expected_materials=_materials(),
            expected_sandbox_policy_bytes=executor.sandbox_policy_bytes,
            expected_seccomp_profile_bytes=_SECCOMP,
            trusted_public_key_bytes=_signer().public_key_bytes,
            image_inspection_bytes=inspection_bytes,
        )
        self.assertEqual(loaded, bundle)

    def test_worker_helpers_preserve_canonical_request_and_bind_output(self) -> None:
        _executor, bundle, _inspection_bytes = self._execute()
        request = parse_hermetic_worker_request(bundle.request_bytes)
        response_bytes = build_hermetic_worker_response(bundle.request_bytes, b'new output')
        response = HermeticExecutionResponse.model_validate_json(response_bytes)

        self.assertEqual(request.invocation_index, 3)
        self.assertEqual(response_bytes, canonical_json_bytes(response))
        self.assertEqual(response.request_sha256, hashlib.sha256(bundle.request_bytes).hexdigest())
        self.assertEqual(base64.b64decode(response.output_base64), b'new output')

    def test_receipt_signature_tamper_fails_closed(self) -> None:
        executor, bundle, inspection_bytes = self._execute()
        tampered = SignedHermeticExecutionReceipt(
            attestation=bundle.receipt.attestation,
            signature_base64=base64.b64encode(b'\x00' * 64).decode(),
        )
        with self.assertRaisesRegex(HermeticExecutionError, 'signature'):
            verify_hermetic_execution_bundle(
                request_bytes=bundle.request_bytes,
                response_bytes=bundle.response_bytes,
                receipt_bytes=canonical_json_bytes(tampered),
                expected_materials=_materials(),
                expected_sandbox_policy_bytes=executor.sandbox_policy_bytes,
                expected_seccomp_profile_bytes=_SECCOMP,
                trusted_public_key_bytes=_signer().public_key_bytes,
                image_inspection_bytes=inspection_bytes,
            )

    def test_wrong_implementation_or_inspection_fails_offline(self) -> None:
        executor, bundle, inspection_bytes = self._execute()
        wrong_materials = HermeticCallbackMaterials(
            implementation_bytes=b'other implementation',
            execution_environment_bytes=_materials().execution_environment_bytes,
            callback_policy_bytes=_CALLBACK_POLICY,
        )
        with self.assertRaisesRegex(ValueError, 'implementation'):
            verify_hermetic_execution_bundle(
                request_bytes=bundle.request_bytes,
                response_bytes=bundle.response_bytes,
                receipt_bytes=bundle.receipt_bytes,
                expected_materials=wrong_materials,
                expected_sandbox_policy_bytes=executor.sandbox_policy_bytes,
                expected_seccomp_profile_bytes=_SECCOMP,
                trusted_public_key_bytes=_signer().public_key_bytes,
                image_inspection_bytes=inspection_bytes,
            )
        changed_inspection = dict(_inspection())
        changed_inspection['unexpected'] = True
        with self.assertRaisesRegex(HermeticExecutionError, 'inspection'):
            verify_hermetic_execution_bundle(
                request_bytes=bundle.request_bytes,
                response_bytes=bundle.response_bytes,
                receipt_bytes=bundle.receipt_bytes,
                expected_materials=_materials(),
                expected_sandbox_policy_bytes=executor.sandbox_policy_bytes,
                expected_seccomp_profile_bytes=_SECCOMP,
                trusted_public_key_bytes=_signer().public_key_bytes,
                image_inspection_bytes=canonical_json_bytes(changed_inspection),
            )

    def test_noncanonical_worker_response_fails_closed(self) -> None:
        executor = self._executor()
        query_results = (
            b'linux|29.2.0\n',
            json.dumps(_inspection()).encode(),
            (b'c' * 64) + b'\n',
            b'',
        )

        def run_worker(_argv, *, input_bytes, **_kwargs):
            result = _process_result_for_request(input_bytes)
            return BoundedProcessResult(
                exit_code=result.exit_code,
                duration_ms=result.duration_ms,
                stdout=result.stdout + b'\n',
                stderr=result.stderr,
                termination=result.termination,
                stdout_truncated=False,
                stderr_truncated=False,
            )

        with (
            patch.object(executor, '_query', side_effect=query_results),
            patch.object(executor, '_request_remove'),
            patch('vaxreplay.operations.hermetic_execution.run_bounded_process', side_effect=run_worker),
            self.assertRaisesRegex(HermeticExecutionError, 'canonical'),
        ):
            executor.execute(
                purpose='adapter',
                invocation_id='adapter-1',
                invocation_index=0,
                input_bytes=_INPUT,
                materials=_materials(),
            )

    def test_timeout_fails_closed_after_verified_cleanup(self) -> None:
        executor = self._executor()
        timed_out = BoundedProcessResult(
            exit_code=None,
            duration_ms=30_000,
            stdout=b'',
            stderr=b'',
            termination='timed_out',
            stdout_truncated=False,
            stderr_truncated=False,
        )
        query_results = (
            b'linux|29.2.0\n',
            json.dumps(_inspection()).encode(),
            (b'c' * 64) + b'\n',
            b'',
        )
        with (
            patch.object(executor, '_query', side_effect=query_results),
            patch.object(executor, '_request_remove'),
            patch('vaxreplay.operations.hermetic_execution.run_bounded_process', return_value=timed_out),
            self.assertRaisesRegex(HermeticExecutionError, 'did not exit normally'),
        ):
            executor.execute(
                purpose='source_verifier',
                invocation_id='verifier-timeout',
                invocation_index=0,
                input_bytes=_INPUT,
                materials=_materials(),
            )

    def test_preflight_rejects_ambient_image_environment(self) -> None:
        executor = self._executor()
        with (
            patch.object(
                executor,
                '_query',
                side_effect=(b'linux|29.2.0\n', json.dumps(_inspection(image_env=['SECRET=value'])).encode()),
            ),
            self.assertRaisesRegex(HermeticExecutionError, 'ambient environment'),
        ):
            executor.execute(
                purpose='adapter',
                invocation_id='adapter-ambient-env',
                invocation_index=0,
                input_bytes=_INPUT,
                materials=_materials(),
            )

    def test_constructor_rejects_wrong_seccomp_or_signing_key(self) -> None:
        with (
            patch('vaxreplay.operations.hermetic_execution._resolve_docker', return_value='/usr/local/bin/docker'),
            self.assertRaisesRegex(ValueError, 'seccomp'),
        ):
            OciHermeticCallbackExecutor(
                sandbox_policy=_policy(),
                seccomp_profile_bytes=b'wrong',
                signer=_signer(),
            )

    def test_offline_verifier_rejects_different_seccomp_bytes(self) -> None:
        executor, bundle, inspection_bytes = self._execute()
        with self.assertRaisesRegex(HermeticExecutionError, 'seccomp'):
            verify_hermetic_execution_bundle(
                request_bytes=bundle.request_bytes,
                response_bytes=bundle.response_bytes,
                receipt_bytes=bundle.receipt_bytes,
                expected_materials=_materials(),
                expected_sandbox_policy_bytes=executor.sandbox_policy_bytes,
                expected_seccomp_profile_bytes=b'different seccomp bytes',
                trusted_public_key_bytes=_signer().public_key_bytes,
                image_inspection_bytes=inspection_bytes,
            )

    def test_execution_receipt_uses_authority_clock(self) -> None:
        _executor, bundle, _inspection_bytes = self._execute()
        self.assertEqual(
            bundle.receipt.attestation.issued_at,
            datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
        )


if __name__ == '__main__':
    unittest.main()
