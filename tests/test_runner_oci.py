from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from vaxreplay.runner._process import BoundedProcessResult
from vaxreplay.runner.backend import BackendPolicyError, IsolationCleanupError
from vaxreplay.runner.oci import OciDevelopmentBackend, build_docker_argv
from vaxreplay.runner.schema import (
    IsolationTier,
    RunnerPolicy,
    SystemSubmissionManifest,
)


def _system() -> SystemSubmissionManifest:
    return SystemSubmissionManifest(
        submission_id='system-1',
        image_ref='registry.example/vax/system@sha256:' + 'a' * 64,
        entrypoint=('/opt/vaxreplay/run', '--literal;touch', '$(not-a-shell)'),
        model_id='model-1',
        harness_id='harness-1',
    )


class RunnerOciTest(unittest.TestCase):
    def test_system_image_must_be_digest_pinned(self) -> None:
        with self.assertRaisesRegex(ValueError, 'pinned'):
            SystemSubmissionManifest(
                submission_id='system-1',
                image_ref='registry.example/vax/system:latest',
                entrypoint=('/opt/vaxreplay/run',),
                model_id='model-1',
                harness_id='harness-1',
            )

    def test_docker_argv_enforces_controls_without_host_mounts_or_shell(self) -> None:
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        argv = build_docker_argv(
            runtime='/usr/local/bin/docker',
            container_name='vaxreplay-test',
            system=_system(),
            policy=policy,
            resolved_image_id='sha256:' + 'b' * 64,
        )

        for pair in (
            ('--pull', 'never'),
            ('--network', 'none'),
            ('--cap-drop', 'ALL'),
            ('--security-opt', 'no-new-privileges:true'),
            ('--user', '65532:65532'),
            ('--ipc', 'private'),
            ('--cgroupns', 'private'),
            ('--log-driver', 'none'),
        ):
            index = argv.index(pair[0])
            self.assertEqual(argv[index + 1], pair[1])
        self.assertIn('--read-only', argv)
        self.assertEqual(argv[1], 'create')
        self.assertNotIn('--rm', argv)
        self.assertIn('--no-healthcheck', argv)
        self.assertIn('--init', argv)
        self.assertNotIn('--privileged', argv)
        self.assertNotIn('--mount', argv)
        self.assertNotIn('--volume', argv)
        self.assertNotIn('-v', argv)
        self.assertIn('--literal;touch', argv)
        self.assertIn('$(not-a-shell)', argv)
        self.assertNotIn('/bin/sh', argv)
        self.assertIn('sha256:' + 'b' * 64, argv)
        self.assertNotIn(_system().image_ref, argv)
        image_index = argv.index('sha256:' + 'b' * 64)
        self.assertEqual(argv[image_index + 1 :], _system().entrypoint[1:])

    def test_official_policy_fails_before_any_runtime_query(self) -> None:
        with patch('vaxreplay.runner.oci._resolve_runtime', return_value='/usr/local/bin/docker'):
            backend = OciDevelopmentBackend()
        with patch.object(backend, '_query_runtime') as query:
            with self.assertRaisesRegex(BackendPolicyError, 'development-only'):
                backend.prepare(_system(), RunnerPolicy())
            query.assert_not_called()

    def test_rejects_image_declared_volumes(self) -> None:
        with patch('vaxreplay.runner.oci._resolve_runtime', return_value='/usr/local/bin/docker'):
            backend = OciDevelopmentBackend()
        query_results = (
            b'linux|29.2.0\n',
            json.dumps(
                {
                    'Id': 'sha256:' + 'b' * 64,
                    'Config': {'Volumes': {'/data': {}}, 'Cmd': None},
                }
            ).encode(),
        )
        with patch.object(backend, '_query_runtime', side_effect=query_results):
            with self.assertRaisesRegex(BackendPolicyError, 'VOLUME'):
                backend.prepare(_system(), RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT))

    def test_rejects_image_declared_command(self) -> None:
        with patch('vaxreplay.runner.oci._resolve_runtime', return_value='/usr/local/bin/docker'):
            backend = OciDevelopmentBackend()
        query_results = (
            b'linux|29.2.0\n',
            json.dumps(
                {
                    'Id': 'sha256:' + 'b' * 64,
                    'Config': {'Volumes': None, 'Cmd': ['implicit-argument']},
                }
            ).encode(),
        )
        with patch.object(backend, '_query_runtime', side_effect=query_results):
            with self.assertRaisesRegex(BackendPolicyError, 'CMD'):
                backend.prepare(_system(), RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT))

    def test_successful_preflight_records_resolved_image(self) -> None:
        with patch('vaxreplay.runner.oci._resolve_runtime', return_value='/usr/local/bin/docker'):
            backend = OciDevelopmentBackend()
        query_results = (
            b'linux|29.2.0\n',
            json.dumps(
                {
                    'Id': 'sha256:' + 'b' * 64,
                    'Config': {'Volumes': None, 'Cmd': None},
                }
            ).encode(),
        )
        with patch.object(backend, '_query_runtime', side_effect=query_results):
            prepared = backend.prepare(
                _system(),
                RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT),
            )

        self.assertEqual(prepared.resolved_image_id, 'sha256:' + 'b' * 64)
        self.assertEqual(prepared.capabilities.isolation_tier, IsolationTier.DEVELOPMENT)

    def test_cleanup_failure_is_a_fatal_infrastructure_error(self) -> None:
        with patch('vaxreplay.runner.oci._resolve_runtime', return_value='/usr/local/bin/docker'):
            backend = OciDevelopmentBackend()
        policy = RunnerPolicy(required_isolation=IsolationTier.DEVELOPMENT)
        metadata = json.dumps(
            {
                'Id': 'sha256:' + 'b' * 64,
                'Config': {'Volumes': None, 'Cmd': None},
            }
        ).encode()
        with patch.object(backend, '_query_runtime', side_effect=(b'linux|29.2.0\n', metadata)):
            prepared = backend.prepare(_system(), policy)
        process_result = BoundedProcessResult(
            exit_code=0,
            duration_ms=1,
            stdout=b'{}',
            stderr=b'',
            termination='exited',
            stdout_truncated=False,
            stderr_truncated=False,
        )
        with (
            patch.object(backend, '_query_runtime', return_value=b'c' * 64),
            patch('vaxreplay.runner.oci.run_bounded_process', return_value=process_result),
            patch.object(backend, '_cleanup_container', return_value=False),
        ):
            with self.assertRaisesRegex(IsolationCleanupError, 'cleanup'):
                backend.run(
                    input_bytes=b'{}',
                    system=_system(),
                    policy=policy,
                    prepared=prepared,
                )


if __name__ == '__main__':
    unittest.main()
