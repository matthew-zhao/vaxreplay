from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = (ROOT / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8')
VERIFY = (ROOT / 'release' / 'alpha-v0.1.0-alpha.1' / 'VERIFY.md').read_text(encoding='utf-8')
REVIEW = (ROOT / 'release' / 'alpha-v0.1.0-alpha.1' / 'RELEASE-REVIEW.md').read_text(encoding='utf-8')
NOTES = (ROOT / 'release' / 'alpha-v0.1.0-alpha.1' / 'RELEASE-NOTES.md').read_text(encoding='utf-8')


def _job_blocks() -> tuple[str, str]:
    jobs = WORKFLOW.split('\njobs:\n', maxsplit=1)[1]
    build, attest = jobs.split('\n  attest:\n', maxsplit=1)
    return build, attest


def test_build_and_attestation_authority_are_separated() -> None:
    build, attest = _job_blocks()

    assert 'contents: read' in build
    assert 'id-token:' not in build
    assert 'attestations:' not in build
    assert 'artifact-metadata:' not in build

    assert 'contents: read' in attest
    assert 'id-token: write' in attest
    assert 'attestations: write' in attest
    assert 'artifact-metadata: write' in attest
    assert 'actions/checkout@' not in attest
    assert 'scripts/' not in attest
    assert '\n          python ' not in attest
    assert '\n          uv ' not in attest

    assert WORKFLOW.count('id-token: write') == 1
    assert WORKFLOW.count('attestations: write') == 1
    assert WORKFLOW.count('artifact-metadata: write') == 1
    assert 'contents: write' not in WORKFLOW


def test_candidate_and_tag_identity_bindings_are_enforced() -> None:
    assert 'test "$GITHUB_REF" = "refs/heads/main"' in WORKFLOW
    assert 'test "$RELEASE_COMMIT" = "$GITHUB_SHA"' in WORKFLOW
    assert WORKFLOW.count('test "$WORKFLOW_SHA" = "$RELEASE_COMMIT"') == 2
    assert 'test "$GITHUB_REF" = "refs/tags/v0.1.0-alpha.1"' in WORKFLOW
    assert 'if [[ "$GITHUB_EVENT_NAME" == "push" ]]' in WORKFLOW
    assert 'args+=(--require-tag)' in WORKFLOW


def test_release_toolchain_actions_and_runner_are_pinned() -> None:
    pins = (
        'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1',
        'actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97',
        'astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9',
        'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a',
        'actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c',
        'actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6',
    )
    for pin in pins:
        assert pin in WORKFLOW

    assert WORKFLOW.count('runs-on: ubuntu-24.04') == 2
    assert "python-version: '3.12.13'" in WORKFLOW
    assert 'version: 0.11.7' in WORKFLOW


def test_release_runs_validation_and_clean_install_contract() -> None:
    for command in (
        'uv sync --locked --extra dev',
        'uv run ruff format --check .',
        'uv run ruff check .',
        'uv run ty check --exclude src/vaxreplay/integrations src/vaxreplay',
        'uv run pytest -q -p no:cacheprovider',
        'uv run python scripts/smoke_test.py',
        '--require-hashes',
        '--only-binary :all:',
        '--no-deps',
        'uv pip check --python "$clean_env/bin/python"',
        'assert vaxreplay.__version__ == "0.1.0a1"',
    ):
        assert command in WORKFLOW

    for entrypoint in (
        'vaxreplay',
        'vaxreplay-iedb',
        'vaxreplay-feasibility',
        'vaxreplay-prospective',
        'vaxreplay-runner',
        'vaxreplay-ops',
        'vaxreplay-release-readiness',
    ):
        assert re.search(rf'^\s+{re.escape(entrypoint)}$', WORKFLOW, flags=re.MULTILINE)
    assert '"$clean_env/bin/$command" --help >/dev/null' in WORKFLOW


def test_checksum_payload_and_sbom_attestations_are_retained() -> None:
    assert WORKFLOW.count('uses: actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6') == 3
    assert 'subject-checksums:' in WORKFLOW
    assert re.search(r'subject-path: .*?/SHA256SUMS$', WORKFLOW, flags=re.MULTILINE)
    assert 'sbom-path:' in WORKFLOW
    for filename in (
        'vaxreplay-v0.1.0-alpha.1-payloads-provenance.sigstore.json',
        'vaxreplay-v0.1.0-alpha.1-checksums-provenance.sigstore.json',
        'vaxreplay-v0.1.0-alpha.1-sbom.sigstore.json',
    ):
        assert filename in WORKFLOW


def test_verification_covers_online_offline_and_candidate_modes() -> None:
    assert 'A `workflow_dispatch` run is a pre-tag candidate only' in VERIFY
    assert 'A manual run on a tag is rejected.' in VERIFY
    assert re.search(r'gh attestation verify \\\n  SHA256SUMS', VERIFY)
    assert 'gh attestation trusted-root > trusted_root.jsonl' in VERIFY
    assert '--custom-trusted-root trusted_root.jsonl' in VERIFY
    assert '--source-ref refs/tags/v0.1.0-alpha.1' in VERIFY
    assert '--source-digest %VAXREPLAY_PUBLIC_COMMIT%' in VERIFY
    assert '--signer-digest %VAXREPLAY_PUBLIC_COMMIT%' in VERIFY
    assert '--deny-self-hosted-runners' in VERIFY
    for filename in (
        'vaxreplay-v0.1.0-alpha.1-payloads-provenance.sigstore.json',
        'vaxreplay-v0.1.0-alpha.1-checksums-provenance.sigstore.json',
        'vaxreplay-v0.1.0-alpha.1-sbom.sigstore.json',
    ):
        assert filename in VERIFY


def test_release_language_is_mode_neutral_and_does_not_mutate_releases() -> None:
    assert 'maintainer approval is external' in REVIEW
    assert 'short-lived GitHub OIDC identity and Sigstore signing certificate' in REVIEW
    assert 'durable, identity-bound artifact attestations' in REVIEW
    assert 'A manually dispatched build is a review candidate only.' in NOTES
    assert 'The release workflow does not create a Git tag, GitHub Release, or PyPI publication.' in NOTES
    assert 'gh release' not in WORKFLOW
    assert 'git tag' not in WORKFLOW
