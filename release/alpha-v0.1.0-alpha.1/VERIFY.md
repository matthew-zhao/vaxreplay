# Verify the VaxReplay alpha bundle

Release: `v0.1.0-alpha.1`

The bundle is built from public commit `%VAXREPLAY_PUBLIC_COMMIT%`. The canonical repository is
`https://github.com/matthew-zhao/vaxreplay`.

The workflow enforces two distinct modes:

- A `workflow_dispatch` run is a pre-tag candidate only. It must run from `refs/heads/main`, and its
  commit input, source commit, and workflow commit must be identical.
- A final release bundle must be rebuilt by the exact `refs/tags/v0.1.0-alpha.1` push. The workflow
  requires that tag to resolve to the source commit. A manual run on a tag is rejected.

The commands below verify the tag-built bundle. For a manual candidate, replace
`refs/tags/v0.1.0-alpha.1` with `refs/heads/main`; that does not make the candidate a final release.

## 1. Verify the checksum manifest online

Use a current GitHub CLI to authenticate the checksum manifest before trusting the list of
payloads it names:

```bash
gh attestation verify \
  SHA256SUMS \
  --repo matthew-zhao/vaxreplay \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
```

Then verify every checksummed payload:

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS` is sorted by artifact name. It covers every payload created before attestation. It
cannot cover itself, so the workflow gives it a separate provenance attestation. The three
`*.sigstore.json` files are added afterward and are verified cryptographically rather than
recursively checksummed.

## 2. Verify payload provenance and the wheel SBOM online

The payload provenance attestation directly binds every entry in `SHA256SUMS` to the tag workflow.
This example verifies the source archive; repeat it for another payload when direct per-file
provenance is required:

```bash
gh attestation verify vaxreplay-v0.1.0-alpha.1-source.tar.gz \
  --repo matthew-zhao/vaxreplay \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
```

Verify the separate SPDX 2.3 SBOM attestation for the wheel:

```bash
gh attestation verify vaxreplay-0.1.0a1-py3-none-any.whl \
  --repo matthew-zhao/vaxreplay \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --predicate-type https://spdx.dev/Document/v2.3 \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
```

The source digest binds the produced artifacts to the exact public source commit. The signer digest
separately binds the workflow definition to that commit. Denying self-hosted runners enforces the
GitHub-hosted runner boundary declared by the release workflow.

## 3. Verify with the retained bundles while offline

On a trusted online machine, obtain current Sigstore trusted roots independently of the release
bundle:

```bash
gh attestation trusted-root > trusted_root.jsonl
```

Transfer the trusted root, a current GitHub CLI, and the complete release bundle into the offline
environment. GitHub recommends obtaining a fresh trusted root whenever new signed material is
imported; do not treat a root supplied by the release archive itself as an independent trust
anchor.

First verify the checksum manifest with its retained bundle, then check all payload bytes:

```bash
gh attestation verify \
  SHA256SUMS \
  --repo matthew-zhao/vaxreplay \
  --bundle vaxreplay-v0.1.0-alpha.1-checksums-provenance.sigstore.json \
  --custom-trusted-root trusted_root.jsonl \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
sha256sum -c SHA256SUMS
```

Verify direct payload provenance with the multi-subject payload bundle:

```bash
gh attestation verify vaxreplay-v0.1.0-alpha.1-source.tar.gz \
  --repo matthew-zhao/vaxreplay \
  --bundle vaxreplay-v0.1.0-alpha.1-payloads-provenance.sigstore.json \
  --custom-trusted-root trusted_root.jsonl \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
```

Verify the wheel's SBOM attestation with the SBOM bundle:

```bash
gh attestation verify vaxreplay-0.1.0a1-py3-none-any.whl \
  --repo matthew-zhao/vaxreplay \
  --bundle vaxreplay-v0.1.0-alpha.1-sbom.sigstore.json \
  --custom-trusted-root trusted_root.jsonl \
  --source-ref refs/tags/v0.1.0-alpha.1 \
  --source-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --predicate-type https://spdx.dev/Document/v2.3 \
  --signer-workflow matthew-zhao/vaxreplay/.github/workflows/release.yml \
  --signer-digest %VAXREPLAY_PUBLIC_COMMIT% \
  --deny-self-hosted-runners
```

## 4. Inspect contents before installation

Review:

- `vaxreplay-v0.1.0-alpha.1-release-binding.json`;
- `vaxreplay-v0.1.0-alpha.1-build-receipt.json`;
- `vaxreplay-v0.1.0-alpha.1-archive-inventory.json`;
- `vaxreplay-v0.1.0-alpha.1.spdx.json`; and
- `vaxreplay-v0.1.0-alpha.1-dependency-licenses.md`.

The archive inventory records every member and the builder rejects absolute paths, traversal,
links, devices, duplicate members, and invalid wheel `RECORD` data.

No local key or unsigned signature placeholder is part of this release process. The workflow
creates attestations and workflow artifacts only; it does not create a Git tag or GitHub Release.
