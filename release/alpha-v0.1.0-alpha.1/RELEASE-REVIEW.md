# Alpha release review

Release: `v0.1.0-alpha.1`

Status: **release gates recorded; maintainer approval is external**

This document records the review boundary and automated gates shared by manual candidates and the
tag-built bundle. It is not a signature, approval record, or claim that a tag or GitHub Release
exists.

## Automated release gates

The release workflow:

- accepts manual candidates only from the exact public `main` commit and accepts a final build only
  from the exact `v0.1.0-alpha.1` tag push;
- requires the source, event, checkout, and workflow commits to agree;
- runs the locked formatter, lint, type, test, and deterministic smoke-test contract;
- requires an exact full public commit and a clean checkout;
- verifies the checked-in public-tree manifest and export metadata;
- creates a deterministic `git archive` source archive using `gzip -n`;
- performs two fresh package builds with pinned `uv` and `SOURCE_DATE_EPOCH`, then byte-compares
  both wheels and both sdists;
- rejects unsafe or duplicate archive members and verifies every wheel `RECORD` entry;
- clean-installs the exact locked runtime plus the wheel into a fresh environment, checks installed
  dependencies and package version, and runs all seven public commands with `--help`;
- checks the static dependency/license policy against the exact runtime closure in `uv.lock`;
- emits an SPDX 2.3 runtime SBOM, inventories, a build receipt, release binding, and sorted SHA-256
  checksums;
- transfers the validated unsigned candidate to a separate no-checkout attestation job, so
  repository code never runs with OIDC or attestation-write authority;
- separately attests the checksummed payloads, `SHA256SUMS`, and the wheel's SPDX SBOM; and
- does not create a release tag, GitHub Release, local signature, or signature placeholder.

The package builds remain network-enabled and are not claimed to be hermetic. The receipt records
the actual tool versions and explicitly records that boundary.

## Candidate and tag-built modes

A manual `workflow_dispatch` is always a review candidate. The workflow permits it only on
`refs/heads/main` and rejects manual dispatches on tag refs. Only the exact tag-push rebuild is
eligible to become the final release bundle.

The tag workflow uses a short-lived GitHub OIDC identity and Sigstore signing certificate to create
durable, identity-bound artifact attestations. Those attestations are machine provenance, not a
substitute for maintainer review.

## Scope and rights boundary

- Original VaxReplay software is Apache-2.0.
- Explicitly identified project-authored fictional fixtures are CC BY 4.0.
- No real cohort, private gold, organizer mapping, provider response, or third-party source dataset
  is distributed in this alpha.
- Previously exposed AACT development cases are permanently reference-only; a held-out or
  commercial evaluation requires a newly selected private cohort.
- The dependency license inventory is informational, version-specific, and not legal advice.

## Third-party names and trademarks

The source uses third-party names—including OpenAI/Codex, Anthropic/Claude Code, Cursor,
Firecracker, IEDB, ImmPort, ClinicalTrials.gov/AACT, VaxSeer, and FluSelect—only to describe
interfaces, data-source concepts, or compatibility work. No third-party logos are distributed and
no affiliation, sponsorship, certification, or endorsement is claimed. See `TRADEMARKS.md`.

## Maintainer approval boundary

Before creating the tag or release, the maintainer should:

- [ ] inspect the manual candidate bundle and authenticated `SHA256SUMS`;
- [ ] confirm the release date and exact public commit;
- [ ] confirm the scope, license, attribution, and trademark statements above;
- [ ] confirm the dependency policy remains accurate for the checked-in `uv.lock`; and
- [ ] explicitly authorize creation of the tag and prerelease.

These checkboxes describe the external pre-tag decision and are intentionally not filled in by the
builder. Their unchecked rendering inside a tag-built bundle does not revoke or replace the
maintainer's separately recorded authorization.
