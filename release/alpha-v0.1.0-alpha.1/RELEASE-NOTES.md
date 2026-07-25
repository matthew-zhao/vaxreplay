# VaxReplay v0.1.0-alpha.1

This is the first VaxReplay technical preview. It publishes infrastructure for temporally
controlled biomedical evaluation, deterministic scoring, contamination controls, and sealed
execution experiments.

The release contains wholly fictional conformance fixtures. It does **not** contain an admitted
real-data benchmark, the private development cohort, private gold, or a scientifically
valid leaderboard. Its outputs must not be used for clinical or vaccine-development decisions.

## Included

- Typed task, evidence, forecast, citation, and score contracts.
- Historical-source adapter interfaces and temporal-admission protocols.
- Deterministic evaluators, aggregation, and research-process scoring components.
- Development-stage sealed-runner, provider-gateway, and harness adapters.
- Public tests, a deterministic smoke test, and fictional compatibility fixtures.
- Reproducible Python package artifacts, a deterministic public-source archive, an SPDX 2.3
  runtime SBOM, dependency/license inventories, checksums, and GitHub artifact attestations.
- Locked formatting, lint, type, test, smoke, and clean-wheel-install release gates.
- Separate attestations for the checksummed payloads, the checksum manifest, and the wheel's SPDX
  SBOM, with all three Sigstore bundles retained for offline verification.

## Important limitations

- APIs and artifact formats may change before beta.
- The public fixtures test software compatibility, not biological or clinical capability.
- A sealed execution environment reduces runtime leakage; it cannot make a contaminated model
  forget training data.
- Previously public development descriptions contaminate the existing AACT pilot and
  development cohort; those cases are permanently ineligible for held-out or commercial scoring.
- Checked-in AACT selection, taxonomy, and clinical-rubric logic is public reference semantics,
  not a secret evaluation policy.
- No production-qualified microVM service or official public leaderboard is included.
- A manually dispatched build is a review candidate only. The final bundle must be rebuilt by the
  exact `v0.1.0-alpha.1` tag push.

The release workflow does not create a Git tag, GitHub Release, or PyPI publication.

See `README.md`, `docs/alpha_scope.md`, and `DATA_LICENSES.md` in the source archive for the full
scope and licensing boundary.
