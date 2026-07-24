# Temporal admission and the Tier A handoff

VaxReplay separates scientific data provenance from execution-time isolation. A microVM can keep
labels secret while a model runs, but it cannot prove that the panel or evidence was chosen before
those labels existed. `src/vaxreplay/temporal_schema.py` defines the sidecar contract for that
earlier proof.

## Why the sidecar is separate

The final `EpisodeManifest` contains `labels_sha256`, which can exist only after outcomes mature.
Its full `evidence_sha256` may also commit post-cutoff canary rows that are never model-visible.
Neither value can be the prospective data seal.

`DecisionSnapshotCommitment` instead binds only:

- split, lineage, task, cutoff, portfolio, forecast targets, rubric version, and reward version;
- the complete candidate universe or closed panel in its fixed order;
- evidence visible at the cutoff, excluding every later row; and
- hashes of the full candidate-enumeration, evidence-acquisition, and outcome-adjudication specs.

Changing the split, lineage, panel, rubric, outcome definition, or visible evidence therefore
changes the decision snapshot without changing legacy episode-manifest hashing.

## Prospective workflow

Before outcomes exist:

1. Freeze the complete panel/universe and its source version, query, inclusion/exclusion rules,
   ordering rule, and completeness audit in the candidate-set definition artifact.
2. Freeze source builds, queries/pages, availability rules, normalization, and candidate mappings
   in the evidence-acquisition artifact.
3. Freeze endpoints, horizons, censoring, relevance-grade mapping, and label derivation in the
   outcome-adjudication artifact.
4. Build the decision snapshot from those hashes plus the candidate and visible-evidence records.
5. Obtain independent timestamp or public transparency-log receipts for the candidate, evidence,
   and combined decision artifacts at or before `decision_at`.

After each prespecified horizon matures:

6. Preserve the raw outcome source and a private label-derivation audit.
7. Create HMAC-bound labels, the per-target outcome-availability record, and an independently
   witnessed outcome receipt.
8. Assemble `TemporalAdmissionEnvelope`, binding the four receipts, final episode manifest, raw
   outcome hash, derivation audit, and the original outcome-adjudication specification.

The timestamps must satisfy:

```text
candidate/evidence availability <= corresponding receipt <= decision receipt <= decision_at
decision_at + target horizon <= target label availability <= outcome receipt <= admission
```

## Private verification boundary

`require_official_temporal_admission` fails closed unless the bundle is a nonsynthetic test episode
with private labels and an HMAC-SHA256 commitment. It reparses the envelope from canonical JSON,
recomputes the decision snapshot, validates bundle and private-label integrity, checks the raw
outcome and protocol artifacts, derives availability from private outcomes, and requires exactly
one proof artifact for every receipt.

The caller must supply a trusted `receipt_verifier`. That verifier is responsible for validating
the actual RFC 3161 token, transparency-log inclusion proof, source signature, or archive proof
against an organizer-approved trust policy. A digest plus a declared `witnessed_at` is not a
timestamp proof, and the fixture verifier in unit tests is not suitable for a release.

For outcome timing, `first_label_available_at` means the earliest time the scored value was
available to any source owner, curator, investigator, or benchmark organizer—not its later public
publication or benchmark-release date. The real receipt verifier and private source audit must
authenticate that interpretation. If they cannot rule out pre-decision private availability, the
episode cannot be Tier A. Likewise, protocol hashes prove immutability; domain adjudicators must
still confirm that the underlying enumeration, acquisition, and label rules are scientifically
complete.

## Implemented Tier A library gates and remaining deployment work

The legacy release-aware path has a versioned admission commitment. It binds the split-admission
hash, inventory-completeness claim, every episode manifest, source tier, and temporal-admission
digest into the public challenge manifest. The authenticated run receipt carries that admission
hash, and the private release loader cross-checks it against the exact scoring episodes and
sidecars before labels are loaded. `build_synthetic_integration_release` exercises those mechanics
with Tier C fictional data only; generic `make-challenge` construction can still omit admission.

The separate prospective path now implements the library-level official gates that were previously
missing:

- the trusted admission gate reverifies every decision seal and the externally witnessed complete
  case universe, requires every preeligible case to map to exactly one decision package, and binds a
  complete lineage-to-split inventory;
- the outcome-free cohort-release builder packages that complete admission, every policy byte, and
  the exact challenge tree, and its loader freshly reverifies both proof families;
- an independent v0.2 release-seal contract freshly replays the canonical approval from original
  campaign/readiness inputs, binds its out-of-band report pin and signed approval time, and
  witnesses the complete reverified tree strictly before submissions open;
- pre-opening attempt reservation derives output-independent attempt and alias-resistant executable
  keys; a separate post-opening typed start authorization binds the exact
  reservation/release/executable identity; after verification, a distinct stateful consumer must
  atomically accept that exact start once immediately before backend work; and the registered
  completion retains either the exact authenticated official run and response records or explicit
  failure bytes; and
- `prospective_cohort_finalizer.py` is the only official Tier A scoring entrypoint. It accepts no
  caller-supplied response, freshly reverifies the release/seal/reservation/start-authorization/completion chain,
  requires evidence for every outcome disposition plus affirmative execution of the frozen
  verifier policy, and aggregates over every preeligible case with invalid or unscored cases equal
  to zero. The per-episode helpers in `prospective_finalizer.py` are lower-level verification and
  adaptation primitives. Their caller-constructible bindings are not verification capabilities, and
  that module exposes no scoring entrypoint.

These gates are tested library contracts with injected trust callbacks, not a deployed benchmark.
A real official run still requires scheduled append-only source capture, a rights- and
domain-reviewed cohort protocol, production timestamp/transparency clients and trust roots, a
durable global registry that atomically enforces alias/attempt/completion uniqueness, a separate
durable post-opening authority that verifies and atomically redeems each typed start target once,
an audited hostile-code backend, real outcome capture and disposition evidence, and a production
implementation of the committed policy verifier. The organizer must also register one canonical finalization
identity append-only and publish a hash-bound redacted result; the private finalization artifact
contains labels and commitment keys and cannot be published as-is.

Until those production systems verify a genuinely prospective real cohort, VaxReplay has no Tier A
episode or official leaderboard suite. Tier B historical sidecars remain a separate retrospective
research track; Tier C reconstructions and the sealed synthetic pilot remain train/debug material.
