# V1 candidate-ranking reward

VaxReplay V1 keeps the V0 forecast and grounding objectives but replaces V0's single ranking
metric with a three-part, fixed-weight ranking reward. The change makes ranking feedback denser: it
scores the order near the portfolio boundary, strict preferences across the full list, and the value
of the selected top-`k` set. It does not increase ranking's 30% share of the total reward.

This document is the normative scoring contract for episodes whose manifest declares
`reward_version: "v1.0"`.

## Inputs and notation

For one episode, let:

- `C` be the manifest's eligible candidates and `n = |C|`;
- `k` be `portfolio_size`;
- `π = (π1, ..., πn)` be the submitted best-first ranking; and
- `g(c)` be candidate `c`'s private integer relevance grade.

The submission must rank every eligible candidate exactly once. It must also contain one forecast
for every candidate/target pair and exactly one assessment for every required dimension of each
top-`k` candidate. The submission schema remains `vaxreplay.v0.1`; the manifest's reward
version selects the scorer.

## Ranking-label contract

Each V1 candidate has exactly one `RankingLabelV1` record in manifest candidate order. The
official evaluator requires:

- integer grades in `{0, 1, 2, 3, 4}`;
- complete coverage, with no duplicate or omitted candidate;
- matching episode IDs;
- no censored grade;
- at least two distinct observed grades;
- `1 <= k < n`;
- positive ideal discounted gain; and
- different best- and worst-achievable top-`k` grade sums.

The label model can represent a censored value with a nonempty `censor_reason` for provenance,
but official V1 bundle validation rejects it. Missing grades are never coerced to zero. A versioned
adjudication rubric must define what each numeric grade means for an episode family. A rubric change
requires a new adjudication version and commitments; a scoring-semantics change requires a new
reward version.

V1 commits to the unchanged private forecast/grounding labels and the ordered ranking-label records
as one canonical payload. IEDB builds use HMAC-SHA256 so low-entropy grades cannot be enumerated from
a public digest. Sealed V1 test manifests require HMAC-SHA256; an unkeyed SHA-256 commitment is
permitted only for non-test development artifacts.

## Ranking metrics

### NDCG at portfolio size

For a submitted order, discounted cumulative gain is:

```text
DCG@k(π) = sum from i=1 to k of (2^g(πi) - 1) / log2(i + 1)
```

`IDCG@k` is the same quantity for candidates sorted by decreasing grade. Candidate ID is used
only to make the ideal ordering deterministic when grades tie; exchanging equal-grade candidates
does not change gain.

```text
NDCG@k = DCG@k / IDCG@k
```

Bundle validation requires `IDCG@k > 0`.

### Strict-pair concordance

Let `P` contain every unordered candidate pair `{a, b}` for which `g(a) != g(b)`.
A pair is concordant when the higher-grade candidate appears first in the submitted ranking.

```text
pairwise_concordance = concordant strict pairs / |P|
```

Gold ties are excluded rather than broken by candidate ID. At least one strict pair is guaranteed by
the requirement for two distinct grades. This metric uses the full ranking, so an inversion below
the top-`k` boundary still changes reward.

### Normalized top-k set utility

Let `S(π)` be the sum of grades in the submitted top-`k` set. Let `S_best` be the sum
of the `k` largest grades and `S_worst` the sum of the `k` smallest grades.

```text
top_k_utility = (S(π) - S_worst) / (S_best - S_worst)
```

Bundle validation requires a nonzero denominator. The metric depends only on membership in the
selected set, not the order within that set.

### Ranking composite

```text
ranking_reward =
    0.50 * NDCG@k
  + 0.25 * pairwise_concordance
  + 0.25 * top_k_utility
```

All three components and the composite lie in `[0, 1]` for a valid episode and submission.

## Total episode reward

The forecast component is one minus mean Brier score over non-censored private outcome records:

```text
forecast_reward = 1 - mean((predicted_probability - binary_outcome)^2)
```

Forecast censoring is separate from ranking censoring: censored forecast outcomes are omitted from
the Brier mean, and at least one observed forecast outcome is required.

The grounding component is:

```text
grounding_reward = evidence_span_F1 * assessment_accuracy
```

Evidence-span matching is exact and one-to-one against gold citations for the submitted top-`k`
candidates. Assessment accuracy is exact agreement on their required assessment conclusions.

The fixed V1 reward is:

```text
reward =
    0.50 * forecast_reward
  + 0.30 * ranking_reward
  + 0.20 * grounding_reward
```

The score-vector model independently recomputes forecast reward, grounding F1, grounding reward,
and both composites, rejecting values that disagree by more than `1e-12` absolute error. A schema,
coverage, citation, or leakage failure produces an
invalid score vector with issues and no scalar evaluator reward. The one-turn RL environment maps
such an invalid result to `-1.0` and reports `valid_submission = 0`.

## Ties, censoring, and aggregation

- Equal gold grades create no pairwise preference. Their order is also neutral in NDCG and top-`k`
  utility, except when set membership crosses candidates with different grades.
- Submitted ties are not representable: a submission is a complete total order of unique candidate
  IDs.
- Official V1 ranking grades must be complete. Censored grades invalidate the episode rather than
  being dropped, imputed, or scored as failures.
- Score each episode independently. For RL rollouts, macro-average the environment reward over the
  fixed episode set, so an invalid episode contributes the environment's `-1.0`. For evaluator
  diagnostics, macro-average each metric over valid episodes and always publish the validity count
  and fixed-set denominator; never silently drop an invalid episode from the report.
  `aggregate_scores` implements this contract against a task-homogeneous `SuiteManifest`. Missing
  expected scores receive `-1.0`; score bindings must match the suite's manifest and label
  commitments. The result records both `suite_manifest_sha256` and `input_scores_sha256`. Do not
  pool candidate pairs, forecast rows, or candidates across episodes, because pooling would let
  large cohorts dominate the benchmark.

  `aggregate_scores` assumes its inputs came from a trusted evaluator; its hashes are commitments,
  not signatures. The evaluator-side CLI exposes `score-suite`, which verifies a preregistered
  suite hash and scores an ordered response file, and `score-run`, which first authenticates and
  validates a sealed-runner artifact. Both load private episode bundles and invoke the same
  in-process deterministic scorer. Sealed scoring requires HMAC-SHA256 label commitments. Private
  bundles and keys are never distributed; environments, oracle baselines, and normal single-episode
  scoring still reject sealed tests. A malformed or non-UTF-8 response row is recorded as missing
  and receives the fixed `-1.0` suite penalty without aborting other rows.

## Reward-hacking resistance

The three ranking views close different shortcuts:

- NDCG emphasizes placing high-grade candidates early, but alone is insensitive to order below
  `k`.
- strict-pair concordance makes every unequal-grade inversion observable, including in the tail;
- normalized top-`k` utility directly checks portfolio membership and cannot be improved merely
  by permuting a fixed selected set; and
- keeping the combined ranking weight at `0.30` prevents the V1 change from simply inflating the
  contribution of ranking relative to calibration and evidence grounding.

Complete labels, nondegenerate episodes, fixed formulas, versioned rubrics, exact score-vector
validation, and commitments over the ranking labels prevent common missing-label, denominator,
weight-tampering, and post hoc relabeling attacks. These controls do not make a label rubric
scientifically valid; cohort construction and adjudication still require external review.

## V0 compatibility

V0 and V1 dispatch on `manifest.reward_version` and deliberately use separate score and private
ranking-label models.

- V0 remains `reward_version: "v0.1"` and uses candidate utility for its sole `NDCG@k`
  ranking term.
- V0's total reward remains `0.50 * forecast + 0.30 * NDCG@k + 0.20 * grounding`.
- V0 bundles do not require `private/ranking_labels.jsonl`, and their private-label commitment
  payload is unchanged.
- V1 stores grades in `private/ranking_labels.jsonl`, binds those grades into the private-label
  commitment, and returns the separate V1 score vector with all three ranking diagnostics.

Thus existing V0 episodes, hashes, submissions, and score-vector serialization do not need a data
migration. A bundle opts into the new behavior only by declaring `reward_version: "v1.0"` and
providing valid committed ranking labels.
