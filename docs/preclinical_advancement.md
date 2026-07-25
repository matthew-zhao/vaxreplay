# Preclinical candidate advancement task

VaxReplay's second task stage is `preclinical_candidate_advancement`. It evaluates whether an
agent can rank an already-defined cohort of opaque candidates for advancement using evidence that
was available at a fixed decision time.

This task does **not** ask the agent to create or modify a candidate, generate a biological
sequence, choose experimental parameters, or propose a laboratory procedure. It is a retrospective
decision benchmark over a closed candidate set.

## Episode contract

A valid episode declares:

```text
task_type: preclinical_candidate_advancement
reward_version: v1.0
```

The public episode contains opaque candidate IDs, a fixed portfolio size, decision-time evidence,
forecast targets, and evidence-assessment dimensions. A submission must:

1. rank every predefined candidate exactly once;
2. forecast the later registered validation outcome for every candidate;
3. assess each required dimension for the selected top-`k` candidates; and
4. ground every citation in an exact span from visible evidence.

The existing V1 reward is reused unchanged. Task stages and reward versions are separate axes:

```text
ranking_reward =
    0.50 * NDCG@k
  + 0.25 * strict-pair concordance
  + 0.25 * normalized top-k set utility

reward =
    0.50 * forecast_reward
  + 0.30 * ranking_reward
  + 0.20 * grounding_reward
```

Creating a new task therefore does not silently redefine what a V1 score means.

## Label and cohort requirements

The candidate cohort must be fixed without reference to future outcomes. An adjudication rubric is
registered before scoring and assigns one complete integer grade from `0` through `4` to every
candidate. The rubric must state what evidence determines each grade; it cannot normalize grades
post hoc within an episode.

Official episodes require:

- a uniform later validation target across candidates;
- the same observation horizon or an explicitly registered candidate-independent censoring rule;
- at least two distinct grades and a nondegenerate top-`k` range;
- complete ranking labels, with no imputation of missing outcomes;
- candidate IDs and record order generated independently of grades; and
- lineage separation across train and evaluation splits.

These are task-level requirements, not sufficient official-leaderboard admission. Under the
[temporal-admission policy](temporal_admission.md), Tier A additionally requires a prospective
pre-outcome seal over the complete candidate panel, decision evidence, decision definition, and
organizer-adjudicated program lineage, followed later by a private outcome receipt. Independently
archived historical versions can support Tier B retrospective research but never an official score;
post hoc reconstructions are Tier C train/debug material.

The checked-in `synthetic_preclinical_v1` fixture uses fictional high-level readiness and response
summaries. It exists only to exercise the contract and contains no candidate sequences or
experimental procedures.

## Temporal and leakage controls

Decision evidence must come from a versioned artifact that predates the advancement decision.
Later results, study conclusions, database annotations, and identifiers that enable future-record
lookup remain private. Public summaries must pass the same re-identification review required by
other VaxReplay adapters.

A real episode is not admitted merely because a paper reports several candidates. It needs an
independently timestamped protocol, registry version, release snapshot, or other credible record of
what was known before outcomes were measured.

For Tier A, that record and the complete candidate set must be captured and committed prospectively,
before the outcomes exist. A history recovered after outcomes exist can establish Tier B provenance
but cannot retroactively create a Tier A seal.

## Real-data admission gates

A proposed source must pass all of these gates before an adapter is implemented:

1. **Decision freeze:** a content-addressed, independently timestamped pre-outcome record exists.
2. **Candidate completeness:** the closed candidate set is recorded without future-conditioned
   filtering.
3. **Uniform labels:** later outcomes are comparable across all candidates and support a
   preregistered grading rubric.
4. **Rights and privacy:** redistribution, participant privacy, and source attribution permit the
   intended release.
5. **Linkage resistance:** public evidence cannot be trivially joined to current records that reveal
   the held-out outcome.
6. **Scale:** enough independent lineage groups exist for train, development, and sealed evaluation.

ImmPort plus a frozen ClinicalTrials.gov protocol history remains a leading feasibility study, not
an approved adapter. An initial strict metadata screen retained a small candidate set, but it
created no episode because supported historical freeze, arm identity, assay timing, outcome
comparability, and redistribution remain unresolved. The next step is a credentialed
three-to-five-lineage audit—not retrospective label construction from current records alone.

## Reporting

Episodes are scored independently and aggregated against a task-homogeneous `SuiteManifest` with
the evaluator-side `score-suite` flow. The suite hash is preregistered, private episode bundles are
used to score ordered raw responses, and every expected episode has equal weight. Invalid,
malformed, or missing responses receive the fixed environment penalty `-1.0` in the all-episode
mean; valid-only diagnostics are reported separately. Candidate rows and pairwise comparisons are
never pooled across episodes. The lower-level `aggregate_scores` API accepts trusted evaluator
score vectors only.
