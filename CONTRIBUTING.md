# Contributing to VaxReplay

VaxReplay welcomes code, evaluation, documentation, fictional-fixture, and metadata-only
data-source proposals. The project is early-stage research software; no contribution should be
represented as a clinical, regulatory, or official leaderboard result unless it has passed the
corresponding review and admission process.

## Choose a contribution path

### AI and evaluation labs

Useful contributions include:

- model and harness adapters;
- fixed-harness and combined-system experiment protocols;
- RL environment integrations;
- adversarial tests and reward-hacking probes;
- contamination-screening methods; and
- reproducibility and receipt-verification tooling.

Start with the [fictional development challenge](benchmarks/development/iedb-fictional-v1/README.md)
and [reference-harness guide](docs/reference_harness_matrix.md). Retain every preregistered first
attempt, including wrapper and schema failures. Commit the experiment protocol before inference;
the [Cursor v0.4 protocol](benchmarks/development/iedb-fictional-v1/experiments/cursor-thinking-v04.md)
is a concrete template. Do not publish raw reasoning streams or private provider credentials.

### Biomedical and data organizations

Start with the [data-partner guide](docs/data_partner_guide.md) and the
[metadata-only issue template](https://github.com/matthew-zhao/vaxreplay/issues/new?template=data-partnership.yml).
A data partnership does not require an initial code contribution.

Never put restricted records, credentials, participant information, unpublished results, private
access instructions, or other sensitive material in a public issue, commit, or pull request. Use a
public issue only to describe metadata and establish an appropriate private review channel.

### Code and documentation contributors

1. Open an issue for substantial changes to benchmark semantics, data contracts, reward functions,
   provenance tiers, or security boundaries.
2. Keep changes narrow and add tests for changed behavior.
3. Update the relevant protocol document when a contract changes.
4. Clearly separate fictional fixtures, retrospective research artifacts, and prospective Tier A
   materials.
5. Do not weaken fail-closed checks merely to admit a new provider or dataset shape.
6. Do not submit real benchmark tasks, private gold, selected real identities, credentials,
   customer outputs, or restricted source material.
7. Use fictional or synthetic records in tests.

## Development setup

VaxReplay requires Python 3.12 or newer.

```bash
git clone https://github.com/matthew-zhao/vaxreplay.git
cd vaxreplay
uv sync --locked --extra dev
uv run python scripts/smoke_test.py
```

Before opening a pull request, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --exclude src/vaxreplay/integrations src/vaxreplay
uv run pytest -q -p no:cacheprovider
uv run python scripts/smoke_test.py
uv build
```

## Benchmark-change requirements

Changes to tasks, labels, rewards, prompts, candidate universes, or model-facing evidence can alter
what the benchmark measures. A pull request that changes one of these should state:

- the decision stage and scientific construct affected;
- whether old scores remain comparable;
- how temporal leakage and model contamination are handled;
- what new reward-hacking opportunities are introduced;
- which artifacts must be versioned or resealed; and
- whether the change affects Tier A, Tier B, Tier C, or development-only data.

Never tune a held-out suite, discard a failed first attempt, or use post-cutoff information to
repair a model response. Synthetic oracle baselines are contract checks, not systems to compare on
a leaderboard.

## Data and scientific review

Real-data work needs explicit review of panel completeness, temporal provenance, candidate identity,
outcome comparability, lineage leakage, privacy and reidentification, redistribution rights, and
domain validity. Passing engineering tests is necessary but is not scientific validation.

References to IEDB, ImmPort, ClinicalTrials.gov, WHO, NCBI, or another source must not imply that the
source endorses or partners with VaxReplay unless a written agreement says so.

## Developer Certificate of Origin

VaxReplay uses the
[Developer Certificate of Origin 1.1](https://developercertificate.org/), not a contributor
license agreement.

Every commit must include a `Signed-off-by` line certifying that the contributor has the right to
submit the work under the applicable repository license:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Create it with `git commit --signoff` or add it through the commit workflow used by your client.
By submitting a contribution, you also represent that you have not included employer-owned,
confidential, or third-party material without authorization.

## Licensing

Original software and project documentation are licensed under Apache License 2.0. Project-authored
fictional fixtures are licensed under CC BY 4.0 only where `DATA_LICENSES.md` says so. Do not submit
third-party or employer-owned material unless you have authority to provide it under the applicable
repository license.
