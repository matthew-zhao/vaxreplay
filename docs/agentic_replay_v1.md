# Agentic Replay V1

Agentic Replay is the task-level VaxReplay track. A harness receives a brokered exact-byte view of
an admitted corpus, private scratch space, bounded local tools, and repeated access to an
organizer-selected model through a narrow metered gateway. It gets one attempt and produces one
final structured submission. It never receives intermediate score or label feedback.

The Direct Replay track remains a useful model-only control. Agentic Replay adds retrieval,
extraction, reconciliation, computation, and evidence-grounded decision making; it is the track for
comparing harness-plus-model systems.

## Contamination boundary

A larger workspace creates a larger artifact-leakage surface. The task instructions, paths,
filenames, catalogs, source contents, extracted text, and deterministic indexes are all
model-visible. Runtime network isolation cannot repair future information already present there.

V1 therefore follows one rule: **messy in task structure, clean in provenance**. A messy workspace
means several source-shaped text/table files plus outcome-blind distractors. It does not mean an
uninspected database dump, mutable webpage, PDF with hidden attachments, nested archive, or
organizer-written summary.

The workspace admission sequence is fail-closed:

1. Precommit the discovery, inclusion, transformation, distractor, and contamination policies.
2. Inventory every discovered source without mounting labels or outcome namespaces.
3. Verify the exact bytes and a conservative availability-time interval for every raw source.
4. Permit only deterministic, networkless, label-blind transformations whose outputs can be
   re-executed from admitted parents.
5. Compile a fresh organizer-side workspace containing only allowlisted regular UTF-8 files.
6. Hash a length-framed, unescaped surface containing every visible path and raw byte. Source
   n-grams remain detectable across newlines, quotes, and backslashes.
7. Contamination-screen that exact surface against a protected post-cutoff corpus whose selection
   policy is cross-bound to the outcome namespace committed during workspace construction.
8. Bind the temporal and contamination admissions to the workspace root.
9. Reverify the exact inventory immediately before execution, serve the committed bytes through the
   logical workspace broker, and bind that same surface into the run receipt.

The compiler tree is not a sealed worker filesystem. Read-only permissions do not hide inode, UID,
mtime, mount, or host-path metadata from arbitrary local code. V1 therefore defines worker input as
a brokered logical namespace with only `LIST`, bounded `READ`, and exact `SEARCH`; its committed
metadata is path, media type, SHA-256, and byte count. Participant code must not receive the compiler
path. The included in-process broker exercises this contract but is not a hostile-code boundary; a
production executor must expose it over authenticated IPC from outside the worker sandbox.

An LLM contamination judge is a secondary alarm. It can nominate paraphrased leaks or ambiguous
completed-result language for quarantine. It cannot establish a timestamp, rewrite a source into
admissibility, select a favorable subset, or upgrade an uncertain source version.

## Assurance profiles

The gate distinguishes four profiles instead of collapsing all historical dates into one claim:

| Profile | Required evidence | Permitted use |
|---|---|---|
| `prospective_exact` | Exact bytes and task definition witnessed before outcomes by an approved prospective authority | Prospective research input; an authenticated release seal and production isolation would still be required for Tier A |
| `independent_exact_byte` | Independent archive or signed immutable digest proves the exact source version by the cutoff | Tier B retrospective research only |
| `source_attested_best_effort` | The source labels a historical snapshot, but VaxReplay obtained or hashed the bytes later | Best-effort integration/research only |
| `fixture` | Local conformance material | Tests and debugging only |

Publication dates, database row dates, DOI metadata, PDF metadata, old URLs, and HTTP
`Last-Modified` values are availability claims, not strict exact-byte proof. Time is represented as an
interval; strict admission requires the conservative upper bound to be no later than the task
cutoff. If an interval crosses the cutoff, the source is quarantined.

The current AACT RSV pilot is `source_attested_best_effort`. Its archive is labeled 2020-02-01, but
the exact ZIP was retrieved and content-addressed in 2026 and the HTTP metadata is later. The pilot
may exercise the workspace and runner contracts, but it must not be relabeled as a strict Tier B or
hidden leaderboard case.

## Workspace

The model-visible logical namespace is:

```text
/input/                           logical broker namespace; not a host mount
  TASK.json
  TASK.md
  source-catalog.json
  sources/
    source-001.txt
    source-002.csv
    source-003.json

/scratch/                         separate writable storage; discarded
/output/                          separate writable storage
  submission.json                 only scored artifact
```

V1 accepts canonical UTF-8 text, Markdown, CSV, TSV, JSON, and JSONL. It rejects symlinks,
hardlinks, hidden files, Unicode/case-colliding paths, NUL bytes, binaries, HTML, PDFs, office
documents, SQLite databases, archives, executables, devices, FIFOs, undeclared metadata, and any
extra file. Rich formats can be added only after recursive extraction and metadata inspection are
implemented.

The fixed candidate universe is public. V1 evaluates retrieval and closed-set prioritization, not
open-set candidate discovery.

Structured public presentations are fail-closed: candidate aliases are exactly
`candidate-001..NNN`, and source IDs, paths, titles, and order are exactly `source-001..NNN`,
`sources/source-NNN.<type>`, and `Source NNN`. A private alias-permutation receipt binds the private
candidate-key commitments and source-byte commitments to those positions, the secret-seed
commitment, permutation algorithm, generator executable/configuration, execution receipt, and both
presentation-order hashes. The build policy precommits the algorithm and generator, while the later
discovery and workspace manifests bind the execution receipt digest and enforce policy → generation
→ discovery time order. This prevents an organizer from silently using identity or discovery order after announcing randomized aliases;
it does not prove that arbitrary free text was identity-scrubbed, so source contents remain inside
the contamination and transformation audit surface.

Each workspace manifest also repeats the bound episode's `synthetic`, split, label-commitment, and
reward-version properties. It exposes only a structural *prospective input* predicate and hard-codes
`official_release_ready: false`. Synthetic, train/dev, SHA-256-label, retrospective, or
non-preregistered episodes cannot satisfy that predicate. Even a structurally eligible input still
requires trusted temporal/contamination admission, an authenticated release seal, and production
isolation before it can support an official run.

## Task output and reward

The final output covers every public fact query and derived metric, then makes one decision:

- fact answers use `observed`, `not_found`, or `conflict` and cite bounded UTF-8 byte spans;
- derived answers use `computed` or `not_computable`, name their formula, and reference fact IDs;
- the decision supplies a complete ranking, portfolio, probabilities, and either `recommend` or
  `insufficient_evidence`.

The evaluator extracts cited bytes itself; it never trusts a model-supplied quote. Exact query and
metric coverage prevents cherry-picking. Duplicate or overlong spans cannot increase credit. A
wrong assertion scores below an honest abstention, while abstention earns positive correctness only
when the private gold says the value is genuinely unavailable.

At the extraction and analysis cell level, wrong / abstain / correct map to `0 / 0.5 / 1`, while a
signed `-1 / 0 / 1` diagnostic is retained. The all-abstain task reward is still zero because it
cannot pass the retrieval and decision bottlenecks.

The V1 scalar is bottlenecked rather than additive:

```text
R = evidence-group retrieval F1 inferred from citations
X = macro typed-fact extraction score
A = macro deterministic-analysis score
C = supported byte-span citation F1
D = 0.5 * continuous NDCG + 0.5 * normalized portfolio utility

process = harmonic_mean(R, X, A, C)
reward  = harmonic_mean(process, D)
```

This prevents a memorized final choice, citation copying, or mechanical extraction from dominating
the task. Forecast calibration remains a published diagnostic with zero scalar weight until the
benchmark contains enough independent lineages. Free-form rationale, chain of thought, tool-call
count, file-read count, plans, and scratch files are not rewarded.

Required baselines and ablations are: all-abstain/uniform, random ranking, lexical retrieval plus a
deterministic table extractor, oracle retrieval, oracle facts, full oracle, relevant-documents-only,
structured-facts-only, no-evidence, identifier-scrubbed, and paired secret alias permutations.

## Agent protocol

The default resource tier permits 20 model calls, 256k aggregate input tokens, 32k aggregate output
tokens, 20 minutes, 4 CPUs, 8 GiB memory, and 1 GiB scratch. Cost and latency are reported separately
from scientific reward.

Participant code may list/read/search through the logical workspace broker, write scratch, run
bounded local computation, and call `model.generate` through a per-run capability. It may not
receive provider credentials or use general network, browsers, web search, remote MCP, provider
file/vector stores, persistent sessions, prior episodes, the host workspace, repository, labels,
scorer, or raw organizer sources.

The gateway chooses the model server-side, enforces calls and tokens, and binds canonical
request/response hashes, resolved model identity, authoritative usage, and stop reasons into a
transcript. Every `model_generate` tool event must correspond one-to-one with a transcript exchange
and bind its call index, canonical request/response hashes and lengths, success state, and in-run
time interval. A local scripted gateway exists for conformance. The gateway and hostile-code
microVM boundary components are implemented as production-shaped, fail-closed contracts: the
gateway uses authenticated canonical frames, a durable at-most-once ledger, server-owned routes,
provider receipts, and a separately authenticated session; the Firecracker supervisor pins every
runtime/image digest, applies no-network/no-API resource isolation, and authenticates the lifecycle
  only after cleanup. A fixed one-shot composition service now packages the closed registry, strict
  verifier, reaper, durable local gateway tombstone, and operator. The task guest and separate
  qualification guest have both booted on a local nested-KVM development host, and a transient
  `SIGKILL`/`OnFailure` cleanup fixture has passed. A fixed-manifest installation, an integrated
  managed-service-to-real-Firecracker drill, a dedicated-host qualification rerun, and abrupt-power-
  loss/reboot recovery remain.

Run receipts bind the workspace manifest/root/surface, temporal admission, contamination admission,
policy, broker attestation, harness and model identities, usage, transcript, tool events, scratch
tree, final response, termination, and timing. The execution policy pins the broker ID, version,
and executable digest; its attestation records that the same logical surface was served before and
after the run without mounting the organizer filesystem. Local vendor-CLI runs must remain
development-only because CLI flags do not provide host-filesystem isolation, narrow provider-only
egress, or authoritative gateway tracing.

## Current implementation status

V1 implements the strict schemas, workspace compiler/loader, neutral alias contract, raw-byte
contamination adapter, temporal and research-only workspace admission, metered gateway transcript,
typed scorer, logical in-process broker, broker-attested run finalizer/loader, authenticated score
artifact, and deterministic conformance tests. Score certification re-authenticates every run
component and recomputes the score inside the trusted boundary from the exact submission and
private gold; it does not sign a caller-supplied score vector.

V1 now also implements a fail-closed Firecracker supervisor contract, authenticated worker
lifecycle, authenticated provider-gateway framing and session evidence, durable replay protection,
a bounded host/guest vsock RPC, hardened direct OpenAI Responses and Anthropic Messages adapters,
and an outer run package that
cross-binds the worker, gateway, authenticated guest RPC session, workspace, policy, harness, model
route, timing, usage, and cost. The guest RPC exposes only bounded list/read/search/model/submit
operations, records host-authoritative sequencing and exact retries, and signs its attempts,
projected events, and final submission. The deterministic and attack-oriented unit tests do not
boot Firecracker or make a live provider call on the macOS development machine. Separately, the
clinical-v2 task guest completed one pinned real AACT-derived task and the qualification guest passed
all seven drills on a local nested-Linux/KVM host; both results remain non-official and used no
external provider/model.

It intentionally cannot mint `official_benchmark` admission. The workspace-admission schema fixes
`official_release_eligible = false` and `authenticated_release_seal_present = false`; even a
prospectively witnessed input remains `prospective_research`. The current run finalizer is a trusted
post-execution handoff. The production-shaped outer handoff removes trust in caller-supplied worker
and gateway claims and now binds a separately authenticated brokered guest RPC session. It still
does not observe guest-local computation or direct scratch-disk writes, so complete tool-trace and
official-release claims remain false. Dedicated-host Linux/KVM qualification, an independently authenticated
release seal, authenticated contamination-judge traces, and externally committed verifier
executables remain deployment work. See the
[production Agentic runner guide](production_agentic_runner.md).

The code is therefore suitable for deterministic local development and integration pilots. Codex
CLI 0.144.3 now has a separate development guest adapter with a real multi-turn local-tool path,
but that adapter is not built into a pinned Linux disk or KVM-qualified. This is not yet a
defensible public leaderboard execution service for Codex, Claude Code, Cursor, or other
participant harnesses.

## Real AACT integration artifact

There is one unique real AACT case in three local organizer artifacts. The original `episode/` is a
non-synthetic, privately scored one-shot pilot with a 2020 post-completion decision view and the
recorded reference-run matrix. A second private-label-bearing/scorable build uses the earliest
permanent archive containing the trial; it is pre-results but not pre-enrollment, and it has no
separate reference-run matrix. Both are development-only and post-hoc, not admitted Tier B
releases. The separate `agentic-v1/` artifact described below was compiled from the original raw
decision slice and is intentionally unscored. Thus “zero admitted real episodes” does not mean that
no real-data episode has been built; it means that the real case has not satisfied the
historical-release gate.

[`build_aact_agentic_v1_integration.py`](../scripts/build_aact_agentic_v1_integration.py) compiles
the existing 2020 AACT decision slice into a real-data Agentic V1 workspace. The compiler admits
exactly the six receipted raw text files, assigns secret-seed neutral source and candidate aliases,
binds source-attested exact-byte proofs, and validates the resulting logical broker inventory. Its
generated local `build/aact-early-clinical-real-pilot-v0/agentic-v1/STATUS.json` is the authoritative
capability statement. Generated build artifacts are intentionally excluded from the source repository.

This artifact supports a meaningful multi-file registry task: retrieve structured fields, join
design groups to interventions, filter the older-adult experimental arms, reconstruct the
dose-by-formulation cross-product, reconcile Day-91 immunogenicity measures, compute declared
metrics, and cite exact UTF-8 spans. It does **not** support candidate ranking. The raw AACT files do
not bind `candidate-001..009` to registry arms, and adding the old normalized evidence would cross
the currently unverified derived-transformation boundary. The correct decision is therefore an
explicit abstention.

The surface also exposes the NCT identifier, sponsor/product strings, formulation names, registry
row IDs, titles, and dates. It is highly reidentifiable and cannot control later outcomes recalled
from model weights. It has no typed gold, scoring contract, temporal admission, contamination
admission, hostile worker isolation, or release seal; it is a compiler/broker integration fixture,
not a leaderboard episode.

## Residual risks

Workspace admission controls artifact leakage. It cannot remove outcomes memorized in proprietary
model weights or deliberately embedded in a participant harness after cases are known. Historical
Agentic Replay therefore always records:

```text
residual_model_weight_contamination = true
residual_harness_embedded_knowledge = true
proves_absence_of_contamination = false
```

Additional research-only limitations remain explicit: transformation and temporal verifier
callbacks are not yet dispatched by an executable-pinned runtime; retrospective task wording is
not yet produced by a preregistered generic generator; contamination judges lack
provider-authenticated gateway traces; and `attempt_reservation_sha256` is a commitment rather than
an atomically consumed one-attempt registry. Until the transformation boundary is upgraded, real
pilots should expose raw admitted sources only.

Use identity masking, reidentification probes, evidence ablations, alias permutations, delayed
aggregate feedback, and common case sets to measure those risks. Prospectively sealed, previously
unseen Tier A cohorts are the only strong long-term control.

## Scientific task requirement

Longer execution must add genuine information work, not busywork. The current RSV pilot contains
protocol definitions but no discriminative pre-decision safety/reactogenicity results. It is useful
for runner integration, not as a serious agentic vaccine-selection score. The first scored cohort
should use historical windows containing real, cutoff-admissible evidence that could have changed
the decision, or be collected prospectively before outcomes exist.
