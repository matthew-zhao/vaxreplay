# VaxReplay threat model

VaxReplay is designed for trainable public episodes and one-shot sealed evaluation. This document
tracks benchmark-specific leakage and reward-hacking risks; it is not a biological safety model.

## Data tiers and independent seals

Only Tier A, prospectively sealed data is eligible for an official leaderboard. Tier B consists of
independently archived but retrospectively assembled episodes and is restricted to a separate
research track. Tier C includes post hoc reconstructions or episodes missing any required receipt
and is train/debug only. Tier A, B, and C episodes must never share a suite or aggregate.

Tier A requires several independently checked commitments. A **prospective data seal** freezes each
universe/panel, evidence view, decision, and organizer-adjudicated pathogen/program lineage before
its outcome exists. A **complete cohort release** then binds every eligible case, split, policy byte,
and exact challenge artifact, and an external **pre-run release seal** witnesses that whole tree
strictly before submissions opening. Its v0.2 target also commits the canonical Tier A approval identity
obtained by replaying the original campaign/readiness inputs under an independently pinned approval
report digest, including its authority-signed verification time. The release witness must be later
than that approval time and strictly earlier than submissions opening. Before opening and execution,
a global append-only registry must
reserve the exact system and its first-and-only attempt under an alias-resistant executable key.
At or after opening and strictly before the deadline, a separate external start authority must
authenticate the exact reservation/release/executable target. After idempotent proof verification,
its stateful consumer must atomically accept that exact attempt/start once immediately before
backend preparation. The attempt then ends in a retained terminal success
or failure whose start cannot predate opening or authorization; success binds the exact response
bytes that alone may be scored. The terminal completion proof must be witnessed at or after the terminal event, and
cohort finalization cannot predate that external proof. After outcomes mature, the private finalizer applies the already
committed disposition policy to every case and keeps labels inside the trusted evaluator. A sealed
runner cannot upgrade Tier B or Tier C data to Tier A. The aggregate denominator is the complete
preeligible universe; invalid and policy-verified unscored cases contribute zero rather than being
dropped.

The four required receipts are:

1. a universe/panel receipt proving the complete eligible opportunity set at the cutoff;
2. an evidence receipt binding exact source bytes/rows, versions, queries, acquisition times, and
   candidate mappings;
3. a decision receipt binding the cutoff, task, portfolio, rubric, and prespecified decision; and
4. a private outcome receipt binding the later endpoint, first availability, horizon, censoring,
   adjudication, and label derivation.

The first three form each episode's prospective data seal; the fourth is attached only after the
endpoint matures. Those episode receipts are necessary but not sufficient: Tier A also requires the
pre-opening complete-release witness, the strictly pre-opening system/attempt reservation, the
post-opening external start authorization, the terminal run and completion records, and exhaustive
whole-cohort outcome disposition. Independent timestamps, content
hashes, and append-only acquisition and attempt registries bind these artifacts to one release,
episode, lineage, and system identity.

The release archive is not allowed to declare its own trust roots. A campaign trust policy must be
distributed out of band and precommit the exact control artifacts and required worker inventory,
including retained external-signer/clock configurations and their exact executable bytes; the
signed campaign manifest must then receive receipts from at least two policy-distinct publishers.
Scope-derived readiness authorities sign the exact release subjects. A separate policy-pinned time
authority signs the exact verification time together with the release and readiness-policy/manifest
bindings. The composed release-decision verifier requires readiness statements to postdate and
cross-bind the campaign publication and rejects reuse of campaign identities, declared organizations,
failure domains, or signing keys by readiness evidence/time authorities. This authenticates the
configured declarations and claims. It does not cryptographically prove that the named organizations
are genuinely independent, that the time authority's clock was honest, or that claims are
substantively correct.

## Two distinct contamination channels

**Artifact leakage** means future information changed an episode artifact: its candidate universe,
evidence, decision, public text, or labels. Temporal receipts and cohort-completeness audits address
this channel. **Pretrained-model or packaged-system contamination** means outcomes were memorized in
weights or embedded in harness code, images, caches, or assets. Opaque IDs, cutoff filtering, and
network denial do not address that channel; it requires pre-reveal system commitments or private
outcome embargo, prospective refreshes, re-identification probes, and contamination canaries.

## Assets that must remain private

- Future outcomes, candidate utilities, and V1 relevance grades
- Future IEDB references and assay identifiers
- Candidate-to-source-database mappings for sealed episodes
- Gold assessment conclusions and evidence spans
- Label/spec/private-audit HMAC keys
- Detailed per-episode sealed-test scores

## Threats and current controls

| Threat | Current control |
|---|---|
| A sealed runner is used to claim that retrospective data are Tier A | The temporal sidecar verifier requires a pre-outcome universe/panel, evidence, and decision seal plus independently verified receipts. Runtime sealing is separate. Tier B has a distinct `retrospective_research` builder, requires a complete split inventory and official isolation policy, remains `sealed_eligible: false`, and cannot enter Tier A aggregates. The prospective cohort-release builder accepts only already witnessed, non-synthetic decision packages; the existence of that library builder does not make historical or fictional inputs Tier A. |
| Individually sealed cases are selectively omitted before systems run | The prospective admission gate requires a complete pre-outcome case universe and split inventory, with every preeligible case bound to exactly one decision package. The cohort-release builder packages that complete admission and exact challenge tree; a separate external release seal witnesses the whole reverified tree before submissions open. |
| A release signer invents its own registry, witness, clock, worker, or publisher roots inside the archive | Campaign verification starts from an expected out-of-band trust-policy digest, exact-precommits every non-archive control artifact and required worker, retains and verifies each signer/clock process configuration plus its exact executable, verifies worker provenance and policy-distinct publication receipts, and rejects extra or missing inventory. Scope-derived readiness evidence is freshness-checked against authenticated signed verification time, and the final decision cross-binds it to the same manifest, trust policy, archive, and index. Real organizational independence and truth of signed claims remain external review facts. |
| A caller backdates `verified_at` to make stale readiness evidence pass | The readiness policy pins a distinct verification-time authority and raw public-key digest. Its signed statement binds the release ID, readiness-policy digest, readiness-manifest digest, and exact `verified_at`; both readiness and final-decision verification require the statement and exact key. A compromised or dishonest time authority can still lie, so protected-clock deployment and review remain required. |
| One operator reuses a campaign authority or key as an apparently external readiness authority | The final decision requires the readiness organizer identity to match the campaign release identity, then rejects readiness evidence/time authorities that overlap campaign release/build/publisher/registry/witness/gossip/worker/sandbox authority IDs, declared organization/failure-domain IDs, or signing-key digests. These are exact policy checks, not proof that distinct labels correspond to independent real organizations. |
| Future bootstrap or provenance material is presented as an older release root | Publication rejects a future-dated manifest and bounds registry/witness bootstrap times, gossip-source issuance times, and worker-provenance creation times by the signed manifest creation time. Publisher receipts must be at or after the manifest, remain within the policy window, and not exceed verification time. |
| Oversized publication artifacts exhaust an offline reviewer or create inconsistent CLI limits | The canonical manifest limits each artifact to 256 MiB, verification limits the complete artifact inventory to 1 GiB, and the campaign/final-decision readers enforce the shared per-artifact bound while checking the exact inventory. These limits do not replace deployment-level archive storage quotas. |
| A readiness policy under-declares the task, source, or model-leaderboard gates while approving a broader release | Prospective campaign archive index v0.2 commits the strict canonical `TierAReleaseScope` and its digest. Semantic approval v0.2 requires exact index/readiness-policy scope equality. Archive loading independently derives the homogeneous suite task and, for official releases, the exact sorted source-category union from canonical promotion handoffs' embedded capture indexes. Unknown production-verifier IDs and verifier/source-namespace mismatches fail closed. Official execution approval identities additionally require `includes_model_leaderboard=true`. |
| A stateful caller mapping returns benign bytes during verification and different bytes during report construction or materialization | Campaign-publication, release-readiness, release-decision, and semantic-approval composition snapshot their mapping/receipt inputs once into exact byte collections before any verification and use only those snapshots downstream. Invalid keys, values, duplicates, or failed iteration are rejected rather than reread. |
| Failure cleanup follows a replaced path and deletes unrelated content, or leaves a partly verified artifact looking installed | The shared archive/materialization, release-seal, reservation, start-authorization, completion, and finalization publisher holds the authenticated parent and operation-owned directory descriptors and device/inode identities. It installs with descriptor-relative atomic no-replace rename. Cleanup detaches only the matching owned tree into its private descriptor quarantine and removes it there. Replacements are left untouched; unavailable safe primitives, identity drift, or incomplete removal fail closed and retain incident evidence. |
| A caller bypasses semantic campaign/readiness approval by sealing an otherwise valid official-looking release | Release-seal target v0.2 can be built and loaded only while freshly replaying semantic approval from its original authority inputs under an out-of-band report digest. The target commits the canonical approval identity, its signed verification time, and exact release/tree/challenge identities. That approval time is a seal prerequisite and the external witness must still precede submissions opening. Reservation hashes that exact seal target; the typed start authorization hashes the reservation plus the release/executable identities; completion hashes the reservation, authorization target, and start proof; and finalization/scoring reload the chain with the same approval pin and replay inputs. Serialized reports and previously returned approval objects are not authorization capabilities. |
| The organizer obtains several runs under renamed submissions and publishes the best | The system/attempt reservation is witnessed strictly before submissions open under a global append-only registry. The alias-resistant executable identity intentionally excludes participant-chosen submission ID; a separate cohort/track/executable alias key lets the trusted registry reject re-entry under a different registered entry or mapped cohort alias. After opening, a separate external authority issues the exact typed authorization; after every static check, its distinct stateful consumer atomically accepts that exact attempt/start once immediately before backend preparation. Replay is a typed non-retryable failure before a second prepare call. Local artifacts and idempotent signature checks do not establish either global property. |
| A run starts before opening or before its authorization, then receives a convenient later proof | Start proofs before opening or at/after the deadline are rejected, execution composition verifies the independently pinned authorization before any backend work, and successful-run or retained-failure start time must be at or after both opening and the authenticated start witness. Completion transitively binds the exact authorization manifest, target, and proof. |
| A failed first run is deleted and retried | The one-attempt policy commits `first_and_only_started_run`, `retry_on_failure=false`, and `failure_disposition=retain_as_invalid`. The external start consumer must atomically enforce global one-shot execution. Ordinary post-consumption backend failures produce registry-ready canonical failure targets; indeterminate or unrepresentable states are typed non-retryable incidents carrying durable attempt/start identities for orphan reconciliation. Only terminal success can carry scoreable response bytes. |
| A participant or organizer replaces the response after seeing outcomes | The externally registered completion binds the reverified official run receipt, complete response file, and each ordinal response record. The cohort-atomic official scorer has no submission argument: it parses only those records and rejects caller-supplied replacement submissions, reconstructed output, mismatched artifacts, or an unreserved run. The lower-level per-episode scoring helper and older post-run timestamp sidecar do not establish an official score or first-attempt fairness by themselves. |
| Missing or conflicting outcomes are classified after seeing scores | In v0, the release's exact `verifier-policy.bin` is also the precommitted selection/missingness/conflict policy. Its hash must equal the complete case-selection audit's policy hash; every sealed-universe case requires one disposition and nonempty evidence; and a caller-supplied trusted verifier must execute the frozen policy over the full universe/audit/evidence map. Aggregation divides by every preeligible case, counting invalid or unscored cases as zero. |
| A Tier B enum is attached to unverified paper metadata | Tier B literature admission binds exact raw and text-view bytes, verifies an independent exact-byte archive proof for every source and later-outcome version, resolves availability from verified witnesses, and rejects organizer attestations. The release builder requires trusted source, universe, and temporal-receipt callbacks. |
| Later outcomes select or shrink the historical panel or whole-case cohort | The decision API accepts no outcome input; an injected deterministic panel-completeness verifier runs over the admitted decision namespace, every matching source row is included or explicitly excluded, and the later join audit retains every frozen candidate exactly once. A separately sealed pre-outcome case universe and exhaustive post-outcome selection audit expose every admitted, predefined-excluded, missing, and conflicting case. |
| An LLM filter silently declares historical cases “clean” | The contamination screen binds the exact final model-facing bytes, protected comparison bytes, policy, judge pins, prompts, configurations, canonical outputs, and calibration results. At least two distinct judges are required. Pass retains `residual_contamination_possible=true` and `proves_absence_of_contamination=false`; disagreement or failed calibration cannot auto-pass. |
| Semantic similarity removes correct pre-cutoff predictions after outcomes are known | Exact and identifier scans nominate review candidates but do not edit evidence. Structured judges must cite exact public and protected spans. N-gram similarity, suspicion, any finding, or judge disagreement routes to manual review; only identifiers, high-severity findings, or contaminated verdicts auto-quarantine. The complete audit covers every preeligible case under one policy. |
| A quarantined case is silently omitted to improve historical scores | `quarantined_contamination` is a first-class case-selection disposition. Tier B packaging requires a complete contamination manifest bound to the sealed universe, accepts only passing admitted cases, and publishes the policy and private-manifest commitments without publishing protected findings. |
| The audit covers source files but not the prompt actually delivered to the model | Release admission recomputes canonical bytes for the final `full` system/user messages and requires every admitted audit to bind their exact hash and byte count. Fixed scrubbed and no-evidence variants have distinct challenge commitments. |
| Prompt-injection text manipulates the organizer-side LLM auditor | Audit material is treated as untrusted data, outputs must satisfy a closed schema and exact-span checks, and model prose cannot set the release disposition directly. Injection canaries belong in the private calibration batch; disagreement or malformed coverage fails closed. |
| A different low-risk subset is chosen for every submitted model | Artifact admission fixes one common passing subset before target systems run. Model-specific recall/reidentification probes annotate exposure risk and never change that system's denominator. |
| Reward configuration is changed after outcomes are visible | `LiteratureEvaluationConfig` is embedded in the decision package and covered by its independent seal. Portfolio size, target/horizon, required dimensions, lineage, split, and adjudication must match the final scoring manifest. |
| Valid but unrelated literature, temporal, and scoring artifacts are mixed | The private source audit binds the episode-manifest hash. Release verification deterministically re-derives candidates, evidence, gold assessments, labels, and ranking grades, then cross-checks the raw outcome and all three protocol artifacts against the Tier B temporal material. |
| A public literature manifest leaks a raw outcome digest or source locator | Literature episodes publish no outcome-derived source provenance. Adversarial tests assert that the outcome canary, raw outcome hash, outcome package hash, source URL, and unmatched rows are absent from the public bundle. |
| An LLM self-asserts that extraction was label blind | Extraction is limited to exact spans in admitted text, cannot select candidates, binds model/code/prompt/config plus a trusted runner receipt, and requires `network_allowed=false` and `outcome_namespace_mounted=false`. The extraction-receipt verifier is organizer-controlled trusted code. |
| A model, harness, or image is updated after outcomes become public | The exact system manifest, including model snapshot, harness commit, image digest, prompt compiler, and policy, is committed before outcome reveal; otherwise outcomes remain privately embargoed through its one-shot run. |
| Future evidence appears in the prompt | Prompt construction filters by `available_at <= decision_at`; bundle integrity requires gold citations to be visible. |
| Future outcomes influence which preclinical candidates enter the cohort | Real development-stage episodes require an independently timestamped pre-outcome candidate set and reject future-conditioned cohort construction. |
| An outcome is privately known before the decision but published later | `first_label_available_at` must mean earliest availability to any source owner, investigator, curator, or organizer. The trusted receipt verifier and raw-source audit must authenticate that claim; inability to rule out earlier private access downgrades the episode from Tier A. |
| A development-stage agent invents a new candidate or procedure | The prompt explicitly forbids generation and procedures; the strict submission schema can rank only manifest candidate IDs and rejects extra output fields. |
| A historical publication is curated after the cutoff | IEDB availability is first observation in a pinned snapshot; publication year is never used as availability. |
| A post-cutoff correction becomes a new outcome | Logical assay first-seen time must be after the cutoff; versions of pre-existing assays cannot label the cohort. |
| Snapshot download crosses an IEDB rebuild | Parsed `api_metrics` captures from before and after acquisition must be identical and match every required table build. |
| Missing/random API pages create false new records | The current adapter accepts one complete page only, requires the endpoint's unique ordering key, exact `Content-Range`, request hash, and identical query/schema across history. |
| Public artifact directly contains labels | `export-public` omits `private/`, rejects symlinks, canonicalizes parsed core models to remove duplicate-key/whitespace channels, reconstructs mutable metadata, and includes decision evidence only with no class counts. |
| Public provenance fields reveal candidate-to-IEDB row mappings | Evidence IDs are HMAC-derived, provenance URLs omit queries, and public derivation text omits raw-table and normalized-row hashes; exact mappings remain private. |
| Public evidence text is re-linked to current IEDB rows | `export-public` currently refuses every non-synthetic IEDB episode. A real release requires a non-reversible aggregate/redacted representation and adversarial linkage test. |
| Binary labels or V1 grades are recovered from a public digest | Sealed V1 test manifests require HMAC-SHA256. IEDB forecast/grounding labels and ordered ranking grades share an evaluator-keyed commitment; episode-spec and private-audit commitments are also keyed. |
| Missing or censored grades are silently scored as failures | Official V1 bundle validation requires exactly one observed integer grade per candidate in manifest order; censored grades are rejected rather than dropped or imputed. |
| An episode advertises a rubric the adapter did not implement | The IEDB spec accepts only the registered `iedb-qualitative-binary-v1` rubric, and the builder independently rejects any other value. |
| Degenerate cohorts make ranking metrics undefined or trivially constant | V1 requires at least two distinct grades, `1 <= portfolio_size < candidate_count`, positive ideal DCG, at least one strict pair, and distinct best/worst top-k sums. |
| Arbitrary tie-breaking creates false reward differences | Gold-grade ties are excluded from strict-pair concordance; NDCG and top-k utility are invariant to permutations within an equal-grade group. |
| Optimizing only the top-k order ignores the rest of the ranking | V1 combines NDCG at `k` and top-k set utility with strict-pair concordance over every unequal-grade pair in the full ranking. |
| Pooling candidate pairs lets large cohorts dominate a release score | `aggregate_scores` gives every episode equal weight; pairwise comparisons, forecast rows, and candidates are not pooled across episodes. |
| Invalid episodes are omitted to inflate a suite score | A task-homogeneous `SuiteManifest` fixes the expected episode bindings; every invalid or missing score receives `-1.0`, with status counts and validity published. |
| A suite summary is detached from its definition or episode scores | `suite_manifest_sha256` and `input_scores_sha256` commit the aggregate to both the suite definition and canonical score inputs. |
| A participant fabricates perfect score vectors | The evaluator-side CLI does not accept score vectors. `score-suite` evaluates ordered raw responses against private labels; `aggregate_scores` is explicitly evaluator-internal and unauthenticated. |
| A participant replaces the suite with a shorter manifest | `score-suite` requires a preregistered suite hash and reconstructs the manifest from the evaluator's supplied episode bundles before scoring. |
| One malformed response aborts scoring for the rest of a sealed batch | `score-suite` treats each malformed or non-UTF-8 row as a missing episode response, assigns the fixed `-1.0` suite penalty, and continues with the remaining rows. |
| Reward weights or component scores are altered after evaluation | `reward_version` selects fixed V1 weights, the score vector exposes every component, and model validation recomputes its derived rewards and composites to `1e-12` absolute tolerance. |
| Whole-document citation stuffing earns grounding reward | Gold quote matching is exact and one-to-one. |
| Extra assessments/citations inflate recall | Submission coverage is exact; duplicate or off-portfolio assessments are invalid. |
| Post-cutoff or invented citations trade off against other reward | Unknown and post-cutoff sources are fatal invalid-leakage results with no scalar reward. |
| Repeated test probes reconstruct labels | Environments, oracle baselines, and normal single-episode scoring reject sealed tests. Only the evaluator-side `score-suite` path or authenticated `score-run` path may score them, every sealed bundle must use an HMAC-SHA256 label commitment, and official evaluation must be one-shot with delayed aggregate reporting. |
| A worker discovers private labels or post-cutoff rows through an episode-directory mount | Challenge construction renders the two public messages into a fresh allowlisted artifact. Workers receive one envelope over stdin and no host mounts; source episode directories are never exposed to the runner backend. |
| A mutable image, implicit image command, or declared volume changes the admitted system | System images must be local and SHA-256 pinned. The OCI preflight records the resolved image ID and rejects image-declared `CMD` and `VOLUME`; the complete entrypoint argv is manifest-bound. |
| A worker uses network, host files, privilege, persistent state, or unbounded scratch | The OCI development backend uses no mounts, network `none`, a read-only root, non-root UID, dropped capabilities, no-new-privileges, private IPC/cgroups, bounded tmpfs/shared memory, CPU/memory/PID/file limits, and one fresh container per episode. Official evaluation still requires a stronger dedicated backend. |
| Output or log flooding exhausts the evaluator | Host-side capture bounds stdout and stderr independently, destroys the isolation unit on overflow or timeout, writes a fixed `null` response row, and retains only bounded private audit streams. Aggregate suite/output limits are checked before launch. |
| A run artifact is edited, forged, or rows are shifted before scoring | The run receipt binds the challenge, suite, canonical system and policy, resolved image, every envelope, response row, and bounded audit stream by hash and size. An organizer HMAC and preregistered key ID authenticate that receipt; `score-run` verifies it, exact file allowlists, and one ordered row per episode before loading private labels. Development and official releases use separate keys. |
| A development container result is presented as officially sealed | Runner policy defaults to `official`; the implemented OCI backend refuses that tier and always emits `sealed=false`. `score-run` rejects development receipts unless an operator explicitly enables the local-testing flag. |
| Censored forecast outcomes are treated as failures | Forecast censoring is independent of ranking: censored outcomes are excluded from Brier scoring, while official V1 ranking grades must be complete. |
| Related episodes cross train/dev/test or task types | Dataset loading rejects episode-ID and `lineage_group_id` overlap across train, dev, and test while pooling task types for the overlap audit. |
| One pathogen/program is fragmented into several opaque lineage IDs | Organizers adjudicate `lineage_group_id` before labels at the pathogen/program level, including aliases and prespecified same-product/platform-family relationships. Hashing IDs is not evidence of independence. |
| Context overflow is cheaper than malformed output | Optional Tinker integration sets context-overflow and parse-failure rewards to the same `-1.0`. |
| Fuzzy study matching creates false or outcome-conditioned ImmPort/registry links | The feasibility inventory links only an explicit unique `NCT\d{8}` supplied by ImmPort. Titles, sponsors, investigators, and intervention text are never match keys. |
| Discovery metadata directly reveals study conclusions | Normalized inventory records use an allowlist of accessions, counts, dates, enums, assay-method names, and hashes; titles, descriptions, contacts, locations, and assay values are excluded. |
| Publishing the feasibility cohort contaminates a future sealed split | `export-summary` omits record-level mappings, hashes, and free text, but the public exact links can still be reconstructed. Any genuinely sealed subset must depend on private authenticated eligibility findings or a later organizer-held split. |
| A current registry record is stripped of results and mislabeled historical | Historical feasibility requires a complete earlier record version. The inventory separately reports whether the history surface is supported, whether the posted date is actual, and whether the version predates results. |
| Current ImmPort assay metadata is treated as if it existed at the study's first release | Real temporal eligibility cannot pass from a declared date alone. It requires a future offline audit of the frozen release-diff artifacts and archive coverage; until then the assay temporal gate stays `not_assessed`. |
| Arm-count equality is mistaken for candidate identity or comparable labels | Arm mapping and outcome comparability remain explicit `not_assessed` gates until authenticated ImmPort tables prove identity and per-arm assay coverage. |
| Tiny or sliced feasibility summaries reveal cohort membership | Real public export requires at least 20 source records and applies primary plus complementary suppression for nonzero counts below 5. Organizers must publish one preregistered whole-inventory summary, not overlapping slices that enable differencing. |

## Required sealed-evaluation deployment controls

The repository alone cannot create a secrecy or global-uniqueness boundary. A sealed evaluator must:

1. admit only a complete prospective cohort whose case/split inventories and episode-level data
   seals all verify;
2. publish the campaign/readiness trust roots through genuinely independent locations, obtain a
   signed release-specific verification time, pin the canonical semantic-approval report digest,
   replay the original approval inputs, and externally witness the exact v0.2 release-seal target
   containing that approval identity before submissions open;
3. reserve the canonical cohort/track and alias-resistant executable identity plus first-and-only
   attempt in a durable global registry before submissions open and execution, rejecting conflicts
   across all organizers and workers;
4. at or after opening and strictly before the deadline, redeem one independently authenticated
   typed start authorization over that exact reservation/release/executable identity, then execute
   once in an official isolation boundary and durably retain terminal success or failure;
5. externally register exactly one completion at or after its terminal time, retaining either the
   authenticated run/response records or explicit failure bytes, and score only that completion;
6. distribute only the exact public challenge artifact and retain outcome files, private audits,
   label commitments, and HMAC keys inside the trusted evaluator;
7. provide evidence for every disposition and run a trusted implementation of the precommitted
   verifier/selection policy over every case, including missing and conflicting outcomes;
8. prevent network and filesystem access beyond the public challenge;
9. aggregate over the complete preeligible-case denominator, treating invalid or policy-verified
   unscored cases as zero, then delay reporting across enough cases to prevent differencing attacks;
10. log release, suite, reservation, terminal record, receipt, response, reward, label-key, model,
    harness, verifier, and evaluator identities; and
11. rotate commitment keys between benchmark releases.

The repository implements fail-closed artifact models, builders, loaders, and injected verifier
boundaries for this lifecycle, plus single-host public static-HTTPS and authenticated ImmPort
capture contracts with a separate inherited-pipe/socket one-shot producer. They have local
structural verification and semantic replay for every
successful supported run, but they are not a production dynamic ingestion service and supply
neither source-wide completeness nor a deployed independent time authority. Durable selection and
witness services and checkpoint-gossip monitors are implemented, not independently operated; a
real Tier A claim needs separately administered deployments, protected keys/clocks, accepted
out-of-band trust anchors, retained runtime process configurations/executables, independent
publishers and readiness/time authorities, monitoring, and archives. The repository checks exact
authority/key separation and timestamp bounds but cannot establish that declared organizations or
failure domains are genuinely independent. The ImmPort producer and bounded supervisor
are implemented, while its secret broker, restricted deployed workload, private storage boundary,
real credentialed qualification, and collector attestation are deployment-supplied trusted
controls. The source-worker supply-chain verifier closes retained-byte and ordered-rootfs
consistency, but does not attest the builder or running host. The general model-runner Docker
backend remains development-tier. The Firecracker task and qualification guests have produced
local nested-KVM development evidence, but the managed end-to-end path and dedicated-host release
deployment remain unqualified; see
`docs/sealed_runner.md` and `docs/production_agentic_runner.md` for the remaining deployment gates.
These controls make the repository Tier-A-capable; they do not make any checked-in fixture,
capture, cohort, run, or dataset Tier A.

## Open risks

- Snapshot hashes prove artifact consistency, not that an organizer honestly obtained the bytes from
  IEDB. Production releases need signed acquisition manifests or a trusted transparency log.
- Future-conditioned cohort replay measures prioritization within a known assayed cohort. It does not
  measure discovery of the opportunity set and can overstate real-world usefulness.
- Public train/dev episodes can contaminate model pretraining or later benchmark submissions. The
  official result needs rolling, prospectively sealed refresh cohorts and contamination canaries.
- IEDB qualitative outcomes are author-reported and assay-context dependent. They are not a universal
  biological threshold or a proxy for clinical protection.
- A WHO vaccine-composition recommendation is a policy/action outcome, not biological ground truth,
  a clinical-protection label, or proof that the selected composition was uniquely optimal.
- The current IEDB adapter derives both binary ranking grades and forecast outcomes from the same
  future qualitative assay signal. Their reward components are correlated and must not be presented
  as independent validation evidence.
- Integer relevance grades make a rubric operational, not scientifically correct. Rubric versions,
  cohort definitions, and adjudication decisions still require domain review and post-evaluation
  audit.
- Human-curated evidence summaries can contain prompt-injection-like text. The adapter uses structured
  fields and strips HTML, but model-side instruction hierarchy must still treat evidence as data.
- Even without direct row identifiers, distinctive evidence text can be reidentified against an
  external IEDB copy. Closed-book/no-network execution is not enough for models that memorized the
  current database, so real IEDB public export remains blocked until the representation changes.
- A submitted image can encode an external lookup table in its weights, code, or assets. Isolation
  controls dynamic retrieval, not parametric or deliberately packaged knowledge.
- Receipt HMACs authenticate the trusted organizer pipeline but are not publicly verifiable
  signatures. Public auditing still benefits from an asymmetric signature or append-only
  transparency log.
- The signed release-verification-time statement prevents an unauthenticated caller from selecting
  another `verified_at`; it cannot prove that the external authority's clock, key custody,
  organization, or failure domain was honest. Those facts require deployment evidence and drills.
- Descriptor-relative publication detects namespace replacement before and after install/cleanup,
  restores a raced unrelated object when that can be done without overwrite, and otherwise retains
  incident state. No userspace rename protocol can defeat an endlessly concurrent process with the
  same write authority over the parent directory. Production parents must therefore be precreated
  for and writable only by the dedicated publication authority; participant, model, adapter, and
  verifier processes must run under different credentials without that namespace capability.
- The Tier A approval, release-seal, reservation, start-authorization, completion, execution, and finalization loaders
  ultimately rely on caller-supplied decision-receipt, case-universe, and source-capture
  proof-verifier callbacks.
  Release sealing does not accept a callback that manufactures an approval: it accepts the typed,
  snapshotted original inputs and invokes the concrete semantic composition in a fresh temporary
  directory. Its out-of-band report digest and exact
  identity are committed in the externally witnessed v0.2 release-seal target; reservation,
  start authorization, completion, and finalization transitively bind that target through canonical
  SHA-256 identities.
  The library tests fail-closed byte and timing contracts, but production trust roots,
  signature/transparency clients, key custody, and verifier deployment remain external.
- The private cohort-finalization manifest content-addresses one exact result, but the library does
  not globally prevent an organizer from building several internally valid finalizations for one
  completion. Production must register one canonical finalization identity append-only and publish
  a hash-bound redacted result while keeping labels and commitment keys private.
- A process-local or test registry cannot prove global attempt uniqueness. Production must provide
  one durable, concurrency-safe, append-only namespace across organizers and workers, publish alias
  decisions, and reject conflicting reservations before any execution starts.
- The library's stateful start-consumer boundary prevents replay through the official composition,
  but Python cannot stop an organizer that also retains direct credentials for the lower-level
  backend from bypassing that composition. In production, only the dedicated trusted launcher may
  hold backend-launch capability; its durable consumer transition must be transactionally coupled
  to that single launch, and the completion registry must reconcile every consumed-but-unterminated
  attempt before accepting another state transition.
- In v0, `verifier-policy.bin` is the committed selection/missingness/conflict policy. The official
  finalizer refuses to proceed without an affirmative trusted-verifier decision over the exact
  policy, universe, exhaustive audit, and per-case evidence. That injected verifier remains a trust
  root: hash equality and a boolean callback do not show that the policy is scientifically
  defensible or faithfully implemented, so both need independent review and audit.
- Tier B proof/completeness callbacks remain organizer-controlled trust roots. Releases commit the
  verifier/selection policy bytes, but production policy implementations and their external
  authority credentials still require independent audit.
- The contamination screen can detect and quarantine artifact leakage and positive recall signals;
  it cannot establish that a proprietary model did not train on later outcomes. Negative probes are
  `no_signal`, not proof. The organizer-controlled LLM audit verifier remains a trust root and needs
  human/SME calibration on the first real release.
- The current runner has no separate non-scoring model-recall response protocol. Full,
  bibliographically scrubbed, and no-evidence challenge views are implemented, but direct-model
  exposure probes still need a receipt-bound runner path.
- The current Docker backend is suitable for development, not hostile multi-tenant official runs.
  Production needs a fresh microVM or equivalently audited boundary and release-specific escape
  canaries, especially when GPU devices are exposed.

Every new reward component or data adapter should add an attack test before it changes the official
reward version.
