# Reward firewall

The reward firewall is the pre-gradient QA and training-admission boundary for VaxReplay. It keeps
the published V0/V1 benchmark metrics intact for reporting while preventing a positive aggregate
score from automatically becoming an optimizer reward.

The boundary is deliberately two phase:

```text
model response
  -> strict parse
  -> primary + reference score
  -> formula and component vetoes
  -> immutable quarantine batch
  -> independently signed vector QA report
  -> Ed25519-signed, single-use admission
  -> exact local replay
  -> optimizer-visible reward
```

No QA detector score is added to the model reward. QA produces `admit`, `quarantine`, or `reject`,
and only `admit` can be signed for gradient use.

## Why reporting reward and training admission differ

V0 and V1 intentionally expose continuous metrics. Forecasting plus ranking carry 80 percent of
the aggregate reward, so a response with zero grounding can still have a positive score. The
uniform smoke-test policy is the concrete example: it receives useful baseline credit while having
zero grounding and zero assessment accuracy.

That is useful benchmark information, but it is not a scientifically acceptable training
trajectory. `RewardContract.component_floors` therefore acts as a non-collapsible veto. A ranking
or forecasting gain cannot compensate for a failed grounding floor.

Changing V0/V1 weights would silently change published results. The firewall instead preserves
those results and controls whether they are training eligible.

## Implemented controls

### Strict untrusted-response parsing

`vaxreplay.qa.parsing.parse_submission` enforces:

- explicit byte and character limits;
- strict UTF-8;
- exactly one JSON document;
- duplicate-key rejection at every object depth;
- rejection of `NaN`, infinities, trailing content, and malformed JSON;
- strict `Submission` schema validation.

The framework-neutral environment now uses this parser as well. It exposes only
`reporting_reward`; attempting to read `.reward` raises. It is a smoke-test/reporting surface, not
an optimizer API. Duplicate JSON members no longer use a parser-dependent "last value wins"
interpretation.

### Independent score checks

`vaxreplay.qa.score_integrity` independently recomputes every V0/V1 scalar relationship and checks
episode, manifest, label, and reward-version bindings. `differential_score` requires the primary
and reference scorer outputs to be byte identical. Each scorer receives a separately reparsed
submission, input mutation is rejected, and copied/constructed submissions and scores are
canonical-schema revalidated.

The reward contract binds distinct primary and reference scorer build hashes. Supplying two
instances of the same implementation is useful for development determinism tests, but does not
establish production independence.

### Metamorphic checks

`vaxreplay.qa.metamorphic` compares already-produced responses and scores for:

- candidate alias/permutation equivariance;
- nuisance invariance;
- expected decision and metric changes after evidence interventions.

Evidence-intervention sensitivity must name exact candidate ranking, forecast, or assessment
paths. An unrelated output perturbation therefore cannot satisfy a causal check. Citations are not
targetable, so removing or changing a citation alone cannot fake scientific sensitivity.

These utilities compare executions; they do not invoke a model or claim that a transformation is
scientifically valid. The QA authority must precommit the transformation and expected relation.

### Versioned attack catalog

`vaxreplay.qa.attack_catalog` defines the public attack taxonomy: parser differentials, scorer
integrity, temporal leakage, parametric memory, candidate shortcuts, evidence gaming, prompt
injection, component collapse, abstention gaming, evaluation awareness, and resource tampering.

Concrete hidden instances and seeds must remain outside the actor and training signal. The catalog
hash, required attack IDs, and QA policy hash are bound by `RewardContract`.

### Vector QA report

`RewardQAReport` embeds the exact reward contract and records:

- per-component values;
- typed findings and their artifact hashes;
- complete required-attack coverage;
- independent-scorer agreement;
- future-taint reachability;
- exact replay;
- tamper success;
- whether item-level private feedback was withheld.

Its disposition is derived by schema:

- a hard control failure or failed `reject` finding derives `reject`;
- a component-floor failure or other failed finding derives `quarantine`;
- only a complete passing vector derives `admit`.

An author cannot label a failing report `admit`.

Before admission, the complete report must also carry an Ed25519 attestation from an independently
pinned QA key. The gradient-admission authority verifies that signature, the exact catalog hash,
required-check coverage, and catalog-defined failure dispositions. The QA and gradient-admission
keys must differ. A launcher-authored report or random artifact digest without the independently
trusted QA service's signature is therefore not sufficient to mint a training grant.

### Signed, single-use gradient admission

`TrainingRunAdmission` binds the admitted report to:

- the exact trajectory batch and reward artifact;
- every episode manifest;
- the independently signed QA-report attestation and QA key identity;
- model, harness, tool policy, environment, and dataset hashes;
- optimizer configuration;
- reward contract and attack catalog;
- an explicit validity interval.

`GradientAdmissionToken` is Ed25519-signed. The optimizer side needs only the independently trusted
public key; it cannot mint its own admissions. Verification requires an injected time and an atomic
single-use token consumer. The in-memory consumer is development-only; production must use an
external transactional service or registry.

### Exact reward release

`TrainingRewardFirewall.quarantine` performs parsing, differential scoring, formula checks, and
component vetoes without returning optimizer reward. It retains the exact raw response, derived
prompt artifact, and opaque recorder trace as hash-bound bytes. The broker can inspect those bytes;
an external recorder format and signature still determine how much tool/resource provenance they
prove. `build_quarantined_batch` binds the ordered trajectory and reward inventories.

`release_training_batch` then:

1. canonical-schema revalidates the batch, submissions, scores, contract, and hashes;
2. reparses the retained raw responses and reconstructs their prompts from the committed bundles;
3. reruns independently identified primary and reference scorers from fresh inputs;
4. recomputes score formulas and component floors;
5. verifies every runtime binding and the trusted Ed25519 signature;
6. atomically consumes the token;
7. returns the replayed rewards.

Any failure occurs before token consumption and before reward release. A successful token cannot be
used twice.

## Minimal framework-neutral flow

```python
pending = firewall.quarantine(
    trajectory_id='batch-7-item-3',
    response=model_response,
    audit_trace=recorder_trace_bytes,
)
batch = build_quarantined_batch([pending])

# The independent QA service inspects this exact batch, executes its hidden
# controls, constructs an admitted RewardQAReport, and returns a signed token.
grant = qa_broker.admit(batch)

rewards = release_training_batch(
    batch,
    reward_contract=contract,
    admission=grant.admission,
    token=grant.token,
    trusted_public_key_bytes=qa_authority_public_key,
    now=trusted_clock(),
    consume_token=global_token_registry.consume,
    expected_model_sha256=model_sha256,
    expected_harness_sha256=harness_sha256,
    expected_tool_policy_sha256=tool_policy_sha256,
    expected_environment_sha256=environment_sha256,
    expected_dataset_sha256=dataset_sha256,
    expected_optimizer_config_sha256=optimizer_config_sha256,
    replay_scorers={
        episode_manifest_sha256: ReplayScorerPair(
            bundle=private_episode_bundle,
            primary_evaluator=primary_scorer,
            reference_evaluator=reference_scorer,
            primary_scorer_sha256=contract.scorer_sha256,
            reference_scorer_sha256=contract.reference_scorer_sha256,
        ),
    },
)
```

The caller must also retain the actual trajectory in an append-only provisional store. The
in-memory batch carries its response, prompt, trace, parsed submission, and score, but the library
does not provide a production object store or sign the recorder trace.

## Tinker boundary

The optional Tinker message adapter is disabled for training. `make_tinker_env` always raises
`UnadmittedTrainingDisabled`, even when a caller supplies a broker, trusted public key, token
consumer, trusted clock, and every runtime hash. There is no compatibility flag that restores the
old direct reward or the newer singleton-admission path.

This is required for two independent reasons:

1. `EnvFromMessageEnv` may create parse-error, length, and context-overflow rewards without calling
   `MessageEnv.step`. A firewall implemented inside `MessageEnv.step` therefore cannot mediate every
   gradient-bearing value.
2. `MessageEnv.step` runs before the correlated rollout group and optimizer batch are complete. A
   singleton admission cannot detect cross-trajectory reward hacking or bind the exact batch that
   will reach `forward_backward`.

A production Tinker integration needs a batch-aware rollout/training coordinator outside the
per-environment message adapter. It must:

1. collect and durably quarantine every trajectory and wrapper termination in the complete
   optimizer batch;
2. reject or exclude the whole batch when any trajectory has an unsigned parser, length, overflow,
   retry, or other wrapper-produced reward;
3. freeze the exact trajectory and reward inventory and obtain one signed QA admission for it;
4. verify and atomically consume that admission while binding it to one optimizer step; and
5. release the admitted reward vector immediately before, and in the same trusted failure domain
   as, the optimizer update.

The current `VaxReplayDatasetBuilder` and `tinker_train` entry point therefore fail closed. They are
not training launchers and must remain unusable until that coordinator is implemented and tested.

## Operational separation

Production should give separate identities and failure domains to:

- actor/harness;
- primary scorer;
- reference scorer;
- QA orchestrator and hidden-control store;
- admission signer;
- single-use token registry;
- trainer.

The actor receives episode inputs and a write-only response channel. It must not receive labels,
scorer code, audit seeds, token material, detailed private findings, network access, or the trusted
clock. The trainer receives only verified reward releases.

## What this does not prove

This boundary prevents an observed or elicited failing trajectory from being reinforced. It does
not prove aligned intent, detect every hidden objective, or prove that model weights do not contain
future outcomes.

Before production RL, the remaining deployment work is:

- implement the independent reference scorer in a separate build/failure domain;
- connect a production model runner to full/scrubbed/no-evidence and intervention probes;
- deploy an external QA signer and globally atomic token-consumption registry;
- pin runtime trust roots outside the training job and derive runtime/scorer hashes from measured
  attestations rather than launcher strings;
- define and sign the recorder-trace schema for tool, resource, clock, retry, and network events;
- retain append-only quarantine artifacts and incident handling;
- calibrate component floors and semantic checks with vaccine-domain reviewers;
- exercise deliberately hacky policies under blind audit;
- keep active prospective Tier A cohorts permanently outside training.

Prospective outcomes, hermetic execution, and strict authority separation remain the strongest
independent anchors.
