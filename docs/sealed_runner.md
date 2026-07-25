# VaxReplay sealed-runner contract

## What V0 controls

The runner controls the information channels outside an executable system. It does not claim to
erase information encoded in model weights, code, or image assets.

This runtime is one boundary inside the larger Tier A lifecycle. Before outcomes exist, episode
receipts must seal the complete universe/panel, cutoff-visible evidence, and decision. Before
submissions open, the exact outcome-free cohort release must pass semantic campaign/readiness
approval and receive its strict v0.2 external seal. The system/attempt is reserved before opening;
at or after opening and strictly before the deadline, an independent authority must authorize the
typed start; after verification, its stateful consumer must atomically accept that exact start once
immediately before this runner begins backend work. The first terminal success or
failure is retained and registered. Private outcomes are joined only later by the exhaustive
cohort finalizer. A retrospective Tier B/Tier C dataset stays Tier B/Tier C even when this runner
executes it perfectly; isolation cannot manufacture prospective provenance.

For each episode, a fresh worker can access only:

1. its digest-pinned OCI image, containing the declared harness and model;
2. one canonical challenge envelope delivered over standard input; and
3. bounded private scratch and shared memory supplied by the isolation backend.

The worker receives no episode-directory mount, repository mount, private labels, HMAC keys,
scoring code, network access, provider credentials, or previous-episode state. A submitted image
can still contain a memorized or deliberately embedded database. This is why fixed-model harness
results, contamination probes, and prospective cohorts remain separate controls.

The exact model snapshot, harness commit, image digest, prompt compiler, and execution policy must
also be committed before outcomes are revealed to entrants. If that timing is impossible, the
outcomes and any re-identifiable source records must remain privately embargoed until the committed
system completes its run. Network denial cannot prevent a post-reveal image from embedding labels.

The current challenge format implements the Core track: the envelope contains the exact system and
user messages produced by the existing VaxReplay prompt compiler. The
[Agentic Replay protocol](agentic_replay_v1.md) now has a versioned Lane A contract, signed guest bootstrap,
frozen-workspace/local-tool boundary, trusted inference gateway, canonical development operator,
and Firecracker qualification collector. That path remains development-only until its exact guest
image, host, provider snapshot, cleanup/reconciliation adapters, and Linux/KVM execution pass the
production qualification and deployment review. The runner must never expose raw source
directories as an accidental shortcut.

## Trust boundary

```text
private episode builder
        |
        | renders only public, pre-cutoff messages
        v
hash-bound challenge bundle  ---> externally preregistered challenge hash
        |
        | one envelope over stdin; no host mounts
        v
fresh model+harness worker
        |
        | one Submission JSON on stdout; bounded logs on stderr
        v
run artifact: canonical responses + private bounded audit streams + HMAC receipt
        |
        | worker is destroyed before this boundary
        v
private score-run process: labels + HMAC keys + deterministic suite scorer
        |
        v
delayed aggregate leaderboard result
```

`run_challenge_bundle` never loads labels or imports scoring. `score-run` never launches contestant
code. The run artifact contains the canonical response rows and bounded raw stdout/stderr under
`audit/`; audit contents remain organizer-private and only their hashes and sizes appear in the
receipt.

Every run receipt is authenticated with an organizer-held HMAC-SHA256 key that is never passed to
the worker. Both execution and scoring require the preregistered key ID, so a contestant cannot
replace the receipt and merely recompute public hashes. Development and official releases must use
different keys, and keys should rotate between releases. HMAC provides organizer authentication,
not public verification; a hosted service should also publish receipt hashes to an append-only log
or add an asymmetric organizer signature when public third-party verification is required.

## Isolation tiers

The default runner policy requires `official` isolation and fails closed.

- `official`: a dedicated ephemeral microVM or equivalently audited hostile-code sandbox, with no
  persistent host-home mount and independently enforced network denial. The repository defines the
  backend protocol and capability checks, but does not pretend that such infrastructure exists on
  every laptop.
- `development`: the implemented Docker OCI backend. It uses no host mounts, no network, a
  read-only root, a non-root user, dropped capabilities, no-new-privileges, private IPC/cgroups,
  bounded tmpfs/shared memory, CPU/memory/PID/file limits, disabled health checks and daemon logs,
  one fresh container per episode, and bounded host-side capture. Docker images must be local,
  pinned by SHA-256, and cannot declare `VOLUME` or `CMD`.

The Docker backend always records `sealed=false`. It refuses an `official` policy before querying
or launching the image. Private scoring likewise refuses a development receipt unless the operator
passes the explicit local-testing flag.

Containers are a useful development boundary, but Docker Desktop, a persistent Docker VM, a
container runtime vulnerability, and GPU device passthrough are outside the V0 guarantee. Official
evaluation should run the same backend protocol on fresh, dedicated infrastructure and gate it on
an escape-canary suite.

## Worker protocol

The image manifest must be immutable:

```json
{
  "schema_version": "vaxreplay.system-submission.v0.1",
  "submission_id": "example-system-v1",
  "image_ref": "registry.example/vax/system@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "entrypoint": ["/opt/vaxreplay/run"],
  "model_id": "example-model-snapshot",
  "harness_id": "example-harness-commit",
  "response_protocol": "vaxreplay.submission-json-stdout.v0.1"
}
```

The complete entrypoint is declared as an argv array and is never executed through a shell. The
worker reads exactly one `vaxreplay.challenge-envelope.v0.1` JSON object from stdin. Its `messages`
field contains one system message followed by one user message. It emits exactly one existing
VaxReplay `Submission` JSON object on stdout. Pretty or multiline JSON is accepted and
canonicalized by the trusted runner. Any extra prose, malformed schema, non-UTF-8 output, nonzero
exit, timeout, or output-limit failure becomes exactly `null\n` in the ordered response file and
therefore receives the existing missing-response penalty.

Stderr is log-only. It is bounded independently from the response and cannot affect scientific
scoring. Internal model/tool traces are observable only if the harness writes them there; an
arbitrary submitted system cannot honestly self-attest its hidden model-call or token usage.

## Operator workflow

The commands below exercise the lower-level challenge interface. Literature-replay pilots should
use the release-aware workflow in [Literature Replay V0](literature_replay.md),
which binds split and temporal admission into the challenge and private package. Generic
`make-challenge` leaves that admission optional.

Build a challenge from private source episode directories. Only rendered messages leave this step:

```bash
vaxreplay make-challenge \
  --challenge-id preclinical-pilot-001 \
  --suite-id preclinical-pilot \
  --episode-dir /private/episodes/episode-a \
  --episode-dir /private/episodes/episode-b \
  --output-dir /sealed/public-challenge

vaxreplay verify-challenge \
  --challenge-dir /sealed/public-challenge \
  --expected-challenge-sha256 PREREGISTERED_CHALLENGE_HASH
```

For a local Docker test, use a policy whose `required_isolation` is explicitly `development`:

```bash
vaxreplay-runner receipt-key-id --receipt-key /organizer/secrets/development-receipt-key.hex

vaxreplay-runner run-oci \
  --challenge-dir /sealed/public-challenge \
  --expected-challenge-sha256 PREREGISTERED_CHALLENGE_HASH \
  --system-manifest /submissions/system.json \
  --policy /organizer/development-policy.json \
  --receipt-key /organizer/secrets/development-receipt-key.hex \
  --expected-receipt-key-id PREREGISTERED_DEVELOPMENT_KEY_ID \
  --runtime /absolute/path/to/docker \
  --output-dir /sealed/run-artifact
```

Then move the run artifact across the one-way handoff and score it in the private evaluator. Local
development receipts require an explicit opt-in:

```bash
vaxreplay score-run \
  --challenge-dir /sealed/public-challenge \
  --expected-challenge-sha256 PREREGISTERED_CHALLENGE_HASH \
  --run-dir /sealed/run-artifact \
  --system-manifest /submissions/system.json \
  --policy /organizer/development-policy.json \
  --receipt-key /organizer/secrets/development-receipt-key.hex \
  --expected-receipt-key-id PREREGISTERED_DEVELOPMENT_KEY_ID \
  --episode-dir /private/episodes/episode-a \
  --episode-dir /private/episodes/episode-b \
  --allow-development-run
```

This generic CLI invocation is not the official Tier A composition. Official use must enter through
the approved prospective lifecycle: load the independently pinned v0.2 release seal, pre-opening
reservation, and post-opening typed start authorization; verify them before backend work; retain
the authorization, then atomically consume that exact attempt/start once immediately before backend
work; externally register the first terminal result; and later score only that completion through
the private cohort finalizer. The service must additionally enforce delayed aggregate feedback, minimum
cohort size, receipt signing/transparency, and submission quotas; these remain deployment controls
rather than properties of a single process invocation.

## Remaining production gates

- Deploy and independently qualify the implemented production-shaped ephemeral-microVM backend on
  the exact pinned Linux/KVM/jailer/guest-image stack.
- Run network, filesystem, process-tree, resource-exhaustion, state-reset, GPU, and secret-sentinel
  escape canaries before each release.
- Add asymmetric receipt signatures or an organizer-controlled transparency log for public
  verification beyond the implemented symmetric organizer HMAC.
- Add organizer-controlled structured model/tool tracing for the fixed-model harness track.
- Deploy and qualify the implemented [Agentic Replay](agentic_replay_v1.md) frozen workspace, mediated
  inference gateway, call tracing, and provider boundary without mounting private source
  directories or exposing provider credentials to participant code.
- Keep temporal provenance admission upstream: sealing a contaminated prompt only makes the
  contamination reproducible.
