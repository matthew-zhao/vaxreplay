# VaxReplay

> [!IMPORTANT]
> VaxReplay `v0.1.0-alpha.1` is a technical preview of infrastructure for temporally controlled
> biomedical evaluation. It contains fictional conformance fixtures and no admitted real-data
> benchmark. Its outputs do not establish vaccine-development capability and must not be used for
> clinical or R&D decisions.

VaxReplay is an early research toolkit for building historical and prospective evaluations in
which a model or agent receives only the evidence allowed at a declared decision cutoff. The
toolkit separates task provenance, model-facing workspaces, private outcome labels, deterministic
scoring, and execution evidence.

## Included in this preview

- typed task, evidence, forecast, citation, and score contracts;
- deterministic evaluators and aggregation;
- historical-source adapter interfaces;
- contamination-control and temporal-admission protocols;
- development-only sealed-runner and provider-gateway components;
- unit tests and a deterministic smoke test; and
- wholly fictional antigen and candidate-ranking fixtures.

## Not included

- real AACT/ClinicalTrials.gov, IEDB, ImmPort, VaxSeer, or FluSelect evaluation data;
- the internal 131-task development cohort or its identities;
- private real-data gold, organizer mappings, or source slices;
- an admitted Tier A or Tier B benchmark;
- a production-qualified execution service; or
- a scientifically valid model leaderboard.

The available fictional fixtures test software compatibility and scoring mechanics. They do not
measure biological knowledge or vaccine-development ability.

## Quick start

VaxReplay requires Python 3.12 and uses `uv` for the locked development environment.

```bash
uv sync --locked --extra dev
uv run python scripts/smoke_test.py
uv run pytest
```

## Evaluation model

VaxReplay distinguishes three provenance tiers:

- **Tier A:** prospectively sealed before outcomes exist;
- **Tier B:** independently archived retrospective evidence; and
- **Tier C:** retrospective development or synthetic material.

Only a separately admitted Tier A release could support an official headline result. Running a
Tier B or Tier C task inside a sealed worker does not upgrade its provenance.

The architecture is designed around one retained evaluation attempt, while allowing multiple
bounded model and workspace-tool calls within that attempt. A fixed-harness model comparison and a
harness-plus-model comparison are reported separately.

## Documentation

- [Technical-preview scope](docs/alpha_scope.md)
- [Architecture](docs/architecture.md)
- [Agentic replay](docs/agentic_replay_v1.md)
- [Temporal admission](docs/temporal_admission.md)
- [Threat model](docs/threat_model.md)
- [Sealed runner](docs/sealed_runner.md)
- [Reward firewall](docs/reward_firewall.md)

## Licensing and data

Original software and project documentation are licensed under Apache License 2.0. The explicitly
listed project-authored fictional fixtures are licensed under CC BY 4.0; see `DATA_LICENSES.md` for
the exact path boundary and attribution. Neither license covers any undistributed real cohort,
private gold, organizer mapping, third-party source material, or model submission.

## Status

APIs and artifact formats may change before beta. No alpha result should be represented as clinical
validation, regulatory evidence, or proof that a model can safely develop a vaccine.
