# ADR-0026: Study Layer and Sweep Orchestration Boundary

- **Status:** Accepted
- **Date:** 2026-04-20
- **Related ADRs:** ADR-0023, ADR-0024, ADR-0025

---

## Context

ADR-0025 established that session construction is the runtime execution
boundary and that richer cross-run reporting is deferred to a future
study layer.

We have now decided how that future study layer fits into the project.

The project needs a higher-level structure for organizing and comparing
multiple runs, especially when performing hyperparameter sweeps. At the
same time, we do not want to reimplement sweep orchestration logic that
is already provided by Hydra and Optuna.

This ADR establishes two boundaries:

- Hydra + Optuna own sweep orchestration
- the study layer owns post hoc aggregation across completed runs

---

## Decision

### 1. We adopted Hydra + Optuna as the sweep orchestration layer

Hyperparameter sweeping is handled through Hydra multirun with the
Optuna sweeper plugin.

This includes:

- search-space definition
- sampler/pruner selection
- trial scheduling
- study naming and storage
- optimization over a scalar objective

We do not implement a project-specific sweep engine.

---

### 2. We defined the study layer as a post hoc aggregation layer

The study layer sits above individual runs and above session execution.

It does not own:

- session construction
- model preparation
- checkpoint loading
- trial scheduling
- parameter suggestion

Instead, it consumes completed run artifacts and sweep metadata and
materializes study-level summaries, rankings, and derived outputs.

---

### 3. We kept the trial objective contract minimal

Each sweep trial returns a scalar optimization target to Hydra/Optuna.

All richer outputs remain persisted as normal run artifacts, including
evaluation artifacts, metadata, checkpoints, and previews when present.

This keeps the optimization boundary simple while allowing the study
layer to reconstruct a richer multi-metric view after the fact.

---

### 4. We established the study layer as the owner of cross-run analysis

The study layer is responsible for:

- linking trials to project run artifacts
- aggregating completed runs under a study boundary
- ranking and selecting candidate runs
- producing study-level summaries for downstream consumption

Detailed schemas and reporting conventions remain open and will be
defined later as the study layer matures.

---

## Consequences

### Positive

- we reuse mature sweep orchestration instead of reimplementing it
- runtime execution remains cleanly separated from cross-run analysis
- study-level aggregation becomes a first-class concern
- richer reporting can evolve independently of session/runtime code

### Trade-offs

- the project now depends on an external sweep orchestration contract
- trial/run identity mapping must be handled explicitly
- study outputs will initially remain lightweight until a later schema
  and reporting ADR is introduced

---

## Follow-up Work

- implement a first study catalog that maps Optuna trials to project run
  artifacts
- define the initial scalar objective contract for sweepable pipelines
- add study-level ranking and best-candidate selection
- draft a later ADR for study artifact schemas and reporting formats

---

## Summary

We have adopted Hydra + Optuna for sweep orchestration and introduced a
separate study layer for post hoc aggregation across runs.

This keeps sweep execution, runtime execution, and cross-run analysis as
three distinct responsibilities and provides a clean foundation for
future study-level reporting.