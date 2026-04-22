# ADR-0026: Study Layer and Sweep Orchestration Boundary

- **Status:** Accepted
- **Date:** 2026-04-20
- **Related ADRs:** ADR-0023, ADR-0024, ADR-0025

---

## Context

ADR-0025 established that session construction is the runtime execution
boundary and that richer cross-run reporting would be deferred to a
future study layer.

We have now implemented the initial structure of that study layer and
clarified how it fits into the project. The project required a clear,
explicit boundary for organizing and comparing multiple runs, especially
when performing hyperparameter sweeps, while avoiding reimplementation
of orchestration logic already provided by mature external tools.

This ADR documents the decisions that have been implemented to separate
sweep orchestration, runtime execution, and cross-run aggregation into
distinct responsibilities.

---

## Decision

### 1. Sweep orchestration is owned by Hydra and Optuna

We adopted Hydra and Optuna as our exclusive sweep orchestration mechanism. However, because the official Hydra Optuna sweeper plugin lacks official support for Optuna 4.0+, we do not use it. Instead, we implemented a custom adapter to bridge the two frameworks. 

This adapter is specifically designed to derive a trial-specific root configuration (structured via Hydra and Python dataclasses) from a dedicated Hydra config group. 

Despite this custom integration layer, the core sweep orchestration responsibilities are still strictly delegated to Hydra and Optuna. These include:
* Search-space definition
* Sampler and pruner selection
* Trial scheduling
* Study naming and storage
* Optimization over a scalar objective

While we maintain the config-derivation adapter to facilitate compatibility, we rely entirely on Optuna for the underlying optimization logic and do not implement or maintain a project-specific sweep engine.

---

### 2. The study layer is defined as a post hoc aggregation layer

We introduced a study layer that sits above individual runs and above
session execution.

The study layer does **not** own:
- session construction or execution
- model preparation or instantiation
- checkpoint loading
- trial scheduling
- parameter suggestion

Instead, the study layer consumes completed run artifacts and sweep
metadata and performs post hoc operations such as trial ranking and
summary generation.

---

### 3. The trial objective contract is intentionally minimal

Each sweep trial returns a **single scalar value** to Hydra/Optuna.

All richer outputs—such as evaluation artifacts, metadata, checkpoints,
logs, and previews—are persisted as normal run artifacts and are not
returned through the optimization interface.

This keeps the optimization boundary simple while allowing the study
layer to reconstruct a richer, multi-metric view after the fact.

---

### 4. Cross-run analysis is explicitly owned by the study layer

The study layer is responsible for:
- linking Optuna trials to project run artifacts
- aggregating completed runs under a study boundary
- ranking and selecting candidate runs
- materializing lightweight study-level summaries for downstream use

Detailed schemas, reporting formats, and visualization conventions were
intentionally deferred and are not part of this ADR.

---

## Consequences

### Positive

- Sweep orchestration reuses mature, well-supported tooling
- Runtime execution remains cleanly separated from cross-run analysis
- Study-level aggregation is a first-class concern without impacting
  runtime code
- Study-related evolution can proceed independently of session and
  training logic

### Trade-offs

- The project depends on an external sweep orchestration contract
- Trial-to-run identity mapping must be handled explicitly
- Study outputs remain intentionally lightweight in the initial
  implementation

---

## Follow-up Work

The following follow-up items are expected to be addressed incrementally
through future ADRs:

- maintain and evolve the mapping between Optuna trials and project run
  artifacts
  
- refine and possibly extend the scalar objective contract for
  sweepable pipelines

- expand the study layer in future ADRs to cover:
  - richer Optuna-related logic, such as custom samplers, pruners,
    callbacks, metadata capture, and study lifecycle policies
  - post hoc study analysis and reporting, including study-level schemas,
    multi-metric aggregation, comparisons, and downstream reporting or
    export formats

These expansions are explicitly out of scope for this ADR and will be
designed incrementally as the study layer matures.

---

## Summary

We have adopted Hydra and Optuna for sweep orchestration and implemented a
separate study layer for post hoc aggregation across completed runs.

This establishes three distinct responsibilities—sweep orchestration,
runtime execution, and cross-run analysis—while providing a clean and
stable foundation for future study-layer enrichment without prematurely
constraining its design.