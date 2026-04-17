# ADR-0023: Public Session Construction API

- **Status:** Proposed
- **Date:** 2026-04-16
- **Related ADRs:** ADR-0021, ADR-0022

---

## Context

ADR-0021 established the session as the primary construction and ownership
boundary for runtime execution. ADR-0022 clarified that sessions may serve
different **intents** (training, overfit, evaluation-only), without introducing
distinct session types or classes.

At present, session construction is exposed through multiple factory functions
(e.g. engine construction, optional runner construction), which are correct
internally but ambiguous as a *public* API. Callers must understand which
functions to combine and in what order.

This ADR defines a **public session construction surface** that is stable,
intent-aware, and minimal, while explicitly deferring orchestration unification
and higher-level experiment control.

---

## Decision

### Define a single public session construction entrypoint

The system will expose **one public session construction API** intended for use
by CLI pipelines and external callers:

    build_session(...)

This function represents the **official session boundary**. All other session
factory helpers are considered internal implementation details.

The public API is responsible for:
- constructing session-owned components
- initializing runtime state
- assembling execution engines
- optionally assembling a runner
- returning a fully constructed, ready-to-use session object

---

### Scope of the public API

The public session construction API:

- **Accepts**
  - prepared `DataSpecs`
  - a constructed model
  - session configuration (components, runtime, phases)
  - execution context (device, logger, paths)

- **Returns**
  - a session object encapsulating:
    - trainer
    - evaluator
    - optional runner
    - runtime metadata

- **Does not**
  - perform artifact ingestion or data preparation
  - perform model selection or hyperparameter search
  - manage multiple runs or compare results across sessions

---

### Session intent handling

Session intent (as defined in ADR-0022) influences *what is assembled*, not the
existence of multiple APIs.

Examples:
- **Training intent** → trainer + evaluator + runner
- **Overfit intent** → trainer + evaluator, runner optional or omitted
- **Evaluation-only intent** → evaluator only (deferred)

Intent may be expressed via configuration or pipeline selection, but does not
require separate construction functions.

---

### What remains internal

The following remain **non-public** and may change without notice:

- engine-level builders (trainer/evaluator instantiation)
- callback wiring and instrumentation details
- runtime state initialization helpers
- intermediate factory functions used by `build_session`

This preserves refactoring freedom inside the session module.

---

## Consequences

### Positive
- Establishes a clear, minimal public API boundary
- Reduces cognitive load for CLI and external callers
- Prevents session internals from leaking into pipelines
- Creates a stable foundation for future orchestration layers

### Costs
- Requires light refactoring to funnel construction through the public API
- Some flexibility is deferred in favor of clarity

These costs are accepted to stabilize the session boundary before introducing
higher-level abstractions.

---

## What this ADR does *not* do

This ADR intentionally does **not**:

- define an experiment or study abstraction
- formalize evaluation-only sessions
- unify all execution workflows
- guarantee long-term API stability beyond the session boundary

Those concerns are explicitly deferred.

---

## Follow-up Work (Deferred)

The following items are expected to be addressed by future ADRs:

- Clarify and formalize **evaluation-only sessions** and reporting outputs
- Introduce an **experiment / study layer** for hyperparameter sweeps,
  model comparison, and final model selection
- Define how session results are aggregated and compared
- Further simplify internal session factory structure once orchestration
  patterns converge

---

## Summary

This ADR defines a **single, explicit public API** for session construction,
solidifying the session as a stable execution boundary while preserving
flexibility for future orchestration and optimization layers.