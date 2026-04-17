
# ADR-0023: Public Session Construction API

- **Status:** Accepted
- **Date:** 2026-04-18
- **Related ADRs:** ADR-0021, ADR-0022

---

## Context

ADR-0021 established the session as the primary construction and ownership
boundary for runtime execution. ADR-0022 clarified that sessions may serve
different **intents** (training, overfit, evaluation-only), without introducing
distinct session types or classes.

Prior to this ADR, session construction was exposed through multiple factory
functions (e.g. engine builders, runner builders). While correct internally,
this made the *public* usage ambiguous: callers needed to understand which
pieces to assemble and in what order.

This ADR formalizes a **single public session construction API** and constrains
all runtime assembly behind that boundary.

---

## Decision

### A single public session construction entrypoint

The system exposes **one public API** for constructing a session:

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

## Public API Surface

### Inputs

The public session construction API accepts:

  - prepared `DataSpecs`
  - a constructed model
  - session configuration (components, runtime, phases)
  - execution context (device, logger, paths)

### Outputs

The API returns a **typed session result object** encapsulating:

  - trainer
  - evaluator
  - optional orchestration (e.g. training runner)
  - session-level metadata

This result object is the only supported integration surface for callers.

---

## Session Intent and Orchestration

Session intent (as defined in ADR-0022) influences *what is assembled*, not the
shape of the public API.

Examples:

- **Training intent** → trainer + evaluator + training orchestration
- **Overfit intent** → trainer + evaluator, orchestration omitted
- **Evaluation-only intent** → evaluator only (deferred)

Orchestration is optional and strategy-specific. It is not the definition of a
session and is not required for all session intents.

---

## Explicit Non-Goals

The public session construction API does **not**:

- perform artifact ingestion or data preparation
- perform model selection or hyperparameter search
- manage multiple runs or compare results across sessions
- define experiment-level or optimization workflows

These concerns are intentionally outside the session boundary.

---

## Internal vs Public Guarantees

Only the following are considered **public and stable**:

- the `build_session(...)` entrypoint
- the returned session result object and its fields

The following remain **internal** and may change without notice:

- engine-level builders
- orchestration internals
- callback wiring and instrumentation
- runtime state initialization helpers
- intermediate factory functions

This preserves refactoring freedom within the session module.

---

## Consequences

### Positive

- Establishes a clear, minimal public API boundary
- Makes orchestration explicitly optional
- Removes ambiguity in CLI and pipeline usage
- Aligns code structure with session-first architecture

### Costs

- Requires funneling all construction through the public API
- Defers some flexibility in favor of clarity

These costs are accepted to stabilize the session boundary before introducing
higher-level orchestration or optimization layers.

---

## Follow-up Work (Deferred)

The following items are expected to be addressed by future ADRs:

- Clarify and formalize **evaluation-only sessions** and reporting outputs
- Introduce an **experiment / study layer** for hyperparameter sweeps,
  model comparison, and final model selection
- Define a session-level result reporting contract for downstream consumers
- Further simplify internal session construction once orchestration patterns
  converge

---

## Summary

This ADR establishes a **single, explicit public API** for session construction
and enforces the session as the authoritative runtime boundary. With the public
surface implemented and typed, future orchestration and optimization layers can
be introduced without destabilizing core execution semantics.