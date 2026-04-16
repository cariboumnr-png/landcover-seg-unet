# ADR-0021: Session Construction, Component Ownership, and Runtime-State Boundaries

- **Status:** Accepted (Implemented with Deferred Extensions)
- **Date:** 2026-04-15
- **Related ADRs:** ADR-0019, ADR-0020

## Context

Following ADR-0019 (Shared Execution Core) and ADR-0020 (Dedicated Evaluator Engine),
core execution mechanics and policy-level orchestration have been separated.
Trainer and evaluator engines now act as thin policy layers over a shared batch
execution core, with unified runtime state and component contracts.

This refactor surfaced architectural pressure at the *session construction* layer.
Historically, the CLI pipelines were responsible for assembling sessions end-to-end,
including component construction, runtime-state initialization, callback wiring,
and engine instantiation. This resulted in implicit ordering constraints, tight
coupling to configuration shape, and limited testability.

The work in this branch addresses those issues by introducing an explicit
*session-owned construction boundary*, while intentionally deferring higher-level
session orchestration unification (e.g., normal training vs. overfit workflows).

---

## Decision

### Session-first construction boundary

The system adopts a **session-first architecture** where:

- The session module owns construction and assembly of runtime elements
- The CLI is reduced to a thin entrypoint that *requests* a session
- Configuration objects serve as *inputs* to session construction, not as drivers
  of runtime wiring

Session assembly responsibilities now include:

- Component construction (data, task, optimization)
- Runtime state initialization
- Callback instantiation and binding
- Batch execution engine construction
- Trainer and evaluator instantiation
- (Optionally) runner construction

All of the above occur behind a session-level factory boundary.

### Explicit ownership and lifecycle boundaries

This ADR establishes the following ownership rules:

- **Components** are constructed once and owned by the session
- **Runtime state** is initialized exactly once per session from components
- **Callbacks** are bound to runtime state and configuration during session
  construction
- **Execution and policy engines** receive fully-initialized state and do not
  participate in assembly

This removes construction order dependencies and makes lifecycle transitions
explicit and auditable.

### Configuration direction

Configuration modeling has been refactored to align with session-level concerns:

- Legacy trainer- and runner-centric configuration shapes are removed
- A unified `SessionConfig` expresses components, runtime, and phases
- Configuration no longer dictates builder order; it is consumed by the session
  factory as input

This reverses the historical dependency direction where runtime architecture was
forced to conform to configuration layout.

---

## Deferred Scope: Session Variants and Orchestration

This ADR intentionally **does not unify all training workflows under a single
session execution model**.

Specifically:

- The system currently supports multiple pipelines (e.g., full end-to-end
  curriculum training and overfit-style training)
- These pipelines have different orchestration needs, and not all require a
  session runner
- Introducing a single, configurable *session execution mode* abstraction would
  require a clearer roadmap of supported session types

As a result:

- The session factory exposes composable construction steps (e.g., engine-only vs.
  engine+runner)
- A single `build_session(...)` convenience API is intentionally deferred
- This ADR focuses on **correct construction boundaries and ownership**, not on
  exhaustively modeling all session variants

This limitation is accepted as a conscious scoping decision, not an architectural
inconsistency.

---

## Consequences

### Positive

- CLI pipelines are thin and declarative
- Session construction is centralized and testable
- Component, state, and callback lifecycles are explicit
- Engines remain policy-only and free of wiring concerns
- Configuration reflects runtime needs rather than build order

### Costs

- Some duplication remains between session construction paths
- A unified session execution abstraction is deferred
- Follow-up design work will be required to generalize session variants

These costs are accepted to avoid premature abstraction and to allow informed
future design.

---

## Non-Goals

This ADR does **not**:

- Redesign batch execution or policy engines
- Mandate a single session execution model
- Introduce new training or evaluation behaviors
- Define a public evaluation CLI

---

## Follow-up Work (Deferred)

- Define a taxonomy of supported session types (e.g., curriculum, overfit,
  evaluation-only)
- Introduce a unified `build_session(...)` API once session variants are
  well-understood
- Further simplify session factory surface area after orchestration convergence

---

## Related Code Areas

- `session/`
- `session/components/`
- `session/state/`
- `session/engine/`
- `session/runner/`
- `session/instrumentation/`
- `cli/pipelines/`
- `configs/`