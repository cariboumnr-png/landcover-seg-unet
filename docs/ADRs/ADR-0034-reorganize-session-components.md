# ADR-0034: Reorganize and Scope Session Components

**Status:** Accepted
**Date:** 2026-05-07

---

## Context

The `landseg.session.components` module historically acted as a catch-all
for all elements required to execute a session, combining three distinct
architectural concerns:

1. task (heads, losses, metrics)
2. data (datasets, dataloaders)
3. optim (optimizers, schedulers)

As the execution engine, runtime, and orchestration layers matured,
this structure no longer reflected the true boundaries of the system.
The coupling of unrelated concerns obscured intent, complicated
dependency management, and conflicted with the decoupled design
principles established in earlier ADRs.

In particular, optimization logic combined two concerns with different
lifecycles:

- Optimizers are persistent, stateful objects tied to model parameters.
- Schedulers are temporal policies that must adapt to training phases
  and may require reconfiguration as training progresses.

---

## Decision

We have reorganized the session architecture to align with functional
boundaries and runtime responsibilities.

### 1. Promote Dataloading (`data`)

We have moved dataset and dataloader construction out of the
`components` namespace into a first-class module: `landseg.session.data`

This module now serves as the adapter layer between geospatial data
representations and session execution. Dataloader construction is
performed explicitly during session/engine building rather than through
a monolithic component factory.

---

### 2. Isolate Task Components (`tasks`)

We have formalized "session components" as strictly the definition of
the neural network task:

- head specifications
- loss functions
- metric computations

These are now located under: `landseg.session.engine.runtime.tasks`

Task construction is handled through a dedicated factory that produces
a cohesive `EngineTasks` bundle consumed by the execution engine.

This clarifies that components are no longer a generic concept, but are
specifically tied to model behavior within the execution engine.

---

### 3. Relocate Optimization to Runtime (`optimization`)

We have moved optimization logic out of the previous components module
into the engine runtime layer: `landseg.session.engine.runtime.optim`

Optimization is now treated as a runtime concern alongside the execution
engine and task definitions.

---

### 4. Decouple Optimizer and Scheduler Lifecycles

We have decoupled optimizer and scheduler responsibilities while
preserving a practical and flexible construction model.

- The optimizer is instantiated once and persists across all phases.
- The scheduler is no longer treated as tightly bound to optimizer
  lifecycle and can be dynamically controlled.

An `Optimization` wrapper provides explicit APIs:

- `step_optimizer()`
- `step_scheduler()`
- `reconfigure(...)`
- `reset_scheduler(...)`
- `rebuild_scheduler()`

This allows the orchestration layer (e.g. curriculum training) to
adapt scheduler behavior at phase boundaries.

Schedulers may be:

- instantiated during initial construction, and
- reconfigured or rebuilt dynamically at runtime by phase policies.

This approach ensures that scheduler behavior remains aligned with
phase-specific requirements (e.g. step counts, durations) without
over-constraining construction responsibilities.

---

### 5. Introduce Engine Runtime Composition

We have introduced a runtime composition layer to assemble all execution
concerns into a coherent structure:

```
EngineRuntime = {
  engine execution core,
  task definitions,
  optimization bundle
}
```

This replaces the previous "components" container and provides a clearer,
more maintainable abstraction aligned with execution semantics.

---

### 6. Remove the `components` Abstraction

We have fully removed the `session.components` concept.

Configuration, construction, and runtime access are now explicitly
scoped to:

- `loader` (data)
- `tasks` (model behavior)
- `optimization` (runtime optimization)
- `runtime` (execution policies)

This eliminates ambiguity and aligns configuration structure with
execution responsibilities.

---

## Consequences

### ✅ Architectural Clarity

The system now cleanly separates:

- data ingestion (`session.data`)
- task definition (`engine.runtime.tasks`)
- execution and optimization (`engine.runtime`)

The meaning of each module is explicit and consistent.

---

### ✅ Runtime Flexibility

Decoupling scheduler lifecycle allows training orchestration to:

- adapt learning rate schedules across phases
- avoid stale scheduling state
- support curriculum and multi-phase training strategies

without requiring rigid factory ownership or complex re-instantiation
pipelines.

---

### ✅ Improved Extensibility

The introduction of `EngineRuntime` provides a stable composition layer
that can be extended with additional runtime concerns (e.g. advanced
instrumentation, distributed execution, adaptive policies).

---

### ✅ Reduced Conceptual Overhead

Removing the generic "components" abstraction simplifies reasoning
about the system and aligns naming with actual responsibilities.

---

### ⚠️ Trade-off: Flexible Scheduler Ownership

Scheduler instantiation is not strictly tied to a single layer
(e.g. exclusively phase runners). Instead, it may be initialized
during construction and later reconfigured dynamically.

This introduces some flexibility in ownership but significantly reduces
complexity and improves usability while preserving the required
lifecycle decoupling.

---

## Summary

We have transitioned from a monolithic "components" model to a
runtime-oriented architecture where:

- data, tasks, and optimization are clearly separated,
- execution is driven by a composed runtime layer, and
- optimization policies are dynamically adaptable to training phases.

This better reflects the evolved system design and provides a stable
foundation for future development.
