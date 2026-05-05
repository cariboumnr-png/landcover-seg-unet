# ADR-0031: Formalize Base Callback Interface and Event Dispatcher

**Status:** Accepted
**Date:** 2026-05-05


## Context

Following ADR-0030, callbacks were made passive, stateless observers. However,
prior to this ADR, orchestration and engine layers still relied on concrete callback
implementations or instrumentation-specific construction logic. This created
unnecessary coupling and made it difficult to swap, layer, or entirely replace
instrumentation concerns (e.g., console output, dashboards, telemetry) without
touching core execution code.

The original intent of this ADR was to introduce a standardized, decoupled way
for orchestration and execution components to emit lifecycle events without knowing
who is consuming them.

---

## Decision (Implemented)

We have implemented a strict Publisher–Subscriber pattern for session lifecycle
events, centered around a formal observer interface and a dispatcher abstraction.

### What we have done

- **Formalized a callback interface**
  - Introduced `SessionObserverLike` (a runtime-checkable protocol) that defines
  all supported lifecycle hooks (epoch, policy, batch, etc.).
  - Defined `BaseCallback` as a concrete convenience base class that implements
  this interface and can be subclassed by all callback implementations.

- **Introduced a CallbackDispatcher**
  - Implemented `CallbackDispatcher` as a concrete publisher that:
    - Holds a list of registered `BaseCallback` instances.
    - Implements `SessionObserverLike` itself.
    - Broadcasts all lifecycle hook calls to its registered callbacks.

- **Decoupled orchestration and execution from instrumentation**
  - The orchestrator, runners, engines, and policies depend **only** on
  `SessionObserverLike`.
  - They no longer import, reference, or construct any concrete callbacks.
  - Instrumentation remains fully contained within `landseg.session.instrumentation`.

- **Shifted callback composition to the composition root**
  - Instead of the orchestrator accepting and registering raw callbacks, a fully
  constructed `SessionObserverLike` (typically a `CallbackDispatcher` preloaded
  with callbacks) is injected.
  - This construction occurs in session factories or higher-level wiring code,
  not in orchestration logic.

---

## Outcome and Design Shift

### Original intended outcome
The original ADR wording assumed that:
- The orchestrator would accept a list of callbacks.
- The orchestrator would be responsible for registering them with a dispatcher.

### Actual implemented outcome
We intentionally refined this design:

- The orchestrator now accepts **only an observer abstraction**, not callbacks.
- Callback registration and dispatcher construction are handled upstream.
- The orchestrator simply emits events against the `SessionObserverLike` interface.

### Rationale for the shift

This shift improves the architecture beyond the original ADR intent:

- **Stronger decoupling**
  - Orchestration code is completely unaware of instrumentation concerns.
  - Instrumentation can be removed, replaced, or extended without touching
  orchestration.

- **Clear separation of responsibilities**
  - Session factories act as the composition root.
  - Orchestrators focus solely on control flow and lifecycle progression.
  - Dispatchers focus solely on event fan-out.

- **Future extensibility**
  - Non-callback observers (e.g., tracing adapters, remote telemetry, testing
  probes) can be injected without changing orchestrator APIs.
  - The observer abstraction is no longer tied conceptually to “callbacks” alone.

This design still fully satisfies — and in practice exceeds — the original goals
of dynamic composition, loose coupling, and extensibility described in this ADR.

---

## Consequences

- **Definition of Done (Revised)**
  - The orchestrator no longer imports or constructs specific callbacks.
  - It interacts only with a `SessionObserverLike`.
  - Callback dispatching and registration are performed externally via a
  dispatcher injected at initialization.
  - All lifecycle events are fired through standardized hooks.

- **Extensibility**
  - Arbitrary observers can be stacked or swapped by changing only session wiring
  code.
  - Instrumentation is fully optional and orthogonal to execution.

- **Architectural clarity**
  - The observer/dispatcher boundary is explicit and enforceable via typing.
  - Instrumentation is no longer a first-class dependency of orchestration.
