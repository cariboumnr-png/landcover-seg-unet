# ADR-0031: Formalize Base Callback Interface and Event Dispatcher

**Status:** Proposed
**Date:** 2026-05-03

## Context
Following ADR-0030, our callbacks are now passive, stateless observers. However, the orchestrator still needs
a standardized, decoupled way to communicate with these observers. If the orchestrator directly instantiates
and calls specific callbacks, it creates a hard dependency. 

To support dynamically swapping or stacking callbacks (e.g., adding a `DashboardCallback` only during sweeps,
or a `ProgressCallback` for local runs), we need a centralized subscription mechanism. 

## Decision
We will introduce a strict Publisher-Subscriber pattern within `src/landseg/session/instrumentation/`:

1.  **`BaseCallback` Interface:** Define an abstract base class outlining the allowed lifecycle hooks (e.g.,
`on_run_start`, `on_epoch_end`, `on_batch_end`). All future callbacks must inherit from this and implement
only the hooks they need.
2.  **`CallbackDispatcher` (or `CallbackList`):** Create a manager class that holds a list of registered
`BaseCallback` objects. 
3.  **Orchestrator Integration:** The orchestrator will only interact with the `CallbackDispatcher`. At the
end of an epoch, the orchestrator will simply call `dispatcher.on_epoch_end(epoch_summary)`. The dispatcher
iterates through all registered callbacks and forwards the payload.

## Consequences
* **Definition of Done:** The orchestrator no longer imports specific callbacks. It accepts a list of callbacks 
at initialization, registers them with the dispatcher, and fires standard hooks.
* **Extensibility:** We can now inject an arbitrary number of observers into a session run without touching
the core engine or orchestrator code.