# ADR-0030: Decouple State Management to Enable Passive Callbacks

**Status:** Accepted
**Date:** 2026-05-02


## Context
In the current architecture, the callback system
(`src/landseg/session/instrumentation/callbacks/`) holds load-bearing
responsibilities for the engine's state lifecycle. Specifically, phase callbacks
(train, val, infer) are responsible for resetting batch contexts and outputting
the engine state.

This violates the principle of maximum decoupling. A true observer pattern
dictates that the core engine should be completely self-sufficient. If all
callbacks are stripped from the execution pipeline, the mathematical training
loop and phase progressions should still execute flawlessly without raising
state-related or memory errors. Because callbacks currently mutate or reset the
engine state, they act as active controllers rather than passive observers,
making it fragile to add or remove logging infrastructure.

As we prepare to introduce a robust, dashboard-driven tracking system (e.g.,
TensorBoard, MLflow), we need a guarantee that adding visualization observers
will not interfere with the core segmentation training logic.

## Decision
We have refactored the engine and callback systems to enforce a strictly passive
observer pattern:

1.  **Engine Owns State Lifecycle:** All `reset()`, `clear()`, and state
  initialization operations have been removed from the callback layer and now
  embedded directly into the core engine and orchestrator routines
  (`src/landseg/session/engine/` and `src/landseg/session/orchestration/`). The
  engine now explicitly manages its own memory space and metric accumulation at
  the start and end of every batch/phase.

2.  **Stateless Callbacks:** Existing phase callbacks have been stripped of
  state-mutating logic and now are practically empty to host future passive
  subscribers.

3.  **Read-Only Engine State:** In the following implementations, callbacks
  will treat the engine state and aggregated metrics as read-only payloads.
  They may maintain their own internal state, e.g., the current progress counter,
  but they will never modify the engine's internal dataclasses.

## Consequences
* **Resilience:** The training loop becomes completely immune to failures in the
  visualization or logging layers.
* **Simplified Extensibility:** This clears the path to easily implement a
  `DashboardingCallback` suite that can be dynamically attached or detached
  depending on whether the execution environment is a local development sweep
  or a production run.

## Notes
* The preview image generation utility is current inactive, while `infer()`
  remains functional to aggregate batch-level inference results and update to
  engine state. This part may be of use for future development of the dashboard
  system.