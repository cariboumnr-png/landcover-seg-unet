# ADR-0016: Refine `trainer/` Module Boundaries and Remove External Config Coupling

**Status:** Accepted
**Date:** 2026-04-09

---

## Context

We observed that the original `landseg.trainer` package exhibited the same boundary issue previously identified in `landseg.models`:

- `trainer.factory` and `engine.engine_config` imported and depended directly on `landseg.configs`.
- Trainer components (losses, metrics, optimization, callbacks) were logically well-separated, but their assembly was tightly coupled to Hydra configuration objects.

This coupling made it difficult to reuse the trainer outside the existing experiment framework and obscured ownership of configuration semantics between the trainer and the application layer.

---

## Decision

We refactored the trainer architecture so that it mirrors the design principles used in `geopipe`:

- Removed direct dependencies on `landseg.configs` from all trainer internals.
- Moved trainer-owned runtime configuration concerns (schedule, precision, monitoring, optimization) behind explicit trainer-side interfaces rather than Hydra-specific dataclasses.
- Redesigned trainer factories so that they:
  - accept explicit, typed configuration objects,
  - do not import or reference Hydra or root configuration trees.

We constrained the trainer’s responsibilities to:

- Orchestration of training, validation, and inference loops
- Coordination of trainer components (losses, metrics, optimizers, callbacks)
- Runtime state management and logging

The CLI / experiment layer remains responsible for instantiating trainer configuration objects from Hydra and passing them into the trainer at assembly time.

---

## Expected Consequences

### Positive

- Cleaner separation of concerns between experiment specification and training mechanics.
- Easier reuse of the trainer in non-Hydra contexts (e.g., programmatic experiments, testing).
- Improved consistency with the architectural boundaries used in `models` and `geopipe`.

---

## Clarification: Session vs Trainer Scope

As part of the implementation, we organized the refactored trainer architecture under a new `landseg.session` namespace.

In this context, **session** represents a concrete execution runtime for training, validation, and inference, composed of:

- `engine.trainer`: step- and epoch-level execution logic
- `components`: assembly of data, task, optimization, and callback components
- `runner`: curriculum and phase orchestration across training stages

This naming reflects runtime composition rather than model-specific training logic, while remaining trainer-centric in responsibility.

---

## Negative Consequences

- Moderate refactor effort was required to migrate existing configuration plumbing.
- Temporary churn occurred in factory signatures and downstream call sites during the transition.

---

## Summary

Overall, this change significantly simplified the mental model of the training stack and improved the system’s extensibility for future work (e.g., alternative trainers, distributed backends, or non-CLI execution contexts).