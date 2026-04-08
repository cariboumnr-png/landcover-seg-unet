# ADR-0016: Refine `trainer/` Module Boundaries and Remove External Config Coupling

- **Status:** Proposed
- **Date:** 2026-04-07

## Context
We observed that the `landseg.trainer` package exhibits the same boundary issue identified in `landseg.models`:

- `trainer.factory` and `engine.engine_config` import and depend directly on `landseg.configs`.
- Trainer components (losses, metrics, optimization, callbacks) are logically well-separated, but their assembly is tightly bound to Hydra configuration objects.

This coupling makes it difficult to reuse the trainer outside the current experiment framework, and obscures the ownership of configuration semantics between the trainer and the application layer.

## Decision
We plan to refactor `landseg.trainer` so that it mirrors the design principles used in `geopipe`:

1. Remove direct dependencies on `landseg.configs` from trainer internals.
2. Move trainer-side configuration dataclasses (schedule, precision, monitoring, optimization) into the `trainer` module.
3. Redesign trainer factories so that they:
   - accept explicit, typed configuration objects,
   - do not import or reference Hydra or root configuration trees.
4. Constrain the trainer’s responsibilities to:
   - orchestration of training/validation/inference loops,
   - coordination of components (losses, metrics, optimizers, callbacks),
   - runtime state management and logging.

The CLI / experiment layer will remain responsible for instantiating trainer configs from Hydra and passing them into the trainer factory.

## Expected Consequences
### Positive
- Cleaner separation of concerns between experiment specification and training mechanics.
- Easier reuse of the trainer in non-Hydra contexts (e.g., programmatic experiments, testing).
- Improved consistency with `models` and `geopipe` boundaries.

### Negative
- Moderate refactor effort to migrate existing config plumbing.
- Temporary churn in factory signatures and downstream call sites.

We expect this change to significantly simplify the mental model of the training stack and make future extensions (e.g., alternative trainers, distributed backends) easier to implement.
