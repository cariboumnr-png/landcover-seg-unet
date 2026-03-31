# ADR-0013: Consolidate Training Components into a First-Class `trainer` Module

- **Status:** Proposed
- **Date:** 2026-03-30

## Context

The training stack is currently distributed across several closely related
modules:

- `trainer_components`
- `trainer_engine`
- `trainer_runner`

Although each module has a clear local responsibility, they collectively
implement a single conceptual concern: **model training orchestration and
execution**. Their current separation is largely historical and increases
navigation overhead, import indirection, and cognitive load when extending or
reviewing training behavior.

Following the successful consolidation of concrete data and domain primitives
into `geopipe.core` (ADR‑0012), the training stack is now the primary area where
related functionality remains fragmented.

## Decision

We will consolidate `trainer_components`, `trainer_engine`, and
`trainer_runner` into a single **first‑class `trainer` module** with a cohesive
namespace and clearly scoped internal submodules.

The target structure will resemble:

```
geopipe/
  model/
    trainer/
      __init__.py
      components.py
      engine.py
      runner.py
      stages.py   # if applicable
```

This change is intentionally **structural only**:

- No training logic will be modified
- Execution order and semantics will remain unchanged
- Configuration behavior will remain intact

## Rationale

- Establishes the trainer as a first‑class architectural concept, on par with
  `model`, `foundation`, and `transform`
- Improves discoverability and readability of the training code
- Reduces cross‑module coupling and simplifies imports
- Creates a clear extension surface for future training features
  (e.g., schedulers, evaluators, exporters)

## Consequences

- Import paths referencing training components will change and must be updated
- Any external references to old module paths will need adjustment
- No compatibility or artifact changes are expected at runtime

## Scope Notes

- This ADR explicitly excludes refactoring trainer logic, APIs, or behavior
- Any functional changes to training workflows should be captured in a
  subsequent ADR

## Status Notes

This ADR proposes a follow‑up architectural cleanup aligned with recent
repository refactors. Implementation is expected to be mechanical and low risk.
