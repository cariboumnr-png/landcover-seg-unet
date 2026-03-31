# ADR-0013: Consolidate Training Components into a First-Class `trainer` Module

- **Status:** Accepted 
- **Date:** 2026-03-31

## Context

The training stack was previously distributed across several closely related
modules:

- `trainer_components`
- `trainer_engine`
- `trainer_runner`

Although each module has a clear local responsibility, they collectively
implemented a single conceptual concern: **model training orchestration and
execution**. Their separation was largely historical and increased
navigation overhead, import indirection, and cognitive load when extending or
reviewing training behavior.

Following the successful consolidation of concrete data and domain primitives
into `geopipe.core` (ADR‑0012), the training stack was the primary area where
related functionality remains fragmented.

## Decision

We have consolidated `trainer_components`, `trainer_engine`, and
`trainer_runner` into a single **first‑class `trainer` module** with a cohesive
namespace and clearly scoped internal submodules.

The implemented structure is as below:

```
geopipe/
models/
trainer/
  common/ # local protocols and alias
  components/
  engine/
  runner/
  factory.py   # module API
```

This change was intentionally **structural only**:

- No training logic was modified
- Execution order and semantics unchanged
- Configuration behavior remains intact

## Rationale

- Establishes the trainer as a first‑class architectural concept, on par with
  `model`, `foundation`, and `transform`
- Improves discoverability and readability of the training code
- Reduces cross‑module coupling and simplifies imports
- Creates a clear extension surface for future training features
  (e.g., schedulers, evaluators, exporters)

## Consequences

- Import paths referencing training components have been updated
- External references to old module paths were properly adjusted
- No compatibility or artifact changes were encountered at runtime

## Scope Notes

- This ADR explicitly excluded refactoring trainer logic, APIs, or behavior
- Any functional changes to training workflows will be captured in a subsequent ADR

## Status Notes

The refactoring is complete. The training stack now follows the recent repository architectural styles.
