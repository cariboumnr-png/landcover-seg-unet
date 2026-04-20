## Current workflow
```
[foundation/world_grids/builder]            (1 World Grid – pure construction)
|
+--> [foundation/world_grids/lifecycle]
|        (grid artifact persistence & validation)
|
+--> [foundation/domain_maps/mapper]        (2 Domain → Grid, optional)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (pure domain feature computation)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (domain artifact persistence)
|
+--> [foundation/data_blocks/mapper]        (3 Imagery/Labels → Grid windows)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (pure block construction)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (catalog & schema update)
|
+--> [geopipe/specification/factory]        (4 Build DataSpecs)
|
+--> [models/factory]                       (5 Model construction)
|
+--> [session/factory]                     (6 Session construction boundary)
|        |
|        +--> [session/components]          (components: loaders, heads, loss, optim)
|        |
|        +--> [session/state]               (runtime state initialization)
|        |
|        +--> [session/instrumentation]     (callbacks, exporters)
|        |
|        +--> [session/engine/batch]        (shared batch execution engine)
|        |
|        +--> [session/engine/policy]       (trainer / evaluator policies)
|        |
|        +--> [session/runner]              (optional: phases & runner)
|
|
+--> [cli/pipelines/*]                              (7 Pipeline execution)
         (explicit command-stage orchestration only;
          resolves config, selects a pipeline, and delegates
          construction/runtime ownership downstream)

    +--> [cli/pipelines/train_model]               (7a Train pipeline)
    |        (full experiment execution)
    |        |
    |        +--> validate / resolve required experiment artifacts
    |        |        (prepared dataset artifacts, manifests, schema)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (build DataSpecs from prepared artifacts)
    |        |
    |        +--> [models/factory]
    |        |        (construct training model)
    |        |
    |        +--> [session/factory]
    |        |        (assemble training session:
    |        |         components, state, callbacks, engines, runner)
    |        |
    |        +--> [session/runner]
    |                 (execute multi-phase train/validate lifecycle)
    |
    +--> [cli/pipelines/evaluate_model]            (7b Evaluate pipeline)
    |        (single evaluation execution)
    |        |
    |        +--> validate / resolve required experiment artifacts
    |        |        (prepared dataset artifacts, manifests, schema,
    |        |         evaluation inputs / model source)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (build evaluation DataSpecs)
    |        |
    |        +--> [models/factory]
    |        |        (construct evaluation model)
    |        |
    |        +--> [session/factory]
    |        |        (assemble evaluation session:
    |        |         components, state, callbacks, evaluator engine)
    |        |
    |        +--> [session/engine/policy]
    |                 (run single evaluation pass and emit metrics/exports)
    |
    +--> [cli/pipelines/train_overfit]             (7c Overfit pipeline)
             (end-to-end stack validation on one block)
             |
             +--> build / select one valid block
             |        (minimal block acquisition path for debug validation)
             |
             +--> build minimal DataSpecs
             |        (single-block / tiny-scope specification)
             |
             +--> [models/factory]
             |        (construct debug/overfit model)
             |
             +--> [session/factory]
             |        (assemble compact training session for overfit test)
             |
             +--> [session/engine/policy]
                      (train repeatedly on the same block until
                       near-perfect fit / expected debug target)
```
---

### Interpretation notes (updated)

- All foundation build steps (grid, domain, blocks) remain pure and deterministic.
- All reuse, overwrite, and validation logic flows through artifacts.Controller /
PayloadController.
- DataSpecs and Model construction occur before the session boundary and are
treated as inputs to session construction.
- Session construction is centralized in session/factory and owns:

  - component construction
  - runtime state initialization
  - callback binding
  - engine instantiation
  - optional runner assembly

- Engines are policy-only and receive fully initialized state and components.
- The CLI executes explicit pipeline stages and requests a session; it does
not assemble runtime internals.
- The workflow cleanly separates:

  - foundation artifacts (ingest & preparation)
  - experiment artifacts (specs, model)
  - training runtime (session-owned lifecycle)