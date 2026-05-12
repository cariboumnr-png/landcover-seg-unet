## Current workflow

Last updated : 2026-05-12

```
[foundation/world_grids/builder]            (1 World Grid – pure construction)
|
+--> [artifacts/controller]
|        (resolve/build/reuse grid artifact under policy)
|
+--> [foundation/world_grids/lifecycle]
|        (grid artifact persistence & validation)
|
+--> [foundation/domain_maps/mapper]        (2 Domain → Grid alignment, optional)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (pure domain feature computation)
|        |
|        +--> [artifacts/controller]
|        |        (resolve/build/reuse domain artifact)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (domain artifact persistence & validation)
|
+--> [foundation/data_blocks/mapper]        (3 Imagery/Labels → Grid window mapping)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (pure block construction)
|        |
|        +--> [artifacts/controller]
|        |        (resolve/build/reuse block artifacts)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (catalog, schema registration, and indexing)
|
+--> [geopipe/specification/factory]        (4 Build DataSpecs from prepared artifacts)
|
+--> [models/factory]                       (5 Model construction & wiring)
|
+--> [session/factory]                      (6 Session construction boundary)
|        |
|        +--> [session/data]                (dataloaders and batching adapters)
|        |
|        +--> [session/engine]
|        |        (batch + epoch execution engines)
|        |
|        +--> [session/instrumentation]
|        |        (callbacks, logging, tracking, preview/export)
|        |
|        +--> [session/orchestration]
|        |        (lifecycle management and phase coordination)
|        |
|        +--> [session/metadata]
|                 (session tracking and runtime context)
|
|
+--> [execution/executor]                  (7 Execution entry point)
         (resolves config, selects pipeline, delegates execution)

    +--> [execution/pipelines/train]       (7a Train pipeline)
    |        (full experiment lifecycle)
    |        |
    |        +--> resolve required artifacts via [artifacts/controller]
    |        |        (grid, domain, blocks, manifests, schema)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (build DataSpecs from resolved artifacts)
    |        |
    |        +--> [models/factory]
    |        |        (construct training model)
    |        |
    |        +--> [session/factory]
    |        |        (assemble training session)
    |        |
    |        +--> [session/orchestration]
    |                 (execute single/multi-phase train/validate lifecycle)
    |
    +--> [execution/pipelines/evaluate]    (7b Evaluate pipeline)
    |        (single-pass evaluation)
    |        |
    |        +--> resolve required artifacts via [artifacts/controller]
    |        |        (dataset artifacts, manifests, model source)
    |        |
    |        +--> [geopipe/specification/factory]
    |        |        (build evaluation DataSpecs)
    |        |
    |        +--> [models/factory]
    |        |        (construct evaluation model)
    |        |
    |        +--> [session/factory]
    |        |        (assemble evaluation session)
    |        |
    |        +--> [session/engine]
    |                 (run evaluation pass and emit metrics/exports)
    |
    +--> [execution/pipelines/overfit]     (7c Overfit pipeline)
             (end-to-end stack validation on minimal scope)
             |
             +--> resolve/build minimal block via [artifacts/controller]
             |        (debug-scale artifact acquisition)
             |
             +--> build minimal DataSpecs
             |        (single-block / constrained specification)
             |
             +--> [models/factory]
             |        (construct debug/overfit model)
             |
             +--> [session/factory]
             |        (assemble compact training session)
             |
             +--> [session/engine]
                      (repeat training on fixed data until convergence)
```
---

### Interpretation notes (updated)

- All foundation build steps (grid, domain, blocks) remain pure,
  deterministic, and side-effect free.

- All artifact reuse, rebuild, overwrite, and validation decisions are
  centralized through artifacts.controller, enforcing policy-driven
  lifecycle management.

- Foundation builders do not handle persistence; they produce in-memory
  outputs that are materialized exclusively via the artifacts layer.

- All downstream stages operate on resolved artifacts, not on
  recomputed or implicit intermediates.

- DataSpecs and model construction occur strictly before the session
  boundary and are treated as fully-resolved, immutable inputs.

- Session construction is centralized in session/factory and owns:

  - data interface construction (dataloaders, samplers)
  - component assembly (model bindings, losses, optimizers)
  - runtime state initialization
  - callback and instrumentation binding
  - execution engine instantiation
  - lifecycle orchestration setup

- Session internals are fully configured prior to execution; no
  structural mutation occurs during runtime.

- Execution engines operate on injected state and components and do
  not encode configuration or lifecycle decisions.

- Lifecycle control (train/validate phases and transitions) is handled
  by session orchestration, not by pipelines.

- The execution layer (execution.executor + pipelines) is responsible
  only for:

  - resolving configuration
  - selecting the pipeline
  - coordinating artifact resolution via artifacts.controller
  - delegating construction to factories (specs, model, session)

- Pipelines do not construct runtime internals and act strictly as
  thin orchestration shells.

- The workflow enforces clear separation across layers:

  - foundation layer
    deterministic data construction (grid, domain, blocks)

  - artifacts layer
    persistence, validation, versioning, and reuse policy

  - experiment layer
    dataset specification (DataSpecs) and model construction

  - session layer
    runtime system, execution, and lifecycle orchestration

  - execution layer
    top-level orchestration and pipeline selection

- The system maintains a unidirectional dependency flow:

  foundation → artifacts → experiment → session → execution

- This structure ensures:

  - reproducibility via explicit artifact control
  - strict separation of build-time vs runtime concerns
  - composable and predictable pipelines
  - deterministic reconstruction from configs + artifacts