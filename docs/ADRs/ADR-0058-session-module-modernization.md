# ADR-0058: Session Module Modernization

**Status:** Accepted<br>
**Date:** 2026-09-20

---

## Context

`landseg.session` grew around a `common` package and a broad `engine.runtime`
package. Builders frequently reached back through enclosing package exports to
access sibling implementations. This made import order fragile, exposed
implementation details as public API, and obscured the boundaries between data
loading, batch execution, epoch execution, optimization, and task assembly.

The previous flat session configuration reflected this history: dataloader,
engine execution, optimizer, task, and scheduling settings were sibling
sections, even though scheduling belongs to the engine rather than to
orchestration.

## Decision

Modernize `landseg.session` around direct module dependencies, explicit
contracts, and canonical builders.

### Contracts and package surface

- Replace `session.common` with `session.contracts` for shared data aliases,
  observer protocol, and phase protocol. Move the session logger to
  `session.logger`.
- Keep package `__init__.py` modules as small lazy public surfaces. Internal
  modules import their direct siblings or the defining module; they do not
  import an enclosing package merely to retrieve a sibling symbol.
- Expose `build_session_runner` as the top-level session construction API.
  Its `session_type` selects `overfit`, `evaluate`, `continuous`, or
  `curriculum` construction while retaining the corresponding pipeline
  behavior.

### Data loading

- Rename the dataloader builder module to `session.data.builder` and retain
  `build_dataloaders` as the data-layer entry point.
- Split batching concerns into `data.collate`, dataset representation into
  `data.dataset`, and optional preloading/cache policy into `data.memory`.
- Rename the Hydra group from `data_loader` to `dataloader` to match the
  schema and public terminology.

### Engine composition

Organize the engine by execution level:

```text
session.engine/
|-- builder.py       Compose a complete epoch runner
|-- batch/           Per-batch state, objective, and batch engine
|-- epoch/           Epoch runner plus train/evaluation policies
|-- optim/           Optimizer and scheduler construction
`-- tasks/           Heads, losses, metrics, constraints, and regularization
```

- Replace `engine.runtime` with these first-class `batch`, `epoch`, `optim`,
  and `tasks` packages. `build_engine` assembles a batch engine, optimization,
  task components, and an epoch runner.
- Refactor `epoch.builder.build_epoch_runner` as the dedicated constructor
  for trainer/evaluator policies and the epoch runner.
- Flatten task implementation paths where they represent one concern:
  `tasks.heads`, `tasks.metrics.diagnostics`,
  `tasks.metrics.segmentation`, and `tasks.constraints.multihead`.
  `tasks.builder.build_engine_tasks` remains the task-composition facade.

### Orchestration and instrumentation

- Keep `orchestration.builder.build_runner` as the canonical constructor for
  continuous and curriculum runners. The runner classes remain available for
  typing and specialized consumers.
- Move callback construction to `instrumentation.builder`; callback,
  dashboard, and formatter implementations remain in focused subpackages.

### Configuration ownership

Nest the execution sections under `session.engine`:

```text
session:
  dataloader: ...
  engine:
    engine_exec: ...
    engine_optim: ...
    engine_schedule: ...
    engine_tasks: ...
  orchestration: ...
```

`engine_schedule` owns validation, inference, checkpoint, and loss-update
frequencies. `orchestration` owns monitoring, curriculum phase selection, and
resume policy.

## Consequences

### Positive

- Dependency direction and ownership are visible in both module paths and
  configuration composition.
- Builders have narrow, testable responsibilities and receive explicit
  contexts instead of recovering dependencies through package exports.
- Data caching, collation, batch execution, epoch policy, and orchestration
  can evolve independently.
- The public import surface is smaller and lazy imports no longer need to
  support internal implementation coupling.

### Costs

- Internal import paths and configuration paths changed. Callers using
  non-public paths must migrate to the new locations.
- The refactor spans construction code and requires focused session,
  configuration, and lazy-import tests whenever those boundaries change.

## Verification

Run the relevant configuration and session unit tests, including lazy-import
coverage, dataloader construction, batch engine, epoch builder, task builder,
orchestration, and session factory tests.
