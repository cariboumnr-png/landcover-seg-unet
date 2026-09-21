# ADR-0058: Modernization of Session Module — Scoped Imports and Canonical Entry Points

**Status:** Proposed  
**Date:** 2026-09-20  

---

## 1. Context

Following the architectural modernization of the geospatial ETL domain in
`landseg.geopipe` (PR #72 / ADR-0057), dataset preparation was successfully
decoupled into explicit semantic stages with in-memory contexts and clean
import scopes. Specifically, ADR-0057 eliminated circular self-imports where
entry points imported their own enclosing package lazy exports, establishing
single canonical entry points (e.g., `view.py` in `geopipe.prepare.dataset`).

While `landseg.session` serves a distinct purpose with a different lifecycle
than `geopipe` (real-time model execution, multi-task optimization, and phase
orchestration rather than offline ETL), an architectural audit of the `session`
module identified significant structural debt in two core areas:

1. **Leaky Import Scopes (Enclosing Package Self-Imports)**:
   Within multiple subpackages in `landseg.session`, scripts that serve as
   subsystem entry points or builders import their own enclosing package (or
   ancestor package) via lazy `__init__.py` exports rather than importing
   sibling modules directly.
   - In `landseg.session.data.loader`, the entry point `build_dataloaders`
     imports `landseg.session.data as data` to consume `data.BlockDatasetContext`
     and `data.MultiBlockDataset` from sibling `dataset.py`.
   - In `landseg.session.instrumentation.callbacks.builder`, `build_dispatcher`
     imports `...callbacks as callbacks` to access `callbacks.BaseCallback`,
     `callbacks.LoggingCallback`, and `callbacks.CallbackDispatcher` from
     sibling modules.
   - In `landseg.session.engine.runtime.tasks.loss.builder`, `build_headlosses`
     imports `...loss as loss` to consume `loss.CompositeLoss` and
     `loss.CompositeLossConfig` from sibling `composite.py`.
   - In `landseg.session.engine.runtime.executor.executor`, `BatchEngine`
     imports `...executor as executor` to consume
     `executor.TrainingObjectives`, `executor.EngineState`, and
     `executor.multihead_objective` from sibling scripts.
   - In `landseg.session.orchestration.runner` and
     `landseg.session.engine.epoch.policy`, concrete implementations
     (`ContinuousRunner`, `CurriculumRunner`, `MultiHeadTrainer`,
     `MultiHeadEvaluator`) import their own package namespace (`...runner as
     runner`, `...policy as policy`) to subclass base classes
     (`runner.BaseRunner`, `policy.EngineBase`).

   This anti-pattern forces `__init__.py` to re-export internal implementation
   classes, polluting public namespaces and creating fragile, circular
   lazy-resolution chains.

2. **Ambiguous / Competing Entry Points and Builder Fragmentation**:
   - In `landseg.session.orchestration`, the package exposes both a unified
     factory function (`build_runner` in `builder.py`) and concrete runner
     classes (`ContinuousRunner`, `CurriculumRunner` in `runner/`), resulting
     in dual competing entry points for runner assembly.
   - In `landseg.session.engine.runtime.tasks`, task building is fragmented
     across five disparate sub-builders (`loss/builder.py`,
     `metrics/segmentation/builder.py`, `heads/specs.py`,
     `constraints/constraints.py`, and `regularization/consistency.py`), lacking
     a single canonical task compilation facade.
   - In `landseg.session.engine.epoch.policy`, `trainer.py` and `evaluator.py`
     exist as un-unified parallel execution entry points without a cohesive
     policy factory.
   - In `landseg.session.common`, cross-cutting types and protocols are grouped
     without clear domain ownership, leading modules like `events.py` to import
     `common` to resolve `PhaseLike` defined in `orchestration.py`.

---

## 2. Decision & Goals

We will modernize `landseg.session` by restructuring its import scopes and
establishing canonical subsystem entry points.

### Primary Constraints
- **Zero Behavioral Change**: The refactoring will not alter any runtime
  behavior, mathematical loss/metric calculations, training dynamics, event
  emissions, checkpointing, or configuration schemas.
- **Strict Backward Compatibility**: External entry points invoked by
  `landseg.execution.pipelines` (`session.build_overfit_session`,
  `session.build_evaluate_session`, `session.build_continous_training_session`,
  `session.build_curriculum_training_session`) will retain their existing
  signatures and contracts.

### Modernization Rules

#### Rule 1: Strict Direct Sibling Imports (Eliminate Self/Ancestor Imports)
- Internal scripts will import sibling modules directly using explicit module
  imports (e.g., `import landseg.session.data.dataset as dataset`) rather than
  importing the enclosing package `__init__.py`.
- Concrete subclasses will inherit from base classes via direct module imports
  (e.g., `from landseg.session.orchestration.runner.base import BaseRunner` or
  `import landseg.session.orchestration.runner.base as runner_base`).
- Child packages will not import ancestor `__init__.py` packages to access
  parent base classes (e.g., `callbacks/tracking/*.py` will import directly
  from `callbacks.base`).
- Package `__init__.py` files will only lazily expose symbols that constitute
  the public contract for external consumers outside that package. Internal
  helper dataclasses, contexts, and sibling utilities will remain encapsulated
  within their respective modules.

#### Rule 2: Single Canonical Entry Point per Subsystem
- **`landseg.session.orchestration`**: `build_runner` will serve as the single
  public entry point for building execution runners. Concrete runner
  implementations (`ContinuousRunner`, `CurriculumRunner`) will remain internal
  to `session.orchestration.runner`, while public callers will interact
  through runner interfaces and configurations.
- **`landseg.session.data`**: `build_dataloaders` will serve as the single
  canonical entry point for constructing session dataloaders. Dataset classes
  (`MultiBlockDataset`, `BlockDatasetContext`) will be internal to data loading
  orchestration.
- **`landseg.session.instrumentation.callbacks`**: `build_dispatcher` will be
  the sole entry point for constructing callback dispatchers, encapsulating
  concrete logging and tracking callbacks.
- **`landseg.session.engine.runtime.tasks`**: `build_engine_tasks` in
  `factory.py` will serve as the unified facade coordinating per-head
  specifications, composite losses, confusion matrices, constraints, and
  regularizers.
- **`landseg.session.engine.runtime`**: `build_engine_runtime` will serve as
  the runtime coordination entry point, cleanly wiring batch execution,
  optimization, and task subsystems.
- **`landseg.session.engine`**: `build_epoch_engine` will remain the canonical
  entry point for epoch engine construction, orchestrating runtime and policy
  execution.

---

## 3. Proposed Implementation Structure

### 3.1. `session.data`
- `loader.py`:
  - Will replace `import landseg.session.data as data` with
    `import landseg.session.data.dataset as dataset`.
  - Will instantiate `dataset.BlockDatasetContext` and
    `dataset.MultiBlockDataset` directly.
- `data/__init__.py`:
  - Will expose: `build_dataloaders`, `DataLoaders`, and `DataLoaderConfig`.
  - Will prune internal dataset classes from the public package surface.

### 3.2. `session.instrumentation`
- `callbacks/builder.py`:
  - Will replace `import ...callbacks as callbacks` with direct imports:
    - `import landseg.session.instrumentation.callbacks.base as base`
    - `import landseg.session.instrumentation.callbacks.dispatcher as dispatcher`
    - `import landseg.session.instrumentation.callbacks.logging as logging`
- `callbacks/dispatcher.py` & `callbacks/logging.py`:
  - Will import `base.BaseCallback` directly from
    `landseg.session.instrumentation.callbacks.base`.
- `callbacks/tracking/*.py`:
  - Will import `base.BaseCallback` directly from
    `landseg.session.instrumentation.callbacks.base`.
- `dashboards/ml_flow.py` & `dashboards/tensor_board.py`:
  - Will import `BaseTracker` directly from
    `landseg.session.instrumentation.dashboards.base`.

### 3.3. `session.engine`
- `runtime/tasks/loss/builder.py`:
  - Will replace `import ...loss as loss` with
    `import landseg.session.engine.runtime.tasks.loss.composite as composite`.
- `runtime/tasks/loss/primitives/*.py`:
  - Will import `PrimitiveLoss` directly from
    `landseg.session.engine.runtime.tasks.loss.primitives.base`.
- `runtime/tasks/metrics/`:
  - Will flatten nested `diagnostics/` and `segmentation/` subdirectories into
    two cohesive sibling modules:
    - `diagnostics.py`: houses `MTLMetricsAggregator` (GEM and constraint
      violation metrics).
    - `segmentation.py`: consolidates `ConfusionMatrix`, `HeadMetrics`, and
      `build_headmetrics`.
  - `metrics/__init__.py` will lazily delegate directly to `.diagnostics` and
    `.segmentation`, eliminating two package directory nesting levels.
- `runtime/tasks/heads`:
  - Will flatten `heads/specs.py` and `heads/__init__.py` into a single flat
    module `runtime/tasks/heads.py`, exporting `HeadSpec`, `HeadSpecs`, and
    `build_headspecs`.
- `runtime/tasks/constraints`:
  - Will rename `constraints.py` to `multihead.py` and update exported names
    to `CompiledMTLConstraint` and `compile_mtl_constraints` for explicit
    domain clarity.
- `runtime/optim/builder.py`:
  - Will replace `import ...optim as optim` with
    `import landseg.session.engine.runtime.optim.optimization as optimization`.
- `runtime/executor/executor.py`:
  - Will replace `import ...executor as executor` with direct imports:
    - `import landseg.session.engine.runtime.executor.objective as objective`
    - `import landseg.session.engine.runtime.executor.state as state`
- `epoch/policy/trainer.py` & `epoch/policy/evaluator.py`:
  - Will import `EngineBase` directly from
    `landseg.session.engine.epoch.policy.base`.

### 3.4. `session.orchestration`
- `runner/continuous.py` & `runner/curriculum.py`:
  - Will import `BaseRunner` directly from
    `landseg.session.orchestration.runner.base`.
- `policy/phase.py`:
  - Will import `EpochPolicy` directly from
    `landseg.session.orchestration.policy.epoch`.
- `orchestration/builder.py`:
  - Will act as the single entry point `build_runner`.
- `orchestration/__init__.py`:
  - Will expose `build_runner`, `BaseRunnerConfig`, and `TrackingConfig`,
    encapsulating direct runner classes.

### 3.5. `session.common`
- `common/events.py`:
  - Will import `PhaseLike` directly from
    `landseg.session.common.orchestration` (or via `typing.TYPE_CHECKING`)
    instead of importing `landseg.session.common as common`.

---

## 4. Consequences & Benefits

### Positive
- **Clean Dependency Graph**: Eliminates circular and self-import loops
  during module initialization.
- **Accurate Discoverability**: Subpackage `__init__.py` files will cleanly
  define public contracts via `__all__`, preventing internal implementation
  leakage.
- **Maintainable Architecture**: Clear separation between builders/entry
  points and internal execution components.
- **Consistent Codebase Standards**: Aligns `session` with the architectural
  conventions established in `geopipe` and `.agents/AGENTS.md`.

### Negative / Risks
- **Refactoring Surface**: Touching import statements across multiple
  `session` submodules requires careful verification of unit test suites
  (`tests/unit/session/` and `tests/unit/test_lazy_imports.py`).

---

## 5. Verification Plan

1. **Lazy Import Integrity**:
   - Run `tests/unit/test_lazy_imports.py` to ensure all public lazy imports
     across `landseg.session` and its subpackages resolve deterministically.
2. **Session Unit Tests**:
   - Execute all targeted unit tests for `session` components:
     - `tests/unit/session/` (data loaders, runners, policies, optimizers,
       tasks, callbacks).
3. **Pylint & Formatting Compliance**:
   - Verify 0 circular import warnings and 100% compliance with
     `.agents/AGENTS.md` (no `pd`/`np` aliases, line length $\le 80$,
     comment length $\le 72$).
