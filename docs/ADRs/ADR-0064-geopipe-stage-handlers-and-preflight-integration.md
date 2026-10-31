# ADR-0064: Geopipe Stage Validators and Catalog-Driven Preflight Integration

**Status:** Accepted — Implemented<br>
**Date:** 2026-10-27 (Implemented: 2026-10-31)

---

## 1. Context

### 1.1. Foundation Laid by Validator Primitives

The introduction of `landseg.utils.validator` established a
zero-dependency module of primitive validation functions
(`require_types`, `require_numeric_range`, `require_value_from`,
`require_mapping_str_v`, `require_file`, `require_string`,
`require_sequence`, `require_numeric_sequence`, etc.) that raise a
canonical `ValidationError`. These primitives were adopted across
`landseg.configs.schema` and `geopipe.harmonize.manifest.normalizer` /
`compiler`, replacing ad-hoc `isinstance` checks, bare `ValueError`
raises, and inline guard clauses with a uniform validation vocabulary.

Subsequent refactoring simplified primitive signatures by removing
superfluous parameter name arguments (`name`, `value_name`, `file_name`),
relying on concise, standardized error messages and type-annotated
overloads. In addition, `BaseConfigSection` aliases `val = validator`
directly, allowing configuration dataclasses to perform schema
validations cleanly without boilerplate wrapper staticmethods.

Key foundations established on the branch:
- `ManifestEntry` and `AllowedCategory` colocated in `normalizer.py`;
  redundant `schema.py` eliminated.
- Functional `normalize_manifest_entry()` and
  `compile_dataset_manifest()` guard root JSON structure and entries
  with validator primitives while rejecting empty manifests and
  duplicate entries.
- Immutable `tuple` schemas adopted across `CategoricalSpec`,
  `LabelScheme`, and `FeatureSchemes`.
- Simplified validation primitive signatures across
  `landseg.utils.validator` with type-annotated overloads.

### 1.2. Preflight Architecture Gaps

[ADR-0060](./ADR-0060-higher-level-workflow-orchestration-and-preflight-validation.md)
established the pre-flight validation engine (`landseg.execution.preflight`)
with a probe-based diagnostic framework. However, two architectural
shortcomings emerged in practice:

1. **Coupling of Stage Validation and Execution Logic**:
   - **`geopipe.grid`**: Mode checks, reference raster existence, and
     extent calculations were interleaved inside `GridLayout`
     construction and `lifecycle.prepare_world_grid()`, preventing
     preflight from probing grid specification validity without
     triggering full artifact persistence.
   - **`geopipe.harmonize`**: Manifest validation, grid report loading,
     and collision hashing were coupled with pipeline initialization
     and I/O.
   - **`geopipe.ingest`**: Raster alignment, feature feasibility, and
     pool grid compatibility checks were spread across `blocks.mapper`,
     `blocks.assembler.builder`, and `context.py`, often interleaved with
     array reading or memory allocations.
   - **`geopipe.prepare`**: Preflight lacked a standard entry point to
     validate requested features and multi-head target hierarchies
     against upstream ingested schema artifacts prior to preparation.

2. **Divergent Preflight Engine and Duplicate Validation Paths**:
   - The preflight engine maintained separate subsystems:
     `prerequisites.py` (imperative gate-checks) and `probes/`
     (heuristic diagnostic evaluations).
   - Target inspection was hardcoded through procedural match-arms
     rather than a declarative suite.
   - `assert_target_ready()` and `inspect_target()` checked different
     subsets of conditions, creating duplicate code paths and subtle
     contract drift between pre-flight assertions and execution gating.

### 1.3. Practical Design Evolution: Handlers to Validators

The initial draft of this ADR envisioned creating standalone
`handlers.py` modules within each geopipe stage containing paired
`validate_<noun>` and `execute_<verb>` functions. During implementation,
practical codebase constraints led to an ad-hoc simplification:

- **Intermediate `execute_*` wrappers were dropped**: Each stage already
  possesses authoritative lifecycle entry points and pipeline runners
  (`prepare_world_grid()`, `run_data_harmonization()`,
  `run_data_ingestion()`, `run_data_preparation()`). Introducing an extra
  `execute_*` layer added boilerplate indirection without architectural
  benefit.
- **Dedicated `handlers.py` files dissolved into `validators` namespace**:
  Instead of adding new submodules, each geopipe stage package exposes a
  clean, public `validators` namespace object directly on its module
  `__init__.py`.
- **Preflight restructured into a catalog-driven engine**: Rather than
  appending probes onto the legacy engine, preflight was restructured into
  a catalog-driven readiness engine (`catalog.py`, `context.py`,
  `engine.py`, `checks/contracts.py`), deprecating `prerequisites.py` and
  unifying diagnostic inspection with fail-fast execution gating.

---

## 2. Decision

We established a **stage-level validation surface** exposed via a
`validators` namespace in each `geopipe.<stage>` module, backed by a
**declarative catalog-driven readiness engine** in
`landseg.execution.preflight`.

Validation functions are pure, read-only helpers that use
`landseg.utils.validator` primitives. The preflight engine organizes all
target checks into declarative suites in `catalog.py`, delegating domain
contract verification directly to stage `validators` through standardized
contract probes in `checks/contracts.py`.

### 2.1. Stage `validators` Namespace Architecture

Each geopipe stage (`grid`, `harmonize`, `ingest`, `prepare`) exposes a
public `validators` singleton on its `__init__.py`, instantiated from an
internal `_Validators` class.

Validators adhere to the following principles:
1. **Purity and Read-Only Operation**: Validators never mutate disk
   artifacts, allocate compute arrays, or perform destructive operations.
2. **Primitive Input Contracts**: Validators accept minimal
   configurations, metadata, or path arguments (`pathlib.Path`), avoiding
   coupling to instantiated pipeline runtime containers.
3. **Fail-Fast Error Propagation**: Failures raise `val.ValidationError`
   or stage-specific domain errors (e.g., `artifacts.ArtifactError`,
   `DatasetManifestError`).
4. **Direct Execution Reuse**: Stage pipeline runners invoke these same
   validation functions prior to processing, guaranteeing preflight and
   execution share an identical domain contract.

### 2.2. Unified Stage Validator Inventory

| Stage | Validator | Responsibility | Delegated Implementation |
|---|---|---|---|
| `grid` | `grid_specs(config)` | Validates grid generation specs: checks reference raster bounds (`'ref'` mode) or enforces CRS pattern, origin, pixel size, and tile/stride bounds (`'manual'` mode). | `geopipe.grid.builder.compile_inputs()` |
| `harmonize` | `grid_reference(grid_report_path)` | Loads and validates world grid reference artifact from upstream grid report JSON, returning restored `GridLayout`, `grid_id`, and path. | `geopipe.harmonize.context.load_grid_reference()` |
| `harmonize` | `manifest_contract(manifest_path)` | Reads, validates, and compiles dataset manifest JSON; enforces non-empty entries and source file existence; rejects duplicate collisions. | `geopipe.harmonize.manifest.compiler.compile_dataset_manifest()` |
| `ingest` | `feature_specs(image_fpath, label_fpath, *, add_spectral, add_topo)` | Verifies requested spectral indices and topographic features against available raster bands without loading full imagery arrays. | `geopipe.ingest.blocks.assembler.io.RasterReadContext.validate_feature_specs()` |
| `ingest` | `pool_compatibility(schema_fpath, grid_fpath)` | Validates incoming world grid block identity against existing dataset schema pool identity. Accepts `pathlib.Path` or `GridLayout`. | `geopipe.ingest.context.verify_pool_grid_compatibility()` |
| `ingest` | `raster_alignment(image_fpath, grid_fpath, label_fpath)` | Reads raster headers to verify CRS compatibility, pixel resolution, and spatial extent intersection with the world grid. | `geopipe.ingest.blocks.mapper.mapper.validate_raster_alignment()` |
| `prepare` | `dataset_semantics(schema, config=None, *, ...)` | Validates requested feature channels, multi-head target hierarchy, and focal head against ingested schema. | `geopipe.prepare.dataset.semantics.resolve_*` |

*Note on Partition Configuration*: Validation of partition ratios,
buffer strides, and AOI metadata is enforced directly within the
configuration schema (`PreparationPartitionConfig`) via validator
primitives, eliminating redundant probe layers.

### 2.3. Catalog-Driven Preflight Engine

`landseg.execution.preflight` was restructured into a modular,
catalog-driven readiness engine:

1. **Declarative Target Suites (`catalog.py`)**: Single source of truth
   defining ordered `ProbeSpec(pid, run)` tuples for each target in
   `SUPPORTED_TARGETS` (`'world-grid'`, `'data-harmonize'`,
   `'data-ingest'`, `'data-prepare'`, `'model-train'`, `'model-evaluate'`,
   `'diagnose-overfit'`). Each suite executes in strict phase order:
   - **Filesystem**: Output directory writability (`checks.filesystem`).
   - **Lineage**: Upstream reports, manifests, and run history (`checks.lineage`).
   - **Policy**: Execution rules such as collision policies (`checks.policy`).
   - **Contract**: Domain validation backed by stage `validators` (`checks.contracts`).
   - **Ledger**: Informational summaries of block pools and prepared datasets (`checks.ledger`).
   - **Hardware & Model**: GPU/RAM resources and model checkpoints (`checks.hardware`, `checks.model`).
2. **Unified Inspection & Fail-Fast Assertions (`engine.py`)**:
   - `inspect_target(target, ctx)`: Executes the target's declarative probe
     suite against an `InspectionContext`. Probe exceptions are contained
     as failing `ProbeResult` entries with category `EXECUTION`, ensuring
     an unexpected probe crash does not abort the diagnostic report.
   - `assert_target_ready(target, config, ...)`: Replaced legacy
     `prerequisites.py` by delegating directly to `inspect_target()`. If
     any probe returns `ProbeStatus.FAIL`, it raises a fail-fast
     `RuntimeError` listing all failure reasons.
3. **Contract Probe Adapters (`checks/contracts.py`)**: Thin adapters
   that invoke stage `validators` and map outcomes into `ProbeResult`
   instances (`grid_specs`, `grid_artifact`, `manifest_contract`,
   `feature_specs`, `pool_compatibility`, `raster_alignment`,
   `prepare_semantics`).

### 2.4. Ingestion Assembler Streamlining & Primitive Paths

To support read-only feature validation and simplify assembler data flow:
- Consolidated fragmented builder dataclasses into `RasterReadContext` and
  unified `BlockBuildingInputs`.
- `build_data_block()` consumes `RasterReadOutput` directly, eliminating
  the previous intermediate builder triad.
- Ingestion validators accept primitive path arguments (`pathlib.Path` or
  `GridLayout`), enabling modular verification on arbitrary inputs without
  instantiating session or pipeline containers.
- Relocated diagnostic block generation (`build_test_block()`) from
  production lifecycle to `landseg.execution.workflows.diagnose_overfit`.

### 2.5. Validator Primitives Simplification

- Removed verbose parameter name arguments (`name`, `value_name`,
  `file_name`) across `landseg.utils.validator` functions.
- Retained type-annotated `@typing.overload` signatures for static analysis.
- Aliased `val = validator` on `BaseConfigSection`, streamlining schema
  validation across configuration dataclasses.

### 2.6. Diagnostic Character vs. Authoritative Stage Gating

A critical architectural distinction must be emphasized: current stage
`validators` and preflight contract probes lean towards **diagnostic**
inspection rather than **authoritative execution guarantees**. A passing
preflight check confirms prerequisite presence, path validity, and
lightweight schema compliance, but:

$$\text{preflight checks passed} \not\equiv \text{geopipe stage guaranteed to run}$$

Authoritative gating remains strictly enforced during actual pipeline
execution through two sequential barriers:
1. **Configuration Schema Validation**: Strong typing and value bounds
   enforced by dataclasses in `landseg.configs.schema` via `val = validator`
   primitives.
2. **Runtime Context Building**: Domain-specific context builders
   (`build_harmonization_context()`, `build_ingestion_context()`,
   `build_preparation_context()`, `GridLayout.from_specs()`) that load full
   artifacts, inspect rasters, and assemble in-memory execution state.

During branch development, there was a temptation to hoist runtime context
building into preflight checks to achieve fully authoritative preflight
guarantees. However, this was deliberately dropped to prevent scope creep:
context building reads extensive rasters, scans file trees, and prepares
runtime objects, which conflicts with preflight's core mandate of being a
fast, non-destructive, zero-footprint dry-run inspection. Hoisting context
construction may be considered in a future refactoring, but was intentionally
not proceeded with on this branch.

---

## 3. Architecture

```text
+-------------------------------------------------------------------------+
|                       execution.preflight                               |
|  catalog.py      -- declarative target suites (SUITES, ProbeSpec)       |
|  engine.py       -- inspect_target(), assert_target_ready()             |
|  context.py      -- InspectionContext (config, artifact paths)          |
|  checks/         -- categorized checks (filesystem, lineage, policy...) |
|     contracts.py -- thin adapters calling geopipe.<stage>.validators    |
+-----------------------------------+-------------------------------------+
                                    | calls
                                    v
+-------------------------------------------------------------------------+
|                       geopipe.<stage>                                   |
|  __init__.py     -- exposes validators = _Validators() namespace        |
|                     pure, read-only domain validation functions         |
|  pipeline.py / lifecycle.py -- authoritative execution entry points     |
+-----------------------------------+-------------------------------------+
                                    | uses
                                    v
+-------------------------------------------------------------------------+
|                     landseg.utils.validator                             |
|  require_types, require_numeric_range, require_file, require_sequence...|
+-------------------------------------------------------------------------+
```

---

## 4. Implementation & Migration Summary

All planned capabilities were implemented across five phases:

1. **Validator Primitives Simplification & Schema Aliasing**: Dropped
   parameter name arguments across `landseg.utils.validator`, introduced
   direct `val = validator` aliasing on `BaseConfigSection`, and aligned
   schema validation across all config dataclasses.
2. **Geopipe Stage Validator Namespaces**: Created `_Validators` classes
   and exposed public `validators` namespaces on `geopipe.grid`,
   `geopipe.harmonize`, `geopipe.ingest`, and `geopipe.prepare` package
   inits; eliminated redundant intermediate `handlers.py` modules.
3. **Ingestion Assembler Streamlining & Primitive Paths**: Consolidated
   assembler data structures into `RasterReadContext` and
   `BlockBuildingInputs`; updated ingestion validators to accept primitive
   `pathlib.Path` or `GridLayout` arguments; relocated `build_test_block()`
   to `diagnose_overfit`.
4. **Preflight Catalog Restructuring & Prerequisites Retirement**:
   Implemented declarative `SUITES` in `catalog.py`, migrated probe
   implementations into categorized `checks/` modules, wired
   `checks.contracts` to stage `validators`, unified `assert_target_ready()`
   with `inspect_target()` in `engine.py`, and removed legacy
   `prerequisites.py` and `probes/`.
5. **Test Suite Realignment & Verification**: Rewrote preflight unit tests
   in `test_preflight.py` to assert against catalog suites and categorized
   checks without prerequisite mocks; aligned stage tests to verify the
   public `validators` namespaces; confirmed full passing test coverage.

---

## 5. Consequences

### Positive

- **Single Source of Truth**: Domain validation rules reside strictly
  inside owning geopipe stage modules. Preflight probes and pipeline
  execution share identical validation functions.
- **Safe, Read-Only Diagnostics**: Preflight diagnostic checks never
  invoke heavy array operations, window mapping, block writing, or
  destructive artifact manipulation.
- **Zero Handler Boilerplate**: Exposing `stage.validators` on package
  inits eliminated intermediate `handlers.py` and `execute_*` wrapper
  indirection.
- **Declarative Readiness**: Target readiness is defined as declarative
  data in `catalog.py`. `assert_target_ready()` and `inspect_target()`
  share identical logic, eliminating prerequisite divergence.
- **Streamlined Validation Primitives**: Leaner validator signatures
  reduce visual noise and call-site complexity across schemas and stages.
- **Primitive Path Contracts**: Ingestion validators operate directly on
  `pathlib.Path` or `GridLayout`, enabling modular validation without
  heavy container dependencies.

### Negative / Considerations

- **Diagnostic Rather Than Authoritative Gating**: Preflight validation
  guarantees prerequisite sanity and fast diagnostics, but is not an
  authoritative execution guarantee (`preflight passed != stage execution
  guaranteed`). Authoritative gating remains within config validation and
  stage context building (`build_*_context`). Converting context building into
  preflight checks was deferred to avoid scope creep.
- **Unit Test Migration Required**: Preflight tests that mocked
  `prerequisites.py` or old `probes/` had to be rewritten to target
  `catalog.py` and `checks/`.
- **Top-Level Lazy Resolution**: `validators` namespaces on `__init__.py`
  must be exported in `__all__` and documented to preserve clean static
  analysis and IDE autocompletion.

---

## 6. Relationship to Prior ADRs

| ADR | Relationship |
|---|---|
| [ADR-0060 (Preflight Engine)](./ADR-0060-higher-level-workflow-orchestration-and-preflight-validation.md) | Restructures the probe-based preflight engine into a declarative, catalog-driven readiness engine with unified fail-fast gating and authoritative contract checks. |
| [ADR-0059 (Incremental Batch Ingestion)](./ADR-0059-incremental-batch-ingestion-and-block-collision-lifecycle.md) | Ingestion validators formalize pool grid compatibility and feature engineering feasibility as pure, path-based validation entry points. |
| [ADR-0057 (Dynamic Dataset Semantics)](./ADR-0057-dynamic-dataset-semantics-in-memory-data-context-and-decoupled-preparation-pipeline.md) | Preparation validators directly wrap the feature channel and multi-head target resolvers introduced in ADR-0057. |
| [ADR-0056 (World Grid Pipeline)](./ADR-0056-first-class-world-grid-pipeline-and-geopipe-modularization.md) | Grid validator encapsulates the grid input compilation and CRS validation previously embedded inside the builder. |
| [ADR-0054 (ETL-First Architecture)](./ADR-0054-etl-first-architecture-and-artifact-renaming.md) | Harmonization validators formalize manifest compilation and grid reference verification for ETL pipelines. |
| [ADR-0050 (Decouple Datablock Assembly)](./ADR-0050-decouple-datablock-assembly.md) | Streamlines assembler containers into `RasterReadContext` and `BlockBuildingInputs`. |
