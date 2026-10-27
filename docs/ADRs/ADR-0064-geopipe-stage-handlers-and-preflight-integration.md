# ADR-0064: Geopipe Stage Handlers and Preflight Integration

**Status:** Proposed<br>
**Date:** 2026-10-27

---

## 1. Context

### 1.1. Foundation Laid by the Validator Primitives Branch

The four most recent commits on `main` introduced
`landseg.utils.validator` -- a zero-dependency module of primitive
validation functions (`require_types`, `require_numeric_range`,
`require_value_from`, `require_mapping_str_v`, `require_file`, etc.)
that raise a canonical `ValidationError`. These primitives were then
adopted across `landseg.configs.schema` and
`geopipe.harmonize.manifest.normalizer` / `compiler`, replacing ad-hoc
`isinstance` checks, bare `ValueError` raises, and inline guard clauses
with a uniform, composable validation vocabulary.

Key changes already on `main`:

- `ManifestEntry` and `AllowedCategory` colocated into
  `normalizer.py`; the redundant `schema.py` was removed.
- `ManifestEntryNormalizer` class was converted to a functional
  `normalize_manifest_entry()`.
- `compile_dataset_manifest()` now guards root JSON structure and
  individual entries with validator primitives, and rejects empty
  manifests.
- Immutable `tuple` schemas were adopted across `CategoricalSpec`,
  `LabelScheme`, and `FeatureSchemes`.

### 1.2. Current Preflight Architecture Gap

ADR-0060 established the pre-flight validation engine
(`landseg.execution.preflight`) with a probe-based diagnostic
framework. Today, the engine works well for its supported categories
(lineage, filesystem, ledger, hardware, spatial, dataset, policy,
model), but its **domain-contract probes rely on external heuristics
rather than the authoritative validation logic living inside each
geopipe stage module**.

Concretely:

- **`geopipe.grid`**: The builder validates mode, reference raster
  existence, and manual-specs presence inline. These checks are
  interleaved with `GridLayout` construction, so preflight cannot
  probe grid-definition validity without triggering the full artifact
  lifecycle in `lifecycle.prepare_world_grid()`.
- **`geopipe.harmonize`**: The manifest compiler and normalizer are
  now clean validation boundaries, but the compiler calls
  `artifacts.Controller.hash(overwrite=False)` and issues I/O, and
  no explicit output-name-uniqueness guard exists yet. The
  processor's defensive checks (e.g., `_tag_label_metadata()`
  rejecting missing categorical specs) are duplicated rather than
  shared with the normalizer.
- **`geopipe.ingest`**: Raster alignment checks, block-config
  feasibility checks, domain-build parameter validation, and
  block-pool compatibility are spread across `blocks.mapper`,
  `blocks.assembler.builder`, `domains.builder`, and `context`.
  Some are interleaved with computation (window mapping, PCA, block
  writing), and `_structural_validation()` in
  `blocks.assembler.lifecycle` has side effects (file deletion) that
  preflight must not trigger.
- **`geopipe.prepare`**: Feature/target resolvers in
  `dataset.semantics` are excellent shared APIs, but inline
  target-override validation is incomplete, preparation-context
  artifact/schema cross-checks are ad-hoc, and partition splitter
  input validation has a check-ordering bug (`shape` indexed before
  `ndim`).

The overarching problem is that **each stage's validation logic and
execution logic are coupled inside the same functions**, making it
impossible for preflight to call the validation path without also
triggering artifact I/O, mutation, or heavy computation.

### 1.3. Design Objective

Let each geopipe stage module expose **stable handlers** -- thin,
public functions whose sole job is to encapsulate either validation
logic or execution logic for that stage. Preflight probes will call
the validation handlers directly, using the same code path as the
stage pipeline itself, guaranteeing that preflight and execution
enforce identical contracts.

---

## 2. Decision

We will introduce a **handler-based contract surface** within each
`geopipe.<stage>` package, split cleanly into validation handlers
and execution handlers. Validation handlers will leverage
`landseg.utils.validator` primitives wherever applicable, and each
handler will be a pure, read-only function that raises
`ValidationError` (or a stage-specific domain error wrapping it) on
failure, returning normalized/validated data on success.

The preflight engine will call these validation handlers instead of
reimplementing equivalent checks as independent probes.

### 2.1. Handler Protocol

Each geopipe stage (`grid`, `harmonize`, `ingest`, `prepare`) will
expose its handlers through a `handlers.py` module (or a `handlers/`
sub-package if the stage requires it). Every handler will follow
these conventions:

1. **Validation handlers** are pure, read-only functions.
   - Named `validate_<noun>` (e.g., `validate_grid_inputs`,
     `validate_data_block_config`).
   - Accept only the minimal inputs needed for the check (config
     objects, metadata, file paths), never raw arrays or open file
     handles.
   - Return a normalized/validated value or `None`; raise
     `ValidationError` on failure.
   - Have zero side effects: no file writes, no artifact mutations,
     no expensive computation.
   - Use `landseg.utils.validator` primitives for type, range,
     membership, and file-existence checks, expanding the primitive
     set as needed.

2. **Execution handlers** are the stage's authoritative entry points
   for producing artifacts.
   - Named `execute_<verb>` (e.g., `execute_grid_build`,
     `execute_block_ingestion`).
   - Call the corresponding validation handler(s) first, then
     proceed with I/O and computation.
   - May have side effects (writes, deletes, artifact registration).

3. The stage's existing `pipeline.py` (or runner) will delegate to
   execution handlers. Existing internal functions remain as private
   helpers.

```text
geopipe.<stage>.handlers
    +-- validate_<A>(...)   ->  normalized value | raises ValidationError
    +-- validate_<B>(...)   ->  normalized value | raises ValidationError
    +-- execute_<X>(...)    ->  calls validate_*, then computes & persists

execution.preflight.probes
    +-- probe_<stage>_<A>(...)  ->  calls validate_<A>(...),
                                   catches ValidationError,
                                   returns ProbeResult(PASS|FAIL)
```

### 2.2. Stage-Specific Handler Inventory

#### 2.2.1. `geopipe.grid.handlers`

| Handler | Responsibility | Validator primitives |
|---|---|---|
| `validate_grid_inputs(mode, tile_size, tile_stride, ref_fpath, manual_specs)` | Validate mode is `'ref'` or `'manual'`; validate tile dimensions are positive integers; in `ref` mode validate reference raster exists and is readable via `require_file`; in `manual` mode validate CRS string, origin, pixel size, extent via `require_string`, `require_numeric_sequence`, `require_numeric_min` | `require_value_from`, `require_numeric_sequence`, `require_numeric_min`, `require_file` |
| `validate_grid_reference_metadata(ref_fpath)` | Inspect reference raster spatial metadata (CRS present, resolution valid, bounds non-degenerate) without constructing `GridLayout` | `require_file`, `require_string` |
| `execute_grid_build(...)` | Call both validators, then delegate to `builder.build_grid()` and `lifecycle.prepare_world_grid()` | -- |

#### 2.2.2. `geopipe.harmonize.handlers`

| Handler | Responsibility | Validator primitives |
|---|---|---|
| `validate_manifest_contract(manifest_fp)` | Compile and validate the complete manifest through the existing `compiler.compile_dataset_manifest()` plus new guards: check effective output-name uniqueness across entries, and verify read-only compiler behavior | `require_file`, existing compiler/normalizer (which already use validator primitives) |
| `validate_harmonization_inputs(manifest_entries, grid)` | Verify that compiled manifest entries are compatible with the target grid (CRS agreement, band-mapping completeness) | `require_string`, `require_sequence` |
| `execute_harmonization(...)` | Call validators, then delegate to `pipeline.exec_harmonize_data()` | -- |

#### 2.2.3. `geopipe.ingest.handlers`

| Handler | Responsibility | Validator primitives |
|---|---|---|
| `validate_raster_alignment(image_fpath, label_fpath, grid)` | Extract raster metadata and check CRS, resolution compatibility, and spatial overlap with the target grid -- without computing tile-window mappings | `require_file`, `require_string` |
| `validate_data_block_config(config, context, has_label)` | Check spectral index support, required bands, topo feature support, DEM availability, and label categorical specs -- without computing arrays | `require_value_from`, `require_sequence`, `require_numeric_range` |
| `validate_domain_build_config(valid_threshold, target_variance)` | Enforce valid-threshold in (0, 1] and target-variance in (0, 1]; replace assertions on integer dtype/nodata with explicit `ValidationError` | `require_numeric_range` |
| `inspect_existing_blocks(pool_dpath, grid)` | Read-only integrity and collision inventory: check NPZ readability and report damaged/colliding blocks without deleting any files | `require_file` |
| `validate_pool_grid_compatibility(schema_fpath, grid)` | Expose the existing `verify_pool_grid_compatibility()` as a stable handler, making unknown-identity handling explicit | `require_file` |
| `execute_block_ingestion(...)` | Call validators, then run `blocks.pipeline`, domain builds, and manifest updates | -- |

#### 2.2.4. `geopipe.prepare.handlers`

| Handler | Responsibility | Validator primitives |
|---|---|---|
| `validate_preparation_inputs(schema_fpath, catalog_fpath, mapped_windows_fpath, config)` | Cross-artifact consistency: schema exists and is loadable, catalog structure matches schema, tile dimensions are positive, overlap is compatible with tile shape, at least one image source path exists with a valid CRS | `require_file`, `require_numeric_min`, `require_numeric_sequence`, `require_mapping` |
| `validate_feature_target_resolution(schema, config)` | Delegate to existing `resolve_feature_channels()`, `resolve_target_heads()`, `resolve_focal_head()` and additionally validate inline target-override schemes (source class indices, group ID contiguity, class-name mapping completeness) | `require_mapping`, `require_numeric_range`, `require_sequence` |
| `validate_partition_config(val_ratio, test_ratio, buffer_step, aoi_config)` | Validate ratio ranges, positive stride (fix zero-stride acceptance), AOI raster availability and spatial metadata; fix `_validate_inputs()` check ordering (`ndim` before `shape` indexing) | `require_numeric_range`, `require_numeric_min`, `require_file` |
| `execute_preparation(...)` | Call validators, then run partitioning, normalization, and schema emission | -- |

### 2.3. Preflight Integration

The preflight engine will introduce new **stage-contract probes**
that call the validation handlers:

```text
execution.preflight.probes
    +-- filesystem.py       (existing)
    +-- hardware.py         (existing)
    +-- ledger.py           (existing)
    +-- spatial.py          (existing -- grid spatial metadata)
    +-- dataset.py          (existing)
    +-- model.py            (existing)
    +-- policy.py           (existing)
    +-- stage_contracts.py  (NEW -- handler-backed domain validation)
```

`stage_contracts.py` will contain probe functions of the form:

```python
def probe_grid_inputs(
    root_config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Validate grid definition through authoritative handler.'''
    try:
        grid_handlers.validate_grid_inputs(
            mode=root_config.data.world_grid.mode,
            tile_size=...,
            tile_stride=...,
            ref_fpath=...,
            manual_specs=...,
        )
    except val.ValidationError as exc:
        return schema.ProbeResult(
            pid='grid_inputs',
            category='Stage Contract',
            status=schema.ProbeStatus.FAIL,
            message=str(exc),
        )
    return schema.ProbeResult(
        pid='grid_inputs',
        category='Stage Contract',
        status=schema.ProbeStatus.PASS,
        message='Grid definition inputs validated.',
    )
```

The engine (`engine.py`) will wire these new probes into the existing
`inspect_target()` dispatch under each target's match arm,
supplementing (and where appropriate replacing) existing spatial and
dataset probes that currently perform equivalent checks independently.

### 2.4. New Validator Primitives (Expansion)

During handler implementation, the following additions to
`landseg.utils.validator` are anticipated:

| Primitive | Purpose |
|---|---|
| `require_directory(path, name, *, must_exist)` | Validate a directory path exists and is a directory; return `Path` |
| `require_nonempty_sequence(value, name)` | Shorthand combining `require_sequence` + non-empty check |
| `require_positive_int(value, name)` | Shorthand for `require_numeric_min(value, name, 1, int)` |

These will be added conservatively and only when multiple handlers
share the same compound check.

---

## 3. Architecture

### 3.1. Layered Responsibility Model

```text
+----------------------------------------------------------+
|                   execution.preflight                    |
|  probes/stage_contracts.py calls validation handlers     |
|  translates ValidationError -> ProbeResult(PASS|FAIL)    |
+-------------------------+--------------------------------+
                          | calls
                          v
+----------------------------------------------------------+
|              geopipe.<stage>.handlers                    |
|  validate_*()  -- pure, read-only, uses validator prims  |
|  execute_*()   -- calls validate_*, then computes/writes |
+-------------------------+--------------------------------+
                          | delegates to
                          v
+----------------------------------------------------------+
|          geopipe.<stage>.{builder,pipeline,...}           |
|  existing computation, I/O, lifecycle management         |
+-------------------------+--------------------------------+
                          | uses
                          v
+----------------------------------------------------------+
|              landseg.utils.validator                     |
|  require_types, require_numeric_range, require_file, ... |
+----------------------------------------------------------+
```

### 3.2. Module Layout After Implementation

```text
src/landseg/
+-- utils/
|   +-- validator.py                   (expanded with new primitives)
+-- geopipe/
|   +-- grid/
|   |   +-- handlers.py                (NEW)
|   |   +-- builder.py                 (delegates to handlers)
|   |   +-- lifecycle.py
|   +-- harmonize/
|   |   +-- handlers.py                (NEW)
|   |   +-- manifest/
|   |   |   +-- compiler.py            (strengthened, called by handler)
|   |   |   +-- normalizer.py          (unchanged -- already refactored)
|   |   +-- context.py
|   |   +-- pipeline.py
|   +-- ingest/
|   |   +-- handlers.py                (NEW)
|   |   +-- blocks/
|   |   |   +-- assembler/
|   |   |   |   +-- builder.py         (config checks extracted)
|   |   |   |   +-- lifecycle.py       (split: inspect vs. mutate)
|   |   |   |   +-- io.py
|   |   |   +-- mapper/
|   |   +-- domains/
|   |   |   +-- builder.py             (assertions -> ValidationError)
|   |   +-- context.py                 (verify_pool_grid -> handler)
|   |   +-- pipeline.py
|   +-- prepare/
|       +-- handlers.py                (NEW)
|       +-- context.py                 (delegates to handler)
|       +-- dataset/
|       |   +-- semantics.py           (inline-override validation added)
|       |   +-- normalizer.py
|       |   +-- view.py
|       +-- partition/
+-- execution/
    +-- preflight/
        +-- probes/
            +-- stage_contracts.py     (NEW -- handler-backed probes)
```

---

## 4. Implementation Plan

### Phase 1: Harmonization Handlers (Lowest Risk -- Already Refactored)

1. Create `geopipe.harmonize.handlers` with
   `validate_manifest_contract()` wrapping
   `compile_dataset_manifest()` plus new effective output-name
   uniqueness check.
2. Add `validate_harmonization_inputs()` for grid-manifest CRS
   agreement.
3. Wire `probes.stage_contracts.probe_manifest_contract()` in
   preflight for `data-harmonize` target.
4. Add unit tests for both handlers.

### Phase 2: Grid Handlers

1. Create `geopipe.grid.handlers` with `validate_grid_inputs()` and
   `validate_grid_reference_metadata()`.
2. Refactor `builder.build_grid()` to call `validate_grid_inputs()`
   first.
3. Wire `probes.stage_contracts.probe_grid_inputs()` in preflight
   for `world-grid` target.
4. Add unit tests.

### Phase 3: Ingestion Handlers (Largest Scope)

1. Create `geopipe.ingest.handlers` with:
   - `validate_raster_alignment()` -- extracted from
     `blocks.mapper.map_rasters()`.
   - `validate_data_block_config()` -- extracted from
     `blocks.assembler.builder.build_data_block()`.
   - `validate_domain_build_config()` -- extracted from
     `domains.builder.build_domain()`, replacing assertions with
     `ValidationError`.
   - `inspect_existing_blocks()` -- read-only split from
     `blocks.assembler.lifecycle._structural_validation()`.
   - `validate_pool_grid_compatibility()` -- thin wrapper around
     existing `context.verify_pool_grid_compatibility()`.
2. Split `_structural_validation()` in `lifecycle.py` into read-only
   inspection and destructive lifecycle operations.
3. Wire probes for `data-ingest` target.
4. Add unit tests.

### Phase 4: Preparation Handlers

1. Create `geopipe.prepare.handlers` with:
   - `validate_preparation_inputs()` -- cross-artifact consistency.
   - `validate_feature_target_resolution()` -- delegating to existing
     `semantics` resolvers plus inline-override validation.
   - `validate_partition_config()` -- fix `_validate_inputs()` check
     ordering and zero-stride acceptance.
2. Wire probes for `data-prepare` target.
3. Add unit tests.

### Phase 5: Validator Primitive Expansion and Polish

1. Add new `validator.py` primitives as identified during Phases 1-4.
2. Sweep remaining `assert` guards on external inputs across all four
   stages and replace with `ValidationError` where the input comes
   from user configuration, file metadata, or upstream artifacts.
3. Run full unit and integration test suites.

---

## 5. Consequences

### Positive

- **Single Source of Truth**: Every domain validation rule lives in
  exactly one handler function. Preflight and execution share the
  same code path -- no duplicated or divergent checks.
- **Safe Preflight Probing**: Validation handlers are pure and
  read-only by construction. Preflight can call them without risk of
  file deletion, artifact mutation, or expensive computation.
- **Composable Validator Stack**: The handler layer composes
  `landseg.utils.validator` primitives into domain-specific checks.
  New primitives can be added conservatively when multiple handlers
  share a pattern.
- **Clearer Module Boundaries**: Each stage's `handlers.py` becomes
  the stable public API surface for both internal pipeline use and
  external preflight integration.
- **Incremental Adoption**: Each phase targets one stage and can be
  merged independently. The branch already includes the four
  foundation commits (validator primitives + harmonize manifest
  refactoring), so Phases 1-2 are low-risk continuations of
  completed work.

### Negative / Considerations

- **Additional Module per Stage**: Four new `handlers.py` files
  increase the module count, though each is a thin coordination
  layer rather than a new subsystem.
- **Potential Over-Extraction**: Not every internal `if`/`assert`
  should become a handler. Guards protecting internal invariants
  during numerical computation should remain as defensive assertions
  at their point of use; only checks expressing a meaningful input
  contract that another caller needs to rely on should be promoted.
- **Existing Probe Overlap**: Some current preflight probes (e.g.,
  `spatial_reference`, `raw_dataset`) perform lightweight checks
  that may now overlap with handler-backed probes. These will be
  evaluated case-by-case: retire the probe if the handler subsumes
  it, or keep it if the probe provides complementary information
  (e.g., telemetry details for the dashboard).

---

## 6. Relationship to Prior ADRs

| ADR | Relationship |
|---|---|
| ADR-0060 (Preflight Engine) | This ADR extends ADR-0060 by providing authoritative stage-level validation entry points that preflight probes will call, replacing independent heuristic checks. |
| ADR-0059 (Incremental Batch Ingestion) | Ingestion handlers formalize the collision-inspection / pool-compatibility checks introduced in ADR-0059 as stable, read-only APIs. |
| ADR-0057 (Dynamic Dataset Semantics) | Preparation handlers delegate to the feature/target resolvers introduced in ADR-0057, adding inline-override validation. |
| ADR-0056 (World Grid Pipeline) | Grid handlers encapsulate the grid-definition validation that is currently inline in the builder introduced by ADR-0056. |
