# ADR-0063: Single-File Artifact Envelopes and Schema Standardization

**Status:** Accepted — Implemented<br>
**Date:** 2026-10-26

---

## 1. Context

In [ADR-0014](./ADR-0014-decouple-artifacts-lifecycle-consistent-handling.md),
we introduced `landseg.artifacts.PayloadController[D, M]` to manage
"split-file payloads" where bulk serialized data was stored in `x.json` and
schema/metadata sidecars were stored in `x_meta.json`. The original goal was
to allow inspection of spatial metadata and `schema_id` without parsing bulk
coordinate arrays.

In practice, this abstraction proved counterproductive:

1. **No Independent Metadata Access**: Across the entire codebase,
   `PayloadController.load()` always loaded both `data.json` and `_meta.json`
   synchronously. No call site ever loaded only `_meta.json`.
2. **Dual-File State Hazards**: The sidecar `_meta.json` introduced
   desynchronization risks when files were moved, renamed, or inspected, and
   cluttered artifact directories.
3. **Incomplete Adoption**: Only two artifacts (`GridLayout` and
   `DomainTileMap`) ever used `PayloadController`. Other large artifacts like
   `data_catalog.json`, `data_schema.json`, and `prepared_schema.json`
   bypassed it entirely and saved as single JSON files.
4. **Metadata Already Handled via Stage Reports**: Preflight probes and
   downstream stages already inspect metadata without loading data arrays by
   reading the lightweight `report.json` emitted by each stage (e.g.,
   `WorldGridReport` in `report.json`).
5. **Inconsistent Schema Versioning**: Schema validation across single-file
   artifacts was fragmented. `DataSchema` used `schema_id`, while
   `PreparedSchema` used `schema_version`, and neither was validated on load
   by `artifacts.Controller`.

---

## 2. Decision

We have retired `PayloadController` and the split-file `_meta.json` sidecar
pattern in favor of **single-file self-contained artifacts**, and standardized
runtime `schema_id` enforcement and lifecycle defaults directly within
`artifacts.Controller`.

### 2.1. Unified Single-File Envelope for Structured Payloads

For structured collections containing metadata and bulk arrays (`GridLayout`
and `DomainTileMap`), we serialized into single self-contained JSON envelopes:

**Grid Layout (`GridPayload`)**:
```json
{
  "schema_id": "grid_layout_payload/v0.1.0",
  "artifact_meta": {
    "gid": "grid_row_256_128_col_256_128",
    "spec": {
      "crs": "EPSG:3161",
      "origin": [0.0, 0.0],
      "pixel_size": [20.0, 20.0],
      "tile_size": [256, 256],
      "tile_stride": [128, 128],
      "tile_overlap": [128, 128]
    }
  },
  "data": [
    [0, 0, 0, 0, 256, 256],
    [0, 128, 0, 128, 256, 256]
  ]
}
```

**Domain Tile Map (`DomainPayload`)**:
```json
{
  "schema_id": "domain_tile_map_payload/v0.1.0",
  "artifact_meta": {
    "crs": "EPSG:3161",
    "explained_variance": 0.95,
    "major_freq_mean": 0.82,
    "major_freq_min": 0.15,
    "pca_axes_n": 4
  },
  "data": {
    "0,0": [1, 0, 0, 0],
    "0,128": [0, 1, 0, 0]
  }
}
```

This guarantees atomic file writes and copies, simplifies path resolution,
and eliminates dangling sidecar files.

### 2.2. Standardized `schema_id` Across Inter-Stage Contracts

All stateful data contracts passed between pipeline stages standardize on a
top-level field `schema_id: str`:

- **`data_schema.json`**: sets `schema_id: 'data_schema/v0.1.0'`.
- **`prepared_schema.json`**: renames `schema_version` to
  `schema_id: 'prepared_schema/v0.1.0'`.
- **`grid.json`**: embeds `schema_id: 'grid_layout_payload/v0.1.0'`.
- **`domains.json`**: embeds `schema_id: 'domain_tile_map_payload/v0.1.0'`.

Operational reports (`report.json`) and audit ledgers (`runs_manifest.json`)
remain unversioned TypedDict schemas as they represent stage execution
summaries rather than domain data contracts.

### 2.3. Native Schema Enforcement and Lifecycle Invariants in `Controller`

Rather than passing schema IDs or strict flags ad-hoc at `fetch()` call sites,
`schema_id` is bound as an invariant property of the `Controller` instance at
creation time and validated bidirectionally:

```python
class Controller(typing.Generic[T]):
    def __init__(
        self,
        file_path: str | pathlib.Path,
        policy: artifacts.LifecyclePolicy | None = None,
        *,
        schema_id: str | None = None,
    ):
        ...
        self.policy = (
            policy if policy is not None
            else artifacts.LifecyclePolicy.LOAD_OR_FAIL
        )
        self.schema_id = schema_id
```

Key lifecycle behaviors:
1. **Default `LOAD_OR_FAIL`**: The default lifecycle policy is
   `LifecyclePolicy.LOAD_OR_FAIL`. Calling `fetch()` on a consumer artifact
   fails fast with `ArtifactError` if the file is missing or corrupt.
2. **Bidirectional Schema Validation**:
   - On `fetch()`: If `self.schema_id` is defined and loaded data is a mapping,
     asserts `data.get('schema_id') == self.schema_id`. Mismatches raise
     `ArtifactError`.
   - On `persist()`: If `self.schema_id` is defined and payload is a mapping,
     asserts `src.get('schema_id') == self.schema_id`. Mismatches raise
     `ArtifactError`.
3. **Explicit Opt-In via `missing_ok`**: `fetch(missing_ok: bool = False)`
   replaces legacy `strict: bool` and `expected_schema_id` arguments. Passing
   `missing_ok=True` gracefully returns `None` strictly when an artifact file
   does not exist on disk under `LOAD_OR_FAIL`. Existing files with corrupt
   JSON or mismatched `schema_id` continue to raise `ArtifactError`.

### 2.4. Call Site Standardization Hierarchy

To ensure code readability and eliminate dangling multi-line argument blocks,
call sites follow a canonical priority hierarchy:

1. **Local Consistency / Symmetry**: When multiple related artifacts are
   loaded or stored in the same function block, prioritize identical visual
   structure across all calls:
   ```python
   Ctrl = artifacts.Controller[dict[str, str]]
   train_splits = Ctrl(train_path).fetch()
   val_splits = Ctrl(val_path).fetch()
   ```
2. **Single-Line `Controller[T](...).fetch()`**: Used when the complete load
   or persist statement fits on a single line ($\le 80$ characters):
   ```python
   report = artifacts.Controller[dict](report_path).fetch()
   ```
3. **Single-Line `Controller[T](...)`**: Used when the controller
   instantiation fits on one line ($\le 80$ characters) before invoking methods
   (e.g., computing a checksum or multiple operations):
   ```python
   ctrl = artifacts.Controller[dict](fpath)
   data = ctrl.fetch()
   checksum = ctrl.compute_hash()
   ```
4. **Scoped `Ctrl = artifacts.Controller[T]`**: Used when parameter bindings
   (`policy=...`, `schema_id=...`) or complex types exceed 80 characters,
   avoiding dangling argument blocks:
   ```python
   Ctrl = artifacts.Controller[geo_core.PreparedSchema]
   schema = Ctrl(
       schema_path,
       schema_id=geo_core.prepared_schema.SCHEMA_ID,
   ).fetch()
   ```

*Note on Typing Imports*: Never introduce unnecessary imports (e.g. `typing`)
solely to parameterize a `Controller` call site when the file does not already
require them. In such cases, use `Ctrl = artifacts.Controller` untyped.

### 2.5. Retirement of `PayloadController` and Legacy Utilities

We have:
1. Migrated `GridLayout` and `DomainTileMap` to single-file envelopes managed
   directly by `artifacts.Controller`.
2. Deleted `src/landseg/artifacts/payload_io.py` and
   `tests/unit/artifacts/test_payload_io.py`.
3. Removed `PayloadController` and `PayloadDict` from package public exports.
4. Retired `Controller.load_json_strict` in favor of standard
   `artifacts.Controller[T](path).fetch()`.
5. Cleaned up legacy split-file test fixtures across the test suite.

---

## 3. Consequences

- **Positive**:
  - Eliminates dual-file state hazards and folder clutter across `world_grids/`
    and `domains/`.
  - Simplifies `landseg.artifacts` to a single, canonical `Controller[T]`
    pattern.
  - Guarantees runtime schema verification on both load and persist operations.
  - Standardizes fail-fast lifecycle loading behavior across pipelines with
    clean `fetch()` call sites.
- **Negative / Neutral**:
  - Reading metadata of large grids requires parsing the single JSON file
    (~15–30 ms), which is negligible and already mitigated by stage
    `report.json`.
- **Supersedes**:
  - Supersedes the split-file `PayloadController` design from
    [ADR-0014](./ADR-0014-decouple-artifacts-lifecycle-consistent-handling.md).
