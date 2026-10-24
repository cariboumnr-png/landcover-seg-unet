# ADR-0063: Single-File Artifact Envelopes and Schema Standardization

**Status:** Proposed<br>
**Date:** 2026-10-24

---

## 1. Context

In [ADR-0014](./docs/ADRs/ADR-0014-decouple-artifacts-lifecycle-consistent-handling.md), we introduced `landseg.artifacts.PayloadController[D, M]` to manage "split-file payloads" where bulk serialized data was stored in `x.json` and schema/metadata sidecars were stored in `x_meta.json`. The original goal was to allow inspection of spatial metadata and `schema_id` without parsing the bulk coordinate arrays.

In practice, this abstraction has proven counterproductive:

1. **No Independent Metadata Access**: Across the entire codebase, `PayloadController.load()` always loads both `data.json` and `_meta.json` synchronously. There is no call site that loads only `_meta.json`.
2. **Dual-File State Hazards**: The sidecar `_meta.json` introduces desynchronization risks when files are moved, renamed, or inspected, and clutters artifact directories.
3. **Incomplete Adoption**: Only two artifacts (`GridLayout` and `DomainTileMap`) ever used `PayloadController`. Much larger artifacts like `data_catalog.json`, `data_schema.json`, and `prepared_schema.json` bypassed it entirely and saved as single JSON files.
4. **Metadata Already Handled via Stage Reports**: Preflight probes and downstream stages already inspect metadata without loading data arrays by reading the lightweight `report.json` emitted by each stage (e.g., `WorldGridReport` in `report.json`).
5. **Inconsistent Schema Versioning**: Schema validation across single-file artifacts is fragmented. `DatasetSchema` uses `schema_id`, while `PreparedSchema` uses `schema_version`, and neither is validated on load by `artifacts.Controller`.

---

## 2. Decision

We will retire `PayloadController` and the split-file `_meta.json` sidecar pattern in favor of **single-file self-contained artifacts**, and we will standardize runtime `schema_id` enforcement directly within `artifacts.Controller`.

### 2.1. Unified Single-File Envelope for Structured Payloads

For structured collections containing metadata and bulk arrays (`GridLayout` and `DomainTileMap`), we will serialize into a single self-contained JSON envelope:

```json
{
  "schema_id": "grid_layout_payload/v1",
  "metadata": {
    "gid": "grid_row_256_128_col_256_128",
    "crs": "epsg:3161",
    "pixel_size": [20.0, 20.0],
    "tile_size": [256, 256],
    "tile_stride": [128, 128],
    "extent": [512, 768]
  },
  "data": [
    [0, 0, 0, 0, 256, 256],
    [0, 128, 0, 128, 256, 256]
  ]
}
```

This ensures atomic file copies, simpler path resolution, and zero dangling sidecar files.

### 2.2. Standardized `schema_id` Across Inter-Stage Contracts

All stateful data contracts passed between pipeline stages will standardize on a top-level field `schema_id: str`:

- **`data_schema.json`**: retains `schema_id: 'data_schema/v1.1'`.
- **`prepared_schema.json`**: renames `schema_version` to `schema_id: 'prepared_schema/v1.0'`.
- **`grid.json`**: embeds `schema_id: 'grid_layout_payload/v1'`.
- **`domains.json`**: embeds `schema_id: 'domain_tile_map_payload/v1'`.

Operational reports (`report.json`) and audit ledgers (`runs_manifest.json`) will remain unversioned TypedDict schemas as they represent stage execution summaries rather than data contracts.

### 2.3. Native Schema Enforcement in `artifacts.Controller`

We will enhance `artifacts.Controller` with native schema verification:

```python
def fetch_strict(self, expected_schema_id: str | None = None) -> T:
    '''Fetch JSON data and verify expected schema_id compatibility.'''
    data = self.fetch()
    if expected_schema_id is not None:
        actual = data.get('schema_id') if isinstance(data, dict) else None
        if actual != expected_schema_id:
            raise artifacts.ArtifactError(
                f"Schema mismatch in {self.filepath.name}: "
                f"expected '{expected_schema_id}', got '{actual}'"
            )
    return data
```

Downstream loaders will invoke `fetch_strict(SCHEMA_ID)` to fail fast if an incompatible artifact is provided.

### 2.4. Retirement of `PayloadController`

We will:
1. Migrate `GridLayout` and `DomainTileMap` to single-file envelopes managed by `artifacts.Controller`.
2. Delete `src/landseg/artifacts/payload_io.py` and `tests/unit/artifacts/test_payload_io.py`.
3. Remove `PayloadController` and `PayloadDict` from `landseg.artifacts.__all__`.
4. Delete legacy `_meta.json` files generated in test fixtures and experiment outputs.

---

## 3. Consequences

- **Positive**:
  - Eliminates dual-file state hazards and folder clutter across `world_grids/` and `domains/`.
  - Simplifies `landseg.artifacts` to a single, canonical `Controller[T]` pattern.
  - Ensures runtime schema version verification across all data contracts.
- **Negative / Neutral**:
  - Reading metadata of large grids requires parsing the single JSON file (~15–30 ms), which is negligible for current scales and already mitigated by stage `report.json`.
- **Supersedes**:
  - Supersedes the split-file `PayloadController` design from [ADR-0014](./docs/ADRs/ADR-0014-decouple-artifacts-lifecycle-consistent-handling.md).
