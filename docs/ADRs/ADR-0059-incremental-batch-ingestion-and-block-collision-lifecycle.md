# ADR-0059: Incremental Batch Ingestion and Block Collision Lifecycle

**Status:** Proposed<br>
**Date:** 2026-09-23

---

## 1. Context

In operational remote sensing, ecological modeling, and forestry inventory pipelines, earth observation imagery and ground-truth survey layers do not arrive as a single monolithic dataset. Instead, spatial data arrives incrementally in discrete batches:
- Periodic flight lines and airborne LiDAR missions covering regional swaths.
- Incremental satellite passes (e.g., seasonal Sentinel-2 cloud-free composites).
- Ongoing field plot inventories and regional land cover update packages.

The existing geospatial ETL pipeline (`landseg.geopipe`) currently implements a decoupled 4-stage lifecycle:
1. `world-grid`: Builds and persists the canonical spatial tiling layout (`GridLayout`) defined by tile dimensions and stride.
2. `data-harmonize`: Warps, resamples, and stacks raw continuous and categorical rasters onto the world grid, producing multi-band VRTs and valid pixel masks. Each execution is recorded as an isolated run folder (`harmonized_data/run_0001`, `run_0002`, etc.) with its own `harmonize_report.json`.
3. `data-ingest`: Slices harmonized rasters into canonical data blocks (`.npz` tensors), produces domain knowledge maps, and records dataset metadata (`catalog.json`, `schema.json`).
4. `data-prepare`: Dynamically partitions the canonical block pool into train, validation, and test splits (via ratio scoring or geographic AOI masks), computes split-isolated normalization statistics, and emits experiment datasets.

### Identified Bottlenecks in Data Ingestion

While `data-harmonize` operates per batch run (`run_xxx`), `data-ingest` was originally architected as a static, single-execution pipeline:

1. **Absence of Ingestion Run History**:
   `data-ingest` writes its summary report directly to `ingested_data/ingest_report.json`. Repeated executions overwrite this report and erase historical telemetry, preventing teams from auditing which harmonization batches were ingested, when they were added, and how the canonical pool grew over time.

2. **Lack of an Incremental Pooling Mental Model**:
   Ingestion treats its target directory as a static destination rather than an accumulating canonical block pool. When a new harmonization batch arrives, there is no standardized protocol for ingesting `run_xxx` into the existing pool while recording the operation as ingestion `run_yyy`.

3. **Uncontrolled Spatial Tile Collisions**:
   Because batches often cover adjacent or overlapping areas (e.g., overlapping flight lines, re-surveyed districts, or multi-temporal imagery over the same region), tiles generated from an incoming batch frequently share the identical grid coordinate `(row, col)` with blocks already present in the canonical pool.
   Currently, collision handling is implicitly tied to the global `rebuild: bool` flag:
   - If `rebuild: false` (`LifecyclePolicy.BUILD_IF_MISSING`), any valid block file existing on disk is silently skipped without recording collision metrics.
   - If `rebuild: true` (`LifecyclePolicy.REBUILD`), all candidate blocks from the batch are written to disk, unconditionally overwriting existing blocks and discarding previous data.
   The framework lacks an explicit, configurable collision rule (`skip`, `overwrite`, or `error`) allowing users to declare intended behavior when incoming tiles intersect with the incumbent block pool.

4. **Missing Provenance in Cumulative Catalogs**:
   The catalog `catalog.json` records raster source hashes, but does not explicitly track batch lineage (i.e., which harmonization run `run_xxx` and ingestion run `run_yyy` produced or last updated each block).

---

## 2. Decision

We will evolve `landseg.geopipe` to support **incremental batch ingestion into a unified canonical block pool** anchored to a **global static grid**, with **first-class ingestion run tracking** and an **explicit collision policy**.

### 2.1. Architectural Mental Model

```text
+-------------------------------------------------------------------------------+
|                             GLOBAL STATIC GRID                                |
|             Defined once via world-grid (e.g. grid_row_256_col_256)           |
+---------------------------------------+---------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                                                |
       v                                                                v
+-------------------------------+                     +-------------------------------+
|  Harmonization Batch 1        |                     |  Harmonization Batch 2        |
|  harmonized_data/run_0001/    |                     |  harmonized_data/run_0002/    |
+---------------+---------------+                     +---------------+---------------+
                |                                                     |
                v                                                     v
+-------------------------------+                     +-------------------------------+
|  Ingestion Run 1 (run_0001)   |                     |  Ingestion Run 2 (run_0002)   |
|  - Ingests: harmonize run_0001|                     |  - Ingests: harmonize run_0002|
|  - Target: canonical pool     |                     |  - Target: canonical pool     |
|  - Collision rule: skip       |                     |  - Collision rule: overwrite  |
+---------------+---------------+                     +---------------+---------------+
                |                                                     |
                +-----------------------+-----------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                             CANONICAL BLOCK POOL                              |
|  ingested_data/                                                               |
|  |-- data_blocks/                                                             |
|  |   |-- blocks/                   <-- Cumulative .npz pool (all batches)     |
|  |   |   |-- row_000000_col_000000.npz                                        |
|  |   |   `-- row_000000_col_000256.npz                                        |
|  |   |-- catalog.json              <-- Cumulative metadata & run provenance   |
|  |   `-- schema.json               <-- Dataset schema specification           |
|  `-- runs/                         <-- Historical ingestion run audits        |
|      |-- run_0001/ingest_report.json                                          |
|      `-- run_0002/ingest_report.json                                          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                           data-prepare (Downstream)                           |
|       Consumes accumulated pool & catalog seamlessly without modification     |
+-------------------------------------------------------------------------------+
```

### 2.2. Dual-Layer Directory Layout in `IngestionPaths`

We will update `landseg.artifacts.paths.IngestionPaths` to distinguish between the **persistent canonical pool** and **incremental execution runs**:

1. **Persistent Pool (`data_blocks/`, `domain_knowledge/`)**:
   - `data_blocks/blocks/`: The single accumulated pool of `.npz` data blocks.
   - `data_blocks/catalog.json`: The cumulative catalog mapping coordinates `(row, col)` to metadata for all blocks currently active in the pool.
   - `data_blocks/schema.json`: Unified dataset schema.
   - Retaining this canonical layout guarantees backwards compatibility: downstream `data-prepare` and `PreparationPaths` will continue reading `artifact_paths.data_ingestion.data_blocks.catalog` without breaking changes.

2. **Execution Runs (`runs/run_XXXX/`)**:
   - `runs/run_0001/ingest_report.json`, `runs/run_0001/config.json`: Per-run telemetry recording the specific batch execution, consumed harmonization run ID, runtime duration, and collision statistics.
   - `IngestionPaths` will implement `init()` and `get_run_folder()` analogous to `HarmonizationPaths`, automatically generating incremental run directories (`run_0001`, `run_0002`, ...).
   - The latest run report will additionally be copied/referenced at `ingested_data/ingest_report.json` as a convenience pointer for top-level inspections.

### 2.3. Explicit Collision Rule Specification

We will introduce `collision_rule` into the ingestion configuration, formalizing three distinct strategies when an incoming tile matches a grid coordinate `(row, col)` already present in the block pool:

| Collision Rule | Action on Colliding Block File | Action in Cumulative Catalog | Failure Mode / Use Case |
| :--- | :--- | :--- | :--- |
| `skip` *(default)* | Preserve incumbent `.npz` file; do not re-slice. | Retain incumbent block metadata; ignore incoming tile. | Incremental non-destructive expansion (new flight lines / tiles). |
| `overwrite` | Replace existing `.npz` block with newly sliced data. | Update entry with new raster hashes, timestamp, and run provenance. | Re-surveys, updated calibration, or higher-quality raster refreshes. |
| `error` | No blocks are written; pipeline aborts immediately. | Catalog remains unchanged. | Strict data auditing; prevents unexpected geographic overlap. |

### 2.4. Configuration Surface Evolution

We will extend `_IngestionCfg` in `landseg.configs.schema.sections.data` and `configs/user.yaml`:

```yaml
data-ingest:
  output_dpath: ./experiment/artifacts/ingested_data
  # Targeted harmonization run folder index, name, or path (null for latest)
  harmonization_run: null
  # Explicit policy when an incoming block collides with an existing block in the pool:
  # Supported: [skip, overwrite, error]
  collision_rule: skip
  # If true, forces rebuilding all candidate blocks in this batch
  rebuild: false
```

- When `rebuild: false`, the incoming batch is processed incrementally according to `collision_rule`.
- When `rebuild: true`, all candidate blocks from the incoming batch are regenerated regardless of pre-existing state.

### 2.5. Assembler & Lifecycle Enhancements

We will enhance `landseg.geopipe.ingest.blocks.assembler.lifecycle`:

1. **Spatial Collision Detection**:
   Prior to scheduling parallel block generation jobs, `_prepare_block_windows` and `_structural_validation` will intersect candidate coordinates from the incoming batch with the coordinates already present on disk:
   $$\mathcal{C}_{\text{candidate}} = \{(r, c) \in \text{Batch Windows}\}$$
   $$\mathcal{C}_{\text{existing}} = \{(r, c) \in \text{Valid Blocks in Pool}\}$$
   $$\mathcal{C}_{\text{collide}} = \mathcal{C}_{\text{candidate}} \cap \mathcal{C}_{\text{existing}}$$
   $$\mathcal{C}_{\text{new}} = \mathcal{C}_{\text{candidate}} \setminus \mathcal{C}_{\text{existing}}$$

2. **Policy Enforcement**:
   - `collision_rule == 'error'`: If $\mathcal{C}_{\text{collide}} \neq \emptyset$, raise an `artifacts.ArtifactError` detailing the colliding coordinates and source rasters.
   - `collision_rule == 'skip'`: Build $\mathcal{C}_{\text{new}}$. Leave $\mathcal{C}_{\text{collide}}$ untouched.
   - `collision_rule == 'overwrite'`: Build $\mathcal{C}_{\text{new}} \cup \mathcal{C}_{\text{collide}}$. Overwrite disk files for $\mathcal{C}_{\text{collide}}$.

3. **Telemetry & Execution Metrics**:
   Update `BlockStats` and `IngestReportSchema` in `landseg.geopipe.contracts.ingestion` to report:
   - `blocks_candidate`: Total valid windows in incoming batch.
   - `blocks_collided`: Count of overlapping coordinates encountered.
   - `blocks_skipped`: Count of colliding blocks retained from incumbent pool.
   - `blocks_overwritten`: Count of colliding blocks replaced by new batch.
   - `blocks_added`: Count of newly created blocks added to pool.
   - `total_pool_blocks`: Total active blocks in the cumulative catalog after ingestion.

### 2.6. Catalog Lineage & Provenance Tracking

We will extend `DatasetBlockMeta` in `landseg.geopipe.core.dataset_catalog` and `manifest/catalog.py`:
- Add `harmonize_run_id: str` (e.g. `'run_0002'`).
- Add `ingest_run_id: str` (e.g. `'run_0001'`).
- In `build_catalog`, when `collision_rule == 'overwrite'`, update the colliding entries with the new batch's SHA-256 and lineage tags. When `collision_rule == 'skip'`, preserve incumbent entries.

### 2.7. Pipeline Orchestration & Upstream Verification

We will update `landseg.execution.executor._validate_upstream_pipelines`:
- Verify that the target harmonization run (`run_xxx`) completed with `status: "SUCCESS"` and shares the same canonical `world_grid` GID as the existing pool. If GIDs mismatch, fail fast before writing to the pool.
- Replace the legacy prompt popup with a clean check against ingestion run history: log an informational note if the target harmonization run has already been ingested into the pool previously.

---

## 3. Consequences

### Positive
- **Real-World Operational Alignment**: Ingestion natively matches remote sensing workflows where data arrives incrementally in multi-temporal or multi-regional batches.
- **Single Global Spatial Anchor**: The global static grid ensures all batches align seamlessly on the same tile boundaries without spatial drift or resampling artifacts.
- **Unambiguous Conflict Resolution**: Teams can choose between non-destructive expansion (`skip`), data updating (`overwrite`), or strict segregation (`error`).
- **Full Traceability & Provenance**: Every block in the canonical pool tracks the exact harmonization run and ingestion run that created or updated it.
- **Zero Downstream Disruption**: `data-prepare` continues consuming the unified canonical pool (`data_blocks/catalog.json`, `data_blocks/blocks/`) with zero changes to dataset partitioning, scoring, or normalization.

### Negative / Considerations
- **Catalog Management Overhead**: Large accumulated catalogs with tens of thousands of tiles require efficient JSON serialization and atomic updates to avoid corruption during concurrent runs.
- **Storage Growth**: As multiple batches are ingested into the pool, storage requirements expand; pruning or archiving legacy runs will require explicit maintenance tools.
