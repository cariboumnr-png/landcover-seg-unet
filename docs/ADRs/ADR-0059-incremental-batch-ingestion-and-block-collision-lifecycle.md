# ADR-0059: Incremental Batch Ingestion and Block Collision Lifecycle

**Status:** Accepted (Phases 1 & 2 Implemented; Phases 3 & 4 Deferred)<br>
**Date:** 2026-09-23 (Updated 2026-09-26)

---

## 1. Context

In operational remote sensing, ecological modeling, and forestry inventory pipelines, earth observation imagery and ground-truth survey layers do not arrive as a single monolithic dataset. Instead, spatial data arrives incrementally in discrete batches:
- Periodic flight lines and airborne LiDAR missions covering regional swaths.
- Incremental satellite passes (e.g., seasonal Sentinel-2 cloud-free composites).
- Ongoing field plot inventories and regional land cover update packages.

The existing geospatial ETL pipeline (`landseg.geopipe`) implements a decoupled 4-stage lifecycle:
1. `world-grid`: Builds and persists the canonical spatial tiling layout (`GridLayout`) defined by tile dimensions and stride.
2. `data-harmonize`: Warps, resamples, and stacks raw continuous and categorical rasters onto the world grid, producing multi-band VRTs and valid pixel masks. Each execution is recorded as an isolated run folder (`harmonized_data/run_0001`, `run_0002`, etc.) with its own `harmonize_report.json` and tracked in `harmonization_runs.json`.
3. `data-ingest`: Slices harmonized rasters into canonical data blocks (`.npz` tensors), produces domain knowledge maps, and records dataset metadata (`catalog.json`, `schema.json`).
4. `data-prepare`: Dynamically partitions the canonical block pool into train, validation, and test splits (via ratio scoring or geographic AOI masks), computes split-isolated normalization statistics, and emits experiment datasets.

### Identified Bottlenecks in Data Ingestion

While `data-harmonize` operates per batch run (`run_xxx`), `data-ingest` was originally architected as a static, single-execution pipeline:

1. **Absence of Ingestion Run History**:
   `data-ingest` wrote its summary report directly to `ingested_data/ingest_report.json`. Repeated executions overwrote this report and erased historical telemetry, preventing teams from auditing which harmonization batches were ingested, when they were added, and how the canonical pool grew over time.

2. **Lack of an Incremental Pooling Mental Model**:
   Ingestion treated its target directory as a static destination rather than an accumulating canonical block pool. When a new harmonization batch arrived, there was no standardized protocol for ingesting `run_xxx` into the existing pool while recording the operation as ingestion `run_yyy`.

3. **Conflation of Affine Warping with Block Pool Compatibility**:
   Early designs assumed a single monolithic global grid for all operations. In practice, raster harmonization only requires coordinate reference, origin, and pixel resolution alignment (`affine_identity = crs|origin|pixel_size`), whereas canonical block pooling additionally requires identical tile dimensions (`block_identity = affine_identity|tile_size`). Sampling stride and bounding extent are local run parameters (sampling density and regional AOI) and should not prevent multi-regional batches from accumulating into the same pool.

4. **Uncontrolled Spatial Tile Collisions**:
   Because batches often cover adjacent or overlapping areas (e.g., overlapping flight lines, re-surveyed districts, or multi-temporal imagery over the same region), tiles generated from an incoming batch frequently share the identical grid coordinate `(row, col)` with blocks already present in the canonical pool.
   The framework lacks an explicit, configurable collision rule (`skip`, `overwrite`, or `error`) allowing users to declare intended behavior when incoming tiles intersect with the incumbent block pool.

5. **Missing Provenance in Cumulative Catalogs**:
   The catalog `catalog.json` records raster source hashes, but does not explicitly track batch lineage (i.e., which harmonization run `run_xxx` and ingestion run `run_yyy` produced or last updated each block).

---

## 2. Decision

We evolve `landseg.geopipe` to support **incremental batch ingestion into a unified canonical block pool** anchored to a **decoupled spatial frame**, with **first-class ingestion run ledgers**, **automatic batch catch-up**, and an **explicit collision policy**.

Implementation is structured across four phases:
- **Phase 1 (Implemented)**: Ingestion run ledger (`ingestion_runs.json`), run-level collision detection (`fingerprint`), automatic batch catch-up resolution, spatial identity decoupling (`affine_identity` vs `block_identity`), and pool grid compatibility verification.
- **Phase 2 (Implemented)**: Fine-grained intra-pool block collision policy (`CollisionPolicy: [skip, overwrite, error]`), mandatory persistence of run collision manifests (`collisions.json`) recording all overlapping coordinates, and cumulative catalog lineage.
- **Phase 3 (Proposed / Deferred)**: Inter-batch boundary seam stitching and nodata-filling for partial border blocks bisected by regional swath seams (deferred to future work; enabled by Phase 2's `collisions.json`).
- **Phase 4 (Proposed / Deferred)**: Higher-level workflow orchestration (`landseg.execution.workflows`) decoupling multi-run and cross-pipeline workflows (e.g., `study-sweep`, end-to-end data intake, and pre-flight telemetry checks) from atomic execution pipelines.

### 2.1. Architectural Mental Model

```text
+-------------------------------------------------------------------------------+
|                       DECOUPLED SPATIAL FRAMES                                |
|  - Continuous Affine Frame (affine_identity: crs|origin|pixel_size)           |
|    Governs raster warping in data-harmonize. Invariant to tiles, stride, AOI. |
|  - Discrete Block Tensor Frame (block_identity: affine_identity|tile_size)    |
|    Governs canonical block pool. Invariant to stride and extent.              |
+---------------------------------------+---------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                                                |
       v                                                                v
+-------------------------------+                     +-------------------------------+
|  Harmonization Batch 1        |                     |  Harmonization Batch 2        |
|  harmonized_data/run_0001/    |                     |  harmonized_data/run_0002/    |
|  - Recorded in runs manifest  |                     |  - Recorded in runs manifest  |
|  - Status: SUCCESS / SKIPPED  |                     |  - Status: SUCCESS / SKIPPED  |
+---------------+---------------+                     +---------------+---------------+
                |                                                     |
                v                                                     v
+-------------------------------+                     +-------------------------------+
|  Ingestion Run 1 (run_0001)   |                     |  Ingestion Run 2 (run_0002)   |
|  - Ingests: harmonize run 1   |                     |  - Ingests: harmonize run 2   |
|  - UID: ingest_<uuid>         |                     |  - UID: ingest_<uuid>         |
|  - Ledger: ingestion_runs.json|                     |  - Ledger: ingestion_runs.json|
+---------------+---------------+                     +---------------+---------------+
                |                                                     |
                +-----------------------+-----------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                             CANONICAL BLOCK POOL                              |
|  ingested_data/                                                               |
|  |-- ingestion_runs.json           <-- Cumulative ledger (lineage, SHA-256)   |
|  |-- run_0001/ingest_report.json   <-- Run 1 telemetry & metrics             |
|  |-- run_0002/ingest_report.json   <-- Run 2 telemetry & metrics             |
|  |-- data_blocks/                                                             |
|  |   |-- blocks/                   <-- Cumulative .npz pool (all batches)     |
|  |   |   |-- row_000000_col_000000.npz                                        |
|  |   |   `-- row_000000_col_000256.npz                                        |
|  |   |-- catalog.json              <-- Cumulative metadata & run provenance   |
|  |   `-- schema.json               <-- Dataset schema with block_identity     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                           data-prepare (Downstream)                           |
|       Consumes accumulated pool & catalog seamlessly without modification     |
+-------------------------------------------------------------------------------+
```

### 2.2. Dual-Layer Directory Layout & Ledger Architecture (Phase 1 Implemented)

`landseg.artifacts.paths.IngestionPaths` manages both the **accumulating canonical pool** and **isolated run telemetry**:

1. **Persistent Canonical Pool (`data_blocks/`, `domain_knowledge/`)**:
   - `data_blocks/blocks/`: The accumulated pool of `.npz` data blocks.
   - `data_blocks/catalog.json`: The cumulative catalog mapping coordinates `(row, col)` to metadata for all active blocks.
   - `data_blocks/schema.json`: Unified dataset schema recording `dataset.block_identity`.
   - Downstream `data-prepare` continues consuming `data_blocks/catalog.json` without changes.

2. **Execution Runs & Manifest Ledger**:
   - `ingested_data/ingestion_runs.json`: Ledger tracking all ingestion runs:
     ```json
     {
       "ingest_1a2b3c4d5e6f7890": {
         "run_uid": "ingest_1a2b3c4d5e6f7890",
         "run_id": "run_0001",
         "harmonization_run_uid": "harmonize_fedcba0987654321",
         "harmonization_run_id": "run_0001",
         "run_folder": ".../ingested_data/run_0001",
         "timestamp": "2026-09-24T18:00:00Z",
         "fingerprint": "a1b2c3...",
         "status": "SUCCESS"
       }
     }
     ```
   - `ingested_data/run_XXXX/`: Run-isolated telemetry, logs, and `ingest_report.json`.

### 2.3. Spatial Identity Decoupling & Pool Grid Verification (Phase 1 Implemented)

`landseg.geopipe.core.grid_layout.GridLayout` exposes two canonical identity properties:
- `affine_identity`: `f'{self.crs}|{self.origin}|{self.pixel_size}'`.
  Determines continuous affine grid alignment for raster warping.
- `block_identity`: `f'{self.affine_identity}|{self.tile_size}'`.
  Determines discrete block tensor compatibility in the canonical block pool.

Before ingesting any batch into an existing block pool:
- `verify_pool_grid_compatibility(schema_fpath, incoming_grid)` compares `incoming_grid.block_identity` against `dataset.block_identity` stored in `schema.json`.
- Mismatches raise an `artifacts.ArtifactError`, preventing grid misalignment and tensor dimension corruption.
- Different regional batches (e.g., Ottawa vs Sudbury) with distinct geographic bounding extents or sampling strides can safely populate the same block pool as long as `block_identity` matches.

### 2.4. Upstream Dynamic Batch Catch-Up & Run Idempotence (Phase 1 Implemented)

1. **Automatic Batch Resolution (`resolve_pending_ingestion_batches`)**:
   - Discovers completed `SUCCESS` runs from `harmonized_data/harmonization_runs.json`.
   - Matches against existing `SUCCESS` runs in `ingested_data/ingestion_runs.json`.
   - Supports four operational modes:
     * `target=None` or `'pending'`: Auto-ingests all uningested harmonization batches in chronological sequence (catch-up mode).
     * `target='latest'`: Ingests only the most recent successful harmonization run.
     * `target=<id>`: Targets a specific run index, folder name, or UID.
     * `rebuild=True`: Forces re-ingestion of already-processed batches.

2. **Run-Level Collision Avoidance**:
   - Both `data-harmonize` and `data-ingest` compute a deterministic SHA-256 fingerprint of inputs, grid identity, and execution configs.
   - If an identical run exists with status `SUCCESS`, the execution is marked `SKIPPED`, avoiding redundant compute and duplicate ledger records.

### 2.5. Intra-Pool Block Collision Policy (Phase 2 Implemented)

When incoming tiles from different batches intersect on grid coordinates `(row, col)` already present in the block pool:
$$\mathcal{C}_{\text{candidate}} = \{(r, c) \in \text{Batch Windows}\}$$
$$\mathcal{C}_{\text{existing}} = \{(r, c) \in \text{Valid Blocks in Pool}\}$$
$$\mathcal{C}_{\text{collide}} = \mathcal{C}_{\text{candidate}} \cap \mathcal{C}_{\text{existing}}$$

We formalized an explicit `CollisionPolicy` enum in `landseg.geopipe.contracts.ingestion`:
- `SKIP` (`'skip'`, default): Preserve incumbent `.npz` file; do not re-slice. Retain incumbent block metadata in `catalog.json`. Ideal for non-destructive incremental pooling.
- `OVERWRITE` (`'overwrite'`): Replace existing `.npz` block with newly sliced data. Update catalog with new raster hashes, timestamps, and lineage tags. Ideal for re-surveys and recalibrated rasters.
- `ERROR` (`'error'`): Abort execution immediately before writing any blocks if $\mathcal{C}_{\text{collide}} \neq \emptyset$. Enforces strict disjoint geographic partitioning.

### 2.6. Mandatory Run Collision Manifest (`collisions.json`) (Phase 2 Implemented)

Regardless of which `CollisionPolicy` is active (`skip`, `overwrite`, or `error`), every ingestion run generates and persists a dedicated collision manifest at `ingested_data/run_XXXX/collisions.json`:

```json
{
  "ingestion_run_id": "run_0002",
  "ingestion_run_uid": "ingest_1a2b3c4d5e6f7890",
  "harmonization_run_id": "run_0002",
  "collision_policy": "skip",
  "total_collided": 42,
  "collided_blocks": [
    {
      "block_name": "row_000512_col_001024",
      "grid_coord": [512, 1024],
      "incumbent_ingest_run": "run_0001",
      "incumbent_harmonize_run": "run_0001",
      "action_taken": "skipped"
    }
  ]
}
```

This permanent audit artifact serves two critical roles:
1. **Auditability & Anomaly Diagnosis**: Downstream training experiments encountering border anomalies or high loss along swath edges can cross-reference block coordinates directly against the collision ledger.
2. **Staging Index for Future Stitching**: Provides an exact, pre-filtered index of overlapping tiles, eliminating the need to brute-force scan hundreds of thousands of pool tensors when performing mosaic reconciliation.

High-level collision counters (`blocks_candidate`, `blocks_collided`, `blocks_skipped`, `blocks_overwritten`, `blocks_added`) are reported in `ingest_report.json`.

### 2.7. Catalog Lineage & Provenance Tracking (Phase 2 Implemented)

We extended `DatasetBlockMeta` in `landseg.geopipe.core.dataset_catalog` and `manifest/catalog.py`:
- Added `harmonize_run_id: str` (e.g. `'run_0002'`).
- Added `ingest_run_id: str` (e.g. `'run_0001'`).
- In `build_catalog`, when `collision_policy == OVERWRITE`, update the colliding entries with the new batch's SHA-256 and lineage tags. When `collision_policy == SKIP`, preserve incumbent entries.

### 2.8. Pipeline Orchestration & Upstream Verification (Phase 1 Implemented)

`landseg.execution.pipelines.data_ingest.exec_ingest_data`:
- Discovers upstream harmonization runs and matches against the downstream ledger.
- Executes pending batches sequentially with individual progress reporting.
- Exits cleanly as a no-op when all batches are up-to-date.

### 2.9. Proposed Phase 3: Border Seam Dilemma and Stitching Deferral

A subtle but critical reality of incremental geospatial data arrival is the **border seam dilemma**:
- When Batch A terminates along a regional acquisition boundary (e.g. flight swath edge or district boundary) that bisects discrete grid tile $(r, c)$, Batch A ingests a **"partial block"** containing valid sensor measurements on one side and `nodata` / background padding on the other.
- When adjacent Batch B subsequently arrives, tile $(r, c)$ contains the complementary data for the opposite half of the tile.
- Under `SKIP`, Batch A's partial tile is preserved and Batch B's valid data is discarded.
- Under `OVERWRITE`, Batch B replaces the block entirely and Batch A's valid data is discarded.

Resolving this boundary condition requires a dedicated **block stitching / mosaic blending pass** that loads both tensors and merges valid pixels where the other contains `nodata`.

**Explicit Deferral Decision**:
We explicitly propose to **defer block stitching to Phase 3**. Attempting to implement multi-band tensor blending, categorical label conflict resolution, and nodata-filling within the current branch would significantly inflate complexity and destabilize core pooling invariants. Because Phase 2 guarantees that all colliding coordinates are immutably cataloged in `collisions.json`, Phase 3 can be introduced later as a clean, decoupled post-processing consolidation utility without disrupting the canonical ingestion contract.

### 2.10. Proposed Phase 4: Higher-Level Workflow Orchestration (`landseg.execution.workflows`)

A structural insight emerging from Phase 1 and 2 implementation is the architectural tension between **atomic pipeline execution** and **higher-level multi-run orchestration**:
- In `landseg.execution.pipelines`, atomic commands (`world-grid`, `data-harmonize`, `data-prepare`, `model-train`, `model-evaluate`) strictly maintain a $1$-to-$1$ relationship with their execution context: $1$ invocation $\rightarrow$ $1$ run folder, $1$ report, and $1$ manifest entry.
- Conversely, operations such as `data-ingest` (which iterates over an unbounded queue of pending harmonization batches) and `study-sweep` (which orchestrates multiple hyperparameter search trials using Optuna) represent **higher-level workflows** that sit above atomic pipeline execution.

To resolve this dichotomy without bloating atomic execution pipelines, Phase 4 will formalize a dedicated `landseg.execution.workflows` module:

1. **Decoupled Workflow Hierarchy**:
   - `landseg.execution.pipelines.*`: Remains strictly scoped to atomic, single-run execution units. `exec_ingest_data` will be simplified to execute an explicit, single-batch ingestion target.
   - `landseg.execution.workflows.*`: Hosts multi-run iterators, composite multi-stage DAGs, and search orchestration.

2. **Core Workflow Candidates**:
   - `study_sweep.py`: Relocated from `pipelines/` into `workflows/` as a dedicated multi-session optimization workflow.
   - `batch_ingest_workflow`: Orchestrates comprehensive data intake (e.g., upstream harmonization run discovery $\rightarrow$ sequential batch catch-up $\rightarrow$ pool verification).
   - `end_to_end_intake_workflow`: Manages continuous ingestion by sequentially invoking `data-harmonize` on incoming raw source packages followed by incremental `data-ingest` into the block pool.

3. **Pre-flight Telemetry & Context Validation**:
   - `workflows/` will implement a unified pre-flight inspection engine.
   - Before launching heavyweight GPU/CPU compute, the workflow layer can query each downstream and upstream pipeline's context in dry-run mode:
     * Verifying grid and schema compatibility.
     * Identifying pending vs. completed runs across all ledgers.
     * Detecting un-harmonized raw inputs or invalid file paths.
     * Emitting an executive status-quo dashboard summarizing pipeline readiness.

**Explicit Deferral Decision**:
Phase 2 implementation remains focused on finalizing intra-pool block collisions, `collisions.json`, and catalog lineage within the existing pipeline surface. Workflow decoupling and the creation of `landseg.execution.workflows` are formally deferred to Phase 4 on a dedicated branch.

---

## 3. Consequences

### Positive
- **Real-World Operational Alignment**: Ingestion natively matches remote sensing workflows where data arrives incrementally in multi-temporal or multi-regional batches.
- **Single Global Spatial Anchor**: The global static grid ensures all batches align seamlessly on the same tile boundaries without spatial drift or resampling artifacts.
- **Unambiguous Conflict Resolution**: Teams can choose between non-destructive expansion (`skip`), data updating (`overwrite`), or strict segregation (`error`).
- **Full Traceability & Provenance**: Every block in the canonical pool tracks the exact harmonization run and ingestion run that created or updated it.
- **Zero Downstream Disruption**: `data-prepare` continues consuming the unified canonical pool (`data_blocks/catalog.json`, `data_blocks/blocks/`) with zero changes to dataset partitioning, scoring, or normalization.
- **Clean Architectural Scaling**: Segregating atomic pipeline runners from multi-run workflows ensures that `execution.pipelines` remains simple and uniform, while complex workflows (sweeps, continuous intake, pre-flight checks) have a dedicated, testable home.

### Negative / Considerations
- **Catalog Management Overhead**: Large accumulated catalogs with tens of thousands of tiles require efficient JSON serialization and atomic updates to avoid corruption during concurrent runs.
- **Storage Growth**: As multiple batches are ingested into the pool, storage requirements expand; pruning or archiving legacy runs will require explicit maintenance tools.
