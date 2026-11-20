# ADR-0068: Cloud Raster I/O, Materialization, and Execution Hardening

**Status:** Accepted — Implemented<br>
**Date:** 2026-11-19 (Updated 2026-11-20)<br>
**Branch:** `cloud-raster-io`<br>
**Related:**
[ADR-0066](ADR-0066-production-notebook-architecture-and-databricks-integration.md),
[ADR-0067](ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md)<br>
**Issues:** #88, #89, #90, #91

---

## 1. Context & Problem Statement

The introduction of decoupled stage paths in
[ADR-0067](ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md) and
the enterprise notebook runtime in
[ADR-0066](ADR-0066-production-notebook-architecture-and-databricks-integration.md)
enabled the `landseg` pipeline to be deployed on Databricks clusters backed by
Unity Catalog (UC) Volumes.

However, during initial deployment and validation with **real production
datasets** (multi-band Sentinel-2 L2A satellite imagery, 20 m DEM/HAG, and
multi-head landcover label rasters across real geographic Areas of Interest),
the system encountered a cascade of operational failures and performance
degradations that were completely masked when running synthetic "dummy data"
on local developer workstations.

### 1.1. The Performance Paradox: Local Hardware vs. Cloud Production

A critical discovery during testing was that **even with real-world production
data** (e.g., ~5 million pixels, 22 channels, 3 raw heads across 361 blocks),
execution on a local developer laptop (e.g., Intel i7-12700H, NVMe) remained
sub-second and built blocks at several per second.

The performance collapse on Databricks was therefore **not caused by a
difference between synthetic dummy data and real data**. Rather, the local
laptop's hardware profile accidentally masked severe architectural and I/O
inefficiencies that collapsed immediately under cloud conditions:

1. **NVMe PCIe Bandwidth vs. Cloud FUSE Latency**:
   * Local NVMe storage operates at 3,000–7,000 MB/s with sub-millisecond
     random-access latency.
   * Databricks Unity Catalog Volumes are remote object-store buckets mounted
     via a user-space FUSE filesystem (streaming at ~124 MB/s sequentially,
     with high round-trip latency for small random window reads).
   * Reprojecting lazily on every read via `WarpedVRT` and reading 15 bands
     individually from pixel-interleaved files was tolerated on local NVMe
     because the local disk bus and kernel page cache brute-forced through the
     redundant I/O. On cloud FUSE, every redundant band read translates into
     network round-trips.

2. **Core Count Asymmetry & Worker Starvation**:
   * On a 20-thread developer laptop, the historical concurrency formula
     `CORE_N = max(1, cpu_count() - 4)` yielded **16 parallel worker
     processes**.
   * On a standard 4-vCPU cloud node (2 physical cores), the exact same
     formula evaluated to `max(1, 4 - 4) = 1` worker.
   * Consequently, block creation that took a couple of minutes across 16
     parallel workers locally was serialized into a single process on the
     cluster, magnifying the per-block read latency by a factor of 16.

3. **Filesystem Write Semantics**:
   * Local POSIX filesystems support arbitrary write-seeking.
   * Cloud FUSE filesystems do not support backward-seeking on open file
     handles, causing `_tiffSeekProc: Input/output error` and truncated files
     during standard GeoTIFF and ZIP/NPZ writing.

4. **Hardware & Numerical Incompatibilities Under Real Datasets**:
   * Cloud CPU instances lacking hardware BF16 instructions (AVX-512 / AMX)
     fell back to slow software emulation or diverged under PyTorch AMP.
   * Real satellite reflectance values introduced unnormalized extremes that
     caused non-finite activations previously masked by silent `nan_to_num`.
   * Real hierarchical label distributions exposed mathematical flaws in
     singleton child IoU evaluation when parent predictions were missed.

---

## 2. Evolution & Structural Decisions

### 2.1. Preliminary Hotfixes

To unblock cluster execution, four targeted fixes were initially applied to
address hard crashes and correctness bugs:

1. **`75a466e` (Issue #88 - Hardware & Numerical Hardening)**:
   * Removed silent `nan_to_num` masking in `HeadManager` to fail fast on
     non-finite activations.
   * Added consecutive skipped-step guards in `MultiHeadTrainer` against
     runaway `GradScaler` bypass.
   * Added CPU BF16 capability detection and fallback to FP32 on CPUs
     lacking hardware acceleration.
2. **`29d61dd` (Hierarchical Evaluation & Grid Contiguity)**:
   * Corrected `ConfusionMatrix` and `BatchEngine` to evaluate child heads
     strictly against actual parent predictions (`parent_pred_1b`), properly
     penalizing missed parents as False Negatives.
   * Fixed independent coordinate bounds and full-grid assertions in
     `__calc_test_grid()` for non-contiguous AOI blocks.
3. **`c69b151` (Issue #89 - Local-First Mask GTiff Writing)**:
   * Introduced local-first writing in `geopipe/rasters/mask.py`, writing to a
     temporary directory before copying sequentially to the target Volume to
     eliminate FUSE write-seek truncation.
4. **`2324016` (Issue #90 - Sequential NPZ Block Streaming)**:
   * Implemented sequential in-memory archive streaming in
     `geopipe.ingest.blocks.assembler.lifecycle` to bypass FUSE
     backward-seek limitations on `.npz` block writes.

### 2.2. Architectural Foundation Established on `main`

While the hotfixes resolved immediate data corruption and crashes, addressing
Issue #91 (raster I/O throughput) required introducing fast local scratch
staging and scaled execution concurrency.

Before implementing raster materialization, the team recognized an architectural
risk: `landseg.utils` was drifting toward becoming an untyped dumping ground for
heterogeneous responsibilities (multiprocessing, device probing, logging,
notebook detection, widgets). Adding more utilities (e.g., scratch managers) to
`utils` would exacerbate domain bloat.

To establish clean architectural boundaries, three foundational commits were
implemented and merged directly onto `main`:

1. **`a14af30` — First-Class `landseg.runtime` Subsystem**:
   * Consolidated execution environment concerns into a first-class subsystem:
     `concurrency` (`ParallelExecutor`), `hardware` (capability checks),
     `platform` (environment detection), `interactive` (widgets and progress),
     and `logger` (unified base logger).
   * Upgraded concurrency worker calculation to
     `max(1, cpu_count() - 1)` with `LANDSEG_MAX_WORKERS` override, resolving
     the 4-vCPU worker starvation issue.
2. **`72538d6` — First-Class `landseg.artifacts.storage.scratch` Substrate**:
   * Introduced a centralized local scratch lifecycle under the `artifacts`
     subsystem: `get_scratch_dir()`, `scratch_file()`, `stage_to_scratch()`,
     and `cleanup_scratch()`.
   * Added `scratch_dir` property to `ExecutionContext` and `set_scratch()`
     builder method to `BaseConfigurator`, supporting environment override via
     `LANDSEG_SCRATCH_DIR` and falling back safely to `tempfile.gettempdir()`.
3. **`3302994` — Clean-Cut Deprecation of Legacy Shims**:
   * Completely removed backward-compatibility shims (`utils.device`,
     `utils.multip`, `utils.logger`, `mlops.env`, `notebooks.env`,
     `notebooks.widgets`).
   * Trimmed `landseg.utils` strictly to generic validation helpers
     (`validator.py`).
   * Relocated unit tests cleanly into `tests/unit/runtime/` and
     `tests/unit/artifacts/storage/`.

### 2.3. Scoping Branch `cloud-raster-io`

With the runtime and scratch substrates fully landed and tested on `main`,
the feature branch `cloud-raster-io` is sharply scoped to deliver the remaining
raster I/O optimizations under Issue #91:
1. Rebase `cloud-raster-io` onto `main` to align with `landseg.runtime` and
   `landseg.artifacts.storage.scratch`.
2. Introduce GDAL cloud runtime profiles in `landseg.geopipe.rasters.env`.
3. Implement materialized tiled GeoTIFF warping in `geopipe.rasters.warp`
   using `stage_to_scratch`.
4. Stream valid-pixel mask generation window-by-window in
   `geopipe.rasters.mask`.
5. Optimize block ingestion reads (single multi-band reads per window and
   selective boundless padding) in `geopipe.rasters.io`.
6. Align Databricks ingestion notebook with `set_scratch()` pointing to
   fast ephemeral NVMe storage (`/local_disk0`).

---

## 3. Architectural Decisions

```mermaid
flowchart TD
    subgraph Raw Layer
        RawVol["Raw Rasters on UC Volume (Remote Object Store)"]
    end

    subgraph Staging & Materialization (Phase 1 & 2)
        Scratch["Local Fast Scratch (artifacts.storage.scratch)"]
        WarpTif["Materialized GeoTIFFs (tiled 256x256, deflate)"]
        VolTif["Harmonized GeoTIFFs on UC Volume (Sequential Copy)"]
        AlignedVRT["harmonized_*_STACKED.vrt (SimpleSource, No Warping)"]
    end

    subgraph Downstream Ingestion & Masking (Phase 3 & 4)
        WinMask["Streaming Windowed Mask Generation (O(tile) RAM)"]
        WinBlock["Direct Aligned Block Window Reads (1 call / block)"]
        Workers["runtime.concurrency: max(1, cpu_count() - 1)"]
    end

    RawVol -->|Single sequential warp pass| Scratch
    Scratch -->|Bake metadata & tags| WarpTif
    WarpTif -->|Sequential copy avoids FUSE seek| VolTif
    VolTif --> AlignedVRT
    AlignedVRT --> WinMask
    AlignedVRT --> WinBlock
    Workers --> WinBlock
```

### Decision 1: Materialize Harmonized Rasters as Tiled GeoTIFFs
**Status:** In Progress on `cloud-raster-io`<br>
**Problem:** `WarpedVRT` recalculates coordinate reprojection, spatial
snapping, and bilinear/cubic interpolation on every single read. Stacking
multiple WarpedVRTs compounds this overhead across network storage.

**Decision:**
* We will discontinue writing virtual `VRTWarpedDataset` files for individual
  harmonized sources in `data-harmonize`.
* `geopipe.rasters.warp` will introduce `write_warped_gtiff()`. Every raw
  raster source will be warped **exactly once** into a materialized, tiled,
  compressed GeoTIFF on the target canonical grid:
  * Format: `driver='GTiff'`, `tiled=True`, `blockxsize=256`, `blockysize=256`,
    `compress='deflate'`.
  * Staged via `landseg.artifacts.storage.stage_to_scratch()`: generated on
    local scratch first, tagged with dataset/band metadata, and copied
    sequentially to the target UC Volume.
* The composite files (`harmonized_features_STACKED.vrt` and
  `harmonized_labels_STACKED.vrt`) will remain lightweight VRTs, but their
  `<SimpleSource>` elements will point directly to aligned GeoTIFFs. All
  runtime coordinate transformation and interpolation during ingestion are
  eliminated.

### Decision 2: Centralized Local Scratch Storage Substrate
**Status:** Accepted & Merged into `main` (`72538d6`)<br>
**Problem:** FUSE filesystems cannot support random-access writes. Ad-hoc
temporary directories were hardcoded across modules without centralized
configuration or lifecycle management.

**Decision:**
* Scratch lifecycle is centralized in `landseg.artifacts.storage.scratch`
  providing `get_scratch_dir()`, `scratch_file()`, `stage_to_scratch()`, and
  `cleanup_scratch()`.
* Resolution precedence:
  1. Explicit programmatic argument `scratch_dir`.
  2. Configuration setting `ExecutionContext.scratch_dir`.
  3. Environment variable `LANDSEG_SCRATCH_DIR` (e.g. `/local_disk0` on
     Databricks).
  4. System fallback: `tempfile.gettempdir()`.
* Fluent configuration support via `BaseConfigurator.set_scratch(path)`.
* Zero hardcoded cloud paths in library code; cloud paths remain in
  user-facing notebook / environment configuration.
* All temporary staging operations (warp outputs, mask generation, block NPZs)
  route through `landseg.artifacts.storage.scratch`.

### Decision 3: Streaming Windowed Valid-Pixel Mask Generation
**Status:** In Progress on `cloud-raster-io`<br>
**Problem:** `unify_nodata_mask` reads all 15 bands across the entire raster
into memory at once, forcing redundant decodes of interleaved S2 tiles and
causing high memory footprints.

**Decision:**
* `geopipe.rasters.mask.unify_nodata_mask` will be refactored into a streaming,
  windowed generator using `src.block_windows(1)`.
* For each block window, all bands will be read simultaneously in a single I/O
  pass (`src.read(window=window)`).
* The valid boolean mask will be computed per block and written directly to a
  tiled scratch GeoTIFF via `stage_to_scratch` before copying to destination.
* Memory usage will scale with tile size ($O(256 \times 256)$), not total
  geographic extent.

### Decision 4: Consolidated Multi-Band Block Reads & Selective Boundless I/O
**Status:** In Progress on `cloud-raster-io`<br>
**Problem:** Ingestion block creation calls `_read_bands` 15 times per block
with `boundless=True`, causing 15 separate network round-trips and boundless
padding overhead.

**Decision:**
* `geopipe.rasters.io._read_bands` will inspect the requested band dtypes:
  * When bands share the same dtype (e.g., all 15 feature bands as `float32`),
    rasterio will read all bands in a **single call**:
    `dataset.read(bands, window=window)`.
  * When bands have mixed dtypes, bands will be grouped by unique dtype,
    executing one read per dtype group.
* `edge_mode='boundless'` will be applied selectively: only windows whose
  coordinates actually intersect or exceed the raster boundary will set
  `boundless=True`. Interior windows will execute direct native block reads.

### Decision 5: First-Class Runtime Subsystem & Auto-Scaled Concurrency
**Status:** Accepted & Merged into `main` (`a14af30`, `3302994`)<br>
**Problem:** `CORE_N = max(1, cpu_count() - 4)` starved 4-vCPU cloud nodes down
to 1 worker, serializing ingestion. Furthermore, execution concerns were
scattered across `utils`, `mlops`, and `notebooks`.

**Decision:**
* Concurrency and execution primitives are elevated into `landseg.runtime`:
  * Worker count scales to
    `CORE_N = int(os.environ.get('LANDSEG_MAX_WORKERS',`
    `max(1, cpu_count() - 1)))`.
  * On 4-vCPU cloud nodes, worker count increases from 1 to 3 (a 300%
    throughput multiplier).
  * `ParallelExecutor` centralizes batch execution with worker lifecycle
    management.
* Hardware detection (`has_cpu_bf16_support()`) is centralized in
  `landseg.runtime.hardware`.
* Complete clean cut: legacy shims in `utils` were deleted; `utils` contains
  only pure validation helpers.

### Decision 6: Cloud-Optimized GDAL Runtime Profiles
**Status:** In Progress on `cloud-raster-io`<br>
**Problem:** Default GDAL settings trigger directory listings on every file
open, maintain an undersized 64 MB cache, and operate single-threaded.

**Decision:**
* Introduce `landseg.geopipe.rasters.env` with `configure_gdal_defaults()`:
  * `GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'`: Eliminates recursive cloud
    directory listing latency on every raster open.
  * `GDAL_NUM_THREADS='ALL_CPUS'`: Enables multi-threaded DEFLATE/LZW
    compression and decompression.
  * `GDAL_CACHEMAX='512MB'`: Expands block cache to prevent tile thrashing.

---

## 4. Consequences & Benefits

### 4.1. Positive Consequences
* **Throughput Multipliers**:
  * Mask generation duration drops from 2 min 22 s to under 5 seconds.
  * Ingestion block creation drops from 14–30 s per block to several blocks per
    second, reducing total ingestion from ~2.5 hours to < 5 minutes.
* **Volume Write Reliability**: Complete elimination of FUSE truncation and
  `_tiffSeekProc` errors across all raster and datablock writing stages.
* **Memory Boundedness**: Windowed streaming ensures memory consumption remains
  constant regardless of the raster's physical pixel dimensions.
* **Clean Subsystem Architecture**: Databricks paths (`/local_disk0`) remain
  entirely in user configuration/notebook space, while library execution and
  storage substrates reside in clean first-class namespaces (`landseg.runtime`,
  `landseg.artifacts.storage`).
* **Repository Hygiene**: `landseg.utils` is preserved as a concise validation
  module rather than an untyped dumping ground.

### 4.2. Neutral / Managed Trade-offs
* **Temporary Disk Footprint**: Materializing GeoTIFFs requires local scratch
  storage proportional to the harmonized imagery extent during the warp step.
  This is safely accommodated by cluster ephemeral storage (e.g. `/local_disk0`
  with > 200 GB available).
* **Harmonization Storage Volume**: Harmonized data on the UC Volume occupies
  real disk space as compressed GeoTIFFs rather than zero-byte virtual VRTs.
  Because all downstream steps (masking, ingestion, previewing) read from these
  files repeatedly, the one-time storage cost is a deliberate and necessary
  trade-off for order-of-magnitude faster pipeline execution.

---

## 5. Implementation & Migration Plan

### Phase 1: Architectural Foundations (Completed on `main`)
* [x] **`a14af30`**: Introduce `landseg.runtime` subsystem (`concurrency`,
  `hardware`, `platform`, `interactive`, `logger`) with auto-scaled concurrency
  `max(1, cpu_count() - 1)`.
* [x] **`72538d6`**: Introduce `landseg.artifacts.storage.scratch` substrate
  (`stage_to_scratch`, `scratch_file`) and `ExecutionContext.scratch_dir`.
* [x] **`3302994`**: Clean-cut migration removing legacy shims in `utils`,
  `mlops`, and `notebooks`, retargeting all call sites and unit tests.

### Phase 2: Cloud Raster I/O Hardening (Active on `cloud-raster-io`)
* [x] **Step 1: Rebase Branch onto `main`**:
  * Rebase `cloud-raster-io` onto `origin/main`.
  * Align preliminary fixes (`75a466e`) with `landseg.runtime.hardware`.
* [x] **Step 2: GDAL Cloud Profile Engine**:
  * Implement `landseg.geopipe.rasters.env` with `configure_gdal_defaults()`.
* [x] **Step 3: Materialized GeoTIFF Warping Engine**:
  * Implement `write_warped_gtiff()` in `geopipe.rasters.warp` using
    `landseg.artifacts.storage.stage_to_scratch`.
* [x] **Step 4: Harmonization & Streaming Mask Pipelines**:
  * Update `geopipe.harmonize.processor` to emit materialized GeoTIFFs.
  * Refactor `geopipe.rasters.mask.unify_nodata_mask` to windowed streaming.
* [x] **Step 5: Consolidated Multi-Band Ingestion Reads**:
  * Optimize `geopipe.rasters.io._read_bands` and `read_window` for single-call
    multi-band reading and selective boundless padding.
* [x] **Step 6: Databricks Notebook Alignment**:
  * Expose `05_scratch_dir` widget in `notebooks/01_data_ingestion.ipynb`
    pointing to `/local_disk0` via `set_scratch()`.
* [x] **Step 7: Verification & Final Merge**:
  * Execute test suites (`tests/unit/geopipe/`, `tests/unit/runtime/`,
    `tests/unit/artifacts/storage/`).
  * Validate end-to-end integration (`tests/integration/test_e2e_intake.py`).
  * Merge `cloud-raster-io` into `main` and close Issues #88, #89, #90, #91.
