# ADR-0068: Cloud Raster I/O, Materialization, and Execution Hardening

**Status:** Accepted — Implemented<br>
**Date:** 2026-11-19 (Updated 2026-11-23)<br>
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

### 2.1. Operational Hardening

To unblock cluster execution, targeted operational fixes were applied to
address hard crashes, numerical instability, and filesystem incompatibilities:

1. **Hardware & Numerical Hardening (Issue #88)**:
   * Removed silent `nan_to_num` masking in `HeadManager` to fail fast on
     non-finite activations.
   * Added consecutive skipped-step guards in `MultiHeadTrainer` against
     runaway `GradScaler` bypass.
   * Added CPU BF16 capability detection and fallback to FP32 on CPUs
     lacking hardware acceleration.
2. **Hierarchical Evaluation & Grid Contiguity**:
   * Corrected `ConfusionMatrix` and `BatchEngine` to evaluate child heads
     strictly against actual parent predictions (`parent_pred_1b`), properly
     penalizing missed parents as False Negatives.
   * Fixed independent coordinate bounds and full-grid assertions in
     `__calc_test_grid()` for non-contiguous AOI blocks.
3. **Local-First Mask GeoTIFF Writing (Issue #89)**:
   * Introduced local-first writing in `geopipe/rasters/mask.py`, writing to a
     temporary directory before copying sequentially to the target Volume to
     eliminate FUSE write-seek truncation.
4. **Sequential NPZ Block Streaming (Issue #90)**:
   * Added `landseg.utils.npzio.savez_compressed`, which builds the archive
     in memory and writes it to the target in one sequential pass, to bypass
     FUSE backward-seek limitations on `.npz` writes.
   * Replaced `numpy.savez_compressed` in `DataBlock.save`,
     `_materialize_one_block` and `Controller._npz_write_dict`.

### 2.2. Architectural Foundation: Runtime & Storage Substrates

While operational fixes resolved immediate data corruption and crashes,
addressing Issue #91 (raster I/O throughput) required introducing fast local
scratch staging and scaled execution concurrency.

Before implementing raster materialization, an architectural risk was addressed:
`landseg.utils` was drifting toward becoming an untyped dumping ground for
heterogeneous responsibilities (multiprocessing, device probing, logging,
notebook detection, widgets). Adding more utilities (e.g., scratch managers)
to `utils` would exacerbate domain bloat.

To establish clean architectural boundaries, foundational substrates were
established:

1. **First-Class `landseg.runtime` Subsystem**:
   * Consolidated execution environment concerns into a first-class subsystem:
     `concurrency` (`ParallelExecutor`), `hardware` (capability checks),
     `platform` (environment detection), `interactive` (widgets and progress),
     and `logger` (unified base logger).
   * Upgraded concurrency worker calculation to
     `max(1, cpu_count() - 1)` with `LANDSEG_MAX_WORKERS` override, resolving
     the 4-vCPU worker starvation issue.
2. **First-Class `landseg.artifacts.storage.scratch` Substrate**:
   * Introduced a centralized local scratch lifecycle under the `artifacts`
     subsystem: `get_scratch_dir()`, `scratch_file()`, `stage_to_scratch()`,
     and `cleanup_scratch()`.
   * Added `scratch_dir` property to `ExecutionContext` and `set_scratch()`
     builder method to `BaseConfigurator`, supporting environment override via
     `LANDSEG_SCRATCH_DIR` and falling back safely to `tempfile.gettempdir()`.
3. **Clean-Cut Deprecation of Legacy Shims**:
   * Completely removed backward-compatibility shims (`utils.device`,
     `utils.multip`, `utils.logger`, `mlops.env`, `notebooks.env`,
     `notebooks.widgets`).
   * Trimmed `landseg.utils` strictly to generic validation helpers
     (`validator.py`).
   * Relocated unit tests cleanly into `tests/unit/runtime/` and
     `tests/unit/artifacts/storage/`.

### 2.3. Scope of Cloud Raster I/O Architecture (Issue #91)

With runtime and scratch substrates established, the cloud raster I/O
architecture was redesigned to eliminate remote object-store latency:
1. **Materialized GeoTIFF Warping**: Transform raw inputs into canonical,
   tiled GeoTIFFs once, avoiding repeated on-the-fly reprojection across
   network storage.
2. **Streaming Windowed Mask Generation**: Bound mask unification memory
   strictly to tile dimensions ($O(256 \times 256)$).
3. **Consolidated Multi-Band Reads**: Batch window reads into single I/O
   operations per data type and selectively apply boundless padding.
4. **Ephemeral Cluster Storage Staging**: Connect pipelines directly to fast
   node-local NVMe scratch storage (`/local_disk0`).

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
**Status:** Accepted — Implemented<br>
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
**Status:** Accepted — Implemented<br>
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
* File-based staging operations (warp outputs, mask generation) route through
  `landseg.artifacts.storage.scratch`. `.npz` archives are not staged on disk;
  they are built in memory by `landseg.utils.npzio` and written sequentially.

### Decision 3: Streaming Windowed Valid-Pixel Mask Generation
**Status:** Accepted — Implemented<br>
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
**Status:** Accepted — Implemented<br>
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
**Status:** Accepted — Implemented<br>
**Problem:** `CORE_N = max(1, cpu_count() - 4)` starved 4-vCPU cloud nodes down
to 1 worker, serializing ingestion. Furthermore, execution concerns were
scattered across `utils`, `mlops`, and `notebooks`.

**Decision:**
* Concurrency and execution primitives are elevated into `landseg.runtime`:
  * Worker count scales to
    `CORE_N = int(os.environ.get('LANDSEG_MAX_WORKERS',`
    `max(1, cpu_count() - 1)))`.
  * On 4-vCPU cloud nodes, worker count increases from 1 to 3 (up to 3x
    block-level parallelism).
  * `ParallelExecutor` centralizes batch execution with worker lifecycle
    management.
* Hardware detection (`is_cpu_bf16_supported()`) is centralized in
  `landseg.runtime.hardware`.
* The runtime subsystem and the scratch substrate (Decision 2) landed on
  `main` ahead of the raster I/O work on this branch.
* Complete clean cut: legacy shims in `utils` were deleted; `utils` contains
  only pure validation helpers.

### Decision 6: Cloud-Optimized GDAL Runtime Profiles
**Status:** Deferred<br>
**Considered:** process-level GDAL options (`GDAL_DISABLE_READDIR_ON_OPEN`,
`GDAL_NUM_THREADS`, `GDAL_CACHEMAX`) for cloud-mounted storage.

**Rationale:** Decisions 1–5 delivered the observed speed-up (§4.1) without
GDAL tuning. The default block cache (5% of RAM) is already adequate,
`ALL_CPUS` would oversubscribe cores under parallel ingestion, and the gain
from skipping directory listings is unmeasured. Revisit only with
stage-scoped options backed by benchmarks.

---

## 4. Consequences & Benefits

### 4.1. Positive Consequences
* **Throughput Improvements** (indicative, not formal benchmarks):
  * On a 4-vCPU Databricks node with UC Volume storage, ingestion block
    creation improved from roughly 14–30 s per block to roughly 3–5 s per
    block.
  * The remaining gap to a 20-thread workstation (16 workers, several blocks
    per second) is consistent with the lower core count rather than with
    storage I/O.
  * Valid-pixel mask generation (previously 2 min 22 s) now streams by window
    from materialized GeoTIFFs; it was not separately timed.
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
  trade-off for substantially faster pipeline execution.

---

## 5. Architectural Implementation Status

The architectural decisions defined in this record have been implemented across
the codebase and validated end-to-end on Databricks (Decision 6 deferred):

### 5.1. Runtime Execution Subsystem (`landseg.runtime`)
* **Dynamic Concurrency**: Upgraded worker calculation to `max(1, cpu_count() - 1)`
with `LANDSEG_MAX_WORKERS` override, resolving cloud node worker starvation.
* **Hardware Capability Probing**: Centralized instruction checks (`is_cpu_bf16_supported`)
to govern mixed-precision fallback policies across heterogeneous CPU nodes.
* **Namespace Isolation**: Relocated concurrency, platform, interactive, and logging
facilities into `landseg.runtime`, deprecating untyped shims in `landseg.utils`.

### 5.2. Local Scratch Storage Substrate (`landseg.artifacts.storage.scratch`)
* **Centralized Lifecycle**: Implemented `get_scratch_dir()`, `scratch_file()`,
`stage_to_scratch()`, and `cleanup_scratch()`.
* **Configurable Precedence**: Structured resolution order prioritizing explicit
arguments, context configuration, environment variables (`LANDSEG_SCRATCH_DIR`),
and system temp fallbacks.
* **Sequential Write Safety**: Local staging with sequential copy publishing for
GeoTIFF and VRT outputs, and in-memory sequential writes for NPZ archives
(`landseg.utils.npzio`), eliminating FUSE write-seek truncation.

### 5.3. Cloud-Optimized Raster Engine (`landseg.geopipe.rasters`)
* **Materialized Warping (`warp.py`)**: Implemented `write_warped_gtiff()` to
warp raw rasters once into materialized, tiled (256x256), compressed GeoTIFFs via
scratch staging, replacing fragile virtual reprojection graphs.
* **Windowed Mask Generator (`mask.py`)**: Streamed valid-pixel mask generation
by block window with constant $O(256 \times 256)$ RAM bounds.
* **Consolidated Block I/O (`io.py`)**: Consolidated multi-band window reading
into single calls for uniform dtypes, grouped calls for mixed dtypes, and applied
boundless padding selectively only to raster perimeters.

### 5.4. Enterprise Pipeline & Notebook Integration
* Exposed `set_scratch()` across all stage configurators.
* Added the optional `05_scratch_dir` widget to `notebooks/01_data_ingestion.ipynb`
(empty by default, falling back to the system temp directory); on Databricks,
set it to node-local scratch (e.g. `/local_disk0`).
