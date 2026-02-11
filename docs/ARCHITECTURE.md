# Architecture Overview

> This document describes the architecture enacted by ADR‑0001 and the
> resulting module boundaries and data contracts.

## Purpose

Promote grid and domain to first‑class, versioned artifacts reusable
across training and inference, with configuration injected via manifests
and strict decoupling from task code.

## Components

### Grid

- **Responsibility:** Deterministic tiling over a projected CRS using a
  `GridSpec`; returns stable pixel‑origin windows and supports raster
  alignment via integer offsets. No raster I/O.
- **Extent Modes (API → Engine):**
  - API: `ref`, `aoi`, `tiles` (chosen by builder).
  - Engine: `bbox` (covers `ref`, `aoi`) and `tiles` (in `GridLayout`).
- **Persistence:** payload `.pkl` + meta `.json`
  (`grid_layout_payload/v1`, canonical hash).

### Domain

- **Responsibility:** For each domain raster (categorical, single‑band),
  align to the world grid, remap labels to `[0..K‑1]`, filter tiles by
  valid‑pixel fraction, compute majority stats and normalized frequency
  vectors, and project to PCA features meeting target variance.
- **Persistence:** JSON payload + metadata
  (`domain_tile_map_payload/v1`, SHA‑256).

### Task

- **Responsibility:** Bind grid/domain artifacts with datasets and model
  configs; orchestrate training/inference; no grid/domain definitions.

## Specifications

### Grid Spec (design)

- Fields: `crs`, `origin`, `pixel_size` (positive magnitudes), `tile_size`,
  `tile_overlap`, plus either `grid_extent` (for `bbox`) or `grid_shape`
  (for `tiles`).
- API extent modes are mapped by builder: `ref|aoi|tiles` → `bbox|tiles`.

### Domain Spec (design)

- Fields: `index_base`, `valid_threshold`, `target_variance`.
- Rules: integer labels; global remap to `[0..K‑1]`; PCA (SVD), `k` by
  cumulative explained variance; JSON persistence with schema id and hash.

## Data Flow (high level)

1. Choose **extent config** (`ref|aoi|tiles`) and **grid profile** → builder
   computes `GridSpec` and constructs `GridLayout`.
2. Prepare **domain** for each configured raster → build `DomainTileMap`
   (alignment, remap, filter, stats, PCA) or load existing artifacts.
3. **Task** consumes grid/domain artifacts for training or inference.

## Invariants & Guards

- Grid and rasters share CRS and pixel_size; alignment uses integer pixel
  offsets; no reprojection/resampling here.
- Domain rasters are integer; nodata is integer (default `-1` if absent).
- Persistence uses schema ids and content hashes for integrity checks
  (grid: canonicalized hash; domain: SHA‑256).

## Future Work (separate ADRs)

- Grid‑keyed dataset caching & cataloging.
- Standard tile and AOI summaries (EDA/QA).
- Unified task manifest and pipeline orchestration.