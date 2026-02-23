# Architecture Overview (Updated)

> This document reflects the architecture originally envisioned under ADR‑0001
> and updated to describe what has been implemented to date, as well as
> divergences and future directions implied by ADR‑0002/0003/0004.

## Purpose
The architectural goal was to **promote Grid and Domain to first‑class,
versioned, reusable artifacts**, decoupled from task logic and consumable by
training/inference workflows. These artifacts were intended to be referenced
via manifests, enabling strict reproducibility and stable inputs.

This document now includes:
- The **original design intent**,
- The **actual implemented behavior**,
- The **gaps or differences** from the ADR‑0001 plan,
- How the system has evolved in practice.

---

## Components

### 1. Grid

#### Original Intent
- Deterministic world‑grid tiling over a projected CRS (`GridSpec`).
- Pixel‑aligned windows and deterministic scanning order.
- Support multiple extent modes (`ref`, `aoi`, `tiles`) unified internally.
- Persist reproducible grid artifacts (`.pkl` + `.json`) with canonical hashing.

#### What Is Implemented
- ✔ `GridLayout` and `GridSpec` exist and produce stable pixel‑aligned windows.
- ✔ Deterministic tiling with support for `bbox` and `tiles` modes exactly as
  intended.
- ✔ Grid persistence is implemented (`pkl` payload + `json` metadata).
- ✔ Hash‑recording exists in line with ADR‑001’s reproducibility goals.
- ✔ Alignment in the data‑prep pipeline uses integer offsets only, ensuring no
  reprojection or implicit interpolation.

#### What Differs / Not Implemented
- ❌ No global grid **version registry**—the grid’s identity is implicit via
  file paths and hash metadata rather than explicit semantic versions.
- ❌ Grid artifacts are not yet referenced by a **unified task manifest**
  (ADR‑0004), though the schema loader achieves a similar result.

---

### 2. Domain

#### Original Intent
- Align domain rasters to the world grid.
- Remap integer labels to a global contiguous space `[0..K‑1]`.
- Filter tiles by valid‑pixel thresholds.
- Compute majority class, frequency distributions, and PCA features
  meeting target variance.
- Persist domain maps with schema id + hash.

#### What Is Implemented
- ✔ Domain rasters are aligned to the grid via pixel offsets.
- ✔ Label remapping and nodata normalization are implemented.
- ✔ Majority‑class computation and normalized frequency vectors are implemented.
- ✔ PCA vectorization for continuous domain features is implemented.
- ✔ Domain artifacts persist as JSON with recorded SHA‑256 values.

#### What Differs / Not Implemented
- ❌ No explicit *domain versioning* or manifest integration.
- ❌ No domain‑level global catalog of available artifacts.
- ❌ Domain QA summaries (e.g., histograms, coverage reports) not generated
  automatically—ADR‑0003 touches on this future direction.

---

### 3. Task Layer

#### Original Intent
- A task should bind:
  - grid artifact,
  - domain artifacts,
  - dataset,
  - model config,
  - output specification.
- Tasks themselves define no grid/domain logic—only consume prebuilt artifacts.

#### What Is Implemented
- ✔ `dataset.load_data()` now **acts as a unified entrypoint**:
  - validates schema,
  - ensures artifacts exist,
  - triggers rebuild on corruption or mismatch.
- ✔ `DataSpecs` aggregates grid layout, label topology, normalization info,
  splits, and domain features.
- ✔ Tasks do not define grid/domain logic; they consume the produced artifacts.

#### What Differs / Not Implemented
- ❌ No explicit “Task Manifest” document authored by the user, though the
  generated `schema.json` now behaves as the effective manifest.
- ❌ No JSON Schema definitions for a manifest (ADR‑0004 future direction).

---

## Specifications

### Grid Spec (as designed vs. implemented)

**Original design fields:**
- `crs`, `origin`, `pixel_size`
- `tile_size`, `tile_overlap`
- either `grid_extent` (for `bbox`) or `grid_shape` (for `tiles`)
- mapping: API modes `ref/aoi/tiles` → engine modes `bbox/tiles`

**Implemented:**
- ✔ All fields are present and expressed in `GridSpec`.
- ✔ Mapping of user-facing extent modes to internal layout construction exists.
- ✔ Enforced invariants:
  - consistent CRS,
  - consistent pixel size,
  - integer‑aligned offsets.

**Not implemented:**
- ❌ Formal schema versioning of grid specs.

---

### Domain Spec (as designed vs. implemented)

**Original design fields:**
- `index_base`, `valid_threshold`, `target_variance`
- global remap to contiguous IDs
- PCA using economical SVD
- JSON persistence w/ schema id + hash

**Implemented:**
- ✔ All functional aspects (remap, filter, PCA) are implemented.
- ✔ Persistence is hashed and stable.

**Not implemented:**
- ❌ No separate “DomainSpec” versioned artifact that is referenced by tasks.
- ❌ No schema id attached to domain JSON beyond normal metadata.

---

## High‑Level Data Flow (Updated)

1. **Grid Construction**
   - Extent selection & grid profile → `GridSpec` + `GridLayout`.
   - Grid artifacts persisted and reloadable.

2. **Domain Construction**
   - Align domain rasters → remap → filter → stats → PCA.
   - Domain artifacts persisted and reloadable.

3. **Dataprep Pipeline**
   - Window mapping → block cache → spectral/topo features → label hierarchy.
   - Block normalization using global stats (Welford algorithm).
   - Scoring + train/val split.
   - Schema generation (acts like a manifest).

4. **Task Consumption**
   - `dataset.load_data()` validates cache integrity and schema.
   - Downstream training/inference consumes `DataSpecs`.

---

## Invariants & Guards (Updated)

- Grid and rasters must share CRS and pixel size; no implicit resampling.
- Domain rasters must be integer typed; nodata coerced to integer (`-1` default).
- Hash‑based integrity checks apply to:
  - all block artifacts,
  - image stats,
  - splits,
  - domain maps,
  - schema.json.
- Schema validation acts as a gateway before task execution.

---

## What Has Evolved Beyond ADR‑0001

- The pipeline uses **schema‑driven reconstruction** rather than a standalone
  manifest (ADR‑0004 influence).
- The system supports reproducibility through **per‑artifact hash tracking**
  rather than a grid‑keyed global catalog (ADR‑0002 influence).
- The block metadata and domain statistics effectively provide the raw material
  needed for **tile‑based QA summaries**, though the reporting layer is not yet
  implemented (ADR‑0003 influence).

---

## Future Work (As Updated by ADR‑0002/0003/0004)

- **Grid‑keyed cache catalog**
  Central registry for grid/domain/dataset combinations and cache reuse.

- **Standard tile + AOI QA summaries**
  Dedicated reporting module that aggregates block‑level metrics into
  human‑readable summaries.

- **Unified task manifest**
  A user-authored manifest that references grid/domain/dataset artifacts
  and fully replaces multi‑config invocation.

---

## Final Notes
The current architecture matches the **intent** of ADR‑0001 while evolving
toward the needs expressed in ADR‑0002–0004.
Grid and domain artifacts are first‑class, reproducible, hash‑validated
components; the schema now serves as the de facto manifest tying these
artifacts together for any downstream task.
