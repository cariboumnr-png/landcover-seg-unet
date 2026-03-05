# Architecture Overview (Updated March 2026)

This document replaces the outdated `ARCHITECTURE.md` and reflects the **current, implemented architecture** across Grid, Domain, Dataprep, Dataset, Training/Controller, CLI/Config, and Experiment I/O. It synthesizes the curent codebase
and all accepted ADRs (0001–0009).

---
## 1. Architectural Principles

The system now consistently follows these principles:

1. **Grid and Domain are first‑class, reproducible, hash‑validated artifacts** (ADR‑0001).
2. **Data-prep is a deterministic, schema‑driven pipeline** producing `schema.json` as a manifest of record (ADR‑0004, ADR‑0005).
3. **Artifact integrity is enforced by per‑file SHA256 hashes**, stored in `hash.json` files (ADR‑0002).
4. **Tasks (training, inference, overfit test) consume only stable artifacts**, never constructing their own grid/domain (ADR‑0001, 0004).
5. **Experiment-level I/O is fully centralized under a user‑specified `exp_root`** (ADR‑0007).
6. **Configuration is packaged, layered, and profile‑driven**, with a built‑in `overfit_test` mode (ADR‑0008, 0009).

---
## 2. Component Architecture

### 2.1 Grid System
**Intent:** Deterministic, pixel‑aligned world grid describing tiling.

**Implemented:**
- `GridSpec` captures CRS, pixel size, origin, tile size, overlap.
- `GridLayout` creates deterministic windows for all tiles.
- Extent modes: `ref`, `aoi`, `tiles` → internally mapped to `bbox` or `tiles`.
- Persistence: `grid_id_meta.json` + `grid_id.pkl`, with hash recording.
- `prep_world_grid()` loads or builds the grid.

**Not Implemented:**
- No semantic *versioning* of grids—identity is derived from file paths + hash (ADR‑0002 revision).

---
### 2.2 Domain System
**Intent:** First‑class per‑tile semantic features aligned to the world grid.

**Implemented:**
- `DomainTileMap` loads categorical rasters, aligns via integer pixel offsets, and:
  - remaps raw labels → contiguous IDs,
  - filters invalid tiles by pixel ratio threshold,
  - computes majority class and frequency,
  - computes per‑tile PCA vectors meeting target variance.
- Fully persisted as JSON payload + metadata (`schema_id='domain_tile_map_payload/v1'` + SHA256).
- Loaded automatically during `load_data()`.

**Not Implemented:**
- No global domain catalog (ADR‑0002 deferred).

---
### 2.3 Dataprep Pipeline
**Intent:** Stable, deterministic pipeline that builds all data artifacts.

**Implemented:**
1. **Raster → world‑grid tiling** (`mapper.map_rasters`)
2. **Block building** (image/label/DEM) with validation & integrity checks.
3. **Block normalization** via Welford global stats.
4. **Label scoring and train/val block split**.
5. **Schema generation** (`schema.json`) containing:
   - grid info,
   - normalization stats,
   - block splits,
   - label topology,
   - domain metadata references.
6. **Single‑block mode** for overfit testing.

**Not Implemented:**
- No full tile‑level or AOI‑level reporting module (ADR‑0003 deferred).

---
### 2.4 Dataset Layer
**Intent:** Convert persisted artifacts + schema into runtime specifications for training/inference.

**Implemented:**
- `validate_schema()` confirms integrity of all dependent artifacts.
- `build_dataspec()` builds `DataSpecs`, containing:
  - meta info (channels, ignore index, block sizes),
  - heads (class counts, logit adjustments, topology),
  - splits (train/val/test dictionaries),
  - domain knowledge (IDs + PCA vectors).
- `load_data()` orchestrates rebuild if artifacts are missing or corrupted.

---
### 2.5 Model + Trainer + Controller
**Model Layer:**
- Multi‑head UNet/UNet++ with optional domain conditioning (`concat`, `film`, `hybrid`, or `none`).
- Dropout, normalization modes (BN/GN/LN), activation, clamping.

**Training Engine (Trainer):**
- Multi‑phase curricula with separate logit‑adjust and head‑state configurations.
- Scheduler, optimizer (AdamW), gradient clipping, AMP.
- Patch sampling and deterministic loaders.

**Controller:**
- Drives multi‑phase execution.
- Manages checkpointing, progress files, and early stopping.

**Overfit Test Mode:** (ADR‑0008, 0009)
- One‑block dataset.
- No regularization, no augmentation, deterministic loader.
- AMP off.
- Dropout disabled.
- Conditioning disabled.
- Used for debugging and CI.

---
### 2.6 CLI & Configuration System
**Implemented:**
- Full package layout under `src/landseg`.
- Hydra‑driven configuration tree placed inside the package.
- Entry point: `experiment_run` console script → `cli/main.py`.
- Profile system (`end_to_end`, `overfit_test`).
- Layered configuration merge: base config → settings.yaml → profile overrides → dev overrides.

---
### 2.7 Experiment I/O Structure
**Implemented (ADR‑0007):**
A single `exp_root` contains **everything**:
```
<exp_root>/
  input/                 # raw user data
    <dataset_name>/
      fit/
      test/
  artifacts/             # persistent, shareable
    world_grids/
    domain_knowledge/
    data_cache/
  results/               # per-experiment folders
    exp_0001/
      logs/
      plots/
      checkpoints/
      previews/
      config.json
```
This structure cleanly separates **stable artifacts** from **experiment outputs**.

---
## 3. High‑Level Data Flow
```
User Data → (Grid) → Tiling
         → (Domain) → DomainTileMaps
         → (Dataprep) → Blocks → Normalization → Split → schema.json
         → (Dataset.load_data) → DataSpecs
         → (Trainer + Controller) → Training / Evaluation
         → (Exp I/O) → results/exp_xxxx/
```

---
## 4. Guarantees & Invariants
- Grid, domain, and all cached artifacts are hash‑validated.
- No implicit resampling: CRS + pixel size must match.
- All pipelines are deterministic unless explicitly using randomness (documented knobs).
- Schema.json is the **manifest of record** and must pass validation before training.

---
## 5. Deviations from Original Architecture
- Global grid/domain catalog **not implemented** (ADR‑0002 revised in ADR‑0005).
- Domain and Grid versioning **implicitly** tracked via hashes rather than semantic versions.
- Task manifest is **generated**, not user‑authored (ADR‑0004, revision in ADR‑0005).
- Tile/AOI reporting layer postponed (ADR‑0003).

---
## 6. Future Work
- Add global cache/catalog for cross‑dataset reuse (ADR‑0002).
- Add unified tile/AOI reporting module (ADR‑0003).
- Optionally introduce a user‑authored task manifest (ADR‑0005).
- Consider richer CLI commands (diagnostics, batch experiments).

---
## 7. Summary
The system now implements a **complete, deterministic geospatial ML pipeline**:
- Reproducible grids and domain maps.
- Robust dataprep with integrity guarantees.
- Strong schema validation.
- Modular training system.
- Clean experiment isolation.
- Standardized overfit testing.

This architecture reflects all accepted ADRs and the current codebase.
