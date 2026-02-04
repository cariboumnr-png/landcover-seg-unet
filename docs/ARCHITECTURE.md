# Architecture Overview

## Purpose
Document the current *coupled* dataset pipeline (grid → domain → training in one sequence) and the target *decoupled* architecture where **grid**, **domain**, and **task** are independent, versioned components.

## Current State (as of 2026-02-04)
**Flow:**
1) Create grid
2) Assign domain along with training data to the grid
3) Train model

**Coupling observed:**
- `dataset/blocks/layout.py` generates tiles directly for a specific dataset/split
- `dataset/domain.py` pulls domain features as part of dataset assembly
- `training/dataloading/dataset.py` assumes domain+imagery+labels are bundled

**Implications:**
- Hard to reproduce the same grid across tasks
- Domain PCA can drift if recomputed per run/split
- Inference tooling can’t reuse grid/domain caches easily

## Target State (Decoupled)
**Contracts:**
- **Grid**: Purely geometric. Deterministic tiling (CRS, tile size, overlap), stable IDs. No labels/domain at this stage.
- **Domain**: Predefined, versioned datasets (eco IDs, geology PCA on a fixed global basis). Computed per grid cell, independent of labels/splits.
- **Task**: Binds a grid spec + domain schema + labels/splits + model config.

**Benefits:**
- Reproducibility (same tile IDs across runs)
- Safe conditioning (global, fixed PCA/normalization)
- General inference (any AOI → grid → domain → run)
- Cache-ability (grid IDs key domain & imagery)

## Interfaces (summary)
- Grid Spec → `docs/specs/grid_spec.v1.yaml`
- Domain Schema → `docs/specs/domain_schema.v1.yaml`
- Task Manifest → `docs/specs/task_manifest.v1.yaml`

## Migration Plan (phased)
- Phase 1: Document current behavior, freeze PCA artifacts, add manifests (no code movement)
- Phase 2: Extract grid as first-class spec + IDs (keep call sites unchanged)
- Phase 3: Extract domain generation to use grid IDs + versioned PCA
- Phase 4: Task manifest becomes the single entrypoint for training/inference

## Versioning
- `grid_spec.version`: e.g., `grid_v1`
- `domain_schema.version`: e.g., `dom_v1` (references `eco_2024a`, `geo_pca_v1.0`)
- PCA artifacts: `(μ, W)` persisted under version; never refit silently