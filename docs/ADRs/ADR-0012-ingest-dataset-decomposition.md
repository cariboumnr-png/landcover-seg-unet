# ADR-0012 (Updated): Dataset Ingestion Pipeline Decomposition

- **Status:** Accepted / Implemented
- **Date:** 2026-03-31
- **Supersedes:** ADR-0012 (Proposed)

## Context

The earlier ingestion pipeline computed normalization before dataset splitting
and sometimes wrote normalized arrays into per-block artifacts, creating a risk
of data leakage and tightly coupling the block catalogue to a particular
experiment. The new implementation decomposes the pipeline so that raw block
artifacts are immutable and experiment-independent, while normalization happens
later using statistics derived **only from the training split**.

While implementing this decomposition, module names diverged from the original
ADR wording:

- **Canonical layer** (ADR: *align* + *blocks*) is implemented under
  `geopipe.foundation` (world grid alignment & raw block building).
- **Materialized layer** (ADR: *split* + *standardize*) is implemented under
  `geopipe.transform` (splitting, normalization, and schema emission).

Additionally, previously implemented **domain knowledge** components were
consolidated by extracting concrete class implementations into
`geopipe.core` (e.g., `DataBlock`, `BlocksCatalog`, `DomainTileMap`,
`GridLayout`), so they can be reused consistently across the pipeline and the
trainer stack.

A stage‑based training runner pipeline (multi‑phase schedule with head
activation/logit‑adjust controls) was also added in this branch, improving
training orchestration but orthogonal to data ingestion per se.

## Decision

Adopt a **two‑layer, four‑stage** pipeline that maps directly to the original
intent, with updated module naming to reflect the codebase:

- **Canonical (read‑only, reusable across experiments)**

  1. **World‑grid alignment** → build/load a persisted `GridLayout` (windows)
     and align rasters on demand (`offset_from`).
  2. **Raw blocks** → build immutable `.npz` blocks with images/labels/metadata
     but **no normalized arrays**; manage a read‑only catalogue.

- **Materialized (experiment‑dependent)**

  3. **Split** → consume the raw catalogue to produce train/val/(optional test)
     manifests; compute label distributions/metrics needed for scoring. No
     normalization here.
  4. **Standardize** → compute per‑band mean/std over **train only**, then
     normalize all splits using these stats; emit a **final schema** tied to
     the normalized artifacts.

## Module Mapping (Implemented)

### Canonical layer — `geopipe.foundation`

- **World grid**: `grid_generator.GridLayout`/`GridSpec` build or load a
  persisted grid; `offset_from` supplies raster alignment as an integer pixel
  offset per dataset.
- **Raw blocks**: `foundation_data_block.DataBlock` constructs per‑window data
  and writes `.npz` without normalized arrays; catalogue/metadata are persisted
  via `BlocksCatalog.save_json(...)` → `catalog.json` and dataset‑level
  `metadata.json`. Existing blocks are **skipped** to avoid collisions.

### Materialized layer — `geopipe.transform`

- **Split**: `partition_blocks(...)` writes raw split manifests and label
  stats (see *Artifacts*), operating purely on the raw catalogue.
- **Standardize**: `build_normalized_blocks(...)` aggregates **train‑only**
  band statistics and writes normalized blocks plus a normalized split registry;
  `build_schema(...)` then emits the final `schema.json` pointing **only** to
  normalized artifacts.

## Artifacts & Conventions

> File names reflect the implemented branch; they replace the earlier ADR’s
> placeholders (e.g., `extract_stats.json`).

- **Raw catalogue (canonical)**
  - `catalog.json`: grid‑indexed block registry with provenance, checksums,
    and per‑block metadata.
  - `metadata.json`: dataset‑level conventions (tensor shapes, dtypes, label
    topology, ignore index, source paths, mapped grids).
- **Split outputs (materialized)** — produced by `partition_blocks(...)` under
  the transform root:
  - `block_source.json`: *raw* block paths per split.
  - `label_stats.json`: per‑head class counts used for scoring/diagnostics.

- **Normalization outputs (materialized)** — produced by `build_normalized_blocks(...)`:
  - `image_stats.json`: **train‑only** per‑band stats (count, mean, std, M2).

  - `block_splits.json`: *normalized* block paths per split.

  - Normalized block files (`.npz`), using array keys:
    - **image** → `'image'`
    - **labels** → `'label_stack'` (hierarchical base/reclass channels)

- **Final dataset schema (materialized)**
  - `schema.json`: references **normalized** splits only; persists
    array‑key conventions (`image`/`label_stack`), statistics, checksums, and
    artifact paths for reproducibility.

## Data Flow (Implemented)

```
# canonical (reusable)
reference raster/extent → GridLayout (persisted) → map rasters to windows
→ build DataBlock (.npz) per window → update catalog.json / metadata.json

# materialized (experiment)
partition_blocks → block_source.json + label_stats.json
→ build_normalized_blocks (train-only stats → image_stats.json)
→ normalized .npz + block_splits.json
→ build_schema → schema.json (consumes block_splits.json)
```

## Rationale

- Prevents data leakage by computing normalization strictly from the **train**
  split and materializing normalized artifacts per experiment.
- Keeps raw blocks stable, reproducible, and **reusable** across multiple
  experiments and splits.
- Allows changing normalization techniques or sampling/scoring strategies
  without touching the raw catalogue.

## Consequences

- Raw blocks **exclude** normalized arrays and remain versionable and portable.

- Experiments depend only on normalized manifests and `schema.json`, enabling
  deterministic training I/O and auditing.
- Additional artifact files exist compared with the original ADR text (e.g.,
  `image_stats.json`, `block_splits.json`), but they clarify responsibilities
  between source vs. normalized splits.

## Implementation Notes (This branch)

- **Domain knowledge consolidation:** concrete implementations for domain tiles,
  world grid, data blocks, and catalog were extracted into `geopipe.core` so
  both data prep and the trainer stack rely on the same primitives.
- **Catalogue behavior:** block creation skips existing artifacts to avoid
  collisions; catalog writes are deterministic and hashed. (Full immutability
  enforcement can be added via a future read‑only “frozen” marker.)
- **Stage‑based training pipeline:** experiment runner supports multi‑phase
  schedules (per‑phase heads, logit‑adjust scheme, LR scaling), implemented in
  the branch’s runner/trainer stack.


## Acceptance

We will **accept this branch as it currently stands**. The implementation
meets the intent of ADR‑0012 in substance, with the updated module naming and
artifact conventions reflected in this document. Any remaining follow‑ups
(e.g., adding a catalogue `frozen` guard or publishing JSON schemas) can be
handled as incremental improvements without blocking adoption.