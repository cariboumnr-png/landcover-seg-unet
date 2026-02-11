# ADR‑0001 — Decouple Grid, Domain, and Task

- **Status:** Accepted
- **Date:** 2026‑02‑10
- **Version:** v1

## Context

Historically, grid generation, domain mapping, and dataset assembly were
interwoven. This coupling made it hard to reproduce the same grid across
runs, introduced PCA drift when recomputed per split, and limited
reusability in inference tooling. A decoupled, versioned architecture for
grid and domain artifacts remedies these issues.

## Decision

We **decouple** the system into three versioned components, each with a
narrow responsibility and a manifest/spec:

1. **Grid** — Deterministic tiling defined by a `GridSpec`.
   - *Public (API) extent modes:* `ref`, `aoi`, `tiles`.
   - *Engine modes:* `bbox` (covers `ref` and `aoi`) and `tiles`.
   - Implemented by `GridLayout` and the builder that maps API → engine
     modes.

2. **Domain** — Per‑tile domain features (majority, majority frequency,
   PCA embedding) computed from categorical rasters, aligned to the grid.
   - PCA uses economical SVD; `k` is chosen to meet a target cumulative
     explained variance; embeddings are `float32`.

3. **Task** — Orchestrates training/inference; **consumes** grid/domain
   artifacts and injects configuration; does **not** define grid/domain.

### Boundaries

- Grid has **no raster I/O**; it provides deterministic windows and
  alignment by computing integer pixel offsets.
- Domain reads rasters to build features but imports no task code.
- Config flows from manifests/specs into modules at runtime.

### Persistence

- **Grid:** payload (`.pkl`) + meta (`_meta.json`) with
  `schema_id='grid_layout_payload/v1'` and canonical hash.
- **Domain:** payload (`.json`) + meta (`_meta.json`) with
  `schema_id='domain_tile_map_payload/v1'` and SHA‑256 hash.

## Consequences

### Benefits
- Reproducible grids and domain embeddings across runs.
- Reusable artifacts for both training and inference.
- Clear boundaries and simpler evolution of downstream tooling.

### Costs
- Initial refactor and manifest work; small learning curve for the
  API→engine extent mode mapping.

## Implementation (enacted)

- **Grid:** `GridSpec` + `GridLayout` with `bbox|tiles` modes and offset
  alignment; no raster I/O.
- **Builder:** API modes `ref|aoi|tiles` collapsed to `bbox|tiles`.
- **Domain:** `DomainTileMap` with remap, valid‑tile filtering, majority
  stats, PCA to target variance; JSON persistence.
- **Specs/Docs:** Grid and domain design specs and module docstrings
  added to the repo.

## Out of Scope (follow‑up ADRs)

Downstream adaptations (dataset caching, cataloging, summarization, and a
unified task manifest) are intentionally excluded and will be addressed
separately.

## Status

**Accepted** — Branch `architecture/decouple-grid-domain-task` is ready to
merge; the decision has been enacted in code and docs.