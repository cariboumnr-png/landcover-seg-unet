# ADR‑0002 — Grid‑Keyed Dataset Caching & Catalog

- **Status:** Proposed
- **Date:** 2026‑02‑10

## Context
With ADR‑0001, grid and domain are stable artifacts. We need reproducible
caching for imagery/features/labels keyed by grid identity to avoid
re‑computation and to enable shareable inference pipelines.

## Decision
Adopt a cache layout keyed by `(grid_id, grid_version, domain_version,
imagery_hash)` with a lightweight catalog (JSON/Parquet) for discovery
and invalidation.

## Consequences
- Faster end‑to‑end runs; cross‑task reuse.
- Clear invalidation when any component version/hash changes.

## Notes
- Cache entries store provenance (CRS, pixel_size, transforms).
- Catalog exposes queries by AOI, date range, and product type.
