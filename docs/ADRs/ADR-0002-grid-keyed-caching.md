# ADR‑0002 — Grid‑Keyed Dataset Caching & Catalog
**Status:** Accepted
**Date:** 2026‑02‑10
**Updated:** 2026‑02‑22

## 1. Context
The original intent of this ADR was to introduce a unified, reproducible
caching mechanism keyed by:
`(grid_id, grid_version, domain_version, imagery_hash)`
The cache would be queryable through a lightweight catalog (JSON/Parquet),
enabling reuse of blocks across tasks and clear invalidation rules.

## 2. Original Decision Summary
- Create a cache layout keyed by the above tuple.
- Maintain a discoverable catalog for existing cached results.
- Support invalidation when any component version changes.
- Store provenance (CRS, pixel_size, transforms) with each entry.

## 3. What Has Been Implemented
- ✔ **Per‑dataset cache roots exist**, and each artifact (blocks, stats, splits)
  is **hash‑tracked** via `hash.json` files (schema validates these).
- ✔ The pipeline already performs **immutable validation**: if hashes mismatch,
  the loader triggers a rebuild.
- ✔ Metadata inside blocks contains CRS, pixel size and transforms, fulfilling
  the “store provenance” portion.
- ✔ The overall behaviour now **achieves reproducibility** similar to a
  grid‑keyed cache, because:
  - the world grid is stable,
  - the domain artifacts are stable,
  - and the data prep schema ties them all together.

## 4. What Has *Not* Been Implemented
- ❌ No **global catalog** keyed by `(grid_id, grid_version, domain_version, imagery_hash)`.
- ❌ No central cache reuse across *different datasets* or *tasks* yet.
- ❌ No explicit versioning logic for grid/domain combinations (versions are
  inferred from artifact paths and hashes rather than explicit keys).

## 5. Final Assessment
The implemented system fulfills the **spirit** of ADR‑0002: reproducible and
self‑validating caching.
However, the **mechanism** differs: instead of a global catalog, reproducibility
is achieved via tight schema + hash integrity checks.

A future improvement may still add a global catalog for cross‑dataset
reuse and CI/CD promotion.
