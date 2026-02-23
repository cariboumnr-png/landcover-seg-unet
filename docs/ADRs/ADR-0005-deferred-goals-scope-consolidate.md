# ADR‑0005 — Consolidation of Deferred Goals & Scope Update
**Status:** Proposed
**Date:** 2026‑02‑22

## Context
ADR‑0002/0003/0004 introduced several intentions that remain partially or
wholly unimplemented following the recent merge. The merged branch achieves
the overarching goals—reproducibility, grid/domain first‑classing, and a
clean pipeline—but some mechanisms differ from the originals and a few are
no longer strictly necessary.

This ADR consolidates the deferred/changed items and updates scope to match
the current architecture and codebase.

- Current pipeline already guarantees reproducibility and integrity via
  per‑artifact hashing, schema validation, and rebuild on failure.
- The architecture emphasizes grid/domain as reusable artifacts and a
  schema that effectively acts as the manifest of record.

## Restated Intent (from ADR‑0002/0003/0004)
- **ADR‑0002 (cache & catalog):** Global cache layout keyed by
  `(grid_id, grid_version, domain_version, imagery_hash)` + discovery catalog.
- **ADR‑0003 (tile reporting):** Standard per‑tile/AOI EDA & QA summaries,
  including class histograms, valid‑pixel ratios, domain PCA distributions,
  and raster QA checks.
- **ADR‑0004 (task manifest):** One manifest to reference grid/domain/dataset
  and model configuration; JSON‑Schema‑validated; single entrypoint.

## What Is Already Achieved (via merged branch)
- **Reproducibility & integrity:** Schema‑gated loading, artifact hashing,
  and automatic rebuild on missing/corrupted artifacts.
- **First‑class grid/domain:** Persisted artifacts with alignment invariants
  and hash‑guarded loading.
- **Manifest‑like behavior:** `schema.json` functions as the effective
  manifest for downstream training/inference (even if not a user‑authored
  manifest).

## Deferred or Changed Items (and revised stance)
1) **Global cache catalog (ADR‑0002)**
   - *Not implemented:* No central catalog keyed by the grid/domain/version tuple;
     caching remains per‑dataset cache roots with hash records.
   - *Revised scope:* Defer catalog until there is a real need to reuse the same
     blocks across **multiple datasets or tasks**; the current schema + hashing
     already gives determinism for single‑dataset pipelines.

2) **Standardized tile/AOI reporting (ADR‑0003)**
   - *Not implemented:* No reporting exporter that aggregates per‑block
     metrics into per‑tile/AOI reports and QA bundles.
   - *Revised scope:* Treat reporting as a thin, optional layer on top of existing
     block metadata (counts, entropy, valid ratios, stats, domain PCA), producing
     JSON/Parquet (and optional plots) colocated with artifacts.

3) **Formal user‑authored task manifest (ADR‑0004)**
   - *Not implemented:* No explicit top‑level manifest file, schema, or CLI that
     consumes it.
   - *Revised scope:* Recognize `schema.json` as the **generated manifest of record**.
     Introduce a *lightweight* user manifest later only if/when external scheduling
     or CI/CD needs require a single declarative file.

## Decision
- Accept the revised stance above. The pipeline’s current approach delivers the
  intended outcomes and keeps complexity low. We will:
  - Postpone a global cache catalog until multi‑dataset reuse is required.
  - Add a small “reporting exporter” that aggregates existing metrics into a
    standard per‑tile/AOI artifact set.
  - Treat the generated `schema.json` as canonical; revisit a user‑authored
    manifest if external orchestration demands it.

## Consequences
- Simpler near‑term maintenance and fewer moving parts.
- We preserve forward compatibility to add a catalog and a formal manifest later.
- A reporting exporter is straightforward with the metadata already in blocks.

## Out of Scope / No Longer Applicable
- Hard requirement for a global catalog in all runs.
- Mandatory user‑authored manifests for local pipelines.

## Status & Follow‑ups
- Status: Proposed (to replace the “pending parts” of ADR‑0002/0003/0004).
- Follow‑ups:
  1) Implement the reporting exporter CLI (tile/AOI JSON/Parquet + optional PNGs).
  2) Add minimal hooks for future catalog indexing (behind a feature flag).