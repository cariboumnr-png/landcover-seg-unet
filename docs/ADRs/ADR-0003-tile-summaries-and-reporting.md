# ADR‑0003 — Tile Summaries & Reporting

- **Status:** Proposed
- **Date:** 2026‑02‑10

## Context
We need consistent EDA/QA summaries on the decoupled grid for both train
and inference areas.

## Decision
Standardize per‑tile and per‑AOI summaries:
- Class histograms and valid‑pixel ratios.
- Domain PCA stats distributions.
- Optional raster QA (nodata coverage, out‑of‑bounds checks).

## Consequences
- Comparable diagnostics across runs and regions.
- Inputs to monitoring and data validation gates.

## Notes
- Outputs stored next to artifacts and indexed in the catalog.