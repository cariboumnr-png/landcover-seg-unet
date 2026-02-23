# ADR‑0003 — Tile Summaries & Reporting
**Status:** Accepted
**Date:** 2026‑02‑10
**Updated:** 2026‑02‑22

## 1. Context
This ADR proposed standardizing **per‑tile** and **per‑AOI** EDA/QA summaries:
- Class histograms
- Valid‑pixel ratios
- Domain PCA distributions
- Raster QA checks (nodata, OOB)
- Outputs stored next to artifacts

## 2. Original Decision Summary
- Provide consistent diagnostic summaries for both train and inference areas.
- Integrate these summaries into monitoring and validation gates.

## 3. What Has Been Implemented
- ✔ Each `DataBlock` computes:
  - class counts,
  - Shannon entropy,
  - valid‑pixel ratios,
  - per‑band statistics (for global stats).
- ✔ Domain PCA (if configured) is computed and stored per block via the
  domain pipeline.
- ✔ All these metrics are already stored in `.npz` block metadata (`meta`).
- ✔ These values indirectly support diagnostics and can be aggregated externally.

## 4. What Has *Not* Been Implemented
- ❌ No **dedicated reporting module** that:
  - aggregates block‑level statistics into per‑tile/per‑AOI summaries,
  - outputs standalone summary reports (JSON/CSV/PDF/plots).
- ❌ No explicit QA step that surfaces nodata coverage, OOB coverage, or
  distribution drift checks.
- ❌ No standard output naming convention for summaries.

## 5. Final Assessment
All **raw metrics needed for tile summaries** now exist and are stored per block.
What is missing is the **reporting layer** that aggregates these metrics into
human‑readable diagnostic artifacts.

This ADR is considered *Accepted* because the pipeline groundwork is complete,
and the decision has effectively shaped your block metadata design.