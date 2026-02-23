# ADR‑0004 — Unified Task Manifest
**Status:** Accepted
**Date:** 2026‑02‑10
**Updated:** 2026‑02‑22

## 1. Context
This ADR proposed using a single **Task Manifest** that bundles:
- grid spec id/version
- domain schema + paths
- dataset splits + artifacts
- model config
- output paths
- all validated via JSON Schema

## 2. Original Decision Summary
- Provide one reproducible entrypoint for training/inference.
- Improve auditability and CI/CD promotion.

## 3. What Has Been Implemented
- ✔ `dataset.load_data()` already acts as a **unified entrypoint**:
  - It validates schema integrity.
  - Triggers full rebuild when artifacts are missing or invalid.
- ✔ The resulting `schema.json` includes:
  - world grid information,
  - normalization files,
  - label topology,
  - splits,
  - training stats,
  - domain metadata.
- ✔ This schema effectively behaves as a manifest for downstream training.
- ✔ The `DataSpecs` object consolidates model‑relevant specs into a single runtime bundle.

## 4. What Has *Not* Been Implemented
- ❌ No explicit **Task Manifest file** created by the user prior to running.
- ❌ No top‑level JSON Schema defined for a formal manifest.
- ❌ No CLI that consumes such a manifest directly.
- ❌ Tasks still depend on multiple config groupings (dataset/artifacts/dataprep)
  rather than a single manifest document.

## 5. Final Assessment
Although the pipeline does not yet expose a **formal task‑manifest file**, the
current system *does* behave as a unified and reproducible workflow.

Thus, ADR‑0004 is considered *Accepted*, with future work possible to introduce:
- a true user‑authored manifest,
- JSON Schema validation,
- CLI consuming a single file.
