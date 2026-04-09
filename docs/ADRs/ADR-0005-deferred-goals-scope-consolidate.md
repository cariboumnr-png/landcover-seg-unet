# ADR‑0005 — Consolidation of Deferred Goals & Scope Update (Updated)

- **Status:** Accepted — Implemented
- **Original Date:** 2026‑02‑22
- **Updated:** 2026‑04‑09
- **Supersedes / Consolidates:** ADR‑0002, ADR‑0003, ADR‑0004 (remaining open items only)

---

## Context

ADR‑0002/0003/0004 originally introduced several architectural intentions
around caching, reporting, and manifests. Following the merge and subsequent
stabilization of `landseg.geopipe`, the pipeline has reached a mature state
that **meets the original goals**, though in some cases via different (and
simpler) mechanisms.

This ADR was introduced to consolidate deferred or changed intentions and to
realign scope with the **actual, shipped architecture**. As of April 2026,
the architecture and codebase now fully reflect the revised stance outlined
here.

The current pipeline emphasizes:
- Deterministic, artifact‑driven reproducibility
- Grid and domain as first‑class, reusable spatial artifacts
- Generated schemas as the canonical manifest of record
- Explicit lifecycle management over implicit orchestration

---

## Restated Intent (from ADR‑0002/0003/0004)

- **ADR‑0002 (cache & catalog):**
  Deterministic reuse of data blocks via stable identifiers and hashing.

- **ADR‑0003 (tile reporting):**
  Availability of per‑tile / per‑AOI statistics and QA signals derived from
  underlying block metadata.

- **ADR‑0004 (task manifest):**
  A single, authoritative description of dataset structure, provenance, and
  configuration for downstream consumption.

---

## What Is Now Fully Achieved

### Reproducibility & Integrity
- All persisted artifacts (grids, mapped windows, blocks, catalogs, schemas,
  domains, transforms) are hash‑tracked and lifecycle‑managed.
- Schema‑gated loading and deterministic rebuild policies ensure integrity.
- Downstream stages consume only persisted, validated artifacts.

### First‑Class Grid & Domain
- World grids are immutable, persisted artifacts (`GridSpec` / `GridLayout`)
  with explicit metadata and alignment invariants.
- Domain knowledge is represented as persisted, grid‑aligned
  `DomainTileMap` artifacts, including validated tile selection, class
  frequencies, and PCA‑reduced features.
- Both grids and domains are reusable across datasets by construction.

### Manifest of Record
- `schema.json` (foundation and transform stages) functions as the
  **generated manifest of record**.
- All downstream consumers (partitioning, normalization, training spec
  assembly) derive configuration exclusively from persisted schema artifacts.
- No user‑authored top‑level manifest is required for local or standard
  workflows.

---

## Deferred or Revised Items (Final Stance)

### 1) Global Cache Catalog (ADR‑0002)
- **Status:** Intentionally not implemented.
- **Rationale:**
  Caching and determinism are guaranteed via per‑dataset catalogs combined
  with artifact hashing and explicit grid/domain identifiers.
- **Decision:**
  A global, cross‑dataset catalog keyed by
  `(grid_id, grid_version, domain_version, imagery_hash)` is postponed until a
  concrete multi‑dataset reuse requirement emerges.
- **Forward compatibility:**
  Existing `CatalogEntry` structure, hashing, and metadata are sufficient to
  introduce such a catalog later without refactoring.

### 2) Standardized Tile / AOI Reporting (ADR‑0003)
- **Status:** Data model satisfied; exporter is optional.
- **Current state:**
  All required metrics already exist at the block and domain levels:
  class histograms, valid‑pixel ratios, entropy, per‑band stats, and domain PCA.
- **Decision:**
  Reporting is treated as a *thin, optional exporter layer* that aggregates
  existing metadata into per‑tile / per‑AOI JSON or Parquet artifacts, with
  optional visualizations.
- **Implementation:**
  Not required for pipeline correctness; tracked as an optional follow‑up
  tooling concern.

### 3) User‑Authored Task Manifest (ADR‑0004)
- **Status:** Explicitly declined.
- **Decision:**
  Generated schemas (`schema.json`) are canonical and sufficient.
  A lightweight user‑authored manifest may be introduced only if required
  by external schedulers or CI/CD systems.
- **Rationale:**
  Avoids duplication, drift, and manual configuration errors.

---

## Decision

The revised stance outlined in the original ADR‑0005 is **fully implemented
and validated** by the current `geopipe` architecture.

We therefore:
- ✅ Accept the absence of a global cache catalog as intentional.
- ✅ Treat reporting as an optional, downstream aggregation concern.
- ✅ Recognize generated schemas as the sole manifest of record.
- ✅ Preserve forward compatibility for future orchestration or cataloging
  needs without adding premature complexity.

---

## Consequences

- The pipeline remains simpler, deterministic, and artifact‑driven.
- Maintenance burden and conceptual surface area are minimized.
- Future extensions (global catalogs, reporting CLIs, external orchestration)
  can be added cleanly without re‑architecting existing stages.

---

## Out of Scope / No Longer Applicable

- Mandatory global catalogs for all runs.
- Required user‑authored manifests for local or single‑dataset pipelines.
- Built‑in reporting exporters as part of the critical path.

---

## Status & Follow‑Ups

- **Status:** Accepted — Implemented.
- **Optional Follow‑Ups (Non‑Blocking):**
  1. Implement an optional reporting exporter CLI (tile/AOI JSON/Parquet +
     optional plots).
  2. Introduce a global catalog index only if multi‑dataset block reuse becomes
     operationally necessary.