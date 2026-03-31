# ADR-0011: Factory Architecture for Raster, Domain, and Data Schema Preparation
**Status:** Accepted
**Date:** 2026-03-12
**Supersedes:** ADR-0011 (2026-03-10 draft)
**Depends on:** ADR-0010 (Structured Configs with Dataclasses)

---

## 1. Context

This ADR clarifies and finalizes the architecture around three preprocessing factories used to transform raw geospatial inputs into structured, validated artifacts consumed by the training system. Earlier ADR drafts described module boundaries but did not fully reflect the evolved project structure after refactoring and separation of concerns.

The project now contains **three independent preprocessing pipelines**, each producing artifacts required downstream:

1. **prep_grid/** → builds the world-grid layout from extent and grid configs.
2. **prep_raster/** → processes image and label rasters into blocks, stats, splits, and image-derived artifacts.
3. **prep_domain/** → processes domain rasters (categorical / continuous) into domain tile maps.
4. **data_schema/** → integrates artifacts from all three preprocessing steps to produce a canonical training schema.

This ADR documents how these stages relate, why they are factory layers, and how they should interact with configuration dataclasses, runtime modules, and each other.

This document reflects the current tree:

```
prep_grid/
prep_raster/
prep_domain/
data_schema/
```

---

## 2. Decision

### 2.1 Introduce a unified factory layer

Factories:
- Consume validated configuration dataclasses (from `configs/`).
- Perform IO-heavy, data-dependent preprocessing.
- Produce stable artifacts stored on disk.
- Emit thin, runtime-safe specs used downstream.

They are **not** runtime modules.

### 2.2 Four preprocessing factories

**prep_grid/**
- builds world grid from CRS, extent, tile size configs.

**prep_raster/**
- processes imagery + labels into blocks, normalization, splits.

**prep_domain/**
- aligns domain rasters, computes PCA, validity, tile maps.

**data_schema/**
- validates & integrates artifacts to produce the final training schema.

### 2.3 Runtime layer remains config-agnostic
Runtime modules consume only:
- DataSpecs
- DomainTileMap
- GridLayout
- narrow protocols
- primitive parameters

### 2.4 Hydra/OmegaConf usage restricted to CLI/config layers
Only CLI + configs may import Hydra/OmegaConf.

### 2.5 Config objects must never be mutated
Factories may not mutate dataclasses.

### 2.6 Execution order without dependency entanglement
```
prep_grid → prep_domain → prep_raster → data_schema
```
Grid must run first; domain and raster are peers; schema integrates all outputs.

---

## 3. Rationale

- Clarifies `prep_domain` as a factory.
- Clean layering and separation of IO/runtime.
- Prevents config leakage.
- Aligns with ADR-0010 recommendations.

---

## 4. Consequences

### Positive
- Discoverable architecture.
- Runtime modules remain light and testable.
- Preprocessing is modular and extensible.

### Negative
- Requires consistency between all factory outputs.
- Developers must understand the four-stage preprocessing pipeline.

---

## 5. Acceptance Criteria

The requirements defined by this ADR have been fully met:

1. `prep_grid`, `prep_raster`, `prep_domain`, and `data_schema` are formally established and treated as factory modules.
2. Configuration dataclasses are imported exclusively within these factory directories.
3. Runtime modules operate without referencing or importing configuration dataclasses.
4. Hydra/OmegaConf usage is confined strictly to the CLI and configuration layers.
5. Factory modules do not mutate configuration dataclasses; all transformations produce new artifacts and specs.
6. The `data_schema` factory successfully integrates and validates outputs from all upstream preprocessing stages.

---

## 6. Appendix – Summary of Each Factory

**prep_grid**
- Builds world grid from CRS and extent.

**prep_raster**
- Maps rasters to windows, builds blocks, normalization stats.

**prep_domain**
- Aligns domain rasters; computes PCA + tile maps.

**data_schema**
- Integrates all artifacts; performs final validation.

---
