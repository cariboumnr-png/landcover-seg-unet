# ADR-0057: Named Schemes and Data Preparation Resolution

**Status:** Accepted — Implemented
**Date:** 2026-09-02

---

## 1. Context

In previous versions of the `landseg` geospatial ETL and preparation pipeline:

1. **Contaminated & Static Sidecar Metadata**:
   Dataset metadata sidecar JSON files mixed intrinsic physical raster
   properties (such as index base, class counts, and colormaps) with fixed
   experiment-level target reclassifications (`reclass` and `reclass_name`),
   locking the dataset into a single static hierarchy during harmonization
   and ingestion.
2. **All-or-Nothing Feature Channels**:
   Ingestion and normalization operated indiscriminately on all channels of
   raw feature rasters without allowing users to configure or slice specific
   feature bands (e.g., RGB vs. RGB-NIR vs. Surface Reflectance) for downstream
   experiments.
3. **Semantic Proximity vs. Experiment Ergonomics**:
   While experiment-level selection belongs in `user.yaml` or notebooks,
   writing raw band names or complex class mappings from scratch for every
   experiment configuration was tedious and severed the connection to
   reusable domain semantics established alongside the data sources.

---

## 2. Decision

We decoupled intrinsic raster metadata from experiment-level data preparation
by introducing **Named Schemes** and a dedicated **Preparation Resolution
Layer** across `landseg.geopipe`.

### 2.1. Pure Categorical Specifications and Unified Index Base
- **Purified `CategoricalSpecs`**:
  Stripped `reclass` and `reclass_name` out of `CategoricalSpecs` (renamed from
  `LabelSpecs`) so that it strictly describes physical GeoTIFF raster properties
  (`index_base`, `num_cls`, `ignore_cls`, `class_name`, `color_map`, `taxonomy`).
  Taxonomy specifications validate against standard profiles and return a typed
  `TaxonomySpecs` dictionary (`profile`, optional `canonical_indices`).
- **Explicit `index_base` Support**:
  Unified 0-based and 1-based raster indexing across label and domain
  categorical rasters, enforcing strict validation against class index bounds.

### 2.2. Named Schemes in Dataset Sidecars and Harmonization Processing
- **Root Manifest and Sidecar Submodule**:
  Root `manifest.json` entries map raw rasters to sidecar files via the
  `"manifest"` key. Manifest validation and compilation are modularized in
  `landseg.geopipe.harmonize.manifest` (`schema.py`, `normalizer.py`,
  `compiler.py`).
- **Manifest Sidecar Schema (`schemes`)**:
  Sidecars declare optional, semantically close named schemes:
  - **`FeatureSchemes`**: Named band groupings (e.g.,
    `rgb: ["blue", "green", "red"]`, `rgb_nir: ["blue", "green", "red", "nir"]`).
  - **`LabelSchemes`**: Named hierarchical target reclassifications (e.g.,
    `binary: {reclass: {"1": [1, 2], "2": [3]}, reclass_name: {"1": "forest", "2": "water"}}`).
  - **Domain Rasters**: Strictly enforced to have `schemes: null`.
- **Harmonized VRT Tagging & Ingestion Schema Propagation**:
  - `data-harmonize`: Coordinated by `harmonize.processor.harmonize_sources`,
    which writes resolved schemes into harmonized VRT dataset metadata tags
    (`schemes={cfg['name']: schemes}`).
  - `data-ingest`: Reads embedded schemes from the source VRTs via
    `io.read_schemes()` and records them into the dataset `schema.json`
    artifact under `dataset.schemes`.

### 2.3. Dedicated Preparation Resolution Layer (`geopipe.prepare.resolver`)
- **Module Separation**:
  Introduced `landseg.geopipe.prepare.resolver` housing pure metadata and
  configuration resolution:
  - `resolve_feature_channels()`: Resolves active band names and 0-based
    channel indices against `image_band_map` using `raster_schemes` or inline
    band lists.
  - `resolve_target_reclass()`: Resolves active multi-head target hierarchies
    against `label_names` using `raster_schemes` or inline reclass
    dictionaries per label layer.
- **Pipeline Consumption (`data_prepare.py`)**:
  `data-prepare` loads `schema.json`, extracts `raster_schemes`, and passes
  them to the resolver functions before slicing and normalizing data blocks.

### 2.4. Selective Statistics Aggregation and Dynamic Normalization
- **Channel-Specific Image Statistics (`stats.py`)**:
  Updated global Welford stats aggregation to compute per-band normalization
  statistics exclusively on the selected feature channels.
- **Dynamic Multi-Head Target Construction (`normalize.py`)**:
  Block normalization slices images to active channels and generates
  multi-head target label stacks via `_reclassify_label_stack` on the fly,
  preserving the immutability of canonical ingested `.npz` data blocks.

### 2.5. Dual Public Interfaces (CLI & Programmatic API)
- **CLI Configuration (`configs/user.yaml` & Hydra)**:
  Exposed `features` and `targets` sections under `data-prepare:`, mapped
  into `data.preparation` via `landseg.adapters.cli.translate`.
- **Programmatic Python API (`DataPreparationConfigurator`)**:
  Added fluent `set_features()` and `set_targets()` methods for notebook and
  script environments.

### 2.6. Ingestion Feature Engineering & Preparation Cohesion
- **Exposed Ingestion Feature Engineering**:
  Exposed `add_topo: bool = False` and `add_spectral: list[str] | None = None`
  across `_DataBlocks` schema, Hydra defaults (`default.yaml`),
  `BlockBuildingParameters`, `data_ingest.py`, and `user.yaml` (`data-ingest:`).
  This enables optional on-the-fly calculation of topographic metrics (slope,
  aspect sine/cosine, TPI) and spectral indices (NDVI, NDMI, NBR) during block
  construction.
- **Explicit Preparation Channel Selection**:
  Ingested blocks record all raw and engineered channels into
  `image_band_map`. Preparation resolution in `geopipe.prepare.resolver`
  is kept deliberately simple and boring: it maps explicit band lists or
  manifest-defined feature schemes directly against `image_band_map` without
  pseudo-dataset groups, fuzzy phrases, or boolean heuristics.

---

## 3. Consequences

### Positive
- **Semantic Proximity**: Domain experts define reusable band combinations
  and reclassification hierarchies alongside the raw dataset sidecars.
- **Experiment Agility**: Data scientists can test different feature subsets
  and target definitions in `user.yaml` or notebooks without re-running data
  harmonization or ingestion.
- **Immutability of Ingested Blocks**: Ingested `.npz` blocks remain pure,
  canonical representations of the underlying geospatial layers.
- **Single Responsibility & Simplicity**: Clear separation between metadata
  resolution (`resolver.py`), catalog filtering (`adapter.py`), and tensor
  operations (`normal_blocks/`). The resolver operates strictly with explicit
  mappings and deterministic types.
- **Fail-Fast Configuration Validation**: Explicit resolver checks detect
  misconfigurations early (e.g. unknown bands or non-existent sidecar schemes)
  before block normalization begins.

### Negative / Migration
- **Sidecar Schema Migration**: Existing dataset manifests and sidecars must
  separate `reclass` and `reclass_name` into `schemes.LabelSchemes` and remove
  them from intrinsic `categorical_specs`.
- **Target Head Selection**: Downstream model heads must specify targets that
  match either canonical raw classes or declared scheme names.

---

## 4. Implementation Summary

1. **Purified Categorical Specs**: Decoupled `reclass` and `reclass_name` from
   `CategoricalSpecs` in manifest definitions and added taxonomy profiling.
2. **Dataset Sidecar Schemes**: Implemented `FeatureSchemes` and `LabelSchemes`
   in sidecars, embedding schemes into harmonized VRT metadata and persisting
   them into `schema.json`.
3. **Dedicated Resolution Engine**: Created `geopipe.prepare.resolver` with
   `resolve_feature_channels()` and `resolve_target_reclass()` to dynamically
   evaluate schemes, inline lists, and reclass mappings.
4. **Dynamic Normalization Pipeline**: Updated block normalization and global
   Welford statistics calculation to slice and normalize only selected
   feature channels on the fly.
5. **Ingestion Feature Engineering**: Exposed `add_topo` and `add_spectral` in
   dataclass schemas, Hydra defaults, execution pipelines, CLI translators,
   and `user.yaml`.
6. **Explicit Channel and Target Resolution**: Kept channel resolution simple
   and explicit against `image_band_map`, removing pseudo-dataset aliases.
7. **Quality Assurance**: Added full unit test suites for schema validation,
   manifest verification, CLI translation, channel resolution, and pipeline
   execution.
