# ADR-0057: Dynamic Dataset Semantics, In-Memory Data Context, and Decoupled Preparation Pipeline

**Status:** Accepted — Implemented
**Date:** 2026-09-02 (Updated: 2026-09-20)

---

## 1. Context

In the baseline version of the `landseg` geospatial ETL and preparation pipeline (on `main`):

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
4. **Premature Materialization & Monolithic Pipeline Coupling**:
   Computing class distributions for reclassified or multi-head targets
   previously required materializing and writing normalized blocks to disk
   before spatial partitioning could occur. This tightly coupled tensor
   generation with dataset splitting, caused unnecessary disk I/O, and
   computed image normalization statistics across all blocks rather than
   strictly over the training split.
5. **Coupled Ingestion Tensors**:
   Ingestion pre-baked multi-head label stacks and artificial reclassifications
   into `.npz` data blocks, violating block immutability and preventing
   subsequent experiments from redefining target schemas without re-ingesting
   the entire dataset.
6. **Fragmented `common/` Submodules & Missing Contract Governance**:
   Each geopipe submodule (`harmonize/common`, `prepare/common`,
   `ingest/common`) maintained an ad-hoc, internal `common/` directory housing
   an inconsistent mix of untyped dictionary schemas, bespoke logger
   implementations, and artifact helpers. There was no central, formal
   contracts system defining standard data exchange interfaces, report
   schemas, or structured summary payloads across pipeline stages.
7. **Tight Inter-Pipeline Coupling & Leaky Module Boundaries**:
   Downstream pipelines were tightly coupled to upstream stage
   implementations:
   - `data-harmonize` directly imported and executed `grid` generation logic.
   - Harmonization summary schemas embedded full world grid report objects
     rather than decoupled primitive references (`grid_id`, `grid_fpath`).
   - Ingestion and diagnostic pipelines directly imported `geopipe.grid` to
     resolve grid layout paths and read reports.
   - Inter-stage data passing relied on legacy adapters (`adapter.py`,
     `harmonization_inputs.py`) that intermingled contract parsing with
     pipeline logic rather than providing clean, in-memory context containers.
8. **Scattered and Duplicated Typing Definitions**:
   Typing primitives (`Coord2d`, `RasterWindow`, `IntArray`, etc.) were
   defined redundantly across disparate `alias.py` files in subpackages,
   leading to inconsistent type checking and cyclic import workarounds.

---

## 2. Decision

We overhauled the end-to-end data lifecycle across `landseg.geopipe` by
introducing **Named Schemes**, **Canonical Ingestion DataBlocks**, an
in-memory **Data Context Layer**, a deferred **Materialize Blocks Pipeline**,
a centralized **Contracts Architecture**, and pure **Inter-Pipeline Decoupling**.

### 2.1. Pure Categorical Specifications, Taxonomy Profiling, and Index Base
- **Purified `CategoricalSpecs`**:
   Stripped `reclass` and `reclass_name` out of `CategoricalSpecs` (retiring
   legacy `LabelSpecs` completely across core, ingest, and prepare) so that it
   strictly describes physical GeoTIFF raster properties (`index_base`,
   `num_cls`, `ignore_cls`, `class_name`, `color_map`, `taxonomy`).
- **Typed Taxonomy Specifications**:
   Taxonomy specifications validate against standard profiles and return a typed
   `TaxonomySpecs` dictionary (`profile`, optional `canonical_indices`).
- **Explicit `index_base` Support**:
   Unified 0-based and 1-based raster indexing across label and domain
   categorical rasters, enforcing strict validation against class index bounds.
- **Harmonize Submodule Decomposition**:
   Decomposed monolithic manifest handling into `landseg.geopipe.harmonize.manifest`
   (`schema.py`, `normalizer.py`, `compiler.py`), separated raster processing
   into `landseg.geopipe.harmonize.rasters` (`stack.py`, `mask.py`, `metadata.py`,
   `spatial.py`), and extracted pipeline coordination into `pipeline.py`.

### 2.2. Named Schemes in Dataset Sidecars and Harmonization Processing
- **Root Manifest and Sidecar Submodule**:
   Root `manifest.json` entries map raw rasters to sidecar files via the
   `"manifest"` key.
- **Manifest Sidecar Schema (`schemes`)**:
   Sidecars declare optional, semantically close named schemes:
   - **`FeatureSchemes`**: Named band groupings (e.g.,
     `rgb: ["blue", "green", "red"]`, `rgb_nir: ["blue", "green", "red", "nir"]`).
   - **`LabelSchemes`**: Named hierarchical target reclassifications (e.g.,
     `binary: {reclass: {"1": [1, 2], "2": [3]}, reclass_name: {"1": "forest", "2": "water"}}`).
   - **Domain Rasters**: Strictly enforced to have `schemes: null`.
- **Harmonized VRT Tagging & Ingestion Schema Propagation**:
   - `data-harmonize`: Coordinated by `harmonize.pipeline.harmonize_sources`,
     which writes resolved schemes into harmonized VRT dataset metadata tags
     (`schemes={cfg['name']: schemes}`).
   - `data-ingest`: Reads embedded schemes from the source VRTs via
     `io.read_schemes()` and records them into the dataset `schema.json`
     artifact under `dataset.label_schemes` and `dataset.image_schemes`.

### 2.3. Canonical Ingested DataBlocks & Core Type Normalization
- **Canonical `DataBlock` Labels**:
   Ingested `DataBlock` artifacts preserve raw categorical base label values
   without pre-baked reclassifications or synthetic label stacks.
- **Zero-Based `label_band_map`**:
   `label_band_map` in `DataSchema` (`io_conventions`) is strictly 0-based,
   enabling direct ndarray slice indexing into `DataBlock.data.label`.
- **Core Module & Type Renaming**:
   Standardized file and class naming across `geopipe.core`: `data_block.py`
   (`DataBlock`, `DataBlockArrays`, `DataBlockManifest`), `dataset_catalog.py`
   (`DatasetCatalog`, `DatasetCatalogEntry`), `dataset_schema.py` (`DataSchema`),
   `domain_tile_map.py` (`DomainTileMap`, `DomainTile`), `categorical.py`
   (`CategoricalSpec`, `TaxonomySpec`), and `grid_layout.py` (`GridLayout`).
- **Standardized `GridLayout` Container Mapping**:
   `GridLayout` strictly implements `collections.abc.Mapping[Coord2d, RasterWindow]`,
   providing consistent coordinate-to-window indexing, bounding box computation,
   and affine transform alignment via `offset_from()`.

### 2.4. Ingestion Feature Engineering (`add_topo` & `add_spectral`)
- **Exposed Ingestion Feature Engineering**:
   Exposed `add_topo: list[str] | None` (supporting `[slope, tpi]`) and
   `add_spectral: list[str] | None` (supporting `[ndvi, ndmi, nbr]`) across
   `_DataBlocks` schema, Hydra defaults, `BlockBuildingParameters`,
   `data_ingest.py`, and `configs/user.yaml`.
- **Automatic Band Mapping**:
   Calculated engineered layers are automatically registered into
   `image_band_map` at block assembly time, allowing them to be selected
   by name during data preparation.

### 2.5. Upstream Context & In-Memory Dataset View (`geopipe.prepare`)
- **Decoupled Architecture**:
   Cleanly separated passive upstream environment discovery from active
   experiment dataset compilation:
   - `prepare.context`: Defines `PreparationContext` and `build_preparation_context()`,
     mirroring `HarmonizationContext` and `IngestionContext`. Resolves upstream
     ingested schema and catalog paths and extracts canvas spatial metadata
     (CRS, transform) with zero business logic or user parameters.
   - `prepare.dataset`: Compiles the active in-memory experiment dataset representation:
     - `view.py`: Defines the immutable `DatasetView` container and
       `DatasetViewParameters` dataclass, coordinating catalog views,
       feature channel selections, target head topologies, and derived
       class counts. `build_dataset_view(catalog_fpath, schema, parameters, ...)`
       coordinates compilation with zero upward module imports.
     - `catalog.py`: Handles catalog view coordinate indexing and valid-pixel
       filtering without preliminary focal count guessing.
     - `semantics.py`: Implements metadata resolution and in-memory class derivation:
       - `resolve_feature_channels()`: Resolves active band names and 0-based channel
         indices against `image_band_map` using `image_schemes` or explicit band lists.
       - `resolve_target_heads()`: Resolves multi-head target hierarchies into
         canonical `TargetHeadsContext` (and `TargetHeadsSchema` in `geo_core`).
         Head ordering is structured as `[base_head, group_head, *child_slices]`.
       - `resolve_focal_head()`: Deterministically identifies the focal target head
         for spatial partitioning, supporting direct head names, class names, or
         defaulting to grouping/base heads.
       - `derive_head_class_counts()`: Computes exact per-class pixel counts for
         all target heads in memory directly from catalog raw base class counts.
- **Decoupled Pre-Partition Distribution**:
   By deriving head class counts purely in memory during dataset view compilation,
   spatial partitioning (`partition`) executes with complete multi-head class
   distribution knowledge without reading or materializing tensor blocks on disk.

### 2.6. Decoupled Spatial Partitioning (`geopipe.prepare.partition`)
- **Metadata-Driven Splitting**:
   Partitioning consumes `DatasetView` and derived class distributions purely
   in memory. It does not touch raw block arrays or write materialized tensors.
- **Spatial AOI Integration**:
   Integrates spatial AOI raster intersection (`test_aoi`, `val_aoi`, `train_aoi`),
   priority conflict resolution (`test > val > train`), and safety buffering.
- **Submodule Reorganization**:
   Modularized partition logic into `geopipe.prepare.partition.operations`
   (`aoi.py`, `filter.py`, `hydrate.py`, `score.py`, `stratify.py`), coordinated
   by `orchestration.py` and invoked via `runner.py`.

### 2.7. Deferred Post-Partition Materialization (`geopipe.prepare.materialize`)
- **Renamed and Decoupled from `normal_blocks`**:
   Modularized tensor processing into `geopipe.prepare.materialize`:
   - `materialize.py`: Implements `materialize_blocks()`, `_normalize_image()`,
     `_reclassify_labels()`, and `_purge()`.
   - `stats.py`: Houses `aggregate_image_stats()` (Welford's online algorithm)
     and `count_label()` (per-split label frequency aggregation).
   - `runner.py`: Coordinates the end-to-end materialization workflow.
   - `schema.py`: Compiles the finalized `PreparedSchema` artifact.
- **Pipeline Execution Sequence**:
   1. `prepare.context`: Resolves upstream ingestion catalog, schema, and canvas.
   2. `prepare.dataset`: Compiles in-memory `DatasetView` and derived head counts.
   3. `prepare.partition`: Splits blocks into train, validation, and test sets
      using derived class distributions and spatial AOI constraints.
   4. `prepare.materialize`: Aggregates global image statistics solely across
      the training partition, applies per-band z-score normalization to selected
      channels, constructs multi-head label stacks via `_reclassify_labels()`,
      and writes compressed `.npz` artifacts.

### 2.8. Streamlined Submodule Naming Across `geopipe`
- **Single-Word Domain Packages**:
   Streamlined all submodule names across `geopipe` into punchy, consistent
   single-word domain packages, eliminating redundant `data_` prefixes and
   verb-noun compounds:
   - `harmonize`: `manifest`, `rasters`, `taxonomy`
   - `ingest`: `blocks` (was `data_blocks`), `domains` (was `domain_maps`)
   - `prepare`: `dataset` (was `data_context`), `partition` (was
     `data_partition`), `materialize` (was `materialize_blocks`)
- **Parallel Module Triad**:
   Every pipeline stage (`harmonize`, `ingest`, `prepare`) now provides the
   identical root file triad:
   - `context.py`: Passive upstream context builder.
   - `logger.py`: Stage structured logger.
   - `pipeline.py`: High-level stage pipeline runner.

### 2.9. Dual Public Interfaces (CLI & Programmatic API)
- **CLI Configuration (`configs/user.yaml` & Hydra)**:
   Exposed `features` and `targets` sections under `data-prepare:`, mapped
   into `data.preparation` via `landseg.adapters.cli.translate`.
- **Programmatic Python API (`DataPreparationConfigurator`)**:
   Added fluent `set_features()` and `set_targets()` methods for notebook and
   script environments (`notebooks/01_data_preparation.ipynb`).

### 2.10. Project-Wide Formatting, Style, and Documentation Standards
- **Standardized Python Scans**:
   Unified Crown copyright preservation, import groupings without blank lines,
   section delimiters (`# ----- `[Symbol]` [lowercase description]`), Google-style
   docstrings, Given / When / Then test docstrings, and strict line length
   limits ($\le 80$ chars code, $\le 72$ chars docstrings/comments) across
   all modules.

### 2.11. Package-Wide Typing Standardization & Common Alias Retirement
- **Retirement of Standalone Alias Files**:
   Completely removed and retired standalone `alias.py` files across `geopipe`
   (`geopipe/alias.py`, `prepare/common/alias.py`, `ingest/common/alias.py`, and
   `ingest/domains/alias.py`).
- **Co-Located Runtime Types (`geopipe.core.grid_layout`)**:
   Relocated `RasterReader`, `RasterWindow`, and `RasterWindowDict` directly into
   `geopipe.core.grid_layout`, co-locating them with the rasterio operations
   and grid layout structures that produce and consume them.
- **Localized Submodule Aliases**:
   - `geopipe.ingest.domains.mapper`: Houses `RasterTileDict` locally, enforcing
     clean boundaries without cross-module alias leakage.
   - `geopipe.prepare.partition.operations.stratify`: Houses NumPy array typing
     primitives locally where stratification arithmetic is performed.
   - `geopipe.utils.raster_context`: Kept purely primitive using standard
     `rasterio` types directly.

### 2.12. Central Contracts Architecture (`geopipe.contracts`) & `common` Retirement
- **Centralized Contracts Module**:
   Established a top-level `landseg.geopipe.contracts` package acting as the
   single source of truth for all pipeline summary schemas and data transfer
   specifications:
   - `contracts.grid`: `WorldGridReport`, `GridReportSchema`.
   - `contracts.harmonization`: `HarmonizationReportSchema`, `ProvenanceRecord`.
   - `contracts.ingestion`: `IngestReportSchema`, `DataBlocksReport`,
     `DomainMapReport`, `BlockStats`, `DomainStats`, `ManifestStats`.
   - `contracts.preparation`: `PreparationReportSchema`, `DataPartitionReport`,
     `NormalizationReport`, `SchemaReport`, `PreparedSchema`,
     `TargetHeadsSchema`, `BlocksPartition`, `PartitionSummary`,
     `ImageBandStats`.
- **Complete Decommissioning of `common/` Submodules**:
   Decommissioned and deleted all legacy `common/` directories
   (`geopipe.harmonize.common`, `geopipe.prepare.common`,
   `geopipe.ingest.common`).
- **Dedicated First-Class Package Loggers**:
   Introduced first-class structured loggers residing directly in each package
   root:
   - `geopipe.grid.logger.GridLogger` (`grid_report.json`)
   - `geopipe.harmonize.logger.HarmonizeLogger` (`harmonize_report.json`)
   - `geopipe.ingest.logger.IngestLogger` (`ingest_report.json`)
   - `geopipe.prepare.logger.PreparationLogger` (`prepare_report.json`)
   Each logger provides consistent run tracking, elapsed duration timing,
   sub-stage report collection, and canonical summary JSON persistence.

### 2.13. Inter-Pipeline Decoupling, Context Loaders, and Core Layout Deserialization
- **Pure Contract Decoupling**:
   Decoupled `contracts.harmonization.HarmonizationReportSchema` from
   `WorldGridReport` by storing primitive reference strings (`grid_id: str`,
   `grid_fpath: str`) rather than embedding the full world grid report payload.
- **In-Memory Pipeline Context Loaders**:
   Introduced dedicated, symmetric context builders that validate upstream
   reports and construct in-memory pipeline contexts without direct inter-package
   coupling:
   - `geopipe.harmonize.context`: Houses `HarmonizationContext` and
     `build_harmonization_context()`, resolving the grid layout and source
     manifest directly via core helpers.
   - `geopipe.ingest.context`: Houses `IngestionContext` and
     `build_ingestion_context()`, replacing legacy `harmonization_inputs.py`
     and `adapter.py`.
   - `geopipe.prepare.context`: Houses `PreparationContext` and
     `build_preparation_context()`, resolving ingested catalog, schema, and canvas
     metadata.
- **Core Grid Deserialization & Report Primitives (`geopipe.core`)**:
   Promoted reusable world grid loading and inspection functions into
   `geopipe.core.grid_layout`:
   - `GridLayout.from_fpath(cls, fpath: str) -> GridLayout`
   - `load_grid_from_fpath(fpath: str) -> GridLayout`
   - `get_grid_report_fpath(output_dpath: str) -> str`
   - `read_grid_report(report_fpath: str) -> WorldGridReport`
   All helpers are lazily exported by `geopipe.core`.
- **Decoupled Execution Pipelines & Clean-Cut Retirement**:
   - Removed direct `geopipe.grid` imports from downstream execution pipelines
     (`data_harmonize.py`, `data_ingest.py`, `diagnose_overfit.py`).
   - Cleanly retired obsolete, redundant lifecycle helpers
     (`load_grid_from_config`, `load_grid_from_fpath`, `read_grid_report`) from
     `geopipe.grid.lifecycle` and `geopipe.grid.__init__`.

### 2.14. Strict Submodule Import Scope and Lazy Resolution
- **Minimal API Surface**:
   Standardized `__all__`, `typing.TYPE_CHECKING`, and dynamic `__getattr__`
   lazy module resolution across all packages (`core`, `grid`, `harmonize`,
   `ingest`, `prepare`, `contracts`), eliminating circular import risks and
   preventing internal implementation leakage.

---

## 3. Consequences

### Positive
- **Zero-I/O Partitioning**: Spatial partitioning runs entirely on catalog
  metadata and in-memory derived class distributions without touching raw
  raster blocks on disk, significantly accelerating preparation runs.
- **Strict Statistics Isolation**: Global image normalization statistics are
  computed exclusively on training split blocks, preventing data leakage into
  validation and test partitions.
- **Semantic Proximity & Reusability**: Domain experts define reusable band
  combinations and reclassification hierarchies alongside raw dataset sidecars.
- **Experiment Agility**: Data scientists test feature subsets and target
  definitions in `user.yaml` or notebooks without re-running data harmonization
  or ingestion.
- **Immutability of Ingested Blocks**: Ingested `.npz` blocks remain pure,
  canonical representations of underlying geospatial layers.
- **Clean Architectural Separation**: Cohesive boundaries between context
  resolution (`geopipe.prepare.context`), dataset view compilation
  (`geopipe.prepare.dataset`), spatial partitioning
  (`geopipe.prepare.partition`), and tensor materialization
  (`geopipe.prepare.materialize`).
- **Contract-Governed Pipeline Reports**: Standardized `geopipe.contracts`
  schemas enforce consistent, typed output summaries across all pipeline
  stages.
- **Pure Pipeline Decoupling**: Downstream pipelines consume dedicated context
  loaders and core primitives without circular or leaky cross-pipeline module
  dependencies.
- **Unified Logging Architecture**: Consistent, first-class logger
  implementations across all geopipe subpackages without fragmented `common/`
  directories.
- **Minimal, Controlled API Surfaces**: Clean encapsulation via lazy module
  resolution prevents accidental coupling between internal implementations
  and external callers.
- **Fail-Fast Validation**: Explicit resolver checks detect misconfigurations
  early (e.g., unknown bands or invalid scheme names) before block processing
  begins.

### Negative / Migration
- **Sidecar Schema Migration**: Existing dataset manifests and sidecars must
  separate `reclass` and `reclass_name` into `schemes.LabelSchemes` and remove
  them from intrinsic `categorical_specs`.
- **Target Head Selection**: Downstream model heads must specify targets that
  match either canonical raw classes or declared scheme names.
- **Contract Import Alignment**: External callers or downstream scripts
  reading pipeline summary JSONs must import TypedDict schemas from
  `landseg.geopipe.contracts` rather than deprecated submodule schemas.

---

## 4. Implementation Summary

1. **Purified Categorical Specs**: Decoupled `reclass` and `reclass_name` from
   `CategoricalSpecs` in manifest definitions, added taxonomy profiling, and
   retired `LabelSpecs`.
2. **Harmonization Modularization**: Decomposed manifest handling into
   `geopipe.harmonize.manifest` (`schema.py`, `normalizer.py`, `compiler.py`),
   modularized raster operations into `geopipe.harmonize.rasters`, and extracted
   the harmonization execution pipeline to `pipeline.py`.
3. **Dataset Sidecar Schemes**: Implemented `FeatureSchemes` and `LabelSchemes`
   in sidecars, embedding schemes into harmonized VRT metadata and persisting
   them into `schema.json`.
4. **Canonical DataBlocks & Ingestion Feature Engineering**: Preserved raw base
   labels in `DataBlock`, established 0-based `label_band_map`, and added
   `add_topo` and `add_spectral` feature engineering at ingestion time.
5. **Decoupled Preparation Context & Dataset View**: Introduced
   `geopipe.prepare.context.PreparationContext` for passive upstream environment
   resolution, and `geopipe.prepare.dataset.DatasetView` (`view.py`, `catalog.py`,
   `semantics.py`) with canonical `TargetHeadsContext` and in-memory multi-head
   count derivation.
6. **Decoupled Partitioning and Materialization**: Modularized into
   `geopipe.prepare.partition` and `geopipe.prepare.materialize`, computing
   training image stats and multi-head label stacks only after spatial
   partitioning completes.
7. **Quality Assurance & Standards**: Unified docstrings, Given / When / Then
   test specifications, line lengths, and added comprehensive unit test suites
   across `geopipe.prepare` (100% passing).
8. **Typing Architecture & Standalone Alias Retirement**: Retired standalone
   `alias.py` files across `geopipe`, co-locating raster types in
   `geopipe.core.grid_layout`, localizing `RasterTileDict` in `domains.mapper`,
   and isolating NumPy array aliases in `stratify.py`.
9. **Contracts Module & Common Decommissioning**: Created `landseg.geopipe.contracts`
   as the central report schema authority (`contracts.grid`, `contracts.harmonization`,
   `contracts.ingestion`, `contracts.preparation`), decommissioned all `common/`
   directories, and established dedicated package loggers (`GridLogger`,
   `HarmonizeLogger`, `IngestLogger`, `PreparationLogger`).
10. **Pipeline Decoupling & Context Loaders**: Decoupled `HarmonizationReport`
    to primitive grid references, introduced `HarmonizationContext`,
    `IngestionContext`, and `PreparationContext` (retiring `harmonization_inputs.py`),
    promoted `GridLayout.from_fpath`, `load_grid_from_fpath`, `get_grid_report_fpath`,
    and `read_grid_report` to `geopipe.core`, and completely decoupled downstream
    execution pipelines from `geopipe.grid`.
11. **Streamlined Submodule Naming Across Geopipe**: Aligned all subpackages across
    `geopipe` to concise 1-word domain packages (`ingest`: `blocks`, `domains`;
    `prepare`: `dataset`, `partition`, `materialize`), standardizing the root file triad
    (`context.py`, `logger.py`, `pipeline.py`) across all three pipeline stages.
