# Architecture Overview

**Last updated: 2026-11-26**

---

## Purpose

A deterministic, artifact-driven framework for preparing geospatial raster data,
running reproducible deep learning training sessions, and conducting diagnostic
and hyperparameter study workflows.

This document defines **how the system fits together** — its high-level
architectural layers, core data and execution flow, structural invariants, and
governing design principles.

For module-level layouts and step-by-step guides, refer to:
- [project_structure.md](./project_structure.md): Comprehensive package and repository directory tree.
- [workflow_chart.md](./workflow_chart.md): Visual stage-by-stage pipeline and execution flow diagram.
- [data_preparation.md](./data_preparation.md): Guide for world grid, raster harmonization, ingestion, and preparation.
- [preflight_readiness.md](./preflight_readiness.md): Pre-flight readiness diagnostic audit guide.
- [experiment_directory_layout.md](./experiment_directory_layout.md): Canonical experiment directory and artifact organization.

---

## Core Flow

```text
Raw Geospatial Data (GeoTIFFs / Rasters / Manifests) & Domain Knowledge
  │
  ▼
Geospatial Pipeline (`landseg.geopipe`) & Knowledge Layer (`landseg.knowledge`)
  [0. Mosaic ──► 1. World Grid ──► 2. Harmonization ──► 3. Ingestion ──► 4. Preparation]
  │
  ▼
Artifacts & Run Ledgers (`landseg.artifacts`)
  [Grids, Harmonized VRTs, Block Pool, DataSpecs, Checkpoints, Ledgers]
  │
  ▼
Pre-Flight Readiness Validation (`landseg.execution.preflight`)
  [Lineage, Storage, Spatial Domain, & Hardware Probes ──► Diagnostic Audit]
  │
  ▼
Execution Dispatch Layer (`landseg.execution`)
  [Unified Dispatcher: Atomic Pipelines vs. Composite Workflows]
  │
  ▼
Model Representation (`landseg.models`) & Runtime Session (`landseg.session`)
  [DataLoaders ──► Engines ──► Multi-Task Objectives ──► Callbacks & Orchestration]
  │
  ▼
Artifact Results (`SessionResults`), Evaluated Checkpoints, Visual Diagnostics, & Study Reports
```

---

## Architectural Layers (What Each Owns)

### 1. Geospatial Pipeline ([`geopipe`](../src/landseg/geopipe))
- Owns pure, deterministic raster transformations, stage orchestration, and spatial layouts:
  - **Rasters Subsystem (`rasters`)**: Consolidates low-level raster I/O, window
    reading, geometry and coordinate validation, CRS checks, bounds
    reprojection, masking, virtual mosaicking, and VRT manipulation. Acts as
    the exclusive raster access surface across the framework, eliminating
    raw raster library dependencies from individual stages.
  - **World Grid (`grid`)**: Defines canonical projected CRS, tiling origin,
    bounding extent, tile dimensions, and stride via [`GridLayout`](../src/landseg/geopipe/grid/layout.py).
  - **Data Harmonization (`harmonize`)**: Reprojects, warps, and stacks
    heterogeneous multi-band rasters into aligned Virtual Rasters (VRT) with
    central manifest metadata compilation (`DatasetManifestBuilder`, `SensorPreset`)
    and processor composition.
  - **Data Ingestion (`ingest`)**: Slices harmonized rasters into unpartitioned
    canonical `.npz` data blocks, extracts categorical domain maps and PCA,
    resolves block collisions (`skip` vs `overwrite`), and registers batch runs.
  - **Data Preparation (`prepare`)**: Provides an in-memory dataset catalog view,
    applies geographic AOI partitioning (train/val/test splits), computes
    online Welford normalization statistics, and materializes datasets.
- Exposes lazy public facades on top-level `landseg.geopipe`:
  - **Stage Runners (`stages.py`)**: Public `run_*_stage` entrypoints encapsulating
    stage loggers, `build_*_context`, execution, and ledger updates.
  - **Contract Validation (`validation.py`)**: Plain-data `validate_*` wrappers for
    pre-flight contract probes.
  - **DataSpecs & Diagnostics (`factory.py`, `diagnostic.py`)**: Emits runtime
    [`DataSpecs`](../src/landseg/core/data_specs.py) (`build_dataspec`, `build_overfit_dataspec`)
    free of session training loop dependencies.

---

### 2. Domain Knowledge & Embeddings ([`knowledge`](../src/landseg/knowledge))
- Owns curated ecological domain assets and taxonomic alignment:
  - **Species Profiles & Taxonomies (`knowledge.taxonomy`)**: Defines
    `TaxonomySpec` and validates target class taxonomies against official Forest
    Resources Inventory (FRI) species codes.
  - **Embeddings & Similarity (`knowledge.embeddings`, `knowledge.resolver`)**:
    Encodes ecological descriptions into dense vector spaces and resolves
    head taxonomies and cosine similarity matrices (`resolve_head_knowledge`)
    for regularizers, `geopipe.factory`, and `geopipe.diagnostic`.
- Decoupled as an acyclic domain asset configured via `execution.knowledge_dir`.
  Missing or null knowledge paths gracefully bypass ecological regularization
  without blocking execution.

---

### 3. Artifacts & Lifecycle Policy ([`artifacts`](../src/landseg/artifacts))
- Centralized persistence, validation, storage primitives (`storage.npzio`,
  `storage.scratch`), and lifecycle governance via
  [`controller`](../src/landseg/artifacts/controller.py).
- Implements [`ArtifactPolicy`](../src/landseg/artifacts/policy.py): determines whether to `build`, `reuse`,
  or `overwrite` artifacts based on file presence, content hashes, and CLI flags.
- Tracks multi-batch run lineage in append-only JSON run ledgers
  (`harmonization_runs.json`, `ingestion_runs.json`).
- Governs canonical disk paths (`artifacts.paths`) via side-effect-free read
  handles (`from_root`, `open_run`, `from_run_folder`), explicit run
  allocation (`allocate_run`), and schema-marker session lookup
  (`resolve_session_run`) for grids, harmonized batches, block pool tiles,
  prepared datasets, checkpoints, logs, plots, and previews.

---

### 4. Pre-Flight Readiness Validation ([`preflight`](../src/landseg/execution/preflight))
- Non-destructive, dry-run diagnostic audit engine executed before allocating
  heavy GPU and filesystem compute resources.
- Evaluates a canonical probe hierarchy:
  - **Lineage**: Upstream stage prerequisites and ledger run presence.
  - **Filesystem**: Output directory writability and safe overwrite detection.
  - **Domain & Spatial**: CRS compatibility, raster origin alignment, raw
    manifest schemas, and model dimension parity (delegating stage contract checks
    through top-level `landseg.geopipe.validate_*`).
  - **Hardware**: CUDA availability, device detection, and VRAM headroom.
- Emits standardized 120-column terminal dashboards and structured JSON reports.
- **Diagnostic, not authoritative**: Preflight passed does not guarantee stage
  execution success (`preflight checks passed != geopipe stage guarantees to run`).
  Authoritative gating occurs via config schema validation and stage runtime context
  building (`build_*_context`); converting context building into preflight checks
  was dropped to prevent scope creeping.

---

### 5. Execution Dispatcher ([`execution`](../src/landseg/execution))
- Defines *what and how execution runs* through a unified dispatcher
  ([`executor.py`](../src/landseg/execution/executor.py)).
- Enforces strict structural separation between:
  - **Atomic Pipelines ([`pipelines`](../src/landseg/execution/pipelines))**: Strict 1:1 execution invariants
    for individual stages (`world_grid`, `data_harmonize`, `data_ingest`,
    `data_prepare`, `model_train`, `model_evaluate`), delegating geospatial stages
    through top-level `landseg.geopipe.run_*_stage` facades.
  - **Composite Workflows ([`workflows`](../src/landseg/execution/workflows))**: Orchestrated multi-stage execution
    flows (`batch_ingest`, `diagnose_overfit`, `e2e_intake`, `e2e_experiment`,
    `study_sweep`, `study_analysis`) consuming top-level `landseg.geopipe`,
    `landseg.session`, and `landseg.study` entrypoints.
- Resolves configuration and artifact lifecycles; delegates domain construction
  to top-level domain facades.

---

### 6. Models ([`models`](../src/landseg/models))
- Neural network architecture representations only (no training loops, loss
  functions, or optimizer logic).
- Modular decomposition:
  - **Backbones (`backbones`)**: Feature extraction encoders/decoders (UNet,
    UNet++, UNet+++).
  - **Conditioners (`core.conditioner`)**: Auxiliary domain conditioning via
    feature concatenation or Feature-wise Linear Modulation (FiLM).
  - **Heads (`core.heads`)**: Single-task and multi-task prediction heads
    (segmentation, classification, regression).
  - **Frames (`frames`)**: Composites that wire backbones, conditioners, and
    prediction heads into unified networks conforming to [`ModelProtocol`](../src/landseg/core/model_protocol.py).
- Instantiated declaratively via [`factory.py`](../src/landseg/models/factory.py).

---

### 7. Session Runtime ([`session`](../src/landseg/session))
- Defines *how training and evaluation execute* at runtime.
- Decomposed into decoupled runtime subsystems:
  - **Data (`session.data`)**: PyTorch datasets, custom collation, memory
    pinning, and dataloader factories.
  - **Engine (`session.engine`)**: Batch runner, epoch runner,
    optimizer/scheduler builders, and multi-task objectives (loss functions,
    evaluation metrics, and spectral/spatial/ecological regularizations).
  - **Instrumentation (`session.instrumentation`)**: Event callback dispatcher,
    tracking facade (TensorBoard, MLflow, CSV/console), and report formatters
    (`formatters.report`).
  - **Orchestration (`session.orchestration`)**: Continuous or curriculum
    runners, phase transitions, and training state preservation, indexing
    saved checkpoints into `CheckpointCatalog` (`checkpoints/status.json`) for
    deterministic `final` selection (`last_phase_best`).
  - **Checkpoint Catalog (`session.orchestration.catalog`)**: Reads and validates
    per-phase checkpoint paths (`checkpoint_catalog/v0.1.0`) with single-file
    legacy fallback.
  - **Preview Inference (`session.preview`)**: Representative test-block selection
    and stitched inference (`select_representative_block`, `run_preview_inference`)
    feeding `landseg.visual` without coupling visualization to session data/models.
- Pre-flight assembled before execution; supports evaluation-only sessions
  without training scaffolding, emitting structured [`SessionResults`](../src/landseg/core/session_results.py).

---

### 8. Dual-Surface Configuration ([`configs`](../src/landseg/configs) & [`adapters`](../src/landseg/adapters))
- Enforces strict separation between core schema contracts and user-facing
  ergonomic surfaces, resolving `adapters.api` and `adapters.cli` lazily from
  [`landseg.adapters`](../src/landseg/adapters/__init__.py):
  - **Core Schemas (`configs.schema`)**: Strongly typed Python dataclass
    hierarchy ([`RootConfig`](../src/landseg/configs/schema/root.py)), strictly isomorphic and isolated from
    CLI/notebook ergonomics, validating sweep preset names against `SWEEP_PRESETS`
    in `_constants.py`.
  - **Hydra Trees (`configs.hydra`)**: Modular YAML composition defaults.
  - **User CLI Surface (`adapters.cli`)**: Task-specific, modular recipes
    (`configs/*.yaml`) resolved directly into structured Hydra configs
    without intermediate translation layers; supports external recipe paths
    via `execution.recipe_fpath` / `--recipe`, and unified environment
    fallbacks (`${oc.env:LANDSEG_*}`).
  - **Programmatic API Surface (`adapters.api`)**: Notebook-first configurators
    with typed, programmatically safe builder methods (without eagerly loading
    Hydra `cli` or Optuna `study` modules). Implements canonical decoupled stage I/O
    (`upstream_dpath`, `output_dpath`) with single-upstream provenance forwarding
    and auxiliary `from_experiment_root(...)` macros (ADR-0067).

---

### 9. Study & Hyperparameter Sweeps ([`study`](../src/landseg/study))
- Orchestrates automated hyperparameter search and post-hoc trial analysis:
  - **Sweep (`study.sweep`)**: Optuna objective presets, trial parameter
    samplers, and multi-trial sweep orchestration.
  - **Analysis (`study.analysis`)**: Pareto frontier evaluation, metric
    summaries, and study report exports.

---

### 10. Enterprise MLOps & Storage Resolution ([`mlops`](../src/landseg/mlops))
- Provides standardized cloud workspace and governance services across CLI and
  notebooks:
  - **Storage Resolution (`mlops.storage`)**: Precedence-based resolution of
    `exp_root` across Unity Catalog Volumes (`/Volumes/...`) and local POSIX
    filesystems; environment-safe two-tiered Optuna SQLite storage routing and
    volume synchronization.
  - **Model Registry Integration (`mlops.registry`)**: Standardized model
    registration to Unity Catalog (`models:/<catalog>.<schema>.<model_name>`)
    with governance tags.

---

### 11. Visual Presentation & Diagnostics ([`visual`](../src/landseg/visual))
- Pure, headless visualization layer (Matplotlib `Agg`) decoupled from host
  interactive globals, artifact layout discovery, training loops, model
  construction, and dataloaders (strictly path-in / path-out with zero
  `landseg.artifacts` imports):
  - **Palettes & Colorization (`visual.color`)**: Standardized land cover and
    forestry palettes (`DEFAULT_ERROR_PALETTE`, `DEFAULT_MATCH_PALETTE`,
    `FORESTRY_FRI_PALETTE`, `NLCD_16_PALETTE`), categorical color mapping,
    and continuous colormaps.
  - **Curves & Trends (`visual.curves`)**: Multi-head training and validation
    metric plotting from in-memory step records to explicit PNG file paths.
  - **Qualitative Previews (`visual.preview`)**: Pure side-by-side input, ground
    truth, and multi-head prediction masks rendered directly from
    [`InferStepResults`](../src/landseg/core/session_results.py) and
    [`DataSpecs`](../src/landseg/core/data_specs.py).
  - **Diagnostic Heatmaps (`visual.diagnostics`)**: Normalized confusion matrices
    and per-class metric bar charts integrated into model evaluation.
  - **Spatial Stage-Zero Visuals (`visual.spatial`)**: World grid tiling layouts,
    AOI partition footprints, class pixel frequency audits, and multi-band
    spectral composites.

---

### 12. Execution Runtime & Concurrency ([`runtime`](../src/landseg/runtime))
- Consolidates foundational host and process runtime facilities:
  - **Platform & Compute Environment (`runtime.platform`)**: Cloud cluster
    detection, Databricks runtime introspection, and git lineage probes.
  - **Hardware Resolution (`runtime.hardware`)**: Execution device resolution,
    device querying, and CPU/GPU hardware capability probes.
  - **Parallel Concurrency (`runtime.concurrency`)**: Core-aware worker pool sizing
    (`ParallelExecutor`) supporting threads and processes with progress bars.
  - **Interactive Session Safety & Widgets (`runtime.interactive`)**: Multi-runtime
    notebook widgets and graceful `KeyboardInterrupt` / CUDA memory cleanup
    via `safe_training_session`.
  - **Unified Logging (`runtime.logger`)**: Formatted file and console loggers
    with structured run reports.
  - **Contract & Type Validation (`runtime.validator`)**: Zero-dependency primitive
    type, choice, and directory boundary validation.

---

## Key Rules

- **Top-level lazy facades across subsystems**: Cross-subsystem consumers in
  `src/landseg/`, `scripts/`, and `notebooks/` import only top-level `landseg.<pkg>`
  public APIs (with zero-dependency `landseg.runtime.validator` leaf imports as the
  only permitted internal submodule exception).
- **No implicit recomputation**: All data exchange passes through verified
  artifacts governed by [`artifacts.controller`](../src/landseg/artifacts/controller.py).
- **Pure geospatial computation & encapsulated stage orchestration**:
  [`geopipe`](../src/landseg/geopipe) internals never track training state or make
  lifecycle decisions outside injected policies; external callers invoke
  `geopipe`'s top-level `stages`, `validation`, and `diagnostic` facades rather
  than internal stage subpackages.
- **No construction in execution pipelines**: Execution pipelines orchestrate
  lifecycle contracts; stage-specific construction is delegated to domain
  factories and facades.
- **Pre-flight readiness before execution**: Non-destructive diagnostic audits
  validate hardware, spatial alignment, and lineage before compute is allocated.
- **Decoupled model from session**: [`models`](../src/landseg/models) strictly owns neural network
  architectures ([`ModelProtocol`](../src/landseg/core/model_protocol.py)); [`session`](../src/landseg/session) owns dataloaders, loss functions,
  metrics, optimization, preview inference, and training loops.
- **Pure visual presentation**: [`visual`](../src/landseg/visual) renders figures
  from in-memory step records, explicit PNG paths, and [`core`](../src/landseg/core)
  contracts without importing [`artifacts`](../src/landseg/artifacts), discovering
  runs, or constructing models, schemas, or dataloaders.
- **DataSpecs is the only dataset contract**: Model, session, and visual layers
  interface with geospatial data exclusively through [`DataSpecs`](../src/landseg/core/data_specs.py).
- **Isomorphic schemas & decoupled surfaces**: Ergonomic user YAML configs and
  API configurators translate into immutable structured config dataclasses without
  pulling `study`/`optuna` or UI formatting into core schemas.
- **Decoupled stage I/O & single-upstream lineage**: Stages operate on
  explicit upstream and output paths without requiring a shared directory root;
  model training stages depend strictly on preparation outputs with lineage
  forwarded via schemas ([ADR-0067](../docs/ADRs/ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md)).
- **Strict 1:1 Atomic Pipelines vs Composite Workflows**: Atomic pipelines
  execute exactly one stage with deterministic lifecycle contracts; multi-stage
  loops belong in composite workflows.

---

## Mental Model

- **Geopipe prepares data**: World grid, raster harmonization, canonical block
  pooling, AOI-aware dataset preparation, and public stage/validation/diagnostic facades.
- **Artifacts guarantee reproducibility**: Centralized lifecycle policy (build,
  reuse, overwrite), canonical path contracts, storage primitives, and run ledgers.
- **Preflight validates readiness**: Non-destructive dry-run audits across
  lineage, storage, spatial alignment, and hardware VRAM headroom.
- **Models define representations**: Modular backbones, conditioning, heads,
  and model frames.
- **Session executes training & previews**: Data streaming, batch/epoch execution
  engines, multi-task objectives, preview inference, callbacks, and curriculum orchestration.
- **Visual renders diagnostics**: Pure headless plots, heatmaps, spatial layouts,
  and qualitative prediction comparisons.
- **Execution orchestrates dispatch**: Unified dispatcher separating atomic
  pipelines from composite workflows.
- **Configuration adapts ergonomics**: Strongly typed Hydra schemas decoupled
  from ergonomic YAML configs and lazy API/CLI adapters.

---

## Guarantees

- **Deterministic outputs**: Identical input data, configs, and seeds yield
  identical artifacts and model states.
- **No silent rebuilds or accidental overwrites**: Lifecycle policy explicitly
  controls reuse vs rebuild; run ledgers maintain full auditability.
- **Clear boundary separation**: Build-time geospatial transformations, runtime
  GPU sessions, and headless figure rendering interact strictly via public facades
  and `landseg.core` contracts.
- **Non-destructive validation**: Pre-flight readiness probes verify environment
  integrity without mutating disk state or consuming training resources.
- **Isomorphic configuration contracts**: Core dataclasses validate contracts
  without leaking user-interface, Optuna, or formatting abstractions.
