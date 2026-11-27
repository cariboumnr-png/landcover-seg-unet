# Multi-Modal Landcover Classification Framework

[English](README.md) | [Francais](README_fr.md)

> Plain-language summary:
> This project prepares geospatial raster data and trains deep-learning models
> for pixel-level land-cover classification. It organizes data into reusable
> artifacts, builds segmentation models from configuration, and runs
> reproducible training, evaluation, and study workflows.

`landseg` is a modular, artifact-driven framework for land-cover segmentation.
It combines satellite imagery, optional topographic inputs, and optional domain
features through a deterministic geospatial preparation pipeline and a
session-based PyTorch runtime.

The current model stack centers on configurable U-Net-style segmentation models,
including U-Net, U-Net++, and U-Net+++ bodies. The runtime supports multi-head
outputs, configurable losses, segmentation metrics, optimizer wiring, callbacks,
and dashboard adapters. Configuration is split between user-facing root settings
and the packaged Hydra/structured-schema configuration tree.

## Project Status

This repository is in active research and experimental development. The main
data preparation and model training workflow is usable, but module boundaries,
configuration surfaces, and advanced study APIs can still evolve.

Currently usable:

- Data ingestion and experiment-scoped data preparation
- Artifact-backed grid, domain, data block, manifest, and dataset construction
- Model training and standalone model evaluation commands
- Overfit diagnostics for end-to-end stack validation
- Pre-flight readiness validation and dry-run diagnostics
- TensorBoard and MLflow dashboard adapter code paths
- Optuna-oriented study sweep and study analysis command entry points
- Production notebook architecture, interactive widgets, and interruption
  safety (`landseg.runtime`, ADR-0066, ADR-0069)
- Enterprise MLOps, Unity Catalog Volumes & Model Registry integration
  (`landseg.mlops`, ADR-0066)
- Headless visualization: training curves, prediction previews, evaluation
  diagnostics, and spatial plots (`landseg.visual`, ADR-0069)

Still maturing:

- Public programmatic API ergonomics
- Study/sweep configuration guarantees
- Standardized evaluation exports and reporting schemas
- Long-term compatibility guarantees for internal config fields

## Documentation

- [Repository structure](./docs/project_structure.md)
- [Workflow chart](./docs/workflow_chart.md)
- [Data preparation guide](./docs/data_preparation.md)
- [Pre-flight readiness guide](./docs/preflight_readiness.md)
- [Interactive notebooks](./notebooks/README.md)
- [Architecture decision records](./docs/ADRs/)

## Core Concepts

### Ingestion Artifacts

Raw rasters are transformed into reusable, grid-aligned artifacts such as world
grids, domain maps, data blocks, manifests, and schemas. These artifacts are the
bridge between geospatial raster formats and tensor-oriented training inputs.

### DataSpecs

Prepared artifacts are assembled into `DataSpecs`, which describe model inputs,
dataset splits, normalization, class structure, and other dataset contracts used
by the runtime.

### Models

Models are built from configuration through `landseg.models`. The model layer
owns neural network construction: backbones, frames, heads, domain helpers,
conditioning, and safety validation. Training objectives and metrics are kept in
the session runtime rather than in model definitions.

### Sessions

Sessions assemble the runtime surface for training or evaluation:

- datasets and dataloaders
- model bindings
- heads, losses, metrics, constraints, and regularization tasks
- optimizers
- epoch/runtime executors
- callbacks, tracking, dashboards, and report formatting
- orchestration policies and runners

### Execution Layer

The execution layer dispatches a named command (atomic execution pipelines or
composite workflows), resolves configuration, coordinates artifact resolution,
and delegates core work to factories and session runners. Pipeline and workflow
implementations are intentionally thin.

## Installation

Python 3.12 or newer is required.

```bash
pip install .
```

This installs the `landseg` console script:

```bash
landseg command=default
```

For running in remote environments (such as Databricks job compute nodes or VMs)
without installing the package, you can run the bootstrap entry point:

```bash
python scripts/run.py command=default
```

## Configuration

Workflows are driven by task-specific modular recipes under `configs/*.yaml`
(such as `world_grid.yaml`, `data_harmonize.yaml`, `data_ingest.yaml`,
`data_prepare.yaml`, `model_train.yaml`, and `end_to_end/`). Recipes resolve
directly into the structured Hydra configuration tree. Users can specify a
recipe via `--recipe <path>` (or `execution.recipe_fpath`).

The configuration layers are:

- Task recipes: `configs/*.yaml` (or custom external recipe files)
- Packaged Hydra composition defaults: `src/landseg/configs/hydra/`
- Structured Python config contracts: `src/landseg/configs/schema/`
- Decoupled stage I/O: individual stages accept explicit `output_dpath` and
  `upstream_dpath`, enabling cloud volumes and lakehouse storage to be targeted
  independently ([ADR-0067](./docs/ADRs/ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md))
- Base path fallbacks: auxiliary convenience preset configurable via
  `execution.exp_root` (`LANDSEG_EXP_ROOT`, default `None` / `./experiment`),
  `execution.recipe_dir` (`LANDSEG_RECIPE_DIR`, default `./configs`), and
  `execution.knowledge_dir` (`LANDSEG_KNOWLEDGE_DIR`, default `./knowledge`)
- Development overrides: resolved from `execution.dev_cfg`
  (typically defaults to the `AUX_SETTINGS_PATH` environment variable)

Before running data pipelines, read the
[data preparation guide](./docs/data_preparation.md) and organize local inputs
under the configured experiment root.

## Command Usage

Command names are registered in `landseg.execution.executor`.

### 0. World Grid Generation

Build and persist the canonical spatial tiling world grid artifact from a
reference raster or explicit extent parameters.

```bash
landseg command=world-grid
```

### 1. Data Harmonization

Harmonize, reproject, and resample raw raster datasets (features, labels,
and domain masks) onto the canonical world grid canvas.

```bash
landseg command=data-harmonize
```

### 2. Data Ingestion

Build canonical unpartitioned data blocks from harmonized rasters and the
world grid.

```bash
landseg command=data-ingest
```

### 3. Data Preparation

Build experiment-scoped artifacts from ingested data blocks, including
geographic AOI partitioning, splits, normalization statistics, and schemas.

```bash
landseg command=data-prepare
```

### 4. Model Training

Construct and run a full training session from prepared artifacts.

```bash
landseg command=model-train
```

### 5. Model Evaluation

Run evaluation from prepared artifacts and a trained checkpoint.

```bash
landseg command=model-evaluate command.model_evaluate.checkpoint=path/to/checkpoint
```

### 6. Overfit Diagnostic

Run a constrained end-to-end diagnostic on a small scope to validate model,
dataset, loss, optimizer, metric, and execution wiring.

```bash
landseg command=diagnose-overfit
```

### 7. Study Sweep

Run the Optuna-oriented study sweep entry point.

```bash
landseg command=study-sweep
```

### 8. Study Analysis

Analyze study results through the study analysis entry point.

```bash
landseg command=study-analysis
```

### 9. Pre-Flight Validation

Run non-destructive pre-flight audits to verify dependencies, ledger integrity,
spatial contracts, and compute resources prior to execution.

```bash
# Audit full environment or specific pipeline targets
landseg command=preflight # run over all commands
landseg command=preflight execution.preflight_target=model-train # targeted command
```

For detailed probe specifications, status codes, and terminal dashboards, see the
[pre-flight readiness guide](./docs/preflight_readiness.md).

## Artifact And Output Layout

In local workstation workflows, experiment I/O is normally rooted under the
configured experiment directory. In the default working tree this corresponds to:

```text
experiment/
|-- input/       Local source inputs
|-- artifacts/   Reusable generated artifacts
`-- results/     Execution and session outputs
```

In distributed and cloud environments (e.g. Databricks Unity Catalog Volumes),
stages operate with fully decoupled I/O ([ADR-0067](./docs/ADRs/ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md)).
Each stage takes explicit `output_dpath` and `upstream_dpath` directories,
allowing world grids, harmonized rasters, ingested pools, prepared datasets,
and training runs to reside across distinct storage volumes or catalogs.

Artifacts are intended to be the source of truth for reproducibility. The
framework resolves, reuses, rebuilds, or validates artifacts through centralized
artifact policy code rather than requiring users to manually manage intermediate
files.

## Package Boundaries

The current source layout is:

```text
src/landseg/
|-- adapters/        CLI and programmatic API entry surfaces
|-- artifacts/       Artifact paths, persistence, lifecycle policy, checkpoints
|-- configs/         Hydra YAML defaults and structured config schemas
|-- core/            Shared contracts and result types
|-- execution/       Command dispatch, pipelines, workflows, and preflight engine
|-- geopipe/         Geospatial rasters subsystem, grid, harmonize, ingest, prepare
|-- knowledge/       Domain taxonomy validation, species embeddings, similarity matrices
|-- mlops/           Unity Catalog model registry and Optuna storage integration
|-- models/          Model frames, backbones, heads, conditioning, factories
|-- runtime/         Host platform, hardware, concurrency, logging, widgets, validation
|-- session/         Runtime data, engines, tasks, instrumentation, orchestration
|-- study/           Sweep and analysis utilities
|-- testing/         Synthetic dummy data for tests and notebook smoke runs
`-- visual/          Headless plotting, palettes, previews, and diagnostics
```

Each top-level package exposes a lazy public facade; other subsystems import
only from that facade (ADR-0069). For a fuller map, see
[docs/project_structure.md](./docs/project_structure.md).

## Tracking And Instrumentation

Training and evaluation events are emitted through callback-based
instrumentation. The current codebase includes:

- callback dispatch and logging callbacks
- training, validation, and inference tracking hooks
- TensorBoard dashboard adapter
- MLflow dashboard adapter
- report formatting helpers, with figures rendered by `landseg.visual`

These surfaces are still being refined, especially around standardized preview
generation, evaluation exports, and comparison reports.

## Roadmap

Recently completed or stabilized:

- Artifact path handles and final checkpoint resolution ([ADR-0070](./docs/ADRs/ADR-0070-artifact-path-handles-and-final-checkpoint-resolution.md)):
  side-effect-free `artifacts.paths` handles with explicit `allocate_run`,
  schema-marker `resolve_session_run`, `session.CheckpointCatalog` resolving
  `.final` via policy `last_phase_best`, and path-in/path-out `landseg.visual`
  with zero `artifacts` imports.
- First-class module boundary hardening ([ADR-0069](./docs/ADRs/ADR-0069-first-class-module-boundary-hardening-and-subsystem-streamlining.md)):
  retired `landseg.utils` and `landseg.notebooks`, lazy public facades for
  every top-level package, one-way `geopipe` -> `knowledge` dependency, and a
  rendering-only `landseg.visual` fed by `session` inference.
- Cloud raster I/O and execution hardening ([ADR-0068](./docs/ADRs/ADR-0068-cloud-raster-io-and-execution-hardening.md)):
  local scratch staging, materialized GeoTIFF warping, sequential npz writes
  for Unity Catalog Volumes, the `landseg.runtime` subsystem, and
  device-aware AMP dtype selection.
- Decoupled Stage I/O and Auxiliary Experiment Root ([ADR-0067](./docs/ADRs/ADR-0067-decoupled-stage-io-and-auxiliary-experiment-root.md)):
  canonical unification of `upstream_dpath` and `output_dpath` across all API
  configurators, single-upstream provenance forwarding via `PreparedSchema` with
  defensive sibling bundle resolution, optional `exp_root` schema decoupling,
  autonomous `ArtifactPaths` with lazy `match-case` validation, and updated
  notebook workflows.
- Production notebook architecture and Databricks integration: multi-runtime
  `NotebookWidgets`, in-cell visual telemetry, process safety
  (`safe_training_session`), and multi-task workflow orchestration readiness
  (ADR-0066; now hosted in `landseg.runtime` and `landseg.visual`).
- Enterprise MLOps subsystem (`landseg.mlops`): Unity Catalog Volumes and Model
  Registry integration, runtime environment detection, lineage harvesting, and
  two-tiered Optuna storage synchronization (ADR-0066).
- Ergonomic dataset manifest compilation fluent API (`DatasetManifestBuilder`,
  `notebooks/00_dataset_manifest.ipynb`, ADR-0065).
- Geopipe stage handlers and non-destructive pre-flight audits (ADR-0064).
- Centralized raster-ops subsystem (`landseg.geopipe.rasters`) consolidating
  windowed I/O, reprojection, and virtual GDAL VRT mosaicking (ADR-0062).

Near-term / Medium-term focus:

- Stabilize metrics reporting formats and cross-run comparison tools.
- Extended model evaluation reporting and prediction visualizers across headless
  runs.

Longer-term goals:

- Stateful workflow runners with DAG orchestration, pause/resume, and
  step-level retry mechanisms (ADR-0060 Section 6.3).
- Continual learning and replay buffer sampling architecture (ADR-0052).
- Add more model families beyond the current U-Net-style stack (e.g., vision
  transformers).
- Define stable production export paths for trained models (ONNX, TorchScript).
- Support richer cross-experiment analysis workflows.

## Contributing

This project is still experimental. Contributions should preserve the current
separation between geospatial preparation, artifact lifecycle, model
construction, session runtime, and execution pipelines.

Before large structural changes, review the ADRs in [docs/ADRs/](./docs/ADRs/)
and add or update an ADR when a decision changes module ownership, runtime
contracts, or user-facing behavior.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) and
[NOTICE](./NOTICE) for details.

Copyright His Majesty the King in right of Ontario, as represented by the
Minister of Natural Resources, 2026.

Copyright King's Printer for Ontario, 2026.
