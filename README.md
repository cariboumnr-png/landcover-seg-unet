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
- Model training and standalone model evaluation pipelines
- Overfit diagnostics for end-to-end stack validation
- TensorBoard and MLflow dashboard adapter code paths
- Optuna-oriented study sweep and study analysis pipeline entry points

Still maturing:

- Notebook-first workflows and examples
- Public programmatic API ergonomics
- Study/sweep configuration guarantees
- Standardized evaluation exports and reporting schemas
- Long-term compatibility guarantees for internal config fields

## Documentation

- [Repository structure](./docs/project_structure.md)
- [Workflow chart](./docs/workflow_chart.md)
- [Data preparation guide](./docs/data_preparation.md)
- [Architecture decision records](./docs/ADRs/)

## Core Concepts

### Foundation Artifacts

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

### Execution Pipelines

The execution layer selects a named pipeline, resolves configuration, coordinates
artifact resolution, and delegates core work to factories and runtime modules.
Pipeline implementations are intentionally thin.

## Installation

Python 3.12 or newer is required.

```bash
pip install .
```

This installs the `landseg` console script:

```bash
landseg pipeline=default
```

For running in remote environments (such as Databricks job compute nodes or VMs)
without installing the package, you can run the bootstrap entry point:

```bash
python scripts/run.py pipeline=default
```

## Configuration

Most user workflows should start from `configs/user.yaml` under the root-level
`configs/` directory. The packaged Hydra tree under `src/landseg/configs/hydra/`
contains internal composition defaults and should be changed carefully.

The configuration layers are:

- `configs/user.yaml`: local project dataset inputs and high-level choices
- `src/landseg/configs/hydra/`: packaged Hydra composition defaults
- `src/landseg/configs/schema/`: structured Python config contracts
- Development Overrides: resolved from the path in `execution.dev_cfg`
  (typically defaults to the `AUX_SETTINGS_PATH` environment variable)

Before running data pipelines, read the
[data preparation guide](./docs/data_preparation.md) and organize local inputs
under the configured experiment root.

## Pipeline Usage

Pipeline names are registered in `landseg.execution.pipelines`.

### 1. Data Ingestion

Build foundation artifacts from raw rasters. This typically runs once per source
dataset or whenever source rasters/grid settings change.

```bash
landseg pipeline=data-ingest
```

### 2. Data Preparation

Build experiment-scoped artifacts from ingested data blocks, including splits,
normalization/statistics, and dataset schemas.

```bash
landseg pipeline=data-prepare
```

### 3. Model Training

Construct and run a full training session from prepared artifacts.

```bash
landseg pipeline=model-train
```

### 4. Model Evaluation

Run evaluation from prepared artifacts and a trained checkpoint.

```bash
landseg pipeline=model-evaluate pipeline.model_evaluate.checkpoint=path/to/checkpoint
```

### 5. Overfit Diagnostic

Run a constrained end-to-end diagnostic on a small scope to validate model,
dataset, loss, optimizer, metric, and execution wiring.

```bash
landseg pipeline=diagnose-overfit
```

### 6. Study Sweep

Run the Optuna-oriented study sweep entry point.

```bash
landseg pipeline=study-sweep
```

### 7. Study Analysis

Analyze study results through the study analysis entry point.

```bash
landseg pipeline=study-analysis
```

## Artifact And Output Layout

Local experiment I/O is normally rooted under the configured experiment
directory. In the default working tree this corresponds to:

```text
experiment/
|-- input/       Local source inputs
|-- artifacts/   Reusable generated artifacts
`-- results/     Pipeline/session outputs
```

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
|-- execution/       Pipeline registry and top-level dispatch
|-- geopipe/         Geospatial foundation and transform pipeline
|-- models/          Model frames, backbones, heads, conditioning, factories
|-- session/         Runtime data, engines, tasks, instrumentation, orchestration
|-- study/           Sweep and analysis utilities
`-- utils/           Shared logging and multiprocessing helpers
```

For a fuller map, see [docs/project_structure.md](./docs/project_structure.md).

## Tracking And Instrumentation

Training and evaluation events are emitted through callback-based
instrumentation. The current codebase includes:

- callback dispatch and logging callbacks
- training, validation, and inference tracking hooks
- TensorBoard dashboard adapter
- MLflow dashboard adapter
- report rendering/formatting helpers

These surfaces are still being refined, especially around standardized preview
generation, evaluation exports, and comparison reports.

## Roadmap

Recently completed or stabilized:

- Programmatic API surfaces for interactive environments and Jupyter Notebooks
  (`TrainingSessionConfigurator`, etc.).
- Hardening model contracts and strict configuration validation boundaries.
- Multi-head label mechanisms, regularized losses (consistency losses), and
  extended evaluation metrics.
- Initial Optuna study sweep presets and objective metrics integrations.

Near-term / Medium-term focus:

- Update workflow charts to match the current session/runtime execution split.
- Document recommended Optuna workflow guides and publish programmatic
  tutorials.
- Stabilize metrics reporting formats and cross-run comparisons.

Longer-term goals:

- Add more model families beyond the current U-Net-style stack.
- Define stable export paths for trained models and evaluation artifacts.
- Support richer cross-experiment analysis workflows.
- Continue consolidating internal boundaries as ADRs settle.

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
