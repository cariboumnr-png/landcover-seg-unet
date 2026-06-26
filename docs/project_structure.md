## Current Project Structure

Last updated: 2026-06-18

This document summarizes the repository's current package layout and the main
responsibility of each area. It favors the working boundaries that matter when
adding or moving code over a fully exhaustive file listing.

```text
./
|-- docs/                         Project documentation, diagrams, and ADRs
|   |-- ADRs/                     Architecture decision records
|   |-- images/                   Documentation images
|   |-- data_preparation.md       Data preparation guide
|   |-- workflow_chart.md         High-level workflow chart
|   `-- project_structure.md      This file
|
|-- dev/                          Ignored developer scratch/reference workspace
|
|-- experiment/                   Ignored local experiment I/O
|   |-- artifacts/                Generated grids, manifests, checkpoints, etc.
|   |-- input/                    Local experiment inputs
|   `-- results/                  Local pipeline/session outputs
|
|-- notebooks/
|   |-- 01_data_preparation.ipynb Data ingest, validation, and preparation demo
|   `-- 02_model_train.ipynb      End-to-end model training demo
|
|-- src/landseg/                  Installable Python package
|   |-- adapters/                 External entry surfaces
|   |-- artifacts/                Artifact paths, lifecycle policy, payload IO, checkpoints
|   |-- configs/                  Hydra YAML configs and structured schema dataclasses
|   |-- core/                     Project-wide contracts and result types
|   |-- execution/                Pipeline registry and top-level execution dispatch
|   |-- geopipe/                  Geospatial foundation and transform pipeline
|   |-- models/                   Model frames, backbones, heads, conditioning, factories
|   |-- session/                  Runtime session construction, orchestration, and engines
|   |-- study/                    Sweep and post-run study analysis utilities
|   |-- utils/                    Shared logging and multiprocessing helpers
|   |-- _constants.py             Shared constants
|   `-- __init__.py
|
|-- configs/                      User-facing configuration files
|   `-- user.yaml                 Main local pipeline-oriented configuration
|
|-- scripts/                      Helper scripts and entry points
|   `-- run.py                    Bootstrap script for VM/Databricks executions
|
|-- pyproject.toml                Package metadata and `landseg` console entry point
|-- README.md                     Project overview
`-- CONTRIBUTING.md               Contribution notes
```

### Package Layout

```text
src/landseg/
|-- adapters/
|   |-- api/
|   |   |-- api.py                Programmatic API facade
|   |   `-- configurators/        API helpers for ingest, prepare, and train flows
|   `-- cli/
|       |-- cli.py                Hydra-driven CLI implementation
|       |-- resolver.py           CLI config resolution helpers
|       `-- translate.py          User config translation layer
|
|-- artifacts/
|   |-- checkpoint.py             Checkpoint save/load helpers
|   |-- controller.py             Policy-driven artifact resolve/build/rebuild logic
|   |-- paths.py                  Canonical artifact path definitions
|   |-- payload_io.py             Structured payload and metadata persistence
|   `-- policy.py                 Artifact lifecycle rules
|
|-- configs/
|   |-- hydra/
|   |   |-- config.yaml           Hydra composition entry point
|   |   |-- dataspecs/            Dataset specification defaults
|   |   |-- foundation/           Data block, domain, and grid defaults
|   |   |-- models/               Model architecture defaults
|   |   |-- pipeline/             data-ingest, data-prepare, train, eval, study configs
|   |   |-- session/              Runtime, loader, optimizer, task, and orchestration configs
|   |   |-- study/                Study/sweep defaults
|   |   `-- transform/            Transform defaults
|   `-- schema/
|       |-- root.py               Root structured config schema
|       |-- utils.py              Schema helper utilities
|       `-- sections/             Section dataclasses for pipeline, session, models, etc.
|
|-- core/
|   |-- data_specs.py             Runtime data specification contract
|   |-- model_protocol.py         Model behavior protocol
|   `-- session_results.py        Structured session outputs
|
|-- execution/
|   |-- executor.py               Unified execution entry point
|   `-- pipelines/
|       |-- _registry.py          Pipeline lookup and registration
|       |-- data_ingest.py        Foundation data ingestion pipeline
|       |-- data_prepare.py       Transform/data-preparation pipeline
|       |-- model_train.py        Training pipeline
|       |-- model_evaluate.py     Evaluation pipeline
|       |-- diagnose_overfit.py   Overfit diagnostic pipeline
|       |-- study_sweep.py        Hyperparameter sweep pipeline
|       `-- study_analysis.py     Study result analysis pipeline
|
|-- geopipe/
|   |-- core/                     Foundation and transform data contracts
|   |-- foundation/
|   |   |-- common/               Shared foundation aliases
|   |   |-- data_blocks/          Data block manifests, mapping, and pipeline
|   |   |-- domain_maps/          Domain map construction and lifecycle
|   |   `-- world_grids/          World grid construction and lifecycle
|   |-- specification/            DataSpecs factory
|   |-- transform/
|   |   |-- common/               Transform aliases
|   |   |-- data_partition/       Split, filter, hydrate, and scoring pipeline
|   |   `-- normal_blocks/        Normalization stats and normalization pipeline
|   `-- utils/                    Raster context and coordinate string helpers
|
|-- models/
|   |-- backbones/
|   |   |-- base.py               Backbone base abstractions
|   |   |-- factory.py            Backbone construction
|   |   `-- unet/                 UNet, UNet++, UNet+++ bodies and components
|   |-- core/
|   |   |-- conditioner/          Concatenation and FiLM conditioning
|   |   |-- config.py             Model config structures
|   |   |-- domains.py            Domain-related model helpers
|   |   |-- heads.py              Prediction head components
|   |   `-- safety.py             Model safety/validation helpers
|   |-- frames/                   Full model frames that wire backbones and heads
|   `-- factory.py                Top-level model construction
|
|-- session/
|   |-- common/                   Shared aliases, events, and orchestration types
|   |-- data/                     Dataset and dataloader adapters
|   |-- engine/
|   |   |-- builder.py            Engine construction
|   |   |-- epoch/                Epoch executor and trainer/evaluator policies
|   |   `-- runtime/
|   |       |-- builder.py        Runtime construction
|   |       |-- executor/         Batch runtime state, objective, and executor
|   |       |-- optim/            Optimizer construction and optimization logic
|   |       `-- tasks/            Heads, constraints, losses, metrics, regularization
|   |-- instrumentation/
|   |   |-- callbacks/            Callback dispatch, logging, and tracking hooks
|   |   |-- dashboards/           TensorBoard and MLflow dashboard adapters
|   |   `-- formatters/           Report rendering and formatting
|   |-- orchestration/
|   |   |-- builder.py            Session orchestration construction
|   |   |-- policy/               Epoch and phase policies
|   |   `-- runner/               Continuous and curriculum runners
|   |-- factory.py                Session factory
|   `-- metadata.py               Session metadata
|
|-- study/
|   |-- analysis/                 Trial/result analysis helpers
|   `-- sweep/                    Optuna objective, config, and optimization helpers
|
`-- utils/
    |-- logger.py                 Logging setup
    `-- multip.py                 Multiprocessing helpers
```

### Current Boundary Notes

- `adapters/` is intentionally thin: it translates external API/CLI inputs into
  configured execution calls.
- `execution/` owns named pipeline dispatch; stage-specific work is delegated to
  `geopipe/`, `session/`, `models/`, `artifacts/`, and `study/`.
- `configs/hydra/` stores runtime YAML composition, while `configs/schema/`
  stores the structured Python config contracts.
- `geopipe/` owns geospatial data preparation up to `DataSpecs`; model training
  data loading begins under `session/data/`.
- `models/` owns neural network construction only. Training objectives, metrics,
  optimizer setup, and runtime task wiring live under `session/engine/runtime/`.
- `session/` is the main runtime layer: orchestration chooses phases/runners,
  epoch policies define train/eval behavior, and runtime tasks compute heads,
  losses, metrics, constraints, and regularization.
