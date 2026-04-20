# Multi-Modal Landcover Classification Framework

[English](README.md) | [Français](README_fr.md)

>***Plain‑language summary:***<br>
>*This project provides tools for preparing satellite imagery and training*
>*models that classify land cover. It helps users organize data, run deep‑learning*
>*models, and reproduce results consistently.*

A modular, reproducible deep-learning framework for pixel‑level landcover mapping.
The system fuses **Landsat spectral imagery**, **DEM‑derived topographical metrics**,
and **domain‑knowledge features** under stable **grid** and **domain** artifacts.
The pipeline is powered by PyTorch U‑Net architectures and a fully specification‑driven
data preparation workflow.

> **Project Status:**
> This repository is currently in **research / experimental** mode.
> Module boundaries and APIs are **not yet stable**.
> Runtime construction is now **session‑owned**, with execution engines acting as
> policy layers over a shared batch core. Public session and engine APIs are still
> evolving and should not yet be considered stable.


---


# 📖 Overview

This repository provides an end‑to‑end, artifact‑driven workflow for preparing
datasets and training land‑cover segmentation models.

- **Grid & Domain Artifacts**
  Deterministic world‑grid tiling and grid‑aligned domain raster mapping,
  persisted as reusable, hash‑guarded artifacts.

- **Dataprep Pipeline**
  Raster geometry validation → grid/window mapping → raw block caching →
  spectral and topographic feature derivation → label hierarchy construction →
  dataset partitioning and scoring → normalization → schema generation.

- **Schemas as Manifest of Record**
  Generated `**schema.json` artifacts serve as the authoritative description of
  dataset structure, provenance, splits, tensor shapes, label topology, and
  normalization. No user‑authored manifest is required for standard workflows.

- **Dataset Specs (`DataSpecs`)**
  A unified runtime representation derived from persisted schemas and catalogs,
  describing model inputs, class structure, splits, normalization, and optional
  domain conditioning.

- **Model Architectures**
  Multi‑head U‑Net and U‑Net variants with optional grid‑aligned domain
  conditioning.

- **Training, Evaluation & Inference**
  Runtime construction is owned by a session layer that assembles components,
  runtime state, callbacks, and execution engines. Training and evaluation are
  driven by policy‑only engines over a shared batch executor, with an optional
  phase‑based runner providing higher‑level orchestration when required.

- **Reproducibility by Construction**
  Training and inference consume only persisted artifacts (schemas, checkpoints)
  under strict hashing, schema‑gated loading, explicit  runtime state, and
  deterministic rebuild‑on‑mismatch policies, ensuring auditable and restartable
  experiments across runs and environments.

More detailed documentation available here:
- [Repository Structure](./docs/project_structure.md)
- [Workflow Chart](./docs/workflow_chart.md)

---


## ▶️ Usage

Before running any experiment, you must prepare your input rasters and organize
your project folder correctly. Please start by reading the data‑preparation guide:

📄 [**Data preparation guide**](./docs/data_preparation.md)

Once your rasters and folders are ready, configure your project using the
root‑level `settings.yaml`. This file provides a stable entry point for specifying
inputs and processing options without modifying the internal Hydra configuration
tree.

Install the framework:

    pip install .

---

### Pipeline stages

This project runs through **explicit, consecutive pipeline stages**.
Each stage produces or consumes well‑defined artifacts governed by explicit
lifecycle policies.

#### 1. Data ingestion

Process raw rasters into **stable, catalogued data blocks** aligned to a world
grid and persisted as reusable foundation artifacts:

    experiment_run pipeline=data-ingest

This stage typically needs to be run **once per dataset**, unless the input
rasters or grid configuration change.

---

#### 2. Experiment‑scoped data preparation

Prepare experiment‑specific artifacts (dataset splits, normalization, statistics,
schemas) from previously ingested data blocks:

    experiment_run pipeline=data-prepare

This stage may be rerun with different experiment configurations without
re‑ingesting raw data.

---

#### 3. Model training

Run a complete training job using the currently prepared dataset artifacts:

    experiment_run pipeline=model-train

This stage constructs a full training session, including runtime state,
execution engines, and a phase‑driven runner, from prepared dataset artifacts.

This stage consumes prepared artifacts but does not modify foundation data.

---

#### 4. Model evaluation
Run a standalone evaluation job using the currently prepared dataset artifacts
and a trained checkpoint:

    experiment_run pipeline=model-evaluate \
      pipeline.model_evaluate.checkpoint=path/to/checkpoint

This stage constructs an evaluation-only session from prepared dataset
artifacts and the supplied checkpoint, then runs inference and metric
computation on the configured evaluation split (for example, `val` or `test`)
without performing training, optimization, or checkpoint creation. It is
intended for post-training assessment and reporting, and consumes prepared
artifacts without modifying foundation data.

---

#### 5. Overfit silo test (optional)

Run a minimal overfit test on a small subset to validate the end‑to‑end stack.
This pipeline constructs a session **without a runner**, exercising the shared
execution engine directly. It does not require prior ingestion or preparation:

    experiment_run pipeline=diagnose-overfit


>🔔 These commands execute Hydra configurations from `src/landseg/configs/`. These
internal files control the framework’s behavior and should only be modified by
advanced users familiar with Hydra and the project structure. For most workflows,
all required inputs should be provided through the root‑level `settings.yaml`.

---

## 🚀 Roadmap

### Near‑Term
- Documentation refresh reflecting the session‑first runtime architecture
  and current pipeline layout (training, overfit, evaluation-only)
- Workflow diagrams and ADR cross‑references for session construction,
  runtime ownership, and evaluation/reporting boundaries
- Improved examples for training, overfit, and standalone evaluation
  pipelines

### Medium‑Term
- Refine and stabilize the public session construction surface around
  explicit session intents and supported runtime guarantees
- Standardize evaluation result outputs and downstream reporting schemas
- Optional user‑authored task / phase manifest for more declarative
  training workflows
- Incremental hardening of session and engine public APIs

### Long‑Term
- Additional model architectures
- Expanded evaluation, export, and reporting utilities built on top of
  the session/evaluator boundary
- Experiment / study‑level workflows for cross‑session comparison,
  selection, and higher‑level orchestration
- Consolidation of stable runtime components into a production‑leaning
  execution surface centered on the session / engine boundary

---

## 🤝 Contributing

This project is in an experimental phase. Module structure, naming, and CLI
behaviour may evolve. Contributions should focus on research usability unless
aligned with an approved Architecture Decision Record (ADR).

Please review active ADRs in `docs/ADRs/` to understand current design decisions.

---

## 📜 License

Licensed under the **Apache License, Version 2.0**.
See the `LICENSE` and `NOTICE` file for details.

© His Majesty the King in right of Ontario,
as represented by the Minister of Natural Resources, 2026.
© King's Printer for Ontario, 2026.
