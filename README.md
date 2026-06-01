# Multi-Modal Landcover Classification Framework

[English](README.md) | [Français](README_fr.md)

>***Plain‑language summary:***<br>
>*This project provides tools for preparing satellite imagery and training*
>*models that classify land cover. It helps users organize data, run deep‑learning*
>*models, and reproduce results consistently.*

A modular, artifact‑driven deep learning framework for pixel‑level landcover mapping.
The system integrates **Landsat spectral imagery**, **DEM‑derived topographic metrics**,
and **domain features** through a structured data preparation pipeline and a
session-based training runtime built on **U‑Net–style segmentation architectures**
(PyTorch implementation). Current model support includes configurable multi-head
U-Net variants, including standard U-Net, U-Net++, and U-Net3+ style backbones,
with optional convolutional, transformer, or hybrid bottlenecks. The default is
set to a conservative convolutional U-Net for baseline stability, while
transformer-enabled bottlenecks are available for experiments requiring
broader spatial context and long-range feature interaction.

> **Project Status:**
> This repository is currently in **research / experimental** mode. Module boundaries
> and APIs are **not yet stable**. Some interfaces are stable for usage, while others
> (notably advanced orchestration and study layers) are still under development.

---

# 📖 Overview

This repository provides an end-to-end workflow for preparing geospatial datasets
and training land-cover segmentation models, with a strong emphasis on reproducibility
and structured data flow.

## Core Concepts

- **Foundation Artifacts (Data Preparation)**

  Raw rasters are transformed into structured, grid-aligned artifacts
  (e.g., data blocks, domain maps). These artifacts act as the bridge between
  geospatial formats (GeoTIFF) and tensor-based model inputs.

- **Experiment Definition (DataSpecs + Model)**

  Prepared artifacts are assembled into `DataSpecs`, which define:

  - Model inputs
  - Dataset splits
  - Normalization
  - Class structure

  Models are constructed from configuration and paired with these specifications.

- **Session (Runtime System)**

  Training and evaluation are executed through a session, which assembles:

  - Dataloaders
  - Models
  - Loss functions and optimizers
  - Execution engines
  - Callbacks and instrumentation

- **Pipelines (Execution Layer)**

  CLI pipelines orchestrate the workflow. They resolve configuration and artifacts,
  then delegate construction to the system (they do not implement core logic).

## Key Features

- **Artifact-Driven Workflow**

  All intermediate and final data are stored as versioned artifacts.
  Users configure the system; artifact lifecycle and reuse are handled automatically.

- **Deterministic Data Preparation**

  Grid and domain alignment ensure consistent spatial structure across runs.

- **Specification-Driven Datasets (`DataSpecs`)**

  A single runtime object defines all model inputs, splits, and normalization.

- **Session-Based Runtime**

  Training and evaluation logic are encapsulated in a structured runtime system.

- **Decoupled Tracking (Early-Stage Dev)**

  TensorBoard support is available via callback-based instrumentation
  (no vendor lock-in).

  Additional backends (e.g., MLflow) are planned.

---

## ⚠️ Stability Notes

- **Stable for Use**

  - Data ingestion and preparation pipelines
  - Artifact-based dataset construction
  - Training and evaluation pipelines
  - TensorBoard integration (basic tracking)

- **Under Active Development**

  - Notebook-based workflows
    *(expected to become the primary entry point)*
  - Study / sweeping layer
    *(Optuna-based experimentation utilities)*
  - Session and execution APIs
    *(may evolve over time)*

---

**More detailed documentation available here**:
- [Repository Structure](./docs/project_structure.md)
- [Workflow Chart](./docs/workflow_chart.md)

---

## ▶️ CLI Entry Usage

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

    landseg pipeline=data-ingest

This stage typically needs to be run **once per dataset**, unless the input
rasters or grid configuration change.

---

#### 2. Experiment‑scoped data preparation

Prepare experiment‑specific artifacts (dataset splits, normalization, statistics,
schemas) from previously ingested data blocks:

    landseg pipeline=data-prepare

This stage may be rerun with different experiment configurations without
re‑ingesting raw data.

---

#### 3. Model training

Run a complete training job using the currently prepared dataset artifacts:

    landseg pipeline=model-train

This stage constructs a full training session, including runtime state,
execution engines, and a phase‑driven runner, from prepared dataset artifacts.

This stage consumes prepared artifacts but does not modify foundation data.

---

#### 4. Model evaluation
Run a standalone evaluation job using the currently prepared dataset artifacts
and a trained checkpoint:

    landseg pipeline=model-evaluate \
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

    landseg pipeline=diagnose-overfit


>🔔 These commands execute Hydra configurations from `src/landseg/configs/`. These
internal files control the framework’s behavior and should only be modified by
advanced users familiar with Hydra and the project structure. For most workflows,
all required inputs should be provided through the root‑level `settings.yaml`.

---

## 📊 Tracking & Visualization

Training metrics and logs are emitted via callback-based instrumentation.

- TensorBoard is currently supported *(local usage)*
- Tracking is decoupled from framework internals
- Future support planned for additional backends *(e.g., MLflow)*

---

## 🧠 Conceptual Model

The system enforces a strict separation of concerns:

- **Foundation Layer**
  Deterministic data construction
  *(grid, domain, blocks)*

- **Artifacts Layer**
  Persistence, validation, and reuse policies

- **Experiment Layer**
  `DataSpecs` and model definition

- **Session Layer**
  Runtime execution and lifecycle orchestration

- **Execution Layer**
  Pipeline selection and coordination

**Dependency Flow**

    foundation → artifacts → experiment → session → execution

## 📦 Artifact behavior (user-facing summary)

Artifacts are automatically generated and reused. Users do not manually manage lifecycle policies.

Prepared dataset artifacts are stored under:

    <user-defined-experiment-root>/artifacts

Session outputs are stored under:

    <user-defined-experiment-root>/results/run_xxxx/

Artifacts serve as the source of truth for reproducibility.

---

## 🚀 Roadmap

### Near‑Term
- Notebook-first workflow (user-friendly entry point)
- Improved experiment visualization and reporting
- Enhanced TensorBoard integration (richer logging, previews)

### Medium‑Term
- Stable study / hyperparameter sweep interface (Optuna)
- MLflow integration for experiment tracking
- Improved session configuration clarity and guarantees
- Standardized evaluation outputs and reporting schemas

### Long‑Term
- Cross-experiment comparison and study workflows
- Additional model architectures
- Production-oriented execution surface
- Extended export and deployment utilities

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