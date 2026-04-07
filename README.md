# Multi-Modal Landcover Classification Framework

[English](README.md) | [Français](README_fr.md)

>***Plain‑language summary:***<br>
>*This project provides tools for preparing satellite imagery and training models that classify land cover. It helps users organize data, run deep‑learning models, and reproduce results consistently.*

A modular, reproducible deep-learning framework for pixel‑level landcover mapping.
The system fuses **Landsat spectral imagery**, **DEM‑derived topographical metrics**, and **domain‑knowledge features** under stable **grid** and **domain** artifacts.
The pipeline is powered by PyTorch U‑Net architectures and a fully specification‑driven data preparation workflow.

> **Project Status:**
> This repository is currently in **research / experimental** mode.
> Module boundaries and APIs are **not yet stable**.
> A production‑leaning runtime (`engine/`) is planned for future milestones but **not yet included**.

---

## 📖 Overview

This repository provides an end‑to‑end workflow for preparing datasets and
training landcover segmentation models:

- **Grid & Domain Artifacts:** Deterministic world‑grid tiling and domain raster alignment.
- **Dataprep Pipeline:** Window mapping → raster block caching → spectral/topo feature derivation → label hierarchy → normalization → scoring & dataset split → schema generation.
- **Dataset Specs:** A unified representation (`DataSpecs`) describing shapes, class topology, splits, and normalization.
- **Model Architectures:** Multi‑head U‑Net / U‑Net with optional domain conditioning.
- **Training Runner:** A unified training/inference controller with callbacks, metrics, losses, and preview generation.
- **Reproducibility:** Strict artifact hashing, schema validation, and rebuild‑on‑mismatch behavior.

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

    experiment_run pipeline=ingest-data

This stage typically needs to be run **once per dataset**, unless the input
rasters or grid configuration change.

---

#### 2. Experiment‑scoped data preparation

Prepare experiment‑specific artifacts (dataset splits, normalization, statistics,
schemas) from previously ingested data blocks:

    experiment_run pipeline=prepare-data

This stage may be rerun with different experiment configurations without
re‑ingesting raw data.

---

#### 3. Model training

Run a complete training job using the currently prepared dataset artifacts:

    experiment_run pipeline=train-model

This stage consumes prepared artifacts but does not modify foundation data.

---

#### 4. Overfit silo test (optional)

Run a minimal overfit test on a small subset to validate the end‑to‑end stack.
This pipeline **does not require prior ingestion or preparation**:

    experiment_run pipeline=train-overfit


>🔔 These commands execute Hydra configurations from `src/landseg/configs/`. These
internal files control the framework’s behavior and should only be modified by
advanced users familiar with Hydra and the project structure. For most workflows, all required inputs should be provided through the root‑level `settings.yaml`.

---

## 🚀 Roadmap

### Near‑Term
- Improve documentation and examples (active)

### Medium‑Term
- Optional user‑authored task manifest

### Long‑Term
- Additional model architectures
- Evaluation & export utilities
- Gradual promotion of stable components into `engine/training`

---

## 🤝 Contributing

This project is in an experimental phase. Module structure, naming, and CLI behaviour may evolve. Contributions should focus on research usability unless aligned with an approved Architecture Decision Record (ADR).

Please review active ADRs in `docs/ADRs/` to understand current design decisions.

---

## 📜 License

Licensed under the **Apache License, Version 2.0**.
See the `LICENSE` and `NOTICE` file for details.

© His Majesty the King in right of Ontario,
as represented by the Minister of Natural Resources, 2026.
© King's Printer for Ontario, 2026.
