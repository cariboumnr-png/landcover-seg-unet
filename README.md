# Multi-Modal Landcover Classification Framework

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
- [Workflow Chart](./docs/workflow.md)

---
## ▶️ Usage

Prior to running, [see here](./docs/data_preparation.md) on for instructions on preparing label and image rasters

Install the framework:

    pip install .

Run a full experiment:

    experiment_run profile=end_to_end

Run an overfit silo test:

    experiment_run profile=overfit_test

These commands execute Hydra configurations from `src/landseg/configs/`. For most users, it is recommended to provide inputs through the root‑level `settings.yaml`, which is designed for safe customization without modifying the internal configuration tree.

---
## 🚀 Roadmap

### Near‑Term (current milestone)
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

Please review active ADRs [here](./docs/ADRs/) to understand current design decisions.

---
## 📜 License

Licensed under the **Apache License, Version 2.0**.
See the `LICENSE` and `NOTICE` file for details.

© His Majesty the King in right of Ontario,
as represented by the Minister of Natural Resources, 2026.
© King's Printer for Ontario, 2026.
