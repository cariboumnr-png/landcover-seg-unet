
# Multi-Modal Landcover Classification Framework

A modular, reproducible deep-learning framework for pixel‑level landcover mapping.
The system fuses **Landsat spectral imagery**, **DEM‑derived topographical metrics**, and **domain‑knowledge features** under stable **grid** and **domain** artifacts.
The pipeline is powered by PyTorch U‑Net architectures and a fully specification‑driven data preparation workflow.

---

## Overview

This repository provides a complete end‑to‑end workflow:

- **Grid & Domain Artifacts:** Deterministic world‑grid tiling and domain raster alignment.
- **Dataprep Pipeline:** Window mapping → raster block caching → spectral/topo feature derivation → label hierarchy → normalization → scoring & dataset split → schema generation.
- **Dataset Specs:** A unified representation (`DataSpecs`) describing shapes, class topology, splits, and normalization.
- **Model Architectures:** Multi‑head U‑Net / U‑Net++ with optional domain conditioning.
- **Training Runner:** A unified training/inference controller with callbacks, metrics, losses, and preview generation.
- **Reproducibility:** Strict artifact hashing, schema validation, and rebuild‑on‑mismatch behavior.

---

## ⚙️ Current Work

**Actively implementing ADR‑0006** (packaging & entrypoints) on branch:
`packaging-entry-points`

The next major steps include:

- Converting the project into a **pip-installable package** under `src/<package_name>/`
- Adding CLI entrypoints:
  - `<package_name> prep` — run full dataprep
  - `<package_name> report` — tile/AOI EDA/QA summaries
  - `<package_name> train` — unified training workflow
  - `<package_name> infer` — optional inference & stitching

---

## 📁 Current Repository Structure (Source‑First Layout)
```
root/src
├── grid/               # generate stable world grid
│   ├── builder.py      <- module API
│   ├── io.py
│   └── layout.py
├── domain/             # mapp domain rasters to world grid
│   ├── io.py
│   ├── mapper.py       <- module API
│   ├── tilemap.py
│   └── transform.py
├── dataprep/           # process raw rasters to stable artifacts
│   ├── blockbuilder/
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py     <- module API
│   └── schema.py
├── dataset/            # wire data schema to trainer dataloading
│   ├── builder.py      <- module API
│   ├── load.py
│   └── validate.py
├── models/             # defines model structure (current: UNet, UNet++)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- module API
├── training/           # builder trainer and controller (as a unified runner)
│   ├── callback/
│   ├── common/
│   ├── controller/
│   ├── dataloading/
│   ├── heads/
│   ├── loss/
│   ├── metrics/
│   ├── optim/
│   ├── trainer/
│   └── factory.py      <- module API
└── utils/              # project utilities
```

## 🧊 Data Foundation

The system operates on **Landsat imagery** and **DEM‑derived terrain metrics**.
The dataprep pipeline:

- generates spectral indices (NDVI, NDMI, NBR)
- produces slope, aspect, TPI from DEM
- builds label hierarchies
- normalizes features globally using Welford statistics
- bundles everything into stable `.npz` blocks

All artifacts are validated via per‑file SHA‑256 + schema hashing.

---

## 🚀 Roadmap (Updated for ADR‑0005 & ADR‑0006)

### Near‑Term (current milestone)
- Package the repo into a proper Python distribution
- Add CLI entrypoints:
  - `<package_name> prep`
  - `<package_name> report`
  - `<package_name> train`
  - `<package_name> infer`
- Improve documentation and examples
- Add unit tests for dataprep + dataset + training

### Medium‑Term
- Standard tile/AOI reporting (ADR‑0005)
- Optional user‑authored task manifest
- Lightweight artifact catalog (opt‑in) for reuse across datasets

### Long‑Term
- Additional model architectures
- Cross‑sensor extension (Sentinel‑2)
- Evaluation & export utilities

---
## Contributing
Contributions welcome once the initial packaging + entrypoint refactor stabilizes.
Please see active ADRs for the current project direction.

## License
To be determined.
