
# Multi-Modal Landcover Classification Framework

A modular, reproducible deep-learning framework for pixel‑level landcover mapping.
The system fuses **Landsat spectral imagery**, **DEM‑derived topographical metrics**, and **domain‑knowledge features** under stable **grid** and **domain** artifacts.
The pipeline is powered by PyTorch U‑Net architectures and a fully specification‑driven data preparation workflow.

> **Project Status:**
> This repository is currently in **research / experimental** mode.
> Module boundaries and APIs are **not yet stable** and may change without notice.
> A production‑leaning runtime (`engine/`) is planned for future milestones but is **not part of the current codebase**.
---

## Overview

This repository provides a complete end‑to‑end workflow:

- **Grid & Domain Artifacts:** Deterministic world‑grid tiling and domain raster alignment.
- **Dataprep Pipeline:** Window mapping → raster block caching → spectral/topo feature derivation → label hierarchy → normalization → scoring & dataset split → schema generation.
- **Dataset Specs:** A unified representation (`DataSpecs`) describing shapes, class topology, splits, and normalization.
- **Model Architectures:** Multi‑head U‑Net / U‑Net with optional domain conditioning.
- **Training Runner:** A unified training/inference controller with callbacks, metrics, losses, and preview generation.
- **Reproducibility:** Strict artifact hashing, schema validation, and rebuild‑on‑mismatch behavior.

> **Note:**
> The current implementation reflects ongoing research and exploration;
> interfaces and components may evolve as the project matures.
---

## 🟦 How to Use This Project (WIP — interfaces may change)

This framework is now packaged as a Python module. After cloning:

    pip install .

You can run the full end‑to‑end experiment pipeline via:

    experiment_run

This will execute the packaged Hydra configuration located at:

    src/landseg/configs/

and apply any overrides you place in the root-level `settings.yaml`.

⚠️ **Important:**
This project is still *research‑stage* and **brittle**.
Interfaces, config structure, and behavior **may change frequently** as the
pipeline stabilizes. Use at your own risk and expect breaking changes.

Additional CLI commands (prep, report, train, infer) will be added as the
interface matures.

## 🛠️ Current Work


 **ADR‑0006 Status:**
Packaging & entrypoint refactor is **complete** and merged.

The project is now installable via `pip install` we and exposes the new CLI entrypoint: `experiment_run` that runs the entire workflow end‑to‑end (dataprep → dataset → model → training).

This is intentionally a single "everything" entrypoint for now. The internal
pipeline is still being stabilized and may be brittle.

As the framework matures, the monolithic flow will be decomposed into dedicated step‑level entrypoints such as:

    prep     — dataprep only
    report   — tile/AOI QA and diagnostics
    train    — model training
    infer    — inference & stitching

These will be introduced gradually once the underlying modules and Hydra config structure become stable.

Hydra configuration is now packaged under `src/landseg/configs/`, and users
may override settings via the root-level `settings.yaml` (WIP, non‑exhaustive).

---

## 📁 Current Repository Structure (Source‑first - collapsed to first class)
```
root/src/landseg
│
├── grid/               # generate stable world grid
│   ├── builder.py      <- module API
│   ├── io.py
│   └── layout.py
|
├── domain/             # map domain rasters to world grid
│   ├── io.py
│   ├── mapper.py       <- module API
│   ├── tilemap.py
│   └── transform.py
|
├── dataprep/           # process raw rasters to stable artifacts
│   ├── blockbuilder/
│   ├── mapper/
│   ├── normalizer/
│   ├── splitter/
│   ├── utils/
│   ├── pipeline.py     <- module API
│   └── schema.py
│
├── dataset/            # consume data schema for traininig.dataloading
│   ├── builder.py      <- module API
│   ├── load.py
│   └── validate.py
│
├── models/             # defines model structure (current: UNet, UNet++)
│   ├── backbones/
│   ├── multihead/
│   └── factory.py      <- module API
│
├── training/           # trainer and its components
│   ├── callback/
│   ├── common/
│   ├── dataloading/
│   ├── heads/
│   ├── loss/
│   ├── metrics/
│   ├── optim/
│   ├── trainer/
│   └── factory.py      <- module API
│
├── controller/         # build controller (experiment run from this)
│   ├── builder.py      <- module API
│   ├── controller.py
│   └── phases.py
│
├── utils/              # project-wide utilities
│
├── configs/            # hydra config tree shipped with package
│
└── cli/                # CLI scripts
    └── end_to_end.py   <- primary entrypoint for `experiment_run`

# see ./docs/current_folder_structure.md for a detailed tree
```
## ⚙️ Current WorkFlow

```
[configs/]
└─> grid/builder.py                     (1 World Grid)
    ├─> domain/mapper.py                    (2 DK → grid, optional)
    └─> dataprep/pipeline.py                (3 Fit/Test → grid)
        └─> dataprep/schema.py                  (4 Data Scheme)
            ├─> models/factory.py                   (5.1 Model)
            ├─> dataset/builder.py                  (5.2 Dataloaders)
            ├─> training/heads                      (5.3 Data-influenced)
            ├─> training/loss                       (5.4 Head-specified)
            ├─. training/metrics                    (5.5 Head-specified)
            └─> training/optim|callback|...         (5.6 Other)
                └─> training/factory.py → Trainer       (6 Trainer 🗸)
                    └─> controller/builder.py + phases      (7 Controller 🗸)
                        └─> cli/end_to_end.py                   (8 Start ➝)
```

## 🧊 Data Foundation

The system operates on **Landsat imagery** and **DEM‑derived terrain metrics**.
The dataprep pipeline:

- generates spectral indices (NDVI, NDMI, NBR)
- produces slope, aspect, TPI from DEM
- builds label hierarchies
- normalizes features globally using Welford statistics
- bundles everything into stable `.npz` blocks

>Artifacts are validated via per‑file SHA‑256 schema hashing.
---

## 🚀 Roadmap (Updated for ADR‑0005)

### Near‑Term (current milestone)
- ADR-0005
- Improve documentation and examples
- Add unit tests for dataprep  dataset  training

### Medium‑Term
- Standard tile/AOI reporting (ADR‑0005)
- Optional user‑authored task manifest
- Lightweight artifact catalog (opt‑in) for reuse across datasets

### Long‑Term
- Additional model architectures
- Cross‑sensor extension (Sentinel‑2)
- Evaluation & export utilities
- Gradually promote stable components from the research trainer into
  `engine/training` as maturity and interface stability improve.
---

## Contributing

The project is currently in an experimental phase. Module APIs, directory
layout, and CLI behavior may change. Contributions should target research usability unless aligned with an accepted ADR defining a stable interface.

> **Note:**
>Please see active ADRs for the current project direction.
---

## License
To be determined.
