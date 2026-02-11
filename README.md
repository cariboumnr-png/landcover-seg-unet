**Caribou Landcover** is a work‑in‑progress project focused on pixel‑level land‑cover segmentation using Landsat imagery and topographic features.

## Overview
This repository provides the scaffolding for a segmentation workflow including data ingestion, model architectures, training routines, and supporting utilities.

## Repository Structure
```
caribou_landcover/
├─ configs/
├─ src/
│  ├─ dataset/
│  ├─ domain/
│  ├─ grid/
│  ├─ models/
│  ├─ training/
│  ├─ utils/
│  └─ alias.py
├─ main.py
└─ run.ps1
```

## Getting Started
Clone the repository:
```bash
git clone https://github.com/cariboumnr-png/caribou_landcover.git
cd caribou_landcover
```

## Data
The model is intended to operate on Landsat imagery and DEM‑derived topographic variables. Dataset loading and preprocessing logic lives under `src/dataset/`.

## Roadmap
- Dataset schemas and formal config files
- Baseline segmentation models and metrics
- Reproducible training workflows
- Unit tests and examples

## Contributing
Contributions welcome after initial scaffolding stabilizes.

## License
To be determined.
