### Multi-Modal Landcover Classification Framework
A PyTorch-based deep learning pipeline for pixel-level segmentation, fusing Landsat spectral imagery with topographical data using U-Net architectures and enhanced by domain knowledge injection.

## Overview
This repository provides the scaffolding for a segmentation workflow including data ingestion, model architectures, training routines, and supporting utilities.

## WIP
Currently working on: ADR-0002 (see docs/ADRs)

## Current Repository Structure
```
project/
├─ configs/
├─ src/
│  ├─ dataset/
│  ├─ domain/
│  ├─ grid/
│  ├─ models/
│  ├─ tiler/   <-- currently active branch here
│  ├─ training/
│  ├─ utils/
│  └─ alias.py
├─ main.py
└─ run.ps1
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
