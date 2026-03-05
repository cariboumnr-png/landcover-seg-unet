# ADR‑0007 — Unified Experiment‑Level I/O Structure
- **Status:** Accepted
- **Date:** 2026‑03-02
- **Updated:** 2026‑03-02

## Context
The project currently scatters I/O across several locations (root‑level logs, user‑configured cache paths, checkpoints in experiment configs). This makes experiments harder to isolate, archive, and reproduce.
Hydra naturally supports run‑scoped output directories, and widely used ML project structures (e.g., Cookiecutter Data Science) keep *data artifacts* separate from *per‑experiment outputs*.

## Decision
Adopt a **single user‑specified `exp_root` directory** that becomes the parent for **all experiment inputs, cached artifacts, and run outputs**.

Following layout is adpoted:
```
    <exp_root>/

        input/                  # user-provided data or just a manifest
            <dataset_name>/     # user-pointed dataset name
                fit/            # raw rasters for training
                test/           # raw rasters for testing (optional)
                configs/        # input rasters metadata (.json file)
            domain/             # input domain rasters (optional)
            extent_ref/         # input extent reference raster (optional)

        artifacts/              # generated during data preparation
            data_cache/         # cached data blocks
            domain_knowledge    # domain mapped to grid (optional)
            world_grids         # grid definitions

        results/                # experiments from
            <exp_0000_>/        # isolated experiments (0000-9999)
                logs/
                checkpoints/
                previews/
                plots/
                config.json     # runtime hydra configs saved per experiment
```

User provides:
- experiment root folder: <exp_root>
- dataset name: <dataset_name>
- ensure input data folder naming and structure is respected

Only the top‑level idea is fixed:
- everything lives under **one root**;
- **data-prep artifacts** are stable, shared across runs;
- **results** are per‑experiment and disposable;
- Hydra config is saved for each experiment.

## Rationale
- **Reproducibility**: each run becomes a fully self‑contained artifact.
- **Simplicity**: paths are no longer scattered; deleting or archiving an experiment is trivial.
- **Flexibility**: this structure aligns with Hydra’s output‑dir model and
common ML workflows; future MLflow/W&B integration maps cleanly.
- **Minimal code changes**: most paths are already config‑driven.

## Consequences
- Old layout needs mild migration.
- User now specifies only `exp_root` (and optionally `experiment_id`).
- Enables cleaner debugging and easier long‑term storage of experiments.

## Notes
Exact filenames, subfolders, and Hydra behavior (`hydra.run.dir`, `job.chdir`)
can be chosen later. The ADR only commits the **structural shape** and the **centralization principle**.