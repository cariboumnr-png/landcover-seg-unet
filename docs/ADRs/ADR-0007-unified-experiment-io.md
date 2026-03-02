# ADR‑0007 — Unified Experiment‑Level I/O Structure
**Status:** Proposed
**Date:** 2026‑03-02

## Context
The project currently scatters I/O across several locations (root‑level logs, user‑configured cache paths, checkpoints in experiment configs). This makes experiments harder to isolate, archive, and reproduce.
Hydra naturally supports run‑scoped output directories, and widely used ML project structures (e.g., Cookiecutter Data Science) keep *data artifacts* separate from *per‑experiment outputs*.

## Decision
Adopt a **single user‑specified `exp_root` directory** that becomes the parent for **all experiment inputs, cached artifacts, and run outputs**.

A minimal, stable high‑level layout:
```
    <exp_root>/
        input/                  # user-provided data or just a manifest
            <dataset_name>/     # user-pointed dataset name
                fit/            # raw rasters for training
                test/           # raw rasters for testing (optional)
                configs/        # input rasters metadata (.json file)
            domain/
            reference/
        artifacts/              # data-prep cache, schema, grids, domains
        results/
            <experiment_id>/    # each training run is isolated here
                logs/
                checkpoints/
                previews/
                plots/
                .hydra/         # (optional) Hydra config snapshot
```

Only the top‑level idea is fixed:
- everything lives under **one root**;
- **data-prep artifacts** are stable, shared across runs;
- **results** are per‑experiment and disposable;
- Hydra (optionally) writes its config snapshot into each run folder.

## Rationale
- **Reproducibility**: each run becomes a fully self‑contained artifact.
- **Simplicity**: paths are no longer scattered; deleting or archiving an experiment is trivial.
- **Flexibility**: this structure aligns with Hydra’s output‑dir model and common ML workflows; future MLflow/W&B integration maps cleanly.
- **Minimal code changes**: most paths are already config‑driven.

## Consequences
- Old layout needs mild migration.
- User now specifies only `exp_root` (and optionally `experiment_id`).
- Enables cleaner debugging and easier long‑term storage of experiments.

## Notes
Exact filenames, subfolders, and Hydra behavior (`hydra.run.dir`, `job.chdir`) can be chosen later. The ADR only commits the **structural shape** and the **centralization principle**.