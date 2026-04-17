# Architecture Overview

**Last updated: 2026-04-16**

----
This document describes the **current implemented architecture** of the project.
It is intentionally concise and optimized for readability. Details live in code
and ADRs; this file explains *how the system fits together*.

---
## Core Principles

- **Deterministic, reproducible artifacts**: grids, domains, blocks, and schemas are hash-validated.
- **Clear stage boundaries**: foundation → transform → dataset → session/runtime.
- **Session-first execution**: sessions own construction; the CLI only requests work.
- **Schema as contract**: `schema.json` is the authoritative dataset manifest.
- **Experiment isolation**: immutable artifacts are separated from per-run outputs.

---
## High-Level Flow

```
Raw Data
  → Foundation (grid, domains, raw blocks)
  → Transform (splits, normalization, schema)
  → Dataset (DataSpecs)
  → Session (model, loaders, losses, runtime)
  → Results (checkpoints, logs, metrics)
```

---
## Major Subsystems

### 1. Foundation (Geospatial Prep)
**Purpose:** Build stable, shareable data artifacts.

- **Grid**: deterministic world grid (CRS, pixel size, tiling).
- **Domains**: per-tile categorical/continuous context aligned to the grid.
- **Data blocks**: windowed image/label blocks with statistics.

Outputs are persisted under `artifacts/foundation/` and validated by SHA256.

---
### 2. Transform (Experiment Materialization)
**Purpose:** Convert foundation artifacts into experiment-ready datasets.

- Block filtering and scoring
- Train/val/test splits
- Train-only normalization
- Final `schema.json`

Outputs live under `artifacts/transform/`.

---
### 3. Dataset Layer
**Purpose:** Build runtime-facing specifications.

- Validates artifact integrity
- Produces `DataSpecs` (channels, heads, splits, domains)
- Acts as the boundary between data prep and training

`DataSpecs` is the stable contract consumed by models and sessions.

---
### 4. Models

- Multi-head UNet / UNet++ backbones
- Optional domain conditioning (`concat`, `FiLM`, `hybrid`, or none)
- Logit adjustment and numerical safety built-in

Models implement a shared **multihead protocol** consumed by trainers.

---
### 5. Session & Runtime
**Purpose:** Own execution and lifecycle.

A session:
- Constructs loaders, losses, optimizers, callbacks
- Initializes runtime state once
- Runs one or more training phases

Trainers/evaluators execute policy only; they do not assemble components.

---
### 6. CLI

- Hydra-based configuration
- Thin orchestration layer
- Pipelines: ingest, prepare, train, overfit

The CLI **does not wire internals**—it delegates to foundation, transform, and session APIs.

---
## Filesystem Layout (Simplified)

```
<exp_root>/
  input/            # user-provided data
  artifacts/        # immutable, shareable
    foundation/
    transform/
  results/          # per-run outputs
    exp_0001/
      checkpoints/
      logs/
      plots/
      config.json
```

---
## Guarantees

- No silent rebuilds: artifacts are hash-checked.
- No implicit resampling: CRS and pixel alignment must match.
- Deterministic pipelines unless explicitly configured otherwise.

---
## What This Document Is (and Is Not)

- ✅ A mental model of the system
- ✅ A guide to where responsibilities live
- ❌ Not an API reference
- ❌ Not a replacement for ADRs

For rationale and tradeoffs, see the ADRs.