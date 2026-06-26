# ADR-0046: Pipeline-Oriented User Configuration and Decoupled Translation Layer

**Status:** Accepted
**Date:** 2026-06-26

## 1. Context

Previously, the user configuration was managed via a single `settings.yaml` file
located at the repository root. This file mirrored the entire nested structure
of the internal configuration tree (over 200 lines). It mixed basic user concerns
(e.g., dataset filepaths, tile specifications, partition splits) with baseline
application configurations and runtime settings (e.g., model architecture
choices, optimizer parameters, loss weights, tracking options, Optuna sweep
spaces, and multi-phase curriculum configurations).

This structure created several major issues:
* **High Cognitive Load:** CLI users had to navigate a complex, nested yaml tree
  and configure parameters unrelated to basic dataset ingestion and preparation.
* **Lack of Isolation:** There was no separation between user inputs (which change
  per-dataset or per-run) and application baselines (which represent core
  system defaults).
* **Databricks Orchestration Friction:** Compute-job orchestrations require a
  clean separation between static dataset coordinates (user inputs) and dynamic,
  node-specific run parameters (application/environment controls).

## 2. Decision

We have retired the monolithic `settings.yaml` file at the root, replacing it
with a clean, simplified user configuration layout that only exposes essential
knobs.

### 2.1. Separation of Configuration Files
* We introduced a dedicated `configs/` directory at the repository root to
  separate user configurations from other files and scripts.
* We replaced the root-level `settings.yaml` with a simplified
  `configs/user.yaml` that focuses strictly on dataset inputs and project-specific
  coordinates.
* The structure in `configs/user.yaml` is organized under three sequential
  pipeline stages matching the workflow:
  - `data-ingest`: Inputs, Coordinate Reference Systems, and tiling specs.
  - `data-prepare`: Train/val/test splits and class balancing rewards.
  - `model-train`: Baseline hyperparameters, model body types, and run lengths.
* We removed the user-facing `grid_mode` setting; the system now defaults
  strictly to `'ref'` (reference raster) mode, enforcing a clean contract for
  spatial extents.

### 2.2. Decoupled Translation Layer
To keep the config schemas in `src/landseg/configs/schema/` clean and isomorphic,
all user-to-system config translation logic is fully isolated to a dedicated
adapter in `src/landseg/adapters/cli/translate.py`.
* The translation module parses the flat `configs/user.yaml` and maps keys onto
  the nested paths of the `RootConfig` dataclass tree using a lookup-driven
  mapping strategy.
* It exposes a `_set_paths(translated, paths, value)` utility that resolves
  multi-path settings (e.g., mapping `tile_size` to row/col specifications)
  and handles single-path structures (such as curriculum `epochs` phases).

### 2.3. Bootstrapping Runner
To support seamless job execution on Databricks clusters and remote VMs, we
introduced a root-level `scripts/run.py` script. The script automatically
inserts the `src/` directory into `sys.path` so the module resolves without
requiring pre-installation or manual `PYTHONPATH` exports.

### 2.4. Future Advanced Configurations
While we currently only expose essential inputs in `configs/user.yaml`, we plan
to support advanced configurations (such as pre-selecting specific subsets of
the configuration tree) under the `configs/` directory in the future. However,
this implementation is deferred until the core project architecture and pipeline
boundaries have stabilized further.

## 3. Consequences

### Positive
* **Separation of Concerns:** Core schema classes remain strictly isomorphic
  and validate clean configurations without holding user-facing translation
  properties or normalizers.
* **Ergonomic Databricks Integration:** Databricks jobs can consume
  `configs/user.yaml` as the project definition file, while CLI runtime
  overrides are supplied cleanly as task arguments.
* **Declarative and Dry Translation**: Replacing inline branch logic with map
  lookups in `translate.py` reduces branching complexity.

### Negative
* **Mapping Sync**: Adding new user-level settings to `configs/user.yaml`
  requires registering their corresponding target paths in the `translate.py`
  mapping dicts.
