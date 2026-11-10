# ADR-0066: Production Notebook Architecture and Databricks Integration

**Status:** Accepted — Implemented<br>
**Date:** 2026-11-09<br>
**Completed:** 2026-11-10

---

## 1. Context

The repository historically included five interactive Jupyter notebooks under
`notebooks/`:
- `00_raster_mosaic.ipynb`: Virtual raster mosaic assembly.
- `00_dataset_manifest.ipynb`: Dataset manifest construction and validation.
- `01_data_preparation.ipynb`: Spatial world grid alignment and data
  preparation.
- `02_model_train.ipynb`: Baseline model training session.
- `03_study_sweep.ipynb`: Optuna hyperparameter study sweeps.

While these notebooks successfully demonstrated standalone pipeline capabilities
on local workstations, they were structured primarily for demonstration
purposes. In our operating model, local production and automation workflows are
canonically driven through the CLI (`landseg.adapters.cli` with Hydra recipes)
or programmatic scripts (`landseg.adapters.api`), whereas interactive and
scheduled notebook workflows are developed and deployed specifically for
enterprise Databricks environments.

Evolving the notebook suite towards production-grade Databricks execution—both
for interactive researcher exploration on GPU clusters and for headless
orchestration via Databricks Workflows—revealed several architectural gaps:

### 1.1. Repetitive Initialization Boilerplate
Every notebook began with a duplicate 30-line initialization block that manually
probed site-packages versus local repository paths (`sys.path.insert`),
inspected environment flags (`DATABRICKS_RUNTIME_VERSION`), and hardcoded
default paths (`EXP_ROOT = './experiment/'`) alongside commented-out cloud
volume paths. Any adjustment to runtime discovery or root path resolution
required tedious manual edits across every individual notebook.

### 1.2. Brittle Configuration via Manual Cell Editing
All I/O directories, geospatial tiling parameters (CRS, tile size, stride),
dataloader settings (batch size, patch size), and model hyperparameters
(architecture, learning rate, epochs) were hardcoded into cell assignments. To
alter parameters, users had to mutate notebook code cells directly. This
prevented notebooks from being dynamically parameterized in Databricks UI
widgets or headless Databricks Workflows, while naive usage of Databricks-
specific APIs (`dbutils.widgets`) caused fatal `NameError` exceptions in local
Jupyter or CI/CD test environments.

### 1.3. Absence of Dedicated Notebook and MLOps Adapter Surfaces
The framework's user-facing layers were historically organized into
`landseg.adapters.cli` (terminal commands and Hydra recipes) and
`landseg.adapters.api` (programmatic workflow execution and pipeline
configurators). There was no dedicated namespace for notebook-exclusive concerns
or enterprise MLOps capabilities. Placing interactive widgets, IPython kernel
hooks, Databricks environment abstractions, and cloud volume resolution inside
`adapters.api` would have violated the Single Responsibility Principle by
polluting headless automation interfaces with interactive UI logic. Conversely,
bundling cloud volume resolution, runtime probing, and Unity Catalog
registration inside a notebook-only module would have prevented headless CLI
commands or automated scripts from reusing those capabilities.

### 1.4. Cloud Storage and Distributed Locking Hazards on Databricks
Operating in Databricks introduced filesystem behaviors fundamentally different
from local POSIX filesystems:
- **Unity Catalog (UC) Volumes vs DBFS**: Modern governance mandates reading and
  writing data in Unity Catalog Volumes
  (`/Volumes/<catalog>/<schema>/<volume>/...`) rather than legacy DBFS root
  mounts (`/dbfs/`).
- **Distributed POSIX Locking Incompatibility**: FUSE-mounted cloud storage (UC
  Volumes or DBFS backed by cloud object stores such as AWS S3, Azure ADLS Gen2,
  or GCP Cloud Storage) does not support POSIX byte-range file locks. Running
  SQLite-backed metadata engines—such as Optuna trial databases in
  `03_study_sweep.ipynb`—directly on `/Volumes/` or `/dbfs/` caused database
  locking failures (`sqlite3.OperationalError: database is locked`) and file
  corruption.
- To run safely on Databricks, SQLite databases and ephemeral scratch buffers
  had to reside on driver-local storage (`/tmp` or `/local_disk0`), while final
  artifacts and checkpoints needed to synchronize to persistent UC Volumes.

### 1.5. Lack of Interactive Training Telemetry and Process Safety
During long training runs in notebooks:
- Rapid console output flooded notebook cell buffers, causing browser rendering
  lag and bloated notebook file sizes.
- Uncaught cell cancellations or kernel interrupts terminated training abruptly
  without saving the in-flight checkpoint, leaving uncommitted logs and dangling
  GPU allocations.
- Model runs were disconnected from Databricks-managed MLflow tracking and the
  Unity Catalog Model Registry.

### 1.6. Lack of Production Compute Invocation Duality
Users could not transition a notebook from interactive exploratory debugging on
a single GPU driver node to an automated, headless Databricks Job cluster
without rewriting the notebook or extracting it into external scripts.

---

## 2. Decision

We formalized and implemented a production-grade notebook and MLOps architecture
that elevated the notebook suite from standalone demos into multi-environment,
production-ready workflows.

We organized this architecture around two dedicated first-class packages and six
core components:
1. **Dedicated `landseg.notebooks` and `landseg.mlops` Packages**:
   First-class namespaces cleanly separating interactive UI/presentation from
enterprise cloud storage, environment probing, and governance lineage.
2. **Unified Root Path Resolution (`resolve_exp_root`)**:
   Single-call, precedence-based workspace configuration.
3. **Ergonomic Multi-Runtime Configuration Widgets (`NotebookWidgets`)**:
   Interactive controls bridging Databricks and local/CI environments with zero
GUI dependencies.
4. **Two-Tiered Cloud Storage and Lock Protection (`resolve_optuna_storage`,
   `sync_optuna_study`)**:
   Driver-local scratch for SQLite vs UC Volumes for persistent assets.
5. **Interactive Training Ergonomics, In-Cell Visuals, and Interruption
   Safety**:
   Live loss/metric curves, qualitative segmentation previews, and graceful
   cancellation handling.
6. **Production Duality and Databricks Workflow Orchestration Readiness**:
   Seamless transition from interactive EDA to automated multi-task Databricks
   Workflows.

---

### 2.1. Dedicated `landseg.notebooks` and `landseg.mlops` Packages

We created two dedicated packages to support interactive workflows without
polluting existing headless layers:
- `landseg.adapters.cli`: Terminal / Hydra CLI entrypoints.
- `landseg.adapters.api`: Headless programmatic workflow runners (`run`,
  `run_preflight`) and pipeline configurators (`TrainingSessionConfigurator`,
  `WorldGridConfigurator`, etc.).
- `landseg.notebooks`: Interactive widget controls (`NotebookWidgets`), notebook
  environment detection (`is_notebook`), in-cell visual telemetry
  (`plot_training_curves`, `plot_qualitative_predictions`,
  `render_prediction_preview`), and process safety (`safe_training_session`).
- `landseg.mlops`: Enterprise MLOps utilities across CLI and notebooks,
  including root and Optuna storage resolution (`resolve_exp_root`,
  `resolve_optuna_storage`, `sync_optuna_study`, `load_study_trials`),
  runtime/cluster probing (`is_databricks`, `get_dbutils`, `get_cluster_id`,
  `get_runtime_version`, `get_git_commit`), and Unity Catalog Model Registry
  integration (`register_unity_catalog_model`).

In notebooks, developers and users import these distinct surfaces cleanly:
```python
import landseg.adapters.api as api
import landseg.mlops as mlops
import landseg.notebooks as nb
```

### 2.2. Unified Experiment Root and Path Resolution

We eliminated repetitive boilerplate cells by introducing
`mlops.resolve_exp_root()`. It resolves the workspace root path across
environments following a strict precedence order:
1. **Explicit argument**: `exp_root=...` passed directly by caller.
2. **Databricks Widget**: Interactive UI parameter or Databricks Job task
   parameter (`dbutils.widgets.get('exp_root')`).
3. **Environment variable**: `LANDSEG_EXP_ROOT` or `EXP_ROOT`.
4. **Default fallback**: Safe local directory (`./experiment`).

```python
# standard imports
import pathlib
import sys
# local imports
try:
    import landseg.adapters.api as api
    import landseg.mlops as mlops
    import landseg.notebooks as nb
except ImportError:
    repo_src = pathlib.Path.cwd().parent / 'src'
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    import landseg.adapters.api as api
    import landseg.mlops as mlops
    import landseg.notebooks as nb

# resolve experiment root across local, CI, or Databricks runtimes
EXP_ROOT = mlops.resolve_exp_root()
```

Rather than introducing an intermediate wrapper for folder paths, canonical
folders are composed directly via standard `pathlib.Path` off `EXP_ROOT` (e.g.
`EXP_ROOT / 'input'`, `EXP_ROOT / 'artifacts'`), and inputs are parameterized
through `widgets.path('input_dir', default=EXP_ROOT / 'input',
create_if_missing=True)`.

### 2.3. Ergonomic Multi-Runtime Configuration Widgets (`NotebookWidgets`)

Because notebooks are canonically developed for and deployed on Databricks, we
designed `NotebookWidgets` to prioritize Databricks-native ergonomics while
keeping local execution strictly headless and dependency-free:

- **Databricks as Canonical Interactive Surface**: In Databricks,
  `NotebookWidgets` natively bridged `dbutils.widgets` (`text`, `dropdown`,
  `combobox`, `multiselect`). This rendered interactive dropdown and text
  controls in the Databricks notebook toolbar for researchers and seamlessly
  captured task parameters injected by automated Databricks Workflow Jobs.
- **Headless Local Execution (No Local Visual Widgets)**: On local developer
  workstations, complex pipelines are canonically run via the CLI
  (`landseg.adapters.cli`). When notebooks are opened locally for debugging or
  run in CI/CD, we intentionally did **not** introduce `ipywidgets` or visual
  GUI widgets. Instead, `NotebookWidgets` acts as a silent, headless fallback
  that resolves the specified typed `default` values (with optional
  `LANDSEG_<KEY>` environment variable overrides). This eliminated heavy Jupyter
  frontend dependencies, avoided browser extension version conflicts in VS Code
  or JupyterLab, and guaranteed 100% deterministic headless testability.
- **Type Safety**: Provided strongly-typed getters (`int()`, `float()`,
  `boolean()`, `path()`, `multiselect()`), removing repetitive string-to-number
  casting boilerplate caused by Databricks `dbutils.widgets.get()` always
  returning raw strings.

```python
# initialize widgets bridge
widgets = nb.NotebookWidgets()

# declare and read typed configuration parameters
batch_size = widgets.int('batch_size', default=16, label='Batch Size')
patch_size = widgets.int('patch_size', default=128, label='Patch Size (px)')
max_epochs = widgets.int('max_epochs', default=5, label='Max Epochs')
model_body = widgets.dropdown(
    'model_body',
    default='unet',
    choices=['unet', 'unetpp', 'unetppp'],
    label='Model Backbone',
)
device = widgets.dropdown(
    'device',
    default='auto',
    choices=['auto', 'cuda', 'cpu'],
    label='Execution Device',
)
```

### 2.4. Two-Tiered Cloud Storage and Locking Resilience

We formalized and implemented a two-tiered storage model for Databricks
execution:
```text
+--------------------------------------------------------------------------+
|                        Databricks Execution Node                         |
|                                                                          |
|   +------------------------------------------------------------------+   |
|   | Driver-Local Scratch Tier (/tmp or /local_disk0)                 |   |
|   | - Optuna SQLite Trial Storage (sqlite:////tmp/optuna.db)         |   |
|   | - Intermittent Tile Cache & Spatial Indexes                      |   |
|   | - Zero distributed FUSE locking hazards                          |   |
|   +------------------------------------------------------------------+   |
|                                    |                                     |
|                       (Sync summaries & checkpoints)                     |
|                                    v                                     |
|   +------------------------------------------------------------------+   |
|   | Persistent Unity Catalog Volume Tier (/Volumes/catalog/schema/…) |   |
|   | - Immutable Dataset Manifests (manifest.json, manifest.yaml)     |   |
|   | - Canonical World Grid Artifacts                                 |   |
|   | - Harmonized & Ingested Tensors (HDF5 / Zarr / NetCDF)           |   |
|   | - Model Checkpoints (.pt) and Metrics Summaries                  |   |
|   +------------------------------------------------------------------+   |
+--------------------------------------------------------------------------+
```

`mlops.resolve_optuna_storage()` inspects the runtime environment:
- If running on Databricks and no external database URI (e.g. PostgreSQL) was
  passed, it automatically routed SQLite storage to driver-local disk
  (`sqlite:////tmp/optuna.db`).
- If running locally, it defaulted to `sqlite:///optuna.db`.
- Upon sweep completion in `03_study_sweep.ipynb`, calling
  `mlops.sync_optuna_study(...)` ranked the top trials, exported a trials
  summary JSON and complete CSV dataframe, and copied the Pareto-optimal
  checkpoint to the persistent volume tier
  (`EXP_ROOT/artifacts/sweeps/<study_name>/`).
- `mlops.load_study_trials()` enabled programmatic reloading of persisted trials
  tables.

### 2.5. Interactive Training Ergonomics, Telemetry, and Interruption Safety

For `02_model_train.ipynb`, we introduced:
1. **Graceful Cancellation Handling (`safe_training_session`)**: Context manager
   intercepting `KeyboardInterrupt` / cell stop signals to locate the latest
   checkpoint, copy it to `checkpoint_interrupted.pt`, tag active MLflow runs as
   `INTERRUPTED`, and cleanly release GPU VRAM allocations via
   `torch.cuda.empty_cache()`.
2. **In-Cell Metrics Curves (`plot_training_curves`)**: Renders real-time side-
   by-side training vs. validation loss curves and tracking metrics (mIoU) with
   peak epoch annotations directly in the cell output.
3. **Qualitative Inspection Previews (`plot_qualitative_predictions`,
   `render_prediction_preview`)**: Side-by-side three-panel visual comparisons
   displaying input RGB imagery, ground truth annotation masks, and model
   segmentation predictions.

### 2.6. MLflow Tracking and Unity Catalog Integration

When running inside Databricks:
- The configurators and notebook hooks automatically associated runs with the
  active Databricks MLflow Experiment.
- Run metadata captured hyperparameters, manifest hashes (`manifest_sha256`),
  cluster ID, and git commit SHA.
- A standardized registration cell in `02_model_train.ipynb` backed by
  `mlops.register_unity_catalog_model()` published validated checkpoints
  directly to the Unity Catalog Model Registry
  (`models:/<catalog>.<schema>.<model_name>`). When run outside Databricks or in
  CI, it safely bypassed remote registration while logging governance metadata
  without raising errors.

### 2.7. Explicit Supported Use Cases & Compute Invocation Duality

To eliminate environment ambiguity and eradicate fragile in-notebook package
installations, we formalized three explicitly supported execution tiers:

1. **Tier 1: Production Automated Compute (Headless)**:
   - **Primary Driver**: Terminal CLI (`landseg.adapters.cli`) executed with
     Hydra / YAML configuration recipes.
   - **Automated Workflows**: When deployed as tasks in Databricks Workflows or
     Databricks Asset Bundles (DABs), notebooks execute headlessly from top to
     bottom. Job parameters are automatically ingested by `NotebookWidgets`.
   - Environments run on automated job clusters with the pre-built `landseg`
     wheel pre-installed.

2. **Tier 2: Interactive Researcher Exploration (Enterprise Databricks)**:
   - **Target Audience**: Geospatial researchers, data scientists, and domain
     colleagues.
   - **Environment**: Interactive Databricks GPU compute where the `landseg`
     wheel is pre-installed cluster-wide via cluster init scripts or cluster-
     scoped libraries. The repository is accessed directly via Databricks Git
     Folders (Repos).
   - **User Experience**: Rich interactive controls (`NotebookWidgets`), in-cell
     telemetry, and Unity Catalog Model Registry publication.
   - **Zero-Boilerplate Startup**: Notebooks assume the environment is already
     prepared. Executable `%pip install` cells were strictly removed to prevent
     REPL restarts and cluster cache invalidations.

3. **Tier 3: Local Developer Sandbox (VS Code / Workstation)**:
   - **Target Audience**: Core library developers and CI test runners.
   - **Environment**: Local repository clone inside the project's canonical root
     virtual environment (`.venv`) with `pip install -e ".[dev]"`.
   - **Headless Fallback**: `NotebookWidgets` operates silently using typed
     defaults and environment overrides without requiring visual `ipywidgets`
     frontend extensions.

---

## 3. Consequences

### 3.1. Positive Consequences
- **Architectural Boundary Integrity**: `landseg.adapters.api` remains strictly
  headless, `landseg.notebooks` encapsulates interactive widget and telemetry
  concerns, and `landseg.mlops` encapsulates cloud storage and governance
  infrastructure.
- **Zero-Boilerplate Startup**: The repetitive 30-line initialization block was
  replaced by a concise, standardized call to `mlops.resolve_exp_root()`.
- **Dynamic Parameterization**: `NotebookWidgets` enabled interactive tuning in
  Databricks and local Jupyter without breaking headless execution.
- **Distributed Lock Safety**: Prevented SQLite file corruption on cloud storage
  volumes by routing trial databases to driver-local storage.
- **Production Readiness**: Notebooks became fully compatible with scheduled
  Databricks Workflows and CI/CD validation.

### 3.2. Trade-offs and Mitigations
- **Dedicated Package Surfaces**:
  - *Trade-off*: Introducing `landseg.notebooks` and `landseg.mlops` added
    package surfaces to maintain.
  - *Mitigation*: Both packages are lightweight, modular, and covered by
    comprehensive automated unit test suites (`tests/unit/notebooks/`,
    `tests/unit/mlops/`).
- **Driver vs. Volume Data Synchronization**:
  - *Trade-off*: Writing Optuna trial databases to `/tmp/` required syncing
    final study summaries back to persistent volume storage.
  - *Mitigation*: Handled automatically via `mlops.sync_optuna_study()`.

### 3.3. Architectural Invariants
1. **Strict Headless Purity for `adapters.api`**: `landseg.adapters.api` must
   never re-export notebook widgets, display helpers, or interactive runtime
   hooks. All interactive notebook logic must reside in `landseg.notebooks`.
2. **No Hardcoded Cloud URIs**: Notebooks will never hardcode absolute
   `/Volumes/...` or `/dbfs/...` paths; all root paths must resolve via
   `mlops.resolve_exp_root()`.
3. **Guarded `dbutils` Access**: Direct invocation of `dbutils` must always be
   guarded through `NotebookWidgets` or feature detection to prevent local
   crashes.
4. **No SQLite on Distributed FUSE Mounts**: SQLite database URIs must never
   point directly to `/Volumes/` or `/dbfs/` in Databricks runtime; driver-local
   paths (`/tmp/` or `/local_disk0/`) or managed cloud databases must be used.
5. **Headless Execution Invariant**: Every notebook must remain capable of
   executing from top to bottom non-interactively with default parameters.
6. **Databricks as Canonical Notebook Target**: Notebooks are developed and
   maintained canonically for Databricks. Local execution is intentionally
   headless without visual GUI dependencies (e.g. no `ipywidgets`), serving as a
   zero-friction developer sandbox and CI verification layer.
7. **Zero In-Notebook `%pip install` Boilerplate**: Notebooks assume the
   execution environment (cluster init script on Databricks or local virtualenv)
   already has `landseg` installed. In-notebook package installations restart
   REPL kernels and disrupt cluster library management.

---

## 4. Implementation & Migration History

### Phase 1: Core Dedicated Packages and Test Suites (Completed)
- Introduced dedicated package `landseg.notebooks` (`src/landseg/notebooks/`)
  containing `NotebookWidgets`, `is_notebook`, `safe_training_session`,
  `plot_training_curves`, `plot_qualitative_predictions`, and
  `render_prediction_preview`.
- Decoupled enterprise cloud storage, environment probing, and governance
  lineage into first-class package `landseg.mlops` (`src/landseg/mlops/`)
  containing `resolve_exp_root`, `resolve_optuna_storage`, `sync_optuna_study`,
  `load_study_trials`, `is_databricks`, `get_dbutils`, `get_cluster_id`,
  `get_runtime_version`, `get_git_commit`, and `register_unity_catalog_model`.
- Ensured `landseg.adapters.api` remained strictly focused on headless
  programmatic execution with polymorphic `str | Path` configurator arguments
  and no notebook re-exports.
- Implemented comprehensive unit test suites in
  `tests/unit/notebooks/test_notebooks.py`,
  `tests/unit/notebooks/test_telemetry.py`, `tests/unit/mlops/test_env.py`,
  `tests/unit/mlops/test_storage.py`, and
  `tests/unit/mlops/test_mlops_registry.py`.

### Phase 2: Notebook Suite Migration (Completed)
- Updated `notebooks/00_dataset_manifest.ipynb`: Standardized initialization
  cell with `mlops.resolve_exp_root()` and configured raw input directory via
  `widgets.path('input_dir', ...)`.
- Updated `notebooks/00_raster_mosaic.ipynb`: Standardized initialization cell
  with `mlops.resolve_exp_root()`.
- Updated `notebooks/01_data_preparation.ipynb`: Standardized initialization
  cell with `mlops.resolve_exp_root()`, and parameterized World Grid CRS, tile
  size, and stride via `NotebookWidgets`.
- Updated `notebooks/02_model_train.ipynb`: Standardized initialization cell
  with `mlops.resolve_exp_root()`, parameterized training batch size, patch
  size, epochs, model body, and execution device via widgets, wrapped training
  execution in `safe_training_session`, added in-cell metrics and segmentation
  previews, and integrated Unity Catalog model registration.
- Updated `notebooks/03_study_sweep.ipynb`: Standardized initialization cell
  with `mlops.resolve_exp_root()`, resolved safe Optuna storage with
  `mlops.resolve_optuna_storage()`, and synchronized completed sweep trials and
  Pareto-optimal checkpoints via `mlops.sync_optuna_study()`.
- Removed all obsolete `%pip install` boilerplate and `ThrottledProgress`
  references across the entire notebook suite.

### Phase 3: Workflow Orchestration Reference & UC Model Registration (Completed)
- Documented reference multi-task Databricks Workflow orchestration DAG in
  `notebooks/README.md`, demonstrating how to chain the notebook suite on GPU
  cluster compute with dynamic parameter forwarding, while keeping deployment
  infrastructure outside the library repository.
- Introduced MLflow Unity Catalog model registration cell in
  `02_model_train.ipynb` backed by
  `landseg.mlops.registry.register_unity_catalog_model`.
- Authored comprehensive operational and compute documentation in
  `notebooks/README.md`.

