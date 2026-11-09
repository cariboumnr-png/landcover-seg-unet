# ADR-0066: Production Notebook Architecture and Databricks Integration

**Status:** Proposed<br>
**Date:** 2026-11-09

---

## 1. Context

The repository currently includes five interactive Jupyter notebooks under `notebooks/`:
- `00_raster_mosaic.ipynb`: Virtual raster mosaic assembly.
- `00_dataset_manifest.ipynb`: Dataset manifest construction and validation.
- `01_data_preparation.ipynb`: Spatial world grid alignment and data preparation.
- `02_model_train.ipynb`: Baseline model training session.
- `03_study_sweep.ipynb`: Optuna hyperparameter study sweeps.

While these notebooks successfully demonstrate standalone pipeline capabilities
on local workstations, they are structured primarily for demonstration purposes.
Evolving them towards production-grade execution—particularly within enterprise
Databricks environments and automated workflow pipelines—reveals several architectural
gaps:

### 1.1. Repetitive Initialization Boilerplate
Every notebook begins with a duplicate 30-line initialization block that manually
probes site-packages versus local repository paths (`sys.path.insert`), inspects
environment flags (`DATABRICKS_RUNTIME_VERSION`), and hardcodes default paths
(`EXP_ROOT = './experiment/'`) alongside commented-out cloud volume paths. Any
adjustment to runtime discovery or root path resolution requires tedious manual
edits across every individual notebook.

### 1.2. Brittle Configuration via Manual Cell Editing
All I/O directories, geospatial tiling parameters (CRS, tile size, stride),
dataloader settings (batch size, patch size), and model hyperparameters (architecture,
learning rate, epochs) are hardcoded into cell assignments. To alter parameters,
users must mutate notebook code cells directly. This prevents notebooks from being
dynamically parameterized in Databricks UI widgets or headless Databricks Workflows,
while naive usage of Databricks-specific APIs (`dbutils.widgets`) causes fatal
`NameError` exceptions in local Jupyter or CI/CD test environments.

### 1.3. Absence of a Dedicated Notebook Adapter Surface
The framework's user-facing layers are currently organized into `landseg.adapters.cli`
(terminal commands and Hydra recipes) and `landseg.adapters.api` (programmatic
workflow execution and pipeline configurators). There is no dedicated namespace
for notebook-exclusive concerns. Placing interactive widgets, IPython kernel hooks,
and Databricks environment abstractions inside `adapters.api` would violate the
Single Responsibility Principle by polluting headless automation interfaces with
interactive UI logic.

### 1.4. Cloud Storage and Distributed Locking Hazards on Databricks
Operating in Databricks introduces filesystem behaviors fundamentally different
from local POSIX filesystems:
- **Unity Catalog (UC) Volumes vs DBFS**: Modern governance mandates reading and
writing data in Unity Catalog Volumes (`/Volumes/<catalog>/<schema>/<volume>/...`)
rather than legacy DBFS root mounts (`/dbfs/`).
- **Distributed POSIX Locking Incompatibility**: FUSE-mounted cloud storage (UC
Volumes or DBFS backed by cloud object stores such as AWS S3, Azure ADLS Gen2, or
GCP Cloud Storage) does not support POSIX byte-range file locks. Running
SQLite-backed metadata engines—such as Optuna trial databases in
`03_study_sweep.ipynb`—directly on `/Volumes/` or `/dbfs/` causes database locking
failures (`sqlite3.OperationalError: database is locked`) and file corruption.
- To run safely on Databricks, SQLite databases and ephemeral scratch buffers must
reside on driver-local storage (`/tmp` or `/local_disk0`), while final artifacts
and checkpoints sync to persistent UC Volumes.

### 1.5. Lack of Interactive Training Telemetry and Process Safety
During long training runs in notebooks:
- Rapid console output floods notebook cell buffers, causing browser rendering
lag and bloated notebook file sizes.
- Uncaught cell cancellations or kernel interrupts terminate training abruptly
without saving the in-flight checkpoint, leaving uncommitted logs and dangling GPU
allocations.
- Model runs are disconnected from Databricks-managed MLflow tracking and the Unity
Catalog Model Registry.

### 1.6. Lack of Production Compute Invocation Duality
Users cannot currently transition a notebook from interactive exploratory debugging
on a single GPU driver node to an automated, headless Databricks Job cluster
without rewriting the notebook or extracting it into external scripts.

---

## 2. Decision

We will formalize and introduce a production-grade notebook architecture that
elevates the notebook suite from standalone demos into multi-environment,
production-ready workflows.

We will organize this architecture around six core components:
1. **Dedicated `landseg.notebooks` Package**:
A first-class namespace hosting all notebook-exclusive logic.
2. **Unified Root and I/O Resolution (`resolve_exp_root`, `resolve_notebook_io`)**:
Single-call, precedence-based workspace configuration.
3. **Ergonomic Multi-Runtime Configuration Widgets (`NotebookWidgets`)**:
Interactive controls bridging Databricks and local/CI environments.

4. **Two-Tiered Cloud Storage and Lock Protection (`resolve_optuna_storage`)**:
Driver-local scratch for SQLite vs UC Volumes for persistent assets.
5. **Interactive Training Telemetry, In-Cell Visuals, and Interruption Safety**:
Live metrics, qualitative mask inspection, and safe shutdown.
6. **Production Duality and Databricks Compute Dispatch**:
Seamless transition from interactive EDA to Databricks Workflows and Asset Bundles (DABs).

---

### 2.1. Dedicated `landseg.notebooks` Package

We will create a dedicated package, `landseg.notebooks` (`src/landseg/notebooks/`),
designed specifically to support interactive notebook workflows without polluting
existing headless layers:
- `landseg.adapters.cli`: Terminal / Hydra CLI entrypoints.
- `landseg.adapters.api`: Headless programmatic workflow runners (`run`, `run_preflight`)
and pipeline configurators (`TrainingSessionConfigurator`, `WorldGridConfigurator`, etc.).
- `landseg.notebooks`: Interactive widget controls, environment detection, notebook
I/O resolution, storage routing, and in-cell telemetry.

In notebooks, developers and users will import these distinct surfaces cleanly:
```python
import landseg.adapters.api as api
import landseg.notebooks as nb
```

### 2.2. Unified Experiment Root and I/O Resolution

We will eliminate repetitive boilerplate cells by providing `nb.resolve_exp_root()`.
It will resolve the workspace root path across environments following a strict
precedence order:
1. **Explicit argument**: `exp_root=...` passed directly by caller.
2. **Databricks Widget**: Interactive UI parameter or Databricks Job task parameter (`dbutils.widgets.get('exp_root')`).
3. **Environment variable**: `LANDSEG_EXP_ROOT` or `EXP_ROOT`.
4. **Default fallback**: Safe local directory (`./experiment`).

```python
# standard imports
import pathlib
import sys
# local imports
try:
    import landseg.adapters.api as api
    import landseg.notebooks as nb
except ImportError:
    repo_src = pathlib.Path.cwd().parent / 'src'
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    import landseg.adapters.api as api
    import landseg.notebooks as nb

# resolve experiment root across local, CI, or Databricks runtimes
EXP_ROOT = nb.resolve_exp_root()
```

In addition, `nb.resolve_notebook_io()` will structure canonical folder relationships for artifacts, results, and raw inputs:
```python
io_layout = nb.resolve_notebook_io(EXP_ROOT, create_dirs=True)
# io_layout.artifacts_dir -> EXP_ROOT / 'artifacts'
# io_layout.results_dir   -> EXP_ROOT / 'results'
# io_layout.input_dir     -> EXP_ROOT / 'input'
```

### 2.3. Ergonomic Multi-Runtime Configuration Widgets (`NotebookWidgets`)

To replace manual code cell edits with interactive UI controls, we will introduce
`NotebookWidgets` in `landseg.notebooks`:
- **Databricks Runtime**: Automatically interfaces with native `dbutils.widgets`
(`text`, `dropdown`, `combobox`, `multiselect`), rendering UI dropdowns and input
fields in the notebook toolbar and capturing parameters passed by Databricks Workflows.
- **Local Jupyter / VS Code / CI**: Gracefully falls back to environment variables
(`LANDSEG_<KEY>`) or specified default values. It will never fail with `NameError`
when `dbutils` is unavailable.
- **Type Safety**: Provides strongly-typed getters (`int()`, `float()`, `boolean()`,
`path()`, `multiselect()`), removing repetitive string-to-number casting logic.

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

We will formalize a two-tiered storage model for Databricks execution:
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

`nb.resolve_optuna_storage()` will inspect the runtime environment:
- If running on Databricks and no external database URI (e.g. PostgreSQL) is passed,
it will automatically route SQLite storage to driver-local disk (`sqlite:////tmp/optuna.db`).
- If running locally, it will default to `sqlite:///optuna.db`.
- Upon sweep completion in `03_study_sweep.ipynb`, study summary dataframes and
Pareto-optimal checkpoints will be saved to the persistent `EXP_ROOT` volume.

### 2.5. Interactive Training Ergonomics, Telemetry, and Interruption Safety

For `02_model_train.ipynb`, we will introduce:
1. **Progress Output Throttling**: Throttled console updates (1/sec) to avoid
notebook buffer overflow.
2. **In-Cell Metrics & Qualitative Visualizations**: Optional callbacks rendering
real-time loss and mIoU curves after each epoch, and qualitative side-by-side plots
of RGB imagery, ground truth, and predicted segmentation masks upon evaluation.
3. **Graceful Cancellation Handling**: Signal interception for `KeyboardInterrupt` / cell
cancellation to flush an emergency checkpoint (`checkpoint_interrupted.pt`), log
the abort state to MLflow, and cleanly release GPU memory.

### 2.6. MLflow Tracking and Unity Catalog Integration

When running inside Databricks:
- The configurators and notebook hooks will automatically associate runs with the
active Databricks MLflow Experiment.
- Run metadata will capture hyperparameters, manifest hashes (`_hash.json`), cluster
ID, and git commit SHA.
- A standardized registration cell will publish validated model checkpoints directly
to the Unity Catalog Model Registry (`models:/<catalog>.<schema>.<model_name>`).

### 2.7. Compute Invocation and Headless Production Duality

We will ensure every notebook operates under a dual-mode pattern:
- **Interactive Exploration Mode**: Researchers run cells step-by-step on interactive
clusters, tweaking parameters via widgets and inspecting visual outputs.
- **Automated Workflow Mode**: The exact same notebook can be triggered as a task
within Databricks Workflows or Databricks Asset Bundles (DABs) without modifying code, receiving parameters via job base parameters.
- **Headless CI Testing**: Automated CI pipelines can run notebooks using `papermill`
or `pytest` with environment variable parameter injection.

---

## 3. Consequences

### 3.1. Positive Consequences
- **Architectural Boundary Integrity**: `landseg.adapters.api` remains strictly
headless, while `landseg.notebooks` encapsulates all interactive and widget concerns.
- **Zero-Boilerplate Startup**: The repetitive 30-line initialization block is
replaced by a concise, standardized call to `nb.resolve_exp_root()`.
- **Dynamic Parameterization**: `NotebookWidgets` enables interactive tuning in
Databricks and local Jupyter without breaking headless execution.
- **Distributed Lock Safety**: Prevents SQLite file corruption on cloud storage
volumes by routing trial databases to driver-local storage.
- **Production Readiness**: Notebooks become fully compatible with scheduled
Databricks Workflows and CI/CD validation.

### 3.2. Trade-offs and Mitigations
- **Dedicated Package Surface**:
  - *Trade-off*: Introducing `landseg.notebooks` adds a package to maintain.
  - *Mitigation*: The package is lightweight, zero-dependency, and covered by
  automated unit tests.
- **Driver vs. Volume Data Synchronization**:
  - *Trade-off*: Writing Optuna trial databases to `/tmp/` requires syncing final
  study summaries back to persistent volume storage.
  - *Mitigation*: Handled automatically in the sweep execution wrapper.

### 3.3. Architectural Invariants
1. **Strict Headless Purity for `adapters.api`**: `landseg.adapters.api` must never
re-export notebook widgets, display helpers, or interactive runtime hooks. All
interactive notebook logic must reside in `landseg.notebooks`.
2. **No Hardcoded Cloud URIs**: Notebooks will never hardcode absolute `/Volumes/...`
or `/dbfs/...` paths; all root paths must resolve via `nb.resolve_exp_root()`.
3. **Guarded `dbutils` Access**: Direct invocation of `dbutils` must always be
guarded through `NotebookWidgets` or feature detection to prevent local crashes.
4. **No SQLite on Distributed FUSE Mounts**: SQLite database URIs must never point
directly to `/Volumes/` or `/dbfs/` in Databricks runtime; driver-local paths
(`/tmp/` or `/local_disk0/`) or managed cloud databases must be used.
5. **Headless Execution Invariant**: Every notebook must remain capable of executing
from top to bottom non-interactively with default parameters.

---

## 4. Implementation & Migration Plan

### Phase 1: Core Dedicated Package and Test Suite
- Introduce dedicated package `landseg.notebooks` (`src/landseg/notebooks/`)
containing `resolve_exp_root`, `resolve_notebook_io`, `resolve_optuna_storage`,
`NotebookWidgets`, `is_databricks`, and `is_notebook`.
- Ensure `landseg.adapters.api` remains strictly focused on headless programmatic
execution with no notebook re-exports.
- Implement comprehensive unit test suite in `tests/unit/notebooks/test_notebooks.py`.

### Phase 2: Notebook Suite Migration
- Update `notebooks/00_dataset_manifest.ipynb`: Import `landseg.notebooks as nb`,
replace initialization cell with `nb.resolve_exp_root()`, and configure raw input
directory via widgets.
- Update `notebooks/00_raster_mosaic.ipynb`: Import `landseg.notebooks as nb`,
replace initialization cell with `nb.resolve_exp_root()`.
- Update `notebooks/01_data_preparation.ipynb`: Import `landseg.notebooks as nb`,
replace initialization cell with `nb.resolve_exp_root()`, and parameterize World
Grid CRS, tile size, and stride via widgets.
- Update `notebooks/02_model_train.ipynb`: Import `landseg.notebooks as nb`, replace
initialization cell with `nb.resolve_exp_root()`, and parameterize training batch
size, patch size, epochs, model body, and execution device via widgets.
- Update `notebooks/03_study_sweep.ipynb`: Import `landseg.notebooks as nb`, replace
initialization cell and Optuna path configuration with `nb.resolve_exp_root()` and
`nb.resolve_optuna_storage()`.

### Phase 3: Databricks Workflow & DABs Assets
- Author reference Databricks Asset Bundle (`databricks.yml`) in `deploy/databricks/`
demonstrating automated orchestration of the notebook suite on GPU cluster compute.
- Introduce optional MLflow Unity Catalog model registration cell in `02_model_train.ipynb`.
