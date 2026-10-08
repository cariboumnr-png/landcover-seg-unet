# ADR-0060: Higher-Level Workflow Orchestration and Pre-Flight Validation Engine

**Status:** Accepted — Implemented<br>
**Date:** 2026-09-29

---

## 1. Context

During the implementation of incremental batch ingestion (ADR-0059 Phases
1 & 2), the geospatial ETL pipeline (`landseg.geopipe`) established
first-class run ledgers (`harmonization_runs.json`, `ingestion_runs.json`),
run-level SHA-256 fingerprinting, explicit block collision policies, and
canonical block pooling.

However, operationalizing these capabilities exposed a fundamental
architectural tension between **atomic pipeline execution** and
**higher-level multi-run workflow orchestration**.

### 1.1. The Atomic Pipeline Invariant

The core execution layer (`landseg.execution.pipelines`) was originally
conceived around a strict 1-to-1 execution invariant:
$$\text{1 Invocation} \longrightarrow \text{1 Run Target} \longrightarrow \text{1 Run Directory} \longrightarrow \text{1 Report} \longrightarrow \text{1 Ledger Record}$$

This invariant holds cleanly for atomic pipeline stages:
- `world-grid`: Generates a single world grid definition and report.
- `data-harmonize`: Warps a single batch of rasters into `run_XXXX`.
- `data-ingest`: Ingests an explicit single batch into the canonical pool.
- `data-prepare`: Partitions the canonical pool into one experiment dataset.
- `model-train`: Runs a single model training session in `run_XXXX`.
- `model-evaluate`: Evaluates a checkpoint across validation/test splits.

### 1.2. The Emergence of Multi-Run Workflows

In contrast, several mission-critical geospatial MLOps capabilities
represent **composite workflows** that sit above atomic pipeline execution:

1. **Multi-Batch Ingestion Catch-Up (`batch_ingest`)**:
   When new spatial imagery arrives across multiple flight lines or
   seasonal passes, `data-ingest` must query `harmonization_runs.json`,
   resolve pending batches, and iterate over an unbounded queue of runs.
   Placing this queue loop inside `pipelines.data_ingest.exec_ingest_data`
   forces an atomic pipeline runner to manage multi-run lifecycles,
   re-entering the context builder and creating multiple run folders
   within a single pipeline invocation.

2. **Hyperparameter Optimization Sweeps (`study_sweep`)**:
   Currently residing in `landseg.execution.pipelines.study_sweep`, hyperparameter
   sweeps iterate over multiple Optuna study trials, spawning dozens of
   distinct training runs. Treating `study-sweep` as an atomic pipeline
   distorts pipeline semantics, report schemas, and execution guarantees.

3. **Continuous End-to-End Data Intake (`e2e_intake`)**:
   When raw GeoTIFF packages arrive, production pipelines must run
   `data-harmonize` followed immediately by `data-ingest` for each batch,
   accumulating data into the canonical block pool without manual
   intervention between stages.

4. **Full Experiment Workflows (`e2e_experiment`)**:
   Running the complete lifecycle (`world-grid` $\rightarrow$ `data-harmonize`
   $\rightarrow$ `data-ingest` $\rightarrow$ `data-prepare` $\rightarrow$
   `model-train`) requires external shell scripts or manual step-by-step
   invocations, lacking unified failure handling or resumption.

### 1.3. Absence of Pre-Flight Telemetry and Validation

Geospatial deep learning jobs are computationally expensive: warping
large rasters and slicing tens of thousands of tiles takes significant
compute, while deep learning sessions consume extensive GPU hours.

Currently, pipelines validate dependencies lazily at execution time. If an
upstream artifact is incompatible (e.g., mismatched CRS, invalid
`block_identity`, corrupted virtual raster path, or unpopulated feature
layer), the failure occurs deep inside a run, wasting compute and leaving
partially materialized directories.

The framework lacks a unified **pre-flight validation engine** capable of
probing pipeline readiness in dry-run mode, validating cross-pipeline
contracts, and presenting an executive status-quo dashboard before any
heavy compute is dispatched.

---

## 2. Decision

We have formalized a dedicated **higher-level workflow orchestration layer**
(`landseg.execution.workflows`) decoupled from atomic pipelines, and
introduced a **unified pre-flight validation engine** (`landseg.execution.preflight`).

We have:
1. **Decoupled Atomic Pipelines from Multi-Run Workflows**:
   - `landseg.execution.pipelines.*` remains strictly scoped to atomic,
     single-target execution units (enforcing the 1-to-1 invariant).
   - `landseg.execution.workflows.*` hosts multi-run iterators, composite
     multi-pipeline DAGs, and hyperparameter search loops.
2. **Refactored Ingestion into Atomic Pipeline and Batch Workflow**:
   - `pipelines.data_ingest` executes an explicit, single harmonization
     batch target (`harmonize_run_id`).
   - `workflows.batch_ingest` resolves pending batches from ledgers,
     handles queue ordering, and invokes the atomic pipeline sequentially.
3. **Relocated `study-sweep` to Workflows**:
   - Relocated `pipelines.study_sweep` to `workflows.study_sweep`, decoupling
     Optuna multi-trial search from atomic training sessions.
4. **Implemented Pre-Flight Validation Engine**:
   - Provides a dry-run inspection framework that probes upstream ledgers,
     spatial grid invariants (`affine_identity`, `block_identity`),
     raster metadata integrity, and system compute resources prior to
     execution.
5. **Introduced Runner Classes for Atomic Pipelines (Deferring Workflow Runners)**:
   - Encapsulated each atomic pipeline within a dedicated runner class
     (`WorldGridGeneration`, `DataHarmonization`, `DataIngestion`,
     `DataPreparation`, `ModelTraining`, `ModelEvaluation`) with a consistent
     `.run()` invocation lifecycle.
   - Retained workflows as structurally simple, lightweight functions
     (`execute_<workflow>(root_config)`) in `landseg.execution.workflows`,
     deferring class-based workflow runners and registry abstractions until
     composite DAG patterns demand stateful orchestration.

---

## 3. Architecture & Design Specification

### 3.1. Layered Execution Hierarchy

```text
+------------------------------------------------------------------------------+
|                             USER CONTROL SURFACE                             |
|          CLI (configs/user.yaml)     |    Programmatic API (Notebooks)       |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                        landseg.execution.executor.py                         |
|      Unified router: Resolves configs and dispatches execution targets       |
+-------------------+----------------------+-----------------------------------+
                    |                      |                                   |
                    v                      v                                   v
+-----------------------+  +-------------------------------+  +----------------+
|  execution.workflows  |  |  execution.pipelines (Atomic) |  |   execution.   |
| (Multi-Run/Composite) |  |-------------------------------|  |   preflight    |
|-----------------------|  | - WorldGridGeneration         |  | (Readiness/    |
| - batch_ingest        |  | - DataHarmonization           |  |  Inspection)   |
| - diagnose_overfit    |  | - DataIngestion               |  |----------------|
| - study_sweep         |  | - DataPreparation             |  | - inspect_target
| - study_analysis      |  | - ModelTraining               |  | - run_preflight|
| - e2e_intake          |  | - ModelEvaluation             |  | - probes/      |
| - e2e_experiment      |  |                               |  | - prerequisites
+-----------+-----------+  +---------------+---------------+  +----------------+
            |                              ^
            +-------(delegates to)---------+
                                           |
                                           v
                           +-------------------------------+
                           |     geopipe / session core    |
                           +-------------------------------+
```

### 3.2. Execution Abstractions: Pipeline Runner Classes vs. Functional Workflows

To balance operational encapsulation with architectural simplicity, execution
units adopt a bifurcated abstraction model:

#### 1. Atomic Pipeline Runner Classes
Atomic single-target pipelines encapsulate run target resolution, execution
contexts, logging lifecycle, and report emission into dedicated runner classes
instantiated with `RootConfig` and executed via `.run()`:
- `pipelines.WorldGridGeneration(root_config).run()`
- `pipelines.DataHarmonization(root_config).run()`
- `pipelines.DataIngestion(root_config).run()`
- `pipelines.DataPreparation(root_config).run()`
- `pipelines.ModelTraining(root_config).run()`
- `pipelines.ModelEvaluation(root_config).run()`

This class structure provides uniform setup, teardown, and deterministic
run-directory creation for every atomic execution target.

#### 2. Functional Workflows (Runner Classes Deferred)
In contrast, workflows in `landseg.execution.workflows` currently remain
structurally simple. They coordinate existing pipelines, invoke loops, or
delegate to optimization engines without requiring internal run directories
or complex local state machines.

Workflows are exposed as straightforward, lightweight functions accepting
`RootConfig`:
- `workflows.execute_batch_ingest(root_config)`
- `workflows.execute_study_sweep(root_config)`
- `workflows.execute_study_analysis(root_config)`
- `workflows.execute_diagnose_overfit(root_config)`
- `workflows.execute_default_action(root_config)`

Introducing formal class-based workflow runners (e.g. `WorkflowRunner` or
`WorkflowProtocol`) and dynamic registry machinery is explicitly deferred.
This avoids premature abstraction while workflow orchestration logic remains
concise and procedural.

### 3.3. Atomic `data_ingest` vs. `batch_ingest` Workflow

Under this separation of concerns:
- **`pipelines.DataIngestion(config).run()`**:
  Accepts an explicit target batch (`target_harmonize_run_id`). It slices
  blocks, resolves collisions against the canonical pool using
  `CollisionPolicy`, writes `collisions.json` and `ingest_report.json` to
  `ingested_data/run_XXXX/`, and registers the run in `ingestion_runs.json`.
- **`workflows.execute_batch_ingest(config)`**:
  Discovers un-ingested batches via `resolve_pending_ingestion_batches()`,
  verifies pool grid compatibility, and invokes `DataIngestion` sequentially
  for each pending batch. It collects individual run reports and emits a
  consolidated `batch_ingest_summary.json`.

### 3.4. Pre-Flight Validation Engine (`landseg.execution.preflight`)

The pre-flight engine implements a non-destructive dry-run inspection
framework capable of evaluating execution prerequisites across both atomic
pipelines and composite workflows prior to committing compute:

```text
[landseg.execution.preflight]
       |
       +--> [Lineage / Prerequisites]
       |        (upstream pipeline reports, manifest completion, required artifacts)
       |
       +--> [Filesystem & Storage]
       |        (target output directory writability, existing report overwrites)
       |
       +--> [Domain Contracts: Spatial / Dataset / Policy / Model]
       |        (spatial grid & CRS reference, raw dataset manifests, collision policy, model architectures)
       |
       +--> [Ledger & Lineage Integrity]
       |        (past runs history, pending harmonization/ingestion batches, canonical block pool state)
       |
       +--> [Hardware & Compute Environment]
       |        (accelerator availability, CUDA/CPU detection, torch version, VRAM headroom)
       |
       `--> [Reporter: Terminal Dashboard & JSON Artifact]
                (emits standardized 120-col status table and preflight_report_<uid>.json)
```

#### Diagnostic Probe Categories & Canonical Ordering:
Probes execute in a strictly defined canonical order:
1. **Lineage**:
   - Evaluates upstream execution dependencies and pipeline report status
     via `_check_prerequisites` and `check_target_prerequisites`.
   - Validates completed upstream reports, manifest records, and required
     disk artifacts.
2. **Filesystem**:
   - Verifies target destination directory writability via `dir_writable`.
   - Detects whether target artifact files already exist and assesses
     rebuild flags via `target_file_exists`.
3. **Domain Contracts**:
   - **Spatial**: Reports world grid spatial reference source
     (`spatial_reference`), target CRS (`crs_info`), pixel resolution
     (`pixel_size`), grid extent (`grid_extent`), origin (`grid_origin`),
     and tile specs (`grid_specs`).
   - **Dataset**: Inspects input raster dataset manifest readability and
     entry count via `raw_dataset`.
   - **Policy**: Verifies block collision handling policy (`skip` vs
     `overwrite`) via `collision_policy`.
   - **Model**: Verifies neural backbone architecture recognition via
     `model_body`, evaluation checkpoint readiness via `checkpoint_ready`,
     and evaluation dataset split via `eval_split`.
4. **Ledger**:
   - Audits cumulative ETL run manifests (`harmonization_runs.json`,
     `ingestion_runs.json`) via `past_runs`.
   - Evaluates whether input datasets are already harmonized via
     `pending_harmonization`.
   - Resolves batches pending ingestion via `pending_ingestion`.
   - Inspects canonical block pool catalog state and block counts via
     `ingestion_pool_state` and prepared split blocks via
     `prepared_blocks_state`.
5. **Hardware**:
   - Inspects compute accelerator state (`cuda_device`) and estimates VRAM
     headroom against batch tensor memory requirements (`vram_headroom`).

#### Command & Programmatic Interface:

We have introduced `command=preflight` as a first-class execution target in
Hydra and `executor.py`, configured via `_PreflightConfig` in
`landseg.configs.schema.sections.commands`:

```python
@dataclasses.dataclass
class _PreflightConfig:
    target: str = 'all'               # 'all', 'model-train', 'batch-ingest', etc.
    strict: bool = False              # if true, warnings cause failure exit code
    export_report: bool = True        # whether to write preflight_report.json
    report_path: str | None = None    # custom destination path for report JSON
    check_gpu: bool = True            # whether to probe CUDA and VRAM headroom
```

Users can invoke pre-flight checks via the CLI:
```bash
# Full environment and data readiness audit
python scripts/run.py command=preflight

# Target-specific pre-flight readiness checks
python scripts/run.py command=preflight target=model-train
python scripts/run.py command=preflight target=batch-ingest strict=true
```

And programmatically via `landseg.adapters.api`:
```python
import landseg

report = landseg.run_preflight(
    target='model-train',
    strict=False,
    config=root_config,
)
if not report.is_ready:
    raise RuntimeError(f'Preflight failed: {report.errors}')
```

---

### 3.5. Command Pre-Flight Output Dashboards & Artifacts

The pre-flight validation engine renders a standardized 120-column ASCII
terminal dashboard and persists a structured JSON report artifact
(`preflight_report_<uid>.json`) under `<exp_root>/preflight/`.

> [!NOTE]
> **Detailed Target Reference**: For exhaustive terminal output dashboards across
> all 8 execution targets, probe interpretation details, and CLI options, refer to
> the standalone [Pre-Flight Readiness Guide](../preflight_readiness.md)
> ([French](../preflight_readiness_fr.md)).

#### Illustrative Terminal Dashboard (`model-train`):
```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: model-train
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "data-prepare" prerequisites verified.
 Filesystem     checkpoint_dir                PASS     Target directory is writable
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Ledger         pending_ingestion             PASS     Harmonization ledger up to date; nothing to ingest
 Ledger         prepared_blocks_state         PASS     Found 3 train / 1 val / 1 test prepared blocks ready
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

#### Pre-Flight Output Artifact (`preflight_report_<uid>.json`)
When report export is enabled, the preflight engine persists full diagnostic records
and summary telemetry as structured JSON files under
`<exp_root>/preflight/preflight_report_<timestamp>_<uid>.json`.

For the complete JSON artifact schema specification and sample payloads, refer to
the [Pre-Flight Readiness Guide](../preflight_readiness.md) (or French
[Guide de préparation avant vol](../preflight_readiness_fr.md)).

---

## 4. Implementation Plan

### Phase 1: Core Execution Infrastructure (Completed)
- Standardized atomic pipelines on dedicated runner classes (`DataHarmonization`,
  `ModelTraining`, etc.) exposing `.run()`.
- Created `src/landseg/execution/workflows/` with lightweight functional entry
  points (`execute_<workflow>(root_config)`), deferring class-based workflow
  runners.
- Updated `executor.py` to route `command=<name>` directly to atomic runner
  classes (`pipelines.*(root_config).run()`) or workflow functions
  (`workflows.execute_*(root_config)`).

### Phase 2: Workflow Migrations (`study_sweep` & `batch_ingest`) (Completed)
- Relocated `pipelines.study_sweep` to `workflows.study_sweep`.
- Refactored `pipelines.data_ingest` to execute single explicit batch targets.
- Implemented `workflows.batch_ingest` to manage multi-batch discovery,
  sequential queue processing, and error containment.
- Added unit tests for `batch_ingest` workflow and isolated single-batch
  `data_ingest` pipeline.

### Phase 3: Pre-Flight Validation Engine (`landseg.execution.preflight`) (Completed)
- Implemented inspection engine and probe registry in `src/landseg/execution/preflight/`.
- Built diagnostic probes: hardware/memory, spatial grid identity,
  ledger integrity, model contracts, and filesystem permissions.
- Added 120-column ASCII terminal dashboard formatting (with pass/warn/fail indicators).
- Wired `command=preflight` entry point and JSON artifact export under experiment root.

### Phase 4: Composite End-to-End Workflows (Completed)
- Implemented `workflows.e2e_intake` (`harmonize` $\rightarrow$ `ingest`).
- Implemented `workflows.e2e_experiment` (`world-grid` $\rightarrow$ `model-train`).
- Provided programmatic notebook helpers in `landseg.adapters.api`
  (`run_e2e_intake`, `run_e2e_experiment`).

---

## 5. Consequences

### Positive
- **Strict Pipeline Invariants**: Atomic pipelines maintain a predictable,
  testable 1-to-1 relationship with run targets, folders, and reports.
- **Clean Separation of Concerns**: Multi-run loops, retry policies, and
  search algorithms are isolated in `workflows/` without cluttering core ETL.
- **Fail-Fast Safety via Pre-Flight Checks**: Catching CRS mismatches,
  missing files, or pending batches in seconds before launching heavy GPU
  jobs saves significant developer time and compute costs.
- **Better Developer Experience**: Clear terminal dashboards report pipeline
  readiness and actionable guidance when prerequisites are missing.

### Negative / Considerations
- **Additional Abstraction Layer**: Developers must distinguish between
  atomic single-run pipelines (`pipelines/`) and multi-run workflows
  (`workflows/`).
- **CLI Complexity**: Command-line arguments must clearly indicate whether
  an invocation targets an atomic pipeline or a composite workflow.

---

## 6. Future Work: Config Boundary Separation and Modular User Recipes

During the migration of the Hydra configuration group from `pipeline` to
`command` and the consolidation of execution workflows, a natural architectural
tension emerged between **declarative domain configurations** and **imperative
command invocation parameters**:

### 6.1. Domain vs. Command Configuration Boundaries
- **Domain Configs** (`data`, `models`, `session`, `study`): Represent
  long-lived, experiment-scoped declarations of capabilities, contracts,
  and state (e.g. data geometries, neural backbones, loss weights, and
  search spaces).
- **Command Configs** (`command`): Represent the ephemeral execution context
  and invocation-specific arguments (e.g. `command.name`, `checkpoint`,
  `split`).
- In future work, we will evaluate decoupling historical sub-configs currently
  tucked under `CommandConfig` (such as `_TrainModel`, `model_evaluate`, and
  `study_sweep`):
  - Flatten invocation-only overrides directly under `CommandConfig`
    (e.g., `command.checkpoint: str | None = None`) to simplify CLI ergonomics
    (`command.checkpoint=...` rather than `command.model_evaluate.checkpoint=...`).
  - Consolidate domain-level evaluation and study properties into their
    natural domain containers (`session` and `study`).

### 6.2. Modularization of `user.yaml` into Recipe Sections
- `configs/user.yaml` is currently structured as an end-to-end, chronological
  recipe (`world-grid` $\rightarrow$ `data-harmonize` $\rightarrow$
  `data-ingest` $\rightarrow$ `data-prepare` $\rightarrow$ `model-train`).
- Fitting ad-hoc operational tasks (`model-evaluate`) and high-level
  orchestrations (`study-sweep`, `study-analysis`) into this single file risks
  re-bloating `user.yaml` with sections unused during standard training.
- In subsequent iterations, we will explore breaking `configs/user.yaml` into
  self-sustained, command-oriented recipe files under a `configs/recipes/`
  directory (e.g. `evaluate.yaml`, `sweep.yaml`, `batch_ingest.yaml`), while
  retaining `configs/user.yaml` as the streamlined master end-to-end wrapper.
- The CLI translation layer (`translate_user_config`) will be expanded to
  transparently support both modular recipe files and direct CLI parameter
  overrides.

### 6.3. Stateful Workflow Runners and DAG Orchestration
- Workflows are currently kept structurally simple as lightweight functional
  procedures (`execute_<workflow>(root_config)`).
- When multi-stage composite pipelines (such as `end_to_end_intake` or
  `e2e_experiment`) mature to require pause/resume capabilities, step-level
  error recovery, or dependency DAG graphs, we will evaluate introducing
  formal class-based workflow runners (`WorkflowRunner`) with unified
  lifecycle hooks.
