# ADR-0060: Higher-Level Workflow Orchestration and Pre-Flight Validation Engine

**Status:** Proposed<br>
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
- `data-prepare`: Partitions the canonical pool into one experiment dataset.
- `model-train`: Runs a single model training session in `run_XXXX`.
- `model-evaluate`: Evaluates a checkpoint across validation/test splits.
- `diagnose-overfit`: Executes a minimal overfit verification session.

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

3. **Continuous End-to-End Data Intake (`end_to_end_intake`)**:
   When raw GeoTIFF packages arrive, production pipelines must run
   `data-harmonize` followed immediately by `data-ingest` for each batch,
   accumulating data into the canonical block pool without manual
   intervention between stages.

4. **Full Experiment Workflows (`end_to_end_experiment`)**:
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

We will formalize a dedicated **higher-level workflow orchestration layer**
(`landseg.execution.workflows`) decoupled from atomic pipelines, and
introduce a **unified pre-flight validation engine** (`workflows.preflight`).

We will:
1. **Decouple Atomic Pipelines from Multi-Run Workflows**:
   - `landseg.execution.pipelines.*` will remain strictly scoped to atomic,
     single-target execution units (enforcing the 1-to-1 invariant).
   - `landseg.execution.workflows.*` will host multi-run iterators, composite
     multi-pipeline DAGs, and hyperparameter search loops.
2. **Refactor Ingestion into Atomic Pipeline and Batch Workflow**:
   - `pipelines.data_ingest` will execute an explicit, single harmonization
     batch target (`harmonize_run_id`).
   - `workflows.batch_ingest` will resolve pending batches from ledgers,
     handle queue ordering, and invoke the atomic pipeline sequentially.
3. **Relocate `study-sweep` to Workflows**:
   - Relocate `pipelines.study_sweep` to `workflows.study_sweep`, decoupling
     Optuna multi-trial search from atomic training sessions.
4. **Implement Pre-Flight Validation Engine**:
   - Provide a dry-run inspection framework that probes upstream ledgers,
     spatial grid invariants (`affine_identity`, `block_identity`),
     raster metadata integrity, and system compute resources prior to
     execution.
5. **Introduce Unified Workflow Registry and Execution Routing**:
   - Extend `landseg.execution` with a `WorkflowRegistry` mirroring the
     existing `PipelineRegistry`, exposing seamless CLI and programmatic
     routing.

---

## 3. Architecture & Design Specification

### 3.1. Layered Execution Hierarchy

```text
+-------------------------------------------------------------------------------+
|                             USER CONTROL SURFACE                              |
|           CLI (configs/user.yaml)    |    Programmatic API (Notebooks)        |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       landseg.execution.executor.py                           |
|       Unified router: Resolves configs and dispatches execution targets       |
+-------------------+-----------------------------------+-----------------------+
                    |                                   |
                    v                                   v
+---------------------------------------+   +-----------------------------------+
|     landseg.execution.workflows       |   |    landseg.execution.pipelines    |
|   (Multi-Run / Composite / Search)    |   |     (Atomic Single-Run Units)     |
|---------------------------------------|   |-----------------------------------|
| - batch_ingest (multi-batch loop)     |   | - world_grid                      |
| - study_sweep (Optuna trial search)   |   | - data_harmonize                  |
| - e2e_intake (harmonize+ingest)       |   | - data_ingest (single batch)      |
| - e2e_experiment (full 5-stage chain) |   | - data_prepare                    |
| - preflight (cross-pipeline dry run)  |   | - model_train                     |
+-------------------+-------------------+   | - model_evaluate                  |
                    |                       | - diagnose_overfit                |
                    +-- (delegates to) ---->+-----------------------------------+
                                                        |
                                                        v
                                            +-----------------------------------+
                                            |       geopipe / session core      |
                                            +-----------------------------------+
```

### 3.2. Workflow Contract (`WorkflowProtocol`)

Workflows will adhere to a standardized contract defined in
`landseg.execution.contracts.workflow`:

```python
class WorkflowProtocol(typing.Protocol):
    '''Standard protocol for multi-run and composite workflows.'''

    @property
    def workflow_name(self) -> str:
        ...

    def validate_prerequisites(
        self,
        config: RootConfig,
    ) -> PreflightInspectionResult:
        ...

    def execute(
        self,
        config: RootConfig,
    ) -> WorkflowExecutionSummary:
        ...
```

### 3.3. Atomic `data_ingest` vs. `batch_ingest` Workflow

Under this separation of concerns:
- **`pipelines.data_ingest.exec_ingest_data(config)`**:
  Accepts an explicit target batch (`target_harmonize_run_id`). It slices
  blocks, resolves collisions against the canonical pool using
  `CollisionPolicy`, writes `collisions.json` and `ingest_report.json` to
  `ingested_data/run_XXXX/`, and registers the run in `ingestion_runs.json`.
- **`workflows.batch_ingest.exec_batch_ingest(config)`**:
  Discovers un-ingested batches via `resolve_pending_ingestion_batches()`,
  verifies pool grid compatibility, and invokes `exec_ingest_data`
  sequentially for each pending batch. It collects individual run reports
  and emits a consolidated `batch_ingest_summary.json`.

### 3.4. Pre-Flight Validation Engine (`workflows.preflight`)

The pre-flight engine will implement a non-destructive dry-run inspection
pipeline:

```text
[PreflightEngine]
       |
       +--> [Probe: Environment & Hardware]
       |        (GPU availability, CUDA memory, VRAM requirements)
       |
       +--> [Probe: Grid & Spatial Identity]
       |        (world grid existence, CRS, affine_identity, block_identity)
       |
       +--> [Probe: Ledger Integrity & Catch-Up]
       |        (harmonization_runs.json, ingestion_runs.json, pending runs)
       |
       +--> [Probe: Dataset Inputs & Paths]
       |        (existence of raw GeoTIFFs, valid VRT pointers, schema)
       |
       `--> [Reporter: Terminal Dashboard & JSON Artifact]
                (emits ANSI status table and preflight_report.json)
```

#### Diagnostic Probe Categories:
1. **Hardware & Environment Probe**:
   - Detects GPU availability, device name, CUDA capability, and free VRAM.
   - Compares batch tensor size ($B \times C \times H \times W$) against available
     memory to warn about potential CUDA Out-Of-Memory (OOM) risks.
2. **Spatial Grid & Tensor Frame Probe**:
   - Validates existence and schema compliance of `world_grids/`.
   - Compares `incoming_grid.block_identity` against canonical `schema.json`.
   - Verifies Coordinate Reference Systems (CRS) across all configured
     feature and label sources.
3. **Ledger & Lineage Probe**:
   - Inspects `harmonization_runs.json` and `ingestion_runs.json` for
     dangling runs, failed attempts, or hash discrepancies.
   - Identifies exact pending batches awaiting ingestion.
4. **Raster & Path Integrity Probe**:
   - Confirms readability of raw source GeoTIFFs and virtual rasters (`.vrt`).
   - Verifies nodata values, data types, and band channel configurations.

#### Pre-Flight Output Artifact (`preflight_report.json`):
```json
{
  "timestamp": "2026-09-30T10:00:00Z",
  "target_pipeline": "model-train",
  "status": "READY",
  "probes": {
    "hardware": {
      "status": "PASS",
      "device": "NVIDIA RTX 4090",
      "vram_free_gb": 22.4
    },
    "grid_compatibility": {
      "status": "PASS",
      "block_identity": "EPSG:3161|(0,0)|10.0|(256,256)"
    },
    "upstream_artifacts": {
      "status": "PASS",
      "prepared_blocks_available": 1420
    },
    "pending_batches": {
      "status": "WARN",
      "message": "2 harmonization runs pending ingestion"
    }
  },
  "warnings": [
    "2 harmonization batches in harmonized_data/ are pending ingestion."
  ],
  "errors": []
}
```

---

## 4. Implementation Plan

### Phase 1: Core Workflow Infrastructure
- Create `src/landseg/execution/workflows/` directory structure.
- Define `WorkflowProtocol`, `WorkflowContext`, and `WorkflowExecutionSummary`.
- Implement `WorkflowRegistry` in `landseg.execution.workflows._registry.py`.
- Update `executor.py` and CLI translation layers to support dispatching
  both atomic pipelines (`pipeline=<name>`) and workflows (`workflow=<name>`).

### Phase 2: Workflow Migrations (`study_sweep` & `batch_ingest`)
- Relocate `pipelines.study_sweep` to `workflows.study_sweep`.
- Refactor `pipelines.data_ingest` to execute single explicit batch targets.
- Implement `workflows.batch_ingest` to manage multi-batch discovery,
  sequential queue processing, and error containment.
- Add unit tests for `batch_ingest` workflow and isolated single-batch
  `data_ingest` pipeline.

### Phase 3: Pre-Flight Validation Engine (`workflows.preflight`)
- Implement `PreflightEngine` and probe registry in `workflows/preflight/`.
- Build diagnostic probes: hardware/memory, spatial grid identity,
  ledger integrity, and raster paths.
- Add rich terminal status table formatting (with pass/warn/fail indicators).
- Wire `--dry-run` and `preflight` CLI entry points across all pipelines.

### Phase 4: Composite End-to-End Workflows
- Implement `workflows.end_to_end_intake` (`harmonize` $\rightarrow$ `ingest`).
- Implement `workflows.e2e_experiment` (`world-grid` $\rightarrow$ `model-train`).
- Provide programmatic notebook helpers in `landseg.adapters.api`.

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
