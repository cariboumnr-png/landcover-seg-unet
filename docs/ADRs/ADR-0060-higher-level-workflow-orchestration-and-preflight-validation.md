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
5. **Introduce Runner Classes for Atomic Pipelines (Deferring Workflow Runners)**:
   - Encapsulate each atomic pipeline within a dedicated runner class
     (`WorldGridGeneration`, `DataHarmonization`, `DataIngestion`,
     `DataPreparation`, `ModelTraining`, `ModelEvaluation`) with a consistent
     `.run()` invocation lifecycle.
   - Retain workflows as structurally simple, lightweight functions
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
+-------------------+-----------------------------------+----------------------+
                    |                                   |
                    v                                   v
+--------------------------------------+     +---------------------------------+
|     landseg.execution.workflows      |     |   landseg.execution.pipelines   |
|   (Multi-Run / Composite / Search)   |     |    (Atomic Single-Run Units)    |
|--------------------------------------|     |---------------------------------|
| - batch_ingest (multi-batch loop)    |     | - WorldGridGeneration           |
| - diagnose_overfit (overfit test)    |     | - DataHarmonization             |
| - study_sweep (Optuna trial search)  |     | - DataIngestion (single batch)  |
| - study_analysis (study reporting)   |     | - DataPreparation               |
| - preflight (cross-pipeline dry run) |     | - ModelTraining                 |
| - e2e_intake (harmonize+ingest)      |     | - ModelEvaluation               |
| - e2e_experiment (full 5-stage chain)|     |                                 |
+------------------+-------------------+     +----------------+----------------+
                   |                                          ^
                   +-----------------(delegates to)-----------+
                                                              |
                                                              v
                                             +---------------------------------+
                                             |     geopipe / session core      |
                                             +---------------------------------+
```

### 3.2. Execution Abstractions: Pipeline Runner Classes vs. Functional Workflows

To balance operational encapsulation with architectural simplicity, execution
units will adopt a bifurcated abstraction model:

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

Workflows will be exposed as straightforward, lightweight functions accepting
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

### 3.4. Pre-Flight Validation Engine (`workflows.preflight`)

The pre-flight engine will implement a non-destructive dry-run inspection
pipeline capable of evaluating execution prerequisites across both atomic
pipelines and composite workflows prior to committing compute:

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
   - Compares batch tensor size ($B \times C \times H \times W$) against
     available memory to warn about potential CUDA Out-Of-Memory (OOM) risks.
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

#### Command & Programmatic Interface:

We will introduce `command=preflight` as a first-class execution target in
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

### 3.5. Command Pre-Flight Output Templates

The pre-flight validation engine will emit standardized terminal status
dashboards and a structured JSON artifact (`preflight_report.json`) tailored
to the specific dependencies of each command:

#### 1. `world-grid` (Grid Definition & CRS)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: world-grid                          
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Spatial    crs_validity           PASS     EPSG:3161 (NAD83 / Ontario MNR)     
 Spatial    pixel_resolution       PASS     Resolution: 10.0m x 10.0m           
 Spatial    block_dimensions       PASS     256x256 px (2,560m x 2,560m tile)   
 Path       bounds_geojson         PASS     'data/aoi/study_area.geojson' exists
 Filesystem output_directory       PASS     'data/world_grids/' is writable     
 Filesystem existing_grid_check    WARN     'world_grid.geojson' exists (overwrite)
================================================================================
 STATUS: READY (1 warning)
 Telemetry: Estimated grid coverage: ~4,200 blocks across AOI
================================================================================
```

#### 2. `data-harmonize` (Raster Reprojection & Resampling)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: data-harmonize                      
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Dependency world_grid_spec        PASS     Found 'data/world_grids/schema.json'
 Path       raw_raster_manifest    PASS     Found 8 source GeoTIFFs (14.2 GB)   
 GDAL       driver_support         PASS     GTiff / VRT drivers available       
 Geometry   spatial_intersection   PASS     Source rasters intersect AOI (100%) 
 Raster     band_channel_match     PASS     All files contain 4 bands (RGBN)    
 Ledger     harmonize_ledger       PASS     'harmonization_runs.json' healthy   
================================================================================
 STATUS: READY
 Telemetry: 8 files to warp -> estimated harmonized output: ~16.8 GB
================================================================================
```

#### 3. `data-ingest` (Atomic Single-Batch Ingestion)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: data-ingest                         
 Target Batch: run_0003                                                         
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Lineage    target_run_exists      PASS     'harmonized_data/run_0003' completed
 Grid       block_identity_match   PASS     EPSG:3161|(0,0)|10.0|(256,256) matches
 Policy     collision_policy       PASS     'skip' policy configured            
 Ledger     canonical_pool_state   PASS     Pool contains 1,280 existing blocks 
 Storage    pool_write_access      PASS     'data/canonical_pool/' writable     
================================================================================
 STATUS: READY
 Telemetry: Batch run_0003 contains ~340 candidate blocks; ~15 known duplicates
================================================================================
```

#### 4. `batch-ingest` (Multi-Batch Ingestion Workflow)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: batch-ingest                        
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Ledger     harmonization_runs     PASS     4 completed runs in ledger          
 Lineage    pending_batch_queue    PASS     2 batches pending (run_0003, run_0004)
 Grid       pool_grid_homogeneity  PASS     All batches match pool CRS & affine 
 Storage    disk_capacity          PASS     Available: 142 GB (est. needed: 4.8 GB)
 Policy     collision_policy       WARN     Policy 'overwrite' replaces blocks  
================================================================================
 STATUS: READY (1 warning)
 Telemetry: Queue: [run_0003, run_0004] | Total incoming blocks: ~680
================================================================================
```

#### 5. `data-prepare` (Partitioning & Label Mapping)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: data-prepare                        
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Lineage    canonical_pool_ready   PASS     1,960 blocks available in pool      
 Labels     scheme_reclassification PASS    All 12 source classes map to targets
 Labels     unmapped_classes       PASS     0 unmapped raster class IDs detected
 Dataset    split_ratios           PASS     train: 0.70 | val: 0.15 | test: 0.15
 Dataset    split_strategy         PASS     'spatial_block_hash' valid          
 Storage    experiment_dir         PASS     'data/prepared/exp_01/' writable    
================================================================================
 STATUS: READY
 Telemetry: Estimated split counts: Train=1,372 | Val=294 | Test=294
================================================================================
```

#### 6. `model-train` (Deep Learning Session)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: model-train                         
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Dataset    prepared_manifest      PASS     Found 1,372 train / 294 val blocks  
 Hardware   cuda_device            PASS     NVIDIA RTX 4090 (Device 0)          
 Memory     vram_headroom          PASS     22.4 GB free / ~2.2 GB est. batch   
 Model      backbone_registry      PASS     'resnet34' recognized by smp/timm   
 Model      channel_compatibility  PASS     Input channels (4) match data (4)   
 Ledger     checkpoint_dir         PASS     'runs/train/run_0012/' writable     
 Lineage    pending_data_warning   WARN     2 batches pending in harmonization  
================================================================================
 STATUS: READY (1 warning)
 Telemetry: Batch: (32, 4, 256, 256) | Precision: amp_bf16 | Est. step: ~42ms  
================================================================================
```

#### 7. `model-evaluate` (Checkpoint Evaluation)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: model-evaluate                      
 Checkpoint: runs/train/run_0010/checkpoints/best_miou.pt                       
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 File       checkpoint_exists      PASS     Weights file exists (86.4 MB)       
 Integrity  torch_state_dict       PASS     Valid checkpoint state dictionary   
 Model      architecture_match     PASS     State dict matches 'unet_resnet34'  
 Dataset    eval_split_exists      PASS     'test' split has 294 samples        
 Metrics    metric_registry        PASS     [mIoU, F1, PixelAccuracy] configured
 Storage    report_destination     PASS     'runs/eval/run_0010/' writable      
================================================================================
 STATUS: READY
 Telemetry: Model parameters: 24.4M | Target split: 'test' (294 tiles)         
================================================================================
```

#### 8. `diagnose-overfit` (Fast Overfit Verification)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: diagnose-overfit                    
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Data       batch_builder          PASS     Batch constructed (2, 4, 256, 256)  
 Compute    device_allocation      PASS     Allocated on cuda:0                 
 Graph      forward_pass           PASS     Logits shape (2, 6, 256, 256)       
 Autograd   backward_gradient      PASS     Loss computed (0.693); no NaNs      
 Memory     peak_overhead          PASS     Peak allocation: 340 MB             
================================================================================
 STATUS: READY
 Telemetry: Minimal autograd step verified. Ready for overfit test.
================================================================================
```

#### 9. `study-sweep` (Optuna Hyperparameter Search)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: study-sweep                         
 Study: canopy_seg_v1 | Storage: sqlite:///optuna.db                           
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Storage    rdbms_connection       PASS     Connected to sqlite:///optuna.db    
 Optuna     study_schema           PASS     Study exists / initialized cleanly  
 Search     parameter_space        PASS     All 6 search ranges valid           
 Pruning    pruner_algorithm       PASS     MedianPruner configured properly    
 Hardware   gpu_concurrency        PASS     1 GPU available (1 worker active)   
 Dependency model_train_contract   PASS     Prepared dataset and backbones ready
 Budget     compute_budget_warning WARN     50 trials x 30 epochs: ~14.5 GPU hrs
================================================================================
 STATUS: READY (1 warning)
 Telemetry: 6 hyperparameters in space | Target metric: 'val_miou' (maximize)  
================================================================================
```

#### 10. `study-analysis` (Optuna Study Reporting)
```text
================================================================================
               PRE-FLIGHT READINESS CHECK: study-analysis                      
 Study: canopy_seg_v1                                                           
================================================================================
 CATEGORY   PROBE ID               STATUS   DETAILS                             
--------------------------------------------------------------------------------
 Storage    rdbms_connection       PASS     Connected to sqlite:///optuna.db    
 Study      study_exists           PASS     Study 'canopy_seg_v1' found         
 Trials     trial_count_threshold  PASS     38 completed trials (min req: 5)    
 Trials     pruned_trial_count     INFO     8 trials pruned                     
 Trials     failed_trial_count     PASS     0 failed trials                     
 Storage    report_directory       PASS     'runs/studies/canopy_seg_v1/' valid 
================================================================================
 STATUS: READY
 Telemetry: Best trial: Trial #24 (val_miou = 0.814)
================================================================================
```

#### 11. `all` (Full System-Wide Readiness Audit)
```text
================================================================================
                   LANDSEG SYSTEM-WIDE READINESS AUDIT                         
================================================================================
 PIPELINE / WORKFLOW    STATUS   KEY FINDINGS                                   
--------------------------------------------------------------------------------
 1. world-grid          READY    Grid EPSG:3161 defined; 4,200 blocks coverage  
 2. data-harmonize      READY    8 raw GeoTIFFs (14.2 GB) ready to warp         
 3. data-ingest         READY    No active single batch selected                
 4. batch-ingest        READY*   2 batches pending ingestion in ledger          
 5. data-prepare        READY    1,960 pool blocks ready; label scheme valid    
 6. model-train         READY*   GPU ready (RTX 4090); pending batches notice   
 7. model-evaluate      BLOCKED  No checkpoint specified (set checkpoint=...)  
 8. diagnose-overfit    READY    Forward/backward autograd healthy              
 9. study-sweep         READY    sqlite:///optuna.db accessible                 
 10. study-analysis     READY    38 trials recorded in database                 
================================================================================
 SYSTEM STATUS: 9 READY | 1 BLOCKED (model-evaluate missing checkpoint parameter)
================================================================================
```

#### Pre-Flight Output Artifact (`preflight_report.json`):
```json
{
  "timestamp": "2026-10-04T12:35:00Z",
  "target": "model-train",
  "status": "READY",
  "strict": false,
  "exit_code": 0,
  "summary": {
    "total_probes": 7,
    "pass": 6,
    "warn": 1,
    "fail": 0,
    "skip": 0
  },
  "probes": [
    {
      "probe_id": "vram_headroom",
      "category": "hardware",
      "status": "PASS",
      "message": "22.4 GB free / ~2.2 GB est. batch footprint",
      "details": {
        "vram_free_bytes": 24051810304,
        "estimated_batch_bytes": 2362232000,
        "headroom_ratio": 10.18
      }
    },
    {
      "probe_id": "pending_data_warning",
      "category": "lineage",
      "status": "WARN",
      "message": "2 batches pending in harmonization ledger",
      "details": {"pending_run_ids": ["run_0003", "run_0004"]}
    }
  ],
  "telemetry": {
    "batch_shape": [32, 4, 256, 256],
    "precision": "amp_bf16",
    "est_step_ms": 42
  }
}
```

---

## 4. Implementation Plan

### Phase 1: Core Execution Infrastructure
- Standardize atomic pipelines on dedicated runner classes (`DataHarmonization`,
  `ModelTraining`, etc.) exposing `.run()`.
- Create `src/landseg/execution/workflows/` with lightweight functional entry
  points (`execute_<workflow>(root_config)`), deferring class-based workflow
  runners.
- Update `executor.py` to route `command=<name>` directly to atomic runner
  classes (`pipelines.*(root_config).run()`) or workflow functions
  (`workflows.execute_*(root_config)`).

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
