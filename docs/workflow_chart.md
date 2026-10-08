## Current workflow

Last updated: 2026-10-07

```
[grid/builder]                           (1 World Grid – canonical spatial definition)
|
+--> [artifacts/controller]
|        (resolve/build/reuse grid artifact under policy)
|
+--> [grid/lifecycle]
|        (grid artifact persistence & validation)
|
+--> [geopipe/harmonize]                 (2 Data Harmonization – warp/resample rasters)
|        |
|        +--> [artifacts/controller]
|        |        (resolve/build/reuse harmonized raster batch)
|        |
|        +--> [geopipe/ledger]
|                 (register batch run in harmonization_runs.json)
|
+--> [geopipe/ingest]                    (3 Data Ingestion – canonical block pool)
|        |
|        +--> [ingest/blocks/assembler]
|        |        (slice raster tiles & domain features)
|        |
|        +--> [ingest/collision]
|        |        (evaluate collision policy: skip vs overwrite)
|        |
|        +--> [artifacts/controller]
|        |        (persist blocks into canonical pool catalog)
|        |
|        +--> [geopipe/ledger]
|                 (register batch run in ingestion_runs.json)
|
+--> [geopipe/prepare]                   (4 Data Preparation – experiment datasets)
|        |
|        +--> [prepare/partitioner]
|        |        (geographic AOI splitting into train/val/test)
|        |
|        +--> [artifacts/controller]
|        |        (persist manifests, normalization stats, DataSpecs)
|
+--> [models/factory]                    (5 Model construction & wiring)
|
+--> [session/factory]                   (6 Session construction boundary)
|        |
|        +--> [session/data]             (dataloaders and batching adapters)
|        |
|        +--> [session/engine]           (batch + epoch execution engines)
|        |
|        +--> [session/instrumentation]  (callbacks, logging, tracking, dashboards)
|        |
|        +--> [session/orchestration]    (lifecycle management and phase transitions)
|
+--> [execution/preflight]               (7 Pre-Flight Readiness Validation Engine)
|        |                               (non-destructive dry-run inspection)
|        +--> [probes/lineage]           (upstream pipeline prerequisites & ledgers)
|        +--> [probes/filesystem]        (directory writability & artifact overwrites)
|        +--> [probes/domain]            (spatial grid CRS/origin, raw manifests, model)
|        +--> [probes/hardware]          (CUDA/CPU detection, VRAM headroom estimation)
|        `--> [reporter]                 (120-col terminal dashboard & JSON export)
|
+--> [execution/executor]                (8 Execution Dispatch Layer: command=<name>)
         |
         +--> [execution/pipelines]      (8a Atomic Pipelines – 1:1 execution invariant)
         |        |
         |        +--> [WorldGridGeneration]   (build & persist world grid)
         |        +--> [DataHarmonization]     (warp single raw raster batch)
         |        +--> [DataIngestion]         (ingest single batch into pool)
         |        +--> [DataPreparation]       (partition pool into DataSpecs)
         |        +--> [ModelTraining]         (full model training session)
         |        +--> [ModelEvaluation]       (single-pass checkpoint evaluation)
         |
         `--> [execution/workflows]      (8b Composite Multi-Run Workflows)
                  |
                  +--> [batch_ingest]          (resolve ledger queue & loop ingestion)
                  +--> [diagnose_overfit]      (minimal-scope end-to-end diagnostic)
                  +--> [study_sweep]           (Optuna hyperparameter study trials)
                  +--> [study_analysis]        (Optuna study reporting & trial metrics)
                  +--> [default]               (system-wide preflight audit)
```
---

### Interpretation notes (updated)

- All foundation build steps (grid, harmonize, ingest, prepare) remain pure,
  deterministic, and side-effect free in computation.

- All artifact reuse, rebuild, overwrite, and validation decisions are
  centralized through `artifacts.controller`, enforcing policy-driven
  lifecycle management.

- Run ledgers (`harmonization_runs.json`, `ingestion_runs.json`) track
  batch history, run fingerprints, and collision resolutions.

- Downstream stages operate on resolved artifacts, not on recomputed or
  implicit intermediates.

- `DataSpecs` and model construction occur strictly before the session
  boundary and are treated as fully-resolved, immutable inputs.

- Session construction is centralized in `session/factory` and owns:
  - data interface construction (dataloaders, samplers)
  - component assembly (model bindings, losses, optimizers)
  - runtime state initialization
  - callback and instrumentation binding
  - execution engine instantiation
  - lifecycle orchestration setup

- Session internals are fully configured prior to execution; no
  structural mutation occurs during runtime.

- Execution engines operate on injected state and components and do
  not encode configuration or lifecycle decisions.

- Lifecycle control (train/validate phases and transitions) is handled
  by session orchestration, not by pipelines.

- Pre-flight readiness validation (`execution.preflight`) provides a
  non-destructive inspection engine capable of validating prerequisites,
  storage writability, domain contracts, ledgers, and compute resources
  before any heavy pipeline or workflow compute is dispatched.

- The execution layer enforces a clear bifurcation between:
  - **Atomic pipelines** (`execution.pipelines`): dedicated runner classes
    (`WorldGridGeneration`, `DataHarmonization`, `DataIngestion`,
    `DataPreparation`, `ModelTraining`, `ModelEvaluation`) enforcing the
    strict 1-to-1 invariant (1 invocation $\rightarrow$ 1 target $\rightarrow$
    1 run directory $\rightarrow$ 1 report).
  - **Composite workflows** (`execution.workflows`): lightweight procedural
    functions coordinating multi-run loops (`batch_ingest`), optimization
    sweeps (`study_sweep`), diagnostics (`diagnose_overfit`), or analyses
    (`study_analysis`).

- The system maintains a unidirectional dependency flow:
  `geopipe (ETL) → artifacts → models → session → execution (preflight, pipelines, workflows)`

- This structure ensures:
  - reproducibility via explicit artifact and ledger tracking
  - fail-fast safety through non-destructive pre-flight validation
  - strict separation of build-time vs runtime concerns
  - composable and predictable pipelines
  - deterministic reconstruction from configs + artifacts