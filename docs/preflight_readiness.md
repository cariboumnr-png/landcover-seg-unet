# Pre-Flight Readiness & Diagnostic Inspection Guide

[English](./preflight_readiness.md) | [Français](./preflight_readiness_fr.md)

Last updated: 2026-10-07

---

## Overview

Geospatial deep learning pipelines require significant computational resources:
raster warping, slicing multi-channel tiles, and running deep neural networks on
GPUs are time- and memory-intensive operations. When pipelines encounter missing
prerequisites, mismatched Coordinate Reference Systems (CRS), unwritable storage,
or missing weights deep inside a run, compute time is wasted and partially
materialized directories can be left in an inconsistent state.

The **Pre-Flight Validation Engine** (`landseg.execution.preflight`) provides a
unified, non-destructive diagnostic audit layer. It inspects pipeline and
workflow prerequisites in dry-run mode prior to execution, emitting standardized
120-column ASCII terminal dashboards and persisting structured JSON report
artifacts.

---

## Contents

- [CLI and Programmatic Invocations](#cli-and-programmatic-invocations)
- [Canonical Probe Ordering](#canonical-probe-ordering)
- [Probe Status Evaluation](#probe-status-evaluation)
- [Target Inspection Reference](#target-inspection-reference)
  - [1. world-grid](#1-world-grid)
  - [2. data-harmonize](#2-data-harmonize)
  - [3. data-ingest](#3-data-ingest)
  - [4. batch-ingest](#4-batch-ingest)
  - [5. data-prepare](#5-data-prepare)
  - [6. model-train](#6-model-train)
  - [7. model-evaluate](#7-model-evaluate)
  - [8. diagnose-overfit](#8-diagnose-overfit)
  - [9. all (System-Wide Audit)](#9-all-system-wide-audit)
- [Pre-Flight Report Artifacts](#pre-flight-report-artifacts)

---

## CLI and Programmatic Invocations

### Command-Line Interface (CLI)

Pre-flight checks are invoked via `command=preflight` through Hydra:

```bash
# Run system-wide readiness audit across all 8 supported targets
python scripts/run.py command=preflight

# Inspect a specific pipeline or workflow target
python scripts/run.py command=preflight command.preflight.target=model-train
python scripts/run.py command=preflight command.preflight.target=batch-ingest

# Enable strict mode (warnings cause execution gate failure)
python scripts/run.py command=preflight command.preflight.target=data-harmonize command.preflight.strict=true

# Disable report JSON export (console dashboard only)
python scripts/run.py command=preflight command.preflight.export_report=false
```

### Programmatic Python API

Pre-flight inspection can also be integrated into notebooks and scripts:

```python
import landseg

# Compose or load RootConfig
config = landseg.load_config()

# Run preflight inspection
result = landseg.run_preflight(config, target='model-train')

# Evaluate readiness
if not result.is_ready:
    print(f'Execution blocked! Errors: {result.errors}')
```

---

## Canonical Probe Ordering

Diagnostic probes are executed in a strictly defined, predictable sequence:

$$\text{Lineage} \longrightarrow \text{Filesystem} \longrightarrow \text{Domain Contracts} \longrightarrow \text{Ledger} \longrightarrow \text{Hardware}$$

1. **`Lineage`**: Upstream pipeline dependencies, completed reports, and
   prerequisite artifacts on disk.
2. **`Filesystem`**: Output directory writability and target file overwrite /
   rebuild status.
3. **`Domain Contracts`**:
   - **`Spatial`**: Reference rasters, CRS definitions, pixel resolution,
     bounds, and tile specifications.
   - **`Dataset`**: Source dataset manifest file existence and item count.
   - **`Policy`**: Block collision resolution policies (`skip` vs `overwrite`).
   - **`Model`**: Architecture recognition in registry, evaluation checkpoints,
     and split definitions.
4. **`Ledger`**: Run history manifests (`harmonization_runs.json`,
   `ingestion_runs.json`), pending batches awaiting ingestion, and canonical
   block pool counts.
5. **`Hardware`**: Compute accelerator detection (CUDA device name) and VRAM
   headroom estimation against batch dimensions.

---

## Probe Status Evaluation

Each probe evaluates to one of four canonical states:

| Status | Meaning | Impact on Target Status |
| :--- | :--- | :--- |
| `PASS` | Condition verified and completely healthy. | Target remains `READY`. |
| `WARN` | Advisory notice (e.g. CPU fallback, existing blocks, nothing pending). | Target remains `READY` in standard mode; fails in `strict=true` mode. |
| `FAIL` | Blocking condition (e.g. missing report, unwritable path, missing checkpoint). | Target status set to `BLOCKED`. |
| `SKIP` | Probe intentionally omitted or not applicable. | Neutral. |

An execution target is considered **`READY`** only when all probes evaluate to
`PASS` or `WARN` (in non-strict mode). If any probe evaluates to `FAIL`, the target
is marked **`BLOCKED`**.

---

## Target Inspection Reference

### 1. `world-grid`

Validates spatial tiling parameters and reference raster sources:

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: world-grid
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Filesystem     world_grid_output             PASS     Target directory is writable
 Filesystem     world_grid_report             PASS     Target file already exists; force_rebuild: False
 Spatial        world_grid_reference          PASS     Reference raster found at: data/reference/ref.tif
 Spatial        crs                           PASS     Target CRS is defined by the reference raster
 Spatial        pixel_size                    PASS     Pixel size is defined by the reference raster
 Spatial        extent                        PASS     Extent is defined by the reference raster
 Spatial        origin                        PASS     Origin is defined by the reference raster
 Spatial        grid_specs                    PASS     Grid specifications valid
========================================================================================================================
 STATUS: READY (0 errors, 0 warnings)
========================================================================================================================
```

### 2. `data-harmonize`

Validates world grid completion, raw input raster manifest, and past run records:

```text
========================================================================================================================
                                       PRE-FLIGHT READINESS CHECK: data-harmonize
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "world-grid" report verified.
 Filesystem     harmonization_output          PASS     Target directory is writable
 Dataset        source_dataset_manifest       PASS     Found 8 rasters available for harmonization at: input/raw
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         harmonized_dataset            PASS     Dataset not yet harmonized
========================================================================================================================
 STATUS: READY (0 errors, 0 warnings)
========================================================================================================================
```

### 3. `data-ingest`

Validates single-batch ingestion prerequisites, collision policies, and ledger state:

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: data-ingest
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream pipeline "data-harmonize" manifest verified (1 succes...
 Filesystem     ingestion_output              PASS     Target directory is writable
 Policy         collision_policy              PASS     Collision policy 'skip' configured
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         past_ingestion_runs           PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         pending_ingestion             WARN     Harmonization ledger up to date; nothing to ingest
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 4. `batch-ingest`

Validates multi-batch catch-up ingestion queue, collision handling, and block pool state:

```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: batch-ingest
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Filesystem     ingestion_output              PASS     Target directory is writable
 Policy         collision_policy              PASS     Collision policy 'skip' configured
 Ledger         past_harmonization_runs       PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         past_ingestion_runs           PASS     Read run history manifest with 1 total runs with 1 success runs
 Ledger         pending_ingestion             WARN     Harmonization ledger up to date; nothing to ingest
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 5. `data-prepare`

Validates canonical block pool availability and prepared block splits:

```text
========================================================================================================================
                                        PRE-FLIGHT READINESS CHECK: data-prepare
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Required artifacts verified for "data-ingest".
 Lineage        pipeline_prerequisites        PASS     Upstream pipeline "data-ingest" manifest verified (1 successfu...
 Filesystem     preparation_output            PASS     Target directory is writable
 Filesystem     preparation_report            PASS     Target file already exists; force_rebuild: False
 Ledger         ingested_blocks_pool          PASS     Ingestion pool contains 15 existing blocks
 Ledger         prepared_blocks_state         WARN     Found existing prepared blocks; train: 3 | val: 1 | test: 1
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 6. `model-train`

Validates data preparation completion, model architecture, pending batches notice, and GPU compute:

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

### 7. `model-evaluate`

Validates model weights checkpoint existence, model architecture, and target evaluation split:

```text
========================================================================================================================
                                       PRE-FLIGHT READINESS CHECK: model-evaluate
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Lineage        pipeline_prerequisites        PASS     Upstream "data-prepare" prerequisites verified.
 Filesystem     eval_output                   PASS     Target directory is writable
 Model          checkpoint_exists             FAIL     Checkpoint file not found: None
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Model          eval_split_configured         PASS     Evaluation target split 'test' configured
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: BLOCKED (1 errors, 1 warnings)
========================================================================================================================
```

### 8. `diagnose-overfit`

Validates fast forward/backward model compute and hardware state:

```text
========================================================================================================================
                                      PRE-FLIGHT READINESS CHECK: diagnose-overfit
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
```

### 9. `all` (System-Wide Audit)

When invoked with `target=all`, each target dashboard is rendered sequentially,
followed by an aggregate system status banner and report export path:

```text
========================================================================================================================
                                         PRE-FLIGHT READINESS CHECK: world-grid
... (individual reports for all 8 targets) ...
========================================================================================================================
                                      PRE-FLIGHT READINESS CHECK: diagnose-overfit
========================================================================================================================
 CATEGORY       PROBE ID                      STATUS   DETAILS
------------------------------------------------------------------------------------------------------------------------
 Model          model_body                    PASS     Configured architecture: "unetppp" recognized in registry
 Hardware       cuda_device                   WARN     CUDA unavailable; compute running on CPU
========================================================================================================================
 STATUS: READY (0 errors, 1 warnings)
========================================================================================================================
 SYSTEM STATUS: 7 READY | 1 BLOCKED
========================================================================================================================
Preflight report saved to: ./experiment/preflight/preflight_report_20261007_200100_a88673.json
```

---

## Pre-Flight Report Artifacts

When report export is enabled (`command.preflight.export_report=true`, default),
reports are written to:

```text
<exp_root>/preflight/preflight_report_<timestamp>_<hex>.json
```

### Schema Structure

```json
{
  "timestamp": "2026-10-07T20:01:00Z",
  "uid": "20261007_200100_a88673",
  "target": "all",
  "status": "BLOCKED",
  "strict": false,
  "is_ready": false,
  "summary": {
    "total_probes": 28,
    "pass": 24,
    "warn": 3,
    "fail": 1,
    "skip": 0
  },
  "targets": [
    {
      "target": "model-train",
      "status": "READY",
      "is_ready": true,
      "probes": [
        {
          "probe_id": "pipeline_prerequisites",
          "category": "Lineage",
          "status": "PASS",
          "message": "Upstream \"data-prepare\" prerequisites verified.",
          "details": {}
        },
        {
          "probe_id": "checkpoint_dir",
          "category": "Filesystem",
          "status": "PASS",
          "message": "Target directory is writable",
          "details": {"target_dir": "experiment/results"}
        },
        {
          "probe_id": "model_body",
          "category": "Model",
          "status": "PASS",
          "message": "Configured architecture: \"unetppp\" recognized in registry",
          "details": {"model_body": "unetppp"}
        },
        {
          "probe_id": "pending_ingestion",
          "category": "Ledger",
          "status": "PASS",
          "message": "Harmonization ledger up to date; nothing to ingest",
          "details": {}
        },
        {
          "probe_id": "prepared_blocks_state",
          "category": "Ledger",
          "status": "PASS",
          "message": "Found 3 train / 1 val / 1 test prepared blocks ready",
          "details": {"train_blocks": 3, "val_blocks": 1, "test_blocks": 1}
        },
        {
          "probe_id": "cuda_device",
          "category": "Hardware",
          "status": "WARN",
          "message": "CUDA unavailable; compute running on CPU",
          "details": {"device_name": "cpu", "cuda": false}
        }
      ],
      "errors": [],
      "warnings": [
        "CUDA unavailable; compute running on CPU"
      ],
      "telemetry": {}
    }
  ]
}
```
