# ADR-0047: Geopipe Schematic Logging and JSON Execution Reports

**Status:** Accepted
**Date:** 2026-07-02

## 1. Context

During the data ingestion (`data-ingest`) and data preparation (`data-prepare`)
pipelines, we need structured telemetry detailing exactly what was processed,
how long each step took, and whether outputs were rebuilt or loaded from
cached states.

Previously:
* Pipeline logging consisted of unstructured standard text console logs and
  arbitrary `print()` statements.
* Metric tracking (e.g. data block counts, training class distribution
  matrices, and PCA explained variance) was not programmatically queryable
  or structured.
* Cache status was not tracked systematically, making it difficult to
  distinguish fresh runs from cached loads in pipeline audits.

## 2. Decision

We have established a unified, structured schematic logging pattern for
data-processing pipelines in `geopipe`.

### 2.1. Specialized Schema Loggers
We introduced two specialized loggers that wrap standard log outputs and
accumulate execution reports in memory:
* `FoundationLogger` (defined in `foundation/common/logger.py`)
* `TransformLogger` (defined in `transform/common/logger.py`)

Both loggers subclass `utils.Logger` to preserve standard logging handlers while
introducing specialized summary fields. When `close()` is called on the logger,
its `on_close()` hook persists the accumulated schema dictionary to a
structured JSON file (`ingest_report.json` or `prep_report.json`).

### 2.2. Typed Schemas
We defined strict TypedDict interfaces to model the exact JSON shape of
execution summaries:
* `IngestReportSchema` (in `foundation/common/schema.py`)
* `TransformReportSchema` (in `transform/common/schema.py`)

This allows type-checking and autocompletion of metric keys (such as
`unwanted_blocks_removed`, `base_label_count`, and `explained_variance`) across
split, normalization, and schema-building submodules.

### 2.3. Semantic Log Output Hierarchy
We standardized the naming and console prefix style to separate
orchestrator-level progress from internal module checkpoints:
* **Pipeline Orchestrators** (e.g. `data_ingest.py` and `data_prepare.py`) log
  pipeline block start and completion boundaries:
  - `[START] <Step Name>`
  - `[COMPLETE] <Step Name> (D_<duration>s)`
* **Internal Stages/Submodules** log progress markers using the standard
  logger prefix:
  - `[CHECKPOINT] Created/Loaded <Artifact>`

### 2.4. Explicit Logging Signatures
To enforce static type-safety:
* Submodule entrypoints (e.g. `run_datablocks_partition`,
  `run_normalize_blocks`, `build_schema`) explicitly accept the specialized
  `TransformLogger` or `FoundationLogger` instances in their signatures
  instead of generic `utils.Logger` types.
* We removed dynamic `hasattr` checks at call sites, allowing direct,
  statically checked calls to summary registration methods (such as
  `logger.set_normalization_report(report)`).

## 3. Consequences

### Positive
* **Traceable Telemetry**: Execution logs now output a single, well-structured
  telemetry file (`prep_report.json` and `ingest_report.json`) detailing
  execution states, which simplifies automated verification and downstream
  dashboarding.
* **Accurate Cache Auditing**: Restored cached states now cleanly log a
  `"loaded"` status flag alongside sub-millisecond execution times, preventing
  logs from masking cache hit successes as build operations.
* **Unified Console Styling**: Standardizing on the
  `[START]` / `[CHECKPOINT]` / `[COMPLETE]` hierarchy provides readable,
  structured terminal feedback.

### Negative
* **Schema Maintenance**: Adding new pipeline statistics requires updating the
  type dictionaries in both `foundation/common/schema.py` and
  `transform/common/schema.py`.
