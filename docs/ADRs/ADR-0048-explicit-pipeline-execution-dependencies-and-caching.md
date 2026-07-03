# ADR-0048: Explicit Pipeline Execution Dependencies and Caching

**Status:** Accepted
**Date:** 2026-07-03

## 1. Context

Following the implementation of structured pipeline logs and telemetry reports (ADR-0047),
`geopipe` generated `ingest_report.json` and `prep_report.json` to summarize ingestion and
preparation runs. However, the system had two major architectural gaps:
* **lack of dynamic caching policy controls**: Users had no option to force rebuild ingestion or
  preparation data artifacts through standard interfaces. Caching behavior was determined
  internally by task lifecycle managers.
* **silent execution with missing or stale upstream data**: Downstream pipelines (e.g. model
  training, evaluation, study analysis) executed assuming all upstream data preparation had
  completed successfully. There was no validation layer to confirm upstream run statuses.
  Furthermore, if a user modified their configuration file (e.g. changing the grid tile size)
  without re-running ingestion or preparation, pipelines ran silently using stale data artifacts
  constructed under different settings.

## 2. Decision

We have established a unified verification layer and dynamic cache control mechanisms in the
pipeline execution layer.

### 2.1. Dynamic Caching Policies via Rebuild Setting
We introduced a user-facing `rebuild` context setting (defaulting to `false`) within the
structured configurations:
* Mapped `rebuild` from `user.yaml` (either globally or within `data-ingest` and `data-prepare`
  blocks) to `execution.rebuild`.
* Exposed the rebuild configuration in python programmatic configurators for Jupyter notebook
  environments.
* Refactored `data_ingest.py` and `data_prepare.py` orchestrators to dynamically resolve the
  `LifecyclePolicy`:
  ```python
  policy = (
      artifacts.LifecyclePolicy.REBUILD
      if config.execution.rebuild
      else artifacts.LifecyclePolicy.BUILD_IF_MISSING
  )
  ```
  This resolved policy is passed directly to all data preparation tasks, allowing force-rebuilding
  of data blocks from scratch when desired.

### 2.2. Relocated Reports and Persistent Configuration Sidecars
To keep execution reports centrally managed under the `artifacts` module rather than directly in
the experiment root directory:
* `ingest_report.json` is relocated to `artifacts/foundation/ingest_report.json`.
* `prep_report.json` is relocated to `artifacts/transform/prep_report.json`.
* Upon successful execution of data-ingest and data-prepare pipelines, a serialized JSON sidecar
  containing the resolved configuration tree is persisted as `config.json` inside the same
  directories.

### 2.3. Pre-Execution Validation Guards
We introduced a unified execution validation hook in `executor.py` executed before dispatching
commands:
* **Upstream validation**:
  - `data-prepare` validates that `data-ingest` has completed successfully by checking if
    `ingest_report.json` exists and its status field is `"SUCCESS"`.
  - Downstream pipelines (`model-train`, `study-sweep`, etc.) validate that both `data-ingest`
    and `data-prepare` have completed successfully (via ingest and prep reports).
  - Any missing or failed upstream telemetry raises an `ArtifactError`.
* **Config staleness checker**:
  - In CLI mode (`execution.cli_mode = True`), the executor recursively compares active
    configuration parameters against the persisted sidecar `config.json`.
  - Slashes and path formats are normalized (via `os.path.abspath`) to prevent OS-specific path
    representations from triggering false positive mismatches.
  - Slices of configuration are targeted: `data-prepare` compares `foundation` configurations;
    downstream pipelines compare both `foundation` and `transform` configurations.

### 2.4. Interactive and Headless Execution Policies
To maintain compatibility across different execution platforms, validation alerts adapt
dynamically:
* **Interactive Shells (TTY)**: If config mismatches are found, the console outputs the key-value
  diffs and prompts the user for confirmation:
  `Stale artifacts detected. Do you want to proceed with execution anyway? [y/N]: `. Denial
  aborts execution with status code 1.
* **Headless Shells (Non-TTY)**: To prevent blocking automated workflows, a warning block
  detailing all configuration differences is logged to output, and execution automatically
  proceeds.
* **Notebook Sessions**: Staleness checking is skipped for notebook executions
  (`execution.cli_mode = False`) to prevent warnings when training with configurator defaults.

## 3. Consequences

### Positive
* **Fail-Fast execution**: Execution fails immediately with descriptive errors if upstream steps
  are missing or failed, preventing silent run misbehavior.
* **Configuration drift awareness**: Users are alerted to stale data artifacts if configuration
  parameters are changed between pipeline stages.
* **Standardized caching**: Caching and force-rebuilding are cleanly managed across CLI and
  notebook API surfaces.

### Negative
* **Execution file overhead**: Persisting additional configuration sidecars and reading JSON
  reports adds small file I/O operations at startup.
