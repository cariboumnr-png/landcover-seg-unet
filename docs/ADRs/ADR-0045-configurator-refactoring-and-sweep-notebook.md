# ADR-0045: Configurator Refactoring, Sweep Notebook Integration, and Databricks Portability

**Status:** Accepted
**Date:** 2026-06-24
**Related ADRs:** ADR-0026, ADR-0029, ADR-0038, ADR-0044

---

## 1. Context

The codebase contains separate configuration classes (e.g.,
`DataIngestionConfigurator`, `DataPreparationConfigurator`, and
`TrainingSessionConfigurator`) to bridge Jupyter notebooks with programmatic
pipeline execution. However:

1. Significant configuration duplication existed between configurators (e.g.,
   duplicated setup paths, dataset name setup, data loading settings, and domain
   source configuration).
2. Advanced orchestration pipelines (such as `study-sweep`) were originally
   CLI-bound under ADR-0029 and lacked a corresponding programmatic notebook
   configurator.
3. Path configurations used relative file coordinates which are error-prone when
   run across diverse notebook execution environments (e.g., local VS Code
   interactive notebooks vs Databricks clusters).
4. Running concurrent SQLite studies in Databricks clusters using shared network 
   volume storage (like DBFS or Unity Catalog Volumes) suffered from sqlite locking 
   errors, resulting in trial database corruption.

***

## 2. Decision

We have unified the programmatic configurator design, integrated the
hyperparameter sweep pipeline into the programmatic API, and improved
environment portability.

The following changes have been implemented:

### 2.1. Unified Configurator Inheritance & Refactoring
*   We created a common base class, `BaseConfigurator`, which centralizes common
    variables (experiment_root, dataset_name, and pipeline output directory
    setup) and configuration validation checks.
*   We moved shared configurator methods to `BaseConfigurator`:
    *   `set_data_loading(batch_size, patch_size)`
    *   `set_domain_source(category_domain, continuous_domain)`
    *   `set_tasks(logit_adjust_alpha, exclude_classes, loss_weights)`
    *   `set_runtime(max_epochs, active_heads, patience_epoch, track_heads)`
*   We refactored existing configurators to inherit from the base class:
    *   `DataIngestionConfigurator` and `DataPreparationConfigurator` now inherit from
        `BaseConfigurator`. `DataPreparationConfigurator` simplifies data scoring by
        exposing `set_oversampling(target_head, reward_classes)` and removing
        the unused `set_hydration` helper.
    *   `TrainingSessionConfigurator` inherits from `BaseConfigurator`, removing
        redundant logic. The loss weights configuration is unified through
        `set_tasks` and matches parameter key names (supporting both `'tv'` and
        `'tv_loss'` mapping to `'total_var'`).

### 2.2. Programmatic Sweep Execution
*   We implemented StudySweepConfigurator to configure hyperparameter sweeps
    (e.g., optimizer learning rates, backbone channel configurations,
    bottlenecks, weight decay) programmatically using Optuna study presets
    defined in ADR-0044.
*   We registered the `study-sweep` pipeline execution path in
    pipelines/_registry.py and added execution logic in study_sweep.py.

### 2.3. Jupyter Notebook Refinements
*   We introduced 03_study_sweep.ipynb to demonstrate programmatic execution
    of hyperparameter sweep trials.
*   We added environment check guards inside 01_data_preparation.ipynb,
    02_model_train.ipynb, and 03_study_sweep.ipynb to warn/prevent DBFS path
    bottlenecks and automatically point Optuna SQLite storage to the
    driver-local `/tmp` directory on Databricks clusters.
*   We corrected relative path formatting issues (replacing malformed
    `..experiment/` strings with `../experiment/`).

### 2.4. Hardened Absolute Path Handling in Geopipe
*   We updated catalog.py and normalize.py to resolve manifest indices using
    `os.path.abspath(fp)`.
*   We relaxed constraints in adapter.py to make focal target validation
    conditional, preventing assertions when no focal target is set.

***

## 3. Consequences

### Positive
*   **Reduced Duplication**: configurator classes are unified under a single
    parent, reducing boilerplate and API drift.
*   **Programmatic Sweeping**: users can now trigger, analyze, and iterate on
    Optuna sweeps directly from Jupyter.
*   **Databricks Ready**: notebooks are fully optimized for cloud execution by
    bypassing SQLite concurrency lock limitations on DBFS.
*   **Robust I/O**: absolute path conversions ensure geopipe operations succeed
    when executing from arbitrary working directories.

### Negative
*   Inheritance introduces coupling between configurators, so updates to the
    base configurator signature must be carefully checked for compatibility.

***

## 4. Outcome

This architectural transition has been fully implemented, validated under python
compile tests, and run from notebook templates. Accordingly, this ADR is now
**Accepted**.
