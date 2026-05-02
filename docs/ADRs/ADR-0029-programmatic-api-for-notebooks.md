# ADR-0029 Transition to Programmatic API for Data & Training Notebook Execution

**Status:** Accepted
**Date:** 2026-05-01

---

## Context

The landcover-seg-unet pipeline (spanning data ingestion, preparation, training,
and sweeping) originally relied exclusively on a CLI driven by Hydra’s `@hydra.main`
decorator. This CLI‑only architecture caused significant UX friction in notebook‑centric
environments such as Databricks, where technical users primarily interact through Jupyter
Notebooks rather than shell workflows.

The project also spans multiple distinct lifecycles—such as one‑off data ingestion
and preparation versus iterative model training. Forcing all operations through
a single CLI entry point and a monolithic configuration structure proved inefficient
and cumbersome, particularly when users needed to selectively execute or iterate on
individual pipeline stages.

To address these limitations, the project was restructured to support programmatic
orchestration of core pipelines directly from notebooks, while retaining the CLI
for advanced batch and orchestration workflows. This ADR documents the completed 
architectural transition and the first migrated lifecycle phase: data ingestion,
data preparation, and standard model training.

---

## Decision

We have decoupled core execution logic from the CLI by introducing a programmatic
API adapter. This enables the `data-ingest`, `data-prepare`, and `model-train`
pipelines to be executed directly from notebook cells without invoking the CLI.

The following changes have been implemented:

- **Programmatic Adapter**
  A new module, `src/landseg/adapters/api.py`, provides a notebook‑safe API that
  programmatically composes Hydra configurations (`DictConfig`) without relying
  on `@hydra.main`.

- **Inspectable and Mutable Configuration**
  The API exposes `get_default_config(pipeline)` for loading base configurations
  by pipeline and supports interactive modification using standard Python data
  structures and OmegaConf semantics. Custom configuration dictionaries are merged
  explicitly prior to execution.

- **Unified Execution Entry Point**
  The API provides a `run(cfg)` function that routes the resolved configuration
  into a shared execution layer. This mirrors CLI execution semantics while avoiding
  interpreter termination and global state side effects.

- **Execution / CLI Separation**
  All pipeline execution logic has been moved into a dedicated `landseg.execution`
  namespace. The CLI now acts as a thin adapter that resolves Hydra configuration
  and delegates execution to the shared executor, ensuring full parity between CLI
  and API execution paths.

- **Notebook Phasing**
  Two template notebooks have been introduced and validated:
  - `01_data_preparation.ipynb`, scoped strictly to `data-ingest` and `data-prepare`
  - `02_model_training.ipynb`, scoped strictly to `model-train`

- **Absolute I/O Paths**
  All pipelines now rely on explicit absolute paths defined in configuration. No
  execution logic depends on the process or notebook working directory, ensuring
  safe operation in notebook environments.

---

## Consequences

### Positive

- **Notebook‑First UX**
  Users can install the package and run preprocessing and training pipelines
  directly from notebook cells without CLI indirection.

- **Clear Separation of Concerns**
  Data preparation and model training are isolated into distinct execution
  contexts, reducing configuration complexity and cognitive load.

- **Shared Execution Semantics**
  CLI and programmatic execution paths share identical resolution, validation,
  and pipeline logic, eliminating drift between interfaces.

- **Validation Retained**
  All dataclass‑based validation and type safety guarantees are preserved through
  the unified resolver and executor layers.

### Negative

- **Path Strictness**
  Relative paths are no longer supported. Users must explicitly provide absolute
  paths for all filesystem interactions.

- **Deferred Functionality**
  Advanced orchestration workflows (e.g., `study-sweep`, `study-analysis`) remain
  CLI‑bound and are intentionally excluded from the notebook API. These are deferred
  to a future ADR if notebook support becomes necessary.

---

## Outcome

This architectural transition has been fully implemented and validated in code.
The programmatic API, refactored CLI, unified execution layer, and notebook
workflows are in active use and behave as designed. Accordingly, this ADR is
now **Accepted**.