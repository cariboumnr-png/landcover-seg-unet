## ADR-0029 Transition to Programmatic API for Data & Training Notebook Execution

**Status:** Proposed  
**Date:** 2026-04-30  

-----

### Context
Currently, the `landcover-seg-unet` pipeline (spanning data ingestion, preparation, training, and sweeping) relies exclusively on a CLI driven by Hydra's `@hydra.main` decorator. This CLI-only architecture creates UX friction for target deployment environments like Databricks, where technical users operate primarily within Jupyter Notebooks.

Because the project contains distinct lifecycles—such as one-off data preparation versus iterative model training—forcing all operations through a single CLI entry point or a monolithic configuration file is inefficient. We need a way to orchestrate these distinct pipelines programmatically from isolated notebooks.

To manage the complexity of this transition, this ADR focuses strictly on establishing the foundational adapter and migrating the first phase of the project lifecycle: data ingestion, data preparation, and a standard model training loop.

### Decision
We will decouple the core execution logic from the CLI by introducing a programmatic API adapter, enabling the `data-ingest`, `data-prepare`, and `model-train` pipelines to be executed entirely from notebook cells.

1. **Programmatic Adapter:** We will create `src/landseg/api.py` utilizing Hydra's `compose` API to programmatically construct the configuration tree (`DictConfig`) without CLI invocation.
2. **Inspect and Modify Workflow:** The API will expose a `get_default_config(pipeline)` function, allowing users to load base configurations for specific pipelines (e.g., `data-prepare` vs `model-train`) and modify parameters using standard Python dot-notation.
3. **Execution Entry Point:** The API will provide a `run(cfg)` function that routes the composed configuration to the appropriate pipeline logic, bypassing `@hydra.main`.
4. **Notebook Phasing:** We will introduce the first two template notebooks:
    * `01_data_preparation.ipynb`: Scoped strictly to `data-ingest` and `data-prepare`.
    * `02_model_training.ipynb`: Scoped strictly to `model-train`.
5. **Absolute I/O Paths:** Because the notebook abstracts the Current Working Directory, all I/O operations (reading raw imagery, saving preprocessed patches, and writing model artifacts) must utilize explicit absolute paths defined in the config.

### Consequences

**Positive:**
* **Frictionless Deployment:** Users can `pip install .` and execute data preprocessing and training loops directly via Python cells.
* **Separation of Concerns:** By isolating data preparation from training into separate notebooks, users avoid loading unnecessary configurations.
* **Validation Retention:** We retain 100% of the type safety and validation provided by our existing nested dataclasses.

**Negative:**
* **Path Strictness:** Users can no longer rely on relative paths; absolute paths must be enforced.
* **Deferred Functionality:** Complex orchestration (e.g., `study-sweep` and `study-analyze`) is explicitly deferred to a future ADR, meaning hyperparameter sweeping remains CLI-bound for this iteration.
