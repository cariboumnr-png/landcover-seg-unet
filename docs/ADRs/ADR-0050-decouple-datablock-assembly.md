# ADR-0050: Decouple Data Block Assembly and Refactor Ingest Pipelines

**Status:** Accepted
**Date:** 2026-07-08

## 1. Context

Previously, the `builder.py` module in `foundation/data_blocks/` was responsible
for orchestrating file reads, calculating spatial and topographic attributes, and
serializing block outputs in a monolithic fashion.

This coupling presented several architectural drawbacks:
* **Difficult Testability**: The core assembly logic could not easily be tested
in isolation without setting up full filesystems or mocking nested I/O layers.
* **Confusing Terminology**: The term `meta` was overloaded to refer both to the
spatial block database (`catalog.json`) and to runtime configurations, leading to
namespace pollution and logical confusion.
* **Tight Coupling**: File paths, tile window scopes, and hyperparameter configs
were mixed inside builder parameters, making pipeline integrations rigid.

## 2. Decision

We have decoupled block construction by introducing a dedicated `assembler` submodule
and refactoring the downstream pipelines to align with clean, isolated boundary contracts.

### 2.1. Introduction of `assembler` Submodule
The monolithic `builder.py` is retired and replaced by a structured `assembler`
package containing:
* `assembler.py`: Core orchestrator and assembly logic.
* `lifecycle.py`: Lifecycle controls for test/dev block builds.
* `io.py`: Standardized filesystem checks and block serialization helper functions.

### 2.2. Isolated Building Contracts
We decoupled block construction inputs into three distinct types:
* **`BlockBuildingInput`**: Filesystem path parameters containing input rasters,
config files, and the output path.
* **`BlockBuildingContext`**: Spatial window maps aligned to the grid.
* **`BlockBuildingConfig`**: Parameters adjusting band layouts, target heads,
and padding.

### 2.3. Rename of `meta` to `manifest`
To eliminate namespace confusion with configuration schemas and downstream datasets:
* All references to `meta` inside `DataBlock` are renamed to `manifest` (representing
the block's serialization manifest).
* The associated properties, classes, and serialization schemas are updated to
reflect the `manifest` taxonomy.

### 2.4. Downstream Pipeline Migration
The data ingestion pipeline (`data_ingest.py`) and the overfit diagnostics pipeline
(`diagnose_overfit.py`) have been updated to utilize the new decoupled `assembler` API.

## 3. Consequences

* **Testability**: The isolated building contracts enabled clean unit tests for
the assembler (implemented in `test_assembler.py`) without nested side effects.
* **Clarity**: Logical boundaries between what is input (filepaths), what is
context (tile arrays), and what is config (parameters) are clearly defined.
* **Ergonomics**: Decoupling the block assembly interfaces makes future transformations
or feature additions easier to integrate.
