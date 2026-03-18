# ADR-0012: Dataset Ingestion Pipeline decomposition
- **Status:** Proposed
- **Date:** 2026-03-13

## Context
In the current pipeline `landseg.ingest_dataset`, normalization is applied before dataset splitting using statistics from all fit blocks. This limits flexibility and introduces potential data leakage. Raw blocks also contain both unnormalized and normalized images, mixing permanent and experiment‑dependent content.

We intend to support multiple grids on the same rasters. Raw blocks should therefore be immutable, reusable across experiments, and independent of downstream normalization strategies.

## Decision
We adopt a **four‑module pipeline**:

## Module — `align`

**Purpose:** align input image/label rasters to the grid.

### Outputs

a `DataWindows` instance

### Note

* simple migration of the old `ingest_dataset.mapper` module.

---

## Module — `catalogue`

**Purpose:** produce immutable raw data blocks.

### Outputs

Raw `.npz` files containing:

* raw image channels (with spectral/topo features as desired)
* label / hierarchical label arrays
* valid mask
* metadata including per-block stats
* **no normalized image array**

Additional outputs:

* `catalogue.json` : block name → raw `.npz` path
* `catalogue_meta.json` : grid signature, rasters used, creation timestamp, version, etc.

### Requirements

* Catalogue is **read-only after production**.
* Multiple grids can coexist; if grids share origin + shape logic, filename collision is acceptable and the builder will skip existing blocks.
* Catalogue **never contains experiment-dependent features such as normalization**.

---

## Module — `split`

**Purpose:** consume raw catalogue and produce dataset splits.

### Responsibilities

* Read raw catalogue metadata.
* Select a subset of blocks:

  * filter by valid pixel ratio
  * scoring
  * geographic sampling
* Produce `${train, val}.json` manifests pointing to raw blocks.
* Compute any label distributions or per-split metrics needed for scoring.

### Important Rules

* **No normalization here.**
* Splits can be recomputed arbitrarily without modifying the catalogue.

---

## Module — `standardize`

**Purpose:** generate model-ready data tailored to the chosen split, including normalization.

### Responsibilities

Compute per-channel mean/std on **train blocks only**.

* Train-only statistics prevent leakage.
* Statistics are stored in `extract_stats.json`.

Normalize image arrays for train/val/test using train statistics and write:

New `.npz` files containing:

* `image_normalized`
* `label_masked`
* `meta` (shallow copy with updated extractor version)

Files are stored under a **new directory versioned by extractor**.

Build the **final schema** based solely on these extracted `.npz` files.

### Advantages

* Every experiment can choose a fresh training subset, run extract, and get fully leakage-free normalized blocks.
* Raw catalogue remains untouched and reusable.
* Extractor can change normalization techniques (e.g., robust z-score, histogram matching) without affecting catalogue stability.

## Rationale
- Prevents data leakage by computing normalization stats strictly from training split.
- Enables multiple downstream experiments to reuse the same raw catalogue.
- Maintains immutability and versioning guarantees of raw data.
- Extraction becomes experiment‑dependent and repeatable.

## Alternatives Considered
### Option A — Keep normalization in catalogue (rejected)
- Normalization would be tied to a specific “fit” subset. Any change in train selection forces catalogue rebuild. Violates reuse goal.
### Option B — Fold extraction into split module (rejected)
- Splitting and extraction have different lifecycles. Extraction depends on split, and splits may be recomputed many times. Keeping extraction separate avoids churn.

## Consequences
- Normalized arrays removed from raw blocks.
- Catalogue is stable and grid‑dependent only.
- Splits and extraction are reproducible and decoupled.
- Final schema reflects only extracted, normalized blocks.

## Work Plan
- Refactor existing blockbuilder into `catalogue`.
- Implement `split` for all scoring/splitting logic.
- Add `standardize` for normalization + final dataset emission.
- Update pipeline naming, ordering, and documentation.

## Note
- To avoid bloating at root, these modules are to be sub-modules under `landseg.ingest_dataset/` and accessed by `landseg.ingenst_dataset/pipeline.py`.
- Expected artifacts output structure:
```
./experiment/artifacts/
├── <dataset_name>/
    ├── fit/
    |   ├── grid_row_256_col_256
    |   |   ├── blocks/
    |   |   ├── windows/
    |   |   |   ├── windows_<gid>.pkl
    |   |   |   ├── windows_<gid>.pkl
    |   |   |   ...
    |   |   ├── metadata.json
    |   |   ├── catalog.json
    |   |   ...
    |   └── grid_row_512_col_512
    └── test
        ├── grid_row_256_col_256
        |   ├── blocks/
        |   ├── metadata.json
        |   ├── catalog.json
        |   ...
        └── grid_row_512_col_512
```