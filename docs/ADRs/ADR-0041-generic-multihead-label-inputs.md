# ADR-0041: Generic Multi-Head Mechanism and Optional Hierarchies

**Status:** Proposed
**Date:** 2026-06-02

## 1. Context

Currently, our multi-head segmentation architecture assumes a strict hierarchy
between prediction heads (e.g., a coarse taxonomy branching into a fine-grained
taxonomy), derived from a single label raster. However, natural landscapes are
often factorial rather than purely nested. For example, assigning a pixel as
"Forest/Non-Forest", defining its "Age", and predicting its "Leading Species"
(e.g., Pine vs. Spruce) creates overlapping, independent attributes. Forcing a
strict hierarchy results in a combinatorial explosion of target classes, creates
artificially rare edge-cases, and handles missing attributes poorly.

While the core `MultiHeadUNet` and `HeadManager` are already agnostic to head
relationships, the data preparation pipelines (`geopipe`), configuration schemas,
and loss orchestration are implicitly hardcoded to expect a single target array
and a unified hierarchical tree.

To support robust pixel-level Multi-Task Learning (MTL) where implicitly related
tasks can mutually inform a shared latent representation without strict nesting,
we need a generic multi-head data ingestion and training mechanism.

## 2. Decision

We are implementing a generic multi-head mechanism where hierarchies are strictly
optional. The configuration, artifact generation, and runtime layers will be
updated to treat heads as arbitrary, independent tasks by default, with hierarchy
applied only when explicitly configured.

### 2.1. Flexible Label Ingestion
The data ingestion pipeline will accept labels in varying formats:
* **Single-band raster:** Defaults to standard single-head training.
* **Multi-band raster:** Each band can be mapped to a distinct prediction head.
* **List of independent rasters:** Multiple separate rasters can be ingested
simultaneously, each mapping to one or more heads.

### 2.2. Explicit Head Configuration
The ingestion and dataset configuration schemas will explicitly define the mapping
between the input raster bands and the model heads.
* Hierarchies are no longer inferred. They must be explicitly declared via
`head_parent` mapping.
* A head can declare `parent: None`, making it entirely independent.
* Hierarchy can be established between different input rasters or between bands
within the same raster.

### 2.3. Multi-Target `.npz` Artifact Generation
When preparing the dataset, the `geopipe` module will pack multiple label arrays
into the individual `.npz` data block files.
* The `DataSpecs` schema will be updated so `Meta.Label` transitions from a single
`array_key: str` to a mapping `array_keys: dict[str, str]`.
* The data loader will yield a dictionary of target tensors (`dict[str, torch.Tensor]`)
matching the output dictionary of logits produced by the model.

### 2.4. Loss and Metric Orchestration
The training runtime will be updated to handle dictionary targets. The `CompositeLoss`
and metric suites (e.g., `conf_matrix`) will iterate over the active heads, computing
losses and logging metrics independently per head, while utilizing the `ignore_index`
to seamlessly mask out missing labels for specific tasks on a per-pixel basis.

## 3. Consequences

### Positive
* **Combinatorial Efficiency:** Reduces the parameter count in the final convolution
layers by preventing the Cartesian product of all possible factorial states.
* **Robustness to Missing Data:** If a polygon contains species data but is missing
age data, the independent heads can apply the `ignore_index` selectively, allowing
the model to learn from the available data without corrupting the entire pixel label.
* **Inductive Transfer:** Training independent heads on the same backbone forces
the network to learn richer shared feature representations (e.g., texture features
useful for age prediction inherently improve species classification).

### Negative
* **Breaking Schema Changes:** Existing `data-ingest` and `data-prepare` YAML
configurations will need to be migrated to explicitly declare the `array_keys`
and `head_parent` structures.

## 4. Migration Notes

* Update `src/landseg/core/data_specs.py` to support `dict[str, str]` for label
array keys.
* Refactor the data block builder in `geopipe` to stack/save multiple target
arrays during the partitioning phase.
* Update the training executor loop to pass `dict[str, torch.Tensor]` to the
loss and metric functions.
