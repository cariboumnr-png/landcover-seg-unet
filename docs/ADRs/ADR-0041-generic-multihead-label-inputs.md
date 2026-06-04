# ADR-0041: Generic Multi-Head Mechanism and Optional Hierarchies

**Status:** Accepted
**Date:** 2026-06-04

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

We have adopted a generic multi-head mechanism that supports both independent
and hierarchical tasks. However, to ensure numerical efficiency and data safety,
we utilize a dense-tensor approach rather than a sparse dictionary approach
for raw data storage and loading.

### 2.1. Flexible Label Ingestion
The data ingestion pipeline (`geopipe`) prioritizes composite rasters:
* **Single-band raster:** Defaults to standard single-head training.
* **Multi-band raster:** Supports multiple targets within a single file.

**Note:** We explicitly do not support a list of individual label rasters. Users
must provide a composite raster prepared outside the project to ensure all
labels are geographically aligned before ingestion.

### 2.2. Explicit Head Configuration
Hierarchies are now defined during the ingestion phase through reclassification
policies rather than arbitrary band mappings.
* **Reclass-Driven Hierarchy:** Parent-child relationships are built by
reclassifying specific bands into coarse and fine-grained targets.
* **Restriction:** Hierarchy is not supported between arbitrary bands of a
composite raster. This ensures that the pipeline can safely manage
hierarchy-induced masking (e.g., ignore indices) automatically.
* **Independent Heads:** Any head not participating in a reclass hierarchy is
treated as a "Factorial" or independent task.

### 2.3. Slicing-Based Artifact Generation
Instead of transitioning to a dictionary of independent arrays in `.npz` files
at this time, we have maintained the **monolithic label stack**.
* The `geopipe` module packs all labels into a single `label_stack` tensor
within the `.npz` block.
* The `BatchEngine` runtime will continue to receive a 4D tensor `[B, S, H, W]`
and use channel-slicing to route data to specific heads.
* **Rationale:** Since landscape data is currently expected to be dense
(consistent attributes per pixel), slicing is more memory-efficient and
simpler to maintain than a sparse dictionary of tensors.

### 2.4. Loss and Metric Orchestration
The internal execution context utilizes a dictionary-based *view*
(`y_dict`) generated dynamically after slicing. The `CompositeLoss` and metric suites
iterate over active heads independently, allowing the pipeline to handle
hierarchical masking safely.

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
* **Pipeline Safety:** Restricted hierarchy definitions prevent invalid masking
configurations and ensure deterministic training behavior.

### Negative
* **Static Slicing:** Adding new independent tasks requires re-ingesting the
composite raster to rebuild the `label_stack`.

## 4. Future Work

While the current implementation prioritizes performance for dense landscape data,
several extensions are identified for future iterations:

### 4.1. Sparse Dictionary-Based Targets
As the framework expands to support heterogeneous tasks (e.g., combining
pixel-level segmentation with patch-level classification), we may transition
from monolithic channel-slicing to named array keys in `.npz` files. This would
allow the `DataLoader` to yield a dictionary of tensors, supporting sparse
attributes and variable spatial dimensions.

### 4.2. Runtime Label Subsetting
Enhancing the `Dataset` and `BatchEngine` to subset label channels dynamically
at runtime based on `active_heads` (rather than relying on the `transform`
layer to filter channels) would improve experiment agility without requiring
re-materialization of artifacts.
