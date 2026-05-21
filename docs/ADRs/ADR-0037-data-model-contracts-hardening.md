# ADR 0037: Harden Model Contracts for Transformer Integration

**Date:** 2026-05-21
**Status:** Accepted and Implemented

## 1. Context

The `landcover-seg-unet` framework is structurally designed around strict boundaries separating deterministic data preparation (Foundation/Artifacts), experiment definition (`DataSpecs`), and execution (Session). Currently, the architecture supports classical U-Net and U-Net++ models.

To capture broad geographical context and multi-modal feature relationships (Landsat spectral bands + DEM topography), we are evolving the architecture to support Transformer-based models (e.g., U-Net with attention bottlenecks, ResNet-ViT Hybrids, and eventually hierarchical Vision Transformers like SegFormer or Swin-UNet).

However, the previous `DataSpecs` and `MultiheadModelLike` contracts contained flexibilities that were safe for Convolutional Neural Networks (CNNs) but fatal for Vision Transformers:
1. **Loose Forward Signatures:** The `MultiheadModelLike` protocol utilized `**kwargs` in its `forward` pass, risking execution engine failures when passing explicit token routing required by Transformers.
2. **Coupled Domain Routing:** Domain conditioning logic was tightly embedded within the specific `MultiHeadUNet` frame, making it difficult for new architectures to reuse the routing mechanisms.
3. **Unconstrained Spatial Dimensions:** CNNs can handle arbitrary spatial padding, whereas Transformers require image dimensions to be strictly divisible by their patch size (typically 16 or 32).
4. **Implicit Multi-Modal State:** The framework lacked clear boundaries for how spectral and topographic channels were parsed for external architectures.

To maintain the architectural guarantee that the execution runner remains completely agnostic to the underlying model architecture, these contracts were fortified.

## 2. Decision

We have explicitly hardened the boundary between the `Execution` layer and the `Models` layer by fortifying the model protocols, introducing architecture-agnostic domain routing, and implementing fail-fast validations at the factory level.

### 2.1. Eradicating `**kwargs` and Architecture-Agnostic Domain Routing
We removed `**kwargs` from the `MultiheadModelLike` and `MultiHeadBaseModel` protocols. The forward pass now requires explicit domain conditioning parameters: `ids_domain` and `vec_domain`. 

Crucially, the domain routing logic—previously embedded directly within the `MultiHeadUNet` frame—has been overhauled and extracted into an architecture-agnostic `model_core.DomainContextRouter`. This independent router parses the explicit domain tensors into a strictly typed `dict[str, DomainTargetPayload]`. Future Transformer-specific conditioners (e.g., token embedding projectors) can now seamlessly plug into this structured payload without requiring any modifications to the core routing logic or the execution engine.

### 2.2. Delegating Spatial Divisibility to the Backbone
Instead of forcing a rigid patch-size constraint at the data preparation layer, we updated the `Backbone` protocol to enforce a new `@property def spatial_divisor(self) -> int`. This allows the specific machine learning architecture (e.g., Swin vs. ResNet) to declare its own spatial constraints, preserving flexibility in the `geopipe` foundations.

### 2.3. Preserving Multi-Modal Schemas
Rather than formalizing a rigid `is_early_fused` boolean in `DataSpecs`, we determined that `geopipe`'s existing schema—which explicitly maps bands to spectral and topographical groups—already provides sufficient metadata for future Vision Transformer adapters to configure token embeddings dynamically.

### 2.4. Build-Time Factory Validation
We implemented a strict "fail-fast" validation gate in `src/landseg/models/factory.py`. When the session builder requests a model, the factory performs a dry run:
1. It asserts `dataspecs.meta.image_specs.height_width % model.spatial_divisor == 0`.
2. It generates a dummy tensor based on the dataset schema.
3. It runs a zero-gradient forward pass.
4. It asserts that the output keys and spatial topologies perfectly match the `DataSpecs.Heads` topology.

### 2.5. Separation of Safety and Head Management
To further stabilize the model interface, orchestrating logic was extracted into dedicated `NumericSafety` and `HeadManager` components. Per-head logit adjustments are now cleanly registered as non-trainable buffers via a dedicated `_register_logit_adjust` method, ensuring deterministic behavior across devices.

## 3. Alternatives Considered

**Enforcing Spatial Divisibility in DataSpecs:** Initially, it was proposed to add a `__post_init__` hook in `DataSpecs` that violently failed if block sizes were not divisible by 32. This was rejected because 32 is a "magic number" specific to certain CNN stems. Enforcing this at the data pipeline layer felt unnatural and overly restrictive for geospatial research, where dynamic block sizes might be necessary. Moving the constraint to the model layer (via `spatial_divisor`) proved much more elegant.

## 4. Consequences

### Positive
* **Execution Safety:** The session runner is completely protected from architectural quirks. It no longer relies on implicit arguments or handles dimension mismatches during the training loop.
* **Plug-and-Play Transformers:** The extraction of the `DomainContextRouter` means any new external model (like a `timm` Hybrid) can be wrapped in a native adapter and immediately inherit all domain conditioning capabilities.
* **Fail-Fast Predictability:** The factory validation gate ensures that if an adapter is written incorrectly or a patch size is incompatible, it crashes instantly during session initialization before allocating GPU memory or spinning up data loaders.
* **XAI Readiness:** Standardizing the forward pass provides a stable hook for future Explainable AI callbacks.

### Negative
* **Slight Factory Overhead:** Instantiating a dummy tensor and running a dry forward pass adds a negligible overhead (fraction of a second) to session construction.
