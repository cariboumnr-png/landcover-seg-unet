# ADR 0037: Harden Model Contracts for Transformer Integration

**Date:** 2026-05-17
**Status:** Proposed

## 1. Context

The `landcover-seg-unet` framework is structurally designed around strict boundaries separating deterministic data preparation (Foundation/Artifacts), experiment definition (`DataSpecs`), and execution (Session). Currently, the architecture supports classical U-Net and U-Net++ models.

To capture broad geographical context and multi-modal feature relationships (Landsat spectral bands + DEM topography), there is a strategic requirement to evolve the architecture to support Transformer-based models (e.g., U-Net with attention bottlenecks, ResNet-ViT Hybrids, and eventually hierarchical Vision Transformers like SegFormer or Swin-UNet).

However, the current `DataSpecs` and `MultiheadModelLike` contracts contain flexibilities that are safe for Convolutional Neural Networks (CNNs) but fatal for Vision Transformers:
1. **Loose Forward Signatures:** The `MultiheadModelLike` protocol currently utilizes `**kwargs` in its `forward` pass. Transformers often require explicit token routing (e.g., domain conditioning), which risks breaking the execution engine if passed implicitly.
2. **Unconstrained Spatial Dimensions:** CNNs can handle arbitrary spatial padding, whereas Transformers require image dimensions to be strictly divisible by their patch size (typically 16 or 32).
3. **Implicit Multi-Modal State:** The framework does not explicitly guarantee whether spectral and topographic channels are fused before passing to the model backbone, a critical requirement for configuring external ViT embedding layers.

To maintain the architectural guarantee that the execution runner remains completely agnostic to the underlying model architecture, these contracts must be fortified before any external dependencies (e.g., `timm` or `transformers`) are introduced.

## 2. Decision

We will explicitly harden the boundary between the `Execution` layer and the `Models` layer by fortifying the `DataSpecs` and `MultiheadModelLike` contracts. 

Furthermore, we formally adopt the **Adapter Pattern** for all non-native architectures. The framework will allow native experimentation (e.g., custom PyTorch attention bottlenecks in `unet.py`), but all external, mature open-source architectures must be shielded behind a native factory adapter.

### 2.1. Eradicating `**kwargs` in the Model Protocol
The `MultiheadModelLike` protocol will be updated to explicitly define domain conditioning parameters in the forward pass, removing `**kwargs` entirely.

### 2.2. Enforcing Spatial Divisibility in DataSpecs
The `DataSpecs.Meta.Image` dataclass will be updated with a `__post_init__` validation hook. If a pipeline generates a spatial grid configuration where `height_width` is not divisible by 32, the `DataSpecs` instantiation will fail violently at build-time, preventing the ingestion of transformer-incompatible blocks.

### 2.3. Formalizing Fusion Strategy
The `DataSpecs.Meta.Image` dataclass will explicitly declare the multi-modal fusion state (e.g., `is_early_fused: bool = True`) and expose a standardized `tensor_shape` property `(C, H, W)`. This provides model adapters with the exact mathematical dimensions required to rewrite external embedding layers (e.g., modifying a `timm` ResNet-50 stem to accept 10 channels instead of 3 RGB channels).

### 2.4. Build-Time Factory Validation
A strict validation gate will be added to `src/landseg/models/factory.py`. When the session builder requests a model, the factory will generate a dummy tensor based strictly on the `DataSpecs.tensor_shape`, run a zero-gradient forward pass, and assert that the output keys and spatial dimensions perfectly match the `DataSpecs.Heads` topology. 

## 3. Consequences

### Positive
* **Execution Safety:** The session runner is completely protected from architectural quirks. It no longer has to guess what to pass to the model or handle dimension mismatches during the training loop.
* **Rapid Experimentation:** Researchers can safely inject complex external models (like `timm` Hybrids) via the Adapter pattern. The factory validation gate ensures that if an adapter is written incorrectly, it fails instantly before any data loading occurs.
* **XAI Readiness:** Standardizing the forward pass and isolating external models behind adapters provides a stable hook for future Explainable AI callbacks (e.g., Attention Rollout, Grad-CAM) required for reporting on landcover classification reasoning.

### Negative
* **Loss of Spatial Flexibility:** The strict `height_width % 32 == 0` constraint forces the `geopipe.foundation.world_grids` configurations to be slightly less flexible. Arbitrary block sizes (e.g., 500x500) are no longer permitted.
* **Slight Factory Overhead:** Instantiating a dummy tensor and running a dry forward pass in the factory adds a negligible (fraction of a second) overhead to session construction.
