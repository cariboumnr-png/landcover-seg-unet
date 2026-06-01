# ADR 0040: Introduce Configurable Transformer Bottlenecks for UNet Backbones

**Date:** 2026-06-01
**Status:** Accepted

## 1. Context

ADR 0037 hardened the model boundary for future transformer integration. It
removed loose model forwarding patterns, introduced explicit domain-conditioning
inputs, extracted domain routing into an architecture-agnostic router, delegated
spatial divisibility constraints to the backbone, and added build-time factory
validation.

Those changes prepared the framework for transformer-compatible architectures,
but the implemented segmentation model family still remained centred on classical
UNet-style convolutional structures. The next incremental step is to introduce
transformer capacity into the existing UNet family without replacing the current
multi-head model frame, execution pipeline, domain routing system, or prediction-head
management.

We introduces this capability by making the UNet backbone a composition of two
independent choices:

- the UNet body topology; and
- the bottleneck implementation.

This allows the framework to preserve the proven encoder-decoder and skip
connection behaviour of UNet, UNet++, and UNet3+, while optionally replacing the
deepest convolutional bottleneck with transformer or hybrid transformer modules.

## 2. Decision

We now support configurable bottlenecks for UNet-family backbones.

A UNet backbone is configured as:

    UNetBackboneConfig
    ├── body: UNetBodyConfig
    └── bottleneck: BottleneckConfig

The `body` field selects the UNet structural topology:

- `unet`
- `unetpp`
- `unetppp`

The `bottleneck` field selects the deepest feature-processing module:

- `conv`
- `transformer`
- `hybrid`

The transformer bottleneck is introduced as a drop-in replacement at the deepest
UNet feature level. It preserves spatial dimensions and channel compatibility,
while adding long-range context modelling through self-attention and feed-forward
transformer blocks.

The hybrid bottleneck combines convolutional processing and transformer
processing. It provides a middle-ground option that preserves local convolutional
inductive bias while still introducing global context modelling.

## 3. Implementation Summary

### 3.1. UNet body and component packages

UNet topology classes are organized under:

    /landseg.models.backbones.unet.body

Reusable convolution blocks, encoders, bottlenecks, and related configuration
contracts are organized under:

    /landseg.models.backbones.unet.components

This separates architecture topology from reusable implementation components.

### 3.2. Bottleneck abstraction

A new bottleneck abstraction is introduced for UNet backbones. Concrete
implementations include:

- `UNetBottleneck`
- `TransformerBottleneck`
- `HybridBottleneck`

All bottleneck implementations are expected to preserve the feature-map contract
required by the decoder: the output tensor must remain compatible in spatial
shape and channel count with the UNet body.

### 3.3. Backbone factory

A dedicated `build_unet_backbone(...)` factory constructs the requested UNet
body, validates spatial divisibility, computes bottleneck spatial size, builds
the configured bottleneck, and returns a complete backbone.

This removes backbone selection and bottleneck assembly from the `MultiHeadUNet`
frame.

### 3.4. MultiHeadUNet responsibility boundary

`MultiHeadUNet` remains responsible for coordinating:

- shared feature extraction;
- domain routing;
- concat and FiLM conditioning;
- prediction heads;
- numerical safety; and
- forward behaviour.

It no longer needs to own the registry or construction logic for UNet body
classes.

## 4. Consequences

### 4.1. Positive consequences

- Transformer bottlenecks can be introduced without replacing the UNet model
  family.
- UNet body topology and bottleneck implementation are now independently
  configurable.
- Existing multi-head output handling remains unchanged.
- Existing domain routing and conditioning mechanisms remain reusable.
- Transformer attention is applied at the lowest-resolution UNet feature level,
  reducing attention cost relative to full-resolution attention.
- The implementation aligns with the contract-hardening work completed in
  ADR 0037.

### 4.2. Negative consequences

- Model configuration becomes more complex because body and bottleneck choices
  are now separate.
- Transformer bottlenecks introduce additional memory and runtime cost.
- Attention cost remains sensitive to patch size and bottleneck spatial size.
- The package refactor may require import-path updates for internal or external
  callers that previously imported UNet classes directly from older module
  locations.
- More architecture combinations must be covered by tests.

## 5. Alternatives Considered

### 5.1. Replace UNet with a full transformer segmentation architecture

Rejected for this ADR.

A full transformer segmentation model would require broader changes to token
embedding, positional encoding, multi-modal feature handling, decoder design,
and possibly training defaults. The bottleneck approach gives the framework a
lower-risk transformer integration path while preserving the current UNet-based
segmentation stack.

### 5.2. Insert attention blocks throughout the encoder and decoder

Rejected for now.

Applying attention at multiple resolutions would increase implementation
complexity and memory usage. The bottleneck is the most controlled insertion
point because the feature map has already been spatially downsampled.

### 5.3. Keep bottleneck construction inside `MultiHeadUNet`

Rejected.

The model frame should coordinate routing, conditioning, feature extraction, and
heads. It should not own the construction registry for UNet internals. Moving
body and bottleneck assembly to a backbone factory improves separation of
concerns.

## 6. Decision Outcome

We implemented configurable bottlenecks for UNet-family backbones as the next
incremental step toward transformer-capable segmentation models.

The default model remains conservative and backwards-compatible with a
convolutional bottleneck. Transformer and hybrid bottlenecks are available by
configuration for experiments requiring broader spatial context. The design
keeps transformer complexity localized within the backbone layer and preserves
the execution-layer and model-frame boundaries established by ADR 0037.
