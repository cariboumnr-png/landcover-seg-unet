# ADR-0017: Spectral and Spatial Regularization Losses

- **Status:** Accepted (Implemented)
- **Date:** 2026-04-10
- **Supersedes:** ADR-0017 (Proposed, 2026-04-07)

## Context

Training previously relied on focal and Dice losses, which address class
imbalance and region overlap but do not explicitly encode expected
spectral–spatial structure present in multispectral and hyperspectral
remote-sensing imagery.

Empirically, this manifested as:
- Salt-and-pepper artifacts in predictions
- Reduced class consistency in spectrally homogeneous regions

## Decision

Two optional regularization losses have been introduced and integrated
into the training stack:

### 1. Spectral Smoothness Loss

- Penalizes disagreement between neighboring pixels with similar input
  spectra
- Uses fixed similarity weights computed from input features
- Does not backpropagate gradients through the input image
- Implemented as `SpectralSmoothnessLoss`, a `PrimitiveLoss` subclass

### 2. Total Variation (TV) Loss

- Penalizes abrupt spatial variation in predicted class probabilities
- Acts as a weak spatial prior to reduce noise
- Independent of spectral content
- Implemented as `TotalVariationLoss`, a `PrimitiveLoss` subclass

Both losses:
- Live under `session/components/task/loss/primitives/`
- Are composed via the existing `CompositeLoss`
- Are fully optional and controlled by configuration weights
- Can be disabled by setting weights to zero

## Implementation Notes

- Loss primitives were refactored into an explicit `primitives` namespace
- Existing Focal and Dice losses were migrated to the same architecture
- Trainer, factory, and config schemas were updated accordingly
- Overfit experiments confirm:
  - Base losses still overfit cleanly
  - Moderate spectral/TV weights preserve convergence
  - Excessive weights correctly prevent overfitting, as expected

## Consequences

### Positive

- Explicit spectral–spatial regularization without architectural changes
- Configuration-driven and backwards-compatible
- Improved qualitative smoothness in predictions
- Clean extensibility for future regularizers

### Negative

- Slight increase in training-time computation
- Additional hyperparameters requiring tuning

## Outcome

The decision has been fully implemented, validated in overfit tests, and
integrated into the training pipeline without disruption. This ADR is
considered complete and closed.