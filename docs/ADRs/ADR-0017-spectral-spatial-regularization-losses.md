# ADR-0017: Introduce Spectral and Spatial Regularization Losses

- **Status:** Proposed
- **Date:** 2026-04-07

## Context
The current training setup relies primarily on focal loss and Dice loss to address class imbalance and region overlap stability.

While effective, we observed that these losses do not explicitly encourage:
- spectral coherence between neighboring pixels, or
- spatial smoothness in predicted segmentation maps.

For multispectral and hyperspectral remote-sensing data, encouraging such structure is desirable and aligns well with known properties of the data.

## Decision
We plan to introduce two new, optional loss components in an incremental and non-disruptive manner:

1. **Spectral Smoothness Loss (L_spectral)**
   - Penalizes differences in predicted class probabilities between neighboring pixels that are spectrally similar.
   - Uses the input image spectra to compute fixed similarity weights (no gradient through inputs).
2. **Total Variation Loss (L_tv)**
   - Penalizes abrupt spatial changes in predicted probability maps.
   - Acts as a weak spatial prior to reduce salt-and-pepper noise.

Both losses will be:
- implemented as `PrimitiveLoss` subclasses,
- placed under `trainer/components/loss/` alongside existing focal and Dice losses,
- optional and controlled entirely via configuration weights.

They will be composed using the existing `CompositeLoss` mechanism, allowing weighted summation with focal and Dice losses.

## Expected Consequences
### Positive
- Explicit encouragement of spectral-spatial coherence in predictions.
- No required changes to model architectures.
- Fully reversible and easy to disable (set weights to zero).
- Per-loss logging and diagnostics via existing trainer mechanisms.

### Negative
- Slight increase in training-time computation.
- Additional hyperparameters (loss weights) requiring tuning.

We expect these losses to improve segmentation smoothness and class consistency, particularly in spectrally homogeneous regions, while preserving the stability of the existing training pipeline.
