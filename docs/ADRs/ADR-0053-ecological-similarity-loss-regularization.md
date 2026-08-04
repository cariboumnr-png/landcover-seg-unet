# ADR-0053: Ecological Similarity Loss Regularization for Tree Species Segmentation

- **Status:** Proposed
- **Date:** 2026-08-03

## Context

The `landseg` training pipeline currently configures loss via `CompositeLoss`,
composing primitives such as Focal Loss, Dice Loss, Spectral Smoothness, and
Total Variation (`session/engine/runtime/tasks/loss/primitives/`). While
these primitives effectively address class imbalance, region overlap, and
spatial smoothness, they treat all species class targets as orthogonal
one-hot vectors.

Under standard discrete one-hot target encodings, loss functions assume that
all incorrect classes are uniformly equidistant from the ground truth. This
fails to account for continuous domain semantics—such as taxonomic kinship,
functional growth forms, and shared ecological niches—treating minor
within-guild misclassifications as severely as major cross-guild or
cross-climatic errors.

For example, misclassifying a Boreal Lowland Black Spruce (`SB`) as an
ecologically similar Tamarack (`LA`) incurs the exact same cross-entropy or
focal penalty as misclassifying it as an ecologically incongruous Carolinian
Sugar Maple (`MH`).

To inject domain knowledge without architectural risk or inference latency,
we will leverage a repository knowledge base (`knowledge/`) containing
138 FRI species profiles and 29 ecological groups encoded via Sentence
Transformers into embedding tensors and an $N \times N$ cosine similarity
matrix (`species_similarity_matrix.pt`).

## Decision

We will introduce an **Ecological Similarity Loss Regularizer** as Phase 1
of our domain knowledge integration strategy.

### Key Aspects of the Proposed Design

1. **Non-Invasive Model Architecture:**
   - The UNet model structure and output layer will remain completely
     unchanged.
   - Inference execution and ONNX/PyTorch deployment speed will incur zero
     overhead.

2. **Similarity Matrix Supervision:**
   - We will load `species_similarity_matrix.pt` ($S \in \mathbb{R}^{N \times N}$) during training.
   - We will compute a domain-aware distance penalty:

     $$
     \mathcal{L}_{\text{eco}} = \sum_{c=1}^N p_c \cdot (1 - S_{y, c})
     $$

     where $p_c$ is the predicted softmax probability for class $c$ and $y$ is the ground truth target class.

3. **Loss Composite Integration:**
   - We will implement `EcologicalSimilarityLoss` as a primitive loss
     subclass under `session/components/task/loss/primitives/`.
   - The loss will be composable via `CompositeLoss` with a configurable
     weight $\lambda$.

## Implementation Plan

1. Create `EcologicalSimilarityLoss` primitive loss module in
   `session/components/task/loss/primitives/`.
2. Add Hydra configuration schema and validation for
   `ecological_similarity_weight`.
3. Register the loss in loss factory builders.
4. Execute overfit and validation benchmarks against baseline
   Cross-Entropy.

## Consequences

### Positive

- Zero added model parameters or inference time latency.
- Softens target distributions and penalizes ecologically absurd
  misclassifications.
- Low-risk, highly modular, and backwards-compatible addition to the training
  stack.

### Negative

- Requires loading the precomputed similarity matrix tensor into GPU memory
  during training (~3 KB).
- Introduces one hyperparameter ($\lambda$) to tune during loss composition.

## Future Phases

While this ADR focuses on Phase 1 (non-invasive loss regularization), future
ADRs will evaluate and propose subsequent phases of ecological domain
knowledge integration:

- **Phase 2 (Topographic Context Conditioning):** Introduce a
  multi-scale spatial neighborhood encoder over DEM, DSM, TPI, and TWI
  channels to condition UNet feature maps via Feature-wise Linear
  Modulation (FiLM).

- **Phase 3 (Shared Text-Visual Embedding Head):** Project UNet
  output features into the Sentence Transformer embedding space to enable
  dot-product zero-shot classification against text vector prompts.
