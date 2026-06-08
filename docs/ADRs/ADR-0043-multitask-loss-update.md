# ADR-0043: Multi-Task Loss Balancing and Logical Consistency Regularization

**Status:** Proposed
**Date:** 2026-06-8

## 1. Context

Following **ADR-0041** (Multi-head Architecture) and **ADR-0042** (Extended Metrics),
we currently calculate loss as a simple arithmetic mean of all head losses.
Each head currently computes a composite loss:

$L_{head} = L_{focal} + L_{dice} + L_{spectral} + L_{tv}$.

While functional, this approach has two primary weaknesses:
1. **Gradient Imbalance:** Different tasks (e.g., Landcover vs. Canopy Height)
may have different loss scales and convergence rates. A simple average allows
tasks with larger loss magnitudes to dominate the shared backbone updates.
2. **Logical Divergence:** There is no explicit penalty during backpropagation
 when Head A and Head B produce a combination of classes that is physically or
 ecologically impossible.

## 2. Decision

We will implement a structured MTL Loss Engine that supports weighted aggregation
and a dedicated "Logical Consistency" loss component.

### 2.1. Weighted Task Aggregation
Instead of a simple average, the total loss will be defined as:

$$L_{total} = \sum_{i \in Heads} (w_i \cdot L_i) + \lambda \cdot L_{consistency}$$

*   **Static Weighting:** Initial implementation will allow users to define $w_i$
in the configuration to manually balance tasks.
*   **Observability:** Each individual component (Focal, Dice, Spectral, TV) per
head, as well as the weighted sub-total per head, must be logged to the experiment
tracker (e.g., MLFlow/WandB).

### 2.2. Logical Consistency Loss ($L_{consistency}$)
We will introduce a differentiable penalty for violating domain constraints
defined in the configuration.
*   **Mechanism:** Using the "Constraint Matrix" (from ADR-0042), we will calculate
the probability of "invalid states."
*   **Soft Constraints:** Rather than a hard check, we use the Softmax outputs of
the involved heads. If Head A predicts probability $P(A_{water})$ and Head B predicts
$P(B_{trees})$, and the combination is invalid, the loss is $P(A_{water}) \cdot P(B_{trees})$.
Minimizing this product forces the model to reduce the confidence of at least one
of the conflicting predictions.

### 2.3. Task-Wise Regularization (Spectral/TV)
The small regularizers (Spectral, TV) will remain local to each head's loss
calculation to ensure spatial smoothness and spectral integrity are maintained
for each specific output domain.

### 2.4. Dynamic Weighting Support (Future-Proofing)
The architecture will be designed to eventually support **Uncertainty Weighting**
(Kendall et al.), where $w_i$ is a learnable parameter that adjusts based on the
homoscedastic uncertainty of each task, preventing "noisy" tasks from corrupting
the backbone.

## 3. Proposed Implementation Structure

1.  **`MTLLossAggregator`**: A class that takes the collection of head outputs
and targets.
2.  **`ConsistencyLoss` Module**: A specific loss module that takes the logic
map and returns a scalar penalty based on cross-head probability distributions.
3.  **Configuration Schema Update**:
    ```yaml
    training:
      loss_weights:
        landcover: 1.0
        age_class: 0.5
        consistency_weight: 0.2  # Lambda
    ```

## 4. Consequences

### Positive
*   **Task Synergy:** Encourages the model to learn features that benefit all
tasks, not just the one with the highest gradient magnitude.
*   **Physically Plausible Maps:** Drastically reduces "impossible" pixel predictions
(e.g., forest in the middle of a lake) by penalizing them during training, not
just flagging them in metrics.
*   **Better Convergence:** Balancing weights prevents "gradient tug-of-war"
where tasks pull the backbone in opposite directions.

### Negative
*   **Hyperparameter Complexity:** Introduces new weights ($w_i$ and $\lambda$)
that may require tuning.
*   **Memory Overhead:** Calculating the consistency loss requires keeping the
full probability distributions (Softmax) for multiple heads in memory simultaneously before reduction.

## 5. Future Work
*   Implementation of **GradNorm** or **Equal Design Loss** to automate the
discovery of optimal $w_i$ values.
