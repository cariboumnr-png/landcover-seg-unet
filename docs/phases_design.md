# ML Research Experimentation Phases

This document defines the transition gates for model development and data expansion.

---

## Phase 1: Overfit Silo (The "Unit Test")
**Goal:** Prove the model/data contract is functional.
* **Setup:** Use 5–10 representative samples. Disable all stochasticity (no augmentation, no dropout, no weight decay, no loss weighting).
* **Success Criteria:** Training IoU > 95% within ~100–200 epochs.
* **Failure Implication:** If this fails, there is a bug in the U-Net architecture, label encoding, or data normalization. **Do not proceed to Baseline.**

## Phase 2: Baseline Run (The "Floor")
**Goal:** Establish a performance benchmark using standard hyperparameters.
* **Setup:** Full dataset with default Hydra configurations. Enable standard augmentations and your primary loss function (e.g., Dice + Cross-Entropy).
* **Success Criteria:** Stable convergence. This provides the "Reference IoU" for all future improvements.
* **Management:** Log this as `type: baseline` in the experiment tracker.

## Phase 3: Hyperparameter Sweeps (The "Tuning")
**Goal:** Maximize performance by optimizing the "knobs."
* **Setup:** Automated searches (Random or Bayesian) over Learning Rate, Weight Decay, and Augmentation strength.
* **Success Criteria:** Pushing the validation IoU toward the 80% target.
* **Management:** Use early-stopping to prune underperforming runs.

## Phase 4: Ablation Studies (The "Justification")
**Goal:** Determine the marginal value of specific inputs (e.g., Landsat vs. Topo).
* **Setup:** Systematically remove modalities or architectural components (e.g., skip connections or specific topo bands).
* **Success Criteria:** Identifying which features are essential for hitting high-performance targets.

---

## Data Management & Expansion Strategies

### 1. Supplementary Data Addition (Phase 2 or 3)
* **When:** If the model generalizes poorly or fails on specific edge cases (e.g., shadows or specific terrain types).
* **Action:** Introduce new, labeled samples into the training set. 
* **Requirement:** After adding significant new data, you must **re-run the Baseline** to ensure the "floor" hasn't shifted.

### 2. Active Sampling / Hard-Example Mining (Phase 3)
* **When:** If IoU is stalled at 75% and you need to break the 80% barrier.
* **Action:** Analyze the error masks. Sample existing data that contains high-error pixels (e.g., rare classes or complex boundaries) and increase their frequency in the training loop using a weighted sampler.

### 3. Data Re-Normalization (Phase 1)
* **When:** If the Overfit Silo fails despite a "correct" architecture.
* **Action:** Re-evaluate the mean/std of your Landsat and Topo bands. Ensure multi-modal inputs are on a similar scale (e.g., 0-1) so one doesn't mask the other.
