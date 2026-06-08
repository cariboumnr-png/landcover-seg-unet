# ADR-0042: Extended Metrics for Multi-Task Learning (MTL) Consistency

**Status:** Proposed
**Date:** 2026-06-08

## 1. Context

With the adoption of **ADR-0041**, the framework supports pixel-level Multi-Task
Learning (MTL) where multiple classification heads (Factorial or Hierarchical)
operate on a shared backbone. Currently, metrics are computed and reported primarily
on a per-head basis (e.g., mIoU for Head A, mIoU for Head B).

However, in spatial segmentation, the relationship between these tasks is often
as important as the individual task performance. We need to evaluate:
1. **Per-Head Performance:** Standard mIoU for all active heads.
2. **Joint Accuracy:** How often a model is "completely correct" across all
dimensions for a single pixel.
3. **Logical Consistency:** Whether the model's predictions violate known domain
constraints (e.g., a pixel cannot be "Old Growth Forest" in an Age head if the
Landcover head predicts "Water").

## 2. Decision

We will extend the evaluation suite within the `Session` engine (specifically
targeting the logic in `src/landseg/session/engine/runtime/tasks`) to support a
"Global MTL Report."

### 2.1. Holistic mIoU Reporting
The `MetricSuite` will be updated to aggregate mIoU across all registered heads
in the `HeadManager`. The final report will include a flattened dictionary of
metrics prefixed by head name (e.g., `val/head_landcover/mIoU`, `val/head_age/mIoU`).

### 2.2. Global Exact Match (GEM)
We introduce the **Global Exact Match** metric.
* **Definition:** A pixel is counted as a "match" if and only if the predicted
class ID matches the ground truth for **every** active head simultaneously.
* **Implementation:** This requires computing a bitwise/logical AND across the
individual "correctness" masks of all heads before spatial reduction.
* **Purpose:** This provides a strict measure of the model's ability to capture
the factorial state of a pixel correctly.

### 2.3. Constraint Violation Metrics (Optional/Configurable)
To support domain-specific logic, we will introduce an optional **Violation Detector**.
* **Input:** A user-defined "Constraint Matrix" or "Constraint Function" passed
via configuration.
* **Mechanism:** The detector evaluates pairs (or sets) of head predictions. A
"Violation" is recorded if `Head_A == Class_X` and `Head_B == Class_Y` where
$(X, Y)$ is an invalid state.
* **Output:** A `ViolationRate` metric ($0.0$ to $1.0$), where $1.0$ indicates
every pixel violates a constraint and $0.0$ indicates perfect logical consistency.

### 2.4. Handling of Ignore Indices
In MTL scenarios, one head might have an `ignore_index` while another is valid
for the same pixel.
* **GEM:** Pixels containing an `ignore_index` in any head will be excluded from
the Global Exact Match calculation (masked out) to ensure the metric only reflects
areas with complete ground truth.
* **Violation Rate:** Violations will only be calculated for pixels where all
involved heads have valid (non-ignored) predictions.

## 3. Proposed Implementation Structure

1. **`MTLMetricAggregator`**: A new component within the task runtime that consumes
the `y_dict` (predictions) and `target_dict` (labels).
2. **Registry-based Constraints**: Users define violations in the experiment YAML, e.g.:

   ```yaml
   metrics:
     constraints:
       - name: "water_cannot_have_age"
         head_a: "landcover"
         val_a: 0 # Water
         head_b: "age"
         invalid_b: [1, 2, 3, 4] # Any age class
   ```

## 4. Consequences

### Positive
* **Deep Insight:** Provides a clearer picture of whether the model is learning
the relationships between tasks or just memorizing them independently.
* **Domain Safety:** Specifically addresses ecological validity, ensuring maps
produced are logically sound.
* **Standardized Reporting:** Centralizes MTL metrics instead of requiring users
to post-process results in notebooks.

### Negative
* **Computational Cost:** Calculating joint masks and iterating over constraint
lists adds overhead to the validation phase.
* **Memory Pressure:** Storing multiple correctness masks during the batch
aggregation might increase CPU/GPU memory usage slightly.

## 5. Future Work

* **Loss-Weight Tuning:** Using the Violation Rate as a signal for "Constraint
Loss" to penalize logical inconsistencies during training.
* **Confusion Matrix Overlays:** Visualizing joint confusion between two heads
 to identify where specific class combinations are failing.


### Summary of the ADR focus:
1.  **Metric Suite Extension:** Moving beyond isolated head reporting.
2.  **GEM (Global Exact Match):** High-bar accuracy tracking for factorial
classification.
3.  **Violation Logic:** Introducing a way to encode "Biological/Physical Reality"
into the evaluation.