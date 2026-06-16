# ADR-0042: Extended Metrics for Multi-Task Learning (MTL) Consistency

**Status:** Accepted
**Date:** 2026-06-08
**Accepted:** 2026-06-16

## 1. Context

With the adoption of **ADR-0041**, the framework supports pixel-level Multi-Task
Learning (MTL) where multiple classification heads (Factorial or Hierarchical)
operate on a shared backbone. Metrics are computed and persisted on a per-head
basis (e.g., mIoU for Head A, mIoU for Head B), but MTL workflows also need
metrics that evaluate cross-head behavior.

However, in spatial segmentation, the relationship between these tasks is often
as important as the individual task performance. We need to evaluate:
1. **Per-Head Performance:** Standard mIoU for all active heads.
2. **Joint Accuracy:** How often a model is "completely correct" across all
dimensions for a single pixel.
3. **Logical Consistency:** Whether the model's predictions violate known domain
constraints (e.g., a pixel cannot be "Old Growth Forest" in an Age head if the
Landcover head predicts "Water").

## 2. Decision

We extended the evaluation suite within the `Session` engine, specifically the
logic in `src/landseg/session/engine/runtime/tasks`, with an MTL metrics
aggregator. The aggregator computes cross-head metrics during validation and
inference and persists them through `ValStepResults` and `InferStepResults`
alongside the existing per-head metrics.

The current implementation focuses on the runtime metric contract and persisted
step results. Console/dashboard reporting remains intentionally lightweight and
can be formalized as part of a broader reporting pass.

### 2.1. Holistic mIoU Reporting (existing)
The `ValStepResults` and `InferStepResults` currently aggregate mIoU across all
registered active heads. The persisted step results include a dictionary of
per-head metrics via `dict[str, AccumulatedMetrics]`.

### 2.2. Global Exact Match (GEM)
We introduced the **Global Exact Match** metric.
* **Definition:** A pixel is counted as a "match" if and only if the predicted
class ID matches the ground truth for **every** active head simultaneously.
* **Implementation:** The runtime aggregator computes a logical AND across the
individual correctness masks of all active heads before spatial reduction.
* **Purpose:** This provides a strict measure of the model's ability to capture
the factorial state of a pixel correctly.

### 2.3. Constraint Violation Metrics (Optional/Configurable)
To support domain-specific logic, we introduced an optional **Violation
Detector**.
* **Input:** User-defined pairwise constraints passed via session engine task
configuration.
* **Mechanism:** The detector evaluates configured head-prediction pairs. A
"Violation" is recorded if `source_head == trigger_val` and `target_head` is in
the configured `forbidden` class list.
* **Output:** A `violation_{name}` metric ($0.0$ to $1.0$), where $1.0$
indicates every valid pixel violates the named constraint and $0.0$ indicates
perfect logical consistency for that constraint.

### 2.4. Handling of Ignore Indices
In MTL scenarios, one head might have an `ignore_index` while another is valid
for the same pixel.
* **GEM:** Pixels containing an `ignore_index` in any head will be excluded from
the Global Exact Match calculation (masked out) to ensure the metric only reflects
areas with complete ground truth.
* **Violation Rate:** Violations will only be calculated for pixels where all
involved heads have valid (non-ignored) predictions.

## 3. Implementation Structure

1. **`MTLMetricsAggregator`**: A task-runtime component that consumes predicted
and target class-ID dictionaries.
2. **Validated Constraints**: Users define violations in the experiment YAML,
using 1-based class IDs consistent with label tensors:

   ```yaml
   session:
     engine_tasks:
       constraints:
         - name: water_cannot_have_age
           source_head: landcover
           trigger_val: 1
           target_head: age
           forbidden: [1, 2, 3, 4]
   ```

Constraint names must be unique, source and target heads must differ, and all
referenced heads/classes are validated against `DataSpecs`.

## 4. Consequences

### Positive
* **Deep Insight:** Provides a clearer picture of whether the model is learning
the relationships between tasks or just memorizing them independently.
* **Domain Safety:** Specifically addresses ecological validity, ensuring maps
produced are logically sound.
* **Standardized Persistence:** Centralizes MTL metrics in session step results
instead of requiring users to post-process raw predictions in notebooks.

### Negative
* **Computational Cost:** Calculating joint masks and iterating over constraint
lists adds overhead to validation and inference phases.
* **Memory Pressure:** Storing multiple correctness masks during the batch
aggregation might increase CPU/GPU memory usage slightly.

## 5. Future Work

* **Loss-Weight Tuning:** Using the Violation Rate as a signal for "Constraint
Loss" to penalize logical inconsistencies during training.
* **Confusion Matrix Overlays:** Visualizing joint confusion between two heads
 to identify where specific class combinations are failing.
* **Reporting Formalization:** Expanding console, dashboard, and report outputs
for all metric families, including MTL metrics.
* **Automated Tests:** Adding focused metric tests once the broader project API
stabilizes.


### Summary of the ADR focus:
1.  **Metric Suite Extension:** Moving beyond isolated head reporting.
2.  **GEM (Global Exact Match):** High-bar accuracy tracking for factorial
classification.
3.  **Violation Logic:** Introducing a way to encode "Biological/Physical Reality"
into the evaluation.
