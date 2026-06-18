# ADR-0043: Consistency Regularization for Multi-Task Learning

**Status:** Accepted
**Date:** 2026-06-18

## 1. Context

Following **ADR-0041** (Multi-head Architecture) and **ADR-0042** (Extended
Metrics), the framework supports multiple active classification heads with
per-head validation metrics and cross-head MTL metrics.

The loss stack already supports configurable weighting at two levels:

1. **Within each head**, `CompositeLoss` combines supervised losses and
regularizers using configured component weights:

   $$L_{head} =
   w_f L_{focal} +
   w_d L_{dice} +
   w_s R_{spectral} +
   w_{tv} R_{tv}$$

2. **Across heads**, `multihead_loss` (now renamed as `multihead_objective`) multiplies each head loss by the corresponding `HeadSpec.weight` before adding it to the shared total.

The remaining gap was a differentiable training signal aligned with the
cross-head consistency metrics introduced in ADR-0042. ADR-0042 detects invalid
multi-head predictions after the forward pass, but metric-only detection does
not discourage the model from assigning probability mass to those invalid
states during training.

## 2. Decision

We have added an optional **MTL consistency regularizer** that reuses the
constraint definitions from ADR-0042 and penalizes invalid cross-head probability
assignments during training.

This term is implemented as a PyTorch module, but conceptually it is a
regularizer. Focal and dice remain the primary supervised classification losses.
Spectral, total variation, and consistency terms are auxiliary losses that
shape the output distribution.

The total training objective is now:

$$L_{total} =
\sum_{h \in Heads} w_h L_h
+ \lambda_{mtl} R_{consistency}$$

where:

* $L_h$ is the existing per-head composite objective.
* $w_h$ is the existing head-level weight from `HeadSpec`.
* $R_{consistency}$ is the cross-head consistency regularizer.
* $\lambda_{mtl}$ is the configured consistency weight.

The implementation stores the consistency weight on the regularizer as
`consistency_lambda`, so the value returned by `ConsistencyRegularizer.forward`
is already the weighted contribution that is added to the total objective.

### 2.1. Consistency Regularizer

The regularizer uses softmax probabilities rather than hard class IDs. For each
configured ADR-0042 constraint:

```yaml
session:
  engine_tasks:
    mtl_constraints:
      - name: water_cannot_have_dep
        source_head: polytype_groups
        trigger_val: 2
        target_head: lastdep
        forbidden: [1, 2, 3]
```

the regularizer computes the probability that a pixel is in the invalid state:

$$P(source = trigger) \cdot P(target \in forbidden)$$

The batch regularizer is reduced over valid pixels and configured constraints.
The supported training reductions are `mean` and `sum`. Per-constraint
diagnostics remain available through `ConsistencyRegularizer.by_constraint`.

### 2.2. Masking and Ignore Handling

Consistency regularization follows the same validity semantics as the ADR-0042
violation metrics:

* Pixels with `ignore_index` in either involved head are excluded.
* Constraints are skipped when either involved head is inactive.
* Constraint class IDs are 1-based in configuration and are mapped to 0-based
logit/probability indices internally.
* For hierarchical heads, existing parent-child masking remains local to the
supervised head losses unless a consistency constraint explicitly involves that
parent/child pair.

### 2.3. Configuration

The regularizer lives with session engine task configuration, next to the
ADR-0042 constraints it consumes:

```yaml
session:
  engine_tasks:
    mtl_constraints:
      - name: water_cannot_have_dep
        source_head: polytype_groups
        trigger_val: 2
        target_head: lastdep
        forbidden: [1, 2, 3]
    mtl_reg_configs:
      consistency_lambda: 0.05
      consistency_reduction: mean
```

`consistency_lambda: 0.0` disables the regularization contribution while
preserving the metric-only constraint behavior from ADR-0042. The
`mtl_reg_configs` namespace is intentionally regularizer-oriented rather than
specific to this one term, because additional multi-head regularizers may be
added later.

### 2.4. Observability

Training results expose multi-head regularization separately from per-head
losses through `TrainStepResults.regularization`.

The reported `mtl_regularization` scalar follows the same convention as the
existing loss reporting: it is the weighted scalar contribution used by the
objective. The corresponding weight is persisted in the run configuration, so
the raw scale can be reconstructed from the saved config when needed. Richer
component-level diagnostics, including raw per-constraint consistency values,
remain future observability work.

## 3. Implementation Structure

1. **`ConsistencyRegularizer`**: A differentiable module consumes
`multihead_preds`, `multihead_targets`, compiled active constraints, and
`ignore_index`. It returns a scalar tensor for objective aggregation and exposes
`by_constraint(...)` for diagnostics.

2. **Task factory wiring**: `build_engine_tasks` compiles ADR-0042 constraints
for tensor indexing when constructing the regularizer. The raw 1-based
constraints continue to feed `MTLMetricsAggregator`, so metrics and
regularization share the same user-facing constraint contract.

3. **Objective aggregation**: `multihead_objective` adds the weighted
regularization tensor after existing per-head objective accumulation.

4. **Training result schema**: `TrainStepResults.regularization` carries
auxiliary regularizer values separately from `head_losses`.

## 4. Consequences

### Positive

* **Metric-aligned training:** The training objective directly addresses the
same invalid states measured by ADR-0042 violation metrics.
* **More plausible maps:** The model is discouraged from placing probability
mass on impossible cross-head combinations, not merely flagged after inference.
* **Scoped extension:** Existing focal/dice losses, spectral regularization, TV
regularization, and head weighting remain intact.
* **Consistent reporting:** Regularization output follows the existing
already-weighted scalar reporting convention for training losses.

### Negative

* **Additional tuning:** The consistency weight introduces another
hyperparameter. Poorly tuned values could under-constrain the model or suppress
valid supervised learning.
* **Constraint dependence:** Incorrect domain constraints can bias training in
the wrong direction.
* **Extra compute:** The regularizer requires softmax probabilities for involved
heads and additional per-pixel reductions during training.
* **Indirect raw observability:** Raw unweighted consistency values are not
persisted in the standard step result payload.

## 5. Future Work

* Dynamic task weighting such as uncertainty weighting, GradNorm, or related
methods for adapting `HeadSpec.weight` during training.
* Richer component-level loss logging for focal, dice, spectral, TV, and raw
consistency terms.
* Constraint schedules, such as warming up the consistency weight after the
supervised heads have started learning stable class boundaries.
* Higher-order constraints involving more than two heads.
* A more general auxiliary regularizer registry under `mtl_reg_configs`.
