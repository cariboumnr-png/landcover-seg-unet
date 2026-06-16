# ADR-0043: Consistency Regularization for Multi-Task Learning

**Status:** Proposed
**Date:** 2026-06-16

## 1. Context

Following **ADR-0041** (Multi-head Architecture) and **ADR-0042** (Extended
Metrics), the framework supports multiple active classification heads with
per-head validation metrics and cross-head MTL metrics.

The current loss stack already supports configurable weighting at two levels:

1. **Within each head**, `CompositeLoss` combines supervised losses and
regularizers using configured component weights:

   $$L_{head} =
   w_f L_{focal} +
   w_d L_{dice} +
   w_s R_{spectral} +
   w_{tv} R_{tv}$$

2. **Across heads**, `multihead_loss` multiplies each head loss by the
corresponding `HeadSpec.weight` before adding it to the shared total.

Therefore, the main missing piece is not basic loss weighting. The missing piece
is a differentiable training signal that aligns with the cross-head consistency
metrics introduced in ADR-0042.

ADR-0042 can detect invalid multi-head predictions after the forward pass, but
training currently has no explicit penalty for assigning probability mass to
those invalid states. This means the model can improve per-head focal/dice
objectives while still learning combinations that are ecologically or physically
inconsistent.

## 2. Decision

We will add an optional **MTL consistency regularizer** that reuses the
constraint definitions from ADR-0042 and penalizes invalid cross-head probability
assignments during training.

This term may be implemented as a PyTorch loss module, but conceptually it is a
regularizer. Focal and dice remain the primary supervised classification losses.
Spectral, total variation, and consistency terms are auxiliary regularizers that
shape the output distribution.

The total training objective becomes:

$$L_{total} =
\sum_{h \in Heads} w_h L_h
+ \lambda_{mtl} R_{consistency}$$

where:

* $L_h$ is the existing per-head composite objective.
* $w_h$ is the existing head-level weight from `HeadSpec`.
* $R_{consistency}$ is the new cross-head consistency regularizer.
* $\lambda_{mtl}$ is a user-configured consistency weight.

### 2.1. Consistency Regularizer

The regularizer uses softmax probabilities rather than hard class IDs. For each
configured ADR-0042 constraint:

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

the regularizer computes the probability that a pixel is in the invalid state:

$$P(source = trigger) \cdot P(target \in forbidden)$$

The batch regularizer is the mean invalid probability over valid pixels and
configured constraints. Minimizing this value encourages the model to reduce
confidence in at least one side of each invalid combination.

### 2.2. Masking and Ignore Handling

Consistency regularization must follow the same validity semantics as the
ADR-0042 violation metrics:

* Pixels with `ignore_index` in either involved head are excluded.
* Constraints are skipped when either involved head is inactive.
* Constraint class IDs are 1-based in configuration and are mapped to 0-based
logit/probability indices internally.
* For hierarchical heads, existing parent-child masking should remain local to
the supervised head losses unless a consistency constraint explicitly involves
that parent/child pair.

### 2.3. Configuration

The regularizer should live with session engine task configuration, next to the
ADR-0042 constraints it consumes:

```yaml
session:
  engine_tasks:
    consistency:
      weight: 0.0
      reduction: mean
    constraints:
      - name: water_cannot_have_dep
        source_head: polytype_groups
        trigger_val: 2
        target_head: lastdep
        forbidden: [1, 2, 3]
```

`weight: 0.0` disables the regularizer while preserving the metric-only
constraint behavior from ADR-0042.

### 2.4. Observability

The training result should expose the consistency regularizer separately from
per-head losses so users can tell whether it is active and whether it is
dominating the supervised objective.

At minimum, training output should distinguish:

* Per-head supervised/composite losses.
* The unweighted consistency regularizer value.
* The weighted consistency contribution to total loss.

Full component-level logging for focal, dice, spectral, and TV remains useful,
but it is a broader loss observability improvement and does not block this ADR.

## 3. Proposed Implementation Structure

1. **`ConsistencyRegularizer`**: A differentiable module that consumes
`multihead_preds`, `multihead_targets`, active constraints, and `ignore_index`.
It returns a scalar tensor plus optional per-constraint diagnostic values.

2. **Task factory wiring**: Reuse the validated ADR-0042 constraints when
constructing the consistency regularizer. Constraint validation should remain
centralized so metrics and regularization share the same contract.

3. **Loss aggregation update**: Extend `multihead_loss` or wrap it with an MTL
loss aggregator that adds
`consistency.weight * consistency_regularizer(...)` after the existing per-head
loss accumulation.

4. **Training result schema update**: Add a small structured field for auxiliary
regularizers, rather than mixing consistency values into per-head loss names.

## 4. Consequences

### Positive

* **Metric-aligned training:** The training objective directly addresses the
same invalid states measured by ADR-0042 violation metrics.
* **More plausible maps:** The model is discouraged from placing probability
mass on impossible cross-head combinations, not merely flagged after inference.
* **Scoped extension:** Existing focal/dice losses, spectral regularization, TV
regularization, and head weighting remain intact.

### Negative

* **Additional tuning:** The consistency weight introduces another hyperparameter.
Poorly tuned values could under-constrain the model or suppress valid supervised
learning.
* **Constraint dependence:** Incorrect domain constraints can bias training in
the wrong direction.
* **Extra compute:** The regularizer requires softmax probabilities for involved
heads and additional per-pixel reductions during training.

## 5. Future Work

* Dynamic task weighting such as uncertainty weighting, GradNorm, or related
methods for adapting `HeadSpec.weight` during training.
* Richer component-level loss logging for focal, dice, spectral, TV, and
consistency terms.
* Constraint schedules, such as warming up the consistency weight after the
supervised heads have started learning stable class boundaries.
* Higher-order constraints involving more than two heads.
