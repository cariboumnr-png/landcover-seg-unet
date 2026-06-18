# ADR-0044: Sweep Objective Presets and Objective Module Structure

**Status:** Proposed
**Date:** 2026-06-18
**Related ADRs:** ADR-0026, ADR-0038, ADR-0039, ADR-0040, ADR-0041, ADR-0042, ADR-0043

## 1. Context

ADR-0026 established the study layer as the boundary for Hydra/Optuna sweep
orchestration and preserved a minimal trial contract: each trial derives a
trial-specific root configuration, runs a session, reports intermediate scalar
values to Optuna, and returns one scalar objective.

The current sweep objective implementation is intentionally small. It supports a
single `base` objective preset that tunes a proof-of-concept search space:

* optimizer learning rate
* optimizer weight decay
* data loader patch size
* data loader batch size

That proof-of-concept is useful, but the model and runtime have grown. The
project now supports richer model bodies and bottlenecks, multi-head training,
extended metrics, cross-head constraints, and consistency regularization. The
sweep layer should therefore offer more curated objective presets without
turning the Optuna adapter into a large conditional block.

This ADR proposes a feature branch for expanding the sweep layer around
conceptual objective preset families and a small module structure that keeps
search-space mutation separate from sweep orchestration.

## 2. Decision

We will expand sweep objectives as named presets grouped by modeling
perspective. A preset is a curated search space that receives a root config and
an Optuna trial, then returns a trial-specific root config.

The existing scalar Optuna objective contract remains unchanged.

The sweep objective layer will continue to own:

* selecting the configured preset
* deriving per-trial config variants
* reporting scalar validation progress to Optuna
* raising `TrialPruned` when requested by Optuna

The sweep objective layer will not own:

* model construction
* training execution
* metric aggregation
* checkpoint selection
* post hoc cross-run analysis
* domain-specific interpretation of completed studies

## 3. Objective Preset Families

### 3.1. Base Optimization Presets

These presets tune broadly applicable runtime knobs. They are the natural
successor to the current `base` proof-of-concept.

Suggested presets:

* `base`: current compatibility preset for learning rate, weight decay, patch
  size, and batch size.
* `optimizer`: optimizer-centered tuning for learning rate, weight decay, and
  scheduler-adjacent knobs when those become exposed.
* `throughput`: memory and throughput tuning for patch size, batch size, AMP,
  and potentially gradient accumulation if introduced later.

Purpose:

* establish stable baseline sweep behavior
* find practical training settings before changing model semantics
* support short smoke sweeps and resource-fit sweeps

### 3.2. Data Geometry Presets

These presets explore how the model sees spatial context.

Suggested presets:

* `data_geometry`: patch size, batch size, and loader-related sampling knobs
  that affect spatial context and memory pressure.
* `context_window`: patch size and, if later exposed, tile/block context or
  overlap-sensitive settings.

Purpose:

* determine the spatial context needed for stable land-cover boundaries
* balance local texture detail against larger ecological or geomorphic context
* separate data-shape effects from optimizer effects

### 3.3. Architecture Presets

These presets tune model capacity and architectural variants exposed by the
configuration layer.

Suggested presets:

* `architecture`: model body, base channels, bottleneck type, and conditioner
  choices where sweepable.
* `bottleneck`: transformer, hybrid, or convolutional bottleneck settings,
  aligned with ADR-0040.
* `conditioning`: domain-conditioning choices such as no conditioning, FiLM, or
  concatenation when compatible with the selected model.

Purpose:

* compare capacity and inductive-bias choices under otherwise stable training
* isolate backbone and bottleneck trade-offs from loss tuning
* support model-selection studies without hand-editing configs between runs

### 3.4. Loss and Regularization Presets

These presets tune the training objective itself.

Suggested presets:

* `loss_balance`: focal and dice weights, preserving valid composite-loss
  semantics.
* `regularization`: spectral, total variation, and related regularization
  weights when active in the configured loss stack.
* `mtl_consistency`: consistency regularization weight and reduction choices
  introduced by ADR-0043.

Purpose:

* tune class-boundary behavior and imbalance handling
* control smoothness and spectral-spatial priors
* align training signals with multi-head consistency metrics

### 3.5. Multi-Task Presets

These presets tune how multiple heads are trained together.

Suggested presets:

* `head_weights`: per-head `HeadSpec.weight` values for active heads.
* `mtl_joint`: head weights plus consistency regularization, scoped to
  multi-head sessions.
* `hierarchy`: parent-child related head weighting and gating-sensitive loss
  settings where exposed.

Purpose:

* find stable weighting across parent and refinement heads
* reduce dominance by easier heads
* make cross-head consistency a tunable modeling signal rather than a fixed
  assumption

### 3.6. Curriculum Presets

These presets tune phase-based training behavior.

Suggested presets:

* `curriculum`: phase epoch counts, phase learning-rate scales, and freeze/joint
  tuning options for configured curriculum schemas.
* `fine_tune`: later-phase learning-rate scales, joint-tuning duration, and
  active/frozen head choices where valid.

Purpose:

* compare continuous and staged multi-head learning strategies
* tune how long parent or refinement heads train before joint updates
* keep phase-level experimentation explicit instead of burying it in ad hoc
  config edits

### 3.7. Composite Presets

Composite presets combine a small number of lower-level families for common
study questions.

Suggested presets:

* `quick`: low-dimensional smoke-search preset, likely equivalent to `base` or
  a narrowed `optimizer` preset.
* `capacity`: architecture plus optimizer settings.
* `mtl_quality`: head weights, loss balance, and consistency regularization.
* `production_candidate`: conservative combination of optimizer, throughput,
  and loss/regularization settings intended for longer candidate-selection
  sweeps.

Purpose:

* provide named, repeatable study entry points
* avoid forcing every branch or notebook to redefine common search spaces
* let broad sweeps remain curated rather than arbitrary

## 4. Module Structure

The current `src/landseg/study/sweep/objectives.py` module should remain the
Optuna-facing adapter. Preset-specific logic should move into a small package
under the sweep layer:

```text
src/landseg/study/sweep/
  objectives.py              # Optuna objective adapter and scalar reporting
  objective_presets/
    __init__.py              # public preset registry exports
    registry.py              # name -> preset resolver
    types.py                 # preset protocol/type aliases
    base.py                  # base/optimizer/throughput presets
    data.py                  # data geometry presets
    architecture.py          # model/body/bottleneck/conditioning presets
    losses.py                # loss and regularization presets
    multitask.py             # head weighting and MTL consistency presets
    curriculum.py            # phase/curriculum presets
    composite.py             # curated combinations of other presets
```

The module names are intentionally conceptual, not tied to one config class.
Each module should group related search-space builders and small helper
functions for applying trial suggestions to a copied root config.

The initial branch may implement fewer files if only a subset of presets is
added. The important boundary is that `objectives.py` should delegate preset
selection and trial-config derivation instead of accumulating preset details.

## 5. Preset Contract

Each preset should follow a simple contract:

```text
(cfg: RootConfigShape, trial: optuna.Trial) -> RootConfigShape
```

Recommended behavior:

* deep-copy the root config before mutation
* use stable, descriptive Optuna parameter names
* keep parameter names grouped by concept, for example
  `optimizer.lr`, `data.patch_size`, or `mtl.consistency_lambda`
* validate incompatible choices close to the preset boundary
* prefer narrow, meaningful search spaces over very broad exploratory spaces
* keep preset logic deterministic except for Optuna suggestions

Preset composition should be explicit. A composite preset should call smaller
preset functions in a predictable order rather than duplicate their mutation
logic.

## 6. Configuration Direction

The existing configuration shape may evolve from:

```yaml
pipeline:
  study_sweep:
    objective: base

study:
  base:
    learning_rate: [1e-5, 1e-1]
```

toward a preset-oriented structure such as:

```yaml
pipeline:
  study_sweep:
    objective: mtl_quality

study:
  objectives:
    base:
      learning_rate: [1e-5, 1e-1]
    loss_balance:
      focal_weight: [0.25, 0.75]
      dice_weight: [0.25, 0.75]
    mtl_consistency:
      consistency_lambda: [0.0, 0.2]
```

The exact schema should be finalized during implementation. The important
direction is that search-space definitions should live under `study`, while the
pipeline-level sweep config chooses which preset to run.

Backward compatibility for `study.base` and `objective: base` should be
preserved during the first expansion branch.

## 7. Consequences

### Positive

* Sweep objectives become easier to extend without bloating the Optuna adapter.
* Search spaces are named by modeling intent rather than by individual
  parameter lists.
* Existing `base` behavior can remain stable while richer presets are added.
* Composite presets provide repeatable branch and notebook entry points.
* The study layer continues to respect the ADR-0026 scalar optimization
  boundary.

### Negative

* More presets create more configuration surface to document and validate.
* Some presets will only be valid for specific model/session modes.
* Composite presets can become too broad if not curated carefully.
* Parameter naming must be stable to keep Optuna study histories interpretable.

## 8. Implementation Notes for the Feature Branch

Suggested first branch sequence:

1. Add the `objective_presets` package and move the current `base` logic into
   `objective_presets/base.py`.
2. Add a registry so `objectives.py` resolves presets by name.
3. Preserve current `objective: base` behavior and tests.
4. Add one low-risk new preset, such as `optimizer` or `loss_balance`.
5. Add schema/config entries only for presets implemented in the branch.
6. Add validation for unsupported preset/session combinations.
7. Document example Hydra overrides for each implemented preset.

The first implementation branch should avoid adding all proposed presets at
once. A good first target is:

* `base`
* `optimizer`
* `loss_balance`
* `mtl_consistency` if the branch is meant to exercise ADR-0043

Architecture, curriculum, and composite presets can follow once the registry
and schema pattern has settled.

## 9. Future Work

* Study-level comparison templates for ranking presets after sweeps complete.
* Optuna parameter importance reporting grouped by preset family.
* Preset metadata describing supported session modes, required config sections,
  and expected computational cost.
* Multi-objective Optuna studies, if the project later decides that the scalar
  objective contract is too restrictive for certain model-selection questions.
* Preset versioning for long-running or persistent Optuna studies.

