# ADR-0044: Sweep Objective Presets and Objective Module Structure

**Status:** Proposed
**Date:** 2026-06-18
**Related ADRs:** ADR-0026, ADR-0038, ADR-0039, ADR-0040, ADR-0041, ADR-0042, ADR-0043

## 1. Context

ADR-0026 established the study layer as the boundary for Hydra/Optuna sweep
orchestration and preserved a minimal trial contract: each trial derives a
trial-specific root configuration, runs a session, reports intermediate scalar
values to Optuna, and returns one scalar objective.

The current sweep objective implementation provides a minimal search space
through a single `base` preset. This preset mutates a small set of configuration
parameters:

* optimizer learning rate
* optimizer weight decay
* data loader patch size
* data loader batch size

The configuration system and runtime have since expanded to support:

* multiple model architectures and bottlenecks
* multi-head training
* extended metric computation
* cross-head consistency constraints and regularization

The sweep layer must support richer configuration mutation while preserving the
existing trial contract and maintaining a clear boundary between mechanical
configuration sampling and human-guided experimentation.

***

## 2. Decision

The sweep objective layer is defined as a **mechanical configuration
transformation system**.

A preset is a deterministic function:

```text
(cfg: RootConfigShape, trial: optuna.Trial) -> RootConfigShape
```

It produces a trial-specific configuration by mutating a copied root config
using Optuna parameter suggestions.

The sweep objective layer strictly owns:

* selecting a preset by name
* deriving per-trial configuration variants
* interacting with Optuna (suggestions, pruning, reporting)

The sweep objective layer explicitly does not encode:

* modeling intent
* training strategy selection
* interpretation of results
* recommendations about when or why to use a preset

Presets are treated as **named configuration mutation functions**. Any semantic
meaning suggested by preset names is external to the system and not enforced by
the implementation.

The scalar Optuna objective contract remains unchanged.

***

## 3. Parameter Grouping Presets

Presets are organized into families based on grouping of related configuration
parameters.

These groupings are determined by configuration structure and parameter
dependencies, not by intended modeling outcomes.

### 3.1. Runtime Optimization Presets

Mutate optimizer and runtime-level parameters:

- optimizer learning rate
- optimizer weight decay
- optimizer-related scheduler parameters (if exposed)

Suggested presets:

- `optimizer`
- `throughput`

***

### 3.2. Data Geometry Presets

Mutate parameters affecting data sampling and spatial structure:

- patch size
- batch size
- data loader parameters influencing spatial context

Suggested presets:

- `data_geometry`
- `context_window`

***

### 3.3. Architecture Presets

Mutate parameters describing model structure:

* model body selection
* channel counts
* bottleneck configuration
* conditioning mechanisms

Suggested presets:

* `architecture`
* `bottleneck`
* `conditioning`

***

### 3.4. Objective (Loss and Regularization) Presets

Mutate parameters associated with the training objective:

* per-head loss weights (e.g., focal, dice)
* regularization weights (e.g., spectral, total variation)
* multi-head consistency parameters

Suggested presets:

* `loss_balance`
* `regularization`
* `mtl_consistency`

***

### 3.5. Multi-Task Presets

Mutate parameters describing relationships between heads:

* per-head weights
* multi-head consistency weighting
* hierarchical head configuration where exposed

Suggested presets:

* `head_weights`
* `mtl_joint`
* `hierarchy`

***

### 3.6. Composite Presets

Composite presets combine multiple parameter-group mutations into a single
preset.

These presets invoke other preset functions in a fixed order.

Suggested presets:

* `quick`
* `capacity`
* `mtl_quality`
* `production_candidate`


### Smoke-test Preset

Additionally, we will provide a minimal, low-dimensional mutation for validation
of the sweep pipeline.

Suggested presets:

- `smoke`

This preset:

- mutates a very small number of parameters
- uses narrow ranges
- is intended for fast validation of sweep mechanics
- does not correspond to any configuration domain

***

## 4. Module Structure

The Optuna-facing adapter remains:

```text
src/landseg/study/sweep/objectives.py
```

Preset logic is moved into a dedicated package:

```text
src/landseg/study/sweep/
  objectives.py
  objective_presets/
    __init__.py
    registry.py
    types.py
    base.py
    data.py
    architecture.py
    losses.py
    multitask.py
    composite.py
```

Module grouping is based on configuration domains, not on specific config types.

`objectives.py` delegates preset resolution and must not contain preset-specific
logic.

***

## 5. Preset Contract

Each preset must:

* accept `(cfg, trial)` and return a new config
* operate on a deep copy of the root config
* use stable and descriptive Optuna parameter names
* group parameter names by configuration domain
* validate incompatible parameter combinations locally
* remain deterministic aside from Optuna suggestions

Additional constraints:

* presets must not read training results or metrics
* presets must not depend on previous trials
* presets must not maintain state
* presets must remain pure functions of `(cfg, trial)`

Composite presets must call other presets explicitly rather than duplicating
mutation logic.

***

## 5.1. Mechanical vs Human-Guided Boundary

The sweep layer operates strictly on the **mechanical axis**:

* configuration mutation
* parameter sampling
* trial isolation

The following are outside the scope of this ADR:

* interpretation of preset names
* selection of appropriate presets
* comparison between presets or studies
* determination of desirable outcomes

This separation ensures that the sweep layer remains:

* stateless
* composable
* independent of evolving modeling practices

***

## 6. Configuration Direction

Configuration evolves toward a preset-oriented structure:

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

Key properties:

* pipeline selects preset by name
* `study.objectives` defines parameter ranges
* presets map parameter suggestions onto the configuration

Backward compatibility with `objective: base` is preserved.

***

## 7. Consequences

### Positive

* Preset logic is isolated and composable
* Sweep layer remains small and deterministic
* Configuration mutation is explicit and inspectable
* Preset expansion does not increase complexity of the Optuna adapter
* The scalar objective contract from ADR-0026 is preserved

### Negative

* Larger number of presets increases configuration surface
* Some presets may only be valid under specific configurations
* Parameter naming must remain stable for reproducibility

***

## 8. Implementation Notes for the Feature Branch

Suggested sequence:

1. Create `objective_presets` package
2. Move existing `base` preset into `base.py`
3. Implement preset registry
4. Update `objectives.py` to resolve presets via registry
5. Preserve current `objective: base` behavior
6. Implement a small number of additional presets (e.g., `optimizer`, `loss_balance`)
7. Add validation for preset/config compatibility
8. Document Hydra overrides for implemented presets

Initial branch scope should remain limited.

***

## 9. Future Work

* Preset metadata (supported configurations, parameter domains)
* Validation schemas for preset applicability
* Parameter grouping visualization and audit tools
* Multi-objective study support if scalar objective becomes limiting
* Preset versioning for persistent Optuna studies
