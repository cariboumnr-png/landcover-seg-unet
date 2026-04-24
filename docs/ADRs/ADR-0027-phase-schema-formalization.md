# ADR-0027: Training Phase Schemas for Hierarchical Multi-Head Models

- **Status:** Accepted
- **Date:** 2026-04-23

> The **Top-Down Baseline Phase Schema** has been implemented as the
> initial, supported phase schema. Other schemas are documented here
> and intentionally deferred.

## Context

The training system now supports multi-phase execution where each phase
controls:

- Which model heads are active or frozen
- Logit adjustment configuration for class imbalance
- Learning rate scaling and scheduling
- Epoch budgets and early stopping behavior

The system targets hierarchical classification problems, where:

- A shared backbone feeds a **parent (coarse) classification head**
- Each parent class may have one or more **child (fine-grained) heads**
- Heads can be independently activated or frozen
- Parent–child relationships may or may not be present, depending on
  upstream configuration

As part of the recent phase-system refactor, the **phase abstractions
have been clarified and stabilized independently of execution
mechanics**, in preparation for a future runner redesign.

## Problem Statement

A flexible phase system enables a wide range of curricula, but also
introduces:

- Increased cognitive and control-flow complexity in the runner
- Ambiguity around which phase schemas are first-class and supported
- Difficulty reasoning about default behavior, especially under HPO

To address this, a clear set of **canonical phase schemas** was needed
to:

- Establish sensible defaults
- Bound system complexity
- Enable future extensions without repeatedly refactoring the runner

## Considered Phase Schemas

The following phase schemas were evaluated conceptually and documented
for reference.

### 1. Top-Down Baseline (Classical Curriculum)

**Description**
A strict hierarchical curriculum where representation learning proceeds
from coarse to fine.

**Phases**
1. Parent head active; all child heads frozen
2. Parent head frozen; selected child heads active
3. (Optional) Parent and child heads jointly active for fine-tuning

**Characteristics**
- Strong inductive bias aligned with hierarchical labels
- Simple head activation logic
- Easy to debug and reason about

### 2. Soft Hierarchy (Overlapping Phases)

**Description**
A smoother curriculum where parent and child heads are active together,
with influence controlled via loss weighting.

**Characteristics**
- Reduced representation drift at phase boundaries
- Requires careful loss-weight configuration
- Harder to inspect than strict freezing

### 3. Conditional Child Activation

**Description**
Child heads are trained only on samples belonging to their associated
parent class.

**Characteristics**
- Limits gradient noise from irrelevant child classes
- Strong alignment with hierarchical semantics
- Requires conditional routing logic in the training loop

### 4. Imbalance-Aware Curriculum

**Description**
A curriculum that introduces child heads progressively based on class
frequency, combined with stronger logit adjustment for rare classes.

**Characteristics**
- Explicitly targets long-tailed distributions
- Increased operational complexity
- Requires dataset-level statistics and policy decisions

## Decision

- The **Top-Down Baseline Phase Schema has been implemented** as the
  **default and only supported schema** in the current phase system.
- A **non-hierarchical default schema** (single-phase or fully joint
  training) is used when no parent–child structure is configured
  upstream.
- The remaining schemas are **documented but not implemented**, and do
  not participate in training execution at this stage.

This implementation prioritizes:

- Clarity and inspectability of training behavior
- Minimal runner complexity during the current refactor phase
- A stable and widely applicable baseline curriculum

## Consequences

### Positive

- The runner and configuration system now have a clear, bounded notion
  of what a “phase” is
- Phase schemas are explicit, named, and configurable without embedding
  logic in the runner
- The system is prepared for a follow-up runner redesign without
  revisiting phase semantics again

### Negative

- More advanced curricula are not yet available
- Some datasets may benefit from deferred schemas
- Phase semantics are currently descriptive rather than strictly
  enforced at runtime

## Follow-Ups

- A subsequent ADR will introduce a **generator-based training runner**.
  The interaction between the new runner and the phase system is
  expected to further refine how phases are executed, without changing
  the conceptual schemas defined here.
- Additional phase schemas (e.g. soft hierarchy, conditional activation)
  will be implemented once the new runner is in place.
- At that time, we will provide:
  - A **reference phase profile** intended for user customization
  - More **explicit and enforced phase semantics** (e.g. validation of
    head activation, routing, and logit-adjust behavior)

This ADR will be revisited once the generator-based runner is stable.
