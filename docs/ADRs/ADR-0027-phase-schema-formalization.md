# ADR-0027: Training Phase Schemas for Hierarchical Multi-Head Models

- **Status:** Accepted
- **Date:** 2026-04-22

> Only the **Top-Down Baseline Phase Schema** will be implemented initially.
> Other schemas are documented and deferred for future implementation.

## Context

The training system supports multi-phase execution where each phase
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

As part of an upcoming runner refactor (towards a generator-based design),
it is necessary to clarify and stabilize the **phase abstractions**
independently of execution mechanics.

## Problem Statement

A flexible phase system enables a wide range of curricula, but introduces:

- Increased cognitive and control-flow complexity in the runner
- Ambiguity around which phase schemas are first-class and supported
- Difficulty reasoning about default behavior, especially under HPO

A clear set of **canonical phase schemas** is needed to:
- Establish sensible defaults
- Bound system complexity
- Enable future extensions without refactoring the runner again

## Considered Phase Schemas

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

---

### 2. Soft Hierarchy (Overlapping Phases)

**Description**
A smoother curriculum where parent and child heads are active together,
but their influence is controlled via loss weighting.

**Phases**
1. Parent-dominant joint training
2. Child-dominant joint training
3. Balanced joint fine-tuning

**Characteristics**
- Reduced representation drift at phase boundaries
- Requires careful loss-weight configuration
- Harder to inspect than strict freezing

---

### 3. Conditional Child Activation

**Description**
Child heads are only trained on samples belonging to their associated
parent class.

**Phases**
1. Parent-only training
2. Conditional child training (parent frozen or active)
3. Relaxed joint training

**Characteristics**
- Limits gradient noise from irrelevant child classes
- Strong alignment with hierarchical semantics
- Requires conditional routing logic in the training loop

---

### 4. Imbalance-Aware Curriculum

**Description**
A curriculum that progressively introduces child heads based on class
frequency, combined with stronger logit adjustment for rare classes.

**Phases**
1. Parent-only training
2. Frequent child heads
3. Rare child heads (stronger logit adjustment)
4. Joint fine-tuning with reduced bias correction

**Characteristics**
- Explicitly targets long-tailed distributions
- Increased operational complexity
- Requires dataset-level statistics and policy decisions

## Decision

- **The Top-Down Baseline Phase Schema** will be implemented as the
  **default and only supported schema** in the initial phase system.
- A **non-hierarchical default schema** (single-phase or fully joint
  training) will be used when no parent–child structure is configured
  upstream.
- The remaining schemas are **documented but deferred**, and are not part
  of the initial implementation.

This decision prioritizes:
- Clarity and inspectability
- Minimal runner complexity during refactoring
- A well-understood and widely applicable training curriculum

## Consequences

### Positive
- Simplifies the runner refactor by reducing branching logic
- Establishes a clear mental model for phase transitions
- Provides a stable baseline for future extensions and experimentation

### Negative
- More advanced curricula are not immediately available
- Some datasets may benefit from deferred schemas
- Future implementations will require revisiting this ADR

## Follow-Ups

- Implement Top-Down Baseline phase configuration in `phase.py`
- Integrate hierarchy-aware defaults via upstream configuration
- Revisit and revise this ADR once the generator-based runner is stable