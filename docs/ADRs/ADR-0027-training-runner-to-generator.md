# ADR-0027: Refactor Training Runner to Generator-Based Execution for Optuna Pruning

- **Status:** Proposed
- **Date:** 2026-04-22

---

## Context

The current training architecture encapsulates the full training lifecycle within a runner abstraction designed primarily for CLI-driven workflows. Each invocation executes a complete session and returns a final scalar metric.

This design has the following characteristics:

* Strong encapsulation of training, evaluation, scheduling, and early stopping
* A single entry point (`fit()`-style execution) that hides intermediate progress
* Callback system oriented toward side effects (e.g., logging, persistence)

While effective for CLI pipelines, this model creates a limitation for hyperparameter optimization:

> A trial is treated as an opaque function with only a final output, preventing access to intermediate metrics.

As a result:

* Pruning strategies in Optuna cannot operate
* Intermediate performance signals are not externally observable
* Sweep logic is forced to treat training as a black box

This misalignment becomes critical as the sweep layer requires fine-grained control over trial execution.

---

## Decision

Refactor the training runner to a **generator-based execution model** that yields structured intermediate results throughout the training process.

The runner will:

* Expose incremental progress (e.g., per epoch or phase step)
* Retain internal control over training logic (e.g., early stopping, scheduling)
* Allow external consumers (e.g., sweep layer) to drive execution

---

## Design Overview

### Execution Model

Replace monolithic execution with a streaming interface:

* The runner produces a sequence of structured step outputs
* Each step represents a meaningful unit of progress (e.g., epoch)
* Execution control shifts to the caller

### Step Interface

Each yielded step should contain:

* Progress identifiers (epoch, phase)
* Training and validation metrics (normalized structure)
* Internal state indicators (e.g., best model flag)
* Early stopping signals (e.g., convergence reached)

This establishes a consistent and programmatically accessible contract between the runner and external systems.

---

## Integration Strategy

### CLI Workflows

Maintain backward compatibility by consuming the generator internally:

* CLI continues to execute full training sessions
* No behavioral change for standard usage
* Generator is abstracted away from end users

### Sweep Layer

Update the sweep objective to:

* Iterate over runner outputs
* Report intermediate metrics to Optuna
* Trigger pruning decisions based on observed performance

This enables full support for pruning strategies without modifying core training logic.

---

## Rationale

### Architectural Alignment

The generator model aligns the runner with its actual role:

> A producer of training state over time, rather than a terminal computation.

### Separation of Concerns

* Runner: responsible for training progression
* Sweep layer: responsible for optimization decisions
* No direct dependency between the two

### Flexibility

This design enables:

* Early termination via pruning
* External monitoring and control
* Reuse across different orchestration contexts

---

## Consequences

### Positive

* Enables Optuna pruning using intermediate metrics
* Improves observability of training dynamics
* Decouples training execution from optimization logic
* Enhances testability through step-wise inspection
* Provides a foundation for more advanced control flows

### Negative

* Requires refactoring of the runner interface
* Introduces a shift in control flow semantics
* May require updates to existing integrations relying on current behavior

---

## Additional Improvements

### 1. Simplify Runner Responsibilities

Current runner responsibilities are broad and intertwined. Refactor to:

* Isolate core execution logic from side effects
* Move logging, checkpointing, and reporting into hook/callback layers

---

### 2. Normalize Metric Structure

Unify metric outputs into a consistent schema:

* Flatten nested structures
* Use standardized naming conventions
* Ensure compatibility across trainer, evaluator, and sweep layers

---

### 3. Improve State Management

Reduce reliance on mutable shared state:

* Keep internal state encapsulated within the runner
* Expose only immutable step outputs
* Avoid implicit dependencies via callbacks

---

### 4. Make Early Stopping Explicit

Surface early stopping decisions in the step output:

* Indicate when and why training stops
* Differentiate convergence from external interruption (e.g., pruning)

---

### 5. Re-evaluate Phase Abstraction

If phases introduce unnecessary complexity:

* Consider flattening execution into a single loop with phase annotations
* Reduce nested control structures

---

### 6. Introduce Controlled Execution Modes

Support partial execution for debugging and testing:

* Limit number of steps
* Enable deterministic short runs

---

## Migration Plan

1. Introduce generator-based execution alongside existing interface
2. Internally adapt existing execution method to consume the generator
3. Update sweep layer to use generator interface
4. Gradually refactor dependent components
5. Deprecate monolithic execution entry point once migration is complete

---

## Summary

The current runner design reflects historical CLI requirements but constrains modern optimization workflows. Transitioning to a generator-based model resolves this limitation by exposing training as a sequence of observable states.

This change enables pruning, improves modularity, and positions the system for more flexible orchestration patterns without introducing tight coupling between training and optimization layers.
