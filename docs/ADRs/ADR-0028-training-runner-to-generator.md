# ADR-0028: Refactor Training Runner to Generator-Based Execution for Optuna Pruning

- **Status:** Proposed
- **Date:** 2026-04-22

---

## Context

The current training architecture encapsulates the full training lifecycle within
a runner abstraction designed primarily for CLI-driven workflows. Each invocation
executes a complete session and returns a final scalar metric.

This design has the following characteristics:

* Strong encapsulation of training, evaluation, scheduling, and early stopping
* A single entry point (`fit()`-style execution) that hides intermediate progress
* Callback system oriented toward side effects (e.g., logging, persistence)

While effective for CLI pipelines, this model creates a limitation for hyperparameter
optimization:

> A trial is treated as an opaque function with only a final output, preventing
> access to intermediate metrics.

As a result:

* Pruning strategies in Optuna cannot operate
* Intermediate performance signals are not externally observable
* Sweep logic is forced to treat training as a black box

This misalignment becomes critical as the sweep layer requires fine-grained control
over trial execution.

The training system already includes a **mature, explicit phase abstraction** used
to structure multi-stage curricula (e.g., head activation schemas, logit adjustment,
learning rate scaling). The runner is responsible only for *executing* these phases,
not defining or interpreting their semantics.

---

## Decision

Refactor the training runner to a **generator-based execution model** that yields
structured intermediate results throughout the training process.

The runner will:

* Expose incremental progress (e.g., per epoch within a phase)
* Respect phase boundaries and configuration defined externally
* Retain internal control over training logic (e.g., early stopping, scheduling)
* Allow external consumers (e.g., sweep layer) to drive execution flow

This refactor explicitly preserves the existing phase model and treats it as a
stable input to runner execution.

---

## Design Overview

### Execution Model

Replace monolithic execution with a streaming interface:

* The runner produces a sequence of structured step outputs
* Each step represents a meaningful unit of progress (e.g., epoch)
* Steps are annotated with phase identifiers
* Execution control shifts to the caller

The runner becomes an **iterator over training state**, rather than a terminal
procedure.

---

### Step Interface

Each yielded step should contain:

* Phase identifiers (phase name, phase index)
* Progress markers (epoch, global step)
* Training and validation metrics (normalized structure)
* Internal state indicators (e.g., best-model-so-far)
* Early stopping signals (e.g., convergence reached)

Phase transitions are observable through step metadata, not implicit control flow.

---

## Integration Strategy

### CLI Workflows

Maintain backward compatibility by internally consuming the generator:

* CLI continues to execute full training sessions
* Phase sequencing and behavior remain unchanged
* Generator mechanics remain invisible to end users

---

### Sweep Layer

Update the sweep objective to:

* Iterate over runner outputs
* Report intermediate metrics to Optuna
* Use phase-aware step metadata to inform pruning strategies

This enables pruning decisions at **epoch-level or phase-level granularity**
without modifying phase definitions or training semantics.

---

## Rationale

### Architectural Alignment

The generator model reflects the runner’s true role:

> A producer of training state over time, structured by an explicitly defined
> phase schema.

---

### Separation of Concerns

* **Phase system:** defines curriculum structure and training intent
* **Runner:** executes phases and emits progress
* **Sweep layer:** makes optimization and pruning decisions

Each layer has a single, well-scoped responsibility.

---

### Flexibility

This design enables:

* Early termination via pruning
* External monitoring and inspection of phase behavior
* Reuse of the runner in different orchestration contexts
* Deterministic partial execution for testing and debugging

---

## Consequences

### Positive

* Enables Optuna pruning using intermediate metrics
* Improves observability across epochs and phases
* Decouples training execution from optimization logic
* Preserves stability of the existing phase abstraction
* Improves testability through step-wise inspection

---

### Negative

* Requires refactoring of the runner interface
* Introduces a new execution paradigm that callers must understand
* Requires updates to integrations that expect monolithic execution

---

## Additional Improvements

### 1. Simplify Runner Responsibilities

Refactor the runner to:

* Isolate core execution logic
* Delegate logging, checkpointing, and reporting to hooks or callbacks
* Avoid embedding policy decisions in execution code

---

### 2. Normalize Metric Structure

Unify metric outputs into a consistent schema:

* Flatten nested metric structures
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
* Distinguish convergence from external interruption (e.g., pruning)

---

### 5. Introduce Controlled Execution Modes

Support partial execution for debugging and testing:

* Limit number of yielded steps
* Enable deterministic short runs
* Facilitate unit and integration testing of runner behavior

---

## Migration Plan

1. Introduce generator-based execution alongside existing interface
2. Internally adapt the current execution method to consume the generator
3. Update sweep layer to use generator interface
4. Gradually refactor dependent components
5. Deprecate monolithic execution entry point once migration is complete

---

## Summary

The existing runner design reflects historical CLI requirements but constrains
modern optimization workflows. Transitioning to a generator-based execution model
exposes training as a sequence of observable, phase-annotated states.

This refactor enables pruning, improves modularity, and cleanly integrates with
the existing phase system without redefining or weakening its semantics.