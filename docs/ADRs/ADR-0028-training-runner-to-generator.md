# ADR-0028: Generator-Based Training Runner for Observable, Prunable Execution

- **Status:** Accepted
- **Date:** 2026-04-29
- **Supersedes:** Monolithic `TrainingRunner` execution model

---

## Context

The original training architecture encapsulated the full training lifecycle inside a
monolithic runner designed primarily for CLI-driven workflows. Each invocation executed
an entire session and returned only a final scalar metric.

This design had several properties:

- Strong encapsulation of training, evaluation, scheduling, and early stopping
- A single, terminal `fit()` / `execute()` entry point
- Callbacks oriented toward side effects (logging, persistence)
- No externally visible intermediate progress

While effective for CLI pipelines, this model fundamentally limited hyperparameter
optimization and external orchestration:

> Training was treated as an opaque function with only a final output.

As a result:

- Optuna pruning could not operate
- Intermediate epoch- and phase-level metrics were inaccessible
- Sweep logic was forced to treat training as a black box

At the same time, the system already employed a **mature, explicit phase abstraction**
used to define curricula (e.g. multi-stage head activation, scheduling, and scaling).
The runner’s responsibility was *execution*, not semantic definition of phases.

This mismatch between execution and orchestration requirements motivated a structural
refactor.

---

## Decision (Implemented)

The training runner **has been refactored into a generator-based execution model** that
exposes training progress as a stream of immutable, structured steps.

Instead of executing training to completion and returning a single value, the runner now:

- Yields one structured step per completed epoch
- Preserves explicit phase boundaries supplied by configuration
- Retains internal responsibility for training logic and state
- Allows external consumers (CLI, sweeps, studies) to control execution by consuming
  the generator

The runner now acts as an **iterator over training state**, not a terminal procedure.

---

## Design Overview

### Execution Model

The monolithic training runner has been replaced by generator-based orchestration:

- Training progress is emitted incrementally
- Each yielded unit represents a completed epoch
- Phase transitions are explicit and observable
- Termination is signaled through structured metadata, not implicit control flow

Execution control resides with the caller, which may:
- Consume the full run
- Stop early (e.g. pruning)
- Observe or persist intermediate results

---

### Public Step Interface

All training progress is exposed through a single public contract:

**`TrainingSessionStep`** (immutable, frozen dataclass)

Each step contains:

- **Phase identity**
  - `phase_name`
  - `phase_index`
- **Progress markers**
  - `epoch`
  - `global_epoch`
- **Metrics**
  - `objective_name`
  - `objective_value`
  - raw `EpochResults` (training + validation)
- **Best-so-far tracking**
  - `best_value_so_far`
  - `best_epoch_so_far`
  - `is_best_epoch`
- **Termination signals**
  - `is_phase_end`
  - `is_run_end`
  - `stop_reason`

This structure is the **sole externally observable execution unit** for training.

---

## Architectural Realignment

### Separation of Concerns (Enforced)

The refactor enforces strict layering:

- **Batch / Epoch Execution**
  - Executes exactly one epoch
  - No phases, no scheduling, no stopping logic
  - Returns aggregated epoch metrics only

- **Orchestration Policies**
  - Manage epoch sequencing within a phase
  - Track best metrics and early stopping
  - Emit structured, immutable events

- **Runners**
  - Translate orchestration events into `TrainingSessionStep`
  - Expose progress exclusively as a generator
  - Do not embed CLI or sweep logic

- **External Consumers (CLI, Sweep, Study)**
  - Consume step streams
  - Decide when and how long to run
  - Report, prune, or terminate based on observed state

Each layer now has a single, well-scoped responsibility.

---

## Phase Model Preservation

The existing phase abstraction has been **preserved and clarified**, not replaced.

- Phases remain pure, declarative inputs:
  - name
  - number of epochs
  - head configuration
- Phases define *what* to train and *for how long*
- Runners and policies define *how* execution proceeds

No training semantics were pushed into phase definitions.

---

## Sweep and Optuna Integration

The sweep subsystem has been updated to consume runner generators directly.

- A sweep trial now builds a **step-producing runner**
- The objective iterates over `TrainingSessionStep`
- Intermediate metrics are reported to Optuna
- Pruning decisions operate at epoch or phase granularity

Training is no longer treated as a black box.

This directly resolves the original limitation that motivated the ADR.

---

## CLI Workflows

CLI workflows remain fully supported.

- CLI pipelines internally consume the same generator interface
- Full end-to-end runs behave identically from a user perspective
- Generator mechanics remain invisible to CLI users

Backward compatibility is achieved through internal consumption of the step stream
rather than duplicating execution logic.

---

## Explicit Termination and Early Stopping

Early stopping and termination are now explicit and observable:

- Stopping causes are surfaced in step metadata
- Convergence, pruning, and external interruption are distinguishable
- Termination is modeled as structured data, not implicit state

This improves debugging, testing, and automated decision-making.

---

## Consequences

### Positive

- Enables Optuna pruning using intermediate metrics
- Makes training progress externally observable
- Decouples execution from optimization logic
- Preserves and strengthens the phase abstraction
- Improves testability via step-wise inspection
- Enables alternative orchestration contexts

### Costs

- Significant refactor of runner and orchestration internals
- Introduction of generator-based execution as the primary model
- Required mechanical updates to CLI and sweep integration

These costs have been paid and absorbed by the current architecture.

---

## Summary

The training execution system has been successfully refactored from a monolithic,
terminal runner into a generator-based orchestration model.

Training is now exposed as a sequence of immutable, phase-aware steps that can be
observed, inspected, and controlled by external consumers without compromising
internal execution semantics.

This change enables pruning, improves modularity, and aligns the execution model
with modern optimization and orchestration requirements while preserving the
existing phase-based training design.