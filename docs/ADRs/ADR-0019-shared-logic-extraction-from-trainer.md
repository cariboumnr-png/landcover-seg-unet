# ADR-0019: Shared Execution Core Extraction from Trainer

- **Status:** Accepted
- **Date:** 2026-04-15

## Context

The current session architecture distinguishes between:

- **Runner**: orchestrates multi-phase execution (curriculum, checkpoints,
  resumption, phase transitions)
- **MultiHeadTrainer**: executes training, validation, and inference logic
  using callbacks, runtime state, and trainer components

Prior to this change, `MultiHeadTrainer` mixed two responsibilities:

1. **Generic execution mechanics**
   - batch parsing
   - forward / loss / backward sequencing
   - AMP and device context management
   - callback invocation order
   - runtime state mutation

2. **Training-specific policy**
   - optimization steps
   - gradient clipping
   - head activation / freezing semantics
   - metric tracking and patience logic

At the same time, the codebase already anticipated reuse of execution
logic beyond training:

- Validation and inference follow the same batch lifecycle with different
  policies
- A future evaluator engine is planned (ADR-0020)
- Session callbacks are already phase-scoped (train / val / infer)

Keeping execution mechanics embedded inside `MultiHeadTrainer` tightly
couples these flows to a single trainer implementation and complicates
reuse, testing, and extension.

## Decision

We extracted a **shared execution core** from `MultiHeadTrainer` and treat it
as a reusable, policy-agnostic component responsible for **batch-level execution**.

### 1. Shared execution core

A new internal execution component:

- owns the batch lifecycle (parse → forward → loss / metrics → state mutation)
- manages precision and evaluation contexts (AMP, train/eval modes)
- mutates `RuntimeState` deterministically
- performs no optimizer, scheduler, or epoch-level logic

The execution core is deliberately **unaware of**:

- curriculum and phase scheduling
- optimization policy
- checkpointing
- early stopping
- logging cadence or aggregation

### 2. Re-scoped MultiHeadTrainer

`MultiHeadTrainer` now acts as a **training policy layer**:

- defines epoch-level workflows (train / validate / infer)
- owns optimizer, scheduler, and gradient management
- controls active/frozen heads and class exclusions
- aggregates epoch-level statistics and best-metric tracking
- emits semantic lifecycle callbacks

Batch-level execution is delegated entirely to the shared core.

### 3. Architectural preparation

This separation establishes a stable foundation for:

- training
- validation-only evaluators
- inference-only evaluators

and directly enables ADR-0020 (Dedicated Evaluator Engine).


## Explicit Non-Decisions

This ADR does **not**:

- Define the public API of a future evaluator engine
- Change callback interfaces or semantics
- Alter `Runner` responsibilities
- Introduce distributed or asynchronous execution
- Redesign optimization or metric policies

Those concerns are deferred to follow-up ADRs.

## Consequences

### Positive

- Clear separation between **execution mechanics** and **training policy**
- Easier implementation of evaluator and inference engines
- Reduced duplication across session flows
- Improved testability of batch-level execution

### Costs

- Requires refactoring core trainer logic
- Introduces an additional abstraction layer
- Demands careful definition of execution-core boundaries

### Known Gaps / Transitional State

As a consequence of extracting execution mechanics, existing trainer logging
paths are temporarily disrupted.

This is an intentional transitional state. Logging behavior will be revisited
once the dedicated evaluator engine (ADR-0020) is introduced, so that logging
can be re-anchored at a consistent policy layer shared across trainer and
evaluator implementations.

## Follow-up Work

- ADR-0020: Dedicated Evaluator Engine and Callback Model
- Refactor `MultiHeadTrainer` to delegate execution sequencing
- Introduce a minimal execution-core interface scoped to batch lifecycle only

## Related Code Areas

- `session/engine/trainer/core.py`
- `session/engine/trainer/state.py`
- `session/common/trainer_engine.py`
- `session/components/callback/*`
- `session/runner/runner.py`
