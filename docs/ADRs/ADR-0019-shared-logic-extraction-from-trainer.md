# ADR-0019: Shared Execution Core Extraction from Trainer

- **Status:** Proposed
- **Date:** 2026-04-13

## Context

The current session architecture distinguishes between:

- **Runner**: orchestrates multi-phase execution (curriculum, checkpoints,
  resumption, phase transitions)
- **MultiHeadTrainer**: executes training, validation, and inference logic
  using callbacks, runtime state, and trainer components

Although this separation exists conceptually, the `MultiHeadTrainer`
currently mixes two responsibilities:

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

At the same time, the codebase already anticipates reuse of execution
logic beyond training:

- Validation and inference follow the same batch lifecycle with different
  policies
- A future evaluator engine is planned (ADR-0020)
- Session callbacks are already phase-scoped (train / val / infer)

Keeping execution mechanics embedded inside `MultiHeadTrainer` tightly
couples these flows to a single trainer implementation and complicates
reuse, testing, and extension.

## Decision

We will extract a **shared execution core** from `MultiHeadTrainer` and
treat it as a reusable engine responsible for **step-level execution and
callback orchestration**.

Specifically:

### 1. Introduce a shared execution core

- A new internal component (e.g., `ExecutionCore` or equivalent) will:
  - own the batch lifecycle (`parse → forward → loss / metrics → callbacks`)
  - manage context managers (AMP, eval/train modes)
  - mutate `RuntimeState` in a consistent, centralized way
- This core will be **policy-agnostic** and unaware of:
  - curriculum logic
  - checkpointing
  - phase scheduling
  - early stopping criteria

### 2. Re-scope `MultiHeadTrainer`

- `MultiHeadTrainer` becomes a **training policy layer** that:
  - configures optimization behavior
  - supplies training-specific hooks to the execution core
  - wires trainer components, runtime state, and model together
- It delegates execution sequencing to the shared core rather than
  implementing it directly.

### 3. Prepare for multiple session engines

- The shared execution core will be reusable by:
  - training
  - validation-only evaluators
  - inference-only evaluators
- This establishes a stable foundation for ADR-0020
  (Dedicated Evaluator Engine).

## Explicit Non-Decisions

This ADR does **not**:

- Define the public API of a future evaluator engine
- Change callback interfaces or semantics
- Alter `Runner` responsibilities
- Introduce distributed or asynchronous execution
- Redesign optimization or metric policies

Those concerns are deferred to follow-up ADRs.

## Rationale

### Why extract execution mechanics

Execution logic is currently:

- large
- stateful
- difficult to reason about in isolation
- duplicated conceptually across train/val/infer paths

Extracting a shared core:

- centralizes correctness-critical sequencing
- reduces cognitive load in `MultiHeadTrainer`
- makes execution behavior testable without training policy
- enables reuse across future session engines

### Why keep policy in `MultiHeadTrainer`

Training remains special due to:

- optimization steps
- gradient management
- early stopping logic
- curriculum-driven head control

Keeping these as a thin policy layer preserves clarity while avoiding
premature generalization.

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

### Risks

- Over-generalizing the execution core may prematurely constrain future
  features
- Partial extraction may leave responsibility split across layers if not
  completed cleanly

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
