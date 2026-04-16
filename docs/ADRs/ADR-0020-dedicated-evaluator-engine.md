# ADR-0020: Dedicated Evaluator Engine and Policy Model

- **Status:** Accepted
- **Date:** 2026-04-15

## Context

Following ADR-0019, batch-level execution mechanics were extracted into a shared,
policy-agnostic execution core. This refactor reduced duplication and clarified
execution semantics, but left evaluation logic either embedded in
`MultiHeadTrainer` or implicitly coupled to training-centric assumptions.

As training and evaluation workflows continued to diverge, this coupling
obscured intent, complicated logging, and limited reuse of evaluation-only
workflows such as validation-only runs, inference, or benchmarking.

## Decision

### Dedicated Evaluator Engine

The system now **adopts a Dedicated Evaluator Engine** that defines evaluation
policy on top of the shared execution core, analogous to how
`MultiHeadTrainer` defines training policy.

The evaluator:

- Orchestrates epoch-level evaluation and inference workflows
- Controls evaluation-time head configuration
- Aggregates batch-level outputs into epoch summaries
- Interprets runtime state without training assumptions
- Emits lifecycle callbacks as semantic markers for logging and observability

The evaluator explicitly does **not**:

- Own batch execution logic
- Perform backward passes or optimization
- Manage training state, early stopping, or schedules

### Shared execution, divergent policy

Both trainer and evaluator:

- Consume the same shared `BatchExecutionEngine`
- Operate on the same unified runtime `State`
- Share component contracts and callback wiring

They differ **only** in policy interpretation and orchestration, not
execution mechanics.

### Logging consolidation

With trainer/evaluator symmetry established:

- Logging is treated as a **policy-level concern**
- Callbacks observe lifecycle signals emitted by engines
- Epoch summaries are finalized by policy layers, not execution code

This resolves the logging ambiguity introduced during ADR-0019 by providing
a single architectural layer where logging semantics belong.

## Results

The following outcomes have been achieved:

- Evaluation logic has been fully removed from `MultiHeadTrainer`
- A new `MultiHeadEvaluator` engine exists as a first-class policy layer
- Execution core, runtime state, and component boundaries are unified
- Callbacks and logging now depend on policy-emitted lifecycle signals
- The Runner remains training-focused but is evaluator-ready without refactor

This work establishes a clean and extensible architectural foundation for
evaluation-only sessions and future workflows.

## Non-Goals (Confirmed)

This ADR did **not** introduce:

- A public CLI or UX for evaluation sessions
- Prescriptive logging backends or formats
- Callback interface redesign
- Ranking, benchmarking, or comparison workflows

These remain intentionally out of scope.

## Consequences

### Positive

- Clear separation between training and evaluation concerns
- Elimination of training-centric assumptions from evaluation
- Centralized ownership of logging and metric aggregation
- Improved architectural symmetry and long-term maintainability

### Trade-offs

- Additional engine abstraction
- Some duplication of orchestration structure between trainer and evaluator

These costs were deemed acceptable given the clarity gained.

## Current State

The `Runner` currently functions as an end-to-end **training orchestration
runner**. Evaluation capability is now architecturally enabled but will be
explicitly integrated in subsequent work without restructuring existing engines.

## Next ADRs

Per ADR-0018, this ADR is followed by:

- **ADR-0021: Session Component and Runtime-State Boundaries**

ADR-0021 will formalize component ownership, runtime-state lifetimes, and
session-level API boundaries building on the trainer/evaluator split.

## Related Code Areas

- `session/engine/core/`
- `session/engine/base.py`
- `session/engine/trainer.py`
- `session/engine/evaluator.py`
- `session/components/callback/*`
- `session/runner/runner.py`
