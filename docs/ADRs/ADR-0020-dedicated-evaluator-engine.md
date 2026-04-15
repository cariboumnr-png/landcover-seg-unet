# ADR-0020: Dedicated Evaluator Engine and Policy Model

- **Status:** Proposed
- **Date:** 2026-04-15

## Context

Following ADR-0019, batch-level execution mechanics have been extracted into a
shared, policy-agnostic execution core. `MultiHeadTrainer` now focuses purely on
training policy and epoch-level orchestration.

The codebase requires a **non-training session engine** that can:

- run validation-only or evaluation-only workflows
- share execution correctness with training
- interpret batch results without optimizer or gradient semantics
- provide a stable anchor for metric aggregation and logging

Currently, evaluation behavior is either:

- embedded in `MultiHeadTrainer`, or
- implicitly coupled to training-centric assumptions

This coupling obscures intent and complicates logging, reuse, and extension.

## Decision

We will introduce a **Dedicated Evaluator Engine** that defines evaluation policy
on top of the shared execution core, analogous to how `MultiHeadTrainer` defines
training policy.

### 1. Evaluator as a policy layer

The evaluator will:

- orchestrate epoch-level evaluation workflows
- control evaluation-specific head configuration
- aggregate batch-level metrics into epoch summaries
- manage evaluation-specific runtime state interpretation
- emit lifecycle callbacks as semantic markers

The evaluator will **not**:

- own batch execution logic
- perform backward passes or optimization steps
- manage training state or early stopping logic

### 2. Shared execution, divergent policy

Both trainer and evaluator will:

- consume the same execution core
- operate on the same `RuntimeState` structure
- emit callbacks with phase-specific semantics

They will differ only in **policy interpretation**, not execution mechanics.

### 3. Logging consolidation point

With trainer and evaluator symmetry established, logging will be:

- redefined as a **policy-level concern**
- implemented consistently across trainer and evaluator
- driven by lifecycle callbacks and finalized epoch summaries

This explicitly resolves the logging disruption introduced during ADR-0019 by
providing a single architectural layer where logging semantics belong.

## Explicit Non-Decisions

This ADR does **not**:

- define a public CLI or UX for evaluation
- mandate specific logging backends or formats
- redesign callback interfaces
- define ranking, comparison, or benchmarking workflows

Those concerns may be addressed in future ADRs as needed.

## Consequences

### Positive

- Clear separation between training and evaluation concerns
- Elimination of training-centric assumptions in evaluation code
- Natural, centralized ownership for logging and metric aggregation
- Improved architectural symmetry and clarity

### Costs

- Additional engine abstraction
- Minor duplication of orchestration logic across trainer and evaluator
- Initial implementation overhead

### Risks

- Over-aligning evaluator policy too closely with trainer abstractions
- Allowing evaluator scope to grow beyond evaluation concerns

## Follow-up Work

- Implement evaluator engine using shared execution core
- Refactor existing validation flows to use evaluator
- Reintroduce and consolidate logging at the policy layer
- Document lifecycle expectations for trainer vs evaluator

## Related Code Areas

- `session/engine/core/`
- `session/engine/trainer.py`
- `session/engine/evaluator.py` (new)
- `session/components/callback/*`
- `session/runner/runner.py`
