# ADR-0024: Evaluation-Only Sessions and Result Reporting

- **Status:** Accepted
- **Date:** 2026-04-18
- **Related ADRs:** ADR-0021, ADR-0022, ADR-0023

---

## Context

ADR-0021 established the session as the primary runtime construction
boundary. ADR-0022 introduced a session taxonomy and identified
**evaluation-only** as a planned session intent alongside training and
overfit workflows. ADR-0023 formalized a single public API for session
construction and made orchestration explicitly optional.

Historically, evaluation behavior existed mainly as part of training
workflows (for example, validation inside training), which made it
unclear how trained models should be evaluated independently of
optimization, how evaluation outputs should be reported, and where that
responsibility should live in the architecture.

This ADR formalizes **evaluation-only** as a first-class session intent
and clarifies its scope, responsibilities, and boundaries.

---

## Decision

### Introduce evaluation-only as a first-class session intent

An **evaluation-only session** is a session constructed explicitly for
the purpose of evaluating one or more trained model checkpoints without
performing training or optimization.

Evaluation-only sessions are built through the same public session
construction boundary as other session intents and assemble a restricted
runtime tailored to evaluation.

### Characteristics of an evaluation-only session

An evaluation-only session:

- consumes prepared dataset artifacts
- consumes one or more trained model checkpoints
- constructs evaluators and required runtime state
- executes inference and metric computation
- produces explicit evaluation outputs and reports

An evaluation-only session **does not**:

- perform training or gradient updates
- step optimizers or schedulers for learning
- construct training orchestration solely for evaluation
- modify model parameters
- produce training checkpoints

### Relationship to training sessions

Training sessions may include **embedded evaluation** (for example,
validation during training), but this is not equivalent to an
evaluation-only session.

Key distinctions:

- Training evaluation supports **optimization and monitoring**
- Evaluation-only sessions support **assessment and reporting**
- Evaluation-only sessions must be runnable independently of training

This separation avoids overloading training workflows with responsibilities
that belong to post-training assessment.

### Result reporting responsibility

Evaluation-only sessions are responsible for producing **explicit
evaluation outputs**, such as:

- aggregated metrics
- per-head or per-class performance summaries
- optional inference artifacts (for example, previews or maps)

In the initial implementation, these outputs may be persisted as
session-level metadata, summaries, logs, and optional exported
artifacts. Their interpretation beyond a single session (for example,
model comparison, ranking, or final selection) remains out of scope.

The **format and persistence** of evaluation outputs are session-level
concerns. Standardization of downstream result schemas is deferred.

---

## Consequences

### Positive

- Clarifies the role of evaluation in the system
- Makes evaluation runnable independently of training
- Decouples assessment/reporting from training orchestration
- Establishes a clean boundary for future experiment-level workflows
- Aligns runtime construction with the session taxonomy introduced by
  ADR-0022

### Costs

- Requires explicit modeling of evaluation-only runtime paths
- Adds conceptual surface area to the session taxonomy
- Introduces additional reporting and persistence concerns at the
  session layer

These costs are accepted in order to avoid keeping evaluation logic
implicitly embedded in training workflows.

---

## What this ADR does _not_ do

This ADR intentionally does **not**:

- introduce an experiment or study abstraction
- define hyperparameter sweep or model selection logic
- prescribe a standardized evaluation result schema
- require a distinct session factory separate from the existing session
  construction boundary
- require dedicated orchestration beyond what evaluation-only execution
  actually needs

Those concerns are deferred to future ADRs.

---

## Follow-up Work (Deferred)

The following items remain deferred:

- Introduce an **experiment / study layer** to compare results across
  sessions
- Define a standardized evaluation result schema for downstream
  consumers
- Refine evaluation result persistence and consumption interfaces
- Decide whether evaluation orchestration needs a dedicated strategy
  beyond the current session/runtime model

---

## Summary

This ADR establishes **evaluation-only sessions** as a first-class
session intent, separating model assessment and reporting from training
and optimization while preserving the session as the primary single-run
execution boundary.

It completes the progression started in ADR-0022 by moving
evaluation-only from a planned concept to an explicit, supported session
intent.