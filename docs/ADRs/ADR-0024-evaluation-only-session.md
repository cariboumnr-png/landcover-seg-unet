# ADR-0024: Evaluation‑Only Sessions and Result Reporting

- **Status:** Proposed
- **Date:** 2026-04-18
- **Related ADRs:** ADR-0021, ADR-0022, ADR-0023

---

## Context

ADR-0021 established the session as the primary runtime construction boundary.
ADR-0022 clarified that sessions may serve different **intents** (training,
overfit, evaluation-only). ADR-0023 formalized a single public API for session
construction and made orchestration explicitly optional.

At present, evaluation logic exists as part of training workflows (e.g.
validation phases, evaluators attached to trainers), but there is no explicit,
first-class notion of a **standalone evaluation session**.

This creates ambiguity around:
- how trained models should be evaluated independently of training
- how evaluation results should be reported and consumed
- how evaluation fits into future experiment or model selection workflows

This ADR introduces evaluation-only sessions as a distinct *session intent* and
clarifies their responsibilities and boundaries.

---

## Decision

### Introduce evaluation-only sessions as a first-class session intent

An **evaluation-only session** is a session constructed explicitly for the
purpose of evaluating a trained model, without performing training or
optimization.

Evaluation-only sessions are built using the same public session construction
API (`build_session(...)`), but assemble a restricted runtime tailored to
evaluation.

---

### Characteristics of an evaluation-only session

An evaluation-only session:

- consumes prepared dataset artifacts
- consumes one or more trained model checkpoints
- constructs evaluators and required runtime state
- executes inference and metric computation
- produces evaluation artifacts and reports

An evaluation-only session **does not**:

- perform training or gradient updates
- construct training orchestration
- modify model parameters
- produce checkpoints

---

### Relationship to training sessions

Training sessions may include *embedded evaluation* (e.g. validation during
training), but this is not equivalent to an evaluation-only session.

Key distinctions:

- Training evaluation supports *optimization and monitoring*
- Evaluation-only sessions support *assessment and reporting*
- Evaluation-only sessions must be runnable independently of training

This separation avoids overloading training workflows with reporting or
selection responsibilities.

---

### Result reporting responsibility

Evaluation-only sessions are responsible for producing **explicit evaluation
outputs**, such as:

- aggregated metrics
- per-head or per-class performance summaries
- optional inference artifacts (e.g. previews, maps)

The **format and persistence** of these outputs are session-level concerns, but
their interpretation (e.g. model comparison or selection) is explicitly out of
scope.

---

## Consequences

### Positive

- Clarifies the role of evaluation in the system
- Enables consistent, reproducible post-training assessment
- Decouples evaluation from training orchestration
- Establishes a clean boundary for future experiment-level logic

### Costs

- Requires explicit modeling of evaluation-only runtime paths
- Adds conceptual surface area to the session taxonomy

These costs are accepted to prevent evaluation logic from being implicitly
embedded in training workflows.

---

## What this ADR does *not* do

This ADR intentionally does **not**:

- introduce an experiment or study abstraction
- define hyperparameter sweep or model selection logic
- prescribe a specific evaluation reporting format
- require new orchestration strategies

Those concerns are deferred to future ADRs.

---

## Follow-up Work (Deferred)

The following items are expected to be addressed by future ADRs:

- Introduce an **experiment / study layer** to compare results across sessions
- Define a standardized evaluation result schema for downstream consumers
- Decide whether evaluation orchestration requires a dedicated strategy
- Integrate evaluation-only sessions into CLI workflows

---

## Summary

This ADR establishes **evaluation-only sessions** as a first-class session
intent, separating model assessment and reporting from training and
optimization. This prepares the system for future experiment-level workflows
while preserving the session as a single-run execution boundary.
