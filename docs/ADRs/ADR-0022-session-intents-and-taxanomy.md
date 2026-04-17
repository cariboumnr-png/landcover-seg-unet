# ADR-0022: Session Intents and Supported Session Taxonomy

- **Status:** Accepted
- **Date:** 2026-04-16
- **Related ADRs:** ADR-0021

---

## Context

ADR-0021 established a **session-first construction boundary**, centralizing
runtime assembly (components, state, callbacks, engines, optional runner)
behind the session module.

With that boundary in place, the codebase now supports multiple *ways of using*
a session:

- full end-to-end training with curriculum phases and a runner
- minimal overfit workflows that bypass the runner
- evaluation and inference logic embedded in evaluators and callbacks

However, these usages are currently **implicit** and expressed only through
pipeline wiring and configuration conventions. There is no explicit statement
of which session usages are supported, nor what guarantees each usage provides.

This ADR introduces a **session taxonomy** to clarify intent without prematurely
freezing APIs or unifying orchestration logic.

---

## Decision

### Introduce the concept of *session intent*

A **session intent** describes *why* a session is constructed and *what
guarantees it provides*, not how it is implemented.

The system formally recognizes the following session intents:

- **Training session**
- **Overfit session**
- **Evaluation-only session** (planned, not yet first-class)

These intents are conceptual and may be expressed via configuration,
pipeline selection, or documentation. They do **not** require distinct
session classes or factories at this stage.

---

### Supported session intents

#### Training session
A training session is the default, full-featured execution mode.

**Characteristics:**
- Uses prepared dataset artifacts
- Constructs trainers and evaluators
- May include a multi-phase runner
- Performs training and validation
- Produces checkpoints, metrics, and logs

**Guarantees:**
- Deterministic runtime construction
- Reproducible execution given identical artifacts and config
- Session-owned lifecycle and state

---

#### Overfit session
An overfit session is a diagnostic and validation tool.

**Characteristics:**
- Operates on a minimal dataset (often a single block)
- Typically disables regularization and augmentation
- May bypass the runner and phases
- Prioritizes speed and determinism

**Guarantees:**
- End-to-end stack validation
- Fast failure detection for data, model, or loss issues
- No mutation of foundation or experiment artifacts

Overfit sessions are **not** intended for model selection or reporting.

---

#### Evaluation-only session (deferred)
An evaluation-only session is planned but not yet fully modeled.

**Intended characteristics:**
- Consumes trained model checkpoints
- Runs inference and metric computation only
- Produces evaluation artifacts and reports
- Does not perform training or optimization

This intent is acknowledged but deferred until evaluation workflows
become first-class.

---

### What this ADR does *not* do

This ADR intentionally does **not**:

- Introduce new session classes or inheritance hierarchies
- Define a unified `build_session(...)` API
- Unify all pipelines under a single execution model
- Introduce experiment-level or hyperparameter orchestration

Those concerns are deferred to future ADRs.

---

## Consequences

### Positive
- Makes existing session usage explicit and intentional
- Clarifies the role of overfit workflows
- Establishes shared vocabulary for future design discussions
- Prevents premature abstraction of session execution modes

### Costs
- Some duplication remains in session construction paths
- Session intent is not yet enforced programmatically

These costs are accepted to preserve flexibility while the session surface
area continues to evolve.

---

## Follow-up Work (Deferred)

The following items are explicitly deferred and expected to be addressed
by future ADRs:

- Define a **public session construction API** once session intents and
  guarantees are fully understood
- Clarify and formalize **evaluation-only sessions** and reporting outputs
- Introduce an **experiment or study layer** for hyperparameter sweeps,
  model comparison, and final model selection
- Further simplify the session factory surface area after orchestration
  patterns converge

---

## Summary

Sessions are now recognized as a **multi-intent execution boundary** rather
than a single monolithic runtime. This ADR formalizes supported session
intents while deliberately postponing API consolidation and higher-level
orchestration until sufficient usage patterns are established.
``