# ADR-0025: Session Construction Contract and Evaluation Artifact Boundary

- **Status:** Accepted
- **Date:** 2026-04-20
- **Related ADRs:** ADR-0023, ADR-0024

---

## Context

ADR-0023 established a single public boundary for session construction.
ADR-0024 established evaluation-only execution as a first-class intent.

As the project evolved, we clarified several architectural boundaries:

- session construction needed a public typed contract
- evaluation execution needed to be execution-ready at the runtime
  boundary
- evaluation outputs needed to be treated as explicit artifacts rather
  than only as metadata-side effects
- the workflow and command surface needed to reflect these boundaries
  more clearly

This ADR records the decisions we have now adopted.

---

## Decision

### 1. We formalized the public session construction contract

Session construction is now treated as a public architectural boundary.

We exposed public types for:

- the minimum session configuration shape required to construct a
  runtime session
- the executable runtime objects returned by session construction

The returned object represents execution-capable session components,
rather than experiment outputs or completed results. It serves as the
typed container for objects assembled at the session boundary, such as:

- evaluator
- optional trainer
- optional training runner

This hardens the boundary introduced in ADR-0023 and gives training and
evaluation pipelines a stable typed contract for session construction.

---

### 2. We kept model preparation outside the session boundary

Session construction accepts a model as input and treats that model as
already prepared for the intended execution path.

Checkpoint loading, weight restoration, and comparable model preparation
steps remain outside the session factory. These concerns belong to the
higher-level orchestration boundary that prepares the model before
runtime assembly.

What we require at the runtime boundary is straightforward:

- when a session is returned for execution, it is execution-ready for
  its declared intent
- evaluation does not depend on post-session mutation of returned
  runtime objects before execution can begin

This preserves the intended separation of concerns:

- model preparation remains outside session construction
- runtime assembly remains inside session construction

---

### 3. We established explicit base evaluation artifacts

Evaluation-only execution now produces explicit persisted outputs.

Evaluation artifacts are no longer treated only as raw logs embedded in
metadata. Instead, evaluation persists first-class runtime outputs that
can be consumed independently of the session log stream.

At minimum, the evaluation boundary now supports:

- machine-readable evaluation metrics and summaries
- explicit evaluation artifact persistence separate from metadata
- preview-style inference exports when supported by the execution path

This ADR defines the base evaluation artifact boundary only. It does not
attempt to formalize richer comparative reporting, study-wide
aggregation, or advanced visual analysis products.

---

### 4. We synchronized the workflow documentation with the runtime boundary

The main project documentation now reflects the runtime architecture more
directly.

In particular, the workflow documentation distinguishes:

- foundation artifacts
- experiment artifacts
- runtime execution

It also reflects:

- the independent evaluation pipeline
- the session construction boundary as the owner of runtime assembly
- the separation between model preparation and session construction

At this stage, the documentation is intentionally architecture-oriented
rather than a frozen API reference.

---

## Consequences

### Positive

- we now have a single explicit session construction boundary
- training and evaluation pipelines use a public typed session contract
- model preparation is no longer conflated with runtime assembly
- evaluation produces explicit persisted outputs
- the codebase, CLI workflow, and project documentation are better
  aligned

### Trade-offs

- the public session surface is now an architectural contract that must
  be maintained as the project evolves
- evaluation outputs must now be treated as a designed artifact boundary
  rather than an incidental logging side effect
- richer reporting remains intentionally deferred to a future layer

---

## Follow-up Work

We will introduce a dedicated **study layer** in a future ADR.

That future layer is expected to:

- group and organize related runs under a higher-level study or
  experiment boundary
- support richer comparative reporting and aggregated evaluation outputs
  across runs
- provide a clearer downstream handoff contract for derived artifacts and
  summaries
- define the detailed schemas, reporting formats, and artifact
  conventions that are intentionally left open here

---

## Summary

This ADR records the boundary we have now established:

- session construction exposes public typed contracts
- session construction returns execution-capable runtime objects
- model preparation remains outside the session boundary
- evaluation persists explicit base artifacts
- richer study-level reporting is deferred to a future study layer

This keeps the runtime architecture clean, preserves separation of
concerns, and leaves room for a later study/experiment layer to take on
cross-run reporting and aggregation responsibilities.