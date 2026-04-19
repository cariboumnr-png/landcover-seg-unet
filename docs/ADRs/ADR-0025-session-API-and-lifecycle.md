# ADR-0025: Session API Hardening and Evaluation Lifecycle

- **Status:** Proposed
- **Date:** 2026-04-19
- **Related ADRs:** ADR-0023, ADR-0024

---

## Context

ADR-0023 establisheda single public boundary for session construction.

ADR-0024 formalized evaluation-only sessions as a first-class intent.

Currently, structural gaps remain in the execution boundary of the `landcover-seg-unet` project.

- The `build_session` entrypoint returns a private `_Session` object.

- Checkpoint loading happens outside the session builder, leaving the returned engine unready for immediate execution.

- Evaluation results are currently dumped as raw logs into the metadata payload,failing to provide
explicit inference artifacts.

This ADR closes these gaps to fully harden the public API.

---

## Decision

### 1. Formalize the Public Return Object

- The private `_Session` dataclass must be formally exposed as `Session` or `SessionResult`.

- The internal `_SessionConfig` protocol must also be publicly exposed.

- This guarantees rigid typing for the evaluation and training pipelines.

### 2. Shift Checkpoint Ingestion

- Model weight ingestion must occur prior to or strictly during session construction.

- A constructed session object must be strictly execution-ready upon return.

### 3. Formalize Evaluation Artifacts

- Evaluation-only sessions must generate explicit outputs.

- This includes visual confidence maps and spatial performance summaries.

- Relying solely on raw validation logs is no longer sufficient.

### 4. Documentation Synchronization

The main project documentation and workflow diagrams must be updated.

They must explicitly reflect the independent evaluation pipeline and the hardened public types.

---

## Consequences

### Positive

- Enforces strict execution boundaries.
- Eliminates intermediate,unloaded model states.
- Satisfies the explicit output requirement established in ADR-0024.

### Costs

- Requires refactoring the current `evaluate_model` pipeline.
- Demands new exporter modules for visual confidence maps.

---

## Follow-up Work (Deferred)

- Define standardized schemas for downstream consumption of the new visual artifacts.

---

## Summary

This ADR finalizes the boundaries established in ADR-0023 and ADR-0024.

It exposes formal `Session` types, shifts checkpoint loading inward, and requires explicit visual exports.
