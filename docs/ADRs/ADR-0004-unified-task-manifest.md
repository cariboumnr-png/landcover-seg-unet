# ADR‑0004 — Unified Task Manifest

- **Status:** Proposed
- **Date:** 2026‑02‑10

## Context
Tasks currently assemble configs ad‑hoc. A single manifest referencing
grid/domain artifacts, datasets, and model hyper‑parameters improves
reproducibility.

## Decision
Define a task manifest schema that imports:
- Grid spec id/version & artifact paths.
- Domain schema version & artifact paths.
- Dataset splits and sources.
- Model config and output destinations.

## Consequences
- One entrypoint for training/inference (CLI or pipeline).
- Easier CI/CD promotion and auditability.

## Notes
- Backed by JSON Schema; validated in CI before execution.