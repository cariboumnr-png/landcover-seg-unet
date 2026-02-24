
# ADR‑0006 — Packaging & Entry Points

**Status:** Accepted
**Date:** 2026‑02‑22
**Updated:** 2026‑02‑23

## 1. Context

The project required proper packaging to ensure that:

- the ./src directory is a fully installable Python package,
- Hydra configuration is shipped with the package rather than living at repo root,
- CLI entry points are user‑facing, stable, and easy to discover,
- installation via `pip install .` works without manual path adjustments,
- users can run workflows without touching internal modules.

This improves maintainability, reproducibility, and usability for both developers and downstream users.

---

## 2. Decision

The following changes have been implemented and validated:

### 2.1 Packaging Layout
- Created dedicated package under:

        ./src/landseg/

- Added pyproject.toml‑based project metadata and build configuration.
- Installation now works cleanly via:

        pip install .

### 2.2 Configuration Structure
- Moved Hydra configuration tree from:

        ./configs/

  to:

        ./src/landseg/configs/

  ensuring configs are packaged and available post‑installation.

- Added a draft root‑level `settings.yaml` that acts as a user‑override layer
  (non‑exhaustive; designed to grow).

### 2.3 CLI Entry Points
- Moved original entry (`root/main.py`) to:

        ./src/landseg/cli/end_to_end.py

- Added the console script installed as:

        experiment_run

  which triggers the full end‑to‑end workflow.

### 2.4 ADR Status
- Implementation is complete, tested, and approved.
- ADR‑0006 is now **Accepted**.

---

## 3. Alternatives Considered

- Flat repository without packaging — rejected due to poor import hygiene and discoverability.
- Configs remaining outside the package — rejected because installed users would lose access to required Hydra configuration.
- Keeping a single Python entry script in root — rejected in favour of standardized, packaged CLI entry points.

---

## 4. Consequences

Positive outcomes:

- Users can execute workflows directly through `experiment_run`.
- Hydra configurations are properly versioned and shipped with the package.
- The project now follows modern Python packaging best practices.
- CI/CD workflows rely on stable, reproducible entry points.
- Provides a foundation for future modular CLI commands.

No negative impacts identified.

---

## 5. Status & Tasks

### Status: Accepted

### Completed
1. Packaging completed under `src/landseg/`.
2. Hydra config tree moved and included in package distribution.
3. Draft `settings.yaml` added at repository root.
4. `pyproject.toml` created; local installation verified.
5. CLI entry module moved to `landseg/cli/end_to_end.py`.
6. New console‑script entry point `experiment_run` added.

### Optional Future Work
- Expand `settings.yaml` to expose more user options.
- Add richer CLI subcommands (e.g., diagnostics, batch processing).
- Publish wheels to internal/external package index.
---
