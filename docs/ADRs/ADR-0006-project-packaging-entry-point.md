# ADR‑0006 — Packaging & Entry Points
**Status:** Proposed
**Date:** 2026‑02‑22

## Context
The code under `./src` should be bundled as a proper Python package with:
- a consistent import namespace,
- a versioned distribution,
- clear console entrypoints for the main workflows (prepare, load, report).

This improves reproducibility, discoverability, and CI/CD integration.

## Decision
- Adopt **src‑layout packaging** with a single top‑level namespace (e.g.,
  `caribou_landcover`) and publish as a wheel.
- Use a `pyproject.toml` (PEP 621) and a modern backend (`hatchling` or
  `setuptools>=64`).
- Provide **console scripts**:
  - `caribou prep` — run the dataprep pipeline end‑to‑end (map → build →
    normalize → split → schema). 
  - `caribou load` — validate schema, (re)build if required, and print a
    concise `DataSpecs` summary.
  - `caribou report` — aggregate per‑tile/AOI diagnostics as per ADR‑0005.
- Treat generated `schema.json` as the **manifest of record**; optionally
  accept a user manifest later (see ADR‑0005).

## Alternatives Considered
- Flat layout without package: rejected; poorer tooling and discoverability.
- Single monolithic CLI binary: rejected; less composable for future modules.

## Consequences
- Clear imports and API stability.
- Easier installation (`pip install .`) and environment management.
- CI can call stable entrypoints; users can discover commands via `--help`.

## Implementation Notes (Non‑normative)
- Namespace: `caribou_landcover` (or shorter if you prefer).
- Minimum Python: 3.12 (project standard).
- Type hints: adhere to existing style (e.g., `str | None`), PEP 8 line widths.
- Versioning: SemVer; keep `__version__` in the package and expose via CLI.
- Dependencies: pin lower bounds; avoid vendor‑locking GDAL—document env setup.
- Testing: `tests/` with unit tests for CLI and critical modules.
- Logging: `logging` with `--log-level` flag; structured if needed.

## Status & Tasks
- Status: Proposed
- Tasks:
  1) Create `pyproject.toml` with build backend and metadata.
  2) Move modules under `src/caribou_landcover/` + fix imports.
  3) Wire console scripts (`caribou prep|load|report`).
  4) Add packaging docs and a quickstart.
