# ADR-0010: Adopt Structured Configs with Dataclasses for Hydra/OmegaConf
- **Status:** Proposed
- **Date:** 2026‑03-08

## Context

The previous landseg pipeline relies on Hydra’s `DictConfig` objects with scattered validation across modules. There is no single authoritative schema for configuration structure, types, or cross-field rules.

## Decision

Adopt **structured configs** backed by **Python dataclasses**, validated at the CLI boundary using `OmegaConf.structured()` and merged with user YAML configurations.

## Motivation

- Centralized, early validation of config structure and types.
- Clearer error messages and earlier failure modes.
- Strong typing tools (IDE autocompletion, mypy).
- Preserve Hydra YAML composition while adding schema guarantees.

## Consequences

### Positive
- Unified config schema, reduced duplication.
- Consistent defaults and type enforcement.
- Improved maintainability.

### Negative
- Migration effort to replace TypedDicts and ConfigAccess.
- Care needed for defaults in nested dataclasses.

## Validation Boundary Realized 

**Upfront (boundary / config-level) validation**:
- Types, enums, field presence.
- Cross-field semantic rules.
- Known values (e.g., model names, modes).

**Runtime (module-level) validation**:
- Anything relying on external environment, I/O, raster data, dynamic dataset state.
- Raster CRS and pixel size checks.
- Block validity, nodata handling, per-block counts.
- Schema/hash integrity checks.
- Metric/loss runtime consistency.
- Numeric stability and training-time safeguards.

This aligns with secure coding guidelines: validate at boundaries, then verify runtime assumptions where data is consumed.

## Alternatives Considered 

- **Status Quo**: continue with DictConfig and scattered validation:
  - Rejected due to inconsistent enforcement, harder to reason about; validation errors appear
  late.
- **Pydantic** : Rejected to keep dependencies light and leverage Hydra's native OmegaConf integration.

## Implementation Status

- [x] Created `RootConfig` and nested section dataclasses.
- [x] Integrated `OmegaConf.structured()` + merge + resolve at the entry point.
- [.] Implemented `__post_init__` semantic checks for experiment parameters - some implemented, more to go.
- [x] Migrated all modules from `ConfigAccess` to typed dataclasses.
- [ ] Verified schema integrity with new test suite - not doing this at current scope as we don't have a test framework implemented yet.


## File Boundary Summary (High-Level)
- **Validate early**: config structure, types, ranges.
- **Validate at runtime**: data-driven conditions (files, rasters, stats, metrics).
- The boundary is: *anything defined in YAML should be validated up front; anything dependent on external data must still be validated where used*.
