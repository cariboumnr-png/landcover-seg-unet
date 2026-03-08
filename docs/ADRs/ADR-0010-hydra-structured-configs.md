# ADR-0010: Adopt Structured Configs with Dataclasses for Hydra/OmegaConf
- **Status:** Proposed
- **Date:** 2026‑03-08

## Context

The current landseg pipeline relies on Hydra’s `DictConfig` objects with scattered validation across modules. There is no single authoritative schema for configuration structure, types, or cross-field rules.

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

## Scope

All major configuration sections (artifacts, dataprep, dataset, domain, extent, grid, experiment, models, trainer) will be modeled as dataclasses.

Hydra will still compose YAML configs, but the resulting DictConfig will be validated and converted into dataclass instances before entering pipelines.

## Validation Boundary Model

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

## Alternatives

- **Status Quo**: continue with DictConfig and scattered validation:
  - Inconsistent enforcement, harder to reason about; validation errors appear
  late.
- Use **Pydantic** instead of dataclasses:
  - Powerful validators & error messages, but not Hydra’s native path; adds a
  heavy dep; several teams still prefer OmegaConf’s native structured configs for seamless composition. (Community patterns exist to mix Hydra with Pydantic, but we prefer fewer moving parts.)
- Use **attrs** instead of **dataclasses**:
  - Also supported by OmegaConf; we can switch if needed.

## Implementation Plan

1. Create RootConfig and nested section dataclasses.
2. Add conversion logic at CLI boundary (`OmegaConf.structured()` + merge + resolve).
3. Add `__post_init__` semantic checks.
4. Migrate modules from ConfigAccess to typed dataclasses.
5. Add tests for configuration schema and error cases.
6. Remove redundant structural checks now covered by schema.

## File Boundary Summary (High-Level)
- **Validate early**: config structure, types, ranges.
- **Validate at runtime**: data-driven conditions (files, rasters, stats, metrics).
- The boundary is: *anything defined in YAML should be validated up front; anything dependent on external data must still be validated where used*.
