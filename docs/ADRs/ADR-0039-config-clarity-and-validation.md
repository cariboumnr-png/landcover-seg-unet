# ADR: Explicit and Self-Validating Configuration Architecture

**Status:** Accepted
**Date:** 2026-05-24

## Context

The configuration system relied heavily on deferred interpolation, implicit path
composition, and partially validated schema state. This created several issues:

* configuration behavior was difficult to reason about before full resolution
* validation logic was inconsistent across schema sections
* runtime failures could occur far from the source of invalid configuration
* curriculum training semantics were ambiguous
* conditioning behavior was implicitly enabled by defaults
* configuration migration and debugging were difficult

The project also required greater flexibility for:

* custom curriculum phase definitions
* explicit dataset and artifact management
* optional domain conditioning
* stricter configuration guarantees at schema boundaries

## Decision

The configuration architecture is updated with the following principles:

1. Prefer explicit resolved filepaths over implicit path composition.
2. Move validation into schema/dataclass validation boundaries.
3. Reduce reliance on deferred OmegaConf interpolation behavior.
4. Clear separation of curriculum schemas into:
   * `single`
   * `baseline`
   * `custom`
5. Make model conditioning explicitly opt-in through `conditioners`.
6. Enforce stronger runtime invariants and parameter validation.
7. Simplify default configuration structure and training semantics.

## Consequences

### Positive

* clearer and more predictable configuration behavior
* earlier validation failures with better error locality
* easier debugging and reproducibility
* simpler mental model for configuration resolution
* improved extensibility for training curricula
* more explicit model conditioning behavior

### Negative

* breaking configuration compatibility
* more verbose user configuration files
* reduced flexibility from implicit interpolation patterns
* migration required for older configs

## Migration Notes

Key breaking changes:

* `conditioner` renamed to `conditioners`
* implicit filename/path composition removed
* curriculum schema semantics changed
* validation is now stricter and eager
* conditioning defaults are now empty/disabled unless explicitly configured

## Design Principle

Configuration should be explicit, fully resolved, and self-validating at schema
boundaries rather than relying on deferred interpolation and implicit composition.
