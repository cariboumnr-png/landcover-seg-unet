# ADR-0015: Refine `models/` Module Boundaries and Remove External Config Coupling

- **Status:** Accepted
- **Date:** 2026-04-08

## Context
We noticed that the `landseg.models` package previously mixed concerns between:
- pure model construction logic, and
- experiment/runtime configuration driven by the Hydra-based `configs` module.

In particular, `models.factory` depended directly on `configs.ModelsCfg`, which coupled model construction to a specific configuration system. This coupling made models harder to reuse, test, and evolve independently of the CLI and experiment layer.

In contrast, the `geopipe` package demonstrates a cleaner separation of responsibilities:
- each module owns its internal logic and configuration semantics,
- configuration objects are passed in explicitly from the application layer, and
- factories are thin, stateless, and free of global config dependencies.

## Decision
We have realigned `landseg.models` with the same boundary discipline used in `geopipe`.

Specifically, we have:

1. Removed all imports of `landseg.configs` from `landseg.models`.
2. Introduced protocol-based structural contracts for model configuration inputs,
   allowing Hydra-backed dataclasses to be supplied without creating a hard
   dependency.
3. Refactored `models.factory` into a small API layer that:
   - owns no global configuration state,
   - receives explicit, keyword-only, typed arguments,
   - constructs models deterministically from those arguments.
4. Retained the dependency on `core.DataSpecs`, as this object is runtime-derived,
   immutable, and correctly represents dataset-specific constraints.

The Hydra / CLI layer remains responsible for:
- loading YAML configuration files,
- instantiating configuration dataclasses,
- invoking the model factory with explicit arguments.

## Consequences

### Positive
- Clear separation between model logic and experiment configuration.
- Models are reusable outside Hydra (e.g., unit tests, notebooks, scripted runs).
- Improved testability and readability of model construction paths.
- Stronger alignment across `models`, `trainer`, and `geopipe` packages.
- A stable, explicit public API for model construction.

### Negative
- One-time refactor cost to update factory signatures and call sites.
- Slightly more verbosity at the CLI layer due to explicit argument passing.

Overall, this change reduces long-term maintenance cost and establishes a
clear architectural boundary that supports future extensions (e.g., new
backbones, export paths, or alternative configuration systems).
