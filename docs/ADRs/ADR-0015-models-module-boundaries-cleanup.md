# ADR-0015: Refine `models/` Module Boundaries and Remove External Config Coupling

- **Status:** Proposed
- **Date:** 2026-04-07

## Context
We noticed that the current `landseg.models` package mixes concerns between:
- pure model construction logic, and
- experiment/runtime configuration driven by the Hydra-based `configs` module.

In particular, `models.factory` depends directly on `configs.ModelsCfg`, which couples model construction to a specific configuration system. This makes models harder to reuse, test, and evolve independently of the CLI and experiment layer.

In contrast, the `geopipe` package demonstrates a cleaner separation of responsibilities:
- each module owns its internal logic and configuration semantics,
- configuration objects are passed in explicitly from the application layer, and
- factories are thin, stateless, and free of global config dependencies.

## Decision
We plan to realign `landseg.models` with the same boundary discipline used in `geopipe`.

Specifically, we will:

1. Remove all imports of `landseg.configs` from `landseg.models`.
2. Move model-side configuration dataclasses (e.g., backbone parameters, head definitions, conditioning options) into the `models` package itself.
3. Refactor `models.factory` into a small API layer that:
   - owns no global configuration state,
   - receives explicit, typed arguments,
   - builds models deterministically from those arguments.
4. Keep the dependency on `core.DataSpecs`, as this object is runtime-derived, immutable, and correctly represents dataset-specific constraints.

The Hydra/CLI layer will remain responsible for:
- loading YAML configuration files,
- instantiating config dataclasses,
- calling the model factory with explicit arguments.

## Expected Consequences
### Positive
- Clear separation between model logic and experiment configuration.
- Models become reusable outside Hydra (e.g., unit tests, notebooks, scripted runs).
- Improved testability and readability of model construction paths.
- Stronger alignment across `models`, `trainer`, and `geopipe` packages.

### Negative
- A one-time refactor cost to update factory signatures and call sites.
- Slightly more verbosity in the CLI layer due to explicit argument passing.

Overall, we expect this change to reduce long-term maintenance cost and make future architectural extensions (e.g., new backbones, export paths) significantly easier.
