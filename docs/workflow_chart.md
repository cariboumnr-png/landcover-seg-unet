## Current workflow
```
[foundation/world_grids/builder]            (1 World Grid – pure construction)
|
+--> [foundation/world_grids/lifecycle]
|        (grid artifact persistence & validation)
|
+--> [foundation/domain_maps/mapper]        (2 Domain → Grid, optional)
|        |
|        +--> [foundation/domain_maps/builder]
|        |        (pure domain feature computation)
|        |
|        +--> [foundation/domain_maps/lifecycle]
|                 (domain artifact persistence)
|
+--> [foundation/data_blocks/mapper]        (3 Imagery/Labels → Grid windows)
|        |
|        +--> [foundation/data_blocks/builder]
|        |        (pure block construction)
|        |
|        +--> [foundation/data_blocks/manifest]
|                 (catalog & schema update)
|
+--> [geopipe/specification/factory]        (4 Build DataSpecs)
|
+--> [models/factory]                       (5.1 Model construction)
|
+--> [trainer/components/]                 (5.2 Heads / Loss / Optim dataloaders, metrics, callbacks)
|
+--> [trainer/engine/engine]                (6 Training engine)
|
+--> [trainer/runner/]            (7 Phases / Runner)
|
+--> [cli/pipelines/*]    (8 Pipeline execution)
```
---

### Interpretation notes

- All **build steps** (grid, domain, blocks) are now **pure and deterministic**.
- All reuse, overwrite, and validation logic flows through:
  **`artifacts.Controller` / `PayloadController`**.
- The CLI executes **explicit pipeline stages** rather than a single
  end‑to‑end implicit run.
- The workflow cleanly separates:
  - **foundation artifacts** (ingest)
  - **experiment artifacts** (transform)
  - **training runtime**