# ADR‑0011: Module Boundaries, Factories, and Config Detachment
*Status: Proposed*
*Date: 2026‑03‑10*
*Depends on:* ADR‑0010 (Structured Configs with Dataclasses)

---

## 1. Context

After introducing **structured dataclass-backed configs** in ADR‑0010, the project now has a clean validation boundary:

- **Upfront (config-level)**: config shape, types, cross-field semantics.
- **Runtime (module-level)**: data‑driven verification (CRS, raster shapes, masks, NaN guards, loss safety, etc.).

ADR‑0010 documents these responsibilities and their placement at the CLI/config ingress versus runtime consumption. However, several runtime modules still depend directly on config dataclasses or Hydra/OmegaConf types. Examples include:

- `training.loss.CompositeLoss` imports `landseg.configs` and expects a `LossConfig`, creating a **domain → config** dependency.
- `training.loss.factory.build_headlosses` mutates `config.types.focal.alpha` per head (side‑effect at runtime).
- `controller.phase.generate_phases` mixes domain types (`Phase`) with config‑level logic.
- `training.dataloading.loader` imports `omegaconf.DictConfig`, leaving Hydra types in a runtime layer.

The project now needs a formal architecture decision that defines **boundary rules** between config ingestion, factory layers, and domain/runtime modules.

---

## 2. Decision

### 2.1 Factories are the *only* modules permitted to depend on config dataclasses
Factories translate **validated dataclasses → domain‑level specs or constructed objects**. Domain layers must not import `landseg.configs` directly.

### 2.2 Domain modules depend on narrow contracts (Protocols/ABCs or thin spec dataclasses)
Examples:
- `CompositeLoss` should accept either `(weight, PrimitiveLoss)` pairs or a `LossSpec`, not a full `LossConfig`.
- Models and trainers already use `DataSpecsLike`—this ADR generalizes that pattern across the codebase.

### 2.3 No factory may mutate config objects
Any mutation (e.g., overriding `alpha` inside `build_headlosses`) must be replaced with **spec construction** or **configured object construction**.

### 2.4 Strict layering rules
```
CLI/Hydra (DictConfig)
    → configs/*  (dataclasses + validation)
        → factories/*  (translation layer)
            → domain/* (models, losses, metrics, datasets,
                        controller runtime, dataprep internal logic)
```
- Only the top two layers may touch OmegaConf/Hydra types.
- Domain must never depend on config schemas.

### 2.5 Validation distribution (reaffirming ADR‑0010)
- **Upfront**: structure, types, enums, ranges, cross‑field rules.
- **Factories**: minimal **integration invariants** only they can see (e.g., per‑head alpha vector size matches class count).
- **Runtime**: safety and tensor validity (ignore_index behavior, NaN guards, block validity, CRS checks, metric safety, etc.).

---

## 3. Rationale

- Prevents *domain ↔ config* coupling and deep import chains.
- Eliminates side‑effects from config mutation.
- Simplifies testing (domain code can be fed with small specs or mocks instead of dataclasses).
- Unifies the architecture across training, models, dataset, controller, and dataprep.
- Fully aligned with ADR‑0010's principle of “validate at the boundary; verify at runtime.”

---

## 4. Consequences

### Positive
- Well‑defined, predictable boundaries improve maintainability.
- Stronger modularity: domain modules become portable and reusable.
- Factories become the central, composable seam for wiring configurations.
- Configuration mutations disappear, improving reproducibility and debuggability.

### Negative
- Requires a one‑time refactor of a few modules (loss factory, composite loss, controller phase factory, dataloading).
- Requires new spec dataclasses or Protocols where missing.

---

## 5. Module-by-Module Verdict

The following assessment is derived from the current project tree.

### 5.1 OVERHAUL Required
| Module | Reason |
|-------|--------|
| `training.loss.composite` | Depends on config dataclasses; should take specs or constructed losses only. |
| `training.loss.factory` | Mutates configs; must output immutable specs or ready-made losses per head. |

### 5.2 TUNE Required
| Module | Reason |
|--------|--------|
| `controller.phase` | Move `generate_phases` into a factory; make `Phase` config‑agnostic. |
| `training.dataloading.loader` | Remove `omegaconf.DictConfig`; feed typed dataclasses/specs only. |

### 5.3 OK
All others follow a clean, layered pattern (factories depend on configs; domain does not):
- CLI & configs
- Controller builder & controller runtime
- Models factory & internals (without configs)
- Trainer runtime, metrics, optim factory
- Dataset (loader/specs)
- Dataprep pipeline & stages
- Grid builder/layout
- Domain mapping/PCA
- Utils, Core protocols

---

## 6. Implementation Plan

### Step 1 — Introduce spec types for domain modules
Examples:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FocalSpec:
    weight: float
    alpha: list[float] | None
    gamma: float
    reduction: str

@dataclass(frozen=True)
class DiceSpec:
    weight: float
    smooth: float

@dataclass(frozen=True)
class LossSpec:
    focal: FocalSpec | None
    dice: DiceSpec | None
```
`CompositeLoss` then consumes `LossSpec` (or a list of `(weight, PrimitiveLoss)`), becoming config‑agnostic.

### Step 2 — Refactor factories to map dataclasses → specs or constructed objects
- Eliminate mutation patterns (`config.types.focal.alpha = ...`).
- Build fully-initialized `PrimitiveLoss` instances per head.

### Step 3 — Relocate factory logic away from domain modules
- Move `generate_phases` entirely into controller factory.

### Step 4 — Remove Hydra/OmegaConf from runtime layers
- Refactor `training.dataloading.loader` to expect pure dataclasses or specs.

### Step 5 — Add minimal integration checks in factories
Examples:
- Per-head alpha vector matches number of classes.
- At least one loss type enabled.
- Metrics/loss structure shapes are consistent where the factory has enough context.

### Step 6 — Maintain runtime safety checks in domain modules
Losses, metrics, trainer, dataprep, and controller already include these (NaNs, Inf, empty pixels, ignore_index, shape guards).

---

## 7. Checklists

### 7.1 Domain Modules Checklist
- [ ] No imports of `landseg.configs`.
- [ ] Public APIs consume only specs or already‑constructed objects.
- [ ] Runtime guards remain (NaN checks, mask logic, shape assertions).
- [ ] Factory code relocated elsewhere.

### 7.2 Factory Modules Checklist
- [ ] May import `landseg.configs`.
- [ ] No in‑place mutation of configs.
- [ ] Construct specs or domain objects.
- [ ] Provide integration asserts.
- [ ] Prefer immutable specs.

### 7.3 Runtime Layer Cleanup
- [ ] Remove any remaining `DictConfig` imports.
- [ ] Ensure Hydra/OmegaConf usage exists only in CLI/config layers.

### 7.4 Testing
- [ ] Domain modules tested using only specs or mocks.
- [ ] Factories tested with real dataclass configs.
- [ ] Regression test ensuring configs remain unmodified after factory calls.

---

## 8. Acceptance Criteria

- No domain module imports or references config dataclasses.
- No factory mutates a config object.
- `controller.phase` is config‑agnostic; phase construction lives in factories.
- `training.dataloading.loader` contains no Hydra/OmegaConf imports.
- All domain modules receive narrow specs or constructed instances.

---
