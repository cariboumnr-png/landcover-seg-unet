# ADR-0061: Configuration Modernization and Modular Recipes

**Status:** Accepted<br>
**Date:** 2026-10-08

---

## 1. Context

The configuration system in `landseg` evolved through Hydra-backed
dataclass schemas (`landseg.configs.schema`) and a single user recipe
surface (`configs/user.yaml`).

While this structure established strong type checking and runtime
validation, several quality-of-life and architectural frictions
emerged:

1. **Monolithic Schema Modules**:
   Sections such as `sections/data.py` and `sections/session.py` bundled
   multiple distinct sub-domains (e.g., world grid, ingestion, partitioning,
   engine optimization, multitask loss, and curriculum orchestration) into
   dense single files.
2. **Disconnected Validation Primitives**:
   Validation utilities (`must_within`, `must_exist`, `file_exists`) resided
   in a standalone `utils.py` module. Schema dataclasses imported and invoked
   them procedurally rather than inheriting a consistent validation
   protocol or lifecycle.
3. **Boundary Bleed Between Domain and Command Configs (ADR-0060 §6.1)**:
   Sub-configs mixed long-lived domain parameters with ephemeral CLI
   invocation parameters. Requiring nested overrides impaired CLI
   ergonomics.
4. **Monolithic User Recipe Surface (ADR-0060 §6.2)**:
   `configs/user.yaml` was designed as an end-to-end pipeline recipe.
   Fitting distinct operational tasks (`model-evaluate`, `study-sweep`,
   `batch-ingest`) into one file bloated the document with unused keys
   during standard workflows and required an intermediate translation layer.

---

## 2. Decision

We modernized the configuration schema and user recipe surfaces through
targeted modular enhancements:

### 2.1. Unified Section Base Class (`BaseConfigSection`)

We introduced a lightweight, field-free base class `BaseConfigSection` in
`landseg.configs.schema.base`:

- **Static Validation Primitives**: Exposed type and range check helpers
  directly on the class, allowing dataclass methods to call
  `self.require_type_range(...)` or `self.require_attr_type_range(...)`.
- **Validation Lifecycle**: Provided a default no-op `validate(self) -> None`
  to guarantee a uniform interface across all configuration nodes.
- **Serialization Helper**: Provided a uniform `as_dict` property returning
  `dataclasses.asdict(self)`.
- **Dataclass Preservation**: Schema classes inherit `BaseConfigSection`
  while strictly preserving `@dataclasses.dataclass`. `BaseConfigSection`
  remains a plain (non-dataclass) class with zero fields to prevent Python
  default-argument ordering conflicts.

### 2.2. Modularization of Monolithic Sections

We decomposed monolithic schema files into focused sub-modules under
facade packages:

- **`data/`**: Split into `world_grid.py`, `harmonziation.py`, `ingestion.py`,
  `preparation.py`, `specification.py`, and composite aggregator
  `_composite.py`.
- **`session/`**: Split into `dataloader.py`, `engine.py`, `evaluation.py`,
  `orchestration.py`, and composite aggregator `_composite.py`.
- **`models/` & `study/`**: Modularized backbones, conditioners, safety,
  optimization, architecture, objectives, and sweeps.
- **Facade Re-exports**: Facade `__init__.py` files re-export all public
  and private dataclasses, maintaining transparent compatibility for all
  tests and import sites.

### 2.3. Command-Aware Partial Validation

Each recipe configures only the subset of the schema required for its
target command. We enhanced `RootConfig.validate(command)` with
command-aware selective branching:

- Only active pipeline sections are validated for the selected execution
  command.
- `SessionConfig.validate(mode)` differentiates between `'train'` and
  `'evaluate'` execution modes, enforcing `evaluation.checkpoint` only
  during evaluation sessions.

### 2.4. Modular User Recipes (`configs/`)

We decoupled `configs/user.yaml` into lean, task-oriented recipe files
located directly in `configs/`:

- Single pipelines: `world_grid.yaml`, `data_harmonize.yaml`,
  `data_ingest.yaml`, `data_prepare.yaml`, `model_train.yaml`,
  `model_evaluate.yaml`.
- Specialized workflows: `diagnose_overfit.yaml`,
  `study_sweep.yaml`, `study_analysis.yaml`.
- Composite lifecycles (`configs/end_to_end/`): `e2e_intake.yaml`
  (continuous intake) and `e2e_experiment.yaml` (full 5-stage lifecycle
  replacing legacy `user.yaml`).
- **Translation Layer Retirement**: Recipes directly mirror the
  structured OmegaConf schema without requiring intermediate translation
  dictionaries; the legacy translation layer (`translate.py`) has been
  retired, and custom recipe paths are supported via
  `execution.recipe_fpath`.

---

## 3. Consequences

### Positive
- **Cohesive Schema Hierarchy**: All configuration nodes share a common
  validation interface and lifecycle hook (`validate()`).
- **Command-Aware Validation**: Workflows only validate their active
  sections, eliminating false validation errors on unused configuration
  sub-trees.
- **Direct Schema Alignment**: Eliminating intermediate recipe translation
  provides direct 1-to-1 parity between YAML recipe keys and dataclass
  fields.
- **Clean Recipe Management**: Users interact with lean, task-specific
  recipes instead of a bloated master configuration file.
- **Zero Breaking Changes**: Preserved module aliases and facades ensure
  no call-site rewrites across pipelines or tests.

### Negative / Considerations
- **Multiple Recipe Templates**: Maintaining separate recipe files under
  `configs/` requires keeping templates aligned as schema defaults
  evolve.
