# ADR-0061: Configuration Modernization and Modular Recipes

**Status:** Proposed<br>
**Date:** 2026-10-08

---

## 1. Context

The configuration system in `landseg` evolved through Hydra-backed
dataclass schemas (`landseg.configs.schema`) and a single user recipe
surface (`configs/user.yaml`).

While this structure established strong type checking and runtime
validation, several quality-of-life and architectural frictions have
emerged:

1. **Monolithic Schema Modules**:
   Sections such as `sections/data.py` (~320 lines) and `sections/session.py`
   (~340 lines) bundle multiple distinct sub-domains (e.g., world grid,
   ingestion, partitioning, engine optimization, multitask loss, and
   curriculum orchestration) into single dense files.
2. **Disconnected Validation Primitives**:
   Validation utilities (`must_within`, `must_exist`, `file_exists`) reside
   in a standalone `utils.py` module. Schema dataclasses import and invoke
   them procedurally rather than inheriting a consistent validation
   protocol or lifecycle.
3. **Boundary Bleed Between Domain and Command Configs (ADR-0060 §6.1)**:
   Sub-configs currently under `CommandConfig` (such as `_TrainModel`,
   `_EvaluateModel`, and `_StudySweep`) mix long-lived domain parameters
   with ephemeral CLI invocation parameters. Requiring nested overrides
   like `command.model_evaluate.checkpoint=...` impairs CLI ergonomics.
4. **Monolithic User Recipe Surface (ADR-0060 §6.2)**:
   `configs/user.yaml` was designed as an end-to-end pipeline recipe.
   Fitting distinct operational tasks (`model-evaluate`, `study-sweep`,
   `batch-ingest`) into one file bloats the document with unused keys
   during standard workflows.

---

## 2. Decision

We will modernize the configuration schema and user recipe surfaces
through targeted quality-of-life enhancements, preserving complete
backward compatibility with existing call sites and Hydra resolvers.

### 2.1. Unified Section Base Class (`BaseConfigSection`)

We will introduce a lightweight, field-free base class `BaseConfigSection`
in `landseg.configs.schema.base`:

- **Static Validation Primitives**: Expose `must_within`, `must_exist`, and
  `file_exists` directly on the class, allowing dataclass methods to call
  `self.must_within(...)` or `BaseConfigSection.must_within(...)`.
- **Validation Lifecycle**: Provide a default no-op `validate(self) -> None`
  to guarantee a uniform interface across all configuration nodes.
- **Serialization Helper**: Provide a uniform `as_dict` property returning
  `dataclasses.asdict(self)`.
- **Dataclass Preservation**: Schema classes will inherit
  `BaseConfigSection` while strictly preserving `@dataclasses.dataclass`.
  `BaseConfigSection` will remain a plain (non-dataclass) class with zero
  fields to prevent Python default-argument ordering conflicts.
- **Backward Compatibility**: `landseg.configs.schema.utils` will re-export
  the validation helpers, ensuring existing imports continue working without
  modification.

### 2.2. Modularization of Monolithic Sections

We will decompose monolithic schema files into focused sub-modules under
facade packages:

- **`sections/data/`**: Split into `grid.py`, `ingestion.py`, `preparation.py`,
  and `specification.py`.
- **`sections/session/`**: Split into `loader.py`, `engine.py`, and
  `orchestration.py`.
- **Facade Re-exports**: `sections/data/__init__.py` and
  `sections/session/__init__.py` will re-export all public and private
  dataclasses, maintaining transparent compatibility for all tests and
  import sites.

### 2.3. Domain vs. Command Boundary Separation

We will decouple long-lived domain declarations from ephemeral invocation
flags:

- **Flatten Command Overrides**: Promote ephemeral CLI flags directly
  under `CommandConfig` (e.g., `command.checkpoint: str | None = None`,
  `command.split: str = 'test'`), enabling clean CLI overrides like
  `command.checkpoint=...`.
- **Domain Property Alignment**: Ensure persistent evaluation and sweep
  parameters reside within their natural domain containers (`session` and
  `study`).

### 2.4. Modular User Recipes (`configs/recipes/`)

We will decouple `configs/user.yaml` into dedicated, task-oriented recipe
files under `configs/recipes/`:

- `configs/recipes/train_e2e.yaml`: Full end-to-end pipeline training recipe.
- `configs/recipes/evaluate.yaml`: Dedicated checkpoint evaluation recipe.
- `configs/recipes/sweep.yaml`: Hyperparameter sweep optimization recipe.
- `configs/recipes/batch_ingest.yaml`: Incremental batch ingestion recipe.
- **Translator Compatibility**: We will retain `configs/user.yaml` as the
  default training wrapper and update `translate_user_config` to seamlessly
  resolve specialized recipe files.

---

## 3. Consequences

### Positive
- **Cohesive Schema Hierarchy**: All configuration nodes share a common
  validation interface and lifecycle hook (`validate()`).
- **Improved Code Organization**: Dense multi-hundred-line files are
  decomposed into readable, single-responsibility modules matching the test
  suite structure.
- **Ergonomic CLI Overrides**: Flattened invocation parameters simplify
  command-line overrides without cumbersome sub-dictionary nesting.
- **Clean Recipe Management**: Users interact with lean, task-specific
  recipes instead of a bloated master configuration file.
- **Zero Breaking Changes**: Preserved module aliases and facades ensure
  no call-site rewrites across pipelines or tests.

### Negative / Considerations
- **Multiple Recipe Templates**: Maintaining separate recipe files under
  `configs/recipes/` requires keeping templates aligned as schema defaults
  evolve.
