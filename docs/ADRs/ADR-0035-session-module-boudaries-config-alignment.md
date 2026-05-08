# ADR-0035: Isomorphic Configuration Schema and Engine Boundary Alignment

**Status:** Proposed
**Date:** 2026-05-08

---

## Context

Following the reorganization of session components in ADR-0034, the system currently features cleanly separated data ingestion, task definitions, batch execution, and macro-level orchestration. However, the structured configuration schema (`SessionConfig`) and the Hydra YAML tree do not yet reflect these new boundaries. 

Currently, the schema relies on a catch-all `runtime` block that mixes macro-orchestration concerns (e.g., maximum epochs, early stopping) with micro-execution mechanics (e.g., AMP precision, logit adjustment). This breaks dependency injection principles, as modules are being passed large, generalized configuration blocks containing unrelated parameters. 

To maintain architectural clarity, the configuration tree must be isomorphic to the Python module tree it configures.

---

## Decision

We are going to adopt an **isomorphic configuration schema** for the `session` module. The configuration tree will explicitly mirror the runtime execution boundaries. 

### 1. Schema Restructuring
We will restructure the `SessionConfig` dataclass and corresponding Hydra YAML files into the following strict partitions:

* **`session.data`**: Will replace `loader`. Owns ingestion mechanics (`patch_size`, `batch_size`).
* **`session.tasks`**: Owns declarative objectives (loss types, focal parameters, exclusions).
* **`session.optimization`**: Will consolidate optimization state (`opt_cls`, `lr`, `grad_clip_norm`, scheduler args).
* **`session.executor`**: **(New)** Will own micro-level batch mechanics consumed purely by the `BatchEngine` (`use_amp`, `logit_adjust`).
* **`session.orchestration`**: **(New)** Will replace macro-`runtime` and `phases`. Consumed by the Runner and Phase policies (`schedule`, `monitor`, `curriculum`).

### 2. Synchronization Across User Entry Points
Because this refactor will alter the root configuration shapes, we will update all primary user entry points to reflect the new schema simultaneously to prevent silent configuration failures. This synchronization will include:
* The base Hydra structured configs in `src/landseg/configs/`
* The user-facing `settings.yaml` at the project root
* The programmatic configuration overrides demonstrated in `notebooks/02_model_train.ipynb`

---

## Consequences

### ✅ Strict Dependency Injection
By passing `config.session.executor` to the `BatchEngine`, the engine will be mathematically isolated from orchestration parameters like `max_epoch` or `patience`. 

### ✅ Clearer YAML Hierarchy
Configuration domains will be instantly recognizable. Future additions, such as distributed data parallel (DDP) or `torch.compile` settings, will have an obvious home under `executor`, rather than being appended to a vague `runtime` list.

### ⚠️ Migration Overhead
Existing user scripts, notebooks, and custom `settings.yaml` files that referenced `session.runtime` or `session.loader` will break and require manual mapping to the new `orchestration`, `executor`, or `data` namespaces.

---

## Summary

We are going to reshape the configuration tree to serve as a 1:1 blueprint of the system's execution architecture. By eliminating catch-all blocks and enforcing strict domain-mapped configurations across `src`, `settings.yaml`, and `notebooks`, we will secure a highly stable and extensible configuration API.
