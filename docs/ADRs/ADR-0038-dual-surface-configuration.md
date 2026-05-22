# ADR-0038: Dual-Surface Configuration and Programmatic Notebook API

**Status:** Proposed
**Date:** 2026-05-22

## 1. Context

Following ADR-0035, our configuration schema is strictly isomorphic, mirroring the decoupled execution architecture (`engine_exec`, `engine_tasks`, `orchestration`). While this provides mathematically isolated dependency injection and robust CLI-driven execution via Hydra, it introduces friction for interactive development. 

Researchers and notebook users engaged in rapid prototyping or debugging currently face the cognitive load of navigating nested YAML structures and lack IDE features like autocomplete and type-hinting. 

Crucially, our execution engine (`execution.executor.execute_pipeline`) relies strictly on the fully resolved `RootConfig` Python dataclass, having already stripped away the intermediate Hydra `DictConfig` layer during resolution.

## 2. Decision

We will implement a **Dual-Surface Configuration Architecture** that provides a specialized interface for interactive environments without duplicating schema logic or altering the core execution pipelines.

### 2.1. The CLI Surface (Production)
The CLI workflow remains the gold standard for production. Users triggering automated jobs, remote Databricks executions, or multi-run Optuna sweeps will continue to use `settings.yaml` and native Hydra command-line overrides. The `execution.resolver` will parse these inputs, merge them via OmegaConf, and cast them to the `RootConfig` dataclass before execution.

### 2.2. The Programmatic API Surface (Interactive)
For Jupyter Notebooks and pure Python scripts, we will bypass Hydra and OmegaConf entirely. We will introduce a `SessionBuilder` class in `src/landseg/adapters/api.py`. 

This builder will use a fluent, method-chaining interface to construct the native `RootConfig` dataclass directly in memory. It will trigger `.validate_all()` and pass the raw dataclass directly to `execute_pipeline`.

### 2.3. Selective Parameter Exposure
To maximize user ergonomics and reduce cognitive load, the `SessionBuilder` will intentionally hide the full nested complexity of the configuration tree. It will expose only high-impact "research knobs" organized by intent:
- `set_data(batch_size, patch_size)`
- `set_model(body, base_channels)`
- `set_optimization(lr, weight_decay, optimizer)`
- `set_duration(epochs, early_stop_patience)`
- `set_objectives(focal, dice, spectral)`

"Infrastructure knobs" (e.g., complex multi-phase curriculums, filesystem paths, domain projection internals) will remain hidden in the builder, automatically falling back to the synchronized defaults defined in the underlying `landseg.configs._schema` dataclasses.

## 3. Consequences

### Positive
* **Zero Schema Duplication:** Both the CLI and the API point to the exact same core dataclass definitions.
* **Notebook Ergonomics:** Users gain native Python type-safety, autocompletion, and instant fail-fast validation right in the cell.
* **Zero Parsing Overhead:** Bypassing OmegaConf in notebooks completely eliminates the notorious complexity of initializing Hydra in interactive environments.
* **Intent-Driven UX:** Exposing parameters by intent (e.g., `set_duration` auto-adjusting the scheduler's `T_max` internally) makes the framework highly accessible for new researchers.

### Negative
* **Builder Maintenance:** If new, critical research parameters are added to the underlying dataclasses in the future, the `SessionBuilder` methods in `adapters/api.py` must be manually updated to expose them to notebook users.
