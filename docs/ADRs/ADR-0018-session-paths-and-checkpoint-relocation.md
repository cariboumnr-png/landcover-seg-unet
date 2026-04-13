# ADR-0018: Checkpoint Ownership and Experiment Path Centralization

- **Status:** Proposed
- **Date:** 2026-04-11

## Context

The project already defines a canonical artifact path system under
`project/artifacts/`, with structured access to persisted outputs from
data ingestion and transformation pipelines (e.g., `foundation/`,
`transform/`).

However, session-related concerns are currently split across multiple
layers:

- The training CLI constructs experiment directories via
  `_init_experiment_folder`
- Checkpoint saving/loading logic lives inside trainer/session utilities
- The artifact path system does **not** currently define session-related paths

This creates two concrete issues:

1. **Checkpoint logic is not owned by the artifact subsystem**, even though
   checkpoints are persisted outputs.
2. **Experiment directory structure is defined ad hoc in the CLI**, rather
   than through a canonical path layer.

At the same time, not all session outputs should be treated the same as
data artifacts:

- Geopipe artifacts (foundation/transform) are structured, stable, and
  version-controlled.
- Training run outputs (logs, previews, intermediate checkpoints) are
  **ephemeral run results**, not strict artifacts in the same sense.

## Decision

This ADR makes two focused changes:

### 1. Checkpoint logic moves to the artifact subsystem

Checkpoint persistence (save/load path resolution) will be owned by the
artifact layer.

- The artifact system becomes responsible for defining **where checkpoints live**
- Trainer/session code will **no longer construct checkpoint paths directly**
- Checkpoints are treated as **artifact-like outputs**, but without imposing
  full artifact lifecycle or versioning semantics

### 2. Experiment folder initialization moves to `artifacts/paths.py`

The logic currently implemented in `_init_experiment_folder` will be
relocated into the artifact path layer.

- `artifacts/paths.py` will define:
  - experiment root creation (e.g., `results/exp_0001/`)
  - subdirectories (`logs/`, `checkpoints/`, `previews/`, `plots/`)
- The training CLI will:
  - call into the artifact path API
  - stop constructing directory structures manually

## Explicit Non-Decision

### Session outputs remain outside `artifacts/`

We explicitly **do not** move session outputs under `artifacts/` at this time.

- The `results/` (or equivalent) directory remains separate from
  `artifacts/`
- Session outputs are **run results**, not canonical data artifacts
- They are not required to follow the same structure, stability, or
  version-control expectations as geopipe outputs

This preserves a clear distinction:

| Type                | Location      | Characteristics                      |
|---------------------|--------------|--------------------------------------|
| Geopipe artifacts   | `artifacts/` | canonical, structured, versioned     |
| Session run outputs | `results/`   | ephemeral, iterative, non-canonical  |

## Scope

This ADR applies only to:

- checkpoint path ownership
- experiment folder creation

This ADR does **not** define:

- full session artifact taxonomy
- evaluator output structure
- artifact controller integration
- retention or lifecycle policies

## Rationale

### Why move checkpoint logic

Checkpointing is shared across:

- training (save)
- resume (load)
- future evaluation

Keeping path ownership inside trainer code makes this harder to reason
about and reuse. Moving it to the artifact layer centralizes path logic
without forcing full artifact semantics.

### Why move experiment folder creation

The current `_init_experiment_folder` duplicates responsibility that
belongs in a path-definition layer. Centralizing this:

- removes path construction from CLI code
- ensures consistent structure across all session tools
- prepares for reuse by evaluators and other runners

### Why keep sessions outside artifacts

Session outputs:

- are iterative and frequently overwritten
- do not require strict reproducibility guarantees
- differ fundamentally from geopipe artifacts

Forcing them into `artifacts/` would blur this distinction and introduce
unnecessary constraints.

## Consequences

### Positive

- checkpoint paths become centrally defined and reusable
- CLI becomes thinner and less responsible for filesystem structure
- consistent experiment directory layout across tools
- clearer separation between **data artifacts** and **run outputs**

### Negative / Costs

- requires refactoring `_init_experiment_folder` into artifact paths
- requires updating checkpoint save/load call sites
- introduces a conceptual split (artifacts vs results) that must be
  maintained consistently

### Risks

- partial migration may temporarily leave dual checkpoint logic paths
- unclear boundaries between “artifact-like” and “run-only” outputs may
  need refinement in future ADRs

## Alternatives Considered

### 1. Move sessions fully under `artifacts/`

Rejected. Session outputs are not equivalent to geopipe artifacts and do
not require strict versioning or canonical structure.

### 2. Keep everything in trainer/CLI

Rejected. This continues duplication and prevents reuse across training,
evaluation, and future tooling.

### 3. Only move checkpoints, keep init logic in CLI

Rejected. Path ownership would remain split, defeating the purpose of a
central path layer.

## Implementation Notes

Follow-up work:

1. Extend `artifacts/paths.py` with:
   - experiment root creation utilities
   - subdirectory path builders (`logs`, `checkpoints`, etc.)
2. Replace `_init_experiment_folder` with artifact path APIs
3. Move checkpoint path resolution into artifact-owned utilities
4. Update trainer/session code to consume these APIs

## Related Code Areas

- `project/artifacts/paths.py`
- training CLI (`train_model.py`)
- trainer checkpoint utilities
- session runner and engine

## Following ADRs

- ADR-0019: Shared Execution Core Extraction from Trainer
- ADR-0020: Dedicated Evaluator Engine and Callback Model
- ADR-0021: Session Component and Runtime-State Boundaries