# ADR-0018: Checkpoint Ownership and Experiment Path Centralization

- **Status:** Accepted
- **Date:** 2026-04-11
- **Implemented:** 2026-04-13

## Context

At the time of this ADR, session-related filesystem concerns were split across
multiple layers. The training CLI constructed experiment and run directories
ad hoc, while checkpoint save/load logic lived inside trainer utilities. The
artifact path system did not define any session-related paths, despite
checkpoints being persisted outputs.

At the same time, we intentionally distinguished between:

- **Geopipe artifacts**: structured, stable, versioned outputs
- **Session run outputs**: ephemeral, iterative products of training runs

## Decision (Implemented)

We implemented two focused structural changes:

### 1. Checkpoint logic moved to the artifact subsystem

- Checkpoint save/load utilities were relocated from trainer/session utilities
  into `landseg.artifacts.checkpoint`.
- The artifact layer now owns checkpoint path resolution and persistence.
- Trainer and runner code no longer constructs checkpoint paths directly.
- Checkpoints are treated as artifact-like outputs without enforcing full
  artifact lifecycle or versioning semantics.

### 2. Experiment folder initialization moved to `artifacts/paths.py`

- Experiment/run directory structure is now defined centrally in the artifact
  path layer.
- Per-run subdirectories (checkpoints, logs, plots, previews) are specified via
  artifact-managed path builders.
- The training CLI no longer creates filesystem structures manually and instead
  consumes artifact path APIs.

## Explicit Non-Decision (Preserved)

### Session outputs remain outside canonical artifacts

- Session run outputs continue to live under `results/`.
- They are treated as run results rather than canonical, versioned artifacts.
- The distinction between geopipe artifacts and session outputs remains
  explicit and enforced in code.

## Implementation Evidence

The following changes satisfy this ADR:

- `src/landseg/session/engine/trainer/utils/checkpoint.py` was moved to
  `src/landseg/artifacts/checkpoint.py`, and re-exported via the artifact module.
- Trainer and runner code now imports checkpoint utilities from
  `landseg.artifacts`.
- `_init_experiment_folder` was removed from the training CLI.
- `ResultsPaths` and per-run directory structure were defined in
  `src/landseg/artifacts/paths.py`.
- The CLI constructs no filesystem paths directly for runs or checkpoints.

## Consequences

### Positive

- Checkpoint ownership is centralized and reusable across training, resume, and
  future evaluation paths.
- The training CLI is thinner and no longer responsible for filesystem layout.
- Run directory structure is consistent across all session tooling.
- The conceptual boundary between **canonical artifacts** and **run outputs** is
  maintained.

### Costs

- Required coordinated refactors across CLI, trainer, runner, and artifact
  modules.
- Artifact paths now cover both long-lived and ephemeral outputs, requiring
  discipline in lifecycle handling.

## Outcome

This ADR has been fully satisfied. The codebase now reflects the intended
ownership model and path centralization, and this ADR is retained primarily
for architectural record-keeping.

## Following ADRs

- ADR-0019: Shared Execution Core Extraction from Trainer
- ADR-0020: Dedicated Evaluator Engine and Callback Model
- ADR-0021: Session Component and Runtime-State Boundaries