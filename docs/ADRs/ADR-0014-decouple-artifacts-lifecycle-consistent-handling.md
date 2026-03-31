# ADR-0014: Decoupling Grid/Domain Construction and Unifying Artifact Handling

- **Status:** Proposed
- **Date:** 2026-03-31

## Context

The current `landseg.geopipe` data ingestion and preparation pipelines (`ingest_data.py`, `prepare_data.py`) reveal several architectural inconsistencies and unnecessary couplings:

1. **World grid coupling during ingestion**
   - `foundation.build_world_grid(...)` both *loads* and *builds* a `GridLayout`.
   - As a result, downstream steps (domain building, block building) require a fully instantiated `GridLayout` object to be passed through the ingestion pipeline.
   - This couples higher-level orchestration logic with lower-level persistence decisions.
   - The `world_grid` submodule is responsible for *more than one concern* (construction + load policy).

2. **Domain building embeds load-or-build behavior**
   - `foundation.build_domains(...)` implicitly loads existing domain artifacts if present.
   - Like the grid, domain construction mixes:
     - domain computation logic
     - artifact lookup, existence checks, and reuse rules
   - This makes domain building less composable and harder to reason about when used outside the canonical ingestion pipeline.

3. **Inconsistent artifact loading semantics**
   - Some artifacts (e.g., world grid, domain maps) are:
     - validated with schema identifiers
     - verified via hashes
   - Other artifacts (e.g., generic JSON files such as `block_source.json`, `label_stats.json`, intermediate configs) are:
     - loaded directly without integrity or compatibility checks
   - There is no unified concept of “artifact loading policy”.

4. **Inconsistent overwrite / existence behavior**
   - Some artifacts:
     - automatically reload if present
     - refuse overwrite implicitly
     - silently overwrite
   - Others expose explicit flags (`remap`, `rebuild`, etc.).
   - The behavior is inconsistent across foundation and transform stages.

These issues complicate:
- reuse of geopipe components in external tools
- testability of individual build steps
- predictability of pipeline behavior
- enforcement of reproducibility guarantees

## Decision

We will **separate construction from loading**, and **standardize artifact handling across geopipe** by adopting the following principles.

### 1. World grid module responsibility is construction only

- The `foundation.world_grid` submodule will:
  - **only build** a `GridLayout` from explicit inputs
  - **never decide** whether to load or reuse an existing grid
- `GridLayout` construction becomes a pure, deterministic operation.

**Resulting changes**
- `build_world_grid(...)` is split conceptually into:
  - `GridLayout.from_spec(...)` (pure construction)
- Load/reuse logic (`load_grid`, hash validation, overwrite rules):
  - moves to **external orchestration utilities** (e.g., CLI, pipeline helpers).

### 2. Domain building becomes pure computation

- Domain-building code will:
  - accept only:
    - a `GridLayout`
    - domain rasters
    - explicit parameters
  - return constructed domain objects
- Artifact lookup, version checks, and reuse decisions are removed from:
  - `build_domains(...)`
  - `DomainTileMap` factory logic

**Resulting changes**
- Domain load/save logic remains in `foundation.domain_maps.io`
- Decision logic (load vs rebuild vs overwrite):
  - handled at the pipeline / CLI level

### 3. Artifact loading is unified under a single policy

We introduce a **uniform artifact contract**:

All persisted artifacts must define:
- schema identifier (if structured)
- deterministic payload hashing
- optional metadata sidecar

All artifact loads must explicitly declare:
- expected schema (if applicable)
- whether integrity verification is required
- expected behavior on mismatch

**Standard load modes**
- `LOAD_ONLY` – must exist and validate
- `BUILD_IF_MISSING` – build if absent, otherwise validate and load
- `REBUILD` – overwrite unconditionally
- `FAIL_IF_EXISTS` – enforce immutability

No artifact is loaded “at will”.

### 4. Explicit overwrite and existence semantics

All pipelines and utilities must:
- expose overwrite / rebuild semantics explicitly
- apply the same rules consistently across:
  - grids
  - domain maps
  - block windows
  - blocks
  - JSON artifacts

Implicit behaviors (e.g., “auto-load if exists”) are disallowed at the module level.

## Consequences

### Positive
- Developers can:
  - reuse grid/domain builders independently
  - reason clearly about when artifacts are built vs reused
- Pipelines become:
  - more deterministic
  - easier to test
  - easier to extend
- Artifact integrity guarantees are consistent and auditable.

### Trade-offs
- CLI pipelines will become slightly more verbose, as they must:
  - explicitly choose artifact policies
- Some refactoring effort is required in:
  - ingestion pipeline
  - domain factory
  - world grid factory

## Follow-up Actions

1. Refactor `foundation.world_grids.factory`:
   - isolate pure grid construction
2. Refactor `foundation.domain_maps.factory`:
   - remove embedded loading logic
3. Introduce shared artifact utilities:
   - load / save / validate / hash
4. Update CLI pipelines:
   - make artifact policies explicit and consistent
5. Document artifact lifecycle and policy usage for contributors
