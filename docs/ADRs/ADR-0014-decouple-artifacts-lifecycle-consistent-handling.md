# ADR-0014: Decoupling Grid/Domain Construction and Unifying Artifact Handling

- **Status:** Accepted
- **Date:** 2026-04-07

---

## Context

The original geopipe data ingestion and preparation pipelines exhibited architectural coupling and inconsistent handling of persisted artifacts. These issues reduced composability, testability, and predictability across the foundation and transform stages.

The following problems were identified before this ADR was implemented:

1. **World grid coupling during ingestion**
   - Grid construction and loading logic were combined.
   - Downstream steps required a fully instantiated grid object whose lifecycle was implicitly managed.
   - The world grid module carried multiple responsibilities (construction + persistence policy).

2. **Domain building embedded load-or-build behavior**
   - Domain builders implicitly reused existing artifacts.
   - Computation logic was mixed with artifact existence checks and reuse rules.
   - This limited reuse outside the canonical pipeline.

3. **Inconsistent artifact loading semantics**
   - Some artifacts were validated with schema IDs and hashes.
   - Others were loaded directly without integrity or compatibility guarantees.
   - There was no unified concept of artifact lifecycle policy.

4. **Inconsistent overwrite and existence behavior**
   - Artifact behavior varied between silent overwrite, implicit reuse, or failure.
   - Semantics differed across grids, domains, blocks, and JSON artifacts.

These issues complicated:
- reuse of geopipe components in external tools
- reasoning about pipeline behavior
- enforcement of reproducibility guarantees
- safe incremental updates of datasets

---

## Decision

We **have decoupled construction from loading** and **have standardized artifact handling across geopipe** by implementing the following design decisions.

---

### ✅ 1. World grid modules are construction-only

**What we have done**

- The `foundation.world_grids` builder layer now:
  - **only constructs** `GridLayout` objects from explicit inputs
  - performs **no loading, reuse, or overwrite decisions**
- Grid construction is now a **pure and deterministic operation**.

**Where responsibility lives now**

- Grid persistence, reuse, validation, and overwrite behavior are handled by:
  - `foundation.world_grids.lifecycle`
  - `artifacts.PayloadController`
  - an explicit `LifecyclePolicy`

✅ Result: grid construction is reusable, testable, and independent of disk state.

---

### ✅ 2. Domain building is pure computation

**What we have done**

- Domain builders now:
  - accept only:
    - a `GridLayout`
    - mapped raster tiles
    - explicit parameters
  - return a fully constructed `DomainTileMap`
  - perform **no artifact lookup or reuse**
- All implicit load-on-exist behaviors were removed.

**Where responsibility lives now**

- Artifact persistence and reuse are handled by:
  - `foundation.domain_maps.lifecycle`
  - `artifacts.PayloadController`
  - explicit lifecycle policies

✅ Result: domain construction is deterministic and reusable outside the CLI pipeline.

---

### ✅ 3. Artifact handling is unified under an explicit lifecycle policy

**What we have done**

We introduced a **central artifact subsystem** (`landseg.artifacts`) that enforces a uniform artifact contract.

**All persisted artifacts now define**
- a schema identifier (where applicable)
- a deterministic payload hash (SHA‑256)
- structured metadata (where relevant)

**All artifact access now requires**
- an explicit lifecycle policy
- controlled load / build / rebuild semantics
- optional integrity verification

**Implemented lifecycle policies**
- `LOAD_ONLY`
- `LOAD_OR_FAIL`
- `BUILD_IF_MISSING`
- `REBUILD`
- `REBUILD_IF_STALE`

✅ Result: no artifact is ever loaded or overwritten implicitly.

---

### ✅ 4. Overwrite and existence semantics are explicit and consistent

**What we have done**

- All pipelines and lifecycle modules now:
  - require an explicit `LifecyclePolicy`
  - apply identical semantics across:
    - world grids
    - domain maps
    - mapped windows
    - data blocks
    - catalogs
    - schemas
    - transform artifacts
- Implicit behaviors (e.g. “auto-load if exists”) were eliminated at the module level.

✅ Result: artifact behavior is predictable, auditable, and reproducible.

---

## Implementation Notes

During implementation, the scope of this ADR naturally expanded to include a **centralized artifact infrastructure**, which now serves as the backbone of geopipe’s persistence model.

We introduced:
- `artifacts.Controller` for single-file artifacts (JSON, NPZ)
- `artifacts.PayloadController[D, M]` for payload + metadata pairs
- `ArtifactPaths`, `FoundationPaths`, and `TransformPaths` for canonical layout
- coordinated manifest management for catalogs and schemas

All geopipe stages now follow the same lifecycle model.

---

## Consequences

### ✅ Positive outcomes

- Grid and domain builders are **pure, deterministic, and reusable**.
- Pipelines are **explicit orchestration layers**, not implicit lifecycle controllers.
- Artifact integrity guarantees are:
  - consistent
  - enforceable
  - auditable
- Incremental dataset updates are safer and easier to reason about.
- The transform pipeline is now fully artifact-driven and policy-governed.

### ⚠️ Trade-offs

- CLI and pipeline code is more explicit and verbose.
- Responsibility has shifted toward clear orchestration rather than convenience.
- Initial refactoring effort was non-trivial.

These trade-offs are intentional and align with long-term maintainability.

---

## Follow-up Actions (Completed ✅)

- ✅ Refactored world grid construction and lifecycle handling
- ✅ Refactored domain map construction and persistence
- ✅ Introduced a unified artifact subsystem
- ✅ Updated all CLI pipelines to use explicit artifact policies
- ✅ Standardized catalog and schema lifecycle management
- ✅ Documented artifact contracts and responsibilities implicitly through code

---

**Final state:**
ADR‑0014 reflects the current architecture accurately and is considered **fully implemented and closed**.