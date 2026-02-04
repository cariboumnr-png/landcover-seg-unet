# ADR-0001: Decouple Grid, Domain, and Task

- **Status**: Proposed
- **Date**: 2026-02-04
- **Context**:
  - The current dataset pipeline couples grid creation, domain assignment, and label assembly.
  - This blocks reproducible grids, global PCA semantics, and general inference tooling.
- **Decision**:
  - Introduce explicit, versioned **Grid Spec**, **Domain Schema**, and **Task Manifest**.
  - The model consumes a stable conditioning interface; upstream is deterministic and versioned.
- **Consequences**:
  - Initial overhead to write manifests and adjust loaders.
  - Large benefits in reproducibility, caching, and safe conditioning.
- **Alternatives**:
  - Keep coupling and patch ad hoc: faster short term, accumulates technical debt.
  - Build a global planetary grid: overkill for current scope.