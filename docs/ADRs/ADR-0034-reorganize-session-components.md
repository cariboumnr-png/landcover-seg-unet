# ADR-0034: Reorganize and Scope Session Components

**Status:** Proposed  
**Date:** 2026-05-06  

## Context
The `landseg.session.components` module has historically served as a catch-all for various elements required to run a session. Currently, it houses three distinctly different architectural concerns:
1. `task` (heads, losses, metrics)
2. `data` (datasets, dataloaders)
3. `optim` (optimizers, learning rate schedulers)

As the execution engine and orchestration layers have matured, this grouping no longer accurately reflects the boundaries of the system. Lumping these together blurs the lines of responsibility, making dependency management harder and violating the decoupled design principles established in recent ADRs.

Furthermore, the components within `optim` exhibit conflicting lifecycles. Optimizers are static, stateful objects bound to the model parameters. Schedulers are temporal policies that must often be re-instantiated dynamically at phase boundaries during curriculum training to align with new step counts.

## Decision
We will reorganize the session module to align the directory structure with the functional boundaries of the execution engine. 

1. **Promote Dataloading (`data`):**
   * Move `dataset.py` and `loader.py` out of components. 
   * They will be reclassified as a first-class module under `landseg.session.data`, acting as the adapter layer between the geospatial foundations and the session.

2. **Isolate Engine Components (`task`):**
   * Retain `heads`, `loss`, and `metrics` as the true definition of a "component." 
   * These will be relocated to `landseg.session.engine.components` as they directly construct the execution task.

3. **Relocate and Decouple Optimization (`optim`):**
   * Move the optimization logic out of components and into `landseg.session.engine.optimization`.
   * To resolve lifecycle conflicts, the construction of optimizers and schedulers will be strictly decoupled.
   * Session builders will instantiate the optimizer once.
   * Phase runners (e.g., `CurriculumRunner`) will be granted the authority to dynamically invoke the scheduler factory at the start of distinct training phases, passing the persistent optimizer into the newly created temporal scheduler.

## Consequences
* **Architectural Clarity:** The definition of a "session component" is now strictly limited to the neural network task elements (heads, losses, metrics).
* **Curriculum Flexibility:** Decoupling the scheduler construction from the optimizer construction fully supports complex, multi-phase curriculum orchestration without carrying over stale learning rate momentum.
* **Refactoring Overhead:** Import paths across the orchestrator, factory, and configuration layers (`configs/session/components/`) will need to be updated to reflect the new namespace mapping.
