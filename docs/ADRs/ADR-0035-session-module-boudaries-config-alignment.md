# ADR-0035: Isomorphic Configuration Schema and Engine Boundary Alignment (Updated)

**Status:** Accepted
**Date:** 2026-05-08

---

## Context

Following the reorganization of session components in ADR-0034, the system
achieves a clear separation between:

- Data ingestion
- Task definition
- Batch execution
- Macro-level orchestration

However, prior to this ADR, the configuration schema (`SessionConfig`) and Hydra
YAML structure did not reflect these boundaries. A catch-all `runtime` block mixed
concerns such as:

- Macro orchestration (e.g., epochs, early stopping)
- Micro execution (e.g., AMP, logit adjustment)

This violated dependency injection principles and obscured the relationship
between configuration and execution modules.

---

## Decision

We adopt an **isomorphic configuration schema** where the configuration tree
mirrors the execution architecture.

### 1. Schema Restructuring

The session configuration is partitioned into explicit domains aligned with
system modules:

- **session.data_loader**
  Owns ingestion mechanics (patch size, batch size)

- **session.engine_exec**
  Owns batch-level execution behavior (e.g., AMP, logit adjustment)

- **session.engine_optim**
  Owns optimization state (optimizer, learning rate, scheduler, clipping)

- **session.engine_tasks**
  Owns training objectives (loss types, class exclusion, weighting)

- **session.orchestration**
  Owns macro-level training control:
  - monitoring (metrics, early stopping)
  - curriculum (phase scheduling)

> NOTE: Naming reflects execution ownership rather than conceptual grouping,
> strengthening alignment between config and runtime modules.

---

### 2. Elimination of Catch-all Runtime

The previous `session.runtime` structure is removed entirely.

Responsibilities are redistributed as follows:

| Previous Runtime Responsibility | New Location |
|--------------------------------|--------------|
| AMP / precision                | engine_exec  |
| logit adjustment               | engine_exec  |
| optimization clipping          | engine_optim |
| monitoring                     | orchestration.monitor |
| scheduling                     | orchestration.curriculum |

This removes cross-domain coupling and enforces strict dependency boundaries.

---

### 3. Engine Boundary Alignment

Execution modules now consume only their dedicated configuration domains.

- BatchEngine consumes:
  - engine_exec
  - engine_optim
  - engine_tasks

- Orchestration consumes:
  - orchestration.monitor
  - orchestration.curriculum

This ensures mathematical isolation between:

- batch execution
- optimization
- orchestration logic

---

### 4. Orchestration Model Change

Training duration is no longer defined by a global schedule.

Instead:

- Epoch control is fully phase-driven
- Total training length is defined as:

    total_epochs = sum(phase.num_epochs)

This consolidates scheduling into curriculum definitions and simplifies
orchestration semantics.

---

### 5. Hydra Configuration Structure

The Hydra configuration tree mirrors the execution architecture:

    session/
      data_loader/
      engine_exec/
      engine_optim/
      engine_tasks/
      orchestration/

Each module owns its configuration exclusively, enabling:

- clear dependency injection
- modular reasoning
- independent extensibility

---

### 6. User Entry Point Strategy

The user-facing configuration (`settings.yaml`) acts as the primary
configuration entry point.

At this stage:

- The internal configuration structure is exposed directly
- Usability is achieved through structured layout and documentation
- No abstraction or translation layer is introduced

Rationale:

- Configuration surface is still evolving
- Premature abstraction would introduce coupling and maintenance overhead
- Hydra composition works best with structurally aligned schemas

---

### 7. Future Direction (Non-binding)

A user abstraction layer may be introduced once the configuration stabilizes.

Potential future improvements include:

- simplified user-facing blocks (e.g., optimization, task)
- intent-based parameters (e.g., max_epochs)
- mapping layer from user surface to internal schema

This is intentionally deferred.

---

## Consequences

### ✅ Positive

- Strict dependency injection across modules
- True config-to-code isomorphism
- Improved extensibility and maintainability
- Clear separation of concerns
- Hydra compatibility preserved without schema duplication

---

### ⚠️ Trade-offs

- User-facing config exposes internal concepts
- Slightly higher cognitive load for new users

Mitigation:

- Extensive inline documentation in `settings.yaml`

---

### ⚠️ Migration Impact

Removed legacy fields:

- session.runtime
- session.loader
- session.phases

Replaced with:

- session.engine_exec
- session.engine_optim
- session.engine_tasks
- session.orchestration

---

## Summary

The configuration system now serves as a 1:1 blueprint of the execution
architecture. Catch-all runtime structures are eliminated, and configuration
domains are strictly aligned with module boundaries.

At the same time, the internal structure is intentionally exposed to users
with enhanced documentation, delaying abstraction until the system stabilizes.

This balances architectural clarity, extensibility, and practical usability.
