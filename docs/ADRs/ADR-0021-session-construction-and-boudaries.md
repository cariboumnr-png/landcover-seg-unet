# ADR-0021: Session Construction, Component Ownership, and Runtime-State Boundaries

- **Status:** Proposed
- **Date:** 2026-04-15
- **Related ADRs:** ADR-0019, ADR-0020

## Context

Following ADR-0019 (Shared Execution Core) and ADR-0020 (Dedicated Evaluator Engine),
core execution mechanics and policy-level orchestration have been successfully
separated. Trainer and evaluator engines now act as thin policy layers over a
shared batch execution core, with unified runtime state and component contracts.

However, this refactor has exposed a new architectural tension at the *session
construction* layer:

- The CLI training pipeline is currently responsible for assembling the entire
  session end-to-end.
- Session component construction, runtime-state initialization, callback wiring,
  and engine instantiation are explicitly orchestrated at the CLI call site.
- Runtime configuration (`RootConfig`) is threaded through nearly every step of
  session assembly, creating implicit coupling and order sensitivity.

In practice, the CLI currently performs the following responsibilities:

1. Constructing the engine component bundle (data, callbacks, heads, losses,
   metrics, optimization)
2. Initializing the runtime state from components and runtime flags
3. Wiring callbacks to runtime state, configuration, and device
4. Constructing the batch execution engine from the initialized state
5. Instantiating trainer and evaluator policy engines

This makes the CLI pipeline difficult to reason about, hard to test in isolation,
and unsuitable as a stable public entrypoint.

### Contributing factor: configuration-driven session shape

A key reason for this situation is historical:

- The project previously relied on a **Hydra + dataclass-based configuration
  system**, where configuration dataclasses and YAML structure were treated as
  the primary source of truth.
- During recent refactors, care was taken not to modify the configuration module
  structure, resulting in the *session module being shaped to match the existing
  config layout*.
- This inverted the intended dependency direction: runtime architecture was made
  to conform to configuration shape, rather than configuration expressing the
  needs of the runtime system.

This ADR formally acknowledges that mismatch and establishes a new direction.

## Decision

### Session-first construction boundary

The system will adopt a **session-first architecture**, where:

- The `session` module owns the construction and assembly of runtime elements
- The CLI is reduced to a thin entrypoint that *requests* a session, rather than
  assembling one
- Configuration objects serve as *inputs* to session construction, not as
  structural drivers of runtime shape

Concretely:

- Session assembly (components, state, callbacks, engines) is moved behind a
  single, well-defined construction boundary
- The CLI no longer performs stepwise wiring of runtime internals

### Explicit ownership and lifecycle boundaries

This ADR establishes clear ownership rules:

- **Components** are constructed once and owned by the session
- **Runtime state** is initialized exactly once per session from components
- **Callbacks** are bound to runtime state and configuration during session
  construction, not lazily or implicitly
- **Execution and policy engines** receive fully-initialized state and do not
  participate in assembly

This removes construction order dependencies and makes lifecycle transitions
explicit and auditable.

### Configuration refactoring (anticipated)

To support this change, we explicitly acknowledge that **major refactoring of the
`configs/` module is expected** in follow-up work:

- Hydra YAML structure and dataclass shapes will be revisited
- Configuration will be reorganized to reflect *session-level concerns*, rather
  than mirroring historical builder order
- The session module will define what configuration it consumes, rather than
  adapting itself to existing config layout

This ADR does not prescribe the new configuration structure, but establishes the
architectural intent to reverse the current dependency direction.

## Consequences

### Positive

- CLI pipelines become thin, declarative entrypoints
- Session construction is centralized, testable, and consistent
- Component and state lifecycles are clearly defined
- Trainer and evaluator engines remain focused on policy, not wiring
- Configuration becomes an expression of runtime needs, not a driver of
  architecture

### Costs

- Refactoring effort in `session/` and `configs/`
- Temporary duplication or adapter layers during migration
- Short-term churn in CLI training and evaluation scripts

These costs are accepted as necessary to stabilize the architecture long-term.

## Non-Goals

This ADR does **not**:

- Redesign batch execution or policy engines
- Introduce new training or evaluation behaviors
- Mandate a specific configuration backend (Hydra remains acceptable)
- Define a public evaluation CLI

## Follow-up Work

- Introduce a session-level construction API (e.g. `build_training_session`)
- Migrate existing CLI pipelines to use the new session boundary
- Refactor `configs/` to align with session-level concerns
- Remove legacy implicit wiring between configuration, components, and state

## Related Code Areas

- `session/`
- `session/components/`
- `session/engine/`
- `session/runner/`
- `cli/pipelines/`
- `configs/`