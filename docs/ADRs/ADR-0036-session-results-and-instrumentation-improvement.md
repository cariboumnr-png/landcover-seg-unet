## ADR-0036: Unify Session Result Model and Decouple Instrumentation Layers

**Status:** Accepted
**Date:** 2026-05-15

---

### Context

Following ADR-0032 (Unified Tracking Facade) and ADR-0033 (Dashboard
Callback and Preview Integration), the session instrumentation and
execution stack evolved significantly. While these changes successfully
introduced vendor-agnostic tracking and event-driven visualization,
they also exposed underlying inconsistencies in the session result
model and instrumentation boundaries.

Several issues emerged:

- **Ambiguity in result semantics**
  Different result types (e.g., TrainingSessionStep, EpochResults,
  TrainerEpochResults) conflated multiple lifecycles, including:
  - batch vs step updates
  - epoch aggregation
  - session-level summaries

- **Inconsistent data flow across layers**
  Orchestration, engine, pipelines, and callbacks consumed and
  transformed results differently, leading to redundant conversions
  and unclear contracts.

- **Callback system becoming overloaded**
  The existing callback design mixed responsibilities, including:
  - scalar tracking
  - preview generation
  - artifact management

- **Blurring between tracking and visualization concerns**
  The instrumentation layer used the term "tracking" to represent both
  logging and dashboarding, creating ambiguity in intent and ownership.

- **Formatter layer complexity**
  The preview utility evolved into a large, file-oriented module that
  mixed formatting, IO, and visualization concerns.

These issues made the system harder to reason about and conflicted with
the decoupled architecture established in earlier ADRs (particularly
ADR-0034).

---

### Decision

We have refactored the session result model and instrumentation layers
to enforce clear data contracts, reduce coupling, and align with an
explicit execution → observation architecture.

#### 1. Unify Session Result Model

A new, consistent result hierarchy was introduced:

- **SessionStepSummary (formally TrainingStep)**
  - Immutable snapshot exposed by runners
  - Represents session-level progress and best metrics

- **SessionStepResults (formarlly EpochResults)**
  - Container for all results produced during a single epoch

- **Specialized result types**
  - TrainStepResults → training updates (batch-driven)
  - ValStepResults → aggregated validation metrics
  - InferStepResults → inference predictions and outputs (*new*)

Legacy result types were removed or renamed accordingly.

This establishes a clear separation between:
- incremental updates (training)
- aggregated metrics (validation)
- aggregated metrics and output artifacts (inference)
- summarized session state (runner-facing)

---

#### 2. Align System-Wide Data Contracts

All system components were updated to use the unified result model:

- Epoch engine returns SessionStepResults
- Runners yield SessionStepSummary
- Callbacks consume strongly typed result objects per phase

This removes the need for intermediate adapters and enforces a consistent
data flow across:
- engine
- orchestration
- pipelines
- instrumentation

---

#### 3. Simplify and Specialize Callback System

The callback system was redesigned to improve separation of concerns.

- Removed monolithic callbacks:
  - TrackingCallback
  - PreviewCallback

- Introduced specialized callbacks:
  - TrainTrackingCallback → scalar metrics
  - ValTrackingCallback → validation metrics
  - InferTrackingCallback → image and inference outputs

Callbacks now operate strictly as lifecycle observers and delegate all
external interactions to tracking backends.

---

#### 4. Decouple Tracking from Dashboards

The instrumentation layer was reorganized:

- Renamed:
  landseg.session.instrumentation.tracking → dashboards

- Introduced a clearer abstraction:
  - **Callbacks** = lifecycle orchestration
  - **Dashboards (trackers)** = external logging and persistence

BaseTracker remains the unified interface defined in ADR-0032, but its
role is now explicitly framed as a dashboard integration layer rather
than generic tracking.

---

#### 5. Refactor Formatter Layer

The preview utility was decomposed into smaller, focused modules:

- **renderer.py**
  - Handles tensor-to-image conversion (e.g., colorize)

- **report.py**
  - Generates structured evaluation outputs (e.g., IoU reports)

The legacy file-based preview exporter was removed.

This separates:
- formatting logic (pure)
- lifecycle orchestration (callbacks)
- persistence (trackers)

---

#### 6. Maintain Phase-Agnostic Batch Execution

No functional changes were made to BatchEngine responsibilities.

- BatchEngine remains responsible for:
  - loss computation
  - metric updates
  - inference aggregation

- Phase-specific semantics were clarified to belong to:
  - Trainer (training policy)
  - Evaluator (validation/inference policy)

Non-functional changes were made to remove misleading
phase-label comments within BatchEngine, reinforcing that it is a
phase-agnostic execution component.

---

### Consequences

#### ✅ Achieved Outcomes

- A consistent and explicit session result model
- Strongly typed contracts across all execution layers
- Simplified and modular callback system
- Clear separation between:
  - execution (engine)
  - orchestration (runner/policies)
  - observation (callbacks)
  - persistence (dashboards)
- Reduced coupling between instrumentation and engine logic
- Improved extensibility for future tracking and visualization features

---

#### ✅ Architectural Improvements

The system now follows a clear data flow:

    Batch Execution → Result Aggregation → Session Results → Callbacks → Dashboards

Responsibilities are explicitly scoped:
- Engine: produces results
- Orchestration: coordinates execution
- Callbacks: observe lifecycle events
- Dashboards: persist metrics and artifacts

This aligns with the design principles established in ADR-0034 and
extends the abstraction boundaries introduced in ADR-0032.

---

#### ⚠️ Trade-offs

- **Implementation complexity**
  The refactor affected multiple system layers, requiring coordinated
  changes and increased development effort.

- **Ongoing evolution**
  The design remains subject to refinement as the system matures.
---

### Summary

The session result model and instrumentation layers have been refactored
to enforce clear contracts, reduce ambiguity, and align with a modular,
event-driven architecture.

- Result types are now explicit and lifecycle-aware
- Instrumentation is cleanly separated into callbacks and dashboards
- Formatting utilities are modular and composable
- Execution and observation responsibilities are clearly defined

This provides a stable foundation for extending training workflows,
instrumentation, and reporting capabilities without increasing coupling
or complexity.