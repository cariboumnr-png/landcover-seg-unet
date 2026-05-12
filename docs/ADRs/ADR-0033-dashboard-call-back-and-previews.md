# ADR-0033: Implement Dashboard Callback and Preview Integration

**Status:** Accepted
**Date:** 2026-05-11

## Context
During the implementation of ADR-0031 (Event Dispatcher) and ADR-0032 (Unified
Tracking Facade), the foundational routing for metrics and telemetry was
established. The `TrackingCallback` now successfully bridges training events and
scalar metrics to the underlying tracking backends (TensorBoard/MLflow).

However, the original intent of ADR-0033—to push visual tracking to the
dashboard—remains partially unfulfilled. Our existing inference preview utility
(`src/landseg/session/instrumentation/exporters/preview.py`) currently operates
as an isolated file-saver. With the tracking facade now supporting a `log_image`
API, this visual preview logic needs to be fully integrated into the event-driven
lifecycle so that stitched image grids (Landsat + DEM + Ground Truth + Prediction)
are automatically tracked epoch-by-epoch alongside scalar metrics.


## Decision

Instead of creating a standalone `DashboardCallback`, we have extended the
existing callback system to fully support image telemetry via a dedicated
`PreviewCallback` integrated with the dispatcher lifecycle.

The implemented solution includes:

1. **In-Memory Preview Generation**
   The legacy disk-based preview exporter was refactored into a pure formatting
   layer. Preview images are now generated in memory using `stitch_patches`,
   producing numpy arrays suitable for direct consumption by tracking backends.

2. **Event-Driven Integration**
   Preview generation is triggered during the training lifecycle via the
   `on_train_step_end` hook. This hook ensures that preview generation occurs
   only when inference results are available, aligning with evaluation epochs
   while remaining compatible with flexible training policies.

3. **Telemetry Routing via Trackers**
   Generated preview mosaics (per head, for both labels and predictions) are
   pushed to active trackers using the `BaseTracker.log_image` API. This enables
   seamless integration with TensorBoard and MLflow without backend-specific logic.

4. **Multi-Head and Phase-Aware Logging**
   The implementation supports multiple prediction heads and logs images with
   phase-aware keys (e.g., `{phase}_{head}_predictions`), ensuring clarity and
   compatibility with multi-phase or curriculum-based training workflows.

5. **Separation of Concerns**
   The design cleanly separates responsibilities:
   - Runtime: aggregates inference outputs (`infer_out`)
   - Formatter: reconstructs spatial mosaics (`stitch_patches`)
   - Callback: orchestrates preview generation (`PreviewCallback`)
   - Tracker: handles persistence (`BaseTracker` implementations)

## Consequences

**Definition of Done (Achieved):**
- Running a training session with the dispatcher configured results in visual
inference previews automatically appearing in TensorBoard/MLflow at the end of
evaluation epochs.

**Deprecation:**
- Legacy file-based preview exporting (PNG writing to local disk) is no longer
required and is superseded by tracker-based artifact logging.

**Performance Consideration:**
- Image logging introduces additional overhead. This is mitigated by:
  - Restricting preview generation to inference-enabled steps
  - Centralizing logging within a callback that can be optionally disabled

**Extensibility:**
- The callback-based design allows future enhancements such as:
  - Custom preview layouts (e.g., raw imagery + GT + predictions panels)
  - Subsampling or throttling image logging for large-scale experiments
  - Additional visualization modalities (e.g., uncertainty maps)

## Summary

ADR-0033 is fully implemented with a cleaner and more extensible design than
originally proposed. Visual telemetry is now a first-class, event-driven component
of the training pipeline, fully integrated with the unified tracking system.
