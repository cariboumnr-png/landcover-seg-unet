# ADR-0033: Implement Dashboard Callback and Preview Integration

**Status:** Proposed
**Date:** 2026-05-06

## Context
During the implementation of ADR-0031 (Event Dispatcher) and ADR-0032 (Unified Tracking Facade), the foundational routing for metrics and telemetry was established. The `TrackingCallback` now successfully bridges training events and scalar metrics to the underlying tracking backends (TensorBoard/MLflow). 

However, the original intent of ADR-0033—to push visual tracking to the dashboard—remains partially unfulfilled. Our existing inference preview utility (`src/landseg/session/instrumentation/exporters/preview.py`) currently operates as an isolated file-saver. With the tracking facade now supporting a `log_image` API, this visual preview logic needs to be fully integrated into the event-driven lifecycle so that stitched image grids (Landsat + DEM + Ground Truth + Prediction) are automatically tracked epoch-by-epoch alongside scalar metrics.

## Decision
Instead of creating a standalone `DashboardCallback` as originally proposed (since metric routing is now handled by the `TrackingCallback`), we will extend the tracking instrumentation to explicitly handle image telemetry:

1. **Preview Extraction:** Refactor the existing preview logic to generate image arrays/buffers in memory rather than writing directly to disk.
2. **Event Hook Integration:** Utilize the `on_eval_end` (or `on_epoch_end`) lifecycle hooks to trigger the preview generation.
3. **Telemetry Routing:** Push the generated preview grid to the active `BaseTracker` instance via the `log_image(key, image, step)` method. 
4. **Separation of Concerns:** This visual logging can either be integrated into the existing `TrackingCallback` or isolated into a dedicated `PreviewTrackingCallback` to allow users to opt-in or opt-out of heavy image logging during rapid sweeps.

## Consequences
* **Definition of Done:** Launching a training session with the appropriate callback attached results in visual inference previews populating the TensorBoard/MLflow UI automatically at the end of evaluation epochs.
* **Deprecation:** Legacy code relying on manual `.png` file saving to local disk from the preview exporter can be deprecated in favor of the artifact tracking store.
* **Performance:** Writing images to the tracker adds overhead. Scoping this to `on_eval_end` ensures it does not block the high-frequency batch training loops.
