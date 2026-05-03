# ADR-0033: Implement Dashboard Callback and Preview Integration

**Status:** Proposed
**Date:** 2026-05-03

## Context
With the event dispatcher (ADR-0031) and the unified tracker (ADR-0032) in place, we lack the final bridge: the actual 
observer that captures epoch-level metrics and visual previews, and routes them to the dashboard.

Additionally, our existing inference preview utility (`src/landseg/session/instrumentation/exporters/preview.py`) currently
operates as an isolated file-saver. It needs to be wired into the modern visualization pipeline.

## Decision
We will implement the `DashboardCallback` to drive all visual and metric tracking:

1.  **`DashboardCallback` Implementation:** Create a new callback inheriting from `BaseCallback`. It will be instantiated with
an `ExperimentTracker` instance.
2.  **Metric Routing:** On the `on_epoch_end` hook, the callback will extract scalar metrics (train loss, validation IoU, etc.)
from the payload and push them to the tracker.
3.  **Preview Stitching:** On the `on_eval_end` (or `on_epoch_end`) hook, the callback will utilize the existing `preview.py`
logic to generate the stitched image grid (Landsat + DEM + Ground Truth + Prediction) and push it directly to the tracker using
the `log_image` API.

## Consequences
* **Definition of Done:** Launching a training run with the `DashboardCallback` attached results in a fully populated TensorBoard
dashboard, including both scalar loss curves and epoch-by-epoch visual inference previews.
* **Deprecation:** Any legacy code manually saving `.png` files to disk from the preview exporter can be refactored or removed in
favor of the dashboard artifact store.