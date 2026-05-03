# ADR-0032: Unified Tracking Facade for TensorBoard and MLflow

**Status:** Proposed
**Date:** 2026-05-03

## Context
We need a robust dashboarding solution for experiment tracking. In local development and sweeps, TensorBoard
is the preferred, environment-agnostic tool. In production on Databricks, MLflow is the standard. 

If we embed TensorBoard or MLflow API calls directly into our callbacks or orchestration code, we create 
vendor lock-in. We need an abstraction layer that standardizes how metrics and images are persisted to
external visualization tools.

## Decision
We will implement an Adapter/Facade pattern for experiment tracking:

1.  **`ExperimentTracker` Facade:** Create a unified tracking interface in `src/landseg/session/instrumentation/tracking/`.
This class will expose generic methods like `log_scalar(name, value, step)` and `log_image(name, tensor, step)`.
2.  **Native TensorBoard Backend:** By default, the `ExperimentTracker` will route these generic calls to
`torch.utils.tensorboard.SummaryWriter`.
3.  **Optional MLflow Backend:** The tracker will accept a configuration flag (e.g., `use_mlflow=True`). When enabled,
it will mirror the routing of metrics and artifacts to the `mlflow` python API.

## Consequences
* **Definition of Done:** A standalone `ExperimentTracker` class exists that can write to a local `runs/` directory for
TensorBoard, and optionally to MLflow, without exposing the underlying 3rd-party APIs.
* **Environment Independence:** The tool can be run anywhere. Colleagues can view rich TensorBoard dashboards locally,
and Databricks runs will automatically populate the MLflow UI.