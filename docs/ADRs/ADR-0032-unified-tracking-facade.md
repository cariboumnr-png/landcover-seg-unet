# ADR-0032: Unified Tracking Facade for TensorBoard and MLflow

**Status:** Accepted
**Date:** 2026-05-06

---

## Context

A robust experiment tracking and dashboarding solution was required.

- Local development and sweeps rely on TensorBoard due to its environment-agnostic
nature.
- Production on Databricks standardizes on MLflow.

Direct integration of either TensorBoard or MLflow into callbacks or orchestration
would introduce vendor lock-in. A unified abstraction layer was required to
standardize metric and artifact logging.

---

## Decision

A **Facade / Adapter pattern** was introduced for experiment tracking.

---

### 1. BaseTracker Facade

A unified interface was added under:

`landseg.session.instrumentation.tracking`

It defines the following methods:

- log_scalar(key, value, step)
- log_params(key, value)
- log_image(key, image, step)
- log_artifact(path)
- flush()
- close()

This interface is fully decoupled from any third-party tracking system.

---

### 2. TensorBoard Backend (Default)

`TensorBoardTracker` is implemented using:

`torch.utils.tensorboard.SummaryWriter`

It is the default backend for:

- Local development
- Interactive experimentation
- Lightweight sweeps

It mirrors the facade interface without exposing TensorBoard APIs externally.

---

### 3. MLflow Backend (Optional)

`MLFlowTracker` provides MLflow integration.

Mappings:

- scalars → mlflow.log_metric
- params → mlflow.log_param
- artifacts → mlflow.log_artifact

Constraints:

- No model registry integration
- No autologging
- No lifecycle orchestration

This keeps the implementation minimal and Databricks-compatible.

---

### 4. Tracking Integration via Callback

A `TrackingCallback` bridges training events to the tracker interface:

training session events → tracker calls

This ensures:

- Tracking is decoupled from training logic
- Instrumentation remains modular

---

## Consequences

### Achieved Outcomes

- A standalone tracking module exists in `landseg.session.instrumentation.tracking`
- A unified `BaseTracker` interface defines the contract
- Two interchangeable backends exist:
  - TensorBoard (default)
  - MLflow (optional)
- No third-party tracking APIs leak into orchestration or engine layers
- Tracking is integrated via `TrackingCallback`

---

### Environment Independence

#### Local Development

- TensorBoard backend
- Writes to `runs/`
- Optimized for fast iteration

#### Databricks Production

- MLflow backend
- Logs directly to MLflow UI

The same training code runs unchanged in both environments.

---

### Architectural Improvements

This work also introduced structural improvements:

- Instrumentation decoupled from orchestration
- Simplified observer interface (`SessionObserverLike`)
- Introduced protocol separation in `session.engine.protocols`
- Reduced coupling between:
  - engine
  - orchestration
  - instrumentation

---

## Limitations (Intentional)

The MLflow backend is intentionally minimal:

- No model registry
- No structured artifact hierarchy
- No experiment lifecycle customization
- No automatic config logging
- No multi-backend composition

These are deferred to future design iterations.

---

## Summary

A vendor-agnostic experiment tracking layer has been established.

- TensorBoard remains the default for local workflows
- MLflow supports Databricks production usage
- Training logic is fully decoupled from tracking systems
- The architecture is now extensible for future MLOps features without increasing
complexity