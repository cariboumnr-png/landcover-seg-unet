# ADR‑0009: Introduce a Built‑In “Overfit Test” Profile
- **Status:** Accepted
- **Date:** 2026‑03-02
- **Updated:** 2026‑03-05

> this ADR is pratically combined with [ADR-0008](./ADR-0008-knob-inventory-control-surface.md) during implementation.
## Context

The project maintains a complex end‑to‑end pipeline:
- **Data pipeline:** tiling, block building, normalization, and dataset splitting.
- **Model pipeline:** multi‑head UNet/UNet++ backbones, optional domain conditioning, logit adjustment, and multiple optimizer/scheduler configurations.
- **Training controller:** multi‑phase curricula, logit adjustment, head masking, patience logic, checkpointing, and logging.

During development or debugging, contributors frequently need a minimal correctness test ensuring that:

- Data loading functions correctly for at least a single block.
- The model can forward‑propagate and back‑propagate without error.
- The loss decreases and the model can memorize a tiny sample.
- The trainer, controller, scheduler, logging, and checkpointing work end‑to‑end.

Today, running such a minimal “overfit a single block” test requires manual configuration edits, disabling multiple training features, and constructing a handcrafted micro dataset. Because many configuration knobs are real (dropout, logit adjustment, weight decay, domain conditioning, etc.), and some others are *not implemented* (mixup, cutmix, rotate, crop), developers lack a clean, reproducible way to run an overfit test.

A standardized, supported **overfit test profile** improves developer experience, reliability, and CI validation.

## Decision

We will add an official Hydra profile named **`overfit_test`**, implemented as
a dedicated override file:

```
configs/profiles/overfit_test.yaml
```

This profile configures the system to intentionally overfit a tiny dataset to validate correctness across the entire pipeline.

### The profile will:

#### 1. Reduce dataset size
- Replace train/val split with a deterministic, single‑block (or tiny) JSON index.
- Disable block scoring and spatial validation.

#### 2. Disable regularization
- `p_drop = 0.0` (DoubleConv dropout)
- `weight_decay = 0.0`
- `grad_clip_norm = null`

#### 3. Disable priors and adjustments
- `logit_adjust.alpha = 0.0`
- Disable logit adjustment for all phases (`train`, `val`, `test`)
- Loss class‑balancing/α‑weighting disabled

#### 4. Disable domain conditioning
- `conditioning: none`

#### 5. Disable augmentation
- No flips
- All other augmentations remain off (not implemented)

#### 6. Deterministic data loader
- `shuffle = false`
- `num_workers = 0`
- Fully deterministic + seeded

#### 7. Scheduler & evaluation
- `eval_interval = null` (no evaluation)
- `patience_epochs = null` (no early stopping)
- `max_epoch = 1000` to guarantee overfitting

#### 8. Disable AMP
- `amp = false` to simplify reproducibility and debugging

#### 9. Minimal batch size
- `batch_size = 1`

## Alternatives Considered

#### **Option A — Modify the base config directly**
**Rejected.**
Would pollute the default configuration, risks accidental use, and mixes debug settings into primary workflows.

#### **Option B — Manual overrides**
**Rejected.**
Inconsistent, error‑prone, and impossible to scale to CI. Developers frequently forget to disable features like dropout or domain conditioning.

#### **Option C — Add a CLI flag (`--overfit`)**
**Possible**, but rejected.
Hydra profiles are already the project’s established and cleaner method for mode switching.

## Consequences

### Positive
- Provides zero‑effort correctness testing for all contributors.
- Enables simple CI regression checks ensuring the pipeline converges on trivial data.
- Prevents failures due to lingering regularization or conditioning settings.
- Reduces confusion about implemented vs. non‑implemented augmentation knobs.

### Negative
- Introduces one additional config file (low maintenance cost).
- Developers must discover the profile (addressed via documentation).

## Implementation

1. Created `configs/profile/overfit_test.yaml`.
2. Added documentation to `README` under **Developer Quick Checks**.
3. The test is now invoked via:
   ```
   python cli/main.py profile=overfit_test
   ```