# ADR‑0009: Knob Inventory & Control Surface
**Status:** Proposed (pre‑work for [ADR‑0008](./ADR-0008-knob-inventory-control-surface.md))
**date:**2026‑03-02

## Context

Before introducing a standardized `overfit_test` Hydra profile, we need a clear view of all *implemented* knobs (what exists today), what’s *missing*, and which knobs should be **public** (user‑facing) vs. **internal** (config‑only for CI/profiles).
This ADR inventories existing controls across **data**, **model**, **training/controller**, and **runtime** surfaces, calling out gaps that prevent determinism (e.g., lack of explicit “off” switches or seeds).

## Decision

1. **Adopt a “Control Surface” taxonomy** for configuration keys:
   - **Public**: Stable, documented, safe for users to tweak.
   - **Internal**: Supported in configs (so Hydra profiles/CI can set them) but not documented for general users.
   - **Experimental**: Present but default‑off and not documented; subject to change.

2. **Introduce deterministic “off” paths** for any behavior that is *on by default* and affects reproducibility—even if those switches remain internal only.

3. **Document a canonical checklist** (below) to validate the control surface before adopting ADR‑00X.

---

## Scope of Inventory (What Exists vs. What’s Missing)

### A) Data / Dataprep / Splits

**Implemented knobs & artifacts**
- Input/output paths for fit/test rasters.
- Block cache directories and JSON index files.
- Block validation threshold (`blk_thres_*`).
- Global per‑channel normalization stats.
- Train/val block splits and label-count metadata.
- Block scoring and spatial selection for validation sets.
- Grid creation and persistence.
- Domain tile maps (categorical raster → PCA‑reduced vectors).
- Rebuild flags: `rebuild_all`, `remap`, `rebuild_blocks`, `renormalize`, `rebuild_split`.

**Gaps to close**
- No standardized single‑block or micro‑split generator for overfit testing.
- No explicit “disable scoring/valid‑block filtering” switch aside from setting thresholds manually.

---

### B) Dataset / Dataloaders

**Implemented knobs**
- Batch size (loader meta).
- Patch sampling (`patch_per_blk`), test grid layout for inference.
- A minimal augmentation flag (e.g., flips) toggled by train/val/test mode.

**Gaps to close**
- Missing deterministic loader knobs:
  `shuffle=false`, `num_workers=0`, `seed=<int>`, `deterministic=true`.
- Missing global augmentation disable switch (`augment.enable=false`).

---

### C) Model (Backbones, Heads, Conditioning)

**Implemented knobs**
- Backbone selection (UNet/UNet++) and hyperparameters (`in_ch`, `base_ch`).
- Block‑level knobs: `p_drop`, normalization mode (`bn|gn|ln|none`), group‑norm groups.
- Multi‑head model: per‑head class counts, logit adjustment buffers, head activation/freeze.
- Domain conditioning modes: none / concat / FiLM / hybrid, with flags for ids/vec/MLP usage.
- Output clamping range.

**Gaps to close**
- Confirm `conditioning.mode=none` is fully neutral (no latent conditioning ops).
- Ensure `p_drop=0.0` becomes globally effective.

---

### D) Training Engine / Controller

**Implemented knobs**
- Phase structure: `{name, num_epochs, head specs, logit-adjust flags, lr_scale}`.
- Schedule: `max_epoch`, `eval_interval`, `patience_epochs`.
- Logit adjustment at runtime: per‑phase flags + global alpha setter.
- Optimization: scheduler hooks, optional gradient clipping.
- AMP toggle (`use_amp`) and scaler/autocast pipeline.

**Gaps to close**
- Need neutral positions for:
  `eval_interval=null`, `patience_epochs=null`, `grad_clip_norm=null`.
- Ensure optimizer exposes `weight_decay` and can be set to `0.0`.
- Confirm that AMP can be fully disabled (no autocast/scaler).

---

### E) Runtime Determinism

**Implemented**
- Only preview‑palette seeding.

**Missing**
- A global seeding system covering Python, NumPy, Torch, and CUDA.
- Deterministic backend toggles.
- Internal config keys such as:
  `runtime.seed`, `runtime.deterministic=true`.

---

## Control Surface: Public vs. Internal

**Public (documented)**
- Dataset/cache paths, grid selection, backbone choice, domain config.
- Major training knobs: `max_epoch`, `eval_interval`, learning rate, weight decay.
- AMP toggle.

**Internal (not documented)**
- `augment.enable`
- `dataloader.shuffle`, `dataloader.num_workers`, `dataloader.seed`
- `runtime.seed`, `runtime.deterministic`
- `conditioning.mode`
- `logit_adjust.*`, `logit_adjust.alpha`
- `p_drop`
- `grad_clip_norm`
- `patience_epochs`
- Micro‑split generation

**Experimental**
- Future augmentation ops
- Advanced schedulers or unreleased loss functions

---

## Checklist: Explicit Items to Implement/Verify

### Data / Splits
- [ ] Add internal single‑block `{train,val}` split generator.
- [ ] Ensure `blk_thres_fit=0` and associated logic cleanly bypass validation.

### Dataloader & Augmentation
- [ ] Add deterministic loader config:
  `shuffle=false`, `num_workers=0`, `seed=<int>`, `deterministic=true`.
- [ ] Add `augment.enable=false` to disable all augmentation.

### Model / Regularization / Conditioning
- [ ] Make sure `p_drop=0.0` is honored everywhere.
- [ ] Confirm `conditioning.mode=none` fully disables domain features.
- [ ] Ensure logit adjustment can be globally neutralized using `alpha=0.0`.

### Training / Scheduler
- [ ] Allow `weight_decay=0.0`.
- [ ] Allow `grad_clip_norm=null`.
- [ ] Allow `eval_interval=null`.
- [ ] Allow `patience_epochs=null`.

### AMP / Precision
- [ ] Ensure `use_amp=false` produces no autocast or scaling.

### Batch Size
- [ ] Ensure `batch_size=1` flows safely end‑to‑end.

### Global Seeds
- [ ] Add internal `runtime.seed` and set all relevant RNG seeds.
- [ ] Add `runtime.deterministic=true`.

### Controller & Checkpointing
- [ ] Ensure checkpoints and previews function even with validation off.

---

## Consequences

**Positive**
- Establishes a clear, maintainable control surface.
- Prevents accidental nondeterministic behavior.
- Enables a reliable `overfit_test` configuration for CI and debugging.

**Negative**
- Slightly larger internal config surface.
- Public/internal distinction must be maintained in docs.

---

## Implementation Plan

1. Add internal configs:
   `runtime.seed`, `runtime.deterministic`, loader knobs, global `augment.enable`, expose `weight_decay`.
2. Add global seeding logic to the CLI entry point.
3. Verify all “neutral paths” (dropout off, conditioning none, logit adjustment off, early‑stop off).
4. Implement micro‑split writer.
5. Document public knobs; keep internal knobs out of main README.

---

## Acceptance Criteria

Running:

```
python cli/end_to_end.py +overfit_test
```

must perform a deterministic, minimal, fully reproducible run that:
- Loads exactly one block,
- Trains with minimal batch size and no regularization/conditioning/augmentation,
- Disables validation, early stopping, and AMP,
- Converges (memorizes) the tiny sample,
- Saves checkpoints without errors.
