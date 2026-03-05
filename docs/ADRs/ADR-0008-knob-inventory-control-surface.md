# ADR‑0008: Knob Inventory & Control Surface
- **Status:** Accepted (pre‑work for [ADR‑0009](./ADR-0008-knob-inventory-control-surface.md))
- **Date:** 2026‑03-02
- **Updated:** 2026‑03-05

## Context

Before introducing a standardized `overfit_test` Hydra profile, we need a clear view of all *implemented* knobs (what exists today), what’s *missing*, and which knobs should be **public** (user‑facing) vs. **internal** (config‑only for CI/profiles).
This ADR inventories existing controls across **data**, **model**, **training/controller**, and **runtime** surfaces, calling out gaps that prevent determinism (e.g., lack of explicit “off” switches or seeds).

## Decision

1. **Adopt a “Control Surface” taxonomy** for configuration keys:
   - **Public**: Stable, documented, safe for users to tweak.
   - **Internal**: Supported in configs (so Hydra profiles/CI can set them) but not documented for general users.
   - **Experimental**: Present but default‑off and not documented; subject to change.

2. **Introduce deterministic “off” paths** for any behavior that is *on by default* and affects reproducibility—even if those switches remain internal only.

3. **Document a canonical checklist** (below) to validate the control surface before adopting ADR‑0009.

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

**Gaps closed**
- No standardized single‑block or micro‑split generator for overfit testing.
> added a pipeline to generate a single data block
- No explicit “disable scoring/valid‑block filtering” switch aside from setting thresholds manually.
> not applicable since the single block generator is now in place

---

### B) Dataset / Dataloaders

**Implemented knobs**
- Batch size (loader meta).
- Patch sampling (`patch_per_blk`), test grid layout for inference.
- A minimal augmentation flag (e.g., flips) toggled by train/val/test mode.

**Gaps to closed**
- Missing deterministic loader knobs:
  `shuffle=false`, `num_workers=0`, `seed=<int>`, `deterministic=true`.
> not applicable since a single block is inherently deterministic and block
searching is determined by a fixed seed (42).
- Missing global augmentation disable switch (`augment.enable=false`).
> now added and exposed

---

### C) Model (Backbones, Heads, Conditioning)

**Implemented knobs**
- Backbone selection (UNet/UNet++) and hyperparameters (`in_ch`, `base_ch`).
- Block‑level knobs: `p_drop`, normalization mode (`bn|gn|ln|none`), group‑norm groups.
- Multi‑head model: per‑head class counts, logit adjustment buffers, head activation/freeze.
- Domain conditioning modes: none / concat / FiLM / hybrid, with flags for ids/vec/MLP usage.
- Output clamping range.

**Gaps to closed**
- Confirm `conditioning.mode=none` is fully neutral (no latent conditioning ops).
- Ensure `p_drop=0.0` becomes globally effective.
> both conditioning and p_drop knobs added and exposed

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
> `grad_clip_norm` knob exposed; rest not applicable as the test is done by directly
calling trainer methods, no controller is involved
- Ensure optimizer exposes `weight_decay` and can be set to `0.0`.
> done
- Confirm that AMP can be fully disabled (no autocast/scaler).
> done

---

### E) Runtime Determinism

**Implemented**
- Only preview‑palette seeding.

**Missing**
- A global seeding system covering Python, NumPy, Torch, and CUDA.
- Deterministic backend toggles.
- Internal config keys such as:
  `runtime.seed`, `runtime.deterministic=true`.
> not applicable/not implemented

---

## Outcome

Now overfit can be run by:

```
python cli/main.py profile=overfit_test
```

which perform a deterministic, minimal, fully reproducible run that:
- Loads exactly one block / one batch
- Trains with no regularization/conditioning/augmentation
- Validation (IoU) on the same block
- No early stopping till max epoch reached or 99% IoU reached