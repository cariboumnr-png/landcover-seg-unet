# Canonical Logit Adjustment (LA) Combinations

## 1. No LA (Neutral Model)
- **Train:** LA = OFF
- **Curriculum Stages:** OFF
- **Validation:** OFF
- **Test:** OFF
Use when you want a fully neutral baseline with no prior-driven bias.

---

## 2. LA as Training-Time Imbalance Correction
### (LA is part of the loss/objective)
- **Train:** LA = ON (alpha ≈ 1.0)
- **Curriculum Stages:** Typically ON for all stages
  (unless scheduling alpha downwards late)
- **Validation:** OFF
- **Test:** OFF
Prevents double-application of priors at eval; training sees the biased logits.

---

## 3. LA as Deployment-Time Calibration
### (Train neutral; calibrate only at inference)
- **Train:** LA = OFF
- **Curriculum Stages:** OFF
- **Validation:** ON *and* OFF (compare both)
- **Test:** ON (alpha tuned by val)
Most common when deployment priors are known and should influence predictions.

---

## 4. Curriculum-Driven LA Schedule (Scaffolding → Neutralization)
### (Use priors early to stabilize, then remove)
- **Stage 1:** LA = ON (alpha = 1.0)
- **Stage 2:** LA = ON (alpha = 0.5)
- **Stage 3+**: LA = OFF
- **Validation:** OFF
- **Test:** OFF or ON depending on deployment intent
Useful when early curriculum stages are highly imbalanced or noisy.

---

## 5. Late-Stage Calibration (Neutral Early → LA Final Stage)
- **Stages 1..K-1:** LA = OFF
- **Stage K:** LA = ON (alpha tuned)
- **Validation:** ON (to choose alpha)
- **Test:** ON
Good when representation learning should be unbiased, but final decisions
should match real-world prevalence.

---

## 6. Ablation Across Curriculum Stages
### (Measure LA's influence under different difficulty levels)
- **Odd Stages:** LA = OFF
- **Even Stages:** LA = ON (alpha fixed)
- **Validation/Test:** match the stage setting or run both
Helps isolate LA’s contribution vs. curriculum difficulty progression.

---

## 7. Alpha Sweep for Operating-Point Selection
- **Train:** LA = OFF
- **Curriculum:** OFF
- **Validation:** Evaluate with alpha ∈ {0.0, 0.25, 0.5, 1.0}
- **Test:** Use best alpha from validation
Provides threshold-free tuning of deployment behavior.

---

## 8. Head-Specific LA (If some heads have priors)
- **Train:** LA = ON/OFF (choice depends on objective)
- **Curriculum:** You may enable LA only for selected heads
- **Validation/Test:** ON only for heads needing calibration
Only heads with `la_{head}` buffers are affected; others remain unchanged.