# Hierarchical Coarse → Fine E2E Roadmap
# (Backbone-Decoupled, Multihead-Compatible)

──────────────────────────────────────────────
Milestone 0 — Establish a Stable Coarse Backbone
──────────────────────────────────────────────
• Train coarse heads only (functional groups).
• Use any backbone implementing `Backbone` interface
  (UNet, UNet++, or your own).                          ← modular/swappable backbone
• Ensure coarse semantics are stable before adding fine tasks.

Outcome:
A reliable shared representation for global semantic grouping.

──────────────────────────────────────────────
Milestone 1 — Freeze Coarse Heads + Protect Structure
──────────────────────────────────────────────
• Freeze coarse heads permanently via `set_frozen_heads(...)`.
• Freeze early backbone layers (global semantics).
• Leave deeper layers trainable (local details).

Outcome:
Coarse routing becomes an anchor; future fine heads cannot corrupt it.

──────────────────────────────────────────────
Milestone 2 — Add Per‑Group Refinement Adapters
──────────────────────────────────────────────
• Insert lightweight group-specific adapter blocks before fine heads:
  - FiLM modulation (supported in your architecture)
  - 1–2 conv bottlenecks per group
  - low‑rank linear projections
• Adapters isolate fine-specific gradients from the backbone.

Outcome:
Fine heads can specialize without destabilizing shared features.

──────────────────────────────────────────────
Milestone 3 — Introduce Fine Heads Progressively
──────────────────────────────────────────────
• Activate ONE fine head at a time using the multihead manager.
• Keep coarse heads inactive during fine-head optimization.
• Use LR warmup for new fine heads.
• Rebuild optimizer each time you add a head (avoid stale states).

Outcome:
Controlled specialization of fine heads without interference.

──────────────────────────────────────────────
Milestone 4 — Hierarchical Gating in Training
──────────────────────────────────────────────
• Fine heads only receive gradients for samples belonging to their
  parent coarse group (mask or data routing).
• Loss is hierarchical:
  coarse CE + conditioned fine CE.
• Backprop flows only within valid coarse regions.

Outcome:
Fine heads learn only relevant distinctions, reducing competition.

──────────────────────────────────────────────
Milestone 5 — Add Feature/Predictive Distillation
──────────────────────────────────────────────
• Distill backbone to preserve coarse‑level feature structure:
  - feature‑matching loss between frozen and current features
  - KL distillation from frozen coarse logits
• Ensures semantic stability across phase shifts.

Outcome:
Prevents catastrophic forgetting as new fine heads are introduced.

──────────────────────────────────────────────
Milestone 6 — Hierarchical Gating at Inference (E2E Integration)
──────────────────────────────────────────────
Pipeline inside a single model:
1. Backbone produces shared features.
2. Coarse head produces group logits → produces routing mask.
3. Fine head(s) refine predictions within masked regions.
4. Combine results into final hierarchical prediction.

Outcome:
True single‑model coarse→fine inference without external routing.

──────────────────────────────────────────────
Milestone 7 — Optional Joint Fine-Tuning (Final Polish)
──────────────────────────────────────────────
• Freeze coarse head(s).
• Activate all fine heads simultaneously.
• Train with small LR to harmonize fine heads and adapters.
• Partial backbone freeze recommended.

Outcome:
Fully integrated, consistent coarse→fine system.

──────────────────────────────────────────────
Milestone 8 — Fully Modular & Swappable Backbone
──────────────────────────────────────────────
• Because the model uses `Backbone` abstract class:
  - You can plug in UNet++, UNet, or any custom architecture.
  - The multihead system remains unchanged.
• The hierarchical logic lives at the head/adapter level,
  not inside the backbone.

Outcome:
Sustainable architecture: backbone-independent, stable hierarchy.