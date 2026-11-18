---
trigger: always_on
description: Enforces mandatory post-phase forensic legitimacy auditing ("Fake vs. Legit") and first-principles mathematical root-cause investigation for any unmet criterion before advancing to the next phase.
---

# Project Caspian — Mandatory Investigation Rule (`investigation`)

After every phase's confirmatory execution and **before** moving forward to the next phase, the agent MUST execute a formal **Investigation** (`experiments/investigation.py` and `docs/phase<N>/INVESTIGATION.md`) covering two mandatory pillars:

## 1. Pillar A — Forensic Legitimacy Audit ("Fake vs. Legit")
Before trusting any positive or negative result, verify and prove in `docs/phase<N>/INVESTIGATION.md` and via `experiments/investigation.py` that the results are **100% legitimate (`LEGIT`) and not manufactured (`FAKE`)**:
1. **Zero Data / Target Leakage Audit:**
   - Verify `leak_rows == 0` across all seeds and splits.
   - Verify mathematically and by AST/source inspection that no model class in `models/` accesses ground-truth simulator internals (`decode`, `truth`, `unmix_ground_truth`, `hidden_state_delta`).
2. **Real Gradient & Weight Movement Audit:**
   - Verify that trainable parameters actually move from initialization (`||theta_final - theta_init|| > 0`), training loss decreases from epoch 0 to final epoch, and predictions are computed by live forward passes rather than cached/hardcoded constants.
3. **Cross-Seed Independence & Non-Duplication Audit:**
   - Verify that all confirmatory seeds produce distinct trajectories, distinct weight matrices, and non-identical metric vectors.
4. **Negative-Control & Shuffled-Label Sanity Check:**
   - Verify that shuffled-label probes fail ($R^2 \le 0.05$) and restricted architectures fail on negative-control worlds where their inductive bias cannot represent the ground-truth function.

## 2. Pillar B — First-Principles Root-Cause Analysis & Solution for Unmet Criteria
Whenever **any** pre-registered hypothesis or sub-criterion is `NOT MET` (even by a single seed):
1. **Isolate the Exact Failing Seeds / Cells:** Identify which seeds, worlds, or architectures missed the threshold.
2. **First-Principles Mathematical Diagnosis:** Derive from first principles (linear algebra, optimization landscape, gradient dynamics, conditioning, or function-class geometry) *why* those specific seeds or architectures failed. Never guess or hand-wave.
3. **Derive & Verify a First-Principles Solution:** Formulate a principled mathematical or architectural fix, verify it experimentally in a transparent post-confirmatory diagnostic study (without overwriting the original locked confirmatory logs), and carry the validated fix forward into the next phase.
