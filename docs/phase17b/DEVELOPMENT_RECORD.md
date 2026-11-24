# CSP-P17B-E001 — Development & Execution Record

1. **Section 3 Pre-Implementation Forensic Audit (`docs/phase17b/P17B_FORENSIC_AUDIT.md`):**
   - Independently recalculated all archived Phase 17 (`seeds 71–80`) metrics (`<= 5e-6` rounding diff).
   - Identified exact cause of `ret_gamma_post4 (0.20292) != forward_transfer_gamma (0.17640)`: probe misretrieval on seeds 74 and 75 (`retrieval = 3/4`).
   - Found `G = 5..8` (mean `7.3`), `rank(Xc) = 3` singular collapse on seed 77 (`cond = 2.83e12`), and `32x` replicate noise variance (`0.0049–0.1584`).
2. **Stage 1 Development (`DEV_SEEDS = [9101..9105]`):**
   - Separated `alignment_mode="original_v3"` (exact Stage B `115d579` path) from `"occam_hierarchy"` and `"six_dof_only"` without overwriting historical logic.
   - Dev seed `9105` revealed that allowing 10-DOF expansion when `G <= 10` caused interpolation overfitting (`0.0401 -> 0.1427`). Added explicit identifiability guard (`mean_offset_norm <= 0.35` stays on 6-DOF when `G <= 10`).
   - Froze manifest (`P17B_MANIFEST.json`), preregistration (`EXPERIMENT_REGISTRY.md`), and code at commit `316e7d0a0c9f7b1895df9c4982647d8bbd2bfcb3`.
3. **Stage 2 Independent Confirmation (`SEEDS17B = [131..145]`, `n = 15` fresh disjoint seeds):**
   - Primary 10-shot outcome: **`Outcome C — No Reliable Improvement`** (`P17B-H1` NOT MET: `delta=+0.0305, CI=[-0.0117, +0.0716], Holm p=0.3354`; `P17B-H2` NOT MET vs `memoryless_structural`: `delta=+0.0405, CI=[-0.0124, +0.0927]`; `P17B-H4` NOT MET due to `4/10` in-subspace rotation detection; `P17B-H3` Retention `MET` & `P17B-H5` Determinism/Leakage `MET`).
   - Secondary exploratory 12-shot diagnostic (strictly separate): `occam_hierarchy_memory` achieved `0.03853` vs `original_v3` `0.08009` (`p = 0.0243`) and vs `memoryless_structural` `0.12412` (`14/15` wins, `p = 0.00195`, $d_z = 1.05$).
   - Preserved all results honestly without tuning on confirmatory seeds or redefining success.
