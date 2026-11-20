# Phase 16 Audit & Independent Confirmatory Pre-Registration (`CSP-P16-E002`)

**Status:** Pre-registered *before* executing any seed in `SEEDS16_INDEPENDENT = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]`.
**Branch:** `feature/multiple-worlds`
**Locked Historical Record (Untouched):** `CSP-P16-E001` (`logs/phase16/confirmatory_matrix.json`, seeds `51–60`).

---

## 1. Critical Audit of `CSP-P16-E001` (Why Independent Confirmation Is Required)

Before building Phase 17 (Continual Learning) on top of Phase 16's `v2 ModularTransferModel`, a strict scientific audit of `CSP-P16-E001` surfaced **four issues** that must be closed before claiming `v2` is confirmed:

1. **Post-Hoc Seed Reuse in Stage C:** `v2` fixes were evaluated in Stage C on the *same* seeds (`51–60`) used to diagnose `v1`. Even though the fixes were derived from first principles, evaluation on previously inspected seeds is diagnostic—not independent confirmation.
2. **Protocol Privilege (`align_sensors=spec.obs_shifted`):** In `CSP-P16-E001`, the experiment harness passed the ground-truth boolean `spec.obs_shifted` into `ModularTransferModel.adapt_few_shot(...)`. An autonomous agent entering a new world does not receive a ground-truth sensor-shift flag.
   - **Audit Fix:** Added unsupervised `ModularTransferModel.detect_sensor_shift(X_tgt)` computing the normalized Frobenius subspace projection residual $r_{\text{subspace}} = \|(X_{\text{tgt}} - \bar{x}_{\text{src}})(I - W_{\text{src}} W_{\text{src}}^\top)\|_F / \|X_{\text{tgt}} - \bar{x}_{\text{src}}\|_F$. `CSP-P16-E002` runs with `autonomous_shift_detection=True` (`align_sensors="auto"`), giving `modular_transfer` **zero** access to `spec.obs_shifted`.
3. **Fair Head-to-Head `v2 scratch_structural` Baseline:** In Stage C of `CSP-P16-E001`, `v2 modular_transfer` was compared against `v1` baseline logs (where `scratch_structural` still lacked $z_k^2$ and LOOCV). In `CSP-P16-E002`, **all 6 architectures**—including `v2 scratch_structural` with the exact same 14-feature $O(4)$-invariant basis and unique-state LOOCV—run fresh head-to-head on seeds `61–70`. Any remaining advantage of `modular_transfer` over `scratch_structural` is therefore **100% pure cross-world structural transfer**, not a basis or regularization discrepancy.
4. **Explicit Scope Boundary:** `W16_obs_shift` applies an orthogonal sensor rotation/projection ($X_{\text{tgt}} = P M_{\text{shift}}$) with zero additive offset ($c = 0$). Generalization to nonzero additive sensor offsets ($c \neq 0$) is explicitly outside `CSP-P16-E002` scope and not claimed.

---

## 2. Independent Confirmatory Design (`CSP-P16-E002`)

- **Disjoint Seeds:** `SEEDS16_INDEPENDENT = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]` (never inspected during `v1` or `v2` development).
- **Target Worlds (4):** `W16_obs_shift`, `W16_shared_plus_new`, `W16_rule_shift`, `W16_negative_control`.
- **Architectures (6):** `modular_transfer` (`v2`, `align_sensors="auto"`), `direct_transfer`, `finetune_mlp`, `scratch_mlp`, `scratch_structural` (`v2`), `oracle_target`.
- **Shot Levels (2):** `20` (`lo_shot`) and `48` (`hi_shot`). Total evaluations: `4 x 6 x 2 x 10 = 480`.

---

## 3. Pre-Registered Hypotheses & Pass Criteria

We report **both** the exact original `CSP-P16-E001` tests (`P16-H1`–`P16-H5`) unchanged for full continuity, and the independent confirmation suite (`P16-IND-1`–`P16-IND-5`) designed to test genuine autonomous transfer without artificial per-seed floor artifacts:

| Hypothesis | World / Regime (20-Shot, Seeds `61–70`) | Pre-Registered Pass Criterion |
| :--- | :--- | :--- |
| **`P16-IND-1`** | `W16_obs_shift` (Autonomous Sensor Alignment) | Unsupervised shift detector triggers on `10/10` seeds; `core_gate = 1.0` on `>= 9/10` seeds; mean `target_nrmse < 0.025`; paired mean improvement with bootstrap 95% CI strictly `> 0` against `direct_transfer`, `scratch_mlp`, `finetune_mlp`, AND `v2 scratch_structural`. |
| **`P16-IND-2`** | `W16_shared_plus_new` (Shared Core + New Rule) | `0/10` false sensor-shift triggers; `core_gate = 1.0` on `>= 8/10` seeds; paired mean improvement with bootstrap 95% CI strictly `> 0` against `scratch_mlp` and `v2 scratch_structural`, and mean `target_nrmse < finetune_mlp`. |
| **`P16-IND-3`** | `W16_rule_shift` (Rule Shift on Shared Latents) | `0/10` false sensor-shift triggers; paired mean improvement with bootstrap 95% CI strictly `> 0` against `direct_transfer`, `scratch_mlp`, and `finetune_mlp`. |
| **`P16-IND-4`** | `W16_negative_control` (Negative-Transfer Guard) | `0/10` false sensor-shift triggers; `direct_transfer >= 0.40` on `>= 9/10` seeds; `modular_transfer` sets `core_gate = 0.0` on `>= 9/10` seeds; paired mean improvement with bootstrap 95% CI strictly `> 0` against `finetune_mlp`. |
| **`P16-IND-5`** | Source Knowledge Retention (All Worlds) | `source_retention_nrmse < 0.02` on `>= 9/10` seeds across both `W16_shared_plus_new` and `W16_rule_shift`, AND beats `finetune_mlp` by `>= 0.05` on `>= 8/10` seeds (identical to original `P16-H5`). |
