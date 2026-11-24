"""Phase 18D (CSP-P18D-E001): Cross-Environment Knowledge Continuity — Selective Transfer,
Adaptation, and Reconstruction of Internal World Models.

Four Controlled Environmental Regimes (each isolating a single structural variable):
- Regime A (Representation-Only Shift): M3 + offset 0.5, unchanged A_drift, B_act, bound=1.8, rule_alpha.
- Regime B (Boundary Shift):            M3 + offset 0.5, unchanged A_drift, B_act, rule_alpha, bound=1.0.
- Regime C (Partial Dynamics Mutation): M3 + offset 0.5, unchanged A_drift, bound=1.8, rule_alpha, ONLY action a=2 mutated.
- Regime D (Full Structural Mutation):  M3 + offset 0.5, mutated B_act_beta, bound=1.1, rule_beta.

Seven Fair Comparison Variants (all given identical N=60 target transitions, zero privilege):
1. full_reconstruction
2. unconditional
3. consequence_only (Phase 18C baseline)
4. dynamics_and_boundary
5. selective
6. uncertainty_guided
7. fair_mlp (+ diagnostic oracle upper bound ONLY)
"""
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from experiments.phase18_experiments import (
    ACTION_COSTS_DEFAULT,
    A_DRIFT_ALPHA,
    B_ACT_ALPHA,
    B_ACT_BETA,
    IN_DIM18,
    M_GAMMA_SHIFTED,
    N_ACTIONS18,
    N_LATENT18,
    OFFSET_GAMMA,
    W18_ALPHA,
    W18_GAMMA_SHIFTED,
    MLPWorldModelBaseline,
    Phase18WorldSpec,
    _rule_alpha,
    _rule_beta,
    ast_leakage_audit_phase18,
    holm_bonferroni,
    oracle_plan_action,
    paired_stats,
    run_episode,
)
from models.planner import AutonomousPlanner
from models.world_model import CaspianWorldModel

DEV_SEEDS_18D = [9401, 9402, 9403, 9404, 9405]
CONFIRMATORY_SEEDS_18D = list(range(231, 246))  # 15 brand-new unseen seeds (231..245)

# Controlled Regime B: Boundary Shift ONLY (state_bound: 1.8 -> 1.0)
W18D_REGIME_B = Phase18WorldSpec(
    name="W18D_regime_B_boundary_shift",
    M=M_GAMMA_SHIFTED, offset=OFFSET_GAMMA,
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_ALPHA,
    rule_fn=_rule_alpha, action_costs=ACTION_COSTS_DEFAULT,
    state_bound=1.0,
)

# Controlled Regime C: Partial Dynamics Mutation ONLY (action a=2 mutated; a=0,1,3,4 unchanged)
B_ACT_PARTIAL = B_ACT_ALPHA.copy()
B_ACT_PARTIAL[2] = np.array([0.00, -0.85, 0.00, 0.60], dtype=float)
W18D_REGIME_C = Phase18WorldSpec(
    name="W18D_regime_C_partial_mutation",
    M=M_GAMMA_SHIFTED, offset=OFFSET_GAMMA,
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_PARTIAL,
    rule_fn=_rule_alpha, action_costs=ACTION_COSTS_DEFAULT,
    state_bound=1.8,
)

# Controlled Regime D: Full Structural Mutation (genuinely non-isomorphic dynamics A_D, B_D + bound=1.1 + rule_beta)
# Note: B_ACT_BETA is a coordinate permutation of B_ACT_ALPHA; Regime D must break O(4) isomorphism!
A_DRIFT_D = np.array([
    [0.55, -0.30,  0.15,  0.00],
    [0.25,  0.60,  0.00, -0.25],
    [0.00,  0.20,  0.50,  0.30],
    [-0.20, 0.00, -0.25,  0.60],
], dtype=float)

B_ACT_D = np.array([
    [ 0.00,  0.00,  0.00,  0.00],
    [-0.50,  0.55, -0.45,  0.20],
    [ 0.65, -0.35,  0.40, -0.50],
    [ 0.30,  0.60,  0.50,  0.45],
    [-0.60, -0.45,  0.30, -0.55],
], dtype=float)

W18D_REGIME_D = Phase18WorldSpec(
    name="W18D_regime_D_full_mutation",
    M=M_GAMMA_SHIFTED, offset=OFFSET_GAMMA,
    A_drift=A_DRIFT_D, B_act=B_ACT_D,
    rule_fn=_rule_beta, action_costs=ACTION_COSTS_DEFAULT,
    state_bound=1.1,
)

REGIMES_18D = {
    "A_rep_shift": W18_GAMMA_SHIFTED,
    "B_boundary_shift": W18D_REGIME_B,
    "C_partial_mut": W18D_REGIME_C,
    "D_full_mut": W18D_REGIME_D,
}

VARIANTS_18D = [
    "full_reconstruction",
    "unconditional",
    "consequence_only",
    "dynamics_and_boundary",
    "selective",
    "uncertainty_guided",
    "fair_mlp",
]

PLAN_INITS_18D = [
    np.array([-0.65, -0.65, 0.3, -0.3], dtype=float),
    np.array([-0.55, -0.60, -0.4, 0.4], dtype=float),
    np.array([-0.70, -0.55, 0.2, 0.2], dtype=float),
]


def _eval_regime_variant(
    seed: int,
    regime_name: str,
    spec: Phase18WorldSpec,
    tr_src: Dict[str, np.ndarray],
    src_bytes: bytes,
    err_src_pre: float,
    tr_tgt: Dict[str, np.ndarray],
    te_tgt: Dict[str, np.ndarray],
    variant: str,
) -> Dict[str, Any]:
    x_scale = float(np.std(te_tgt["X_t"])) + 1e-8
    S_det = np.array([spec.step_true(te_tgt["S_t"][i], int(te_tgt["A_t"][i]), rng=None)[0] for i in range(len(te_tgt["S_t"]))])
    X_det = spec.observe(S_det)
    hit_bnd = np.any(np.abs(S_det) >= spec.state_bound - 0.005, axis=1)

    # Multi-step H=3 open-loop test trajectories
    rng = np.random.default_rng(seed * 31 + hash(regime_name) % 997)
    n_roll = 20
    roll_x0, roll_acts, roll_true_x3 = [], [], []
    for _ in range(n_roll):
        s_c = rng.uniform(-0.8 * spec.state_bound, 0.8 * spec.state_bound, size=N_LATENT18)
        acts = rng.integers(0, N_ACTIONS18, size=3)
        roll_x0.append(spec.observe(s_c))
        roll_acts.append(acts)
        st = s_c.copy()
        for h in range(3):
            st, _ = spec.step_true(st, int(acts[h]), rng=None)
        roll_true_x3.append(spec.observe(st))
    roll_x0_arr = np.array(roll_x0)
    roll_acts_arr = np.array(roll_acts)
    roll_true_x3_arr = np.array(roll_true_x3)

    if variant == "fair_mlp":
        mlp = MLPWorldModelBaseline(seed=seed)
        mlp.fit(tr_tgt["X_t"], tr_tgt["A_t"], tr_tgt["X_next"], tr_tgt["dE"])
        px1, _ = mlp.predict_step(te_tgt["X_t"], te_tgt["A_t"])
        sq1 = np.mean((px1 - X_det) ** 2, axis=1)
        x_h1 = float(np.sqrt(np.mean(sq1)) / x_scale)
        int_err = float(np.sqrt(np.mean(sq1[~hit_bnd])) / x_scale) if np.any(~hit_bnd) else x_h1
        bnd_err = float(np.sqrt(np.mean(sq1[hit_bnd])) / x_scale) if np.any(hit_bnd) else x_h1
        rx, _ = mlp.rollout(roll_x0_arr, roll_acts_arr)
        x_h3 = float(np.sqrt(np.mean((rx[:, 3, :] - roll_true_x3_arr) ** 2)) / x_scale)
        ret = float(np.mean([
            run_episode(spec, lambda x, s, t: mlp.plan_action(x, horizon=3), s0, steps=20, seed=seed * 5000 + i)["total_return"]
            for i, s0 in enumerate(PLAN_INITS_18D)
        ]))
        return {
            "x_h1": x_h1, "x_h3": x_h3, "interior_nrmse": int_err, "boundary_nrmse": bnd_err,
            "plan_return": ret, "pi95_cov": 0.0, "forgetting_src": 0.0, "decisions": {},
        }

    cwm = CaspianWorldModel.from_bytes(src_bytes)
    cwm.transfer_mode = variant
    eid_src = 0
    eid_tgt = cwm.observe_transitions(tr_tgt["X_t"], tr_tgt["A_t"], tr_tgt["X_next"], tr_tgt["dE"])

    # Verify zero forgetting on source engram
    p_src_post = cwm.predict_step(tr_src["X_t"][:40], tr_src["A_t"][:40], eid=eid_src)["x_next"]
    err_src_post = float(np.sqrt(np.mean((p_src_post - tr_src["X_next"][:40]) ** 2)))
    forgetting_src = max(0.0, err_src_post - err_src_pre)

    pred1 = cwm.predict_step(te_tgt["X_t"], te_tgt["A_t"], eid=eid_tgt)
    sq1 = np.mean((pred1["x_next"] - X_det) ** 2, axis=1)
    x_h1 = float(np.sqrt(np.mean(sq1)) / x_scale)
    int_err = float(np.sqrt(np.mean(sq1[~hit_bnd])) / x_scale) if np.any(~hit_bnd) else x_h1
    bnd_err = float(np.sqrt(np.mean(sq1[hit_bnd])) / x_scale) if np.any(hit_bnd) else x_h1
    pi95_cov = float(np.mean(np.abs(pred1["de_mean"] - te_tgt["dE"]) <= 1.96 * pred1["de_std"]))

    roll3 = cwm.rollout_latent(roll_x0_arr, roll_acts_arr, eid=eid_tgt)
    x_h3 = float(np.sqrt(np.mean((roll3["x_traj"][:, 3, :] - roll_true_x3_arr) ** 2)) / x_scale)

    pl = AutonomousPlanner(cwm, horizon=3, gamma=0.95, beta_uncertainty=0.5)
    ret = float(np.mean([
        run_episode(spec, lambda x, s, t: pl.plan_action(x, eid=eid_tgt)[0], s0, steps=20, seed=seed * 5000 + i)["total_return"]
        for i, s0 in enumerate(PLAN_INITS_18D)
    ]))

    return {
        "x_h1": x_h1, "x_h3": x_h3, "interior_nrmse": int_err, "boundary_nrmse": bnd_err,
        "plan_return": ret, "pi95_cov": pi95_cov, "forgetting_src": forgetting_src,
        "decisions": cwm.last_transfer_decisions,
    }


def run_single_seed_18d(seed: int, variants: List[str] = VARIANTS_18D) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tr_src = W18_ALPHA.collect_transitions(n_steps=350, seed=seed * 100 + 1)
    # Fit source world ONCE per seed and share exact serialized source engram across all variants
    cwm_src = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True, transfer_mode="consequence_only")
    eid_src = cwm_src.observe_transitions(tr_src["X_t"], tr_src["A_t"], tr_src["X_next"], tr_src["dE"])
    p_src_pre = cwm_src.predict_step(tr_src["X_t"][:40], tr_src["A_t"][:40], eid=eid_src)["x_next"]
    err_src_pre = float(np.sqrt(np.mean((p_src_pre - tr_src["X_next"][:40]) ** 2)))
    src_bytes = cwm_src.to_bytes()

    out: Dict[str, Any] = {"seed": seed, "regimes": {}}

    for r_idx, (rname, spec) in enumerate(REGIMES_18D.items()):
        tr_tgt = spec.collect_transitions(n_steps=60, seed=seed * 100 + 20 + r_idx)
        te_tgt = spec.collect_transitions(n_steps=120, seed=seed * 100 + 60 + r_idx)
        oracle_ret = float(np.mean([
            run_episode(spec, lambda x, s, t: oracle_plan_action(spec, s, horizon=3), s0, steps=20, seed=seed * 5000 + i)["total_return"]
            for i, s0 in enumerate(PLAN_INITS_18D)
        ]))
        r_res: Dict[str, Any] = {"oracle_return": oracle_ret, "variants": {}}
        for v in variants:
            r_res["variants"][v] = _eval_regime_variant(seed, rname, spec, tr_src, src_bytes, err_src_pre, tr_tgt, te_tgt, v)
        out["regimes"][rname] = r_res

    out["wall_time_s"] = round(time.perf_counter() - t0, 3)
    return out


def aggregate_18d(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    seeds = [r["seed"] for r in rows]

    def vec(regime: str, variant: str, key: str) -> np.ndarray:
        return np.array([r["regimes"][regime]["variants"][variant][key] for r in rows], dtype=float)

    # Means table per regime x variant
    table_means: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rname in REGIMES_18D:
        table_means[rname] = {}
        for v in VARIANTS_18D:
            table_means[rname][v] = {
                k: round(float(np.mean(vec(rname, v, k))), 5)
                for k in ("x_h1", "x_h3", "interior_nrmse", "boundary_nrmse", "plan_return", "pi95_cov", "forgetting_src")
            }

    # Component-level decision accuracy for uncertainty_guided across all 4 regimes
    # Expected ground-truth component compatibility:
    # - Regime A: all 5 actions TRANSFER/ADAPT, boundary TRANSFER
    # - Regime B: all 5 actions TRANSFER/ADAPT, boundary ADAPT
    # - Regime C: actions [0,1,3,4] TRANSFER/ADAPT, action 2 RECONSTRUCT, boundary TRANSFER/ADAPT
    # - Regime D: all 5 actions RECONSTRUCT, boundary RECONSTRUCT
    act_correct, act_total = 0, 0
    bnd_correct, bnd_total = 0, 0
    harmful_xfer_d = 0
    unnecessary_rec_a = 0
    c_mut_detected = 0
    c_unchanged_preserved = 0
    c_unchanged_total = 0

    for r in rows:
        dA = r["regimes"]["A_rep_shift"]["variants"]["uncertainty_guided"]["decisions"]
        for a in range(N_ACTIONS18):
            act_total += 1
            if dA["action_decisions"][a] in ("TRANSFER", "ADAPT"):
                act_correct += 1
            else:
                unnecessary_rec_a += 1
        bnd_total += 1
        if dA["boundary_decision"] == "TRANSFER":
            bnd_correct += 1

        dB = r["regimes"]["B_boundary_shift"]["variants"]["uncertainty_guided"]["decisions"]
        for a in range(N_ACTIONS18):
            act_total += 1
            if dB["action_decisions"][a] in ("TRANSFER", "ADAPT"):
                act_correct += 1
        bnd_total += 1
        if dB["boundary_decision"] in ("ADAPT", "RECONSTRUCT"):
            bnd_correct += 1

        dC = r["regimes"]["C_partial_mut"]["variants"]["uncertainty_guided"]["decisions"]
        for a in range(N_ACTIONS18):
            act_total += 1
            if a == 2:
                if dC["action_decisions"][a] == "RECONSTRUCT":
                    act_correct += 1
                    c_mut_detected += 1
            else:
                c_unchanged_total += 1
                if dC["action_decisions"][a] in ("TRANSFER", "ADAPT"):
                    act_correct += 1
                    c_unchanged_preserved += 1
        bnd_total += 1
        if dC["boundary_decision"] in ("TRANSFER", "ADAPT"):
            bnd_correct += 1

        dD = r["regimes"]["D_full_mut"]["variants"]["uncertainty_guided"]["decisions"]
        for a in range(N_ACTIONS18):
            act_total += 1
            if dD["action_decisions"][a] == "RECONSTRUCT":
                act_correct += 1
            else:
                harmful_xfer_d += 1
        bnd_total += 1
        if dD["boundary_decision"] == "RECONSTRUCT":
            bnd_correct += 1

    # Pre-registered Hypotheses P18D-H1 through P18D-H7
    # H1: Selective dynamics transfer in Regime A beats consequence_only (P18C) & full_reconstruction on x_h1 (Holm p < 0.05)
    h1_vs_p18c = paired_stats(vec("A_rep_shift", "uncertainty_guided", "x_h1"), vec("A_rep_shift", "consequence_only", "x_h1"), higher_is_better=False, seed=1)
    h1_vs_rec = paired_stats(vec("A_rep_shift", "uncertainty_guided", "x_h1"), vec("A_rep_shift", "full_reconstruction", "x_h1"), higher_is_better=False, seed=2)
    h1_holm = holm_bonferroni({"consequence_only": h1_vs_p18c["perm_p"], "full_reconstruction": h1_vs_rec["perm_p"]})
    h1_met = bool(h1_vs_p18c["ci_strictly_positive"] and h1_vs_rec["ci_strictly_positive"] and all(p < 0.05 for p in h1_holm.values()))

    # H2: Boundary continuity in Regime A beats consequence_only on boundary_nrmse AND adapts in Regime B beating dynamics_and_boundary (unconditional bounds)
    h2_a_bnd = paired_stats(vec("A_rep_shift", "uncertainty_guided", "boundary_nrmse"), vec("A_rep_shift", "consequence_only", "boundary_nrmse"), higher_is_better=False, seed=3)
    h2_b_bnd = paired_stats(vec("B_boundary_shift", "uncertainty_guided", "boundary_nrmse"), vec("B_boundary_shift", "dynamics_and_boundary", "boundary_nrmse"), higher_is_better=False, seed=4)
    h2_holm = holm_bonferroni({"regimeA_vs_p18c": h2_a_bnd["perm_p"], "regimeB_vs_dyn_bnd": h2_b_bnd["perm_p"]})
    h2_met = bool(h2_a_bnd["ci_strictly_positive"] and h2_b_bnd["ci_strictly_positive"] and all(p < 0.05 for p in h2_holm.values()))

    # H3: Selective adaptation in Regime C (partial mutation) beats BOTH unconditional/dynamics_and_boundary AND full_reconstruction on x_h1
    h3_vs_uncond = paired_stats(vec("C_partial_mut", "uncertainty_guided", "x_h1"), vec("C_partial_mut", "dynamics_and_boundary", "x_h1"), higher_is_better=False, seed=5)
    h3_vs_rec = paired_stats(vec("C_partial_mut", "uncertainty_guided", "x_h1"), vec("C_partial_mut", "full_reconstruction", "x_h1"), higher_is_better=False, seed=6)
    h3_holm = holm_bonferroni({"dynamics_and_boundary": h3_vs_uncond["perm_p"], "full_reconstruction": h3_vs_rec["perm_p"]})
    h3_met = bool(h3_vs_uncond["ci_strictly_positive"] and h3_vs_rec["ci_strictly_positive"] and all(p < 0.05 for p in h3_holm.values()))

    # H4: Structural change detection in Regime D rejects harmful transfer (beats unconditional & dynamics_and_boundary, harmful_xfer_rate < 0.15)
    h4_vs_uncond = paired_stats(vec("D_full_mut", "uncertainty_guided", "x_h1"), vec("D_full_mut", "unconditional", "x_h1"), higher_is_better=False, seed=7)
    h4_vs_db = paired_stats(vec("D_full_mut", "uncertainty_guided", "x_h1"), vec("D_full_mut", "dynamics_and_boundary", "x_h1"), higher_is_better=False, seed=8)
    harmful_rate_d = harmful_xfer_d / max(len(rows) * N_ACTIONS18, 1)
    h4_met = bool(h4_vs_uncond["ci_strictly_positive"] and h4_vs_db["ci_strictly_positive"] and h4_vs_db["perm_p"] < 0.05 and harmful_rate_d < 0.15)

    # H5: Knowledge-efficient reconstruction: action decision accuracy >= 85% & boundary decision accuracy >= 75% across all 4 regimes
    act_acc = act_correct / max(act_total, 1)
    bnd_acc = bnd_correct / max(bnd_total, 1)
    h5_met = bool(act_acc >= 0.85 and bnd_acc >= 0.75)

    # H6: Cross-environment predictive & planning generalization in Regime A: beats fair_mlp & full_reconstruction on planning return (Holm p < 0.05)
    h6_vs_mlp = paired_stats(vec("A_rep_shift", "uncertainty_guided", "plan_return"), vec("A_rep_shift", "fair_mlp", "plan_return"), higher_is_better=True, seed=9)
    h6_vs_rec = paired_stats(vec("A_rep_shift", "uncertainty_guided", "plan_return"), vec("A_rep_shift", "full_reconstruction", "plan_return"), higher_is_better=True, seed=10)
    h6_vs_p18c = paired_stats(vec("A_rep_shift", "uncertainty_guided", "plan_return"), vec("A_rep_shift", "consequence_only", "plan_return"), higher_is_better=True, seed=11)
    h6_holm = holm_bonferroni({"fair_mlp": h6_vs_mlp["perm_p"], "full_reconstruction": h6_vs_rec["perm_p"]})
    h6_met = bool(h6_vs_mlp["ci_strictly_positive"] and h6_vs_rec["ci_strictly_positive"] and all(p < 0.05 for p in h6_holm.values()))

    # H7: Uncertainty-guided selective beats or matches fixed-threshold selective across all 4 regimes (mean x_h1 lower or equal) & preserves 0.0 forgetting & 0 AST leaks
    mean_ug_all = float(np.mean([table_means[r]["uncertainty_guided"]["x_h1"] for r in REGIMES_18D]))
    mean_sel_all = float(np.mean([table_means[r]["selective"]["x_h1"] for r in REGIMES_18D]))
    max_forg = float(np.max([np.max(vec(r, "uncertainty_guided", "forgetting_src")) for r in REGIMES_18D]))
    ast_aud = ast_leakage_audit_phase18()
    h7_met = bool(mean_ug_all <= mean_sel_all + 0.005 and max_forg < 1e-6 and ast_aud["passed"] and act_acc >= 0.85)

    verdicts = {
        "P18D-H1": h1_met, "P18D-H2": h2_met, "P18D-H3": h3_met,
        "P18D-H4": h4_met, "P18D-H5": h5_met, "P18D-H6": h6_met, "P18D-H7": h7_met,
    }
    return {
        "experiment_id": "CSP-P18D-E001",
        "split": label,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "hypotheses_met_count": f"{sum(verdicts.values())}/7",
        "verdicts": verdicts,
        "regime_variant_means": table_means,
        "decision_audit": {
            "action_decision_accuracy": round(act_acc, 4),
            "boundary_decision_accuracy": round(bnd_acc, 4),
            "regimeA_unnecessary_reconstruction_rate": round(unnecessary_rec_a / max(len(rows) * N_ACTIONS18, 1), 4),
            "regimeC_mutated_action_detected_rate": round(c_mut_detected / max(len(rows), 1), 4),
            "regimeC_unchanged_actions_preserved_rate": round(c_unchanged_preserved / max(c_unchanged_total, 1), 4),
            "regimeD_harmful_transfer_rate": round(harmful_rate_d, 4),
        },
        "H1": {"met": h1_met, "vs_p18c": h1_vs_p18c, "vs_rec": h1_vs_rec, "holm_p": h1_holm},
        "H2": {"met": h2_met, "regimeA_bnd_vs_p18c": h2_a_bnd, "regimeB_bnd_vs_db": h2_b_bnd, "holm_p": h2_holm},
        "H3": {"met": h3_met, "vs_dyn_bnd": h3_vs_uncond, "vs_rec": h3_vs_rec, "holm_p": h3_holm},
        "H4": {"met": h4_met, "vs_uncond": h4_vs_uncond, "vs_dyn_bnd": h4_vs_db, "harmful_transfer_rate": round(harmful_rate_d, 4)},
        "H5": {"met": h5_met, "action_acc": round(act_acc, 4), "boundary_acc": round(bnd_acc, 4)},
        "H6": {"met": h6_met, "vs_mlp": h6_vs_mlp, "vs_rec": h6_vs_rec, "vs_p18c": h6_vs_p18c, "holm_p": h6_holm},
        "H7": {"met": h7_met, "mean_ug_x_h1": round(mean_ug_all, 5), "mean_sel_x_h1": round(mean_sel_all, 5), "max_forgetting": max_forg, "ast_passed": ast_aud["passed"]},
        "per_seed": rows,
    }


def run_phase18d(seeds: List[int], label: str, out_path: str) -> Dict[str, Any]:
    rows = [run_single_seed_18d(s) for s in seeds]
    summary = aggregate_18d(rows, label=label)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    if mode == "ablation":
        res = run_phase18d(DEV_SEEDS_18D, "dev_ablation", "logs/phase18d/p18d_ablation_results.json")
    elif mode == "confirmatory":
        res = run_phase18d(CONFIRMATORY_SEEDS_18D, "confirmatory", "logs/phase18d/p18d_confirmatory_results.json")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(f"=== Phase 18D ({res['split']}, n={res['n_seeds']}) ===")
    print("Hypotheses met:", res["hypotheses_met_count"], res["verdicts"])
    print("Decisions:", res["decision_audit"])
    for rname in REGIMES_18D:
        print(f"  [{rname}]")
        for v in VARIANTS_18D:
            m = res["regime_variant_means"][rname][v]
            print(f"    {v:22s}: x_h1={m['x_h1']:.5f} x_h3={m['x_h3']:.5f} bnd={m['boundary_nrmse']:.5f} ret={m['plan_return']:6.1f}")
