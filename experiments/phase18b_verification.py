"""Phase 18B (CSP-P18B-E001): Controlled Ablations & Independent Confirmatory Verification.

Evaluates:
- Stage 3-4 Controlled Ablations on dev seeds 9301-9305 (n=5):
    1. p18_baseline   (boundary_aware=False, basis_sync=False)
    2. boundary_only  (boundary_aware=True,  basis_sync=False)
    3. basis_sync_only(boundary_aware=False, basis_sync=True)
    4. combined_p18b  (boundary_aware=True,  basis_sync=True)
- Stage 6 Locked Confirmatory Evaluation on fresh unseen seeds 191-205 (n=15):
    P18B-H1 through P18B-H7 with paired bootstrap CI, permutation tests, Holm-Bonferroni,
    multi-regime data-budget scaling (N=80, 160, 350), basis-rotation equivalence proof,
    serialization round-trip, determinism, and AST leakage audit.
"""
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from experiments.phase18_experiments import (
    IN_DIM18,
    N_ACTIONS18,
    N_LATENT18,
    W18_ALPHA,
    W18_ALPHA_MUTATED,
    W18_BETA,
    W18_GAMMA_SHIFTED,
    W18_TRAP,
    MLPWorldModelBaseline,
    ast_leakage_audit_phase18,
    compute_auroc,
    holm_bonferroni,
    oracle_plan_action,
    paired_stats,
    run_episode,
    spearman_corr,
)
from models.planner import AutonomousPlanner
from models.world_model import CaspianWorldModel

DEV_SEEDS_18B = [9301, 9302, 9303, 9304, 9305]
CONFIRMATORY_SEEDS_18B = list(range(191, 206))  # 15 brand-new unseen seeds (191..205)

SADDLE_INITS = [
    np.array([-0.60, -0.60, 0.0, 0.0], dtype=float),
    np.array([-0.65, -0.55, -0.2, 0.1], dtype=float),
    np.array([-0.55, -0.65, 0.1, -0.1], dtype=float),
    np.array([-0.70, -0.60, 0.0, 0.0], dtype=float),
    np.array([-0.60, -0.70, -0.1, 0.0], dtype=float),
]

UNSEEN_SHIFT_INITS = [
    np.array([-0.65, -0.65, 0.3, -0.3], dtype=float),
    np.array([-0.55, -0.60, -0.4, 0.4], dtype=float),
    np.array([-0.70, -0.55, 0.2, 0.2], dtype=float),
]


def _eval_ablation_variant(seed: int, boundary_aware: bool, basis_sync: bool) -> Dict[str, float]:
    """Evaluate one (boundary_aware, basis_sync) variant on prediction (interior/boundary) + mutation recovery."""
    rng = np.random.default_rng(seed)
    tr_a = W18_ALPHA.collect_transitions(n_steps=350, seed=seed * 100 + 1)
    tr_b = W18_BETA.collect_transitions(n_steps=250, seed=seed * 100 + 9)
    x_scale = float(np.std(tr_a["X_t"])) + 1e-8

    cwm = CaspianWorldModel(
        in_dim=IN_DIM18, n_latent=N_LATENT18, n_actions=N_ACTIONS18,
        seed=seed, boundary_aware=boundary_aware, basis_sync=basis_sync,
    )
    eid_a = cwm.observe_transitions(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])

    # Open-loop H=1, 3 prediction + interior vs boundary breakdown
    err_h1, err_h3 = [], []
    for _ in range(30):
        s_cur = rng.uniform(-1.1, 1.1, size=N_LATENT18)
        x0 = W18_ALPHA.observe(s_cur)
        acts = rng.integers(0, N_ACTIONS18, size=3)
        true_x = [x0]
        s_tmp = s_cur.copy()
        for h in range(3):
            s_tmp, _ = W18_ALPHA.step_true(s_tmp, int(acts[h]), rng=None)
            true_x.append(W18_ALPHA.observe(s_tmp))
        roll = cwm.rollout_latent(x0, acts.reshape(1, -1), eid=eid_a)
        err_h1.append(float(np.mean((roll["x_traj"][0, 1] - true_x[1]) ** 2)))
        err_h3.append(float(np.mean((roll["x_traj"][0, 3] - true_x[3]) ** 2)))

    # Boundary vs interior 1-step evaluation
    te = W18_ALPHA.collect_transitions(n_steps=160, seed=seed * 100 + 77)
    S_det = np.zeros_like(te["S_next"])
    for i in range(len(te["S_t"])):
        S_det[i], _ = W18_ALPHA.step_true(te["S_t"][i], int(te["A_t"][i]), rng=None)
    X_det = W18_ALPHA.observe(S_det)
    hit_bnd = np.any(np.abs(S_det) >= 1.799, axis=1)
    pred_te = cwm.predict_step(te["X_t"], te["A_t"], eid=eid_a)
    sq_te = np.mean((pred_te["x_next"] - X_det) ** 2, axis=1)
    int_nrmse = float(np.sqrt(np.mean(sq_te[~hit_bnd])) / x_scale)
    bnd_nrmse = float(np.sqrt(np.mean(sq_te[hit_bnd])) / x_scale) if np.any(hit_bnd) else int_nrmse

    # Mutation recovery (learn Beta, then mutate Alpha)
    cwm.observe_transitions(tr_b["X_t"], tr_b["A_t"], tr_b["X_next"], tr_b["dE"])
    cwm.memory.active_engram_id = eid_a
    pl = AutonomousPlanner(cwm, horizon=3, gamma=0.95, beta_uncertainty=0.5, surprise_threshold=3.0, surprise_window=3)
    mut = W18_ALPHA_MUTATED.collect_transitions(n_steps=60, seed=seed * 100 + 12)
    pre_e, post_e = [], []
    bx, ba, bxn, bde = [], [], [], []
    rec = False
    for t in range(60):
        xt, at, xnt, det = mut["X_t"][t], int(mut["A_t"][t]), mut["X_next"][t], float(mut["dE"][t])
        _, de_hat, _ = pl.compute_surprise(xt, at, det)
        ae = abs(det - de_hat)
        bx.append(xt); ba.append(at); bxn.append(xnt); bde.append(det)
        if not rec:
            pre_e.append(ae)
            if pl.should_recover() and len(bx) >= 30:
                pl.trigger_recovery(np.array(bx), np.array(ba), np.array(bxn), np.array(bde))
                rec = True
        else:
            post_e.append(ae)
    pre_mae = float(np.mean(pre_e[:10])) if pre_e else 1.0
    post_mae = float(np.mean(post_e[-10:])) if post_e else pre_mae
    rec_red = float(1.0 - post_mae / max(pre_mae, 1e-6))

    # Post-recovery x_next transition RMSE on mut[50:60]
    p_rec = cwm.predict_step(mut["X_t"][50:60], mut["A_t"][50:60], eid=eid_a)
    post_x_rmse = float(np.sqrt(np.mean((p_rec["x_next"] - mut["X_next"][50:60]) ** 2)))

    return {
        "x_nrmse_h1": float(np.sqrt(np.mean(err_h1)) / x_scale),
        "x_nrmse_h3": float(np.sqrt(np.mean(err_h3)) / x_scale),
        "interior_nrmse": int_nrmse,
        "boundary_nrmse": bnd_nrmse,
        "pre_mut_mae": pre_mae,
        "post_rec_mae": post_mae,
        "recovery_reduction": rec_red,
        "post_rec_x_rmse": post_x_rmse,
    }


def run_ablations(seeds: List[int] = DEV_SEEDS_18B, out_path: str = "logs/phase18b/p18b_ablation_results.json") -> Dict[str, Any]:
    variants = {
        "p18_baseline": (False, False),
        "boundary_only": (True, False),
        "basis_sync_only": (False, True),
        "combined_p18b": (True, True),
    }
    raw: Dict[str, List[Dict[str, float]]] = {k: [] for k in variants}
    for s in seeds:
        for name, (ba, bs) in variants.items():
            raw[name].append(_eval_ablation_variant(s, boundary_aware=ba, basis_sync=bs))

    summary: Dict[str, Any] = {"seeds": seeds, "n": len(seeds), "variants": {}}
    for name, rows in raw.items():
        summary["variants"][name] = {
            k: round(float(np.mean([r[k] for r in rows])), 5)
            for k in rows[0].keys()
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_single_seed_18b(seed: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Verify semantic firewall
    W18_ALPHA.make_neutral_observation(np.zeros(N_LATENT18), 10.0, 0, None)

    tr_a = W18_ALPHA.collect_transitions(n_steps=350, seed=seed * 100 + 1)
    x_scale = float(np.std(tr_a["X_t"])) + 1e-8
    de_scale = float(np.std(tr_a["dE"])) + 1e-8

    # Candidate (Phase 18B) + Phase 18 Baseline + MLP Baseline
    cwm = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True)
    eid_a = cwm.observe_transitions(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])

    cwm_p18 = CaspianWorldModel(seed=seed, boundary_aware=False, basis_sync=False)
    eid_p18 = cwm_p18.observe_transitions(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])

    mlp = MLPWorldModelBaseline(seed=seed)
    mlp.fit(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])

    # --- P18B-H1: Boundary-Aware Prediction (H=1, 2, 3, 5 + interior/boundary + PI coverage) ---
    e_cwm = {h: [] for h in (1, 2, 3, 5)}
    e_p18 = {h: [] for h in (1, 3)}
    e_mlp = {h: [] for h in (1, 3)}
    for _ in range(30):
        s_cur = rng.uniform(-1.1, 1.1, size=N_LATENT18)
        x0 = W18_ALPHA.observe(s_cur)
        acts = rng.integers(0, N_ACTIONS18, size=5)
        true_x = [x0]
        s_tmp = s_cur.copy()
        for h in range(5):
            s_tmp, _ = W18_ALPHA.step_true(s_tmp, int(acts[h]), rng=None)
            true_x.append(W18_ALPHA.observe(s_tmp))
        rc = cwm.rollout_latent(x0, acts.reshape(1, -1), eid=eid_a)
        rp = cwm_p18.rollout_latent(x0, acts.reshape(1, -1), eid=eid_p18)
        mx, _ = mlp.rollout(x0, acts.reshape(1, -1))
        for h in (1, 2, 3, 5):
            e_cwm[h].append(float(np.mean((rc["x_traj"][0, h] - true_x[h]) ** 2)))
        for h in (1, 3):
            e_p18[h].append(float(np.mean((rp["x_traj"][0, h] - true_x[h]) ** 2)))
            e_mlp[h].append(float(np.mean((mx[0, h] - true_x[h]) ** 2)))

    te_bnd = W18_ALPHA.collect_transitions(n_steps=200, seed=seed * 100 + 77)
    S_det = np.zeros_like(te_bnd["S_next"])
    for i in range(len(te_bnd["S_t"])):
        S_det[i], _ = W18_ALPHA.step_true(te_bnd["S_t"][i], int(te_bnd["A_t"][i]), rng=None)
    X_det = W18_ALPHA.observe(S_det)
    hit = np.any(np.abs(S_det) >= 1.799, axis=1)
    pc_b = cwm.predict_step(te_bnd["X_t"], te_bnd["A_t"], eid=eid_a)
    pp_b = cwm_p18.predict_step(te_bnd["X_t"], te_bnd["A_t"], eid=eid_p18)
    sq_c = np.mean((pc_b["x_next"] - X_det) ** 2, axis=1)
    sq_p = np.mean((pp_b["x_next"] - X_det) ** 2, axis=1)
    in_pi_c = (np.abs(pc_b["de_mean"] - te_bnd["dE"]) <= 1.96 * pc_b["de_std"]).astype(float)

    h1 = {
        "cwm_x_h1": float(np.sqrt(np.mean(e_cwm[1])) / x_scale),
        "cwm_x_h2": float(np.sqrt(np.mean(e_cwm[2])) / x_scale),
        "cwm_x_h3": float(np.sqrt(np.mean(e_cwm[3])) / x_scale),
        "cwm_x_h5": float(np.sqrt(np.mean(e_cwm[5])) / x_scale),
        "p18_x_h1": float(np.sqrt(np.mean(e_p18[1])) / x_scale),
        "p18_x_h3": float(np.sqrt(np.mean(e_p18[3])) / x_scale),
        "mlp_x_h1": float(np.sqrt(np.mean(e_mlp[1])) / x_scale),
        "mlp_x_h3": float(np.sqrt(np.mean(e_mlp[3])) / x_scale),
        "cwm_interior_nrmse": float(np.sqrt(np.mean(sq_c[~hit])) / x_scale),
        "cwm_boundary_nrmse": float(np.sqrt(np.mean(sq_c[hit])) / x_scale),
        "p18_interior_nrmse": float(np.sqrt(np.mean(sq_p[~hit])) / x_scale),
        "p18_boundary_nrmse": float(np.sqrt(np.mean(sq_p[hit])) / x_scale),
        "pi95_interior": float(np.mean(in_pi_c[~hit])),
        "pi95_boundary": float(np.mean(in_pi_c[hit])),
    }

    # --- P18B-H2: Representation Synchronization Equivalence & Serialization ---
    cwm_rot = CaspianWorldModel.from_bytes(cwm.to_bytes())
    # Serialization round-trip diff
    p_orig = cwm.predict_step(te_bnd["X_t"][:20], te_bnd["A_t"][:20], eid=eid_a)
    p_ser = cwm_rot.predict_step(te_bnd["X_t"][:20], te_bnd["A_t"][:20], eid=eid_a)
    ser_diff = float(np.max(np.abs(p_orig["x_next"] - p_ser["x_next"])) + np.max(np.abs(p_orig["de_std"] - p_ser["de_std"])))

    # Apply arbitrary random orthogonal rotation Q in O(4) + shift to cwm_rot's adapter and synchronize!
    Q_rand, _ = np.linalg.qr(rng.standard_normal((N_LATENT18, N_LATENT18)))
    e_rot = cwm_rot._get_engram(eid_a)
    W_old_r = e_rot.model.W_adapter.copy()
    xm_old_r = e_rot.model.x_mean.copy()
    b_old_r = e_rot.model.b_adapter.copy()
    e_rot.model.W_adapter = W_old_r @ Q_rand
    e_rot.model.b_adapter = b_old_r @ Q_rand + rng.normal(0.0, 0.2, size=(1, N_LATENT18))
    synced_ok, orth_err = cwm_rot.sync_dynamics_to_basis(eid_a)
    cwm_rot.assert_basis_synchronized(eid_a)

    # Check exact predictive equivalence in observation space (mean AND full propagated covariance!)
    # Use interior points so clipping box corners (which are axis-aligned in W_old) don't confound linear equivalence
    Z_int_pts = rng.uniform(-0.8, 0.8, size=(15, N_LATENT18))
    X_int_pts = (Z_int_pts - b_old_r) @ W_old_r.T + xm_old_r
    A_int_pts = rng.integers(0, N_ACTIONS18, size=15)
    p_before = cwm.predict_step(X_int_pts, A_int_pts, eid=eid_a)
    p_after = cwm_rot.predict_step(X_int_pts, A_int_pts, eid=eid_a)
    mean_equiv_err = float(np.max(np.abs(p_before["x_next"] - p_after["x_next"])))
    # Compare full predictive covariance mapped to observation space: W P_next W^T
    W_new_r = e_rot.model.W_adapter
    P_obs_before = np.einsum("ij,njk,lk->nil", W_old_r, p_before["P_next"], W_old_r)
    P_obs_after = np.einsum("ij,njk,lk->nil", W_new_r, p_after["P_next"], W_new_r)
    var_equiv_err = float(np.max(np.abs(P_obs_before - P_obs_after)))

    h2 = {
        "synced_ok": bool(synced_ok),
        "orth_err": float(orth_err),
        "mean_equiv_err": mean_equiv_err,
        "var_equiv_err": var_equiv_err,
        "serialization_diff": ser_diff,
        "basis_version_after_sync": int(cwm_rot.dynamics[eid_a].basis_version),
    }

    # --- P18B-H3 & P18B-H4: Continual Learning, Zero Forgetting, Retrieval & Synchronized Mutation Recovery ---
    te_a_ret = W18_ALPHA.collect_transitions(n_steps=60, seed=seed * 100 + 8)
    err_a_pre = float(np.sqrt(np.mean((cwm.predict_step(te_a_ret["X_t"], te_a_ret["A_t"], eid=eid_a)["de_mean"] - te_a_ret["dE"]) ** 2)) / de_scale)

    tr_b = W18_BETA.collect_transitions(n_steps=250, seed=seed * 100 + 9)
    eid_b = cwm.observe_transitions(tr_b["X_t"], tr_b["A_t"], tr_b["X_next"], tr_b["dE"])
    cwm_p18.observe_transitions(tr_b["X_t"], tr_b["A_t"], tr_b["X_next"], tr_b["dE"])

    pa = W18_ALPHA.collect_transitions(n_steps=8, seed=seed * 100 + 10)
    pb = W18_BETA.collect_transitions(n_steps=8, seed=seed * 100 + 11)
    ra, _, _ = cwm.retrieve_context(pa["X_t"], pa["A_t"], pa["X_next"], pa["dE"])
    rb, _, _ = cwm.retrieve_context(pb["X_t"], pb["A_t"], pb["X_next"], pb["dE"])
    retr_acc = float((int(ra == eid_a) + int(rb == eid_b)) / 2.0)

    err_a_post = float(np.sqrt(np.mean((cwm.predict_step(te_a_ret["X_t"], te_a_ret["A_t"], eid=ra)["de_mean"] - te_a_ret["dE"]) ** 2)) / de_scale)
    forg_a = max(0.0, err_a_post - err_a_pre)

    # Mid-episode mutation recovery: Phase 18B (synchronized) vs Phase 18 baseline (unsynchronized)
    def _run_mut(model_obj, eid_target):
        model_obj.memory.active_engram_id = eid_target
        pl = AutonomousPlanner(model_obj, horizon=3, gamma=0.95, beta_uncertainty=0.5, surprise_threshold=3.0, surprise_window=3)
        mut = W18_ALPHA_MUTATED.collect_transitions(n_steps=60, seed=seed * 100 + 12)
        pre_e, post_e = [], []
        bx, ba, bxn, bde = [], [], [], []
        rec = False
        for t in range(60):
            xt, at, xnt, det = mut["X_t"][t], int(mut["A_t"][t]), mut["X_next"][t], float(mut["dE"][t])
            _, dh, _ = pl.compute_surprise(xt, at, det)
            ae = abs(det - dh)
            bx.append(xt); ba.append(at); bxn.append(xnt); bde.append(det)
            if not rec:
                pre_e.append(ae)
                if pl.should_recover() and len(bx) >= 30:
                    pl.trigger_recovery(np.array(bx), np.array(ba), np.array(bxn), np.array(bde))
                    rec = True
            else:
                post_e.append(ae)
        pre_m = float(np.mean(pre_e[:10])) if pre_e else 1.0
        post_m = float(np.mean(post_e[-10:])) if post_e else pre_m
        p_x = model_obj.predict_step(mut["X_t"][50:60], mut["A_t"][50:60], eid=eid_target)["x_next"]
        x_rmse = float(np.sqrt(np.mean((p_x - mut["X_next"][50:60]) ** 2)))
        return pre_m, post_m, float(1.0 - post_m / max(pre_m, 1e-6)), x_rmse

    pre_18b, post_18b, red_18b, xrmse_18b = _run_mut(cwm, eid_a)
    pre_18, post_18, red_18, xrmse_18 = _run_mut(cwm_p18, eid_p18)
    cwm.assert_basis_synchronized(eid_a)

    te_b_ret = W18_BETA.collect_transitions(n_steps=60, seed=seed * 100 + 13)
    beta_ret_post = float(np.sqrt(np.mean((cwm.predict_step(te_b_ret["X_t"], te_b_ret["A_t"], eid=eid_b)["de_mean"] - te_b_ret["dE"]) ** 2)) / de_scale)

    h3 = {
        "pre_mut_mae_18b": pre_18b,
        "post_rec_mae_18b": post_18b,
        "recovery_reduction_18b": red_18b,
        "post_rec_x_rmse_18b": xrmse_18b,
        "post_rec_mae_p18": post_18,
        "recovery_reduction_p18": red_18,
        "post_rec_x_rmse_p18": xrmse_18,
    }
    h4 = {
        "forgetting_alpha": forg_a,
        "retrieval_acc": retr_acc,
        "beta_retention_post_mutation": beta_ret_post,
        "memory_footprint": cwm.memory_footprint_bytes(),
    }

    # --- P18B-H5: Planning Robustness (W18_alpha N=350 + Trap World risk-aware vs risk-neutral) ---
    # Re-create clean cwm on W18_alpha for planning (since eid_a was mutated in H3)
    cwm_pl = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True)
    eid_pl = cwm_pl.observe_transitions(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])
    pl_casp = AutonomousPlanner(cwm_pl, horizon=3, gamma=0.95, beta_uncertainty=0.5)
    pl_mem_g = AutonomousPlanner(cwm_pl, horizon=1, gamma=0.95, beta_uncertainty=0.0)

    rets_350 = {k: [] for k in ["caspian", "oracle", "mlp", "memory_no_planner", "reactive", "random"]}
    for ei, s0 in enumerate(SADDLE_INITS):
        es = seed * 2000 + ei
        rr = np.random.default_rng(es + 99)
        rets_350["caspian"].append(run_episode(W18_ALPHA, lambda x, s, t: pl_casp.plan_action(x, eid=eid_pl)[0], s0, steps=20, seed=es)["total_return"])
        rets_350["oracle"].append(run_episode(W18_ALPHA, lambda x, s, t: oracle_plan_action(W18_ALPHA, s, horizon=3), s0, steps=20, seed=es)["total_return"])
        rets_350["mlp"].append(run_episode(W18_ALPHA, lambda x, s, t: mlp.plan_action(x, horizon=3), s0, steps=20, seed=es)["total_return"])
        rets_350["memory_no_planner"].append(run_episode(W18_ALPHA, lambda x, s, t: pl_mem_g.plan_action(x, eid=eid_pl)[0], s0, steps=20, seed=es)["total_return"])
        rets_350["reactive"].append(run_episode(W18_ALPHA, lambda x, s, t: mlp.plan_action(x, horizon=1), s0, steps=20, seed=es)["total_return"])
        rets_350["random"].append(run_episode(W18_ALPHA, lambda x, s, t: int(rr.integers(0, N_ACTIONS18)), s0, steps=20, seed=es)["total_return"])

    tr_trap = W18_TRAP.collect_transitions(n_steps=200, seed=seed * 100 + 5, s_clip_train=(-1.3, 0.45))
    cwm_trap = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True)
    eid_trap = cwm_trap.observe_transitions(tr_trap["X_t"], tr_trap["A_t"], tr_trap["X_next"], tr_trap["dE"])
    pl_safe = AutonomousPlanner(cwm_trap, horizon=3, gamma=0.95, beta_uncertainty=3.5)
    pl_neut = AutonomousPlanner(cwm_trap, horizon=3, gamma=0.95, beta_uncertainty=0.0)
    s_ti = np.array([0.2, 0.2, 0.0, 0.0], dtype=float)
    ret_safe = float(np.mean([run_episode(W18_TRAP, lambda x, s, t: pl_safe.plan_action(x, eid=eid_trap)[0], s_ti, steps=15, seed=seed * 1000 + ep)["total_return"] for ep in range(5)]))
    ret_neut = float(np.mean([run_episode(W18_TRAP, lambda x, s, t: pl_neut.plan_action(x, eid=eid_trap)[0], s_ti, steps=15, seed=seed * 1000 + ep)["total_return"] for ep in range(5)]))

    h5 = {f"ret350_{k}": float(np.mean(v)) for k, v in rets_350.items()}
    h5["oracle_ratio"] = float(h5["ret350_caspian"] / max(h5["ret350_oracle"], 1e-6))
    h5["trap_safe"] = ret_safe
    h5["trap_neutral"] = ret_neut

    # --- P18B-H6: Multi-Regime Data Budgets (N=80, N=160) & Unseen Shifted World (W18_gamma_shifted) ---
    h6: Dict[str, float] = {}
    for budget in (80, 160):
        tr_bgt = W18_ALPHA.collect_transitions(n_steps=budget, seed=seed * 100 + budget)
        cwm_b = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True)
        eid_bgt = cwm_b.observe_transitions(tr_bgt["X_t"], tr_bgt["A_t"], tr_bgt["X_next"], tr_bgt["dE"])
        mlp_b = MLPWorldModelBaseline(seed=seed)
        mlp_b.fit(tr_bgt["X_t"], tr_bgt["A_t"], tr_bgt["X_next"], tr_bgt["dE"])
        pl_b = AutonomousPlanner(cwm_b, horizon=3, gamma=0.95, beta_uncertainty=0.5)
        rc_b = [run_episode(W18_ALPHA, lambda x, s, t: pl_b.plan_action(x, eid=eid_bgt)[0], s0, steps=20, seed=seed * 4000 + budget + ei)["total_return"] for ei, s0 in enumerate(SADDLE_INITS[:3])]
        rm_b = [run_episode(W18_ALPHA, lambda x, s, t: mlp_b.plan_action(x, horizon=3), s0, steps=20, seed=seed * 4000 + budget + ei)["total_return"] for ei, s0 in enumerate(SADDLE_INITS[:3])]
        h6[f"ret_n{budget}_caspian"] = float(np.mean(rc_b))
        h6[f"ret_n{budget}_mlp"] = float(np.mean(rm_b))

    cwm_sh = CaspianWorldModel(seed=seed, boundary_aware=True, basis_sync=True)
    cwm_sh.observe_transitions(tr_a["X_t"], tr_a["A_t"], tr_a["X_next"], tr_a["dE"])
    tr_g = W18_GAMMA_SHIFTED.collect_transitions(n_steps=60, seed=seed * 100 + 14)
    eid_g = cwm_sh.observe_transitions(tr_g["X_t"], tr_g["A_t"], tr_g["X_next"], tr_g["dE"])
    mlp_g = MLPWorldModelBaseline(seed=seed)
    mlp_g.fit(tr_g["X_t"], tr_g["A_t"], tr_g["X_next"], tr_g["dE"])
    pl_g = AutonomousPlanner(cwm_sh, horizon=3, gamma=0.95, beta_uncertainty=0.5)
    sh_c = [run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: pl_g.plan_action(x, eid=eid_g)[0], s0, steps=20, seed=seed * 3000 + ei)["total_return"] for ei, s0 in enumerate(UNSEEN_SHIFT_INITS)]
    sh_m = [run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: mlp_g.plan_action(x, horizon=3), s0, steps=20, seed=seed * 3000 + ei)["total_return"] for ei, s0 in enumerate(UNSEEN_SHIFT_INITS)]
    sh_r = [run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: mlp_g.plan_action(x, horizon=1), s0, steps=20, seed=seed * 3000 + ei)["total_return"] for ei, s0 in enumerate(UNSEEN_SHIFT_INITS)]
    h6["shifted_caspian"] = float(np.mean(sh_c))
    h6["shifted_mlp"] = float(np.mean(sh_m))
    h6["shifted_reactive"] = float(np.mean(sh_r))

    # --- P18B-H7: Integrated Consistency (Causal F1, Counterfactuals, OOD AUROC) ---
    e_pl = cwm_pl._get_engram(eid_pl)
    B_true_z = (W18_ALPHA.B_act @ W18_ALPHA.M) @ e_pl.model.W_adapter
    tmask = (np.abs(B_true_z[1:]) > 0.20).astype(int)
    pmask = cwm_pl.dynamics[eid_pl].causal_action_mask[1:].astype(int)
    tp = int(np.sum((tmask == 1) & (pmask == 1)))
    fp = int(np.sum((tmask == 0) & (pmask == 1)))
    fn = int(np.sum((tmask == 1) & (pmask == 0)))
    causal_f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))

    cf_rhos, cf_top1 = [], []
    for _ in range(30):
        sq = rng.uniform(-1.1, 1.1, size=N_LATENT18)
        xq = W18_ALPHA.observe(sq)
        ao = int(rng.integers(0, N_ACTIONS18))
        ns = rng.normal(0.0, W18_ALPHA.noise_s, size=N_LATENT18)
        ne = float(rng.normal(0.0, W18_ALPHA.noise_e))
        so = np.clip(W18_ALPHA.A_drift @ sq + W18_ALPHA.B_act[ao] + ns, -W18_ALPHA.state_bound, W18_ALPHA.state_bound)
        deo = float(W18_ALPHA.rule_fn(so.reshape(1, -1))[0]) - float(W18_ALPHA.action_costs[ao]) + ne
        tcf = np.array([
            float(W18_ALPHA.rule_fn(np.clip(W18_ALPHA.A_drift @ sq + W18_ALPHA.B_act[a] + ns, -W18_ALPHA.state_bound, W18_ALPHA.state_bound).reshape(1, -1))[0])
            - float(W18_ALPHA.action_costs[a]) + ne
            for a in range(N_ACTIONS18)
        ])
        pcf = cwm_pl.evaluate_counterfactuals(xq, observed_action=ao, X_next_obs=W18_ALPHA.observe(so), dE_obs=deo, eid=eid_pl)["de_mean"]
        cf_rhos.append(spearman_corr(pcf, tcf))
        cf_top1.append(1.0 if int(np.argmax(pcf)) == int(np.argmax(tcf)) else 0.0)

    te_id = W18_ALPHA.collect_transitions(n_steps=50, seed=seed * 100 + 3)
    te_oo = W18_BETA.collect_transitions(n_steps=50, seed=seed * 100 + 4)
    ood_auroc = compute_auroc(
        np.concatenate([cwm_pl.subspace_ood_score(te_id["X_t"], eid=eid_pl), cwm_pl.subspace_ood_score(te_oo["X_t"], eid=eid_pl)]),
        np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]),
    )

    h7 = {
        "causal_f1": causal_f1,
        "cf_spearman_rho": float(np.mean(cf_rhos)),
        "cf_top1_acc": float(np.mean(cf_top1)),
        "ood_auroc": float(ood_auroc),
    }

    return {
        "seed": seed,
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7,
    }


def run_confirmatory_18b(
    seeds: List[int] = CONFIRMATORY_SEEDS_18B,
    out_path: str = "logs/phase18b/p18b_confirmatory_results.json",
) -> Dict[str, Any]:
    rows = [run_single_seed_18b(s) for s in seeds]

    def vec(sec: str, key: str) -> np.ndarray:
        return np.array([r[sec][key] for r in rows], dtype=float)

    # H1
    h1_vs_mlp1 = paired_stats(vec("H1", "cwm_x_h1"), vec("H1", "mlp_x_h1"), higher_is_better=False, seed=1)
    h1_vs_mlp3 = paired_stats(vec("H1", "cwm_x_h3"), vec("H1", "mlp_x_h3"), higher_is_better=False, seed=2)
    h1_vs_p18b = paired_stats(vec("H1", "cwm_boundary_nrmse"), vec("H1", "p18_boundary_nrmse"), higher_is_better=False, seed=3)
    h1_holm = holm_bonferroni({"mlp_h1": h1_vs_mlp1["perm_p"], "mlp_h3": h1_vs_mlp3["perm_p"]})
    h1_met = bool(
        float(np.mean(vec("H1", "cwm_x_h1"))) < 0.05
        and h1_vs_mlp1["ci_strictly_positive"] and h1_vs_mlp3["ci_strictly_positive"]
        and all(p < 0.05 for p in h1_holm.values())
        and h1_vs_p18b["ci_strictly_positive"] and h1_vs_p18b["perm_p"] < 0.05
    )

    # H2
    max_mean_eq = float(np.max(vec("H2", "mean_equiv_err")))
    max_var_eq = float(np.max(vec("H2", "var_equiv_err")))
    max_ser = float(np.max(vec("H2", "serialization_diff")))
    all_synced = all(r["H2"]["synced_ok"] for r in rows)
    h2_met = bool(all_synced and max_mean_eq < 1e-11 and max_var_eq < 1e-11 and max_ser == 0.0)

    # H3
    h3_red = float(np.mean(vec("H3", "recovery_reduction_18b")))
    h3_vs_p18 = paired_stats(vec("H3", "post_rec_mae_18b"), vec("H3", "post_rec_mae_p18"), higher_is_better=False, seed=4)
    h3_met = bool(h3_red >= 0.50 and h3_vs_p18["ci_strictly_positive"] and h3_vs_p18["perm_p"] < 0.05)

    # H4
    h4_forg = float(np.mean(vec("H4", "forgetting_alpha")))
    h4_retr = float(np.mean(vec("H4", "retrieval_acc")))
    h4_beta = float(np.mean(vec("H4", "beta_retention_post_mutation")))
    h4_met = bool(h4_forg < 1e-6 and h4_retr >= 0.95 and h4_beta < 0.10)

    # H5
    casp350 = vec("H5", "ret350_caspian")
    h5_comps = {
        b: paired_stats(casp350, vec("H5", f"ret350_{b}"), higher_is_better=True, seed=10 + i)
        for i, b in enumerate(["memory_no_planner", "reactive", "random", "mlp"])
    }
    h5_holm = holm_bonferroni({k: h5_comps[k]["perm_p"] for k in ("memory_no_planner", "reactive", "random")})
    h5_trap = paired_stats(vec("H5", "trap_safe"), vec("H5", "trap_neutral"), higher_is_better=True, seed=15)
    h5_orac = float(np.mean(vec("H5", "oracle_ratio")))
    h5_met = bool(
        h5_orac >= 0.90
        and all(h5_comps[k]["ci_strictly_positive"] for k in ("memory_no_planner", "reactive", "random"))
        and all(p < 0.05 for p in h5_holm.values())
        and h5_trap["ci_strictly_positive"] and h5_trap["perm_p"] < 0.05
    )

    # H6
    h6_n80 = paired_stats(vec("H6", "ret_n80_caspian"), vec("H6", "ret_n80_mlp"), higher_is_better=True, seed=20)
    h6_n160 = paired_stats(vec("H6", "ret_n160_caspian"), vec("H6", "ret_n160_mlp"), higher_is_better=True, seed=21)
    h6_sh_mlp = paired_stats(vec("H6", "shifted_caspian"), vec("H6", "shifted_mlp"), higher_is_better=True, seed=22)
    h6_sh_react = paired_stats(vec("H6", "shifted_caspian"), vec("H6", "shifted_reactive"), higher_is_better=True, seed=23)
    h6_holm = holm_bonferroni({"n80_vs_mlp": h6_n80["perm_p"], "shifted_vs_mlp": h6_sh_mlp["perm_p"]})
    h6_met = bool(
        h6_n80["ci_strictly_positive"] and h6_sh_mlp["ci_strictly_positive"]
        and all(p < 0.05 for p in h6_holm.values())
    )

    # H7 + Integrity
    det_r = run_single_seed_18b(seeds[0])
    det_diff = abs(det_r["H5"]["ret350_caspian"] - rows[0]["H5"]["ret350_caspian"])
    ast_aud = ast_leakage_audit_phase18()
    h7_f1 = float(np.mean(vec("H7", "causal_f1")))
    h7_rho = float(np.mean(vec("H7", "cf_spearman_rho")))
    h7_top1 = float(np.mean(vec("H7", "cf_top1_acc")))
    h7_auroc = float(np.mean(vec("H7", "ood_auroc")))
    h7_met = bool(h7_f1 >= 0.90 and h7_rho >= 0.85 and h7_top1 >= 0.85 and h7_auroc >= 0.90 and det_diff == 0.0 and ast_aud["passed"])

    verdicts = {
        "P18B-H1": h1_met, "P18B-H2": h2_met, "P18B-H3": h3_met,
        "P18B-H4": h4_met, "P18B-H5": h5_met, "P18B-H6": h6_met, "P18B-H7": h7_met,
    }
    out = {
        "experiment_id": "CSP-P18B-E001",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "hypotheses_met_count": f"{sum(verdicts.values())}/7",
        "verdicts": verdicts,
        "H1": {
            "met": h1_met,
            "means": {k: round(float(np.mean(vec("H1", k))), 5) for k in rows[0]["H1"].keys()},
            "vs_mlp_h1": h1_vs_mlp1, "vs_mlp_h3": h1_vs_mlp3, "vs_p18_boundary": h1_vs_p18b,
            "holm_p": h1_holm,
        },
        "H2": {
            "met": h2_met,
            "all_synced": all_synced,
            "max_orth_err": float(np.max(vec("H2", "orth_err"))),
            "max_mean_equiv_err": max_mean_eq,
            "max_var_equiv_err": max_var_eq,
            "max_serialization_diff": max_ser,
        },
        "H3": {
            "met": h3_met,
            "means": {k: round(float(np.mean(vec("H3", k))), 5) for k in rows[0]["H3"].keys()},
            "vs_p18_post_rec_mae": h3_vs_p18,
        },
        "H4": {
            "met": h4_met,
            "mean_forgetting_alpha": round(h4_forg, 6),
            "mean_retrieval_acc": round(h4_retr, 5),
            "mean_beta_retention_post_mutation": round(h4_beta, 5),
            "memory_footprint": rows[0]["H4"]["memory_footprint"],
        },
        "H5": {
            "met": h5_met,
            "means": {k: round(float(np.mean(vec("H5", k))), 4) for k in rows[0]["H5"].keys()},
            "comparisons": h5_comps,
            "holm_p": h5_holm,
            "trap_vs_neutral": h5_trap,
        },
        "H6": {
            "met": h6_met,
            "means": {k: round(float(np.mean(vec("H6", k))), 4) for k in rows[0]["H6"].keys()},
            "n80_vs_mlp": h6_n80, "n160_vs_mlp": h6_n160,
            "shifted_vs_mlp": h6_sh_mlp, "shifted_vs_reactive": h6_sh_react,
            "holm_p": h6_holm,
        },
        "H7": {
            "met": h7_met,
            "mean_causal_f1": round(h7_f1, 5),
            "mean_cf_spearman_rho": round(h7_rho, 5),
            "mean_cf_top1_acc": round(h7_top1, 5),
            "mean_ood_auroc": round(h7_auroc, 5),
            "determinism_diff": float(det_diff),
            "ast_leakage_passed": ast_aud["passed"],
            "ast_violations": ast_aud["violations"],
            "mean_wall_time_s": round(float(np.mean([r["wall_time_s"] for r in rows])), 3),
        },
        "per_seed": rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    if mode == "ablation":
        res = run_ablations()
        print("=== Phase 18B Stage 3-4 Ablations (Dev Seeds 9301-9305, n=5) ===")
        for vname, m in res["variants"].items():
            print(f"  {vname:16s}: h1={m['x_nrmse_h1']:.5f} h3={m['x_nrmse_h3']:.5f} int={m['interior_nrmse']:.5f} bnd={m['boundary_nrmse']:.5f} | rec_red={m['recovery_reduction']*100:5.1f}% post_mae={m['post_rec_mae']:.3f} post_x={m['post_rec_x_rmse']:.4f}")
    elif mode == "confirmatory":
        res = run_confirmatory_18b()
        print(f"=== Phase 18B Confirmatory (Seeds 191-205, n={res['n_seeds']}) ===")
        print("Hypotheses met:", res["hypotheses_met_count"], res["verdicts"])
        print("H1:", res["H1"]["means"]["cwm_x_h1"], "vs mlp", res["H1"]["means"]["mlp_x_h1"], "| h3:", res["H1"]["means"]["cwm_x_h3"], "vs mlp", res["H1"]["means"]["mlp_x_h3"], "| bnd:", res["H1"]["means"]["cwm_boundary_nrmse"], "vs p18", res["H1"]["means"]["p18_boundary_nrmse"])
        print("H2:", res["H2"])
        print("H3:", res["H3"]["means"], "p=", res["H3"]["vs_p18_post_rec_mae"]["perm_p"])
        print("H4:", res["H4"])
        print("H5:", res["H5"]["means"], "trap_p=", res["H5"]["trap_vs_neutral"]["perm_p"])
        print("H6:", res["H6"]["means"], "holm_p=", res["H6"]["holm_p"])
        print("H7:", res["H7"])
    else:
        raise ValueError(f"Unknown mode: {mode}")
