"""Phase 19 (CSP-P19-E001): Real-World Transfer & Operational Decision-Support Prototype Trials.

Evaluates CaspianOpsAgent (wrapping untouched Phase 17C/18D world model & planner) across:
- Trial 1 (P19-H1): Nominal operational control on Cluster_A vs ThresholdAutoscaler,
  ReactiveController (H=1), Fair MLP Controller (H=3), Random, and Oracle upper bound.
- Trial 2 (P19-H2): Few-shot (N=60) migration to Cluster_B (new telemetry schema + offset)
  vs Fair MLP, ThresholdAutoscaler, and Scratch Caspian (full_reconstruction).
- Trial 3 (P19-H3): Mid-stream hardware/workload incident on Cluster_A_incident
  (autonomous detection, human-oversight escalation, MAE recovery, post-incident control).
- Trial 4 (P19-H4/H5/H6): Uncertainty-driven OOD oversight AUROC, nominal false-escalation rate,
  zero-forgetting retention, autonomous retrieval, zero raw telemetry privacy, decision latency,
  determinism, and AST leakage audit.
"""
import ast
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List
import numpy as np

from experiments.phase18_experiments import (
    MLPWorldModelBaseline,
    ast_leakage_audit_phase18,
    compute_auroc,
    holm_bonferroni,
    paired_stats,
)
from prototype.caspian_ops_agent import CaspianOpsAgent
from prototype.ops_env import (
    CLUSTER_A_INCIDENT,
    CLUSTER_A_NOMINAL,
    CLUSTER_B_MIGRATED,
    OPS_N_ACTIONS,
    OperationalClusterEnv,
    ThresholdAutoscalerBaseline,
)

SMOKE_SEEDS_19 = [9501, 9502, 9503]
CONFIRMATORY_SEEDS_19 = list(range(251, 266))  # 15 disjoint seeds (251..265)

# Overloaded initial cluster states requiring multi-step recovery (Scale-Out + Cooling)
OVERLOAD_INITS = [
    np.array([-0.65, -0.65, 0.0, 0.0], dtype=float),
    np.array([-0.70, -0.55, -0.2, 0.1], dtype=float),
    np.array([-0.55, -0.70, 0.1, -0.1], dtype=float),
    np.array([-0.60, -0.60, -0.1, 0.0], dtype=float),
]


def oracle_ops_action(env: OperationalClusterEnv, s_t: np.ndarray, horizon: int = 3, gamma: float = 0.95) -> int:
    """DIAGNOSTIC UPPER BOUND ONLY — never accessible to CaspianOpsAgent."""
    seqs = np.array(list(itertools.product(range(OPS_N_ACTIONS), repeat=horizon)), dtype=int)
    N = len(seqs)
    S_cur = np.tile(np.asarray(s_t, dtype=float).reshape(1, -1), (N, 1))
    scores = np.zeros(N, dtype=float)
    for h in range(horizon):
        acts = seqs[:, h]
        S_cur = np.clip(S_cur @ env.A_dyn.T + env.B_act[acts], -env.state_bound, env.state_bound)
        de_h = env.sla_fn(S_cur) - env.action_costs[acts]
        scores += (gamma ** h) * de_h
    return int(seqs[int(np.argmax(scores)), 0])


def run_ops_episode(
    env: OperationalClusterEnv,
    policy_fn: Callable[[np.ndarray, np.ndarray, int], int],
    s_init: np.ndarray,
    steps: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    s = np.asarray(s_init, dtype=float).copy()
    total_score = 0.0
    breaches = 0
    for t in range(steps):
        x = env.observe_telemetry(s)
        a = int(policy_fn(x, s, t))
        s, de, breach = env.step(s, a, rng=rng)
        total_score += de
        breaches += int(breach)
    return {"total_score": float(total_score), "breach_rate": float(breaches / steps)}


def ast_leakage_audit_phase19() -> Dict[str, Any]:
    base = ast_leakage_audit_phase18()
    forbidden = {
        "cluster_id", "world_id", "s_true", "A_dyn", "B_act", "sla_fn",
        "telemetry_matrix", "oracle", "hidden_state_delta",
    }
    agent_path = Path("prototype/caspian_ops_agent.py")
    tree = ast.parse(agent_path.read_text(encoding="utf-8"))
    violations = list(base["violations"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            violations.append(f"{agent_path}:{node.lineno}: Name '{node.id}'")
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            violations.append(f"{agent_path}:{node.lineno}: Attr '{node.attr}'")
    return {
        "files_scanned": base["files_scanned"] + [str(agent_path)],
        "violations": violations,
        "passed": len(violations) == 0,
    }


def run_single_seed_19(seed: int) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Verify Semantic Firewall on neutral Observation
    CLUSTER_A_NOMINAL.make_neutral_observation(np.zeros(4), 0.0, 0, None)

    # Collect Cluster_A operational telemetry log (350 ticks)
    log_a = CLUSTER_A_NOMINAL.collect_telemetry_log(n_steps=350, seed=seed * 100 + 1)
    de_scale_a = float(np.std(log_a["dE"])) + 1e-8

    # Fit CaspianOpsAgent & conventional baselines on exact same log_a
    agent = CaspianOpsAgent(seed=seed, horizon=3, beta_uncertainty=0.5, transfer_mode="uncertainty_guided")
    eid_a = agent.ingest_operational_log(log_a["X_t"], log_a["A_t"], log_a["X_next"], log_a["dE"])

    thresh_base = ThresholdAutoscalerBaseline()
    thresh_base.fit(log_a["X_t"], log_a["A_t"], log_a["dE"])

    mlp_a = MLPWorldModelBaseline(seed=seed)
    mlp_a.fit(log_a["X_t"], log_a["A_t"], log_a["X_next"], log_a["dE"])

    # --- Trial 1 (P19-H1): Nominal Operational Control on Cluster_A ---
    scores_a = {k: [] for k in ("caspian", "oracle", "mlp", "threshold", "reactive", "random")}
    breaches_a = {k: [] for k in ("caspian", "oracle", "mlp", "threshold", "reactive", "random")}
    for ei, s0 in enumerate(OVERLOAD_INITS):
        es = seed * 2000 + ei
        rr = np.random.default_rng(es + 77)
        for name, fn in [
            ("caspian", lambda x, s, t: agent.recommend_action(x, eid=eid_a)[0]),
            ("oracle", lambda x, s, t: oracle_ops_action(CLUSTER_A_NOMINAL, s, horizon=3)),
            ("mlp", lambda x, s, t: mlp_a.plan_action(x, horizon=3)),
            ("threshold", lambda x, s, t: thresh_base.select_action(x)),
            ("reactive", lambda x, s, t: mlp_a.plan_action(x, horizon=1)),
            ("random", lambda x, s, t: int(rr.integers(0, OPS_N_ACTIONS))),
        ]:
            ep = run_ops_episode(CLUSTER_A_NOMINAL, fn, s0, steps=20, seed=es)
            scores_a[name].append(ep["total_score"])
            breaches_a[name].append(ep["breach_rate"])

    h1 = {f"score_{k}": float(np.mean(v)) for k, v in scores_a.items()}
    h1.update({f"breach_{k}": float(np.mean(v)) for k, v in breaches_a.items()})
    h1["oracle_ratio"] = float(h1["score_caspian"] / max(h1["score_oracle"], 1e-6))

    # --- Trial 4a (P19-H4): Uncertainty-Driven Human Oversight (Before Cluster_B Migration!) ---
    te_a_nom = CLUSTER_A_NOMINAL.collect_telemetry_log(n_steps=60, seed=seed * 100 + 3)
    te_b_unseen = CLUSTER_B_MIGRATED.collect_telemetry_log(n_steps=60, seed=seed * 100 + 4)
    ood_a = agent.world_model.subspace_ood_score(te_a_nom["X_t"], eid=eid_a)
    ood_b = agent.world_model.subspace_ood_score(te_b_unseen["X_t"], eid=eid_a)
    ood_auroc = compute_auroc(np.concatenate([ood_a, ood_b]), np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)]))
    false_esc_rate = float(np.mean(ood_a > agent.ood_escalation_threshold))
    true_esc_rate = float(np.mean(ood_b > agent.ood_escalation_threshold))
    h4 = {"ood_auroc": float(ood_auroc), "nominal_false_escalation_rate": false_esc_rate, "unseen_escalation_recall": true_esc_rate}

    # --- Trial 2 (P19-H2): Few-Shot (N=60) Telemetry / Cluster Migration to Cluster_B ---
    err_a_pre = float(np.sqrt(np.mean((agent.world_model.predict_step(te_a_nom["X_t"], te_a_nom["A_t"], eid=eid_a)["de_mean"] - te_a_nom["dE"]) ** 2)) / de_scale_a)

    log_b = CLUSTER_B_MIGRATED.collect_telemetry_log(n_steps=60, seed=seed * 100 + 10)
    te_b = CLUSTER_B_MIGRATED.collect_telemetry_log(n_steps=100, seed=seed * 100 + 11)
    x_scale_b = float(np.std(te_b["X_t"])) + 1e-8

    eid_b = agent.ingest_operational_log(log_b["X_t"], log_b["A_t"], log_b["X_next"], log_b["dE"])

    scratch_b = CaspianOpsAgent(seed=seed, horizon=3, beta_uncertainty=0.5, transfer_mode="full_reconstruction")
    eid_sb = scratch_b.ingest_operational_log(log_b["X_t"], log_b["A_t"], log_b["X_next"], log_b["dE"])

    mlp_b = MLPWorldModelBaseline(seed=seed)
    mlp_b.fit(log_b["X_t"], log_b["A_t"], log_b["X_next"], log_b["dE"])

    thresh_b = ThresholdAutoscalerBaseline()
    thresh_b.fit(log_b["X_t"], log_b["A_t"], log_b["dE"])

    pb_casp = agent.world_model.predict_step(te_b["X_t"], te_b["A_t"], eid=eid_b)["x_next"]
    pb_scr = scratch_b.world_model.predict_step(te_b["X_t"], te_b["A_t"], eid=eid_sb)["x_next"]
    pb_mlp, _ = mlp_b.predict_step(te_b["X_t"], te_b["A_t"])

    scores_b = {k: [] for k in ("caspian", "scratch", "mlp", "threshold")}
    for ei, s0 in enumerate(OVERLOAD_INITS[:3]):
        es = seed * 3000 + ei
        for name, fn in [
            ("caspian", lambda x, s, t: agent.recommend_action(x, eid=eid_b)[0]),
            ("scratch", lambda x, s, t: scratch_b.recommend_action(x, eid=eid_sb)[0]),
            ("mlp", lambda x, s, t: mlp_b.plan_action(x, horizon=3)),
            ("threshold", lambda x, s, t: thresh_b.select_action(x)),
        ]:
            scores_b[name].append(run_ops_episode(CLUSTER_B_MIGRATED, fn, s0, steps=20, seed=es)["total_score"])

    h2 = {
        "nrmse_caspian": float(np.sqrt(np.mean((pb_casp - te_b["X_next"]) ** 2)) / x_scale_b),
        "nrmse_scratch": float(np.sqrt(np.mean((pb_scr - te_b["X_next"]) ** 2)) / x_scale_b),
        "nrmse_mlp": float(np.sqrt(np.mean((pb_mlp - te_b["X_next"]) ** 2)) / x_scale_b),
        "score_caspian": float(np.mean(scores_b["caspian"])),
        "score_scratch": float(np.mean(scores_b["scratch"])),
        "score_mlp": float(np.mean(scores_b["mlp"])),
        "score_threshold": float(np.mean(scores_b["threshold"])),
    }

    # --- Trial 4b (P19-H5): Continual Retention & Zero-cluster_id Retrieval ---
    probe_a = CLUSTER_A_NOMINAL.collect_telemetry_log(n_steps=8, seed=seed * 100 + 20)
    probe_b = CLUSTER_B_MIGRATED.collect_telemetry_log(n_steps=8, seed=seed * 100 + 21)
    ra, _, _ = agent.retrieve_cluster_context(probe_a["X_t"], probe_a["A_t"], probe_a["X_next"], probe_a["dE"])
    rb, _, _ = agent.retrieve_cluster_context(probe_b["X_t"], probe_b["A_t"], probe_b["X_next"], probe_b["dE"])
    retr_acc = float((int(ra == eid_a) + int(rb == eid_b)) / 2.0)

    err_a_post = float(np.sqrt(np.mean((agent.world_model.predict_step(te_a_nom["X_t"], te_a_nom["A_t"], eid=ra)["de_mean"] - te_a_nom["dE"]) ** 2)) / de_scale_a)
    forgetting_a = max(0.0, err_a_post - err_a_pre)
    footprint = agent.footprint_audit()

    # --- Trial 3 (P19-H3): Mid-Stream Hardware/Workload Incident Detection & Autonomous Recovery ---
    # Clone clean agent before mutating Cluster_A engram for incident trial
    inc_agent = CaspianOpsAgent(seed=seed, horizon=3, beta_uncertainty=0.5, transfer_mode="uncertainty_guided")
    inc_agent.world_model = agent.world_model.from_bytes(agent.world_model.to_bytes())
    inc_agent.planner.world_model = inc_agent.world_model
    inc_agent.world_model.memory.active_engram_id = eid_a

    # Static (non-adapting) reference uses original agent on eid_a
    inc_log = CLUSTER_A_INCIDENT.collect_telemetry_log(n_steps=60, seed=seed * 100 + 30)
    pre_errs, post_errs = [], []
    bx, ba, bxn, bde = [], [], [], []
    escalated = False
    recovered = False
    for t_i in range(60):
        xt, at, xnt, det = inc_log["X_t"][t_i], int(inc_log["A_t"][t_i]), inc_log["X_next"][t_i], float(inc_log["dE"][t_i])
        m_info = inc_agent.monitor_tick(xt, at, det, eid=eid_a)
        ae = abs(det - m_info["de_hat"])
        bx.append(xt); ba.append(at); bxn.append(xnt); bde.append(det)
        if m_info["incident_escalated"]:
            escalated = True
        if not recovered:
            pre_errs.append(ae)
            if escalated and len(bx) >= 30:
                inc_agent.adapt_to_incident(np.array(bx), np.array(ba), np.array(bxn), np.array(bde))
                recovered = True
        else:
            post_errs.append(ae)

    pre_mae = float(np.mean(pre_errs[:10])) if pre_errs else 1.0
    post_mae = float(np.mean(post_errs[-10:])) if post_errs else pre_mae
    rec_red = float(1.0 - post_mae / max(pre_mae, 1e-6))

    # Post-incident control score: adapted inc_agent vs static non-adapting agent
    sc_adapted = float(np.mean([
        run_ops_episode(CLUSTER_A_INCIDENT, lambda x, s, t: inc_agent.recommend_action(x, eid=eid_a)[0], s0, steps=20, seed=seed * 4000 + i)["total_score"]
        for i, s0 in enumerate(OVERLOAD_INITS[:3])
    ]))
    sc_static = float(np.mean([
        run_ops_episode(CLUSTER_A_INCIDENT, lambda x, s, t: agent.recommend_action(x, eid=eid_a)[0], s0, steps=20, seed=seed * 4000 + i)["total_score"]
        for i, s0 in enumerate(OVERLOAD_INITS[:3])
    ]))

    h3 = {
        "incident_escalated": bool(escalated),
        "incident_recovered": bool(recovered),
        "pre_incident_mae": pre_mae,
        "post_recovery_mae": post_mae,
        "recovery_mae_reduction": rec_red,
        "post_incident_score_adapted": sc_adapted,
        "post_incident_score_static": sc_static,
    }
    h5 = {
        "forgetting_cluster_a": forgetting_a,
        "retrieval_acc": retr_acc,
        "raw_telemetry_stored_bytes": footprint["raw_observations_stored_bytes"],
        "footprint": footprint,
    }
    h6 = {
        "mean_latency_ms": round(agent.mean_latency_ms(), 4),
        "persistent_bytes": footprint["minimal_persistent_pickle_bytes"],
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }

    return {"seed": seed, "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6}


def aggregate_phase19(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    seeds = [r["seed"] for r in rows]

    def vec(sec: str, key: str) -> np.ndarray:
        return np.array([r[sec][key] for r in rows], dtype=float)

    # H1
    casp_a = vec("H1", "score_caspian")
    h1_comps = {b: paired_stats(casp_a, vec("H1", f"score_{b}"), higher_is_better=True, seed=1 + i) for i, b in enumerate(["threshold", "reactive", "mlp", "random"])}
    h1_holm = holm_bonferroni({k: v["perm_p"] for k, v in h1_comps.items()})
    h1_orac = float(np.mean(vec("H1", "oracle_ratio")))
    h1_met = bool(h1_orac >= 0.85 and all(v["ci_strictly_positive"] for v in h1_comps.values()) and all(p < 0.05 for p in h1_holm.values()))

    # H2
    h2_sc_mlp = paired_stats(vec("H2", "score_caspian"), vec("H2", "score_mlp"), higher_is_better=True, seed=10)
    h2_sc_thr = paired_stats(vec("H2", "score_caspian"), vec("H2", "score_threshold"), higher_is_better=True, seed=11)
    h2_sc_scr = paired_stats(vec("H2", "score_caspian"), vec("H2", "score_scratch"), higher_is_better=True, seed=12)
    h2_er_mlp = paired_stats(vec("H2", "nrmse_caspian"), vec("H2", "nrmse_mlp"), higher_is_better=False, seed=13)
    h2_er_scr = paired_stats(vec("H2", "nrmse_caspian"), vec("H2", "nrmse_scratch"), higher_is_better=False, seed=14)
    h2_holm = holm_bonferroni({
        "score_mlp": h2_sc_mlp["perm_p"], "score_threshold": h2_sc_thr["perm_p"], "score_scratch": h2_sc_scr["perm_p"],
        "nrmse_mlp": h2_er_mlp["perm_p"], "nrmse_scratch": h2_er_scr["perm_p"],
    })
    h2_met = bool(
        all(c["ci_strictly_positive"] for c in (h2_sc_mlp, h2_sc_thr, h2_sc_scr, h2_er_mlp, h2_er_scr))
        and all(p < 0.05 for p in h2_holm.values())
    )

    # H3
    h3_esc = float(np.mean([1.0 if r["H3"]["incident_escalated"] else 0.0 for r in rows]))
    h3_red = float(np.mean(vec("H3", "recovery_mae_reduction")))
    h3_vs_stat = paired_stats(vec("H3", "post_incident_score_adapted"), vec("H3", "post_incident_score_static"), higher_is_better=True, seed=20)
    h3_met = bool(h3_esc >= 0.90 and h3_red >= 0.50 and h3_vs_stat["ci_strictly_positive"] and h3_vs_stat["perm_p"] < 0.05)

    # H4
    h4_auroc = float(np.mean(vec("H4", "ood_auroc")))
    h4_fpr = float(np.mean(vec("H4", "nominal_false_escalation_rate")))
    h4_met = bool(h4_auroc >= 0.95 and h4_fpr < 0.05)

    # H5
    h5_forg = float(np.mean(vec("H5", "forgetting_cluster_a")))
    h5_retr = float(np.mean(vec("H5", "retrieval_acc")))
    h5_raw = int(np.max(vec("H5", "raw_telemetry_stored_bytes")))
    h5_met = bool(h5_forg < 1e-6 and h5_retr >= 0.95 and h5_raw == 0)

    # H6
    det_r = run_single_seed_19(seeds[0])
    det_diff = abs(det_r["H1"]["score_caspian"] - rows[0]["H1"]["score_caspian"])
    ast_aud = ast_leakage_audit_phase19()
    lat_ms = float(np.mean(vec("H6", "mean_latency_ms")))
    pers_b = int(np.max(vec("H6", "persistent_bytes")))
    h6_met = bool(lat_ms < 15.0 and pers_b < 50000 and det_diff == 0.0 and ast_aud["passed"])

    verdicts = {"P19-H1": h1_met, "P19-H2": h2_met, "P19-H3": h3_met, "P19-H4": h4_met, "P19-H5": h5_met, "P19-H6": h6_met}
    return {
        "experiment_id": "CSP-P19-E001",
        "split": label,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "hypotheses_met_count": f"{sum(verdicts.values())}/6",
        "verdicts": verdicts,
        "H1": {"met": h1_met, "means": {k: round(float(np.mean(vec("H1", k))), 4) for k in rows[0]["H1"]}, "comparisons": h1_comps, "holm_p": h1_holm},
        "H2": {"met": h2_met, "means": {k: round(float(np.mean(vec("H2", k))), 5) for k in rows[0]["H2"]}, "score_vs_mlp": h2_sc_mlp, "score_vs_scratch": h2_sc_scr, "score_vs_threshold": h2_sc_thr, "nrmse_vs_mlp": h2_er_mlp, "nrmse_vs_scratch": h2_er_scr, "holm_p": h2_holm},
        "H3": {"met": h3_met, "escalation_rate": round(h3_esc, 4), "mean_recovery_reduction": round(h3_red, 5), "mean_pre_mae": round(float(np.mean(vec("H3", "pre_incident_mae"))), 4), "mean_post_mae": round(float(np.mean(vec("H3", "post_recovery_mae"))), 4), "mean_score_adapted": round(float(np.mean(vec("H3", "post_incident_score_adapted"))), 2), "mean_score_static": round(float(np.mean(vec("H3", "post_incident_score_static"))), 2), "vs_static": h3_vs_stat},
        "H4": {"met": h4_met, "mean_ood_auroc": round(h4_auroc, 5), "mean_false_escalation_rate": round(h4_fpr, 5), "mean_unseen_escalation_recall": round(float(np.mean(vec("H4", "unseen_escalation_recall"))), 5)},
        "H5": {"met": h5_met, "mean_forgetting": round(h5_forg, 6), "mean_retrieval_acc": round(h5_retr, 5), "raw_telemetry_stored_bytes": h5_raw, "footprint": rows[0]["H5"]["footprint"]},
        "H6": {"met": h6_met, "mean_latency_ms": round(lat_ms, 4), "persistent_bytes": pers_b, "determinism_diff": float(det_diff), "ast_passed": ast_aud["passed"], "ast_violations": ast_aud["violations"]},
        "per_seed": rows,
    }


def run_phase19(seeds: List[int], label: str, out_path: str) -> Dict[str, Any]:
    rows = [run_single_seed_19(s) for s in seeds]
    summary = aggregate_phase19(rows, label=label)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        s = run_phase19(SMOKE_SEEDS_19, "exploratory_smoke", "logs/phase19/p19_smoke_results.json")
    elif mode == "confirmatory":
        s = run_phase19(CONFIRMATORY_SEEDS_19, "confirmatory", "logs/phase19/p19_confirmatory_results.json")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(f"=== Phase 19 ({s['split']}, n={s['n_seeds']}) ===")
    print("Hypotheses met:", s["hypotheses_met_count"], s["verdicts"])
    print("H1:", s["H1"]["means"]["score_caspian"], "oracle:", s["H1"]["means"]["score_oracle"], "ratio:", s["H1"]["means"]["oracle_ratio"], "mlp:", s["H1"]["means"]["score_mlp"], "thresh:", s["H1"]["means"]["score_threshold"])
    print("H2:", s["H2"]["means"])
    print("H3:", s["H3"]["escalation_rate"], s["H3"]["mean_recovery_reduction"], "adapted:", s["H3"]["mean_score_adapted"], "vs static:", s["H3"]["mean_score_static"])
    print("H4:", s["H4"])
    print("H5:", s["H5"]["mean_forgetting"], s["H5"]["mean_retrieval_acc"], "raw_bytes:", s["H5"]["raw_telemetry_stored_bytes"])
    print("H6:", s["H6"])
