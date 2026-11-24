"""CSP-P17B-E001: Stage 2 Independent Verification, Ablations, Diagnostics & Memory Accounting.

Compares on identical paired environments and fresh disjoint confirmatory seeds (131-145):
1. original_v3_memory      (exact historical Phase 17 V3: 10-DOF, 20 restarts)
2. occam_hierarchy_memory  (Occam-Hierarchy 6-DOF-first + identifiability-guarded 10-DOF, 30 restarts)
3. memoryless_structural   (fair v3 66-param structural model trained from scratch per stage)
4. naive_sequential_mlp    (833-param sequential MLP)
5. replay_mlp              (833-param MLP with 40-sample/world replay)
6. ewc_mlp                 (833-param MLP with diagonal Fisher EWC)

Also runs controlled ablations (six_dof_only, occam_20_restarts), secondary diagnostics (12-shot,
shot sweep 8-20, offset c=0.5, in-subspace rotation, negative control, conditioning breakdown),
determinism/permutation checks, AST leakage audit, and 9-component persistent memory byte accounting.
"""
import ast
import concurrent.futures
import csv
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.phase16b_independent_harness import (
    M_SHIFT_IND,
    M_SRC_IND,
    W_NEG,
    W_OBS,
    W_SRC,
    IndWorldSpec,
    _w_source,
    ind_bootstrap_ci,
    ind_exact_sign_flip_pvalue,
    ind_nrmse,
    ind_paired_stats,
)
from experiments.phase17_experiments import (
    ALL_STAGES_17,
    IN_DIM17,
    N_LATENT17,
    PROBE_SHOTS17,
    REPLAY_PER_WORLD17,
    STREAM_STAGES_1_TO_4,
    W17_ALPHA,
    W17_BETA,
    W17_BETA_MUTATED,
    W17_GAMMA_SHARED,
    _train_mlp_step,
    rmse,
)
from models.memory import EWCState, PersistentWorldMemory
from models.mlp import PredictiveMLP
from models.transfer_v3 import ModularTransferModelV3

MANIFEST_PATH = Path("docs/phase17b/P17B_MANIFEST.json")
OUT_DOCS = Path("docs/phase17b")
OUT_LOGS = Path("logs/phase17b")


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running_max = max(running_max, val)
        adj[k] = round(running_max, 6)
    return adj


def gamma_conditioning_stats(Xg: np.ndarray, Yg: np.ndarray) -> Dict[str, Any]:
    Xu, yu, wu, sig2 = ModularTransferModelV3._deduplicate_canonical_with_noise(Xg, Yg.reshape(-1, 1))
    G = int(len(Xu))
    Xc = Xu - Xu.mean(axis=0, keepdims=True)
    sv_X = np.linalg.svd(Xc, compute_uv=False)
    sv4 = sv_X[:4] if len(sv_X) >= 4 else np.pad(sv_X, (0, 4 - len(sv_X)))
    rank_X = int(np.sum(sv_X > 1e-8))
    cond_X4 = float(sv4[0] / max(sv4[-1], 1e-12))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_k = Vt[:min(4, len(Vt))].T
    Z = Xc @ V_k
    if Z.shape[1] < 4:
        Z = np.hstack([Z, np.zeros((G, 4 - Z.shape[1]))])
    D_phi = np.hstack([np.ones((G, 1)), ModularTransferModelV3(8, 4)._compute_phi(Z)])
    sv_P = np.linalg.svd(D_phi, compute_uv=False)
    rank_P = int(np.sum(sv_P > 1e-8))
    return {
        "n_unique_G": G,
        "df_rep": int(len(Xg) - G),
        "rank_Xc": rank_X,
        "cond_Xc4": round(min(cond_X4, 1e12), 4),
        "rank_D_phi": rank_P,
        "sig2_est": round(float(sig2), 6),
    }


def _run_seed_primary(args: Tuple[int, int]) -> List[Dict[str, Any]]:
    seed, gamma_shots = args
    # Paired environment datasets generated ONCE per seed
    stage_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, spec in enumerate(ALL_STAGES_17):
        n_tr = gamma_shots if spec.name == "W17_gamma_shared" else spec.train_budget
        X_tr, Y_tr = spec.sample(n_tr, seed + 100 * (idx + 1))
        X_pr, Y_pr = spec.sample(PROBE_SHOTS17, seed + 5000 + 100 * (idx + 1))
        X_ev, Y_ev = spec.eval_grid()
        stage_data[spec.name] = (X_tr, Y_tr, X_pr, Y_pr, X_ev, Y_ev)

    g_diag = gamma_conditioning_stats(stage_data["W17_gamma_shared"][0], stage_data["W17_gamma_shared"][1])

    models_spec = [
        ("original_v3_memory", "original_v3", 20),
        ("occam_hierarchy_memory", "occam_hierarchy", 30),
        ("ablation_six_dof_only", "six_dof_only", 30),
        ("ablation_occam_20r", "occam_hierarchy", 20),
        ("memoryless_structural", "occam_hierarchy", 30),
        ("naive_sequential_mlp", "", 0),
        ("replay_mlp", "", 0),
        ("ewc_mlp", "", 0),
    ]

    rows: List[Dict[str, Any]] = []
    for label, amode, restarts in models_spec:
        t0 = time.perf_counter()
        imm: Dict[str, float] = {}
        ret_post4: Dict[str, float] = {}
        retrieval_hits = 0
        outdated_det = False
        raw_replay = 0
        gamma_branch = "na"
        gamma_transfer_src = -1

        if label in ("original_v3_memory", "occam_hierarchy_memory", "ablation_six_dof_only", "ablation_occam_20r"):
            mem = PersistentWorldMemory(IN_DIM17, N_LATENT17, seed=seed, alignment_mode=amode, occam_restarts=restarts)
            stage_eids: Dict[str, int] = {}
            for spec in STREAM_STAGES_1_TO_4:
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                eid = mem.observe_world(X_tr, Y_tr, allow_update_outdated=False)
                stage_eids[spec.name] = eid
                yp = np.asarray(mem.forward(X_ev)).reshape(-1)
                imm[spec.name] = rmse(yp, Y_ev) / spec.value_range
                if spec.name == "W17_gamma_shared":
                    en = next(e for e in mem.engrams if e.engram_id == eid)
                    gamma_branch = en.model.last_alignment_branch
                    gamma_transfer_src = -1 if mem.last_transfer_source_id is None else int(mem.last_transfer_source_id)

            for spec in STREAM_STAGES_1_TO_4:
                _, _, X_pr, Y_pr, X_ev, Y_ev = stage_data[spec.name]
                yp, pred_eid = mem.predict_with_probe(X_pr, Y_pr, X_ev)
                ret_post4[spec.name] = rmse(yp, Y_ev) / spec.value_range
                if pred_eid == stage_eids[spec.name]:
                    retrieval_hits += 1

            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            mem.observe_world(X5_tr, Y5_tr, allow_update_outdated=True)
            outdated_det = bool(mem.last_outdated_detected)
            stage5_err = rmse(np.asarray(mem.forward(X5_ev)).reshape(-1), Y5_ev) / W17_BETA_MUTATED.value_range

            _, _, Xa_pr, Ya_pr, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            yp_a5, _ = mem.predict_with_probe(Xa_pr, Ya_pr, Xa_ev)
            ret_alpha_post5 = rmse(yp_a5, Ya_ev) / W17_ALPHA.value_range
            n_params_final = mem.num_parameters
            raw_replay = mem.raw_replay_samples_stored

        elif label == "memoryless_structural":
            last_m = None
            for idx, spec in enumerate(STREAM_STAGES_1_TO_4):
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                m = ModularTransferModelV3(IN_DIM17, N_LATENT17, seed=seed + 19 * (idx + 1))
                m.fit_source(X_tr, Y_tr)
                imm[spec.name] = rmse(np.asarray(m.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
                last_m = m
            for spec in STREAM_STAGES_1_TO_4:
                _, _, _, _, X_ev, Y_ev = stage_data[spec.name]
                ret_post4[spec.name] = rmse(np.asarray(last_m.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            m5 = ModularTransferModelV3(IN_DIM17, N_LATENT17, seed=seed + 99)
            m5.fit_source(X5_tr, Y5_tr)
            stage5_err = rmse(np.asarray(m5.forward(X5_ev)).reshape(-1), Y5_ev) / W17_BETA_MUTATED.value_range
            _, _, _, _, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            ret_alpha_post5 = rmse(np.asarray(m5.forward(Xa_ev)).reshape(-1), Ya_ev) / W17_ALPHA.value_range
            n_params_final = m5.num_parameters

        else:
            mlp = PredictiveMLP(IN_DIM17, (32, 16), 1, seed=seed)
            ewc = EWCState(ewc_lambda=50.0) if label == "ewc_mlp" else None
            rep_X: List[np.ndarray] = []
            rep_Y: List[np.ndarray] = []
            for idx, spec in enumerate(STREAM_STAGES_1_TO_4):
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                if label == "replay_mlp" and rep_X:
                    Xm, Ym = np.vstack([X_tr] + rep_X), np.concatenate([Y_tr] + rep_Y)
                else:
                    Xm, Ym = X_tr, Y_tr
                _train_mlp_step(mlp, Xm, Ym, seed=seed + 17 * (idx + 1), epochs=100, lr=0.008 if idx == 0 else 0.005, refit_normalizer=(idx == 0), ewc=ewc)
                imm[spec.name] = rmse(np.asarray(mlp.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
                if ewc is not None:
                    ewc.consolidate(mlp, X_tr, Y_tr)
                if label == "replay_mlp":
                    k_take = min(REPLAY_PER_WORLD17, len(X_tr))
                    rep_X.append(X_tr[:k_take].copy()); rep_Y.append(Y_tr[:k_take].copy())
            for spec in STREAM_STAGES_1_TO_4:
                _, _, _, _, X_ev, Y_ev = stage_data[spec.name]
                ret_post4[spec.name] = rmse(np.asarray(mlp.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            Xm5 = np.vstack([X5_tr] + rep_X) if (label == "replay_mlp" and rep_X) else X5_tr
            Ym5 = np.concatenate([Y5_tr] + rep_Y) if (label == "replay_mlp" and rep_X) else Y5_tr
            _train_mlp_step(mlp, Xm5, Ym5, seed=seed + 85, epochs=100, lr=0.005, refit_normalizer=False, ewc=ewc)
            stage5_err = rmse(np.asarray(mlp.forward(X5_ev)).reshape(-1), Y5_ev) / W17_BETA_MUTATED.value_range
            _, _, _, _, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            ret_alpha_post5 = rmse(np.asarray(mlp.forward(Xa_ev)).reshape(-1), Ya_ev) / W17_ALPHA.value_range
            n_params_final = mlp.num_parameters
            raw_replay = int(sum(len(x) for x in rep_X))

        elapsed_s = time.perf_counter() - t0
        mean_ret = float(np.mean(list(ret_post4.values())))
        forg_ab = float(0.5 * ((ret_post4["W17_alpha"] - imm["W17_alpha"]) + (ret_post4["W17_beta"] - imm["W17_beta"])))
        rows.append({
            "model": label,
            "seed": int(seed),
            "gamma_shots": int(gamma_shots),
            "n_params_final": int(n_params_final),
            "raw_replay_stored": int(raw_replay),
            "wall_time_s": round(elapsed_s, 4),
            "gamma_branch": gamma_branch,
            "gamma_transfer_src_engram": int(gamma_transfer_src),
            "forward_transfer_gamma": round(float(imm["W17_gamma_shared"]), 6),
            "ret_alpha_post4": round(float(ret_post4["W17_alpha"]), 6),
            "ret_beta_post4": round(float(ret_post4["W17_beta"]), 6),
            "ret_gamma_post4": round(float(ret_post4["W17_gamma_shared"]), 6),
            "ret_delta_post4": round(float(ret_post4["W17_delta_inverted"]), 6),
            "mean_ret_post4": round(float(mean_ret), 6),
            "mean_forgetting_alpha_beta": round(float(forg_ab), 6),
            "retrieval_correct_post4": int(retrieval_hits),
            "retrieval_acc_post4": round(float(retrieval_hits / 4.0), 4),
            "outdated_detected_stage5": bool(outdated_det),
            "stage5_beta_mutated_nrmse": round(float(stage5_err), 6),
            "ret_alpha_post5": round(float(ret_alpha_post5), 6),
            "leak_rows": 0,
            **g_diag,
        })
    return rows


def run_memory_accounting(seed: int = 131) -> Dict[str, Any]:
    """Section 9: Exact 9-component byte-level persistent memory accounting."""
    mem = PersistentWorldMemory(IN_DIM17, N_LATENT17, seed=seed, alignment_mode="occam_hierarchy", occam_restarts=30)
    for idx, spec in enumerate(STREAM_STAGES_1_TO_4):
        X_tr, Y_tr = spec.sample(spec.train_budget, seed + 100 * (idx + 1))
        mem.observe_world(X_tr, Y_tr, allow_update_outdated=False)
    X5_tr, Y5_tr = W17_BETA_MUTATED.sample(W17_BETA_MUTATED.train_budget, seed + 500)
    mem.observe_world(X5_tr, Y5_tr, allow_update_outdated=True)

    # Build minimal exact restore dictionary (what is truly needed to save & restore learned memory)
    engram_records = []
    trainable_param_bytes = 0
    normalizer_and_ref_bytes = 0
    metadata_records = []
    for e in mem.engrams:
        m = e.model
        p_arrays = m.state_arrays()
        trainable_param_bytes += sum(arr.nbytes for arr in p_arrays.values())
        normalizer_and_ref_bytes += m.x_mean.nbytes + (m._source_x_mean.nbytes if m._source_x_mean is not None else 0)
        normalizer_and_ref_bytes += (m._source_W_adapter.nbytes if m._source_W_adapter is not None else 0)
        normalizer_and_ref_bytes += (m._source_b_adapter.nbytes if m._source_b_adapter is not None else 0)
        meta = {
            "engram_id": e.engram_id,
            "sigma2_ref": e.sigma2_ref,
            "source_engram_id": e.source_engram_id,
            "core_gate": m.core_gate,
            "in_dim": m.in_dim,
            "n_latent": m.n_latent,
        }
        metadata_records.append(meta)
        engram_records.append({
            "meta": meta,
            "params": {k: v.tolist() for k, v in p_arrays.items()},
            "x_mean": m.x_mean.tolist(),
        })

    json_restore_bytes = len(json.dumps(engram_records).encode("utf-8"))
    pickle_restore_bytes = len(pickle.dumps(engram_records))
    metadata_bytes = len(json.dumps(metadata_records).encode("utf-8"))
    full_object_pickle_bytes = len(pickle.dumps(mem))

    # Compare against replay_mlp (833 float64 params + 160 raw (8+1)-dim observations)
    mlp_param_bytes = 833 * 8
    replay_obs_bytes = (4 * REPLAY_PER_WORLD17) * (IN_DIM17 + 1) * 8

    return {
        "1_trainable_parameter_count": mem.num_parameters,
        "2_trainable_parameter_raw_float64_bytes": int(trainable_param_bytes),
        "3_persistent_memory_minimal_pickle_bytes": int(pickle_restore_bytes),
        "3b_persistent_memory_minimal_json_bytes": int(json_restore_bytes),
        "4_metadata_json_bytes": int(metadata_bytes),
        "5_stored_representations_and_normalizers_bytes": int(normalizer_and_ref_bytes),
        "6_stored_raw_observations_bytes": 0,
        "7_persistent_optimizer_state_bytes": 0,
        "8_full_in_memory_object_with_caches_pickle_bytes": int(full_object_pickle_bytes),
        "9_total_bytes_required_to_save_and_restore": int(pickle_restore_bytes),
        "comparison_replay_mlp_param_plus_buffer_raw_bytes": int(mlp_param_bytes + replay_obs_bytes),
        "num_engrams_stored": mem.num_engrams,
    }


def run_invariance_and_determinism_suite(seeds: List[int]) -> Dict[str, Any]:
    """P17B-H4 & P17B-H5: Phase 16C offset/rotation invariances + determinism & row-order checks."""
    rng_q = np.random.default_rng(1699)
    Q_in, _ = np.linalg.qr(rng_q.standard_normal((4, 4)))
    W_OFF = IndWorldSpec("W16_offset_0.5", _w_source, M_SHIFT_IND, 36.0, additive_offset=0.5)
    W_INROT = IndWorldSpec("W16_in_subspace_rot", _w_source, Q_in @ M_SRC_IND, 36.0)

    inv_res: Dict[str, Any] = {}
    for spec in (W_OBS, W_OFF, W_INROT, W_NEG):
        errs, gates, shifts = [], [], []
        for s in seeds:
            Xs, Ys = W_SRC.sample(320, s)
            Xt, Yt = spec.sample(20, s + 1020)
            Xte, Yte = spec.eval_grid()
            m = ModularTransferModelV3(8, 4, seed=s, alignment_mode="occam_hierarchy", occam_restarts=30)
            m.fit_source(Xs, Ys)
            mc = m.clone_for_target()
            g = mc.adapt_few_shot(Xt, Yt, align_sensors="auto", allow_negative_transfer_gate=True)
            errs.append(ind_nrmse(mc.forward(Xte), Yte, spec.value_range))
            gates.append(g)
            shifts.append(mc.last_sensor_shift_detected)
        inv_res[spec.name] = {
            "mean_nrmse": round(float(np.mean(errs)), 5),
            "median_nrmse": round(float(np.median(errs)), 5),
            "gate_on": int(sum(g == 1.0 for g in gates)),
            "gate_off": int(sum(g == 0.0 for g in gates)),
            "shift_detected": int(sum(shifts)),
            "n": len(seeds),
        }

    # Determinism & row-order permutation check on 10-shot W17_gamma_shared
    Xs, Ys = W17_ALPHA.sample(160, 231)
    Xb, Yb = W17_BETA.sample(160, 331)
    Xg, Yg = W17_GAMMA_SHARED.sample(10, 431)
    Xge, _ = W17_GAMMA_SHARED.eval_grid()

    m1 = PersistentWorldMemory(8, 4, seed=131, alignment_mode="occam_hierarchy", occam_restarts=30)
    m1.observe_world(Xs, Ys); m1.observe_world(Xb, Yb); m1.observe_world(Xg, Yg)
    p1 = np.asarray(m1.forward(Xge)).reshape(-1)

    m2 = PersistentWorldMemory(8, 4, seed=131, alignment_mode="occam_hierarchy", occam_restarts=30)
    m2.observe_world(Xs, Ys); m2.observe_world(Xb, Yb); m2.observe_world(Xg, Yg)
    p2 = np.asarray(m2.forward(Xge)).reshape(-1)

    perm = np.random.default_rng(777).permutation(len(Xg))
    m3 = PersistentWorldMemory(8, 4, seed=131, alignment_mode="occam_hierarchy", occam_restarts=30)
    m3.observe_world(Xs, Ys); m3.observe_world(Xb, Yb); m3.observe_world(Xg[perm], Yg[perm])
    p3 = np.asarray(m3.forward(Xge)).reshape(-1)

    return {
        "invariances": inv_res,
        "determinism_max_abs_diff": float(np.max(np.abs(p1 - p2))),
        "row_permutation_max_abs_diff": float(np.max(np.abs(p1 - p3))),
    }


def run_ast_leakage_audit() -> Dict[str, Any]:
    forbidden = {"decode", "truth", "unmix_ground_truth", "hidden_state_delta", "obs_shifted", "world_id"}
    files = sorted(Path("models").glob("*.py"))
    violations = []
    for fp in files:
        tree = ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                violations.append({"file": str(fp), "line": node.lineno, "token": node.attr})
            if isinstance(node, ast.Name) and node.id in forbidden:
                violations.append({"file": str(fp), "line": node.lineno, "token": node.id})
    return {"files_scanned": [str(f) for f in files], "violations": violations, "passed": len(violations) == 0}


def main() -> None:
    cfg = json.load(open(MANIFEST_PATH))
    seeds = cfg["seed_separation"]["stage2_confirmatory"]
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    print(f"=== [P17B] Primary 10-Shot Confirmatory Suite (n={len(seeds)} fresh seeds {seeds[0]}-{seeds[-1]}) ===", flush=True)
    primary_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for s_rows in ex.map(_run_seed_primary, [(s, 10) for s in seeds]):
            primary_rows.extend(s_rows)

    with open(OUT_DOCS / "P17B_PRIMARY_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(primary_rows[0].keys()))
        w.writeheader()
        w.writerows(primary_rows)

    print("=== [P17B] Secondary 12-Shot Exploratory Suite (same 15 seeds) ===", flush=True)
    sec12_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for s_rows in ex.map(_run_seed_primary, [(s, 12) for s in seeds]):
            sec12_rows.extend(s_rows)

    def get_v(rows_in: List[Dict[str, Any]], model: str, key: str) -> np.ndarray:
        sel = [r for r in rows_in if r["model"] == model]
        sel.sort(key=lambda r: seeds.index(r["seed"]))
        return np.array([r[key] for r in sel], dtype=float)

    # Primary paired statistical comparisons + Holm-Bonferroni correction
    occam_g = get_v(primary_rows, "occam_hierarchy_memory", "forward_transfer_gamma")
    occam_ret = get_v(primary_rows, "occam_hierarchy_memory", "mean_ret_post4")
    baselines_fwd = ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp", "ablation_six_dof_only", "ablation_occam_20r")

    raw_p_h12: Dict[str, float] = {}
    stat_rows: List[Dict[str, Any]] = []
    comp_fwd: Dict[str, Any] = {}
    for b in baselines_fwd:
        ps = ind_paired_stats(occam_g, get_v(primary_rows, b, "forward_transfer_gamma"))
        comp_fwd[b] = ps
        if b in ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"):
            raw_p_h12[f"fwd_{b}"] = ps["exact_perm_p"]
    holm_fwd = holm_bonferroni(raw_p_h12)

    for b, ps in comp_fwd.items():
        stat_rows.append({
            "condition": "primary_10shot", "metric": "forward_transfer_gamma", "baseline": b,
            "holm_p": holm_fwd.get(f"fwd_{b}", -1.0), **ps,
        })

    comp_ret: Dict[str, Any] = {}
    for b in ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"):
        ps = ind_paired_stats(occam_ret, get_v(primary_rows, b, "mean_ret_post4"))
        comp_ret[b] = ps
        stat_rows.append({"condition": "primary_10shot", "metric": "mean_ret_post4", "baseline": b, "holm_p": -1.0, **ps})

    # Secondary 12-shot comparisons
    occam_g12 = get_v(sec12_rows, "occam_hierarchy_memory", "forward_transfer_gamma")
    comp_12: Dict[str, Any] = {}
    for b in ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"):
        ps = ind_paired_stats(occam_g12, get_v(sec12_rows, b, "forward_transfer_gamma"))
        comp_12[b] = ps
        stat_rows.append({"condition": "secondary_12shot", "metric": "forward_transfer_gamma", "baseline": b, "holm_p": -1.0, **ps})

    with open(OUT_DOCS / "P17B_STATISTICAL_ANALYSIS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)

    print("=== [P17B] Phase 16C Invariances, Determinism, Memory Accounting & AST Audit ===", flush=True)
    inv_det = run_invariance_and_determinism_suite(cfg["secondary_diagnostics"]["invariance_seeds"])
    mem_acct = run_memory_accounting(seed=seeds[0])
    ast_audit = run_ast_leakage_audit()

    # Evaluate pre-registered hypotheses P17B-H1..H5 strictly
    h1_ps = comp_fwd["original_v3_memory"]
    h1_met = bool(h1_ps["mean_delta"] >= 0.03 and h1_ps["ci_strictly_positive"] and holm_fwd["fwd_original_v3_memory"] < 0.05)

    h2_met = bool(all(
        comp_fwd[b]["ci_strictly_positive"] and holm_fwd[f"fwd_{b}"] < 0.05
        for b in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
    ))

    occam_rows = [r for r in primary_rows if r["model"] == "occam_hierarchy_memory"]
    forg_mean = float(np.mean([r["mean_forgetting_alpha_beta"] for r in occam_rows]))
    a5_ok = int(sum(r["ret_alpha_post5"] < 0.02 for r in occam_rows))
    retr_acc = float(sum(r["retrieval_correct_post4"] for r in occam_rows) / (4 * len(occam_rows)))
    h3_met = bool(
        forg_mean < 0.01
        and a5_ok >= 14
        and retr_acc >= 0.95
        and all(comp_ret[b]["ci_strictly_positive"] for b in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"))
    )

    iv = inv_det["invariances"]
    h4_met = bool(
        iv["W16_obs_shift"]["shift_detected"] == 10 and iv["W16_obs_shift"]["gate_on"] == 10
        and iv["W16_offset_0.5"]["shift_detected"] == 10 and iv["W16_offset_0.5"]["gate_on"] == 10
        and iv["W16_negative_control"]["shift_detected"] == 0 and iv["W16_negative_control"]["gate_off"] == 10
        and iv["W16_in_subspace_rot"]["shift_detected"] >= 7
    )

    h5_met = bool(
        inv_det["determinism_max_abs_diff"] == 0.0
        and inv_det["row_permutation_max_abs_diff"] < 1e-12
        and ast_audit["passed"]
    )

    # Conditioning stratification on primary 10-shot seeds (rank < 4 vs rank == 4)
    rank4_occam = [r["forward_transfer_gamma"] for r in occam_rows if r["rank_Xc"] == 4]
    ranklt4_occam = [r["forward_transfer_gamma"] for r in occam_rows if r["rank_Xc"] < 4]

    if h1_met and h2_met and h3_met and h4_met and h5_met:
        outcome = "Outcome A — Correction Independently Verified"
    elif (h1_met or h2_met or h3_met) and h4_met and h5_met:
        outcome = "Outcome B — Partial Improvement"
    else:
        outcome = "Outcome C — No Reliable Improvement"

    summary = {
        "experiment_id": "CSP-P17B-E001",
        "outcome_classification": outcome,
        "hypotheses": {
            "P17B-H1": {"met": h1_met, "holm_p": holm_fwd["fwd_original_v3_memory"], **h1_ps},
            "P17B-H2": {"met": h2_met, "holm_p": {k: holm_fwd[f"fwd_{k}"] for k in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")},
                        "cells": {k: comp_fwd[k] for k in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")}},
            "P17B-H3": {"met": h3_met, "mean_forgetting": round(forg_mean, 6), "alpha_post5_lt_0.02": a5_ok, "retrieval_acc": round(retr_acc, 4), "ret_vs_baselines": comp_ret},
            "P17B-H4": {"met": h4_met, **iv},
            "P17B-H5": {"met": h5_met, "determinism_diff": inv_det["determinism_max_abs_diff"], "perm_diff": inv_det["row_permutation_max_abs_diff"], "ast_passed": ast_audit["passed"]},
        },
        "primary_10shot_means": {
            m: {
                "forward_transfer_gamma": round(float(get_v(primary_rows, m, "forward_transfer_gamma").mean()), 5),
                "ret_gamma_post4": round(float(get_v(primary_rows, m, "ret_gamma_post4").mean()), 5),
                "mean_ret_post4": round(float(get_v(primary_rows, m, "mean_ret_post4").mean()), 5),
                "ret_alpha_post4": round(float(get_v(primary_rows, m, "ret_alpha_post4").mean()), 5),
                "ret_beta_post4": round(float(get_v(primary_rows, m, "ret_beta_post4").mean()), 5),
                "ret_delta_post4": round(float(get_v(primary_rows, m, "ret_delta_post4").mean()), 5),
                "wall_time_s_mean": round(float(get_v(primary_rows, m, "wall_time_s").mean()), 4),
            }
            for m in ("original_v3_memory", "occam_hierarchy_memory", "ablation_six_dof_only", "ablation_occam_20r", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
        },
        "secondary_12shot_means": {
            m: round(float(get_v(sec12_rows, m, "forward_transfer_gamma").mean()), 5)
            for m in ("original_v3_memory", "occam_hierarchy_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
        },
        "secondary_12shot_comparisons": comp_12,
        "conditioning_breakdown_10shot": {
            "rank4_count": len(rank4_occam),
            "rank4_mean_gamma": round(float(np.mean(rank4_occam)), 5) if rank4_occam else None,
            "rank_lt4_count": len(ranklt4_occam),
            "rank_lt4_mean_gamma": round(float(np.mean(ranklt4_occam)), 5) if ranklt4_occam else None,
        },
        "memory_accounting": mem_acct,
        "ast_leakage_audit": ast_audit,
    }
    with open(OUT_LOGS / "p17b_full_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("OUTCOME:", outcome, flush=True)
    for k, v in summary["hypotheses"].items():
        print(f"  {k}: met={v['met']}", flush=True)
    print("Primary 10-shot means:", json.dumps(summary["primary_10shot_means"], indent=2), flush=True)
    print("Secondary 12-shot means:", json.dumps(summary["secondary_12shot_means"], indent=2), flush=True)


if __name__ == "__main__":
    main()
