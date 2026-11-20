"""Phase 16C (CSP-P16C-E001): Independent Verification of ModularTransferModelV3.

Verifies the four first-principles fixes for Phase 16B on:
1. Phase 16B diagnostic seeds (101–110) — direct before (v2) vs after (v3) comparison.
2. Fresh unseen independent seeds (121–130) — out-of-sample confirmation of v3 across all 4 worlds,
   Gate C, Gate D, and all Experiment F boundary stress tests (c=0.5 offset, in-subspace rotation,
   sample ordering permutation invariance).
"""
import concurrent.futures
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.phase16b_independent_harness import (
    M_SHIFT_IND,
    M_SRC_IND,
    TARGET_WORLDS_IND,
    W_NEG,
    W_OBS,
    W_RS,
    W_SPN,
    W_SRC,
    WORLD_MAP_IND,
    IndWorldSpec,
    _w_source,
    ind_nrmse,
    ind_paired_stats,
    ind_train_mlp,
    summarize_distribution,
)
from models.mlp import PredictiveMLP
from models.transfer_v3 import ModularTransferModelV3

SEEDS16B = list(range(101, 111))
SEEDS16C_FRESH = list(range(121, 131))


def _worker_v3_seed(args: Tuple[str, int]) -> Dict[str, Any]:
    wname, seed = args
    spec = WORLD_MAP_IND[wname]

    Xs, Ys = W_SRC.sample(320, seed)
    Xse, Yse = W_SRC.eval_grid()
    src_v3 = ModularTransferModelV3(8, 4, seed=seed)
    src_v3.fit_source(Xs, Ys)

    src_mlp = PredictiveMLP(8, (32, 16), 1, seed=seed)
    ind_train_mlp(src_mlp, Xs, Ys, seed=seed, epochs=120, lr=0.01, batch_size=32, refit_normalizer=True)

    Xt, Yt = spec.sample(20, seed + 1020)
    Xte, Yte = spec.eval_grid()

    # v3 modular_transfer (autonomous)
    m = src_v3.clone_for_target()
    gate = float(m.adapt_few_shot(Xt, Yt, align_sensors="auto", allow_negative_transfer_gate=True))
    mt_err = ind_nrmse(m.forward(Xte), Yte, spec.value_range)
    mt_ret = ind_nrmse(m.predict_source(Xse), Yse, W_SRC.value_range)
    ortho_err = float(np.linalg.norm(m.W_adapter.T @ m.W_adapter - np.eye(4), ord="fro"))

    # direct_transfer
    md = src_v3.clone_for_target()
    dt_err = ind_nrmse(md.forward(Xte), Yte, spec.value_range)

    # scratch_mlp & finetune_mlp
    ms = PredictiveMLP(8, (32, 16), 1, seed=seed + 13)
    ind_train_mlp(ms, Xt, Yt, seed=seed + 13, epochs=120, lr=0.01, batch_size=16, refit_normalizer=True)
    sm_err = ind_nrmse(ms.forward(Xte), Yte, spec.value_range)

    mf = copy.deepcopy(src_mlp)
    ind_train_mlp(mf, Xt, Yt, seed=seed + 7, epochs=80, lr=0.005, batch_size=16, refit_normalizer=False)
    ft_err = ind_nrmse(mf.forward(Xte), Yte, spec.value_range)
    ft_ret = ind_nrmse(mf.forward(Xse), Yse, W_SRC.value_range)

    # fair v3 scratch_structural
    ss = ModularTransferModelV3(8, 4, seed=seed + 19)
    ss.fit_source(Xt, Yt)
    ss_err = ind_nrmse(ss.forward(Xte), Yte, spec.value_range)

    return {
        "world": wname,
        "seed": seed,
        "mt_err": mt_err,
        "mt_ret": mt_ret,
        "gate": gate,
        "shift_det": bool(m.last_sensor_shift_detected),
        "ortho_err": ortho_err,
        "dt_err": dt_err,
        "sm_err": sm_err,
        "ft_err": ft_err,
        "ft_ret": ft_ret,
        "ss_err": ss_err,
    }


def evaluate_seed_suite(seeds: List[int], label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label, "seeds": seeds, "worlds": {}}
    for spec in TARGET_WORLDS_IND:
        tasks = [(spec.name, s) for s in seeds]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(_worker_v3_seed, tasks))
        mt = np.array([r["mt_err"] for r in rows])
        ret = np.array([r["mt_ret"] for r in rows])
        sm = np.array([r["sm_err"] for r in rows])
        ft = np.array([r["ft_err"] for r in rows])
        dt = np.array([r["dt_err"] for r in rows])
        ss = np.array([r["ss_err"] for r in rows])
        gates = np.array([r["gate"] for r in rows])
        shifts = np.array([r["shift_det"] for r in rows])
        out["worlds"][spec.name] = {
            "modular_v3": summarize_distribution(mt),
            "source_retention": summarize_distribution(ret),
            "direct_transfer": summarize_distribution(dt),
            "scratch_mlp": summarize_distribution(sm),
            "finetune_mlp": summarize_distribution(ft),
            "scratch_structural_v3": summarize_distribution(ss),
            "gate_on": int(np.sum(gates == 1.0)),
            "gate_off": int(np.sum(gates == 0.0)),
            "shift_detected": int(np.sum(shifts)),
            "max_ortho_err": float(max(r["ortho_err"] for r in rows)),
            "vs_scratch_mlp": ind_paired_stats(mt, sm),
            "vs_finetune_mlp": ind_paired_stats(mt, ft),
            "vs_direct_transfer": ind_paired_stats(mt, dt),
            "vs_scratch_structural_v3": ind_paired_stats(mt, ss),
        }

    # Evaluate Gate C (beats both MLPs all 4 worlds CI > 0 + retention < 0.02) and Gate D (negative control)
    beats_both_mlps_all_4 = all(
        out["worlds"][w.name]["vs_scratch_mlp"]["ci_strictly_positive"]
        and out["worlds"][w.name]["vs_finetune_mlp"]["ci_strictly_positive"]
        for w in TARGET_WORLDS_IND
    )
    ret_all_ok = all(out["worlds"][w.name]["source_retention"]["max"] < 0.02 for w in TARGET_WORLDS_IND)
    neg_w = out["worlds"]["W16_negative_control"]
    gate_d_pass = bool(
        neg_w["gate_off"] >= 9
        and neg_w["direct_transfer"]["min"] >= 0.40
        and neg_w["vs_finetune_mlp"]["ci_strictly_positive"]
        and neg_w["vs_scratch_mlp"]["ci_strictly_positive"]
    )
    out["gates_v3"] = {
        "beats_both_mlps_all_4_worlds_ci_gt_0": beats_both_mlps_all_4,
        "source_retention_max_lt_0.02_all_worlds": ret_all_ok,
        "Gate_D_negative_transfer_PASS": gate_d_pass,
    }
    return out


def evaluate_boundary_fixes(seeds: List[int]) -> Dict[str, Any]:
    # 1. Sample-ordering permutation invariance on W16_obs_shift
    spreads = []
    for s in seeds[:5]:
        Xs, Ys = W_SRC.sample(320, s)
        Xt, Yt = W_OBS.sample(20, s + 1020)
        Xte, Yte = W_OBS.eval_grid()
        m0 = ModularTransferModelV3(8, 4, seed=s); m0.fit_source(Xs, Ys)
        rng = np.random.default_rng(s + 999)
        errs = []
        for _ in range(5):
            perm = rng.permutation(len(Xt))
            mc = m0.clone_for_target(); mc.adapt_few_shot(Xt[perm], Yt[perm], align_sensors="auto")
            errs.append(ind_nrmse(mc.forward(Xte), Yte, W_OBS.value_range))
        spreads.append(float(max(errs) - min(errs)))

    # 2. Nonzero additive sensor offset c = 0.5
    w_off = IndWorldSpec("W16_offset_0.5", _w_source, M_SHIFT_IND, 36.0, additive_offset=0.5)
    off_errs, off_gates = [], []
    for s in seeds:
        Xs, Ys = W_SRC.sample(320, s); Xt, Yt = w_off.sample(20, s + 1020); Xte, Yte = w_off.eval_grid()
        m = ModularTransferModelV3(8, 4, seed=s); m.fit_source(Xs, Ys)
        mc = m.clone_for_target(); g = mc.adapt_few_shot(Xt, Yt, align_sensors="auto")
        off_errs.append(ind_nrmse(mc.forward(Xte), Yte, w_off.value_range)); off_gates.append(g)

    # 3. In-subspace pure rotation M_in = Q @ M_src
    rng_q = np.random.default_rng(1699)
    Q_in, _ = np.linalg.qr(rng_q.standard_normal((4, 4)))
    w_inrot = IndWorldSpec("W16_in_subspace_rot", _w_source, Q_in @ M_SRC_IND, 36.0)
    in_errs, in_gates, in_shifts = [], [], []
    for s in seeds:
        Xs, Ys = W_SRC.sample(320, s); Xt, Yt = w_inrot.sample(20, s + 1020); Xte, Yte = w_inrot.eval_grid()
        m = ModularTransferModelV3(8, 4, seed=s); m.fit_source(Xs, Ys)
        mc = m.clone_for_target(); g = mc.adapt_few_shot(Xt, Yt, align_sensors="auto")
        in_errs.append(ind_nrmse(mc.forward(Xte), Yte, w_inrot.value_range))
        in_gates.append(g); in_shifts.append(mc.last_sensor_shift_detected)

    return {
        "sample_ordering_obs_shift_max_spread": float(max(spreads)),
        "additive_offset_0.5": {
            "nrmse": summarize_distribution(np.array(off_errs)),
            "gate_on": int(sum(g == 1.0 for g in off_gates)),
            "n": len(seeds),
        },
        "in_subspace_rotation": {
            "nrmse": summarize_distribution(np.array(in_errs)),
            "shift_detected": int(sum(in_shifts)),
            "gate_on": int(sum(g == 1.0 for g in in_gates)),
            "n": len(seeds),
        },
    }


def main() -> None:
    print("=== [P16C] Evaluating ModularTransferModelV3 on Phase 16B seeds (101-110) ===", flush=True)
    res_16b = evaluate_seed_suite(SEEDS16B, "phase16b_seeds_101_110")
    print("P16B seeds gates_v3:", json.dumps(res_16b["gates_v3"], indent=2), flush=True)

    print("=== [P16C] Evaluating ModularTransferModelV3 on FRESH UNSEEN seeds (121-130) ===", flush=True)
    res_fresh = evaluate_seed_suite(SEEDS16C_FRESH, "fresh_unseen_seeds_121_130")
    print("Fresh seeds gates_v3:", json.dumps(res_fresh["gates_v3"], indent=2), flush=True)

    print("=== [P16C] Evaluating Boundary Fixes on FRESH UNSEEN seeds (121-130) ===", flush=True)
    res_bound = evaluate_boundary_fixes(SEEDS16C_FRESH)
    print("Boundary fixes:", json.dumps(res_bound, indent=2), flush=True)

    Path("logs/phase16c").mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "CSP-P16C-E001",
        "phase16b_seeds": res_16b,
        "fresh_unseen_seeds": res_fresh,
        "boundary_fixes_fresh": res_bound,
    }
    with open("logs/phase16c/p16c_v3_verification.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved logs/phase16c/p16c_v3_verification.json", flush=True)


if __name__ == "__main__":
    main()
