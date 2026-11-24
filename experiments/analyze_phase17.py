"""Registered analysis for Phase 17 (CSP-P17-E001)."""
import json
from typing import Any, Dict, List
import numpy as np

from experiments.phase11_experiments import bootstrap_ci
from experiments.phase17_experiments import ARCHS17, SEEDS17


def _vec(rows: List[Dict[str, Any]], arch: str, seeds=SEEDS17) -> List[Dict[str, Any]]:
    sel = [r for r in rows if r["arch"] == arch]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (arch, len(sel))
    return sel


def paired_ci_summary(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    lo, hi = bootstrap_ci(diff)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    dz = float(diff.mean() / sd) if sd > 1e-12 else 0.0
    return {
        "mean_diff": round(float(diff.mean()), 5),
        "sd_diff": round(sd, 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "d_z": round(dz, 4),
        "wins": int((diff > 0.0).sum()),
        "n": int(len(diff)),
        "ci_gt_zero": bool(lo > 0.0),
    }


def main(
    matrix_path: str = "logs/phase17/confirmatory_matrix.json",
    out_path: str = "logs/phase17/confirmatory_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    out: Dict[str, Any] = {
        "experiment_id": "CSP-P17-E001",
        "seeds": list(SEEDS17),
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
        "exclusions": [r for r in rows if not r["finite"]],
        "archs": {},
        "tests": {},
    }

    metrics = (
        "mean_ret_post4",
        "mean_forgetting_alpha_beta",
        "forward_transfer_gamma",
        "stage5_beta_mutated_nrmse",
        "ret_alpha_post5",
        "ret_alpha_post4",
        "ret_beta_post4",
        "ret_gamma_post4",
        "ret_delta_post4",
    )

    for a in ARCHS17:
        s = _vec(rows, a)
        out["archs"][a] = {
            "n_params_final_mean": float(np.mean([r["n_params_final"] for r in s])),
            "raw_replay_stored_mean": float(np.mean([r["raw_replay_stored"] for r in s])),
            "retrieval_acc_mean": round(float(np.mean([r["retrieval_acc_post4"] for r in s])), 4),
            "outdated_detected_seeds": int(sum(bool(r["outdated_detected_stage5"]) for r in s)),
        }
        for mk in metrics:
            arr = np.array([r[mk] for r in s], dtype=float)
            lo, hi = bootstrap_ci(arr)
            out["archs"][a][mk] = {
                "mean": round(float(arr.mean()), 5),
                "sd": round(float(arr.std(ddof=1)), 5),
                "median": round(float(np.median(arr)), 5),
                "min": round(float(arr.min()), 5),
                "max": round(float(arr.max()), 5),
                "ci95": [round(float(lo), 5), round(float(hi), 5)],
                "per_seed": arr.round(5).tolist(),
            }

    def g(arch: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in _vec(rows, arch)], dtype=float)

    # P17-H1: Retention & anti-forgetting across Stages 1-4
    pm_ret = g("persistent_memory", "mean_ret_post4")
    pm_forg = g("persistent_memory", "mean_forgetting_alpha_beta")
    h1_cells = {
        b: paired_ci_summary(pm_ret, g(b, "mean_ret_post4"))
        for b in ("naive_sequential_mlp", "replay_mlp", "ewc_mlp", "memoryless_structural")
    }
    ret_lt_025 = int((pm_ret < 0.025).sum())
    mean_forg = float(pm_forg.mean())
    out["tests"]["P17-H1"] = {
        "ret_lt_0.025_seeds": ret_lt_025,
        "mean_forgetting_alpha_beta": round(mean_forg, 5),
        "cells": h1_cells,
        "met": bool(ret_lt_025 >= 9 and mean_forg < 0.01 and all(v["ci_gt_zero"] for v in h1_cells.values())),
    }

    # P17-H2: Autonomous memory retrieval accuracy >= 0.95 (38/40)
    retr_hits = int(sum(r["retrieval_correct_post4"] for r in _vec(rows, "persistent_memory")))
    retr_tot = int(sum(r["retrieval_total_post4"] for r in _vec(rows, "persistent_memory")))
    out["tests"]["P17-H2"] = {
        "correct": retr_hits,
        "total": retr_tot,
        "accuracy": round(float(retr_hits / retr_tot), 4),
        "met": bool(retr_hits / retr_tot >= 0.95),
    }

    # P17-H3: Underdetermined forward transfer on Stage 3 (W17_gamma_shared, 10-shot)
    pm_gam = g("persistent_memory", "forward_transfer_gamma")
    h3_cells = {
        b: paired_ci_summary(pm_gam, g(b, "forward_transfer_gamma"))
        for b in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
    }
    out["tests"]["P17-H3"] = {
        "cells": h3_cells,
        "met": bool(all(v["ci_gt_zero"] for v in h3_cells.values()) and h3_cells["memoryless_structural"]["wins"] >= 8),
    }

    # P17-H4: Outdated memory detection & rule-change adaptation on Stage 5 (W17_beta_mutated)
    st5_pm = g("persistent_memory", "stage5_beta_mutated_nrmse")
    a5_pm = g("persistent_memory", "ret_alpha_post5")
    out_det = int(sum(bool(r["outdated_detected_stage5"]) for r in _vec(rows, "persistent_memory")))
    st5_lt_025 = int((st5_pm < 0.025).sum())
    a5_lt_02 = int((a5_pm < 0.02).sum())
    h4_cells = {b: paired_ci_summary(st5_pm, g(b, "stage5_beta_mutated_nrmse")) for b in ("ewc_mlp", "replay_mlp")}
    out["tests"]["P17-H4"] = {
        "outdated_detected_seeds": out_det,
        "stage5_lt_0.025_seeds": st5_lt_025,
        "alpha_post5_lt_0.02_seeds": a5_lt_02,
        "cells": h4_cells,
        "met": bool(out_det == 10 and st5_lt_025 >= 9 and a5_lt_02 >= 9 and all(v["ci_gt_zero"] for v in h4_cells.values())),
    }

    # P17-H5: Memory efficiency (<= 330 total params, 0 raw replay, beats replay_mlp & ewc_mlp on retention)
    max_params = int(max(r["n_params_final"] for r in _vec(rows, "persistent_memory")))
    max_replay = int(max(r["raw_replay_stored"] for r in _vec(rows, "persistent_memory")))
    out["tests"]["P17-H5"] = {
        "max_params_final": max_params,
        "max_raw_replay_stored": max_replay,
        "beats_replay_and_ewc_retention": bool(h1_cells["replay_mlp"]["ci_gt_zero"] and h1_cells["ewc_mlp"]["ci_gt_zero"]),
        "met": bool(max_params <= 330 and max_replay == 0 and h1_cells["replay_mlp"]["ci_gt_zero"] and h1_cells["ewc_mlp"]["ci_gt_zero"]),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res["tests"].items():
        print(k, v["met"])
