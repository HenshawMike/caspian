"""Registered analysis for Phase 14 (CSP-P14-E001)."""
import json
from typing import Any, Dict, List
import numpy as np
from experiments.phase11_experiments import bootstrap_ci, paired_summary
from experiments.phase14_experiments import ARCHS14, SEEDS14, WORLDS14


def vec(rows: List[Dict[str, Any]], seeds=SEEDS14, **kw) -> List[Dict[str, Any]]:
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (kw, len(sel))
    return sel


def main(
    matrix_path: str = "logs/phase14/confirmatory_matrix.json",
    out_path: str = "logs/phase14/confirmatory_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    out: Dict[str, Any] = {
        "tests": {},
        "worlds": {},
        "exclusions": [r for r in rows if not r["finite"] or r["n_train_events"] < 50],
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
    }

    for spec in WORLDS14:
        wname = spec.name
        w_steps = sorted({r["steps"] for r in rows if r["world"] == wname})
        out["worlds"][wname] = {}
        for lv, st in zip(("lo", "hi"), w_steps):
            for a in ARCHS14:
                s = vec(rows, world=wname, arch=a, steps=st)
                h = np.array([r["heldout_nrmse"] for r in s])
                ind = np.array([r["indist_nrmse"] for r in s])
                pr = np.array([r["interaction_precision"] for r in s])
                rc = np.array([r["interaction_recall"] for r in s])
                f1 = np.array([r["interaction_f1"] for r in s])
                fp = np.array([r["false_interaction_count"] for r in s])
                me = np.array([r["main_effect_nmae"] for r in s])
                lo_ci, hi_ci = bootstrap_ci(h)
                out["worlds"][wname][f"{lv}|{a}"] = {
                    "n_params": s[0]["n_params"],
                    "heldout_mean": float(h.mean()),
                    "heldout_sd": float(h.std(ddof=1)),
                    "heldout_ci95": [lo_ci, hi_ci],
                    "heldout_per_seed": h.round(5).tolist(),
                    "indist_mean": float(ind.mean()),
                    "indist_sd": float(ind.std(ddof=1)),
                    "precision_mean": float(pr.mean()),
                    "recall_mean": float(rc.mean()),
                    "f1_mean": float(f1.mean()),
                    "f1_sd": float(f1.std(ddof=1)),
                    "false_positives_mean": float(fp.mean()),
                    "zero_fp_seeds": int((fp == 0).sum()),
                    "f1_ge_0.95_seeds": int((f1 >= 0.95).sum()),
                    "f1_ge_0.90_seeds": int((f1 >= 0.90).sum()),
                    "f1_ge_0.85_seeds": int((f1 >= 0.85).sum()),
                    "main_effect_nmae_mean": float(me.mean()),
                }

    def g(wname: str, arch: str, key: str, lv: str = "hi") -> np.ndarray:
        w_steps = sorted({r["steps"] for r in rows if r["world"] == wname})
        st = w_steps[0 if lv == "lo" else 1]
        return np.array([r[key] for r in vec(rows, world=wname, arch=arch, steps=st)], dtype=float)

    # P14-H1: W14_indep zero false interactions in >= 9/10 seeds + beat mlp_small (0.02) and additive_residual (0.01)
    fp_indep = g("W14_indep", "sparse_interaction", "false_interaction_count")
    zero_fp_s = int((fp_indep == 0).sum())
    h1_mlp = paired_summary(g("W14_indep", "sparse_interaction", "heldout_nrmse"), g("W14_indep", "mlp_small", "heldout_nrmse"), 0.02)
    h1_res = paired_summary(g("W14_indep", "sparse_interaction", "heldout_nrmse"), g("W14_indep", "additive_residual", "heldout_nrmse"), 0.01)
    out["tests"]["P14-H1"] = {
        "zero_fp_seeds": zero_fp_s,
        "vs_mlp_small": h1_mlp,
        "vs_additive_residual": h1_res,
        "met": bool(zero_fp_s >= 9 and h1_mlp["met"] and h1_res["met"]),
    }

    # P14-H2: W14_sparse_pair F1 >= 0.95 in >= 9/10 seeds + beat additive, additive_residual, mlp_small (margin 0.02)
    f1_sp = int((g("W14_sparse_pair", "sparse_interaction", "interaction_f1") >= 0.95).sum())
    h2_add = paired_summary(g("W14_sparse_pair", "sparse_interaction", "heldout_nrmse"), g("W14_sparse_pair", "additive", "heldout_nrmse"), 0.02)
    h2_res = paired_summary(g("W14_sparse_pair", "sparse_interaction", "heldout_nrmse"), g("W14_sparse_pair", "additive_residual", "heldout_nrmse"), 0.02)
    h2_mlp = paired_summary(g("W14_sparse_pair", "sparse_interaction", "heldout_nrmse"), g("W14_sparse_pair", "mlp_small", "heldout_nrmse"), 0.02)
    out["tests"]["P14-H2"] = {
        "f1_ge_0.95_seeds": f1_sp,
        "vs_additive": h2_add,
        "vs_additive_residual": h2_res,
        "vs_mlp_small": h2_mlp,
        "met": bool(f1_sp >= 9 and h2_add["met"] and h2_res["met"] and h2_mlp["met"]),
    }

    # P14-H3: W14_multi_pair F1 >= 0.90 in >= 9/10 seeds + beat additive, additive_residual, mlp_small (margin 0.02)
    f1_mp = int((g("W14_multi_pair", "sparse_interaction", "interaction_f1") >= 0.90).sum())
    h3_add = paired_summary(g("W14_multi_pair", "sparse_interaction", "heldout_nrmse"), g("W14_multi_pair", "additive", "heldout_nrmse"), 0.02)
    h3_res = paired_summary(g("W14_multi_pair", "sparse_interaction", "heldout_nrmse"), g("W14_multi_pair", "additive_residual", "heldout_nrmse"), 0.02)
    h3_mlp = paired_summary(g("W14_multi_pair", "sparse_interaction", "heldout_nrmse"), g("W14_multi_pair", "mlp_small", "heldout_nrmse"), 0.02)
    out["tests"]["P14-H3"] = {
        "f1_ge_0.90_seeds": f1_mp,
        "vs_additive": h3_add,
        "vs_additive_residual": h3_res,
        "vs_mlp_small": h3_mlp,
        "met": bool(f1_mp >= 9 and h3_add["met"] and h3_res["met"] and h3_mlp["met"]),
    }

    # P14-H4: W14_higher_order F1 >= 0.90 in >= 9/10 seeds + beat additive, additive_residual, mlp_small (margin 0.02)
    f1_ho = int((g("W14_higher_order", "sparse_interaction", "interaction_f1") >= 0.90).sum())
    h4_add = paired_summary(g("W14_higher_order", "sparse_interaction", "heldout_nrmse"), g("W14_higher_order", "additive", "heldout_nrmse"), 0.02)
    h4_res = paired_summary(g("W14_higher_order", "sparse_interaction", "heldout_nrmse"), g("W14_higher_order", "additive_residual", "heldout_nrmse"), 0.02)
    h4_mlp = paired_summary(g("W14_higher_order", "sparse_interaction", "heldout_nrmse"), g("W14_higher_order", "mlp_small", "heldout_nrmse"), 0.02)
    out["tests"]["P14-H4"] = {
        "f1_ge_0.90_seeds": f1_ho,
        "vs_additive": h4_add,
        "vs_additive_residual": h4_res,
        "vs_mlp_small": h4_mlp,
        "met": bool(f1_ho >= 9 and h4_add["met"] and h4_res["met"] and h4_mlp["met"]),
    }

    # P14-H5: W14_noisy_pair F1 >= 0.85 in >= 8/10 seeds + beat additive_residual and mlp_small (margin 0.015)
    f1_np = int((g("W14_noisy_pair", "sparse_interaction", "interaction_f1") >= 0.85).sum())
    h5_res = paired_summary(g("W14_noisy_pair", "sparse_interaction", "heldout_nrmse"), g("W14_noisy_pair", "additive_residual", "heldout_nrmse"), 0.015)
    h5_mlp = paired_summary(g("W14_noisy_pair", "sparse_interaction", "heldout_nrmse"), g("W14_noisy_pair", "mlp_small", "heldout_nrmse"), 0.015)
    out["tests"]["P14-H5"] = {
        "f1_ge_0.85_seeds": f1_np,
        "vs_additive_residual": h5_res,
        "vs_mlp_small": h5_mlp,
        "met": bool(f1_np >= 8 and h5_res["met"] and h5_mlp["met"]),
    }

    # P14-H6: Mean false_interaction_count across all 5 worlds lower than additive_residual, factorised, mlp_small, mlp_large by >= 1.0
    def mean_fp_per_seed(arch: str) -> np.ndarray:
        mats = [g(w.name, arch, "false_interaction_count") for w in WORLDS14]
        return np.mean(np.stack(mats, axis=0), axis=0)

    fp_sparse = mean_fp_per_seed("sparse_interaction")
    h6_cells = {}
    for b in ("additive_residual", "factorised", "mlp_small", "mlp_large"):
        h6_cells[b] = paired_summary(fp_sparse, mean_fp_per_seed(b), 1.0)
    out["tests"]["P14-H6"] = {
        "cells": h6_cells,
        "met": all(v["met"] for v in h6_cells.values()),
    }

    # P14-H7: Main-effect recovery nMAE lower than additive and mlp_small in W14_sparse_pair and W14_higher_order (margin 0.02)
    h7_sp_add = paired_summary(g("W14_sparse_pair", "sparse_interaction", "main_effect_nmae"), g("W14_sparse_pair", "additive", "main_effect_nmae"), 0.02)
    h7_sp_mlp = paired_summary(g("W14_sparse_pair", "sparse_interaction", "main_effect_nmae"), g("W14_sparse_pair", "mlp_small", "main_effect_nmae"), 0.02)
    h7_ho_add = paired_summary(g("W14_higher_order", "sparse_interaction", "main_effect_nmae"), g("W14_higher_order", "additive", "main_effect_nmae"), 0.02)
    h7_ho_mlp = paired_summary(g("W14_higher_order", "sparse_interaction", "main_effect_nmae"), g("W14_higher_order", "mlp_small", "main_effect_nmae"), 0.02)
    out["tests"]["P14-H7"] = {
        "sp_vs_additive": h7_sp_add,
        "sp_vs_mlp_small": h7_sp_mlp,
        "ho_vs_additive": h7_ho_add,
        "ho_vs_mlp_small": h7_ho_mlp,
        "met": bool(h7_sp_add["met"] and h7_sp_mlp["met"] and h7_ho_add["met"] and h7_ho_mlp["met"]),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res["tests"].items():
        print(k, v.get("met", v))
