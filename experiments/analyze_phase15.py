"""Registered analysis for Phase 15 (CSP-P15-E001)."""
import json
from typing import Any, Dict, List
import numpy as np
from experiments.phase11_experiments import bootstrap_ci, paired_summary
from experiments.phase15_experiments import ARCHS15, SEEDS15, STEPS15, WORLDS15


def paired_higher_better(a: np.ndarray, b: np.ndarray, threshold: float) -> Dict[str, Any]:
    return paired_summary(-np.asarray(a, dtype=float), -np.asarray(b, dtype=float), threshold)


def vec(rows: List[Dict[str, Any]], seeds=SEEDS15, **kw) -> List[Dict[str, Any]]:
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (kw, len(sel))
    return sel


def main(
    matrix_path: str = "logs/phase15/confirmatory_matrix.json",
    out_path: str = "logs/phase15/confirmatory_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    out: Dict[str, Any] = {
        "tests": {},
        "worlds": {},
        "exclusions": [r for r in rows if not r["finite"] or r["n_train_events"] < 50],
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
    }

    for spec in WORLDS15:
        wname = spec.name
        out["worlds"][wname] = {}
        for lv, st in zip(("lo", "hi"), STEPS15):
            for a in ARCHS15:
                s = vec(rows, world=wname, arch=a, steps=st)
                si = np.array([r["single_int_nrmse"] for r in s])
                mi = np.array([r["multi_int_nrmse"] for r in s])
                ate = np.array([r["ate_nmae"] for r in s])
                cf = np.array([r["counterfactual_nrmse"] for r in s])
                f1 = np.array([r["causal_f1"] for r in s])
                fp = np.array([r["false_causal_edges"] for r in s])
                lo_ci, hi_ci = bootstrap_ci(si)
                out["worlds"][wname][f"{lv}|{a}"] = {
                    "n_params": s[0]["n_params"],
                    "single_int_mean": float(si.mean()),
                    "single_int_sd": float(si.std(ddof=1)),
                    "single_int_ci95": [lo_ci, hi_ci],
                    "single_int_per_seed": si.round(5).tolist(),
                    "multi_int_mean": float(mi.mean()),
                    "multi_int_sd": float(mi.std(ddof=1)),
                    "ate_nmae_mean": float(ate.mean()),
                    "counterfactual_mean": float(cf.mean()),
                    "counterfactual_sd": float(cf.std(ddof=1)),
                    "causal_f1_mean": float(f1.mean()),
                    "causal_f1_ge_0.95_seeds": int((f1 >= 0.95).sum()),
                    "false_edges_mean": float(fp.mean()),
                }

    def g(wname: str, arch: str, key: str, lv: str = "hi") -> np.ndarray:
        st = STEPS15[0 if lv == "lo" else 1]
        return np.array([r[key] for r in vec(rows, world=wname, arch=arch, steps=st)], dtype=float)

    # P15-H1: Breaking spurious correlation & observed confounding (W15_corr_vs_cause, W15_obs_confounder)
    h1_cells = {}
    for w in ("W15_corr_vs_cause", "W15_obs_confounder"):
        for b in ("observational_mlp", "observational_linear", "random_intervention"):
            h1_cells[f"{w}|{b}"] = paired_summary(g(w, "causal_intervention", "single_int_nrmse"), g(w, b, "single_int_nrmse"), 0.05)
    out["tests"]["P15-H1"] = {
        "cells": h1_cells,
        "met": all(v["met"] for v in h1_cells.values()),
    }

    # P15-H2: Hidden-variable confounding (W15_hidden_confounder)
    h2_si_mlp = paired_summary(g("W15_hidden_confounder", "causal_intervention", "single_int_nrmse"), g("W15_hidden_confounder", "observational_mlp", "single_int_nrmse"), 0.05)
    h2_si_lin = paired_summary(g("W15_hidden_confounder", "causal_intervention", "single_int_nrmse"), g("W15_hidden_confounder", "observational_linear", "single_int_nrmse"), 0.05)
    h2_ate_mlp = paired_summary(g("W15_hidden_confounder", "causal_intervention", "ate_nmae"), g("W15_hidden_confounder", "observational_mlp", "ate_nmae"), 0.05)
    h2_ate_lin = paired_summary(g("W15_hidden_confounder", "causal_intervention", "ate_nmae"), g("W15_hidden_confounder", "observational_linear", "ate_nmae"), 0.05)
    out["tests"]["P15-H2"] = {
        "si_vs_obs_mlp": h2_si_mlp,
        "si_vs_obs_lin": h2_si_lin,
        "ate_vs_obs_mlp": h2_ate_mlp,
        "ate_vs_obs_lin": h2_ate_lin,
        "met": bool(h2_si_mlp["met"] and h2_si_lin["met"] and h2_ate_mlp["met"] and h2_ate_lin["met"]),
    }

    # P15-H3: Multi-variable interventions (W15_multi_cause)
    h3_obs_mlp = paired_summary(g("W15_multi_cause", "causal_intervention", "multi_int_nrmse"), g("W15_multi_cause", "observational_mlp", "multi_int_nrmse"), 0.03)
    h3_obs_lin = paired_summary(g("W15_multi_cause", "causal_intervention", "multi_int_nrmse"), g("W15_multi_cause", "observational_linear", "multi_int_nrmse"), 0.03)
    h3_mlp_int = paired_summary(g("W15_multi_cause", "causal_intervention", "multi_int_nrmse"), g("W15_multi_cause", "mlp_intervention", "multi_int_nrmse"), 0.03)
    out["tests"]["P15-H3"] = {
        "vs_obs_mlp": h3_obs_mlp,
        "vs_obs_lin": h3_obs_lin,
        "vs_mlp_int": h3_mlp_int,
        "met": bool(h3_obs_mlp["met"] and h3_obs_lin["met"] and h3_mlp_int["met"]),
    }

    # P15-H4: Counterfactual outcome prediction (W15_hidden_confounder)
    h4_obs_mlp = paired_summary(g("W15_hidden_confounder", "causal_intervention", "counterfactual_nrmse"), g("W15_hidden_confounder", "observational_mlp", "counterfactual_nrmse"), 0.05)
    h4_mlp_int = paired_summary(g("W15_hidden_confounder", "causal_intervention", "counterfactual_nrmse"), g("W15_hidden_confounder", "mlp_intervention", "counterfactual_nrmse"), 0.05)
    out["tests"]["P15-H4"] = {
        "vs_obs_mlp": h4_obs_mlp,
        "vs_mlp_int": h4_mlp_int,
        "met": bool(h4_obs_mlp["met"] and h4_mlp_int["met"]),
    }

    # P15-H5: Causal parent graph recovery across all 4 worlds
    def mean_across_worlds(arch: str, key: str) -> np.ndarray:
        return np.mean(np.stack([g(w.name, arch, key) for w in WORLDS15], axis=0), axis=0)

    f1_ci = mean_across_worlds("causal_intervention", "causal_f1")
    f1_seeds_95 = int((f1_ci >= 0.95).sum())
    h5_vs_obs = paired_higher_better(f1_ci, mean_across_worlds("observational_mlp", "causal_f1"), 0.15)
    h5_vs_mlp_int = paired_higher_better(f1_ci, mean_across_worlds("mlp_intervention", "causal_f1"), 0.15)
    out["tests"]["P15-H5"] = {
        "mean_f1_causal_intervention": float(f1_ci.mean()),
        "f1_ge_0.95_seeds": f1_seeds_95,
        "vs_obs_mlp": h5_vs_obs,
        "vs_mlp_int": h5_vs_mlp_int,
        "met": bool(f1_seeds_95 >= 9 and h5_vs_obs["met"] and h5_vs_mlp_int["met"]),
    }

    # P15-H6: Structured causal model vs unstructured interventional MLP on ATE nMAE across all 4 worlds
    ate_ci = mean_across_worlds("causal_intervention", "ate_nmae")
    ate_mlp_int = mean_across_worlds("mlp_intervention", "ate_nmae")
    h6 = paired_summary(ate_ci, ate_mlp_int, 0.015)
    out["tests"]["P15-H6"] = h6

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res["tests"].items():
        print(k, v.get("met", v))
