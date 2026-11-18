"""Registered analysis for Phase 13 (CSP-P13-E001)."""
import json
from typing import Any, Dict, List
import numpy as np
from experiments.phase11_experiments import bootstrap_ci, paired_summary
from experiments.phase13_experiments import ARCHS13, SEEDS13, W1_RAW, W2_RAW, W3_RAW


def paired_higher_better(a: np.ndarray, b: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Test 'a higher than b by >= threshold': equivalent to paired_summary(-a, -b, threshold)."""
    return paired_summary(-np.asarray(a, dtype=float), -np.asarray(b, dtype=float), threshold)


def vec(rows: List[Dict[str, Any]], seeds=SEEDS13, **kw) -> List[Dict[str, Any]]:
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (kw, len(sel))
    return sel


def main(
    matrix_path: str = "logs/phase13/confirmatory_matrix.json",
    cka_path: str = "logs/phase13/confirmatory_cka.json",
    out_path: str = "logs/phase13/confirmatory_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    cka_rows = json.load(open(cka_path))

    out: Dict[str, Any] = {
        "tests": {},
        "worlds": {},
        "cka": {},
        "exclusions": [r for r in rows if not r["finite"] or r["n_train_events"] < 50],
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
    }

    for spec in (W1_RAW, W2_RAW, W3_RAW):
        wname = spec.name
        w_steps = sorted({r["steps"] for r in rows if r["world"] == wname})
        out["worlds"][wname] = {}
        for lv, st in zip(("lo", "hi"), w_steps):
            for a in ARCHS13:
                s = vec(rows, world=wname, arch=a, steps=st)
                h = np.array([r["heldout_nrmse"] for r in s])
                ind = np.array([r["indist_nrmse"] for r in s])
                sm = np.array([r["slot_match_r2"] for r in s])
                pr = np.array([r["probe_cross_combo_r2"] for r in s])
                ps = np.array([r["probe_shuffled_r2"] for r in s])
                fu = np.array([r["flip_unseen_nmae"] for r in s])
                fs = np.array([r["flip_seen_nmae"] for r in s])
                sel = np.array([r["intervention_selectivity"] for r in s])
                ds = np.array([r["distractor_sensitivity"] for r in s])
                pw = np.array([r["pairwise_acc"] for r in s])
                lo_ci, hi_ci = bootstrap_ci(h)
                out["worlds"][wname][f"{lv}|{a}"] = {
                    "n_params": s[0]["n_params"],
                    "heldout_mean": float(h.mean()),
                    "heldout_sd": float(h.std(ddof=1)),
                    "heldout_ci95": [lo_ci, hi_ci],
                    "heldout_per_seed": h.round(5).tolist(),
                    "indist_mean": float(ind.mean()),
                    "indist_sd": float(ind.std(ddof=1)),
                    "slot_match_r2_mean": float(sm.mean()),
                    "slot_match_r2_sd": float(sm.std(ddof=1)),
                    "slot_match_ge_0.90_seeds": int((sm >= 0.90).sum()),
                    "probe_r2_mean": float(pr.mean()),
                    "probe_shuffled_r2_mean": float(ps.mean()),
                    "flip_unseen_nmae_mean": float(fu.mean()),
                    "flip_seen_nmae_mean": float(fs.mean()),
                    "intervention_selectivity_mean": float(sel.mean()),
                    "distractor_sensitivity_mean": float(ds.mean()),
                    "pairwise_acc_mean": float(pw.mean()),
                }

            for a in ARCHS13:
                if a == "oracle_channels_additive":
                    continue
                c_vals = np.array([
                    c["cka"] for c in cka_rows
                    if c["world"] == wname and c["arch"] == a and c["level"] == lv
                ])
                if len(c_vals) > 0:
                    out["cka"][f"{wname}|{lv}|{a}"] = {
                        "mean": float(c_vals.mean()),
                        "sd": float(c_vals.std(ddof=1)) if len(c_vals) > 1 else 0.0,
                        "n_pairs": int(len(c_vals)),
                    }

    def g(wname: str, arch: str, key: str, lv: str = "hi") -> np.ndarray:
        w_steps = sorted({r["steps"] for r in rows if r["world"] == wname})
        st = w_steps[0 if lv == "lo" else 1]
        return np.array([r[key] for r in vec(rows, world=wname, arch=arch, steps=st)], dtype=float)

    # P13-H1: Compositional extrapolation from raw observations in W1_raw and W2_raw
    h1_w1 = paired_summary(g("W1_raw", "latent_additive", "heldout_nrmse"), g("W1_raw", "raw_mlp_small", "heldout_nrmse"), 0.02)
    h1_w2 = paired_summary(g("W2_raw", "latent_additive", "heldout_nrmse"), g("W2_raw", "raw_mlp_small", "heldout_nrmse"), 0.02)
    out["tests"]["P13-H1_W1_latent_additive_vs_raw_mlp_small"] = h1_w1
    out["tests"]["P13-H1_W2_latent_additive_vs_raw_mlp_small"] = h1_w2
    out["tests"]["P13-H1"] = {"W1_met": h1_w1["met"], "W2_met": h1_w2["met"], "met": bool(h1_w1["met"] and h1_w2["met"])}

    # P13-H2: Superiority over pca_statistical, untrained_encoder, random_repr (margin 0.05)
    h2_cells = {}
    for w in ("W1_raw", "W2_raw"):
        for b in ("pca_statistical", "untrained_encoder", "random_repr"):
            h2_cells[f"{w}|{b}"] = paired_summary(g(w, "latent_additive", "heldout_nrmse"), g(w, b, "heldout_nrmse"), 0.05)
    out["tests"]["P13-H2_cells"] = h2_cells
    out["tests"]["P13-H2"] = {"all_cells_met": all(v["met"] for v in h2_cells.values()), "met": all(v["met"] for v in h2_cells.values())}

    # P13-H3: Necessity of unmixing before nonlinear additive readout in W2_raw (vs raw_naive_additive)
    h3_nrmse = paired_summary(g("W2_raw", "latent_additive", "heldout_nrmse"), g("W2_raw", "raw_naive_additive", "heldout_nrmse"), 0.02)
    h3_dist = paired_summary(g("W2_raw", "latent_additive", "distractor_sensitivity"), g("W2_raw", "raw_naive_additive", "distractor_sensitivity"), 0.005)
    out["tests"]["P13-H3_W2_nrmse_vs_raw_naive_additive"] = h3_nrmse
    out["tests"]["P13-H3_W2_distractor_vs_raw_naive_additive"] = h3_dist
    out["tests"]["P13-H3"] = {"nrmse_met": h3_nrmse["met"], "distractor_met": h3_dist["met"], "met": bool(h3_nrmse["met"] and h3_dist["met"])}

    # P13-H4: Autonomous single-slot property recovery (slot_match_r2 >= 0.90 in >= 9/10 seeds + beat pca_statistical and raw_mlp_small by 0.20)
    sm_w1 = g("W1_raw", "latent_additive", "slot_match_r2")
    sm_w2 = g("W2_raw", "latent_additive", "slot_match_r2")
    h4_w1_pca = paired_higher_better(sm_w1, g("W1_raw", "pca_statistical", "slot_match_r2"), 0.20)
    h4_w1_mlp = paired_higher_better(sm_w1, g("W1_raw", "raw_mlp_small", "slot_match_r2"), 0.20)
    h4_w2_pca = paired_higher_better(sm_w2, g("W2_raw", "pca_statistical", "slot_match_r2"), 0.20)
    h4_w2_mlp = paired_higher_better(sm_w2, g("W2_raw", "raw_mlp_small", "slot_match_r2"), 0.20)
    w1_s90 = int((sm_w1 >= 0.90).sum())
    w2_s90 = int((sm_w2 >= 0.90).sum())
    out["tests"]["P13-H4"] = {
        "W1_seeds_ge_0.90": w1_s90,
        "W2_seeds_ge_0.90": w2_s90,
        "W1_vs_pca": h4_w1_pca,
        "W1_vs_mlp": h4_w1_mlp,
        "W2_vs_pca": h4_w2_pca,
        "W2_vs_mlp": h4_w2_mlp,
        "met": bool(w1_s90 >= 9 and w2_s90 >= 9 and h4_w1_pca["met"] and h4_w1_mlp["met"] and h4_w2_pca["met"] and h4_w2_mlp["met"]),
    }

    # P13-H5: Cross-seed representation consistency (inter-seed Linear CKA >= 0.85 and > untrained_encoder, raw_mlp_small by 0.15)
    cka_w1_la = out["cka"]["W1_raw|hi|latent_additive"]["mean"]
    cka_w1_un = out["cka"]["W1_raw|hi|untrained_encoder"]["mean"]
    cka_w1_mlp = out["cka"]["W1_raw|hi|raw_mlp_small"]["mean"]
    cka_w2_la = out["cka"]["W2_raw|hi|latent_additive"]["mean"]
    cka_w2_un = out["cka"]["W2_raw|hi|untrained_encoder"]["mean"]
    cka_w2_mlp = out["cka"]["W2_raw|hi|raw_mlp_small"]["mean"]
    out["tests"]["P13-H5"] = {
        "W1_latent_additive_cka": cka_w1_la,
        "W1_untrained_cka": cka_w1_un,
        "W1_raw_mlp_small_cka": cka_w1_mlp,
        "W2_latent_additive_cka": cka_w2_la,
        "W2_untrained_cka": cka_w2_un,
        "W2_raw_mlp_small_cka": cka_w2_mlp,
        "met": bool(
            cka_w1_la >= 0.85 and cka_w2_la >= 0.85
            and (cka_w1_la - cka_w1_un >= 0.15) and (cka_w1_la - cka_w1_mlp >= 0.15)
            and (cka_w2_la - cka_w2_un >= 0.15) and (cka_w2_la - cka_w2_mlp >= 0.15)
        ),
    }

    # P13-H6: Causal intervention flip error and intervention selectivity
    h6_w1_flip = paired_summary(g("W1_raw", "latent_additive", "flip_unseen_nmae"), g("W1_raw", "raw_mlp_small", "flip_unseen_nmae"), 0.02)
    h6_w2_flip = paired_summary(g("W2_raw", "latent_additive", "flip_unseen_nmae"), g("W2_raw", "raw_mlp_small", "flip_unseen_nmae"), 0.02)
    h6_w1_sel = paired_higher_better(g("W1_raw", "latent_additive", "intervention_selectivity"), g("W1_raw", "raw_mlp_small", "intervention_selectivity"), 0.15)
    h6_w2_sel = paired_higher_better(g("W2_raw", "latent_additive", "intervention_selectivity"), g("W2_raw", "raw_mlp_small", "intervention_selectivity"), 0.15)
    out["tests"]["P13-H6_W1_flip_unseen"] = h6_w1_flip
    out["tests"]["P13-H6_W2_flip_unseen"] = h6_w2_flip
    out["tests"]["P13-H6_W1_selectivity"] = h6_w1_sel
    out["tests"]["P13-H6_W2_selectivity"] = h6_w2_sel
    out["tests"]["P13-H6"] = {
        "met": bool(h6_w1_flip["met"] and h6_w2_flip["met"] and h6_w1_sel["met"] and h6_w2_sel["met"]),
    }

    # P13-H7: Non-additive W3_raw negative control for latent_additive + residual recovery by latent_residual
    w3_ind_la = g("W3_raw", "latent_additive", "indist_nrmse")
    w3_ind_seeds = int((w3_ind_la >= 0.05).sum())
    h7_res_vs_add = paired_summary(g("W3_raw", "latent_residual", "heldout_nrmse"), g("W3_raw", "latent_additive", "heldout_nrmse"), 0.02)
    h7_res_vs_mlp = paired_summary(g("W3_raw", "latent_residual", "heldout_nrmse"), g("W3_raw", "raw_mlp_small", "heldout_nrmse"), 0.02)
    out["tests"]["P13-H7_W3_residual_vs_additive"] = h7_res_vs_add
    out["tests"]["P13-H7_W3_residual_vs_mlp_small"] = h7_res_vs_mlp
    out["tests"]["P13-H7"] = {
        "latent_additive_indist_ge_0.05_seeds": w3_ind_seeds,
        "res_vs_add": h7_res_vs_add,
        "res_vs_mlp": h7_res_vs_mlp,
        "met": bool(w3_ind_seeds >= 9 and h7_res_vs_add["met"] and h7_res_vs_mlp["met"]),
    }

    Path_out = open(out_path, "w")
    json.dump(out, Path_out, indent=1)
    Path_out.close()
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res["tests"].items():
        print(k, json.dumps(v)[:300])
