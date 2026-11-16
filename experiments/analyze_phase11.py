"""Registered analysis for Phase 11 (docs/phase11/EXPERIMENT_REGISTRY.md). Reads confirmatory logs, writes logs/phase11/confirmatory_analysis.json."""
import json, collections
import numpy as np
from experiments.phase11_experiments import paired_summary, SEEDS, bootstrap_ci

def load():
    m = json.load(open("logs/phase11/confirmatory_matrix.json"))
    return m["rows"], m["cka"], json.load(open("logs/phase11/confirmatory_a1.json")), json.load(open("logs/phase11/confirmatory_adaptation.json"))

def vec(rows, world, lvl, arch, key):
    sel = [r for r in rows if r["world"] == world and r["arch"] == arch and r["steps"] == lvl]
    sel.sort(key=lambda r: SEEDS.index(r["seed"]))
    return np.array([r[key] if "." not in key else r[key.split(".")[0]][key.split(".")[1]] for r in sel])

def main():
    rows, cka, a1, ad = load()
    steps = {w: sorted({r["steps"] for r in rows if r["world"] == w}) for w in ("W1", "W2")}
    out = {"steps": steps, "descriptives": {}, "tests": {}, "exclusions": [r for r in rows if not r["finite"] or r["n_train_events"] < 50], "leak_rows_total": sum(r["leak_rows"] for r in rows)}
    for w in ("W1", "W2"):
        for li, lvl in enumerate(steps[w]):
            lv = ("lo", "hi")[li]
            for arch in sorted({r["arch"] for r in rows}):
                for var in ("additive", "interaction"):
                    x = vec(rows, w, lvl, arch, f"heldout_nrmse_{var}")
                    lo, hi = bootstrap_ci(x)
                    out["descriptives"][f"{w}|{lv}|{arch}|{var}"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci95": [lo, hi], "per_seed": x.round(5).tolist(), "n_params": int(vec(rows, w, lvl, arch, "n_params")[0])}
                x = vec(rows, w, lvl, arch, "indist_nrmse"); out["descriptives"][f"{w}|{lv}|{arch}|indist"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1))}
            for bname in ("memorisation", "additive_fit", "oracle_additive_formula"):
                for var in ("additive", "interaction"):
                    x = np.array([r[f"baselines_nrmse_{var}"][bname] for r in rows if r["world"] == w and r["steps"] == lvl and r["arch"] == "mlp_small"])
                    out["descriptives"][f"{w}|{lv}|baseline:{bname}|{var}"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1))}
    def T(name, w, lv, a_arch, b_arch, key, m):
        lvl = steps[w][("lo", "hi").index(lv)]
        a, b = vec(rows, w, lvl, a_arch, key), vec(rows, w, lvl, b_arch, key)
        out["tests"][name] = {"world": w, "level": lv, "candidate": a_arch, "baseline": b_arch, "metric": key, "margin": m, **paired_summary(a, b, m)}
    for lv in ("hi", "lo"):
        T(f"H1_{lv}", "W1", lv, "additive", "mlp_small", "heldout_nrmse_additive", 0.02)
        T(f"H2_{lv}", "W1", lv, "mlp_large", "mlp_small", "heldout_nrmse_additive", 0.02)
        T(f"H3_{lv}", "W1", lv, "factorised", "mlp_small", "heldout_nrmse_additive", 0.02)
        T(f"H5_{lv}", "W2", lv, "additive", "mlp_small", "heldout_nrmse_additive", 0.02)
        T(f"H7a_{lv}", "W1", lv, "additive", "mlp_small", "flip_additive.unseen_mae", 0.02)
        T(f"H7b_{lv}", "W2", lv, "additive", "mlp_small", "distractor_sensitivity", 0.005)
        T(f"extra_additive_vs_additive_mlp_{lv}", "W1", lv, "additive", "additive_mlp", "heldout_nrmse_additive", 0.02)
    hi = steps["W1"][1]
    x = vec(rows, "W1", hi, "additive", "heldout_nrmse_interaction")
    out["tests"]["H4"] = {"additive_interaction_nrmse_per_seed": x.round(4).tolist(), "seeds_ge_0.25": int((x >= 0.25).sum()), "met": bool((x >= 0.25).sum() >= 9 and out["leak_rows_total"] == 0)}
    # A1 (Phase 10 pipeline, full input) vs reduced-input mlp_small
    a1x = np.array([r["heldout_nrmse_additive"] for r in sorted(a1, key=lambda r: SEEDS.index(r["seed"]))])
    out["A1_phase10_pipeline_W1_hi"] = {"nrmse_additive_mean": float(a1x.mean()), "sd": float(a1x.std(ddof=1)), "n_params": a1[0]["n_params"], "per_seed": a1x.round(5).tolist()}
    # per-combination, pairwise, flips, probes (means at hi, W1)
    mech = {}
    for w in ("W1", "W2"):
        for arch in sorted({r["arch"] for r in rows}):
            sel = [r for r in rows if r["world"] == w and r["steps"] == steps[w][1] and r["arch"] == arch]
            mech[f"{w}|{arch}"] = {"pairwise_acc_additive": float(np.mean([r["pairwise_acc_heldout_pairs_additive"] for r in sel])),
                                   "flip_unseen_mae": float(np.mean([r["flip_additive"]["unseen_mae"] for r in sel])), "flip_seen_mae": float(np.mean([r["flip_additive"]["seen_mae"] for r in sel])),
                                   "effect_std_across_contexts": float(np.mean([r["flip_additive"]["effect_std_across_contexts"] for r in sel])),
                                   "probe_cross_combo_r2": float(np.mean([r["probe_cross_combo_r2_additive"] for r in sel])), "probe_shuffled_r2": float(np.mean([r["probe_cross_combo_r2_shuffled"] for r in sel])),
                                   "distractor_sensitivity": float(np.mean([r["distractor_sensitivity"] for r in sel])) if "distractor_sensitivity" in sel[0] else None,
                                   "eta2": {k: float(np.mean([r["latent_marginal_eta2"][k] for r in sel])) for k in sel[0]["latent_marginal_eta2"]}}
    out["mechanism_hi"] = mech
    ck = collections.defaultdict(list)
    for c in cka: ck[(c["world"], c["steps"], c["pair"])].append(c["cka"])
    out["cka_mean"] = {f"{k[0]}|{k[1]}|{k[2]}": float(np.nanmean(v)) for k, v in ck.items()}
    # H6 and forgetting (W1 interaction, replay, n=20, epoch 30)
    adr = collections.defaultdict(list)
    for r in ad:
        adr[(r["world"], r["arch"], r["variant"], r["mode"], r["n_reveal"])].append(r)
    h6 = {}
    for (w, arch, var, mode, n), v in adr.items():
        gain_unrev = np.array([x["curve"]["0"]["unrevealed"] - x["curve"]["30"]["unrevealed"] for x in v])
        gain_rev = np.array([x["curve"]["0"]["revealed"] - x["curve"]["30"]["revealed"] for x in v])
        forget = np.array([x["curve"]["30"]["train_combos"] - x["curve"]["0"]["train_combos"] for x in v])
        # average the two reveal orders within seed first
        per_seed = collections.defaultdict(list)
        for x, gu, gr, fg in zip(v, gain_unrev, gain_rev, forget): per_seed[x["seed"]].append((gu, gr, fg))
        gu_s = np.array([np.mean([t[0] for t in per_seed[s]]) for s in SEEDS]); gr_s = np.array([np.mean([t[1] for t in per_seed[s]]) for s in SEEDS]); fg_s = np.array([np.mean([t[2] for t in per_seed[s]]) for s in SEEDS])
        lo, hi_ = bootstrap_ci(gu_s)
        h6[f"{w}|{arch}|{var}|{mode}|n{n}"] = {"unrevealed_gain_mean": float(gu_s.mean()), "unrevealed_gain_ci95": [lo, hi_], "revealed_gain_mean": float(gr_s.mean()), "train_combo_error_increase_mean": float(fg_s.mean()),
                                                "initial_unrevealed_nrmse": float(np.mean([x["curve"]["0"]["unrevealed"] for x in v]))}
    out["adaptation"] = h6
    for arch in ("mlp_small", "additive_mlp", "additive"):
        k = f"W1|{arch}|interaction|replay|n20"
        g = h6[k]
        out["tests"][f"H6_{arch}"] = {"gain": g["unrevealed_gain_mean"], "ci95": g["unrevealed_gain_ci95"], "met_ge_0.05_and_ci_lo_gt_0": bool(g["unrevealed_gain_mean"] >= 0.05 and g["unrevealed_gain_ci95"][0] > 0)}
    json.dump(out, open("logs/phase11/confirmatory_analysis.json", "w"), indent=1)
    return out

if __name__ == "__main__":
    o = main()
    print(json.dumps({k: v for k, v in o["tests"].items()}, indent=1))
