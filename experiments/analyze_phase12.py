"""Registered analysis for Phase 12."""
import json, collections
import numpy as np
from experiments.phase11_experiments import paired_summary, bootstrap_ci
from experiments.phase12_experiments import SEEDS12, GRID, ARCHS12

def vec(rows, **kw):
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: SEEDS12.index(r["seed"]))
    assert len(sel) == len(SEEDS12), (kw, len(sel))
    return sel

def main():
    w3 = json.load(open("logs/phase12/confirmatory_w3.json")); sens = json.load(open("logs/phase12/confirmatory_sensitivity.json"))
    out = {"tests": {}, "w3": {}, "sensitivity": {}, "exclusions": [r for r in w3 + sens if not r["finite"] or r["n_train_events"] < 50], "leak_rows_total": sum(r["leak_rows"] for r in w3 + sens)}
    steps3 = sorted({r["steps"] for r in w3}); hi3 = steps3[1]
    for lv, st in zip(("lo", "hi"), steps3):
        for a in ARCHS12:
            s = vec(w3, arch=a, steps=st); h = np.array([r["heldout_nrmse"] for r in s]); i = np.array([r["indist_nrmse"] for r in s])
            lo, hi = bootstrap_ci(h)
            out["w3"][f"{lv}|{a}"] = {"heldout_mean": float(h.mean()), "heldout_sd": float(h.std(ddof=1)), "heldout_ci95": [lo, hi], "indist_mean": float(i.mean()), "n_params": s[0]["n_params"], "heldout_per_seed": h.round(5).tolist(),
                                      "per_combo_mae": {c: float(np.mean([r["per_combo"][c]["mae"] for r in s])) for c in s[0]["per_combo"]}}
        for b in ("memorisation", "additive_fit"):
            x = np.array([r["baselines_nrmse"][b] for r in vec(w3, arch="mlp_small", steps=st)]); out["w3"][f"{lv}|baseline:{b}"] = {"heldout_mean": float(x.mean())}
    g = lambda lv_steps, a, key: np.array([r[key] for r in vec(w3, arch=a, steps=lv_steps)])
    ind_add = g(hi3, "additive", "indist_nrmse")
    out["tests"]["P12-H1"] = {"indist_ge_0.05_seeds": int((ind_add >= 0.05).sum()), **{"T": paired_summary(g(hi3, "mlp_small", "heldout_nrmse"), g(hi3, "additive", "heldout_nrmse"), 0.02)}}
    out["tests"]["P12-H1"]["met"] = bool(out["tests"]["P12-H1"]["indist_ge_0.05_seeds"] >= 9 and out["tests"]["P12-H1"]["T"]["met"])
    # T(a vs b) says a better; H1 wants mlp_small better than additive -> candidate=mlp_small, baseline=additive
    out["tests"]["P12-H2a_W3_residual_vs_additive"] = paired_summary(g(hi3, "additive_residual", "heldout_nrmse"), g(hi3, "additive", "heldout_nrmse"), 0.02)
    steps1 = sorted({r["steps"] for r in sens if r["world"] == "W1"}); st1 = steps1[0]
    def sv(world, arch, lr, ep, key):
        return np.array([r[key] for r in vec(sens, world=world, arch=arch, lr=lr, epochs=ep)])
    # P12-H2b needs W1 at default config (lr 0.01, 200 epochs) from the sensitivity grid (same code path as Phase 11 defaults)
    out["tests"]["P12-H2b_W1_residual_vs_mlp_small_default_config"] = paired_summary(sv("W1", "additive_residual", 0.01, 200, "heldout_nrmse"), sv("W1", "mlp_small", 0.01, 200, "heldout_nrmse"), 0.02)
    mem = np.array([r["baselines_nrmse"]["memorisation"] for r in vec(w3, arch="mlp_small", steps=hi3)])
    out["tests"]["P12-H3_mlp_small_vs_memorisation_W3"] = paired_summary(g(hi3, "mlp_small", "heldout_nrmse"), mem, 0.02)
    # H4: all configs
    h4 = {}
    for w, m in (("W1", 0.02), ("W2", 0.0)):
        for lr, ep in GRID:
            a = sv(w, "additive", lr, ep, "heldout_nrmse").mean(); b = sv(w, "mlp_small", lr, ep, "heldout_nrmse").mean()
            h4[f"{w}|lr{lr}|ep{ep}"] = {"additive": float(a), "mlp_small": float(b), "diff": float(b - a), "ok": bool(b - a >= m if m > 0 else b - a > 0)}
    out["sensitivity"]["H4_grid"] = h4
    out["tests"]["P12-H4"] = {"all_configs_ok": all(v["ok"] for v in h4.values()), "n_configs": len(h4)}
    # selection by in-dist nRMSE
    sel = {}
    for w in ("W1", "W2"):
        for a in ARCHS12:
            best = min(GRID, key=lambda c: sv(w, a, c[0], c[1], "indist_nrmse").mean())
            sel[f"{w}|{a}"] = {"lr": best[0], "epochs": best[1], "indist": float(sv(w, a, *best, "indist_nrmse").mean()), "heldout_mean": float(sv(w, a, *best, "heldout_nrmse").mean()),
                               "heldout_sd": float(sv(w, a, *best, "heldout_nrmse").std(ddof=1))}
    out["sensitivity"]["selected"] = sel
    pick = lambda w, a: sv(w, a, sel[f"{w}|{a}"]["lr"], sel[f"{w}|{a}"]["epochs"], "heldout_nrmse")
    out["tests"]["P12-H5_W1_selected_additive_vs_mlp_small"] = paired_summary(pick("W1", "additive"), pick("W1", "mlp_small"), 0.02)
    out["tests"]["P12-H6_W1_selected_mlp_large_vs_mlp_small"] = paired_summary(pick("W1", "mlp_large"), pick("W1", "mlp_small"), 0.02)
    out["tests"]["extra_W2_selected_additive_vs_mlp_small"] = paired_summary(pick("W2", "additive"), pick("W2", "mlp_small"), 0.0)
    out["sensitivity"]["grid_means"] = {f"{w}|{a}|lr{lr}|ep{ep}": float(sv(w, a, lr, ep, "heldout_nrmse").mean()) for w in ("W1", "W2") for a in ARCHS12 for lr, ep in GRID}
    json.dump(out, open("logs/phase12/confirmatory_analysis.json", "w"), indent=1)
    return out

if __name__ == "__main__":
    o = main()
    for k, v in o["tests"].items(): print(k, json.dumps(v)[:400])
