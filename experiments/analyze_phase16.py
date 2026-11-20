"""Registered analysis for Phase 16 (CSP-P16-E001)."""
import json
from typing import Any, Dict, List
import numpy as np
from experiments.phase11_experiments import bootstrap_ci, paired_summary
from experiments.phase16_experiments import ARCHS16, SEEDS16, SHOTS16, TARGET_WORLDS16


def vec(rows: List[Dict[str, Any]], seeds=SEEDS16, **kw) -> List[Dict[str, Any]]:
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (kw, len(sel))
    return sel


def main(
    matrix_path: str = "logs/phase16/confirmatory_matrix.json",
    out_path: str = "logs/phase16/confirmatory_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    out: Dict[str, Any] = {
        "tests": {},
        "worlds": {},
        "exclusions": [r for r in rows if not r["finite"]],
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
    }

    for spec in TARGET_WORLDS16:
        wname = spec.name
        out["worlds"][wname] = {}
        for lv, sh in zip(("lo_shot", "hi_shot"), SHOTS16):
            for a in ARCHS16:
                s = vec(rows, world=wname, arch=a, n_shot=sh)
                t = np.array([r["target_nrmse"] for r in s])
                ret = np.array([r["source_retention_nrmse"] for r in s])
                cg = np.array([r["core_gate"] for r in s])
                lo_ci, hi_ci = bootstrap_ci(t)
                out["worlds"][wname][f"{lv}|{a}"] = {
                    "n_params": s[0]["n_params"],
                    "target_mean": float(t.mean()),
                    "target_sd": float(t.std(ddof=1)),
                    "target_ci95": [lo_ci, hi_ci],
                    "target_per_seed": t.round(5).tolist(),
                    "retention_mean": float(ret.mean()),
                    "retention_sd": float(ret.std(ddof=1)),
                    "core_gate_zero_seeds": int((cg == 0.0).sum()),
                }

    def g(wname: str, arch: str, key: str, sh: int = SHOTS16[0]) -> np.ndarray:
        return np.array([r[key] for r in vec(rows, world=wname, arch=arch, n_shot=sh)], dtype=float)

    # P16-H1: W16_obs_shift lo_shot: modular_transfer beats direct_transfer, scratch_mlp, finetune_mlp (0.03)
    h1 = {
        b: paired_summary(g("W16_obs_shift", "modular_transfer", "target_nrmse"), g("W16_obs_shift", b, "target_nrmse"), 0.03)
        for b in ("direct_transfer", "scratch_mlp", "finetune_mlp")
    }
    out["tests"]["P16-H1"] = {"cells": h1, "met": all(v["met"] for v in h1.values())}

    # P16-H2: W16_shared_plus_new lo_shot: modular_transfer beats scratch_mlp, finetune_mlp, scratch_structural (0.03)
    h2 = {
        b: paired_summary(g("W16_shared_plus_new", "modular_transfer", "target_nrmse"), g("W16_shared_plus_new", b, "target_nrmse"), 0.03)
        for b in ("scratch_mlp", "finetune_mlp", "scratch_structural")
    }
    out["tests"]["P16-H2"] = {"cells": h2, "met": all(v["met"] for v in h2.values())}

    # P16-H3: W16_rule_shift lo_shot: modular_transfer beats direct_transfer, scratch_mlp, finetune_mlp (0.03)
    h3 = {
        b: paired_summary(g("W16_rule_shift", "modular_transfer", "target_nrmse"), g("W16_rule_shift", b, "target_nrmse"), 0.03)
        for b in ("direct_transfer", "scratch_mlp", "finetune_mlp")
    }
    out["tests"]["P16-H3"] = {"cells": h3, "met": all(v["met"] for v in h3.values())}

    # P16-H4: W16_negative_control lo_shot: direct_transfer >= 0.40 (9/10), modular core_gate == 0 (9/10), beat finetune_mlp (0.05)
    dt_fail = int((g("W16_negative_control", "direct_transfer", "target_nrmse") >= 0.40).sum())
    gate_off = int((g("W16_negative_control", "modular_transfer", "core_gate") == 0.0).sum())
    h4_ft = paired_summary(g("W16_negative_control", "modular_transfer", "target_nrmse"), g("W16_negative_control", "finetune_mlp", "target_nrmse"), 0.05)
    out["tests"]["P16-H4"] = {
        "direct_transfer_ge_0.40_seeds": dt_fail,
        "modular_gate_off_seeds": gate_off,
        "vs_finetune_mlp": h4_ft,
        "met": bool(dt_fail >= 9 and gate_off >= 9 and h4_ft["met"]),
    }

    # P16-H5: Source knowledge retention on W16_shared_plus_new & W16_rule_shift
    ret_spn = g("W16_shared_plus_new", "modular_transfer", "source_retention_nrmse")
    ret_rs = g("W16_rule_shift", "modular_transfer", "source_retention_nrmse")
    spn_ok = int((ret_spn < 0.02).sum())
    rs_ok = int((ret_rs < 0.02).sum())
    h5_spn = paired_summary(ret_spn, g("W16_shared_plus_new", "finetune_mlp", "source_retention_nrmse"), 0.05)
    h5_rs = paired_summary(ret_rs, g("W16_rule_shift", "finetune_mlp", "source_retention_nrmse"), 0.05)
    out["tests"]["P16-H5"] = {
        "spn_ret_lt_0.02_seeds": spn_ok,
        "rs_ret_lt_0.02_seeds": rs_ok,
        "spn_vs_finetune": h5_spn,
        "rs_vs_finetune": h5_rs,
        "met": bool(spn_ok >= 9 and rs_ok >= 9 and h5_spn["met"] and h5_rs["met"]),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res["tests"].items():
        print(k, v.get("met", v))
