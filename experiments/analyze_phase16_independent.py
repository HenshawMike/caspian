"""Independent Confirmatory Analysis for Phase 16 v2 (CSP-P16-E002, disjoint seeds 61-70)."""
import json
from typing import Any, Dict, List
import numpy as np

from experiments.phase11_experiments import bootstrap_ci, paired_summary
from experiments.phase16_experiments import ARCHS16, SEEDS16_INDEPENDENT, SHOTS16, TARGET_WORLDS16


def _vec(rows: List[Dict[str, Any]], seeds=SEEDS16_INDEPENDENT, **kw) -> List[Dict[str, Any]]:
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    sel.sort(key=lambda r: seeds.index(r["seed"]))
    assert len(sel) == len(seeds), (kw, len(sel))
    return sel


def paired_ci_gt_zero(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Paired improvement (b - a) where lower is better; met if bootstrap 95% CI lower bound > 0."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    lo, hi = bootstrap_ci(diff)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    dz = float(diff.mean() / sd) if sd > 1e-12 else 0.0
    return {
        "mean_diff": round(float(diff.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "d_z": round(dz, 4),
        "wins": int((diff > 0.0).sum()),
        "n": int(len(diff)),
        "ci_gt_zero": bool(lo > 0.0),
    }


def main(
    matrix_path: str = "logs/phase16/independent_v2_matrix.json",
    out_path: str = "logs/phase16/independent_v2_analysis.json",
) -> Dict[str, Any]:
    rows = json.load(open(matrix_path))
    out: Dict[str, Any] = {
        "experiment_id": "CSP-P16-E002",
        "seeds": list(SEEDS16_INDEPENDENT),
        "autonomous_shift_detection": True,
        "exclusions": [r for r in rows if not r["finite"]],
        "leak_rows_total": int(sum(r["leak_rows"] for r in rows)),
        "worlds": {},
        "original_tests": {},
        "independent_tests": {},
    }

    for spec in TARGET_WORLDS16:
        wname = spec.name
        out["worlds"][wname] = {}
        for lv, sh in zip(("lo_shot", "hi_shot"), SHOTS16):
            for a in ARCHS16:
                s = _vec(rows, world=wname, arch=a, n_shot=sh)
                t = np.array([r["target_nrmse"] for r in s], dtype=float)
                ret = np.array([r["source_retention_nrmse"] for r in s], dtype=float)
                cg = np.array([r["core_gate"] for r in s], dtype=float)
                sh_det = np.array([r.get("sensor_shift_detected", False) for r in s], dtype=bool)
                lo_ci, hi_ci = bootstrap_ci(t)
                out["worlds"][wname][f"{lv}|{a}"] = {
                    "n_params": s[0]["n_params"],
                    "target_mean": round(float(t.mean()), 5),
                    "target_sd": round(float(t.std(ddof=1)), 5),
                    "target_ci95": [round(float(lo_ci), 5), round(float(hi_ci), 5)],
                    "target_per_seed": t.round(5).tolist(),
                    "retention_mean": round(float(ret.mean()), 5),
                    "retention_max": round(float(ret.max()), 5),
                    "core_gate_on_seeds": int((cg == 1.0).sum()),
                    "core_gate_zero_seeds": int((cg == 0.0).sum()),
                    "shift_detected_seeds": int(sh_det.sum()),
                }

    def g(wname: str, arch: str, key: str, sh: int = SHOTS16[0]) -> np.ndarray:
        return np.array([r[key] for r in _vec(rows, world=wname, arch=arch, n_shot=sh)], dtype=float)

    # --- 1. Exact original CSP-P16-E001 tests (for transparent continuity) ---
    h1 = {b: paired_summary(g("W16_obs_shift", "modular_transfer", "target_nrmse"), g("W16_obs_shift", b, "target_nrmse"), 0.03)
          for b in ("direct_transfer", "scratch_mlp", "finetune_mlp")}
    out["original_tests"]["P16-H1"] = {"cells": h1, "met": all(v["met"] for v in h1.values())}

    h2 = {b: paired_summary(g("W16_shared_plus_new", "modular_transfer", "target_nrmse"), g("W16_shared_plus_new", b, "target_nrmse"), 0.03)
          for b in ("scratch_mlp", "finetune_mlp", "scratch_structural")}
    out["original_tests"]["P16-H2"] = {"cells": h2, "met": all(v["met"] for v in h2.values())}

    h3 = {b: paired_summary(g("W16_rule_shift", "modular_transfer", "target_nrmse"), g("W16_rule_shift", b, "target_nrmse"), 0.03)
          for b in ("direct_transfer", "scratch_mlp", "finetune_mlp")}
    out["original_tests"]["P16-H3"] = {"cells": h3, "met": all(v["met"] for v in h3.values())}

    dt_fail = int((g("W16_negative_control", "direct_transfer", "target_nrmse") >= 0.40).sum())
    gate_off = int((g("W16_negative_control", "modular_transfer", "core_gate") == 0.0).sum())
    h4_ft = paired_summary(g("W16_negative_control", "modular_transfer", "target_nrmse"), g("W16_negative_control", "finetune_mlp", "target_nrmse"), 0.05)
    out["original_tests"]["P16-H4"] = {
        "direct_transfer_ge_0.40_seeds": dt_fail,
        "modular_gate_off_seeds": gate_off,
        "vs_finetune_mlp": h4_ft,
        "met": bool(dt_fail >= 9 and gate_off >= 9 and h4_ft["met"]),
    }

    ret_spn = g("W16_shared_plus_new", "modular_transfer", "source_retention_nrmse")
    ret_rs = g("W16_rule_shift", "modular_transfer", "source_retention_nrmse")
    spn_ok = int((ret_spn < 0.02).sum())
    rs_ok = int((ret_rs < 0.02).sum())
    h5_spn = paired_summary(ret_spn, g("W16_shared_plus_new", "finetune_mlp", "source_retention_nrmse"), 0.05)
    h5_rs = paired_summary(ret_rs, g("W16_rule_shift", "finetune_mlp", "source_retention_nrmse"), 0.05)
    out["original_tests"]["P16-H5"] = {
        "spn_ret_lt_0.02_seeds": spn_ok,
        "rs_ret_lt_0.02_seeds": rs_ok,
        "spn_vs_finetune": h5_spn,
        "rs_vs_finetune": h5_rs,
        "met": bool(spn_ok >= 9 and rs_ok >= 9 and h5_spn["met"] and h5_rs["met"]),
    }

    # --- 2. Pre-registered Independent Confirmatory Tests (P16-IND-1..5) ---
    obs_mt = g("W16_obs_shift", "modular_transfer", "target_nrmse")
    obs_shift_det = int(g("W16_obs_shift", "modular_transfer", "sensor_shift_detected").sum())
    obs_gate_on = int((g("W16_obs_shift", "modular_transfer", "core_gate") == 1.0).sum())
    ind1_cells = {b: paired_ci_gt_zero(obs_mt, g("W16_obs_shift", b, "target_nrmse"))
                  for b in ("direct_transfer", "scratch_mlp", "finetune_mlp", "scratch_structural")}
    out["independent_tests"]["P16-IND-1"] = {
        "shift_detected_seeds": obs_shift_det,
        "core_gate_on_seeds": obs_gate_on,
        "mean_target_nrmse": round(float(obs_mt.mean()), 5),
        "cells": ind1_cells,
        "met": bool(obs_shift_det == 10 and obs_gate_on >= 9 and float(obs_mt.mean()) < 0.025 and all(v["ci_gt_zero"] for v in ind1_cells.values())),
    }

    spn_mt = g("W16_shared_plus_new", "modular_transfer", "target_nrmse")
    spn_shift_fp = int(g("W16_shared_plus_new", "modular_transfer", "sensor_shift_detected").sum())
    spn_gate_on = int((g("W16_shared_plus_new", "modular_transfer", "core_gate") == 1.0).sum())
    ind2_cells = {b: paired_ci_gt_zero(spn_mt, g("W16_shared_plus_new", b, "target_nrmse"))
                  for b in ("scratch_mlp", "finetune_mlp", "scratch_structural")}
    out["independent_tests"]["P16-IND-2"] = {
        "false_shift_triggers": spn_shift_fp,
        "core_gate_on_seeds": spn_gate_on,
        "mean_target_nrmse": round(float(spn_mt.mean()), 5),
        "cells": ind2_cells,
        "met": bool(
            spn_shift_fp == 0
            and spn_gate_on >= 8
            and ind2_cells["scratch_mlp"]["ci_gt_zero"]
            and ind2_cells["scratch_structural"]["ci_gt_zero"]
            and ind2_cells["finetune_mlp"]["mean_diff"] > 0.0
        ),
    }

    rs_mt = g("W16_rule_shift", "modular_transfer", "target_nrmse")
    rs_shift_fp = int(g("W16_rule_shift", "modular_transfer", "sensor_shift_detected").sum())
    ind3_cells = {b: paired_ci_gt_zero(rs_mt, g("W16_rule_shift", b, "target_nrmse"))
                  for b in ("direct_transfer", "scratch_mlp", "finetune_mlp")}
    out["independent_tests"]["P16-IND-3"] = {
        "false_shift_triggers": rs_shift_fp,
        "mean_target_nrmse": round(float(rs_mt.mean()), 5),
        "cells": ind3_cells,
        "met": bool(rs_shift_fp == 0 and all(v["ci_gt_zero"] for v in ind3_cells.values())),
    }

    neg_mt = g("W16_negative_control", "modular_transfer", "target_nrmse")
    neg_shift_fp = int(g("W16_negative_control", "modular_transfer", "sensor_shift_detected").sum())
    ind4_ft = paired_ci_gt_zero(neg_mt, g("W16_negative_control", "finetune_mlp", "target_nrmse"))
    out["independent_tests"]["P16-IND-4"] = {
        "false_shift_triggers": neg_shift_fp,
        "direct_transfer_ge_0.40_seeds": dt_fail,
        "modular_gate_off_seeds": gate_off,
        "vs_finetune_mlp": ind4_ft,
        "met": bool(neg_shift_fp == 0 and dt_fail >= 9 and gate_off >= 9 and ind4_ft["ci_gt_zero"]),
    }

    out["independent_tests"]["P16-IND-5"] = out["original_tests"]["P16-H5"]

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    print("=== Original P16-H1..H5 on Independent Seeds 61-70 ===")
    for k, v in res["original_tests"].items():
        print(f"  {k}: {v['met']}")
    print("=== Pre-Registered Independent P16-IND-1..5 on Seeds 61-70 ===")
    for k, v in res["independent_tests"].items():
        print(f"  {k}: {v['met']}")
