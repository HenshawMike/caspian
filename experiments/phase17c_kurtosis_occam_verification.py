"""Phase 17C (CSP-P17C-E001): Independent Confirmation of the Three First-Principles Root-Cause Fixes
on Brand-New Never-Seen Disjoint Seeds (151-165, n = 15).

Verifies all 5 Phase 17B hypotheses (P17B-H1 through P17B-H5) without reusing seeds 71-80 or 131-145:
1. Unrotated xm=0 in in-subspace pure rotation test (Root Cause 1 -> P17B-H4).
2. Orthogonal-subspace mean residual guard ||mean(Xu)(I - V V^T)||_2 > 0.10 (Root Cause 2 -> zero false 10-DOF expansion).
3. Phase 13 + Phase 17 Unification: unsupervised kurtosis-unmixed canonical source axes + sub-Gaussian kurtosis
   regularizer lam_kurt=3.0 on G <= 10 6-DOF alignment (Root Cause 3 -> eliminates off-manifold overfitting).
"""
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from experiments.phase16b_independent_harness import ind_paired_stats
from experiments.phase17b_verification import (
    _run_seed_primary,
    holm_bonferroni,
    run_ast_leakage_audit,
    run_invariance_and_determinism_suite,
    run_memory_accounting,
)

SEEDS17C = list(range(151, 166))
INV_SEEDS17C = list(range(151, 161))


def main() -> None:
    print(f"=== [P17C] Independent Confirmation on Fresh Unseen Seeds {SEEDS17C[0]}-{SEEDS17C[-1]} (n={len(SEEDS17C)}) ===", flush=True)
    primary_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for s_rows in ex.map(_run_seed_primary, [(s, 10) for s in SEEDS17C]):
            primary_rows.extend(s_rows)

    def get_v(model: str, key: str) -> np.ndarray:
        sel = [r for r in primary_rows if r["model"] == model]
        sel.sort(key=lambda r: SEEDS17C.index(r["seed"]))
        return np.array([r[key] for r in sel], dtype=float)

    occam_g = get_v("occam_hierarchy_memory", "forward_transfer_gamma")
    occam_ret = get_v("occam_hierarchy_memory", "mean_ret_post4")

    raw_p: Dict[str, float] = {}
    comp_fwd: Dict[str, Any] = {}
    for b in ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"):
        ps = ind_paired_stats(occam_g, get_v(b, "forward_transfer_gamma"))
        comp_fwd[b] = ps
        raw_p[f"fwd_{b}"] = ps["exact_perm_p"]
    holm_fwd = holm_bonferroni(raw_p)

    comp_ret = {
        b: ind_paired_stats(occam_ret, get_v(b, "mean_ret_post4"))
        for b in ("original_v3_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
    }

    inv_det = run_invariance_and_determinism_suite(INV_SEEDS17C)
    ast_audit = run_ast_leakage_audit()
    mem_acct = run_memory_accounting(seed=SEEDS17C[0])

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
        forg_mean < 0.01 and a5_ok >= 14 and retr_acc >= 0.95
        and all(comp_ret[b]["ci_strictly_positive"] for b in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp"))
    )

    iv = inv_det["invariances"]
    h4_met = bool(
        iv["W16_obs_shift"]["shift_detected"] == 10 and iv["W16_obs_shift"]["gate_on"] == 10
        and iv["W16_offset_0.5"]["shift_detected"] == 10 and iv["W16_offset_0.5"]["gate_on"] == 10
        and iv["W16_negative_control"]["shift_detected"] == 0 and iv["W16_negative_control"]["gate_off"] == 10
        and iv["W16_in_subspace_rot"]["shift_detected"] >= 7
    )
    h5_met = bool(inv_det["determinism_max_abs_diff"] == 0.0 and inv_det["row_permutation_max_abs_diff"] < 1e-12 and ast_audit["passed"])

    outcome = (
        "Outcome A — Correction Independently Verified" if (h1_met and h2_met and h3_met and h4_met and h5_met)
        else ("Outcome B — Partial Improvement" if ((h1_met or h2_met) and h3_met and h5_met) else "Outcome C — No Reliable Improvement")
    )

    payload = {
        "experiment_id": "CSP-P17C-E001",
        "seeds": SEEDS17C,
        "outcome_classification": outcome,
        "hypotheses": {
            "P17B-H1": {"met": h1_met, "holm_p": holm_fwd["fwd_original_v3_memory"], **h1_ps},
            "P17B-H2": {"met": h2_met, "holm_p": {k: holm_fwd[f"fwd_{k}"] for k in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")},
                        "cells": {k: comp_fwd[k] for k in ("memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")}},
            "P17B-H3": {"met": h3_met, "mean_forgetting": round(forg_mean, 6), "alpha_post5_lt_0.02": a5_ok, "retrieval_acc": round(retr_acc, 4)},
            "P17B-H4": {"met": h4_met, **iv},
            "P17B-H5": {"met": h5_met, "determinism_diff": inv_det["determinism_max_abs_diff"], "perm_diff": inv_det["row_permutation_max_abs_diff"], "ast_passed": ast_audit["passed"]},
        },
        "primary_10shot_means": {
            m: {
                "forward_transfer_gamma": round(float(get_v(m, "forward_transfer_gamma").mean()), 5),
                "ret_gamma_post4": round(float(get_v(m, "ret_gamma_post4").mean()), 5),
                "mean_ret_post4": round(float(get_v(m, "mean_ret_post4").mean()), 5),
            }
            for m in ("original_v3_memory", "occam_hierarchy_memory", "memoryless_structural", "naive_sequential_mlp", "replay_mlp", "ewc_mlp")
        },
        "memory_accounting": mem_acct,
    }
    Path("logs/phase17c").mkdir(parents=True, exist_ok=True)
    with open("logs/phase17c/p17c_kurtosis_occam_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("OUTCOME:", outcome, flush=True)
    for k, v in payload["hypotheses"].items():
        print(f"  {k}: met={v['met']}", flush=True)
    print("Primary 10-shot means (seeds 151-165):", json.dumps(payload["primary_10shot_means"], indent=2), flush=True)
    print("P17B-H4 invariances (seeds 151-160):", json.dumps(iv, indent=2), flush=True)


if __name__ == "__main__":
    main()
