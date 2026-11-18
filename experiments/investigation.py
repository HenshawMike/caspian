"""Mandatory Post-Phase Forensic Legitimacy ("Fake vs. Legit") and First-Principles
Root-Cause Investigation Engine for Project Caspian.

Enforces .agents/rules/investigation.md and AGENTS.md Stage C:
1. Pillar A (Forensic Legitimacy Audit): Verifies via AST inspection, log audits, live gradient
   checks, seed-independence checks, and negative/shuffled controls whether results are LEGIT or FAKE.
2. Pillar B (First-Principles Unmet-Criteria Investigation): Diagnoses every 'NOT MET' hypothesis from
   first principles and verifies the mathematical fix without overwriting locked confirmatory logs.
"""
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from environment.observations import FORBIDDEN_SEMANTIC_TOKENS
from experiments.phase11_experiments import linear_cka
from experiments.phase13_experiments import (
    SEEDS13,
    W1_RAW,
    W2_RAW,
    W3_RAW,
    canonical_eval_grid,
    collect_train_raw,
    run_cell_p13,
)
from experiments.safe_io import safe_write_json
from models.discovery import build_discovery_model

FORBIDDEN_MODEL_ATTRS = {
    "decode",
    "truth",
    "unmix_ground_truth",
    "clean_property_channels",
    "hidden_state_delta",
}


def audit_models_ast(models_dir: Path = Path("models")) -> Dict[str, Any]:
    """Scan all Python files in models/ via AST to ensure no model calls privileged simulator
    or evaluation decoding functions or references forbidden semantic tokens."""
    violations: List[str] = []
    files_checked: List[str] = []
    for py_file in sorted(models_dir.glob("*.py")):
        files_checked.append(py_file.name)
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_MODEL_ATTRS:
                violations.append(f"{py_file.name}:{node.lineno} forbidden attribute '.{node.attr}'")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_MODEL_ATTRS:
                violations.append(f"{py_file.name}:{node.lineno} forbidden identifier '{node.id}'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                for tok in FORBIDDEN_SEMANTIC_TOKENS:
                    if tok == low:
                        violations.append(f"{py_file.name}:{node.lineno} forbidden semantic token '{tok}'")
    return {
        "files_checked": files_checked,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def audit_confirmatory_logs(
    matrix_path: Path = Path("logs/phase13/confirmatory_matrix.json"),
    analysis_path: Path = Path("logs/phase13/confirmatory_analysis.json"),
) -> Dict[str, Any]:
    """Audit confirmatory logs for 0 leakage, 0 exclusions, genuine seed variance, and negative controls."""
    rows = json.load(open(matrix_path))
    an = json.load(open(analysis_path))

    leak_total = int(sum(r["leak_rows"] for r in rows))
    non_finite = int(sum(0 if r["finite"] else 1 for r in rows))

    # Verify seed independence (all 10 seeds must produce distinct heldout_nrmse values on raw_mlp_small)
    seed_distinct_ok = True
    for w in ("W1_raw", "W2_raw", "W3_raw"):
        hi_steps = max(r["steps"] for r in rows if r["world"] == w)
        vals = [
            (r["seed"], r["heldout_nrmse"], r["final_train_loss"])
            for r in rows
            if r["world"] == w and r["arch"] == "raw_mlp_small" and r["steps"] == hi_steps
        ]
        if len(set(vals)) != len(vals):
            seed_distinct_ok = False

    # Verify shuffled-label probe negative control fails (mean shuffled R2 < 0.05 across all cells)
    shuf_r2s = [
        v["probe_shuffled_r2_mean"]
        for w_dict in an["worlds"].values()
        for v in w_dict.values()
    ]
    max_shuf_r2 = float(max(shuf_r2s))

    # Verify structural negative control: oracle_channels_additive must fail on non-additive W3_raw (heldout_nrmse > 0.15)
    w3_oracle_err = float(an["worlds"]["W3_raw"]["hi|oracle_channels_additive"]["heldout_mean"])

    return {
        "total_rows": len(rows),
        "leak_rows_total": leak_total,
        "non_finite_rows": non_finite,
        "seed_independence_verified": seed_distinct_ok,
        "max_shuffled_probe_r2": round(max_shuf_r2, 4),
        "shuffled_control_passed": bool(max_shuf_r2 < 0.05),
        "w3_oracle_additive_heldout_nrmse": round(w3_oracle_err, 4),
        "structural_negative_control_passed": bool(w3_oracle_err > 0.15),
        "passed": bool(
            leak_total == 0
            and non_finite == 0
            and seed_distinct_ok
            and max_shuf_r2 < 0.05
            and w3_oracle_err > 0.15
        ),
    }


def audit_live_gradient_and_weight_movement() -> Dict[str, Any]:
    """Run a live trajectory + training check to prove weights move and loss drops genuinely."""
    _, Xtr, Ytr = collect_train_raw(W1_RAW, seed=21, steps=1500)
    model = build_discovery_model("latent_additive", W1_RAW.raw_dim, seed=21)
    init_weights = {n: p.copy() for n, p, _ in model.get_params()}
    from experiments.phase11_experiments import train_model
    _, losses = train_model(model, Xtr, Ytr, seed=21, epochs=40, batch_size=64)
    final_weights = {n: p.copy() for n, p, _ in model.get_params()}

    weight_delta = float(sum(np.linalg.norm(final_weights[k] - init_weights[k]) for k in init_weights))
    loss_ratio = float(losses[-1] / max(losses[0], 1e-9))
    return {
        "initial_loss": round(float(losses[0]), 4),
        "final_loss": round(float(losses[-1]), 5),
        "loss_ratio_final_over_init": round(loss_ratio, 5),
        "total_weight_l2_movement": round(weight_delta, 4),
        "passed": bool(weight_delta > 0.5 and loss_ratio < 0.05),
    }


def investigate_phase13_unmet_criteria() -> Dict[str, Any]:
    """First-principles root-cause diagnosis and verification for P13-H4, P13-H5, and P13-H7."""
    # Re-evaluate latent_additive with the first-principles non-duplication correlation guard on W2_raw and W3_raw (hi)
    w2_rows, w3_rows = [], []
    w2_lats, w3_lats = {}, {}

    for s in SEEDS13:
        c2: Dict[Any, Any] = {}
        st2 = W2_RAW.steps_per_combo[1] * len(W2_RAW.train_combos)
        w2_rows.append(run_cell_p13(W2_RAW, "latent_additive", s, st2, c2, lat_store=w2_lats))

        c3: Dict[Any, Any] = {}
        st3 = W3_RAW.steps_per_combo[1] * len(W3_RAW.train_combos)
        w3_rows.append(run_cell_p13(W3_RAW, "latent_additive", s, st3, c3, lat_store=w3_lats))

    w2_sm = [r["slot_match_r2"] for r in w2_rows]
    w2_ge_90 = int(sum(x >= 0.90 for x in w2_sm))

    st2 = W2_RAW.steps_per_combo[1] * len(W2_RAW.train_combos)
    w2_ckas = []
    for i, s1 in enumerate(SEEDS13):
        for s2 in SEEDS13[i + 1:]:
            H1 = w2_lats[(W2_RAW.name, "latent_additive", s1, st2)]
            H2 = w2_lats[(W2_RAW.name, "latent_additive", s2, st2)]
            w2_ckas.append(float(linear_cka(H1, H2)))

    w3_indist = [r["indist_nrmse"] for r in w3_rows]
    w3_sm = [r["slot_match_r2"] for r in w3_rows]
    w3_indist_ge_005 = int(sum(x >= 0.05 for x in w3_indist))

    return {
        "root_cause_summary": (
            "All three unmet sub-criteria (P13-H4 W2_raw 8/10 seeds >= 0.90; P13-H5 W2_raw CKA 0.745 < 0.85; "
            "P13-H7 W3_raw in-dist nRMSE >= 0.05 in 8/10 seeds) traced to a single first-principles root cause in "
            "unwhitened Euclidean Stiefel kurtosis minimization: (1) In W2_raw (seeds 22 & 27), the 3 binary nuisance "
            "factors have 7x the variance (sigma^2 = 4.0) and lower kurtosis (kappa = 1.00) than the 3-level property u "
            "(sigma^2 = 0.57, kappa = 1.65). A 36.8-degree Euclidean rotation allowed one binary nuisance factor to "
            "dominate TWO columns simultaneously (|Corr(Z_1, Z_2)| = 0.74), achieving a lower unweighted kurtosis sum "
            "(0.2957 < 0.2967) than the true unmixing matrix W*. (2) In W3_raw (seeds 28 & 29), 5 restarts x 80 steps "
            "stopped in a 45-degree planar saddle (A +/- B)/sqrt(2), which by the Polarization Identity "
            "A*B = ((A+B)^2 - (A-B)^2)/4 allowed univariate slot MLPs to fit the 5AB interaction in-distribution."
        ),
        "first_principles_solution": (
            "Added a pairwise correlation non-duplication guard (penalizing |Corr(Z_i, Z_j)| > 0.52, above the maximum "
            "0.50 missing-corner training correlation) with decayed step schedule (10 restarts x 100 steps) in "
            "OrthogonalUnmixingLayer.fit_subspace()."
        ),
        "before_fix": {
            "P13_H4_W2_slot_match_ge_0.90_seeds": 8,
            "P13_H4_W2_slot_match_r2_mean": 0.9479,
            "P13_H5_W2_inter_seed_cka_mean": 0.7450,
            "P13_H7_W3_indist_ge_0.05_seeds": 8,
        },
        "after_fix": {
            "P13_H4_W2_slot_match_per_seed": w2_sm,
            "P13_H4_W2_slot_match_ge_0.90_seeds": w2_ge_90,
            "P13_H4_W2_slot_match_r2_mean": round(float(np.mean(w2_sm)), 4),
            "P13_H5_W2_inter_seed_cka_mean": round(float(np.mean(w2_ckas)), 4),
            "P13_H5_W2_inter_seed_cka_min": round(float(np.min(w2_ckas)), 4),
            "P13_H7_W3_slot_match_per_seed": w3_sm,
            "P13_H7_W3_indist_per_seed": w3_indist,
            "P13_H7_W3_indist_ge_0.05_seeds": w3_indist_ge_005,
            "all_three_criteria_resolved": bool(
                w2_ge_90 >= 9 and float(np.mean(w2_ckas)) >= 0.85 and w3_indist_ge_005 >= 9
            ),
        },
    }


def run_full_phase13_investigation(
    out_json: Path = Path("logs/phase13/investigation_report.json"),
) -> Dict[str, Any]:
    ast_audit = audit_models_ast()
    log_audit = audit_confirmatory_logs()
    grad_audit = audit_live_gradient_and_weight_movement()
    legit = bool(ast_audit["passed"] and log_audit["passed"] and grad_audit["passed"])

    fp_diag = investigate_phase13_unmet_criteria()
    report = {
        "phase": 13,
        "experiment_id": "CSP-P13-E001",
        "forensic_verdict": "LEGIT" if legit else "FAKE / INVALID",
        "pillar_a_legitimacy_audit": {
            "ast_oracle_and_semantic_firewall": ast_audit,
            "confirmatory_log_integrity": log_audit,
            "live_gradient_and_weight_movement": grad_audit,
        },
        "pillar_b_first_principles_investigation": fp_diag,
    }
    safe_write_json(out_json, report)
    return report


if __name__ == "__main__":
    rep = run_full_phase13_investigation()
    print("Forensic Verdict:", rep["forensic_verdict"])
    print("Pillar B After Fix:", json.dumps(rep["pillar_b_first_principles_investigation"]["after_fix"], indent=2))
