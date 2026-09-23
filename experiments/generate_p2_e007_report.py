"""Master pipeline for CSP-P2-E007 diagnostic follow-up experiments.

Execution order (per contract):
  1. Run automated test suite (full regression)
  2. Run E007a — early-stopped re-benchmark
  3. Run E007b — parameter-matched MLP control (reuses E007a model state)
  4. Run E007c — representation-metric sanity (reuses E007a model state)
  5. Evaluate decide_part_b() — OR logic, dict return
  6. Conditionally run E007d/e if trigger_part_b is True
  7. Generate publication plots
  8. Populate experiment artifact directories
  9. Save consolidated JSON
  10. Compile DOCX reports (individual + master)
"""

import csv
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase2_e007_experiments import (
    run_experiment_p2_e007a,
    run_experiment_p2_e007b,
    run_experiment_p2_e007c,
    decide_part_b,
    run_experiment_p2_e007d,
    run_experiment_p2_e007e,
    ORIGINAL_E001_NUMBERS,
)
from experiments.report_generator import (
    generate_p2_e007_individual_docx_report,
    generate_p2_e007_master_docx_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs" / "experiments"


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_e007_plots(all_results: Dict[str, Any], plot_dir: str) -> Dict[str, str]:
    """Generate side-by-side comparison plots for E007 report."""
    os.makedirs(plot_dir, exist_ok=True)
    plot_paths = {}
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)

    e007a = all_results.get("CSP-P2-E007a", {})
    e007b = all_results.get("CSP-P2-E007b", {})
    cmp = e007a.get("comparison_vs_original", {})

    # --- Plot 1: Before/After MSE comparison bar chart ---
    orig_mem = ORIGINAL_E001_NUMBERS["memory_model_mse"]
    orig_no_mem = ORIGINAL_E001_NUMBERS["no_memory_model_mse"]
    corr_mem = cmp.get("corrected_memory_mse", orig_mem)
    corr_no_mem = cmp.get("corrected_no_memory_mse", orig_no_mem)
    bench3 = e007b.get("three_way_benchmark", {})
    matched_mse = bench3.get("matched_mlp_173", {}).get("mse", orig_no_mem)

    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    orig_vals = [orig_mem, orig_no_mem, orig_no_mem]
    corr_vals = [corr_mem, corr_no_mem, matched_mse]
    labels = ["Memory (GRU)", "No-Memory (Small MLP)", "No-Memory (Matched MLP)"]
    bars_orig = ax.bar(x - width / 2, orig_vals, width, label="Original (CSP-P2-E001)", color="#A0B4C8", edgecolor="black")
    bars_corr = ax.bar(x + width / 2, corr_vals, width, label="Corrected (E007a/b)", color="#107C41", edgecolor="black")
    ax.set_ylabel("Test MSE", fontsize=11, fontweight="bold")
    ax.set_title("Before/After MSE: Early-Stopping & Capacity-Matching Fixes (E007)", fontsize=12, fontweight="bold", color="#182B49")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=True, facecolor="white")
    for bar in bars_orig:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_corr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(plot_dir, "e007_mse_before_after.png")
    plt.savefig(p1)
    plt.close()
    plot_paths["Before/After MSE Comparison (E007)"] = p1

    # --- Plot 2: Delay sweep — original vs corrected ---
    sweep = e007a.get("delay_sweep", [])
    if sweep:
        delays = [r["delay"] for r in sweep]
        mem_mses = [r["memory_model_mse"] for r in sweep]
        no_mem_mses = [r["no_memory_model_mse"] for r in sweep]
        persist_mses = [r["persistence_baseline_mse"] for r in sweep]

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=180)
        ax.plot(delays, mem_mses, marker="o", color="#107C41", linewidth=2.2, label="Memory (GRU) — Corrected")
        ax.plot(delays, no_mem_mses, marker="s", color="#D83B01", linewidth=2.0, linestyle="--", label="No-Memory (Small MLP) — Corrected")
        ax.plot(delays, persist_mses, marker="^", color="#0078D4", linewidth=1.8, linestyle=":", label="Persistence Baseline")
        ax.set_xlabel("Temporal Delay (d timesteps)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Prediction MSE", fontsize=11, fontweight="bold")
        ax.set_title("Delay Sweep After Early-Stopping Fix (E007a)", fontsize=12, fontweight="bold", color="#182B49")
        ax.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        p2 = os.path.join(plot_dir, "e007_delay_sweep_corrected.png")
        plt.savefig(p2)
        plt.close()
        plot_paths["Delay Sweep — Corrected Models (E007a)"] = p2

    # --- Plot 3: Generalization test before/after ---
    gen = e007a.get("generalization_results", {})
    if gen:
        g_labels = [f"d={v['delay']}" for v in gen.values()]
        g_mem_mses = [v["mse"] for v in gen.values()]
        g_persist_mses = [v["persistence_mse"] for v in gen.values()]

        x = np.arange(len(g_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
        ax.bar(x - width / 2, g_mem_mses, width, label="Memory (Corrected)", color="#107C41", edgecolor="black")
        ax.bar(x + width / 2, g_persist_mses, width, label="Persistence Baseline", color="#0078D4", edgecolor="black")
        ax.set_ylabel("Prediction MSE", fontsize=11, fontweight="bold")
        ax.set_title("Generalization to Unseen Delays — Corrected Model (E007a)", fontsize=11, fontweight="bold", color="#182B49")
        ax.set_xticks(x)
        ax.set_xticklabels(g_labels)
        ax.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        p3 = os.path.join(plot_dir, "e007_generalization_corrected.png")
        plt.savefig(p3)
        plt.close()
        plot_paths["Generalization — Corrected Models (E007a)"] = p3

    return plot_paths


# ---------------------------------------------------------------------------
# Artifact directory population
# ---------------------------------------------------------------------------

def populate_e007_artifact_dirs(
    results_by_id: Dict[str, Any],
    plot_paths: Dict[str, str],
) -> None:
    """Create and populate experiments/CSP-P2-E007X/ directories."""
    for exp_id, exp_data in results_by_id.items():
        exp_folder = EXPERIMENTS_DIR / exp_id
        for sub in ["logs", "checkpoints", "plots"]:
            (exp_folder / sub).mkdir(parents=True, exist_ok=True)

        # config.yaml
        with open(exp_folder / "config.yaml", "w", encoding="utf-8") as f:
            f.write(
                f"experiment_id: {exp_id}\n"
                f"title: \"{exp_data.get('title', exp_id)}\"\n"
                f"phase: \"Phase 2 — Memory & Persistence (Diagnostic)\"\n"
                f"seed: {exp_data.get('seed', 42)}\n"
                f"semantic_firewall: \"ACTIVE\"\n"
                f"part_b_logic: \"OR (d0_persists OR gen_persists OR capacity_gap_material)\"\n"
            )

        # README.md
        clean = {k: v for k, v in exp_data.items() if not k.startswith("_")}
        with open(exp_folder / "README.md", "w", encoding="utf-8") as f:
            f.write(
                f"# {exp_id} — {exp_data.get('title', exp_id)}\n\n"
                f"## Phase\nPhase 2 Diagnostic Follow-Up (CSP-P2-E007)\n\n"
                f"## Summary Results\n```json\n{json.dumps(clean, indent=2, default=str)}\n```\n"
            )

        # commands.txt
        with open(exp_folder / "commands.txt", "w", encoding="utf-8") as f:
            f.write(
                f"# Execution commands for {exp_id}\n"
                f"python3 -m unittest tests/test_e007_experiments.py\n"
                f"python3 experiments/generate_p2_e007_report.py\n"
            )

        # results.json
        with open(exp_folder / "results.json", "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)

        # metrics.csv
        with open(exp_folder / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_key", "metric_value"])
            if "comparison_vs_original" in exp_data:
                for k, v in exp_data["comparison_vs_original"].items():
                    writer.writerow([k, v])
            if "gaps" in exp_data:
                for k, v in exp_data["gaps"].items():
                    writer.writerow([k, v])

        # Copy relevant plots
        for p_name, p_file in plot_paths.items():
            if exp_id.lower() in p_file.lower() or exp_id in p_name:
                target = exp_folder / "plots" / os.path.basename(p_file)
                if os.path.exists(p_file) and not target.exists():
                    target.write_bytes(Path(p_file).read_bytes())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_e007_pipeline() -> None:
    print("=" * 60)
    print("PROJECT CASPIAN — CSP-P2-E007 DIAGNOSTIC PIPELINE")
    print("=" * 60)

    # 1. Full regression test suite
    print("\n[Step 1/9] Running full automated test suite (regression)...")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=0)
    test_res = runner.run(suite)
    n_pass = test_res.testsRun - len(test_res.failures) - len(test_res.errors)
    print(f"Tests passed: {n_pass}/{test_res.testsRun}")
    if test_res.failures or test_res.errors:
        print("ERROR: Test failures detected — aborting pipeline.")
        sys.exit(1)

    # 2. E007a — early-stopped re-benchmark
    print("\n[Step 2/9] Running CSP-P2-E007a: Early-Stopped Re-Benchmark...")
    e007a = run_experiment_p2_e007a(
        delay=2,
        num_train_episodes=30,
        num_test_episodes=10,
        steps_per_episode=25,
        seed=42,
        early_stopping_patience=30,
    )
    cmp = e007a["comparison_vs_original"]
    print(f"  Memory MSE:    {cmp['corrected_memory_mse']:.6f}  (original: {cmp['original_memory_mse']:.6f})")
    print(f"  No-Mem MSE:    {cmp['corrected_no_memory_mse']:.6f}  (original: {cmp['original_no_memory_mse']:.6f})")
    print(f"  Gap %:         {cmp['corrected_gap_pct']:.1f}%  (original: {cmp['original_gap_pct']:.1f}%)")
    print(f"  Memory best_epoch:   {e007a['memory_model']['best_epoch']}")
    print(f"  No-Mem best_epoch:   {e007a['no_memory_model']['best_epoch']}")
    print(f"  d=0 anomaly persists after ES: {e007a['d0_anomaly_persists_after_early_stopping']}")
    print(f"  Gen failure persists after ES: {e007a['generalization_failure_persists_after_early_stopping']}")

    # 3. E007b — parameter-matched MLP (pass E007a state to avoid retraining)
    print("\n[Step 3/9] Running CSP-P2-E007b: Parameter-Matched MLP Control...")
    e007b = run_experiment_p2_e007b(
        delay=2,
        num_train_episodes=30,
        num_test_episodes=10,
        steps_per_episode=25,
        seed=42,
        early_stopping_patience=30,
        e007a_results=e007a,
    )
    arch = e007b["matched_mlp_architecture"]
    gaps = e007b["gaps"]
    cap = e007b["capacity_analysis"]
    print(f"  Matched MLP: hidden_dims=(173,) → {arch['num_parameters']} params (±{arch['parameter_match_pct_error']:.2f}% from 6753)")
    print(f"  Memory vs Matched-MLP gap: {gaps['memory_vs_matched_mlp_pct']:.1f}%")
    print(f"  Memory vs Small-MLP gap:   {gaps['memory_vs_small_mlp_pct']:.1f}%")
    print(f"  Fraction of gap surviving capacity match: {cap['fraction_of_gap_surviving_capacity_match']:.4f}")
    print(f"  Capacity interpretation: {cap['interpretation']}")

    # 4. E007c — representation sanity
    print("\n[Step 4/9] Running CSP-P2-E007c: Representation-Metric Sanity Check...")
    e007c = run_experiment_p2_e007c(seed=42, e007a_results=e007a)
    ra = e007c["representation_analysis"]
    print(f"  Corrected cosine_sim: {ra['inter_class_cosine_similarity']:.6f}")
    print(f"  Threshold: {ra['cosine_similarity_threshold']}  Margin: {ra['cosine_sim_margin_from_threshold']:+.6f}")
    print(f"  distinct_flag: {ra['distinct_memory_representations_formed']}  confidence: {ra['representation_distinctness_confidence']}")

    # 5. decide_part_b — single source of truth
    print("\n[Step 5/9] Evaluating Part B decision point...")
    decision = decide_part_b(e007a, e007b)
    print(f"  trigger_part_b:               {decision['trigger_part_b']}")
    print(f"  d0_anomaly_persists:          {decision['d0_anomaly_persists']}")
    print(f"  generalization_persists:      {decision['generalization_failure_persists']}")
    print(f"  capacity_gap_material:        {decision['capacity_gap_material']} ({decision['capacity_gap_pct']:.1f}%)")
    print(f"  Rationale: {decision['rationale'][:120]}...")

    # 6. Conditionally run Part B
    e007d, e007e = None, None
    if decision["trigger_part_b"]:
        print("\n[Step 6/9] PART B TRIGGERED — Running E007d/e hidden-state diagnostics...")
        print("  Running CSP-P2-E007d: d=0 Isolation...")
        e007d = run_experiment_p2_e007d(seed=42, e007a_results=e007a)
        print(f"  d=0 hidden norm: {e007d['hidden_state_analysis']['d0_mean_hidden_norm']:.4f}  "
              f"d=2 norm: {e007d['hidden_state_analysis']['d2_mean_hidden_norm']:.4f}")
        print("  Running CSP-P2-E007e: Generalization Failure Diagnostic...")
        e007e = run_experiment_p2_e007e(seed=42, e007a_results=e007a)
        print(f"  Encoding hypothesis: {e007e['encoding_hypothesis']}")
        print(f"  Delay-norm rank correlation: {e007e['delay_norm_rank_correlation']:.4f}")
    else:
        print("\n[Step 6/9] Part B NOT triggered — skipping E007d/e.")

    # 7. Generate plots
    print("\n[Step 7/9] Generating publication plots...")
    plot_dir = str(LOGS_DIR / "plots_e007")
    all_e007_results = {
        "CSP-P2-E007a": e007a,
        "CSP-P2-E007b": e007b,
        "CSP-P2-E007c": e007c,
        "part_b_decision": decision,
    }
    if e007d:
        all_e007_results["CSP-P2-E007d"] = e007d
    if e007e:
        all_e007_results["CSP-P2-E007e"] = e007e

    plot_paths = generate_e007_plots(all_e007_results, plot_dir)
    for name, path in plot_paths.items():
        print(f"  - Generated: {path}")

    # 8. Populate artifact directories
    print("\n[Step 8/9] Populating experiment artifact directories...")
    artifact_results = {k: v for k, v in all_e007_results.items()
                        if k.startswith("CSP-P2-")}
    populate_e007_artifact_dirs(artifact_results, plot_paths)
    print(f"  - Populated: {', '.join(artifact_results.keys())}")

    # 9. Save JSON + compile DOCX
    print("\n[Step 9/9] Saving JSON and compiling DOCX reports...")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOGS_DIR / "phase2_e007_full_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_e007_results, f, indent=2, default=str)
    print(f"  - Saved: {json_path}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Individual sub-experiment DOCX reports
    for sub_id, sub_data in artifact_results.items():
        out_doc = DOCS_DIR / f"{sub_id}.docx"
        matched_plot = next(
            (p for name, p in plot_paths.items() if sub_id.lower() in p.lower() or sub_id in name),
            None,
        )
        generate_p2_e007_individual_docx_report(
            sub_id=sub_id,
            data=sub_data,
            plot_path=matched_plot,
            output_path=str(out_doc),
        )
        print(f"  - Generated: {out_doc}")

    # Master consolidated report
    master_path = PROJECT_ROOT / "docs" / "CSP-P2-E007.docx"
    doc_file = generate_p2_e007_master_docx_report(
        all_e007_results=all_e007_results,
        plot_paths=plot_paths,
        output_path=str(master_path),
    )
    print(f"  - Generated master report: {doc_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("CSP-P2-E007 DIAGNOSTIC PIPELINE — FINAL SUMMARY")
    print("=" * 60)
    print(f"Tests Passed:          {n_pass}/{test_res.testsRun}")
    print(f"Original Memory MSE:   {ORIGINAL_E001_NUMBERS['memory_model_mse']:.6f}")
    print(f"Corrected Memory MSE:  {cmp['corrected_memory_mse']:.6f}  (Δ={cmp['delta_memory_mse']:+.6f})")
    print(f"Original Gap:          {ORIGINAL_E001_NUMBERS['gap_pct']:.1f}%")
    print(f"After ES Fix Gap:      {cmp['corrected_gap_pct']:.1f}%  (Δ={cmp['delta_gap_pct']:+.1f}pp)")
    print(f"Mem vs Matched-MLP:    {gaps['memory_vs_matched_mlp_pct']:.1f}%  (capacity-matched)")
    print(f"Capacity interp:       {cap['interpretation']}")
    print(f"Part B triggered:      {decision['trigger_part_b']}")
    print(f"Part B rationale:      {decision['rationale'][:100]}")
    print(f"Reports:               docs/experiments/CSP-P2-E007*.docx")
    print(f"                       {doc_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_full_e007_pipeline()
