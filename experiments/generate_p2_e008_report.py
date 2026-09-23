"""Master pipeline for CSP-P2-E008: Reproducibility, Confound Ablations & 7-Criteria Reconciliation.

Execution Steps:
  1. Run automated regression test suite
  2. Run E008a — Confound ablations & triple-run determinism proof
  3. Run E008b — Canonical delay sweep & generalization with 3-way side-by-side
  4. Run E008c — Criterion 1 re-test (deficit-step error reduction vs 70% threshold)
  5. Run E008d — Criterion 3 re-test (8-seed reproducibility & win rate)
  6. Run E008e — Final 7-criteria reconciliation table & Phase 3 status
  7. Generate publication diagnostic plots
  8. Populate experiment artifact directories (E008a through E008e)
  9. Save results JSON and compile individual & master DOCX reports
"""

import os
import sys
import json
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase2_e008_experiments import (
    run_experiment_p2_e008a,
    run_experiment_p2_e008b,
    run_experiment_p2_e008c,
    run_experiment_p2_e008d,
    run_experiment_p2_e008e,
)
from experiments.report_generator import (
    generate_p2_e008_individual_docx_report,
    generate_p2_e008_master_docx_report,
)


def run_tests() -> bool:
    """Run full automated test discovery."""
    print("\n[Step 1/9] Running full automated test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    print(f"Tests passed: {res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun}")
    return res.wasSuccessful()


def generate_plots(all_results: Dict[str, Any], output_dir: str) -> None:
    """Generate publication diagnostic figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Delay Sweep Comparison (Original vs Corrected Matched)
    e008b = all_results.get("CSP-P2-E008b", {})
    sbs = e008b.get("side_by_side_comparisons", {})
    sweep_data = sbs.get("delay_sweep", [])
    if sweep_data:
        delays = [r["delay"] for r in sweep_data]
        orig_mem = [r["original_memory_mse"] for r in sweep_data]
        orig_nomem = [r["original_no_mem_mse"] for r in sweep_data]
        corr_mem = [r["corrected_memory_mse"] for r in sweep_data]
        corr_matched = [r["corrected_matched_mlp_mse"] for r in sweep_data]

        plt.figure(figsize=(9, 5.5))
        plt.plot(delays, orig_mem, "b--o", label="Original Memory GRU (E003)", alpha=0.7)
        plt.plot(delays, orig_nomem, "r--s", label="Original No-Memory MLP (E003)", alpha=0.7)
        plt.plot(delays, corr_mem, "b-o", linewidth=2.2, label="Corrected Memory GRU (E008b)")
        plt.plot(delays, corr_matched, "g-^", linewidth=2.2, label="Corrected Capacity-Matched MLP (E008b)")

        plt.xlabel("Temporal Delay (d)", fontsize=12)
        plt.ylabel("Prediction MSE (Holdout)", fontsize=12)
        plt.title("CSP-P2-E008: Temporal Delay Sweep Side-by-Side Comparison", fontsize=14, fontweight="bold")
        plt.legend(frameon=True, facecolor="white", loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p1_path = os.path.join(output_dir, "e008_delay_sweep_side_by_side.png")
        plt.savefig(p1_path, dpi=200)
        plt.close()
        print(f"  - Generated: {p1_path}")

    # Plot 2: 3-Way Generalization Ablations
    e008a = all_results.get("CSP-P2-E008a", {})
    abl = e008a.get("ablations", {})
    if abl:
        labels = ["Baseline E004", "Abl A (30 eps)", "Abl B (p=0.30)", "Abl C (Early Stop)", "Combined"]
        d3_mses = [
            abl.get("baseline_e004_settings", {}).get("eval_results", {}).get("delay_3", {}).get("mse", 0),
            abl.get("ablation_a_episode_count_30_vs_36", {}).get("eval_results", {}).get("delay_3", {}).get("mse", 0),
            abl.get("ablation_b_interaction_density_030_vs_025", {}).get("eval_results", {}).get("delay_3", {}).get("mse", 0),
            abl.get("ablation_c_early_stopping_vs_none", {}).get("eval_results", {}).get("delay_3", {}).get("mse", 0),
            abl.get("combined_corrected_settings", {}).get("eval_results", {}).get("delay_3", {}).get("mse", 0),
        ]
        d6_mses = [
            abl.get("baseline_e004_settings", {}).get("eval_results", {}).get("delay_6", {}).get("mse", 0),
            abl.get("ablation_a_episode_count_30_vs_36", {}).get("eval_results", {}).get("delay_6", {}).get("mse", 0),
            abl.get("ablation_b_interaction_density_030_vs_025", {}).get("eval_results", {}).get("delay_6", {}).get("mse", 0),
            abl.get("ablation_c_early_stopping_vs_none", {}).get("eval_results", {}).get("delay_6", {}).get("mse", 0),
            abl.get("combined_corrected_settings", {}).get("eval_results", {}).get("delay_6", {}).get("mse", 0),
        ]

        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 5.5))
        plt.bar(x - width/2, d3_mses, width, label="d=3 (Interpolation) MSE", color="#2b5c8f")
        plt.bar(x + width/2, d6_mses, width, label="d=6 (Extrapolation) MSE", color="#d95f02")
        plt.axhline(3.93, color="black", linestyle="--", alpha=0.7, label="Persistence Baseline (~3.93)")
        plt.xticks(x, labels, fontsize=10)
        plt.ylabel("Prediction MSE", fontsize=12)
        plt.title("CSP-P2-E008a: Generalization Confound Ablation Study", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p2_path = os.path.join(output_dir, "e008_generalization_ablations.png")
        plt.savefig(p2_path, dpi=200)
        plt.close()
        print(f"  - Generated: {p2_path}")

    # Plot 3: 8-Seed Reproducibility Bar Chart
    e008d = all_results.get("CSP-P2-E008d", {})
    records = e008d.get("records", [])
    if records:
        seeds = [str(r["seed"]) for r in records]
        mem_mses = [r["memory_mse"] for r in records]
        matched_mses = [r["matched_mlp_mse"] for r in records]

        x = np.arange(len(seeds))
        width = 0.35
        plt.figure(figsize=(9, 5))
        plt.bar(x - width/2, mem_mses, width, label="Memory GRU", color="#1B365D")
        plt.bar(x + width/2, matched_mses, width, label="Matched MLP (6,748 params)", color="#E06D53")
        plt.xticks(x, [f"Seed {s}" for s in seeds])
        plt.ylabel("Prediction MSE", fontsize=12)
        plt.title("CSP-P2-E008d: 8-Seed Reproducibility vs Capacity-Matched Baseline", fontsize=13, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p3_path = os.path.join(output_dir, "e008_8seed_reproducibility.png")
        plt.savefig(p3_path, dpi=200)
        plt.close()
        print(f"  - Generated: {p3_path}")


def populate_artifact_dirs(all_results: Dict[str, Any], base_dir: str) -> None:
    """Populate experiment artifact folders for E008a through E008e."""
    for exp_id, res in all_results.items():
        exp_dir = os.path.join(base_dir, "experiments", exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        json_path = os.path.join(exp_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)

        readme_path = os.path.join(exp_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# {exp_id}: {res.get('title', exp_id)}\n\n")
            f.write(f"**Phase:** Phase 2 — Memory & Persistence (CSP-P2-E008 Reconciliation)\n\n")
            f.write("```json\n")
            f.write(json.dumps(res, indent=2))
            f.write("\n```\n")

        csv_path = os.path.join(exp_dir, "metrics.csv")
        with open(csv_path, "w") as f:
            f.write("metric,value\n")
            f.write(f"experiment_id,{exp_id}\n")
            f.write(f"title,{res.get('title', '')}\n")

    print(f"  - Populated artifact directories: {', '.join(all_results.keys())}")


def main():
    print("=" * 60)
    print("PROJECT CASPIAN — CSP-P2-E008 FINAL RECONCILIATION PIPELINE")
    print("=" * 60)

    base_path = Path(__file__).resolve().parent.parent

    # Step 1: Run regression test suite
    if not run_tests():
        print("ERROR: Test suite failed. Halting pipeline.")
        sys.exit(1)

    all_results = {}

    # Step 2: Run E008a
    print("\n[Step 2/9] Running CSP-P2-E008a: Reproducibility & Confound Ablations...")
    e008a = run_experiment_p2_e008a(seed=42)
    all_results["CSP-P2-E008a"] = e008a
    det = e008a["determinism_proof"]
    print(f"  Triple-run deterministic: {det['triple_run_identical']}")
    print(f"  Negative control divergent: {det['negative_control_divergent']}")
    print(f"  Ablation C (Early stop) d=3 beats persist: {e008a['ablations']['ablation_c_early_stopping_vs_none']['eval_results']['delay_3']['beats_persistence']}")

    # Step 3: Run E008b
    print("\n[Step 3/9] Running CSP-P2-E008b: Canonical Delay Sweep & Generalization...")
    e008b = run_experiment_p2_e008b(seed=42)
    all_results["CSP-P2-E008b"] = e008b
    gen_res = e008b["canonical_generalization"]
    print(f"  Canonical generalization to all unseen delays: {gen_res['generalizes_to_all_unseen_delays']}")
    print(f"    d=3 MSE: {gen_res['results']['delay_3']['memory_mse']} (beats persistence: {gen_res['results']['delay_3']['beats_persistence']})")
    print(f"    d=6 MSE: {gen_res['results']['delay_6']['memory_mse']} (beats persistence: {gen_res['results']['delay_6']['beats_persistence']})")

    # Step 4: Run E008c
    print("\n[Step 4/9] Running CSP-P2-E008c: Criterion 1 Deficit-Step Isolation...")
    e008c = run_experiment_p2_e008c(delay=2, seed=42)
    all_results["CSP-P2-E008c"] = e008c
    def_res = e008c["deficit_step_results"]
    std_res = e008c["standard_step_results"]
    print(f"  Deficit-step error reduction vs matched MLP: {def_res['error_reduction_vs_matched_pct']}% (Target >70%: {def_res['cleared_70pct_threshold']})")
    print(f"  Standard-step error reduction vs matched MLP: {std_res['error_reduction_vs_matched_pct']}% (Original was -219.9%)")

    # Step 5: Run E008d
    print("\n[Step 5/9] Running CSP-P2-E008d: Criterion 3 8-Seed Reproducibility...")
    e008d = run_experiment_p2_e008d(seeds=[1, 2, 3, 4, 5, 6, 7, 8], delay=2)
    all_results["CSP-P2-E008d"] = e008d
    sum_d = e008d["summary"]
    print(f"  Win rate vs capacity-matched MLP: {sum_d['win_rate']}")
    print(f"  Consistent superiority across all seeds: {sum_d['consistent_superiority']}")
    print(f"  Mean advantage gap: {sum_d['mean_advantage_gap_pct']}%")

    # Step 6: Run E008e
    print("\n[Step 6/9] Running CSP-P2-E008e: Final 7-Criteria Reconciliation...")
    e008e = run_experiment_p2_e008e(
        e008a_results=e008a,
        e008b_results=e008b,
        e008c_results=e008c,
        e008d_results=e008d,
        seed=42,
    )
    all_results["CSP-P2-E008e"] = e008e
    print(f"  Criteria Passed: {e008e['criteria_passed']} / {e008e['total_criteria']}")
    print(f"  Definitive Scientific Conclusion: {e008e['scientific_conclusion']}")
    print(f"  Phase 3 Status: {e008e['phase3_status']['decision']}")

    # Step 7: Generate Publication Plots
    print("\n[Step 7/9] Generating publication plots...")
    plots_dir = os.path.join(base_path, "logs", "plots_e008")
    generate_plots(all_results, plots_dir)

    # Step 8: Populate artifact directories
    print("\n[Step 8/9] Populating experiment artifact directories...")
    populate_artifact_dirs(all_results, str(base_path))

    # Step 9: Save JSON and compile DOCX reports
    print("\n[Step 9/9] Saving JSON and compiling DOCX reports...")
    full_json_path = os.path.join(base_path, "logs", "phase2_e008_full_results.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  - Saved: {full_json_path}")

    docs_exp_dir = os.path.join(base_path, "docs", "experiments")
    for exp_id, res in all_results.items():
        doc_path = os.path.join(docs_exp_dir, f"{exp_id}.docx")
        generate_p2_e008_individual_docx_report(exp_id, res, doc_path)
        print(f"  - Generated: {doc_path}")

    master_doc_path = os.path.join(base_path, "docs", "CSP-P2-E008.docx")
    generate_p2_e008_master_docx_report(all_results, master_doc_path)
    print(f"  - Generated master report: {master_doc_path}")

    print("\n" + "=" * 60)
    print("CSP-P2-E008 FINAL RECONCILIATION — COMPLETE SUMMARY")
    print("=" * 60)
    for c in e008e["criteria_table"]:
        print(f"  Criterion {c['criterion_id']}: {c['title']:<40} [{c['final_status']}]")
    print(f"\nOverall Scientific Conclusion: {e008e['scientific_conclusion']}")
    print(f"Phase 3 Status:                {e008e['phase3_status']['decision']}")
    print(f"Master DOCX:                   docs/CSP-P2-E008.docx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
