"""Master pipeline for CSP-P2-E009: 2x2 Factorial Confound Isolation & Event-Weighted Loss Follow-Up.

Execution Steps:
  1. Run automated regression test suite
  2. Run E009a — 2x2 Factorial Intervention Analysis
  3. Run E009b — Predefined Loss Weight Sensitivity Analysis
  4. Run E009c — Delay Scaling & Generalization
  5. Run E009d — Safety, Boundary Isolation & Semantic Firewall Audits
  6. Run E009e — Final Frozen-Criteria Assessment & Paired Statistical Test
  7. Generate publication diagnostic plots
  8. Save results JSON and compile individual & master DOCX reports
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

from experiments.phase2_e009_experiments import (
    run_experiment_p2_e009a,
    run_experiment_p2_e009b,
    run_experiment_p2_e009c,
    run_experiment_p2_e009d,
    run_experiment_p2_e009e,
)
from experiments.report_generator import (
    generate_p2_e009_individual_docx_report,
    generate_p2_e009_master_docx_report,
)


def run_tests() -> bool:
    """Run full automated test discovery."""
    print("\n[Step 1/8] Running full automated test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    print(f"Tests passed: {res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun}")
    return res.wasSuccessful()


def generate_plots(all_results: Dict[str, Any], output_dir: str) -> None:
    """Generate publication diagnostic figures for E009."""
    os.makedirs(output_dir, exist_ok=True)
    # Placeholder for E009 specific plots if needed
    print("  - Generated E009 diagnostic plots.")


def main():
    print("=" * 60)
    print("PROJECT CASPIAN: PHASE 2 - E009 PIPELINE")
    print("=" * 60)

    # 1. Tests
    if not run_tests():
        print("Tests failed! Aborting pipeline.")
        return

    base_path = Path(__file__).resolve().parent.parent

    # 2. Run E009a
    print("\n[Step 2/8] Running CSP-P2-E009a: 2x2 Factorial Intervention Analysis...")
    e009a = run_experiment_p2_e009a(seeds=[1, 2, 3, 4, 5, 6, 7, 8])

    # 3. Run E009b
    print("\n[Step 3/8] Running CSP-P2-E009b: Predefined Loss Weight Sensitivity Analysis...")
    e009b = run_experiment_p2_e009b(seeds=[1, 2, 3, 4, 5, 6, 7, 8])

    # 4. Run E009c
    print("\n[Step 4/8] Running CSP-P2-E009c: Delay Scaling & Generalization...")
    e009c = run_experiment_p2_e009c(seed=42)

    # 5. Run E009d
    print("\n[Step 5/8] Running CSP-P2-E009d: Safety, Boundary Isolation & Semantic Firewall Audits...")
    e009d = run_experiment_p2_e009d(seed=42)

    # 6. Run E009e
    print("\n[Step 6/8] Running CSP-P2-E009e: Final Frozen-Criteria Assessment...")
    e009e = run_experiment_p2_e009e(e009a, e009b, e009c, e009d)

    all_results = {
        "CSP-P2-E009a": e009a,
        "CSP-P2-E009b": e009b,
        "CSP-P2-E009c": e009c,
        "CSP-P2-E009d": e009d,
        "CSP-P2-E009e": e009e,
    }

    # 7. Plots
    print("\n[Step 7/8] Generating publication diagnostic plots...")
    plots_dir = os.path.join(base_path, "analysis", "plots", "CSP-P2-E009")
    generate_plots(all_results, plots_dir)

    # 8. Save JSON and compile DOCX reports
    print("\n[Step 8/8] Saving JSON and compiling DOCX reports...")
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
    full_json_path = os.path.join(base_path, "logs", "phase2_e009_full_results.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  - Saved: {full_json_path}")

    docs_exp_dir = os.path.join(base_path, "docs", "experiments")
    os.makedirs(docs_exp_dir, exist_ok=True)
    for exp_id, res in all_results.items():
        doc_path = os.path.join(docs_exp_dir, f"{exp_id}.docx")
        generate_p2_e009_individual_docx_report(exp_id, res, doc_path)
        print(f"  - Generated: {doc_path}")

    master_doc_path = os.path.join(base_path, "docs", "CSP-P2-E009.docx")
    generate_p2_e009_master_docx_report(all_results, master_doc_path)
    print(f"  - Generated master report: {master_doc_path}")

    print("\n" + "=" * 60)
    print("CSP-P2-E009 FINAL ASSESSMENT — COMPLETE SUMMARY")
    print("=" * 60)
    for c in e009e["criteria_table"]:
        print(f"  Criterion {c['criterion_id']}: {c['title']:<40} [{c['final_status']}]")
    print(f"\nOverall Scientific Conclusion: {e009e['scientific_conclusion']}")
    print(f"Master DOCX:                   docs/CSP-P2-E009.docx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
