"""Master pipeline for Phase 4 (World Model & Planning) Experiments.

Execution Steps:
  1. Run automated regression test suite
  2. Run CSP-P4-E001 — Multi-Step Forward World Model Rollouts
  3. Run CSP-P4-E002 — Model-Based Planning & Counterfactual Simulation
  4. Run CSP-P4-E003 — Multi-Seed Reproducibility Analysis (8 Seeds)
  5. Run CSP-P4-E004 — Boundary Isolation, Semantic Firewall & Criteria Assessment
  6. Save results JSON and compile individual & master DOCX reports
"""

import os
import sys
import json
import unittest
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase4_experiments import (
    run_experiment_p4_e001,
    run_experiment_p4_e002,
    run_experiment_p4_e003,
    run_experiment_p4_e004,
)
from experiments.report_generator import (
    generate_p4_individual_docx_report,
    generate_p4_master_docx_report,
)


def run_tests() -> bool:
    """Run full automated test discovery."""
    print("\n[Step 1/6] Running full automated test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    print(f"Tests passed: {res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun}")
    return res.wasSuccessful()


def main():
    print("=" * 60)
    print("PROJECT CASPIAN: PHASE 4 — WORLD MODEL & PLANNING PIPELINE")
    print("=" * 60)

    # 1. Tests
    if not run_tests():
        print("Tests failed! Aborting pipeline.")
        return

    base_path = Path(__file__).resolve().parent.parent

    # 2. Run E001
    print("\n[Step 2/6] Running CSP-P4-E001: Multi-Step Forward World Model Rollouts...")
    e001 = run_experiment_p4_e001(seed=42)

    # 3. Run E002
    print("\n[Step 3/6] Running CSP-P4-E002: Model-Based Planning & Counterfactual Simulation...")
    e002 = run_experiment_p4_e002(seed=42)

    # 4. Run E003
    print("\n[Step 4/6] Running CSP-P4-E003: Multi-Seed Reproducibility Analysis (8 Seeds)...")
    e003 = run_experiment_p4_e003(seeds=[1, 2, 3, 4, 5, 6, 7, 8])

    # 5. Run E004
    print("\n[Step 5/6] Running CSP-P4-E004: Boundary Isolation, Semantic Firewall & Criteria Assessment...")
    e004 = run_experiment_p4_e004(e001, e002, e003)

    all_results = {
        "CSP-P4-E001": e001,
        "CSP-P4-E002": e002,
        "CSP-P4-E003": e003,
        "CSP-P4-E004": e004,
    }

    # 6. Save JSON and compile DOCX reports
    print("\n[Step 6/6] Saving JSON and compiling DOCX reports...")
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
    full_json_path = os.path.join(base_path, "logs", "phase4_full_results.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  - Saved: {full_json_path}")

    docs_exp_dir = os.path.join(base_path, "docs", "experiments")
    os.makedirs(docs_exp_dir, exist_ok=True)
    for exp_id, res in all_results.items():
        doc_path = os.path.join(docs_exp_dir, f"{exp_id}.docx")
        generate_p4_individual_docx_report(exp_id, res, doc_path)
        print(f"  - Generated: {doc_path}")

    master_doc_path = os.path.join(base_path, "docs", "CSP-P4.docx")
    generate_p4_master_docx_report(all_results, master_doc_path)
    print(f"  - Generated master report: {master_doc_path}")

    print("\n" + "=" * 60)
    print("CSP-P4 FINAL ASSESSMENT — COMPLETE SUMMARY")
    print("=" * 60)
    for c in e004["criteria_table"]:
        print(f"  Criterion {c['criterion_id']}: {c['title']:<40} [{c['status']}]")
    print(f"\nOverall Scientific Conclusion: {e004['scientific_conclusion']}")
    print(f"Master DOCX:                   docs/CSP-P4.docx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
