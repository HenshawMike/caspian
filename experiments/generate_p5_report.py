"""Master pipeline for Phase 5 (Emergent Categories) Experiments.

Execution Steps:
  1. Run automated regression test suite
  2. Run CSP-P5-E001 — Latent Representation Clustering & Cosine Similarity
  3. Run CSP-P5-E002 — Probing Latent Structure & Silhouette Score Quality
  4. Run CSP-P5-E003 — Category Generalization to Unseen Entity Instances
  5. Run CSP-P5-E004 — Multi-Seed Reproducibility Analysis (8 Seeds)
  6. Run CSP-P5-E005 — Boundary Isolation, Semantic Firewall & Criteria Assessment
  7. Save results JSON and compile individual & master DOCX reports
"""

import os
import sys
import json
import unittest
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.phase5_experiments import (
    run_experiment_p5_e001,
    run_experiment_p5_e002,
    run_experiment_p5_e003,
    run_experiment_p5_e004,
    run_experiment_p5_e005,
)
from experiments.report_generator import (
    generate_p5_individual_docx_report,
    generate_p5_master_docx_report,
)


def run_tests() -> bool:
    """Run full automated test discovery."""
    print("\n[Step 1/7] Running full automated test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    print(f"Tests passed: {res.testsRun - len(res.failures) - len(res.errors)}/{res.testsRun}")
    return res.wasSuccessful()


def main():
    print("=" * 60)
    print("PROJECT CASPIAN: PHASE 5 — EMERGENT CATEGORIES PIPELINE")
    print("=" * 60)

    # 1. Tests
    if not run_tests():
        print("Tests failed! Aborting pipeline.")
        return

    base_path = Path(__file__).resolve().parent.parent

    # 2. Run E001
    print("\n[Step 2/7] Running CSP-P5-E001: Latent Representation Clustering & Cosine Similarity...")
    e001 = run_experiment_p5_e001(seed=42)

    # 3. Run E002
    print("\n[Step 3/7] Running CSP-P5-E002: Probing Latent Structure & Silhouette Quality...")
    e002 = run_experiment_p5_e002(seed=42)

    # 4. Run E003
    print("\n[Step 4/7] Running CSP-P5-E003: Category Generalization to Unseen Entity Instances...")
    e003 = run_experiment_p5_e003(seed=42)

    # 5. Run E004
    print("\n[Step 5/7] Running CSP-P5-E004: Multi-Seed Category Emergence Analysis (8 Seeds)...")
    e004 = run_experiment_p5_e004(seeds=[1, 2, 3, 4, 5, 6, 7, 8])

    # 6. Run E005
    print("\n[Step 6/7] Running CSP-P5-E005: Boundary Isolation, Semantic Firewall & Criteria Assessment...")
    e005 = run_experiment_p5_e005(e001, e002, e003, e004)

    all_results = {
        "CSP-P5-E001": e001,
        "CSP-P5-E002": e002,
        "CSP-P5-E003": e003,
        "CSP-P5-E004": e004,
        "CSP-P5-E005": e005,
    }

    # 7. Save JSON and compile DOCX reports
    print("\n[Step 7/7] Saving JSON and compiling DOCX reports...")
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
    full_json_path = os.path.join(base_path, "logs", "phase5_full_results.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  - Saved: {full_json_path}")

    docs_exp_dir = os.path.join(base_path, "docs", "experiments")
    os.makedirs(docs_exp_dir, exist_ok=True)
    for exp_id, res in all_results.items():
        doc_path = os.path.join(docs_exp_dir, f"{exp_id}.docx")
        generate_p5_individual_docx_report(exp_id, res, doc_path)
        print(f"  - Generated: {doc_path}")

    master_doc_path = os.path.join(base_path, "docs", "CSP-P5.docx")
    generate_p5_master_docx_report(all_results, master_doc_path)
    print(f"  - Generated master report: {master_doc_path}")

    print("\n" + "=" * 60)
    print("CSP-P5 FINAL ASSESSMENT — COMPLETE SUMMARY")
    print("=" * 60)
    for c in e005["criteria_table"]:
        print(f"  Criterion {c['criterion_id']}: {c['title']:<40} [{c['status']}]")
    print(f"\nOverall Scientific Conclusion: {e005['scientific_conclusion']}")
    print(f"Master DOCX:                   docs/CSP-P5.docx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
