"""Generate the Phase 5 DOCX experiment report from saved results (logs/phase5_all_experiments_results.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import generate_p5_docx_report

if __name__ == "__main__":
    with open("logs/phase5_all_experiments_results.json", "r", encoding="utf-8") as f:
        all_results = json.load(f)["experiments"]

    out = generate_p5_docx_report(all_results, output_path="docs/CSP-P5-E001.docx")
    print(f"Report saved to: {out}")
