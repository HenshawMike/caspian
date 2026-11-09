"""Generate the Phase 9 DOCX report from logs/phase9_all_experiments_results.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import generate_p9_docx_report

if __name__ == "__main__":
    with open("logs/phase9_all_experiments_results.json", "r", encoding="utf-8") as f:
        all_results = json.load(f)["experiments"]
    out = generate_p9_docx_report(all_results, output_path="docs/CSP-P9-E001.docx")
    print(f"Report saved to: {out}")
