"""CLI entry point to generate docs/CSP-P17-E001.docx from Phase 17 analysis and investigation JSON."""
import json
from experiments.report_generator import generate_p17_docx_report

if __name__ == "__main__":
    an = json.load(open("logs/phase17/confirmatory_analysis.json"))
    out = generate_p17_docx_report(an, "docs/CSP-P17-E001.docx")
    print(out)
