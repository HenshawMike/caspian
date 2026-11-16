"""Generate the Phase 11 DOCX from logs/phase11/confirmatory_analysis.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import generate_p11_docx_report
if __name__ == "__main__":
    an = json.load(open("logs/phase11/confirmatory_analysis.json"))
    print(generate_p11_docx_report(an, "docs/CSP-P11-E001.docx"))
