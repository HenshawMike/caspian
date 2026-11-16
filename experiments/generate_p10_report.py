"""Generate the Phase 10 DOCX report from logs/phase10 results."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import generate_p10_docx_report
if __name__ == "__main__":
    e1 = json.load(open("logs/phase10/phase10_e001_results.json"))["experiments"]
    e2 = json.load(open("logs/phase10/phase10_e002_results.json"))["experiments"]["CSP-P10-E002"]
    print(generate_p10_docx_report({"CSP-P10-E001": e1["CSP-P10-E001"]}, {"summary": e2["summary"]}, "docs/CSP-P10-E001.docx"))
