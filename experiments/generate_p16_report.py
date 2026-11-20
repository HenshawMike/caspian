import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import generate_p16_docx_report
if __name__ == "__main__":
    print(generate_p16_docx_report(json.load(open("logs/phase16/confirmatory_analysis.json")), "docs/CSP-P16-E001.docx"))
