# Phase 16B (`CSP-P16B-E001`) — Exact Reproduction Instructions

## 1. Environment & Frozen Candidate Verification
- **Repository Commit (Stage A Pre-Registration):** `2de525eefda1e057f3de788e3412d8cd5535d71d`
- **Frozen Candidate:** `models/transfer.py`
- **Required SHA-256:** `c6ec7df4a194bd84cfe93078550b138872d83c457828ebce9eacb798f2ae0cb0`
- **Tested Environment:** Python `3.14.7`, NumPy `2.5.2`, Pytest `9.2.0` on Linux (`x86_64`).

Verify the frozen candidate hash before running:
```bash
sha256sum models/transfer.py
# Expected: c6ec7df4a194bd84cfe93078550b138872d83c457828ebce9eacb798f2ae0cb0  models/transfer.py
```

## 2. Run Unit & Freeze Integrity Tests
```bash
.venv/bin/pytest tests/test_phase16b.py -q
```

## 3. Execute Full Independent Replication, Forensic Audit, Robustness & Gate F Check
```bash
PYTHONPATH=. .venv/bin/python experiments/phase16b_independent_harness.py
```
This regenerates:
- `docs/phase16b/P16B_REPLICATION_RESULTS.csv` (`480` rows, seeds `101–110`)
- `docs/phase16b/P16B_STATISTICAL_ANALYSIS.csv` (paired $\Delta_i$, CIs, exact permutation $p$-values, oracle-impossibility counts)
- `docs/phase16b/P16B_FORENSIC_AUDIT.json` (Experiment A, Section 11 AST/leakage audit, Gate F subprocess diff `0.0`, Gates A–F)
- `logs/phase16b/p16b_full_summary.json`, `logs/phase16b/checkpoints/*.json`, and `logs/phase16b/predictions/*.json`.

## 4. Regenerate the Permanent DOCX Report
```bash
PYTHONPATH=. .venv/bin/python experiments/generate_p16b_report.py
```
Produces `docs/CSP-P16B-E001.docx`.
