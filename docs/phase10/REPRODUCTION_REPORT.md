# Reproduction Report (Stage 10.0)

Date: 2026-11-12. Source: `caspian-phase9.zip` as uploaded by the project owner.

## Baseline execution (before any modification)
- Python 3.13.18, numpy 2.5.3, pytest 9.1.1, python-docx 1.2.0.
- `python -m pytest -q tests`: **88 passed** on the unmodified recovered code.

## Bit-for-bit re-execution of logged results
| Check | Command | Logged | Re-run | Result |
|---|---|---|---|---|
| Phase 9 E001 cell C1 / 8000 steps / small / seed 42 | `run_experiment_e001(seeds=[42], sizes={"small": (8,)}, data_steps=[8000], levels=["C1"])` | in-dist 0.9986, unseen 0.9978 | identical | PASS |
| Phase 7 E007 seed 42, sigma 0 (all three arms, swapped and matched R2) | `run_experiment_e007(seeds=[42], noise_levels=[0.0])` | see log | identical | PASS |

Conclusion: seeded experiments in this codebase are deterministic on this platform, so logged per-seed metrics can be regenerated exactly. Not every experiment was re-run; only the two above, chosen from different phases.

## Changes made after the baseline (all with tests)
1. `experiments/safe_io.py`: result files are never overwritten silently (a timestamped sibling is written unless `CASPIAN_ALLOW_OVERWRITE=1`); wired into the Phase 6-9 `run_all_*` functions.
2. `run_all_phase7_experiments` now also runs E005-R2, E006a, E006b and E007 (previously these results were merged into the log by hand).
3. Phase 9 report text corrected from "3 of 6" to "4 of 6" hypotheses met; the original report is archived at `logs/archive_pre_phase10/CSP-P9-E001.original.docx`.
4. Pre-Phase-10 result logs copied to `logs/archive_pre_phase10/`.
- Tests after changes: 90 passed.
