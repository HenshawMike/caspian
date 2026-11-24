# Phase 18B (`CSP-P18B-E001`) — Stage 7 Post-Confirmatory Forensic Investigation

- **Locked Confirmatory Results (Untouched):** [logs/phase18b/p18b_confirmatory_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18b/p18b_confirmatory_results.json) (`seeds 191–205`, $n = 15$).
- **Locked Pre-Confirmatory Commit:** `b187add60b6cb9cc14f5b65387170acbfe2b3d3b` on `feature/planning-sync`.
- **Official Confirmatory Outcome:** **`6/7` Pre-Registered Hypotheses `MET`** (`P18B-H2, H3, H4, H5, H6, H7` `MET`; `P18B-H1` `NOT MET` by `0.00047` on scalar cutoff `0.05047 > 0.05` despite beating MLP at both $H=1,3$ and cutting boundary error `39.3%`).

## Pillar A — Forensic Legitimacy Audit ("Fake vs. Legit")
**Verdict: `LEGIT`.**
1. **Zero Leakage & Immutable Baselines:** `models/memory.py`, `models/transfer_v3.py`, `models/transfer.py`, `logs/phase18/`, and `docs/CSP-P18-E001.docx` remain 100% untouched. Static AST audit across all 17 files in `models/` returned **`0` violations**.
2. **Exact Mathematical Equivalence Under Basis Rotation (`P18B-H2`):** Across all 15 confirmatory seeds under arbitrary random $Q \in O(4)$ rotations + offset shifts, `max_mean_equiv_err = 1.78e-15`, `max_var_equiv_err = 2.17e-17`, and `max_serialization_diff = 0.0`.
3. **Determinism & Unit/Regression Suite:** `determinism_diff = 0.0`; `tests/test_phase18b.py`, `tests/test_phase18.py`, `tests/test_phase17b.py`, `tests/test_phase17.py`, and `tests/test_phase16c.py` all pass with zero regressions.
4. **Honest Reporting of Neutral/Negative Findings:** `P18B-H1` (`0.05047` vs `< 0.05`) is reported honestly as `NOT MET`, and medium-data (`N = 160`) parity against `mlp_world_model` (`427.35 vs 431.36`, $p = 0.939$) is reported transparently.

## Pillar B — First-Principles Diagnosis of `P18B-H1` (`0.05047` vs `< 0.05`)
- **What Passed Inside `P18B-H1`:**
  - Beat `mlp_world_model` at $H = 1$ (**`0.05047 vs 0.06033`**, `14/15` wins, $d_z = 0.935$, Holm $p = 0.0018$).
  - Beat `mlp_world_model` at $H = 3$ (**`0.06896 vs 0.10497`**, `15/15` wins, $d_z = 2.722$, Holm $p = 0.0002$).
  - Cut boundary 1-step error by **`39.3%`** vs Phase 18 baseline (**`0.06538 vs 0.10776`**, `15/15` wins, $d_z = 2.640$, $p = 0.0001$).
  - Improved 95% PI calibration to **`94.7%` (interior) / `93.4%` (boundary)** — nearly exact nominal 95% coverage.
- **Why `cwm_x_h1 = 0.05047` Missed `< 0.05` by `0.00047`:**
  A smooth degree-2 polynomial $\Phi(z)$ approximates the non-differentiable corner of $\text{clip}(\cdot, -1.8, 1.8)$ to within `0.065` at the boundary and `0.039` in the interior, yielding an overall weighted nRMSE of `0.05047` (within `0.94%` of `0.05`). Achieving `< 0.03` on hard box-clipped transitions would require a piecewise hinge basis, at the cost of breaking exact $O(K)$ rotational closure — a clear mathematical trade-off between smooth rotational invariance and non-smooth corner approximation.
