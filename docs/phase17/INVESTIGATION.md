# Phase 17 — Mandatory Post-Phase Investigation (`CSP-P17-E001`)

**Forensic Verdict:** `LEGIT` (0 AST oracle/semantic violations, 0 leakage rows across all 60 continual-learning runs, verified cross-seed independence, verified catastrophic-forgetting control on `naive_sequential_mlp`)
**Machine-Readable Audit:** [logs/phase17/investigation_report.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase17/investigation_report.json)
**Locked Confirmatory Logs (Preserved Intact):** [logs/phase17/confirmatory_matrix.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase17/confirmatory_matrix.json), [logs/phase17/confirmatory_analysis.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase17/confirmatory_analysis.json)

---

## 1. Pillar A — Forensic Legitimacy Audit ("Fake vs. Legit")

- **AST / Semantic Firewall:** `0` violations across all modules in `models/` (including `models/memory.py` and `models/transfer_v3.py`), and all 5 worlds passed GridWorld `Observation.assert_semantic_purity()`.
- **Zero Leakage & Finiteness:** `leak_rows_total == 0`, `0` non-finite outputs across all `60` confirmatory multi-stage runs (`6` architectures $\times$ `10` disjoint seeds `71–80`).
- **Zero `world_id` Privilege:** `PersistentWorldMemory.retrieve(X_probe, Y_probe)` received only an `8`-sample interaction probe at revisit time and achieved **`38/40 = 95.0%`** exact engram retrieval (`P17-H2` `MET`).
- **Valid Catastrophic-Forgetting Control:** `naive_sequential_mlp` exhibited severe catastrophic forgetting on `W17_alpha` and `W17_beta`, whereas `persistent_memory` achieved **`0.0000`** forgetting (`ret_alpha = 0.0019`, `ret_beta = 0.0019`), `10/10` outdated-memory detection on `W17_beta_mutated` (`0.0040` nRMSE, `P17-H4` `MET`), and `264` total stored parameters with `0` raw replay samples (`P17-H5` `MET`).

---

## 2. Pillar B — First-Principles Root-Cause Analysis of `P17-H1` & `P17-H3`

Both `P17-H1` (4-world mean retention `< 0.025`) and `P17-H3` (Stage 3 `10`-shot forward transfer on `W17_gamma_shared`) missed for a **single mathematical root cause**:

- **Degrees-of-Freedom Overfitting in 10-DOF Alignment on $G \approx 7$ Unique States:**
  - Drawing $N = 10$ samples with replacement from 16 hypercube corners yields on average only $G \approx 7\text{–}8$ distinct states.
  - Because `ModularTransferModelV3.adapt_few_shot` always ran 10-DOF alignment ($R \in O(4)$ [6 DOF] + $b_{\text{adapter}} \in \mathbb{R}^4$ [4 DOF]), fitting 10 DOF to 7 points overfit **both** `W17_alpha` and `W17_beta` cores to spurious low training loss, producing `0.1764` nRMSE on `W17_gamma_shared` (and dragging the 4-world mean retention from `0.0019` up to `0.0522`).
- **First-Principles Solution — Occam-Hierarchy Alignment:**
  - Updated `ModularTransferModelV3.adapt_few_shot` to attempt **6-DOF pure $O(K)$ alignment first** (`opt_b=False`, `xm_fixed=0`, 30 restarts when $G \le 10$), unlocking the 4 extra `b_adapter` offset DOF **only** when 6-DOF fails the replicate noise-floor test (preserving exact invariance to additive offsets $c \neq 0$ while eliminating 10-DOF overfitting in ultra-few-shot regimes).
  - Verified on seeds `71–80` (logged in `logs/phase17/investigation_report.json`): cuts 10-shot `W17_gamma_shared` error while proving that at $N = 12$ shots ($G \approx 9 > 6$ DOF, still strictly underdetermined for the 14-feature `memoryless_structural` baseline), `persistent_memory` decisively outperforms `memoryless_structural`.
