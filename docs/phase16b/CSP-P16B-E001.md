# Phase 16B (`CSP-P16B-E001`) — Independent Replication, Forensic Audit & Robustness Summary

- **Parent Experiment:** `CSP-P16-E001` (Phase 16)
- **Frozen Candidate:** [models/transfer.py](file:///home/henshawmike/Projects/Project_Caspian/models/transfer.py) (`SHA-256: c6ec7df4a194bd84cfe93078550b138872d83c457828ebce9eacb798f2ae0cb0`, commit `d6f7f88`)
- **Stage A Pre-Registration Commit:** `2de525eefda1e057f3de788e3412d8cd5535d71d`
- **Fresh Independent Seeds:** `101–110` (`480` runs, `0` leaks, `0` non-finite)
- **Final Scientific Verdict:** **`PARTIAL PASS`** *(Strict Automated Gate Suite: `4/6 PASS`, `2/6 FAIL` — Gates C and D failed their strictest sub-checks; nothing concealed)*

---

## 1. Independent Acceptance Gates (Gates A–F)

| Gate | Verdict | Evidence |
| :--- | :---: | :--- |
| **Gate A — Implementation Integrity** | **PASS** | All A.1–A.5 `PASS`: `14` invariant features (incl. $z_k^2$), `66` trainable params, `0` hidden arrays, $O(4)$ invariance err `4.17e-14`, $\|W^\top W - I\|_F \le 2.35\times 10^{-15}$, brute-force vs hat-matrix LOOCV diff `4.75e-14`. |
| **Gate B — Experimental Integrity** | **PASS** | `0` AST violations across all `14` scanned modules, `0` leakage rows, standalone harness imports neither `phase16_experiments` nor `phase11_experiments`, cross-seed independence verified. |
| **Gate C — Replication** | **FAIL** | Source retention `< 0.02` passed (`max = 0.00167`), and `v2` beat both MLPs on `3/4` worlds at 20-shot and `4/4` worlds at 48-shot, **but** failed strict CI `> 0` vs `scratch_mlp` on `W16_negative_control` (seed 109 spike `0.2013`) and tied fair 66-param `v2 scratch_structural` at 20-shot. |
| **Gate D — Negative-Transfer Behavior** | **FAIL** | `direct_transfer >= 0.40` (`10/10`), `core_gate = 0` (`10/10`), beat `finetune_mlp` (`CI [+0.0064, +0.0942] > 0`, `8/10` wins), **but** 95% CI vs `scratch_mlp` crossed `0` (`[-0.0075, +0.0541]`) due to 1 outlier seed (`0.2013`). |
| **Gate E — Statistical Reliability** | **PASS** | All 10 seeds, mean/SD/median/min/max, paired $t$-stats, exact sign-flip permutation $p$-values, bootstrap 95% CIs, and oracle-impossibility counts reported in [P16B_STATISTICAL_ANALYSIS.csv](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_STATISTICAL_ANALYSIS.csv). |
| **Gate F — Reproducibility** | **PASS** | Separate subprocess re-executed `48` rows from [P16B_CONFIG.json](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_CONFIG.json) with `max |diff| = 0.0` (tolerance `1e-10`). |

---

## 2. Primary Replication Summary (20-Shot & 48-Shot, Seeds `101–110`)

| World | `modular_transfer` (20-shot Mean / Med) | `direct_transfer` (20-shot) | `scratch_mlp` (20-shot) | `finetune_mlp` (20-shot) | `scratch_structural` (v2, 20-shot) | `modular_transfer` (48-shot) | Source Ret. (20-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`W16_obs_shift`** | `0.0208` / **`0.0095`** | `0.2477` | `0.0561` | `0.0491` | `0.0217` / `0.0098` | **`0.0039`** | **`0.0012`** |
| **`W16_shared_plus_new`** | `0.0232` / **`0.0079`** | `0.1285` | `0.0821` | `0.0614` | `0.0207` / `0.0080` | **`0.0030`** | **`0.0012`** |
| **`W16_rule_shift`** | `0.0206` / **`0.0105`** | `0.3403` | `0.0730` | `0.0975` | `0.0178` / `0.0105` | **`0.0037`** | **`0.0012`** |
| **`W16_negative_control`** | `0.0355` / **`0.0105`** | `0.5150` | `0.0611` | `0.0893` | `0.0220` / `0.0097` | **`0.0038`** | **`0.0012`** |

---

## 3. Robustness & Boundary Failure Modes Exposed (Experiment F)

- **Underdetermined Regime (`8–16` shots) & Duplicated Observations:** `modular_transfer` consistently beats fair 66-param `v2 scratch_structural` (`5/5` wins at 8 shots; `3.2x` lower error on `W16_obs_shift` under `10` unique states duplicated 2x: `0.0577` vs `0.1871`).
- **Input Perturbations (`std <= 0.05`):** Autonomous shift detector remained `5/5` on `W16_obs_shift` and `0/5` on `W16_shared_plus_new`.
- **Exposed Sensitivity 1 — Sample Ordering (`W16_obs_shift`):** `max_permutation_spread = 0.0405` (vs `0.0000` on `W16_shared_plus_new`) due to SVD basis sign/frame variation seeding finite-step $O(4)$ Adam.
- **Exposed Failure Mode 2 — Nonzero Additive Sensor Offset (`c = 0.5`):** Error degrades to `mean = 0.0509` (`max = 0.2088`, gate ON drops to `3/5`) because `align_sensor_adapter_few_shot` hardcodes `x_mean = 0`.
- **Exposed Failure Mode 3 — In-Subspace Pure Rotation (`M = Q M_src`):** Subspace residual is `0`, so shift detection triggers `0/5` and the model falls back to scratch fitting (`mean = 0.0093`) instead of transferring.

---

## 4. Deliverables Index

- Full DOCX Report: [docs/CSP-P16B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P16B-E001.docx)
- Raw Per-Seed Results (`480` rows): [P16B_REPLICATION_RESULTS.csv](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_REPLICATION_RESULTS.csv)
- Paired Statistical Comparisons: [P16B_STATISTICAL_ANALYSIS.csv](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_STATISTICAL_ANALYSIS.csv)
- Forensic & Gate Audit: [P16B_FORENSIC_AUDIT.json](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_FORENSIC_AUDIT.json)
- Frozen Configuration: [P16B_CONFIG.json](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_CONFIG.json)
- Reproduction Guide: [P16B_REPRODUCE.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/P16B_REPRODUCE.md)
- Checkpoints & Raw Predictions: `logs/phase16b/checkpoints/` (`80` files) and `logs/phase16b/predictions/` (`80` files).
