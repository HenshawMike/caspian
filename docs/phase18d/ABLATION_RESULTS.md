# Phase 18D (`CSP-P18D-E001`) — Stage 2 Controlled Ablation Results (`Dev Seeds 9401–9405`, $n = 5$)

- **Machine-Readable Log:** [logs/phase18d/p18d_ablation_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18d/p18d_ablation_results.json)
- **Dev Seeds:** `9401–9405` (strictly disjoint from confirmatory seeds `231–245`).

| Regime | `full_rec` (`x_h1`) | `uncond` (`x_h1`) | `conseq_only` (P18C) | `dyn+bnd` (`x_h1`) | `selective` (`x_h1`) | **`uncertainty_guided`** | `fair_mlp` (`x_h1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A (Rep Shift)** | `0.08965` | `0.33723` | `0.08578` | `0.02466` | `0.02259` | **`0.02259`** (`ret 464.5`) | `0.11825` (`ret 435.3`) |
| **Regime B (Boundary Shift)** | `0.16268` | `0.54910` | `0.16841` | `0.26402` | `0.12175` | **`0.15706`** *(refined to 35th-pct $m_a^2$)* | `0.20458` |
| **Regime C (Partial Mut)** | `0.10622` | `0.46170` | `0.11656` | `0.34882` | `0.08887` | **`0.08086`** (`100%` mut detected) | `0.11829` |
| **Regime D (Full Mut)** | `0.12090` | `1.02002` | `0.11952` | `0.78917` | `0.11939` | **`0.11939`** (`0.0%` harmful xfer) | `0.23060` |

## Key Stage 2 Takeaways
- **Decision Audit (`uncertainty_guided`):** Action decision accuracy `89.0%`, boundary decision accuracy `75.0%`, Regime A unnecessary reconstruction `0.0%`, Regime C mutated action detected `100.0%` (`5/5`), Regime C unchanged actions preserved `95.0%` (`19/20`), Regime D harmful transfer `0.0%`, forgetting `0.0000`.
- **Stage 2 → Stage 3 Refinement:** Switched $m_a^2$ quantile from `50th` to `35th` percentile so Regime B's `~50%` boundary-clipping rate does not inflate interior action compatibility scores.
