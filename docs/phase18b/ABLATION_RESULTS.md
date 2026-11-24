# Phase 18B (`CSP-P18B-E001`) — Stage 3–4 Controlled Ablation Results (`Dev Seeds 9301–9305`, $n = 5$)

- **Machine-Readable Log:** [logs/phase18b/p18b_ablation_results.json](file:///home/henshawmike/Projects/Project_Caspian/logs/phase18b/p18b_ablation_results.json)
- **Seeds:** `9301, 9302, 9303, 9304, 9305` (strictly disjoint from locked confirmatory seeds `191–205`).

| Variant | `x_nRMSE (H=1)` | `x_nRMSE (H=3)` | Interior `H=1` | Boundary `H=1` | Recovery Reduction | Post-Rec MAE | Post-Rec `x_rmse` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`p18_baseline`** (`False, False`) | `0.05296` | `0.07871` | `0.05106` | `0.10722` | `33.7%` | `3.991` | `0.3518` |
| **`boundary_only`** (`True, False`) | **`0.05004`** | **`0.06927`** | **`0.05001`** | **`0.06663`** (`-37.9%`) | `34.0%` | `4.006` | `0.3473` |
| **`basis_sync_only`** (`False, True`) | `0.05296` | `0.07871` | `0.05106` | `0.10722` | **`82.0%`** (`+48.3 pts`) | **`1.064`** (`3.75x`) | **`0.0458`** (`7.68x`) |
| **`combined_p18b`** (`True, True`) | **`0.05004`** | **`0.06927`** | **`0.05001`** | **`0.06663`** | **`85.9%`** (`+52.2 pts`) | **`0.863`** (`4.62x`) | **`0.0337`** (`10.44x`) |

## Attribution Conclusions
1. **Boundary-Aware $O(K)$-Invariant Degree-2 Dynamics (`boundary_only`):** Solely responsible for cutting boundary 1-step nRMSE by **`37.9%`** (`0.10722 -> 0.06663`) and 3-step open-loop nRMSE by **`12.0%`** (`0.07871 -> 0.06927`), with zero effect on mutation recovery (`34.0%`).
2. **Atomic Affine-Orthogonal Basis Synchronization (`basis_sync_only`):** Solely responsible for eliminating the `7.68x` post-mutation dynamics collapse (`0.3518 -> 0.0458`) and raising mutation recovery reduction from `33.7%` to **`82.0%`**.
3. **Combined Synergy (`combined_p18b`):** Preserves both improvements simultaneously and yields additional recovery synergy (`85.9%`, `post_rec_mae = 0.863`, `post_rec_x_rmse = 0.0337`, **`10.44x` lower** than Phase 18 baseline).
