# Phase 18D (`CSP-P18D-E001`) — Stage 0 Forensic Architecture Audit

- **Branch:** `feature/selective-transfer` (branched from `main` `3ee8c8e`)
- **Immutable Baselines Preserved:** Phase 17C (`models/memory.py`, `models/transfer_v3.py`), Phase 18 (`f059132`), Phase 18B (`b187add`), Phase 18C (`c603ed8`, `bfba922`).

## 1. Execution Path Trace (`models/world_model.py` & `models/memory.py`)

When `CaspianWorldModel.observe_transitions(X_t, A_t, X_next, dE_t)` encounters a new shifted environment (`W18_gamma_shifted`, `N = 60` transitions) after learning `W18_alpha` (`N = 350`):
1. `self.memory.observe_world(X_next, dE_t)` runs Phase 17C's `ModularTransferModelV3.adapt_few_shot()`, aligning `W_adapter` to minimize **scalar consequence prediction error** against `w_core`, setting `engram.source_engram_id = eid_alpha` and `core_gate = 1.0`.
2. `CaspianWorldModel` encodes `Z_t, Z_next` in the new engram's `W_adapter` basis.
3. Because `len(Z_t) = 60 < 80`, `_fit_orthogonal_box_boundary` is skipped (`U_box = I`, `z_min/z_max = min/max ± 0.25`).
4. Because `len(idx_a) ~ 12 >= K + 3 = 7`, `theta_trans` is refit from scratch per action on ~12 transitions (`src_dyn` is never used).

## 2. Critical Stage 0 Empirical Finding (Refuting Naive Direct Transfer!)

Read-only forensic trace on dev seeds `9401–9405` (`W18_alpha (350) -> W18_gamma_shifted (60)`):

| Seed | `source_engram_id` | `core_gate` | Latent Alignment RMSE (`Z_a` vs `Z_g`) | Phase 18C Refit `x_nRMSE` | Naive Direct `dyn_a` Transfer `x_nRMSE` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `9401` | `0` | `1.0` | **`1.7888`** | `0.11686` | **`0.80292`** (`6.87x` worse!) |
| `9402` | `0` | `1.0` | **`0.9934`** | `0.09999` | **`0.60418`** (`6.04x` worse!) |
| `9403` | `0` | `1.0` | **`0.7570`** | `0.08761` | **`0.50107`** (`5.72x` worse!) |
| `9404` | `0` | `1.0` | **`1.0625`** | `0.09092` | **`0.59334`** (`6.53x` worse!) |
| `9405` | `0` | `1.0` | **`0.6440`** | `0.10187` | **`0.40187`** (`3.94x` worse!) |

### Root-Cause Discovery (Consequence-Symmetry Non-Identifiability)
Aligning `W_adapter` via the scalar consequence function $R_\alpha(s) = 3.5 s_0 + 3.5 s_1 + 6.0 s_0 s_1 + 1.2 s_2 + 0.8 s_3 - 0.4(s_0^2 + s_1^2)$ is **only identifiable up to the stabilizer (symmetry group) of $R_\alpha$** (notably swapping $s_0 \leftrightarrow s_1$ leaves $R_\alpha$ identically invariant!). Therefore, `Z_g` and `Z_a` are related by a **non-trivial residual orthogonal transformation $Q_{\text{dyn}} \in O(K)$ and shift $c_{\text{dyn}}$** (`align_rmse = 0.64–1.79`), even though `w_core` transfers cleanly!

**Implication:** Unconditionally copying `src_dyn` into `PersistentWorldMemory`'s consequence-aligned coordinate frame is mathematically invalid and catastrophically harmful (`4x–7x` higher error). Dynamics and boundary transfer **requires** an explicit dynamics-frame alignment $(Q_{\text{dyn}}, c_{\text{dyn}})$ followed by **component-level Bayesian compatibility testing** before deciding what to preserve, adapt, reconstruct, or reject.
