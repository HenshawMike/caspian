# Phase 18B (`CSP-P18B-E001`) — Stage 1 Architectural Audit

- **Immutable Phase 18 Baseline Commit:** `f05913200ef3a7679087bfe06005386984b55d4e` (`main`: `3804c5f`).
- **Untouched Phase 17C Modules:** `models/memory.py` (`PersistentWorldMemory`), `models/transfer_v3.py` (`ModularTransferModelV3`).
- **Phase 18 Integration Modules:** `models/world_model.py` (`CaspianWorldModel`, `BayesianDynamicsEngram`), `models/planner.py` (`AutonomousPlanner`).

## 1. Exact Dependency Map Between Components

| Dependent Component | Upstream Dependency | Basis Coupling in Phase 18 (`f059132`) | Defect Identified |
| :--- | :--- | :--- | :--- |
| `BayesianDynamicsEngram.theta_trans` | `WorldEngram.model.(x_mean, W_adapter, b_adapter)` | Implicitly parameterized in $W_{\text{old}}$ | **Unsynchronized:** `PersistentWorldMemory.observe_world(..., allow_update_outdated=True)` replaces `W_adapter` with $W_{\text{new}} = W_{\text{old}} Q$ (`||Q - I||_F = 2.47–3.50`), causing `reuse_trans` check to fail (`0/7` seeds) and discarding 350-step dynamics |
| `BayesianDynamicsEngram.cov_trans` & `sigma2_trans` | `W_adapter` basis | Diagonal `sigma2_trans [A, K]` in $W_{\text{old}}$ | Diagonal noise covariance is **not** closed under $Q \in O(K)$ rotations; requires full $K \times K$ covariance `sigma_cov_trans [A, K, K]` |
| `BayesianDynamicsEngram.z_min`, `z_max` | Observed `Z_both` | `min(Z) - 0.25`, `max(Z) + 0.25` | Artificial `±0.25` slack disables clamping at actual empirical boundary; meanwhile `20.3%` clipped training transitions attenuate interior linear drift `4.43x` (`0.0109 -> 0.0484`) |
| `AutonomousPlanner.plan_action` | `CaspianWorldModel.rollout_latent` | No pre-planning basis-version check | Could silently roll out an out-of-sync dynamics engram if memory adapter changed |

## 2. Backward Compatibility & Toggle Discipline
To guarantee controlled ablations against the exact untouched Phase 18 baseline (`f059132`), `CaspianWorldModel` exposes explicit boolean feature flags:
- `boundary_aware: bool = True` (Stage 3–4 ablation toggle; `False` reproduces exact Phase 18 linear + `0.25` slack behavior)
- `basis_sync: bool = True` (Stage 3–4 ablation toggle; `False` reproduces exact Phase 18 unsynchronized behavior)
`models/memory.py` and `models/transfer_v3.py` remain **100% unmodified**.
