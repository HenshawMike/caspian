# Phase 18B (`CSP-P18B-E001`) — Research Specification

- **Experiment ID:** `CSP-P18B-E001`
- **Parent Experiment (Immutable):** `CSP-P18-E001` (`f059132`, `4/7` hypotheses `MET`, `logs/phase18/p18_confirmatory_results.json`)
- **Verified Foundation (Untouched):** Phase 17C (`models/memory.py`, `models/transfer_v3.py`)
- **Development Branch:** `feature/planning-sync`
- **Research Objective:** Resolve from first principles the three limitations exposed by Phase 18 — (A) boundary clipping attenuation (`P18-H1`), (B) memory–dynamics coordinate-basis desynchronization (`P18-H6`), and (C) multi-regime planning comparison against fair MLP baselines (`P18-H5`) — and verify whether the resulting integrated system maintains exact mathematical coherence across representation updates, Bayesian uncertainty estimation, causal dynamics, persistent memory, and autonomous planning.

## Core Scientific Questions
1. **Boundary-Aware Prediction:** Does empirical censored-interior dynamics fitting + tight empirical envelope clamping + boundary-crossing uncertainty inflation eliminate the `4.43x` interior attenuation and `2.47x` boundary overshoot without any oracle knowledge of environment bounds?
2. **Memory–Dynamics Basis Synchronization:** When `PersistentWorldMemory` rotates `W_adapter` by $Q \in O(K)$ during an engram update, does applying the exact affine-orthogonal transformation law $(\Theta'_a = T^{-1}\Theta_a Q,\; \Sigma'_{a,j} = T^{-1}(\sum_k Q_{kj}^2 \Sigma_{a,k}) T^{-\top},\; \Sigma'_{\epsilon,a} = Q^\top \Sigma_{\epsilon,a} Q)$ preserve machine-precision predictive and covariance equivalence and restore robust mutation recovery?
3. **Multi-Regime Fair Planning:** Across low-data (`N=80`), medium-data (`N=160`), abundant-data (`N=350`), shifted, and mutated regimes, where does Caspian's structured world model genuinely outperform an 800-parameter MLP world model, and where does the MLP reach statistical parity?
