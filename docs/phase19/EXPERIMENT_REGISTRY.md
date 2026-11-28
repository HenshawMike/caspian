# Phase 19 (`CSP-P19-E001`) — Pre-Registered Experimental Protocol: Real-World Transfer & Operational Decision-Support Prototype

- **Experiment ID:** `CSP-P19-E001`
- **Target Branch:** `feature/application-prototype`
- **Application Domain:** Bounded Adaptive Cloud-Compute & Thermal-Load Operations Decision-Support (`prototype/ops_env.py`, `prototype/caspian_ops_agent.py`).
- **Verified Foundation Reused Unmodified:** Phase 17C (`models/memory.py`, `models/transfer_v3.py`), Phase 18–18D (`models/world_model.py`, `models/planner.py`).
- **Exploratory Smoke Seeds:** `9501, 9502, 9503` ($n = 3$)
- **Locked Confirmatory Seeds:** `251, 252, ..., 265` ($n = 15$, strictly disjoint from all prior Phases `0–18D` `1–245`).
- **Statistical Methodology:** Paired two-sided permutation test ($B = 10{,}000$), paired 95% bootstrap CI ($B = 5{,}000$), Cohen's $d_z$, Holm-Bonferroni correction ($\alpha = 0.05$).

## Application Boundaries, Privacy & Human Oversight Constraints
1. **Explicit Action Permissions:** Agent may ONLY emit discrete integer codes in `{0, 1, 2, 3, 4}` (Monitor, Scale-Out, Throttle, Rebalance, Boost-Cooling); invalid actions raise `PermissionError`.
2. **Semantic Firewall:** Agent receives strictly neutral 8D telemetry vectors + scalar operational metric passing `Observation.assert_semantic_purity()` (zero semantic labels, zero hidden state access, zero `cluster_id`).
3. **Privacy Constraint:** Zero raw telemetry logs stored in persistent state (`raw_observations_stored_bytes == 0`).
4. **Human-Oversight Escalation Gate:** When `ood_score > 0.25` or standardized surprise triggers, `CaspianOpsAgent` emits an explicit `escalation_flag = True` advisory for human review rather than acting blindly on unfamiliar telemetry.

## Pre-Registered Hypotheses (`P19-H1`–`P19-H6`)

| Hypothesis | Operational Metric(s) | Pre-Registered Acceptance Criterion ($n = 15$, seeds `251–265`) |
| :--- | :--- | :--- |
| **`P19-H1`** *(Nominal Operational Control)* | Mean episode operational score & SLA breach rate on `Cluster_A` vs conventional baselines | Beats `threshold_autoscaler`, `reactive_controller`, `fair_mlp_controller`, and `random` on operational score (all 95% CIs `> 0`, Holm $p < 0.05$) AND achieves $\ge 85\%$ of `oracle_controller` |
| **`P19-H2`** *(Telemetry / Cluster Migration Transfer)* | Few-shot (`N = 60`) 1-step telemetry nRMSE & operational score on migrated `Cluster_B` (new schema + offset) | Beats `fair_mlp_controller`, `threshold_autoscaler`, and `scratch_caspian` on `Cluster_B` operational score & telemetry nRMSE (all 95% CIs `> 0`, Holm $p < 0.05$) |
| **`P19-H3`** *(Mid-Stream Incident Detection & Recovery)* | Incident detection rate, human-oversight escalation recall, and post-incident MAE recovery on `Cluster_A_incident` | Detects & escalates $\ge 90\%$ of incidents; reduces post-incident prediction MAE by $\ge 50\%$ within 30 observations; beats static (non-adapting) controller ($p < 0.05$) |
| **`P19-H4`** *(Uncertainty-Driven Human Oversight)* | OOD escalation AUROC separating nominal `Cluster_A` telemetry from unfamiliar `Cluster_B` telemetry before migration | Escalation OOD AUROC $\ge 0.95$ and false-escalation rate on nominal operations $< 5\%$ |
| **`P19-H5`** *(Continual Retention & Privacy)* | `Cluster_A` retention forgetting after migrating to `Cluster_B` + incident recovery, autonomous retrieval, and privacy | `0.0000` forgetting on `Cluster_A`, retrieval $\ge 95\%$, `0 B` raw telemetry stored |
| **`P19-H6`** *(Operational Reliability & Cost)* | Mean per-decision inference latency, persistent footprint, determinism, and AST leakage | Mean decision latency $< 15\text{ ms/step}$, persistent footprint $< 50\text{ KB}$, `determinism_diff = 0.0`, `0` AST violations |
