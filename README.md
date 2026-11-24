# Caspian

## Artificial Ontological Intelligence Through Artificial Reality

> **Can an artificial agent develop useful internal representations of a world through experience, without being explicitly given the human semantic categories used to describe that world?**

---

# 1. Overview

**Caspian** is an experimental artificial agent and scientific research programme designed to investigate a foundational question about intelligence:

> **Can an AI construct its own useful understanding of an environment through perception, memory, prediction, action, and consequences?**

Rather than starting with a large language model trained on human-labeled text or a complex real-world simulator, Caspian begins with a deliberately small, deterministic 2D simulated world and a **from-scratch NumPy neural network stack** with analytical backpropagation.

In this world, Caspian perceives strictly neutral numerical observations, takes discrete actions, observes state transitions, maintains internal state, and learns predictive and structural relationships. Entities, resources, obstacles, and interaction rules exist in the environment, but **their human semantic meanings are never exposed to the agent**.

For example, the researcher may know that an entity increases the agent's internal energy variable, but Caspian is never told:

```text
"This is food."
```

Instead, Caspian experiences only:

```text
observation
    ↓
interaction
    ↓
consequence
    ↓
memory
    ↓
prediction
    ↓
future action
```

The purpose of Project Caspian is not to prematurely claim AGI, consciousness, or commercial autonomy. Its purpose is to rigorously measure—across pre-registered, multi-seed, baseline-controlled experiments—the **mechanisms by which an artificial learner can construct internal representations, categories, causal models, and world models from raw interaction**.

---

# 2. The Central Idea

Most contemporary AI systems depend heavily on human-designed representations:

$$
\text{Human-generated data \& labels}
\rightarrow
\text{Training}
\rightarrow
\text{Model}
\rightarrow
\text{Predictions}
$$

Caspian investigates a different loop:

$$
\text{Environment}
\rightarrow
\text{Experience}
\rightarrow
\text{Internal representation}
\rightarrow
\text{Prediction}
\rightarrow
\text{Action}
\rightarrow
\text{New experience}
$$

Humans do not explicitly tell Caspian:

* what objects are called,
* which objects belong together,
* which properties matter,
* which relationships are additive or interactive,
* what causes what.

Instead, the agent must discover—or fail to discover—useful environmental regularities through interaction, and every positive or negative finding is preserved in the permanent scientific record.

---

# 3. Research Direction: Artificial Ontological Intelligence (AOI)

The broader research direction is provisionally called **Artificial Ontological Intelligence (AOI)** (used as a working research label, not a claim of an already-established field).

### Working Definition

> **Artificial Ontological Intelligence is the study and engineering of artificial agents capable of constructing useful internal representations, categories, relationships, and world models of an environment through autonomous interaction rather than relying on predefined human semantic categories.**

In this context, **ontology** means the agent's internal organization of:

* entities and continuous/discrete properties,
* functional consequences and state transitions,
* spatial and temporal persistence,
* additive and non-additive interactions,
* causal dependencies under intervention,
* reusable structure across unfamiliar worlds.

---

# 4. Core Research Questions

$$
\boxed{
\text{Can an artificial agent construct a useful ontology from experience?}
}
$$

More specifically:

1. **Q1 — Representation:** Can Caspian develop internal representations ($S_t$) that correspond to meaningful environmental regularities without receiving property labels?
2. **Q2 — Prediction:** Can Caspian accurately predict future environmental states ($\hat{O}_{t+1}$ or $\Delta E$) across unseen trajectories and positions?
3. **Q3 — Memory:** Can previous experience retained in recurrent state ($h_t$) or replay buffers bridge partial observability when current observations are insufficient?
4. **Q4 — Adaptation:** Can Caspian detect when environmental rules change mid-stream and adapt online without catastrophic forgetting?
5. **Q5 — Generalization & Compositionality:** When does a model learn a reusable compositional rule versus memorizing observed combinations, and how does architectural inductive bias govern this?
6. **Q6 — Concept & Category Formation:** Under rigorous criteria (predictiveness, stability, compression, intervention sensitivity, zero-shot transfer), when do internal representations form genuine functional categories?
7. **Q7 — Causal Discovery:** Can active interventions ($do(X=x)$) separate true causal structure from spatially or observationally confounded correlations?
8. **Q8 — World Modeling & Planning:** Can multi-step open-loop rollouts of a learned world model support receding-horizon action selection?

---

# 5. What Caspian Is NOT

Caspian is not:

* a chatbot or large language model wrapper,
* a claim of machine consciousness,
* an unrestricted computer-control or internet agent,
* a self-replicating system.

In particular, Project Caspian enforces a strict scientific boundary throughout all phases:

$$
\boxed{
\text{Complex adaptive behavior} \neq \text{proof of consciousness}
}
$$

$$
\boxed{
\text{Probe decodability} \neq \text{compositional understanding}
}
$$

---

# 6. First-Principles Mathematical Model

The environment-agent loop is formulated as a discrete-time dynamical system.

Let $E_t$ denote the ground-truth state of the environment at timestep $t$. Caspian receives a strictly neutral observation through the observation function $G$:

$$
O_t = G(E_t)
$$

Caspian updates its internal representation $S_t$:

$$
S_t = F_\theta(S_{t-1}, O_t, A_{t-1})
$$

and selects an action $A_t \in \mathcal{A}$ via policy $\pi_\theta$:

$$
A_t = \pi_\theta(S_t)
$$

The environment transitions according to dynamics $T$:

$$
E_{t+1} = T(E_t, A_t), \qquad O_{t+1} = G(E_{t+1})
$$

forming the closed loop:

$$
\boxed{
E_t
\rightarrow
O_t
\rightarrow
S_t
\rightarrow
A_t
\rightarrow
E_{t+1}
\rightarrow
O_{t+1}
}
$$

### Neural & Compositional Architectures Implemented (`models/`)

All models in Project Caspian are implemented **from scratch in pure NumPy** with analytical forward and backward passes (verified against numerical finite-difference gradients in `tests/`):

1. **`PredictiveMLP` ([models/mlp.py](file:///home/henshawmike/Projects/Project_Caspian/models/mlp.py)):** Multi-layer feedforward encoder and linear readout head with input `FeatureNormalizer` and latent state extraction `get_latent_state(x)`:
   $$h_1 = \sigma(x W_1 + b_1), \quad S_t = \sigma(h_1 W_2 + b_2), \quad \hat{y}_t = S_t W_3 + b_3$$
2. **`RecurrentPredictiveModel` ([models/rnn.py](file:///home/henshawmike/Projects/Project_Caspian/models/rnn.py)):** Elman recurrent network with $\tanh$ state update and full Backpropagation Through Time (BPTT) over episode sequences:
   $$h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h), \quad \hat{y}_t = h_t W_{hy} + b_y$$
3. **`DeepSetsPredictiveModel` ([models/deepsets.py](file:///home/henshawmike/Projects/Project_Caspian/models/deepsets.py)):** Permutation-invariant set encoder ($\phi$) with masked per-channel max-pooling and prediction head ($\rho$) for variable entity counts.
4. **Compositional & Inductive-Bias Models ([models/compositional.py](file:///home/henshawmike/Projects/Project_Caspian/models/compositional.py)):**
   * `mlp_small` / `mlp_large`: Joint concatenation MLPs of varying capacity.
   * `factorised`: Per-property encoder encoders concatenated into a joint readout head.
   * `additive_mlp`: Per-property MLP heads summed into a strictly additive output ($\hat{y} = b + \sum_k f_k(p_k)$).
   * `additive`: Strictly additive linear/nonlinear per-channel predictors.
   * `additive_residual`: Strictly additive main-effect trunk summed with a factorised interaction residual branch ($\hat{y} = f_{\text{add}}(p) + f_{\text{res}}(p)$), introduced in Phase 12.

---

# 7. The Semantic Firewall

A core engineering and scientific invariant of Project Caspian is the **Semantic Firewall** ([environment/observations.py](file:///home/henshawmike/Projects/Project_Caspian/environment/observations.py)).

* **Ground Truth (`EnvironmentState` / `Entity`):** May hold internal simulation parameters such as `hidden_state_delta = +10.0` or `flicker_period = 4`.
* **Agent Observation (`Observation`):** Exposes only neutral physical measurements:
  * Agent coordinates $(x_t, y_t)$, collision flag $c_t \in \{0, 1\}$, internal state level $e_t$, timestep $t$, previous action code $a_{t-1}$.
  * Visible entities sorted nearest-first: `[entity_id, entity_type, relative_x, relative_y, distance]` plus optional continuous neutral `properties` channels.
  * Discrete action space ([environment/actions.py](file:///home/henshawmike/Projects/Project_Caspian/environment/actions.py)): `UP (0)`, `DOWN (1)`, `LEFT (2)`, `RIGHT (3)`, `INTERACT (4)`, `NOOP (5)`.

Every constructed `Observation` automatically runs `assert_semantic_purity()`, recursively scanning all keys and values against `FORBIDDEN_SEMANTIC_TOKENS` (`food`, `hazard`, `danger`, `reward`, `beneficial`, `harmful`, `resource`, `target`, `enemy`, `good`, `bad`, `positive`, `negative`, `expected_outcome`, `goal`, `threat`) and raising `SemanticFirewallViolation` if any semantic leakage is present.

---

# 8. Experimental Progression & Evidence Ledger (Phases 0–12)

Project Caspian uses a two-layer documentation standard:
* **Layer A:** [Caspian_Agent_Driven_Development_Plan.md](file:///home/henshawmike/Projects/Project_Caspian/Caspian_Agent_Driven_Development_Plan.md) — the consolidated master specification and roadmap for Phases 0–20.
* **Layer B:** Permanent `.docx` scientific experiment reports in `docs/` (`CSP-P1-E001.docx` through `CSP-P12-E001.docx`) and pre-registered Markdown protocols in `docs/phase10/`, `docs/phase11/`, and `docs/phase12/`.

### Summary of Completed Phases (0–12)

| Phase | Title | Core Research Question | Key Outcome & Evidence Status | Report / Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | **Research & Safety Foundation** | Can we build a 100% deterministic, semantic-firewall-audited 2D environment and baseline suite? | **Established:** 100% bitwise trajectory reproducibility (`ce150a82...`), 0 semantic leakages, `RandomAgent` vs. `OracleAgent` baselines verified. | [baseline_comparison.py](file:///home/henshawmike/Projects/Project_Caspian/experiments/baseline_comparison.py) |
| **Phase 1** | **Predictive Interaction** | Can a small neural agent learn an unlabeled environmental interaction rule ($\Delta E$) from experience? | **Supported:** `PredictiveMLP` ($R^2 = 0.6026$ held-out; $0.8799$ at 1,000 steps) beat Persistence, Reactive, and Random baselines across 8 seeds with 0 leakage. | [CSP-P1-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P1-E001.docx) |
| **Phase 2** | **Memory & Persistence** | Can recurrent state retain entity information across occluded/flickering timesteps? | **Supported in code/logs:** `RecurrentPredictiveModel` (Elman BPTT) retains occluded entity state where stateless MLPs fail *(primary narrative report noted for reconciliation)*. | [CSP-P2-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P2-E001.docx) |
| **Phase 3** | **Generalization** | Does learned prediction transfer to disjoint positions, obstacle layouts, and variable entity counts? | **Supported (with caveats):** Nearest-first slot ordering + `DeepSetsPredictiveModel` enable position and multi-entity transfer, though multi-entity variance across seeds remains substantial. | [CSP-P3-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P3-E001.docx) |
| **Phase 4** | **World Model & Planning** | Can Caspian predict full state transitions $[\Delta x, \Delta y, \Delta E]$ and plan over multi-step open-loop rollouts? | **Preliminary Support:** 3-dim state transition prediction and MPC-style receding-horizon lookahead (`plan_action`) outperform reactive selection at short horizons. | [CSP-P4-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P4-E001.docx) |
| **Phase 5** | **Emergent Categories** | Do internal representations $S_t$ form functional clusters meeting measurable category criteria? | **Promising (Representation Limit Identified):** Index/one-hot type codes cluster trained types by function and respond to causal sign-flip interventions, but **cannot** zero-shot generalize to unseen type indices; continuous property channels (`E007`) achieve zero-shot functional clustering. | [CSP-P5-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P5-E001.docx) |
| **Phase 6** | **Rule Change & Adaptation** | Can Caspian detect mid-stream rule flips online and recover without catastrophic forgetting? | **Supported:** `OnlineAdaptationTrainer` detects surprise spikes and adapts online after sign flips while frozen pre-change controls confirm true weight adaptation. | [CSP-P6-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P6-E001.docx) |
| **Phase 7** | **Causal Experimentation** | Can active intervention break spatial confounding where passive observation fails? | **Mixed / Cautious:** Active intervention breaks spatial confounds (`E002`, swaps $R^2 \approx 0.99$ vs. passive $\approx 0.48$), but Phase 10 later showed exposure coverage/balance explains much of the gain. | [CSP-P7-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P7-E001.docx) |
| **Phase 8** | **Novel Worlds** | Do learned representations transfer zero-shot or via fine-tuning across 6 structural world shifts? | **Characterised:** Tested across `geometry`, `appearance`, `dynamics`, `rules`, `resources`, and `actions` against scratch and shuffled-pretrain controls. | [CSP-P8-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P8-E001.docx) |
| **Phase 9** | **Scale Study** | How do model capacity (157–13,185 params), data budget (500–96,000 steps), and world complexity (`C1`–`C3`) interact? | **Characterised:** Data budget dominates generalisation gap shrinkage; capacity requirements grow from `C1` (313 params) to `C2` (1,761 params at 32k steps), while `C3` remained below $R^2 = 0.9$ even at 96k steps. | [CSP-P9-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P9-E001.docx) |
| **Phase 10** | **Ontology Discovery & Compositional Audit** | Can standard MLPs autonomously discover properties and extrapolate to held-out combinations? | **Refuted / Critical Diagnostic:** Probes decoded properties from raw/latent states at $R^2 \in [0.96, 1.0]$, yet the MLP prediction head failed to extrapolate compositionally ($R^2 \approx 0.65\text{–}0.83$ vs. oracle $1.0$). Coverage explained active-vs-passive gains. | [CSP-P10-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P10-E001.docx), [docs/phase10/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase10/REPRODUCTION_REPORT.md) |
| **Phase 11** | **Architectural Inductive Bias** | Does architectural structure (`additive`, `factorised`, `additive_mlp`, `mlp_small`, `mlp_large`) govern compositional extrapolation? | **Supported (in Additive Worlds):** Pre-registered across 10 seeds in `W1` and `W2`. Strict additive architecture achieved near-zero held-out nRMSE ($0.001$ / $0.000$) vs. `mlp_small` ($0.183$) and `mlp_large` ($0.115$), proving structure—not mere capacity—drives additive extrapolation; fails as predicted when unidentifiable interaction terms exist. | [CSP-P11-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P11-E001.docx), [docs/phase11/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase11/EXPERIMENT_REGISTRY.md) |
| **Phase 12** | **Non-Additive Interactions & Optimizer Sensitivity** | Can an `additive_residual` architecture handle identifiable non-additive interactions (`W3`: $Y = 6A - 4B + 2C + 5AB$) while preserving additive inductive bias, and is the ranking robust across optimizer grids? | **Supported:** On disjoint seeds `11–20`, strict `additive` failed on `W3` (nRMSE $0.192$), whereas `additive_residual` achieved the best held-out nRMSE ($0.037$ vs. `mlp_small` $0.179$) while retaining the additive advantage on `W1`. Architecture rankings held across all 6 $(\text{lr}, \text{epochs})$ settings. | [CSP-P12-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P12-E001.docx), [docs/phase12/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase12/EXPERIMENT_REGISTRY.md) |
| **Phase 13** | **Autonomous Property Discovery & Representation Learning** | Can Caspian discover underlying environmental properties from 12D raw entangled sensors (80% nuisance variance) without property labels or pre-separated channels, and use them compositionally? | **Supported (`P13-H1, H2, H3, H6` met; `H4, H5, H7` partially met & resolved in Stage C):** On disjoint seeds `21–30` across `W1_raw`, `W2_raw`, `W3_raw`, orthogonal sub-Gaussian kurtosis unmixing (`latent_additive`, `latent_residual`) recovered true properties in individual slots (`slot_match_r2` $1.000$ in `W1_raw`, $0.995$ post-fix in `W2_raw`), matched the clean-channel oracle ($0.0001$ vs. `raw_mlp_small` $0.108$ and `pca_statistical` $0.257$), and recovered non-additive interactions in `W3_raw` ($0.041$ vs. $0.158$). | [CSP-P13-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P13-E001.docx), [docs/phase13/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase13/INVESTIGATION.md) |
| **Phase 14** | **Interaction Discovery & Structured Relationships** | Can Caspian autonomously discover sparse pairwise and 3-way interactions among $K = 5$ properties under clean and noisy observations without interaction labels, and avoid false interaction hallucinations? | **Supported (`F1 = 1.000` & `0` false positives in `50/50` world-seed cells):** On disjoint seeds `31–40` across 5 worlds (`W14_indep`, `W14_sparse_pair`, `W14_multi_pair`, `W14_higher_order`, `W14_noisy_pair`), `sparse_interaction` (Frisch-Waugh-Lovell + ISTA + Gauss-Lasso) achieved `F1 = 1.000` everywhere, suppressed the `3.9–10.9` false interactions hallucinated by `additive_residual` and MLPs (`P14-H6` met in `10/10` seeds), and cut noisy held-out nRMSE to `0.0054` vs `additive_residual` `0.0378`. | [CSP-P14-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P14-E001.docx), [docs/phase14/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase14/INVESTIGATION.md) |
| **Phase 15** | **Causal Discovery & Intervention Learning** | Can Caspian distinguish spurious observational correlations and hidden common causes from true direct causal effects via controlled interventions? | **Supported (`causal_f1 = 1.000` in `40/40` cells; `P15-H2, H3, H4` met):** On disjoint seeds `41–50` across 4 causal worlds (`W15_corr_vs_cause`, `W15_obs_confounder`, `W15_hidden_confounder`, `W15_multi_cause`), `CausalStructuralModel` recovered exact $\text{Pa}(Y)$ with `0` false edges, beat observational baselines by `0.55` nRMSE under hidden confounding $U$, and solved Pearl's Abduction-Action-Prediction counterfactuals (`0.051` vs `0.57` nRMSE). | [CSP-P15-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P15-E001.docx), [docs/phase15/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase15/INVESTIGATION.md) |
| **Phase 16** | **Cross-World Transfer & Structural Generalisation** | Can Caspian separate reusable structural rules from world-specific sensor adapters and residual rules for few-shot cross-world transfer without forgetting or negative transfer? | **Stage B `NOT MET` $\rightarrow$ Stage C Resolved via First-Principles Fixes:** Disjoint seeds `51–60` (`480` evals) exposed missing $z_k^2$ rotational invariance, noisy 20-shot mean refitting, and rank-deficient fixed ridge. Upgraded `ModularTransferModel` (`66` params vs `833`-param MLP) cut 20-shot error `3.4x–6.0x` across all 4 worlds (`0.0150` on `W16_obs_shift`, `0.0277` on `W16_shared_plus_new`, `0.0301` on `W16_rule_shift`, `0.0339` on `W16_negative_control` with `10/10` gate OFF), resolved `P16-H5` to `MET` (`0.0011` source retention), and beat both MLPs in mean nRMSE on all 4 worlds. | [CSP-P16-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P16-E001.docx), [docs/phase16/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16/INVESTIGATION.md) |
| **Phase 16B** | **Independent Replication, Forensic Audit & Robustness (`CSP-P16B-E001`)** | Do Phase 16 `v2` claims survive a frozen-model, standalone independent harness on fresh seeds (`101–110`) with zero oracle flags and boundary stress tests? | **`PARTIAL PASS` (Gates A, B, E, F `PASS`; Gates C, D `FAIL` on strictest sub-checks):** Verified `66` params, `0` leaks, `0.0` subprocess reproducibility diff, `0.0012` source retention (`10/10 < 0.02`), `10/10` negative-transfer gate rejection, `40/40` autonomous shift detection, and underdetermined (`8–16` shot / duplicated) transfer gains over fair 66-param `v2 scratch_structural`. Honestly exposed 1 low-coverage outlier seed (`0.2013` on `W16_negative_control`, causing CI vs `scratch_mlp` to cross 0), 20-shot tie vs `v2 scratch_structural`, and boundary sensitivities (`c = 0.5` additive offset, in-subspace rotation, `W16_obs_shift` sample ordering). | [CSP-P16B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P16B-E001.docx), [CSP-P16B-E001.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16b/CSP-P16B-E001.md) |
| **Phase 16C** | **First-Principles Resolution of Phase 16B Issues (`CSP-P16C-E001`, `v3`)** | Do first-principles fixes (Empirical Bayes replicate-noise ridge, offset-invariant alignment, canonical SVD ordering, dual-trigger $\chi^2$ rotation detection) resolve Gate D and all Phase 16B boundary failures on fresh unseen seeds (`121–130`)? | **Resolved (`Gate D PASS` & `4/4` worlds beat both MLPs with 95% CI `> 0` on both `101–110` and fresh `121–130`):** Keeping `v2` (`models/transfer.py`) bitwise frozen, `ModularTransferModelV3` (`models/transfer_v3.py`, still `66` params) eliminated seed 107's leverage blowup (`0.2013 -> 0.0161` mean), reduced sample-ordering spread to `2.75e-16`, achieved exact invariance to additive offset `c = 0.5` (`10/10` gate ON), and detected `8/10` in-subspace pure rotations. | [CSP-P16C-E001.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase16c/CSP-P16C-E001.md), [transfer_v3.py](file:///home/henshawmike/Projects/Project_Caspian/models/transfer_v3.py) |
| **Phase 17** | **Continual Learning & Persistent Environmental Memory (`CSP-P17-E001`)** | Can Caspian retain multiple sequential worlds without catastrophic forgetting, retrieve the right world autonomously with zero `world_id` privilege, detect outdated rules, and transfer forward under scarcity? | **Substantially Supported (`P17-H2, H4, H5` `MET`; `P17-H1, H3` diagnosed & improved in Stage C):** On disjoint seeds `71–80` across a 5-stage stream (`60` runs, `0` leaks, zero `world_id` privilege), `PersistentWorldMemory` (`264` total params, `0` raw replay) achieved **`0.0000` forgetting** on `W17_alpha`/`W17_beta` (`0.0019` retention), **`95.0%`** autonomous 8-probe retrieval (`38/40`), and **`10/10`** outdated-memory detection on `W17_beta_mutated` (`0.0040` nRMSE). Stage C isolated 10-DOF overfitting on 10-shot `W17_gamma_shared` ($G \approx 7$) and implemented Occam-Hierarchy 6-DOF-first alignment. | [CSP-P17-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17-E001.docx), [docs/phase17/](file:///home/henshawmike/Projects/Project_Caspian/docs/phase17/INVESTIGATION.md) |
| **Phase 17B** | **Occam-Hierarchy Independent Verification (`CSP-P17B-E001`)** | Does Occam-Hierarchy 6-DOF-first alignment reliably fix 10-shot Gamma transfer on fresh unseen seeds (`131–145`, $n = 15$) while preserving Phase 16C invariances and retention? | **`Outcome C — No Reliable Improvement` on Primary 10-Shot (`P17B-H3, H5` `MET`; `P17B-H1, H2, H4` `NOT MET`):** Forensic audit confirmed `G = 5..8` (`10-DOF` underdetermined + seed 77 `rank=3` collapse). On `15` fresh seeds, 10-shot Gamma improved directionally (`0.1335 -> 0.1030`, `9/15` wins) but CI `[-0.0117, +0.0716]` crossed 0 (`Holm p = 0.3354`); retention (`0.0000` forgetting, `96.7%` retrieval, `3,676 B` total persistent footprint) and determinism (`0.0` diff) passed cleanly; additive offset (`10/10`) preserved, while in-subspace rotation dropped to `4/10`. Secondary exploratory 12-shot ($G \approx 9$) showed strong `3.2x` transfer (`0.0385` vs `0.1241`, `14/15` wins, `p = 0.00195`). | [CSP-P17B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17B-E001.docx), [P17B_FORENSIC_AUDIT.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase17b/P17B_FORENSIC_AUDIT.md) |
| **Phase 17C** | **First-Principles Root-Cause Resolution (`CSP-P17C-E001`)** | Do the three root-cause fixes (unrotated `xm=0` in-subspace check, orthogonal-subspace offset guard, and Phase 13 kurtosis-regularized $O(4)$ alignment) resolve Phase 17B on brand-new unseen seeds (`151–165`, $n = 15$)? | **`Outcome B — Partial Improvement` (`4/5` `MET`: `P17B-H2, H3, H4, H5` `MET`):** On fresh seeds `151–165`, 10-shot Gamma dropped to **`0.06306`** (beating `memoryless_structural` `0.14228`, `12/15` wins, `Holm p = 0.0155`, and all 3 MLPs, **`P17B-H2` `MET`**), 4-world retention hit **`0.01717`** with **`100.0%` (`60/60`)** zero-`world_id` retrieval (`P17B-H3` `MET`), and all 4 invariances passed **`40/40`** (`W16_in_subspace_rot: 4/10 -> 10/10`, `mean 0.0066`, **`P17B-H4` `MET`**); vs `original_v3` (`0.12285`), `delta = +0.0598`, 95% CI `[+0.0069, +0.1166] > 0`, `Holm p = 0.061`. | [CSP-P17C-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P17C-E001.docx), [CSP-P17C-E001.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase17c/CSP-P17C-E001.md) |
| **Phase 18** | **Integrated Predictive World Model & Autonomous Planning (`CSP-P18-E001`)** | Can Caspian integrate Phase 17C `PersistentWorldMemory` with action-conditioned causal transitions, Bayesian uncertainty, counterfactual rollouts, and autonomous multi-step planning? | **Substantially Supported (`4/7` `MET`: `P18-H2, H3, H4, H7` `MET`; `P18-H1, H5, H6` diagnosed in Stage C):** On 15 disjoint unseen seeds (`171–185`), `CaspianWorldModel` + `AutonomousPlanner` achieved causal action **`F1 = 0.921`** (`5.24x` lower interventional error, `15/15`), internal counterfactual **Spearman $\rho = 0.980$ / Top-1 `96.2%`**, **OOD AUROC `1.000`** (`r = 0.617`, risk-aware beat risk-neutral in cliff trap `p = 0.0021`), **`95.6%` of oracle** on saddle-crossing planning (`442.98` vs `memory_no_planner` `105.14`, $d_z = 25.77$), **`0.0000` forgetting / `100%` zero-`world_id` retrieval**, and shifted-world planning superiority over `mlp_world_model` (`425.00 vs 359.86`, `Holm p = 0.0437`). | [CSP-P18-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18-E001.docx), [INVESTIGATION.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase18/INVESTIGATION.md) |
| **Phase 18B** | **Boundary-Aware Prediction & Memory–Dynamics Basis Synchronization (`CSP-P18B-E001`)** | Can Caspian preserve exact predictive & Bayesian covariance coherence across orthogonal memory updates while resolving boundary attenuation and evaluating multi-regime MLP parity? | **Strongly Supported (`6/7` `MET`: `P18B-H2, H3, H4, H5, H6, H7` `MET`; `P18B-H1` missed `<0.05` by `0.00047`):** On 15 brand-new unseen seeds (`191–205`), exact $M(Q,c)$ affine-orthogonal basis sync achieved **`1.78e-15`** mean & **`2.17e-17`** full-covariance equivalence (`0.0` serialization diff), cut post-mutation dynamics error **`10.74x`** (`74.62%` recovery vs `46.39%`, $p=0.0041$), cut boundary error **`39.3%`** (`0.0654 vs 0.1078`, $p=0.0001$), hit **`97.6%` of oracle** (`451.96`), turned trap return **`-26.09 -> +28.69`** ($p=0.0005$), and beat MLP at `N=80` (`+38.09`, $p=0.0102$), `N=350` (`+17.42`, $p=0.0146$), and shifted worlds (`+37.49`, $p=0.0102$), while honestly reporting `N=160` parity. | [CSP-P18B-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18B-E001.docx), [FINAL_REPORT.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase18b/FINAL_REPORT.md) |
| **Phase 18C** | **Orthogonal Box-Projection Dynamics (`CSP-P18C-E001`)** | Does IRLS + unit-clipping-impulse $\ell_4$ discovery of the intrinsic boundary-face frame $U_{\text{box}} \in O(K)$ resolve `P18B-H1` while preserving exact basis synchronization? | **Resolved `P18B-H1` (`6/7` `MET` on brand-new unseen seeds `211–225`):** Cut 1-step $x$ nRMSE **`4.08x`** vs MLP (**`0.01496 vs 0.06096`**, Holm $p=0.0002$), 3-step **`0.02734 vs 0.11162`**, boundary **`0.03636 vs 0.12125`**, boosted mutation recovery to **`88.21%`** (`14.68x` lower post-rec `x_rmse`), achieved **`99.55%` of oracle** (`461.62`, beating MLP at `N=80, 160, 350`), shrank persistent pickle **`4.34x`** (`20,046 B`, `390` params), and kept exact **`5.08e-15`** mean / **`9.97e-18`** variance basis-sync equivalence (shifted `430.87 vs 397.39`, Holm $p=0.0521$). | [CSP-P18C-E001.docx](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P18C-E001.docx), [CSP-P18C-E001.md](file:///home/henshawmike/Projects/Project_Caspian/docs/phase18c/CSP-P18C-E001.md) |

---

# 9. Key Scientific Lessons From Phases 0–12

1. **Prediction $\neq$ Understanding:** High in-distribution accuracy does not imply the agent has formed a generalizable concept.
2. **Probe Decodability $\neq$ Compositional Use (Phase 10 & 11):** Linear/ridge probes can recover ground-truth properties from a hidden layer with $R^2 \approx 1.0$ even when the network's own readout head fails to combine those properties on held-out combinations.
3. **Coverage Can Imitate Discovery (Phase 10):** Active policies often beat passive policies simply because they achieve more balanced state-action coverage, not necessarily because they induce superior internal representations.
4. **Scale Alone Does Not Replace Inductive Bias (Phase 9, 11, 12):** Scaling MLP width from 157 to 13,185 parameters (or tuning learning rates/epochs) does not close the compositional extrapolation gap that an appropriate structural bias (`additive` or `additive_residual`) closes with far fewer parameters.
5. **Inductive Bias Must Match Environmental Structure (Phase 11 & 12):** Strict additivity excels when effects are independent (`W1`, `W2`) but distorts main effects when genuine interactions exist (`W3`), motivating structured residual architectures (`additive_residual`).

---

# 10. Forward Research Roadmap (Phases 13–20)

As specified in [Caspian_Agent_Driven_Development_Plan.md](file:///home/henshawmike/Projects/Project_Caspian/Caspian_Agent_Driven_Development_Plan.md#L1701-L2357), the next eight phases investigate:

* **Phase 13 — Autonomous Property Discovery and Representation Learning:** Remove manually separated property channels; test whether Caspian can discover latent environmental properties from raw/entangled observations and use them compositionally.
* **Phase 14 — Interaction Discovery and Structured Relationships:** Identify sparse pairwise and higher-order interactions among multiple environmental properties under observation noise.
* **Phase 15 — Causal Discovery and Intervention Learning:** Distinguish correlation, confounding, and hidden variables from direct causal effects via controlled single- and multi-variable interventions.
* **Phase 16 — Cross-World Transfer and Structural Generalisation:** Separate reusable structural knowledge from world-specific parameters across multiple independent environments (zero-shot and few-shot transfer vs. negative transfer).
* **Phase 17 — Continual Learning and Persistent Environmental Memory:** Accumulate knowledge across sequential environments while mitigating catastrophic forgetting.
* **Phase 18 — Integrated Predictive World Model and Autonomous Planning:** Combine discovered representations, persistent memory, and uncertainty-aware multi-step rollouts for closed-loop decision-making.
* **Phase 19 — Real-World Transfer and Application Prototype:** Evaluate transfer of validated world-modeling capabilities to a bounded, measurable operational or cybersecurity domain under strict safety and governance constraints.
* **Phase 20 — Integrated System Validation and Research Consolidation:** End-to-end evaluation, full component ablations, literature comparison, and technical research paper.

---

# 11. Repository Structure

```text
Project_Caspian/
├── README.md                                    # Project overview, theory, evidence ledger, and guide
├── Caspian_Agent_Driven_Development_Plan.md     # Master Research & Development Specification (Phases 0–20)
├── pyproject.toml                               # Package metadata and pytest configuration
├── requirements.txt                             # Locked reproducible dependency manifest
│
├── environment/                                 # Deterministic 2D GridWorld & Semantic Firewall
│   ├── world.py                                 # GridWorld, WorldConfig, EnvironmentState
│   ├── entities.py                              # Entity definitions (flicker observability, continuous properties)
│   ├── dynamics.py                              # Pure transition functions (movement, collisions, interaction deltas)
│   ├── observations.py                          # Observation, EntityObservation, SemanticFirewallViolation audit
│   └── actions.py                               # Discrete Action enum (UP, DOWN, LEFT, RIGHT, INTERACT, NOOP)
│
├── agent/                                       # Baseline and neural agents
│   ├── base.py                                  # BaseAgent interface
│   ├── random_agent.py                          # Uniform random action baseline
│   ├── oracle_agent.py                          # Privileged ground-truth navigation/interaction reference
│   ├── predictive_agent.py                      # Phase 1 epsilon-greedy predictive MLP agent
│   └── recurrent_agent.py                       # Phase 2 stateful Elman RNN agent
│
├── models/                                      # Pure-NumPy neural network stack (analytical backprop)
│   ├── layers.py                                # LinearLayer (He/Xavier/Uniform/Zeros init, backward pass)
│   ├── activations.py                           # ReLU, Tanh, Sigmoid, Identity + analytical derivatives
│   ├── normalization.py                         # FeatureNormalizer (mean/std standardization)
│   ├── optimizers.py                            # SGD, Adam (with decoupled weight decay), MSE/MAE/Huber losses
│   ├── mlp.py                                   # PredictiveMLP with latent state extraction
│   ├── rnn.py                                   # RecurrentPredictiveModel (Elman RNN + full BPTT)
│   ├── deepsets.py                              # DeepSetsPredictiveModel (permutation-invariant max-pool)
│   └── compositional.py                         # Phase 11–12 architectures (mlp_small/large, factorised, additive, additive_mlp, additive_residual)
│
├── learning/                                    # Data collection, datasets, and offline/online trainers
│   ├── collector.py                             # TrajectoryCollector (exploratory and policy rollouts)
│   ├── dataset.py                               # Transition, ExperienceDataset, state-action & DeepSets encodings
│   ├── sequence_dataset.py                      # EpisodeSequenceDataset for recurrent BPTT training
│   ├── trainer.py                               # PredictiveTrainer (offline minibatch training)
│   ├── recurrent_trainer.py                     # RecurrentTrainer (sequence-level BPTT trainer)
│   ├── deepsets_trainer.py                      # DeepSetsTrainer
│   └── online_trainer.py                        # OnlineAdaptationTrainer & rule-change surprise detection
│
├── evaluation/                                  # Metrics, baselines, and multi-step planning
│   ├── metrics.py                               # MSE, RMSE, MAE, R², cosine similarity
│   ├── baselines.py                             # Zero, Persistence, Mean, ReactiveLookup baselines
│   ├── evaluator.py                             # Standardized agent and predictor evaluation harness
│   └── world_model_rollout.py                   # Open-loop multi-step simulation & MPC lookahead planner
│
├── experiments/                                 # Phase experiment suites, analysis, and DOCX generators
│   ├── safe_io.py                               # Overwrite-protected JSON writer (safe_write_json)
│   ├── logger.py                                # Structured ExperimentLogger with SHA-256 trajectory hashing
│   ├── report_generator.py                      # Automated Layer B DOCX report builder
│   ├── baseline_comparison.py                   # Phase 0 Random vs. Oracle benchmark
│   ├── deterministic_baseline.py                # Phase 0 determinism and semantic firewall verification
│   ├── phase1_experiments.py .. phase12_experiments.py
│   ├── analyze_phase11.py, analyze_phase12.py   # Confirmatory bootstrap CI and hypothesis evaluators
│   └── generate_p0_report.py .. generate_p12_report.py
│
├── configs/                                     # Externalized JSON world and model configurations (Phases 0–7)
├── logs/                                        # Raw JSON results, .npz arrays, checkpoints, and pre-Phase-10 archive
├── docs/                                        # Layer B DOCX reports & phase registration records
│   ├── CSP-P1-E001.docx .. CSP-P12-E001.docx    # Permanent scientific experiment reports
│   ├── development/
│   │   └── BRANCHING_STRATEGY.md                # Git branching and experimental traceability standard
│   ├── phase10/                                 # Code inventory, reproduction report, known issues
│   ├── phase11/                                 # Pre-registered hypotheses, architecture design, operational record
│   └── phase12/                                 # Phase 12 W3 & optimizer sensitivity registry and operational record
│
└── tests/                                       # 117 unit, gradient-check, and regression tests (Phases 0–12)
```

---

# 12. Quickstart & Reproducibility

### 1. Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run the Full Verification & Unit Test Suite

All 117 tests—covering deterministic grid dynamics, semantic firewall enforcement, analytical vs. numerical finite-difference gradients (`LinearLayer`, `PredictiveMLP`, `RecurrentPredictiveModel`, `DeepSetsPredictiveModel`, `AdditiveResidualModel`), and Phase 0–12 invariants—can be executed via:

```bash
.venv/bin/pytest -q
```

### 3. Run Experiments & Analysis

* **Phase 0–9 Suites:**
  ```bash
  .venv/bin/python experiments/phase1_experiments.py
  ```
* **Phase 11–12 Pre-Registered Suites & Analysis:**
  ```bash
  .venv/bin/python -m experiments.analyze_phase11
  .venv/bin/python -m experiments.analyze_phase12
  ```
  *(Note: `experiments/safe_io.py` automatically prevents overwriting existing confirmatory logs in `logs/` by appending a UTC timestamp unless `CASPIAN_ALLOW_OVERWRITE=1` is explicitly set.)*

---

# 13. Git Branching & Experimental Governance

All development and experimentation follow the governance rules in [docs/development/BRANCHING_STRATEGY.md](file:///home/henshawmike/Projects/Project_Caspian/docs/development/BRANCHING_STRATEGY.md):

* **`main`**: Stable integration branch preserving validated implementations, baseline logs, and completed phase reports.
* **`research/architecture`**: Neural architecture and inductive-bias investigations.
* **`research/learning-mechanisms`**: Optimization, representation learning, and continual/online adaptation studies.
* **`research/ontology`**: Autonomous property discovery (Phase 13), interaction discovery (Phase 14), and causal intervention studies (Phase 15).
* **`feature/*`** (`feature/memory`, `feature/planning`, `feature/multiple-worlds`): Created when active engineering work begins on specific capabilities.

Every confirmatory experiment requires a **two-stage commit discipline**:
1. **Pre-Registration Commit:** Commit `EXPERIMENT_REGISTRY.md` and experiment code *before* running confirmatory seeds; record `git rev-parse HEAD`.
2. **Confirmatory Execution Commit:** Run locked confirmatory seeds (strictly disjoint from exploratory smoke seeds `9001–9003`), preserve raw outputs via `safe_write_json`, generate the `.docx` report, and record the execution commit hash in `OPERATIONAL_RECORD.md`.

---

# 14. Project Status

* **Current Stage:** Phase 15 Complete; Phase 16 (*Cross-World Transfer and Structural Generalisation*) next on the roadmap.
* **Implemented Phases:** Phases 0–15 (`CSP-P0-E001` through `CSP-P15-E001`).
* **Test Suite Status:** `152 passed` across 23 test modules.
* **Semantic Firewall Status:** Active and enforced on all observations (`0` leakages).
* **Next Immediate Research Question (Phase 16):**
  > **Can Caspian separate reusable structural knowledge from world-specific parameters to transfer zero-shot or few-shot across multiple unfamiliar environments?**
