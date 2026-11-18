"""Generates and updates the living First-Principles Study & Defense Curriculum
for Project Caspian (`docs/Caspian_First_Principles_Study_Curriculum.docx` and
`docs/curriculum/CASPIAN_STUDY_CURRICULUM.md`).

This script is designed to be extended after every phase (Phases 0-20) so the
researcher has a complete, first-principles textbook and viva-defense guide
linked directly to the codebase and empirical ledger.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.report_generator import (
    docx,
    Inches,
    Pt,
    RGBColor,
    WD_TABLE_ALIGNMENT,
    _set_cell_background,
    _set_cell_margins,
)


CURRICULUM_MODULES: List[Dict[str, Any]] = [
    {
        "module_id": "Module 0",
        "title": "Mathematical, Algorithmic & Experimental Foundations (First Principles)",
        "code_files": [
            "environment/world.py",
            "environment/observations.py",
            "models/layers.py",
            "models/activations.py",
            "models/normalization.py",
            "models/optimizers.py",
            "experiments/investigation.py",
        ],
        "first_principles": (
            "1. Agent-Environment Dynamical Loop: Let E_t be the hidden ground-truth simulator state at discrete step t. "
            "The agent never sees E_t directly; it receives a projection O_t = G(E_t) stripped of all human semantic labels. "
            "The agent updates its internal representation S_t = F_theta(S_{t-1}, O_t, A_{t-1}), selects action A_t = pi(S_t), "
            "and observes the consequence O_{t+1} = G(T(E_t, A_t)).\n"
            "2. Analytical Backpropagation in Pure NumPy: Every LinearLayer computes Z = X W + b (where X is B x D_in, W is D_in x D_out). "
            "Given upstream gradient dL/dZ (B x D_out), by matrix chain rule: dL/dW = X^T (dL/dZ), dL/db = sum_b (dL/dZ)_b, "
            "and downstream gradient dL/dX = (dL/dZ) W^T. Every model in models/ is verified against central finite differences: "
            "[L(theta + eps) - L(theta - eps)] / (2 * eps) with tolerance < 1e-5.\n"
            "3. Variance-Preserving Initialization & Feature Normalization: For ReLU networks, He initialization sets Var(W_ij) = 2 / D_in "
            "so activation variance neither explodes nor vanishes across depth. Because raw observations combine coordinates of different scales, "
            "FeatureNormalizer applies z-score standardization (X - mu_train) / sigma_train fit strictly on training data.\n"
            "4. Statistical Inference & Pre-Registration: Primary error is normalized RMSE (nRMSE = sqrt(mean((y_hat - y)^2)) / value_range). "
            "Paired hypothesis test T(a vs b, margin m) requires: (i) mean paired improvement d = metric_b - metric_a >= m, "
            "(ii) 95% bootstrap confidence interval (10,000 resamples) lower bound > 0, and (iii) d >= m in >= 80% of seeds (8/10)."
        ),
        "empirical_link": (
            "All 20 test modules (129 unit, gradient, and forensic tests) enforce these foundations continuously. "
            "Every confirmatory experiment uses a two-stage commit protocol (Stage A pre-registration before data collection; "
            "Stage B confirmatory execution on disjoint seeds; Stage C forensic 'Fake vs. Legit' & first-principles investigation)."
        ),
        "defense_qa": [
            (
                "Why build the entire neural stack in pure NumPy instead of PyTorch or JAX?",
                "Using pure NumPy eliminates black-box autograd magic and forces every forward pass, backward Jacobian, "
                "normalizer, and regularizer (such as BPTT in Phase 2, permutation-invariant max-pooling in Phase 3, and "
                "Riemannian Stiefel kurtosis unmixing in Phase 13) to be derived analytically from first principles and "
                "verified against numerical finite differences."
            ),
            (
                "How do you guarantee the agent is not cheating via hidden semantic labels or data leakage?",
                "Three independent barriers enforce this: (1) Observation.assert_semantic_purity() recursively scans every observation "
                "against FORBIDDEN_SEMANTIC_TOKENS at runtime; (2) train_combos and heldout_combos are mathematically disjoint and "
                "audited for leak_rows == 0; and (3) experiments/investigation.py parses the AST of all files in models/ to prove "
                "zero references to simulator decoding or ground-truth methods."
            ),
        ],
    },
    {
        "module_id": "Phase 0",
        "title": "Research & Safety Foundation — Determinism, Semantic Firewall & Baselines",
        "code_files": [
            "environment/world.py",
            "environment/entities.py",
            "environment/dynamics.py",
            "environment/observations.py",
            "agent/random_agent.py",
            "agent/oracle_agent.py",
            "experiments/deterministic_baseline.py",
        ],
        "first_principles": (
            "Before testing learning, a scientific instrument must have zero measurement noise unless explicitly injected. "
            "GridWorld transitions E_{t+1} = T(E_t, A_t) are pure functions of state and seeded NumPy Generator instances. "
            "Trajectory reproducibility is proven by hashing full state-action-observation rollouts with SHA-256. "
            "Two bounding policies anchor the performance spectrum: RandomAgent (zero-knowledge lower bound, uniform over 6 actions) "
            "and OracleAgent (privileged ground-truth upper bound using Manhattan pathfinding to beneficial entities)."
        ),
        "empirical_link": (
            "Report CSP-P0-E001 verified 100% bitwise trajectory reproducibility (identical SHA-256 digest ce150a82... across repeated runs), "
            "0 SemanticFirewallViolation exceptions, and established the baseline gap between RandomAgent and OracleAgent."
        ),
        "defense_qa": [
            (
                "Why is an OracleAgent allowed to see hidden_state_delta if Caspian forbids semantic leakage?",
                "OracleAgent is strictly an external evaluation upper-bound reference in agent/oracle_agent.py, never part of Caspian's "
                "learning architecture. It establishes the maximum achievable return given the geometry of the world."
            ),
        ],
    },
    {
        "module_id": "Phase 1",
        "title": "Predictive Interaction — Learning Consequence Rules from Unlabeled Experience",
        "code_files": [
            "models/mlp.py",
            "learning/dataset.py",
            "learning/trainer.py",
            "agent/predictive_agent.py",
            "evaluation/baselines.py",
            "experiments/phase1_experiments.py",
        ],
        "first_principles": (
            "Can an agent learn to predict internal state changes Delta_E = E_{t+1} - E_t from neutral state-action vectors x_t = [O_t; one_hot(A_t)]? "
            "PredictiveMLP maps x_t in R^37 through two ReLU hidden layers (32, 16) to a scalar prediction y_hat = Delta_E_hat: "
            "h_1 = ReLU(x_t W_1 + b_1), S_t = ReLU(h_1 W_2 + b_2), y_hat = S_t W_3 + b_3. "
            "Trained offline via mini-batch Adam on MSE loss L = (1/N) sum (y_hat_i - Delta_E_i)^2, and compared against "
            "ZeroBaseline, MeanBaseline, PersistenceBaseline, and ReactiveLookupBaseline."
        ),
        "empirical_link": (
            "Report CSP-P1-E001 (8 seeds): PredictiveMLP achieved held-out R^2 = 0.6026 at 500 steps and R^2 = 0.8799 at 1,000 steps, "
            "outperforming all non-neural baselines and enabling epsilon-greedy action selection that improved survival/energy return."
        ),
        "defense_qa": [
            (
                "Why is predicting Delta_E better than predicting absolute E_{t+1}?",
                "Absolute internal energy E_t is non-stationary (it drifts as a random walk over time), whereas the physical interaction rule "
                "Delta_E = E_{t+1} - E_t is a stationary function of local entity properties and action. Predicting the residual Delta_E "
                "removes the trivial identity shortcut E_{t+1} approx E_t."
            ),
        ],
    },
    {
        "module_id": "Phase 2",
        "title": "Memory & Temporal Persistence — Bridging Partial Observability via Elman BPTT",
        "code_files": [
            "environment/entities.py",
            "models/rnn.py",
            "learning/sequence_dataset.py",
            "learning/recurrent_trainer.py",
            "tests/test_rnn_gradient_check.py",
        ],
        "first_principles": (
            "When entities flicker (visible only when t mod flicker_period < on_steps) or leave a finite radius of vision, "
            "the observation O_t becomes a Partially Observable Markov Decision Process (POMDP): P(Delta_E | O_t, A_t) != P(Delta_E | O_{1:t}, A_{1:t}). "
            "A stateless MLP F(O_t, A_t) must fail during occluded steps because O_t contains no entity features. "
            "RecurrentPredictiveModel maintains hidden state h_t = tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h), y_hat_t = h_t W_{hy} + b_y. "
            "Analytical Backpropagation Through Time (BPTT) unrolls gradients backward from t = T down to 1: "
            "delta_t = ((dL/dy_hat_t) W_{hy}^T + delta_{t+1} W_{hh}^T) * (1 - h_t^2)."
        ),
        "empirical_link": (
            "Report CSP-P2-E001 and test_rnn_gradient_check.py verified exact analytical BPTT gradients (< 1e-6 finite-difference error) "
            "and demonstrated that recurrent state h_t retains occluded entity information across dark steps where stateless MLPs collapse."
        ),
        "defense_qa": [
            (
                "How do you prevent exploding or vanishing gradients during pure-NumPy BPTT?",
                "Episodes are segmented into bounded sequences in EpisodeSequenceDataset, weights are initialized with scaled Xavier/orthogonal "
                "bounds, and gradient norms are clipped analytically before optimizer steps."
            ),
        ],
    },
    {
        "module_id": "Phase 3",
        "title": "Spatial & Set Generalisation — Nearest-First Canonical Ordering & DeepSets",
        "code_files": [
            "environment/observations.py",
            "models/deepsets.py",
            "learning/deepsets_trainer.py",
            "tests/test_observation_entity_ordering.py",
        ],
        "first_principles": (
            "If an environment contains a variable number of entities {e_1, ..., e_m}, an arbitrary slot ordering makes the input representation "
            "sensitive to entity index permutations. By Zaheer et al. (2017), a function f({e_1, ..., e_m}) is permutation-invariant iff it can "
            "be decomposed as f(E) = rho(pool_{i=1..m} phi(e_i)). DeepSetsPredictiveModel applies a shared entity encoder phi to each relative "
            "entity vector, aggregates via masked channel-wise max-pooling over visible entities, concatenates agent self-state and action, "
            "and passes the result through prediction network rho."
        ),
        "empirical_link": (
            "Report CSP-P3-E001: Relative coordinate encoding + nearest-first canonical ordering + DeepSetsPredictiveModel enabled transfer "
            "across unseen spatial coordinates, obstacle layouts, and variable entity counts, while documenting multi-entity seed variance."
        ),
        "defense_qa": [
            (
                "How does analytical backpropagation work through a max-pooling operation over a variable-sized entity set?",
                "During the forward pass over valid (non-padded) entities, DeepSetsPredictiveModel records the argmax entity index m*(b, c) "
                "for each batch item b and embedding channel c. During backward(), upstream gradient dL/d(pool)_{b, c} is routed exclusively "
                "to the winning entity slot m*(b, c) (subgradient of max), with zero gradient sent to masked/non-maximal slots."
            ),
        ],
    },
    {
        "module_id": "Phase 4",
        "title": "World Modeling & Counterfactual Lookahead Planning (MPC)",
        "code_files": [
            "evaluation/world_model_rollout.py",
            "tests/test_world_model_rollout.py",
            "experiments/phase4_experiments.py",
        ],
        "first_principles": (
            "A scalar reward/energy predictor only evaluates immediate 1-step consequences. A true world model predicts the full state transition "
            "vector Delta_s_hat = [Delta_x_hat, Delta_y_hat, Delta_E_hat] = M_theta(O_t, A_t). Given a horizon H and candidate action sequence "
            "a_{0:H-1}, open-loop rollout recursively updates virtual agent coordinates (x_hat_{k+1} = x_hat_k + Delta_x_hat_k), recomputes "
            "relative entity distances d_hat_{k+1}, and accumulates predicted discounted state changes sum_{k=0}^{H-1} gamma^k Delta_E_hat_k. "
            "Model Predictive Control (MPC) selects a_0* = argmax_{a_0} max_{a_{1:H-1}} V_hat(O_t, a_{0:H-1}) and replans at t+1."
        ),
        "empirical_link": (
            "Report CSP-P4-E001 demonstrated accurate 3D transition prediction [Delta_x, Delta_y, Delta_E] and showed short-horizon MPC "
            "lookahead outperforming myopic 1-step reactive selection when beneficial entities require multi-step navigation."
        ),
        "defense_qa": [
            (
                "Why does open-loop world-model rollout degrade at long horizons H, and why use receding-horizon MPC?",
                "Single-step prediction errors epsilon_k compound auto-regressively over H steps (roughly O(H * epsilon) or exponential in "
                "Lipschitz constant L_M^H). Receding-horizon MPC executes only the first action a_0* and re-anchors on real sensory observation "
                "O_{t+1} at the next step, preventing open-loop drift from accumulating unchecked."
            ),
        ],
    },
    {
        "module_id": "Phase 5",
        "title": "Emergent Functional Categories & The Index-Encoding Representation Limit",
        "code_files": [
            "experiments/phase5_experiments.py",
            "tests/test_entity_properties.py",
        ],
        "first_principles": (
            "When do internal hidden states S_t form genuine functional categories? Phase 5 tested five quantitative criteria: "
            "(1) functional cluster separation (silhouette / within-vs-between cosine similarity of S_t for positive vs negative Delta_E entities), "
            "(2) stability across seeds, (3) compression, (4) causal reorganization when an entity type's Delta_E sign is flipped, and "
            "(5) zero-shot generalisation to an unseen entity type. "
            "First-Principles Limit Derived: If entity types are represented as discrete index codes or one-hot vectors e_k, an unseen type k_new "
            "activates an untrained input column W_{1, k_new,:} (which remains at random initialization). Therefore, discrete type IDs CANNOT "
            "generalize zero-shot; zero-shot category formation requires continuous physical property channels p in R^K shared across types."
        ),
        "empirical_link": (
            "Report CSP-P5-E001: Experiments E002-E004 proved that index/one-hot encodings clustered trained types and reorganized under "
            "sign flips, but failed zero-shot transfer (refuted as predicted by theory). Experiment E007 introduced continuous property channels "
            "and achieved zero-shot functional clustering on held-out entity types."
        ),
        "defense_qa": [
            (
                "Why is the failure of E002-E004 on unseen entity type indices considered a central scientific success of Phase 5?",
                "Because it falsified naive category claims and proved mathematically and empirically that zero-shot ontology formation is "
                "impossible on nominal type IDs (since unseen columns receive zero gradient during training), directly motivating the continuous "
                "property representations investigated in Phases 6-13."
            ),
        ],
    },
    {
        "module_id": "Phase 6",
        "title": "Non-Stationary Environments, Surprise Detection & Online Adaptation",
        "code_files": [
            "learning/online_trainer.py",
            "tests/test_online_trainer.py",
            "experiments/phase6_experiments.py",
        ],
        "first_principles": (
            "When environmental rules change mid-stream at step t_flip (e.g., Delta_E -> -Delta_E), a frozen model suffers catastrophic prediction "
            "error. OnlineAdaptationTrainer monitors an exponentially weighted moving average (EWMA) of prediction residuals e_t = |y_hat_t - y_t|. "
            "When e_t exceeds baseline variance by a threshold tau_surprise, the trainer triggers online gradient updates over a sliding replay "
            "buffer of recent transitions, adapting weights theta_t without full offline retraining from scratch."
        ),
        "empirical_link": (
            "Report CSP-P6-E001: Surprise spikes reliably detected mid-stream rule inversions, and OnlineAdaptationTrainer recovered prediction "
            "accuracy within tens of post-flip interactions while frozen pre-change controls remained permanently inverted."
        ),
        "defense_qa": [
            (
                "Why compare online adaptation against both a frozen pre-change model and a reset-from-scratch model?",
                "The frozen control proves the environment shift genuinely invalidated old weights, while the reset-from-scratch control measures "
                "whether retaining shared representation structure accelerates recovery or suffers from negative transfer."
            ),
        ],
    },
    {
        "module_id": "Phase 7",
        "title": "Spurious Correlation, Spatial Confounding & Active Intervention",
        "code_files": [
            "experiments/phase7_experiments.py",
            "tests/test_phase7_helpers.py",
        ],
        "first_principles": (
            "In observational data, if entity type A always spawns in the northern quadrant (y > 3) and type B in the southern quadrant (y < 2), "
            "spatial position y and entity property p are collinear: P_obs(p, y) factors such that E[Delta_E | p, y] can be fit equally well by "
            "w_y y or w_p p. Passive observational risk R_obs(theta) cannot distinguish the true causal parent p from the spatial confounder y. "
            "Under an intervention do(position = swap) or active exploration that breaks the P(p | y) coupling, a spatially confounded predictor "
            "collapses whereas a property-grounded predictor remains invariant."
        ),
        "empirical_link": (
            "Report CSP-P7-E001: Under spatial-swap interventions (E002), actively de-confounded training maintained R^2 approx 0.99 vs passive "
            "confounded training R^2 approx 0.48. Later in Phase 10, a deeper audit revealed that balanced combination exposure coverage accounted "
            "for much of the active-vs-passive difference—a critical methodological lesson."
        ),
        "defense_qa": [
            (
                "What did Phase 10 later reveal about Phase 7's active vs. passive comparison?",
                "Phase 10 showed that when passive data collection is balanced to have the exact same combination exposure distribution as "
                "active collection, the gap between active and passive policies largely disappears—proving that state-space coverage balance, "
                "not policy agency per se, was the primary driver."
            ),
        ],
    },
    {
        "module_id": "Phase 8 & Phase 9",
        "title": "Cross-World Distribution Shifts (Phase 8) & Scaling Laws of Capacity vs. Data (Phase 9)",
        "code_files": [
            "experiments/phase8_experiments.py",
            "experiments/phase9_experiments.py",
            "tests/test_phase8.py",
            "tests/test_phase9.py",
        ],
        "first_principles": (
            "Phase 8 systematically varied 6 structural axes of the world (geometry, appearance, dynamics, rules, resources, actions) comparing "
            "zero-shot transfer, fine-tuning, training from scratch, and shuffled-weight pretraining controls.\n"
            "Phase 9 investigated the neural scaling law across model capacity P in [157, 13185] parameters, data budget N in [500, 96000] steps, "
            "and world complexity C1 (linear additive), C2 (nonlinear multi-entity), and C3 (high-order compositional/temporal). "
            "By statistical learning theory, total risk decomposes into approximation error (governed by function class capacity P and inductive bias) "
            "plus estimation error O(sqrt(P / N))."
        ),
        "empirical_link": (
            "Reports CSP-P8-E001 & CSP-P9-E001: Data budget N dominated generalisation-gap shrinkage from 500 to 96,000 steps; capacity thresholds "
            "scaled from 313 params (C1) to 1,761 params (C2), whereas C3 remained below R^2 = 0.90 even at 13,185 params and 96,000 steps—showing "
            "that brute-force MLP scaling cannot solve compositional structure."
        ),
        "defense_qa": [
            (
                "Why did scaling the MLP to 13,185 parameters and 96,000 steps still fail to master complexity level C3?",
                "Because an unconstrained MLP lacks structural inductive bias: as input dimensionality and interaction order grow, the number of "
                "conjunctive hypercube corners grows exponentially, so local ReLU interpolation fails on held-out combinations regardless of width."
            ),
        ],
    },
    {
        "module_id": "Phase 10",
        "title": "The Compositional Audit — Why Probe Decodability != Compositional Understanding",
        "code_files": [
            "experiments/phase10_experiments.py",
            "docs/phase10/REPRODUCTION_REPORT.md",
            "docs/phase10/KNOWN_ISSUES.md",
        ],
        "first_principles": (
            "Phase 10 audited whether standard MLPs autonomously discover compositional ontology. "
            "Fundamental Theorem Demonstrated: A linear ridge probe W_probe fit on hidden layer S_t (or even directly on raw input X!) can decode "
            "ground-truth properties p_1, p_2, p_3 with R^2 in [0.96, 1.00], because any invertible linear/smooth map preserves linear subspace "
            "information. Yet the network's own prediction head W_out fails to extrapolate when held-out combinations (e.g., A=1, B=1) are tested! "
            "Why? Because on training combinations where (A=1, B=1) is absent, infinitely many functions f(A, B, C) = 6A - 4B + 2C + gamma * A * B "
            "(for ANY scalar gamma in R!) have ZERO training error! An unconstrained ReLU MLP learns non-zero spurious interaction gamma != 0."
        ),
        "empirical_link": (
            "Report CSP-P10-E001 (Refuted/Diagnostic): Latent probes achieved R^2 = 0.96-1.00, but held-out compositional prediction R^2 dropped "
            "to 0.65-0.83 vs oracle 1.00. Exposed and documented 15 methodological issues in docs/phase10/KNOWN_ISSUES.md."
        ),
        "defense_qa": [
            (
                "If a linear probe can decode property A and property B from hidden layer S_t with R^2 = 0.99, why doesn't that prove the model understands A and B?",
                "A linear probe can also decode A and B from the raw input layer with R^2 = 1.00! Probe decodability only measures whether "
                "information is preserved (not destroyed) by the encoder; it says nothing about whether the downstream readout head composes "
                "A and B additively rather than memorizing a conjunctive lookup table."
            ),
        ],
    },
    {
        "module_id": "Phase 11",
        "title": "Architectural Inductive Bias — How Function-Class Structure Governs Extrapolation",
        "code_files": [
            "models/compositional.py",
            "experiments/phase11_experiments.py",
            "experiments/analyze_phase11.py",
            "docs/phase11/EXPERIMENT_REGISTRY.md",
            "tests/test_phase11.py",
        ],
        "first_principles": (
            "To resolve the Phase 10 identifiability problem on held-out combinations, Phase 11 compared five architectures on pre-separated "
            "property channels p = (p_1, ..., p_K): (1) mlp_small (673 params), (2) mlp_large (2,369 params), (3) factorised (per-channel encoders "
            "concatenated into a joint MLP head, 921 params), (4) additive_mlp (per-channel embeddings summed into a nonlinear MLP head, 665 params), "
            "and (5) strictly additive (per-channel encoders summed into a linear readout: f(p) = b + sum_k f_k(p_k), 513 params).\n"
            "First-Principles Proof of Identifiability: If the hypothesis class is restricted to H_add = {b + sum_k f_k(p_k)}, then the mixed "
            "partial derivative d^2 f / (dp_i dp_j) = 0 identically for all i != j. Provided each individual level of every property p_k appears "
            "at least once in training alongside a reference baseline (e.g., 000, 100, 010, 001), the univariate effects f_k(p_k) are 100% uniquely "
            "identified on ALL 2^K combinations, including unseen corners (110, 111)!"
        ),
        "empirical_link": (
            "Report CSP-P11-E001 (10 seeds, W1 & W2): Strictly additive model achieved held-out nRMSE = 0.001 (W1) and 0.000 (W2) vs mlp_small "
            "0.183 and mlp_large 0.115 (H1 & H5 MET in 10/10 seeds). As predicted by theory (H4 negative control), when an unidentifiable +5AB "
            "interaction was injected on the held-out corner only, the additive model erred by 0.417 nRMSE."
        ),
        "defense_qa": [
            (
                "Why did 'factorised' (separate per-property encoders concatenated into a shared MLP head) not match the strictly 'additive' model?",
                "Because although 'factorised' separates properties at the first encoding layer, concatenating the embeddings [z_1; z_2; z_3] into "
                "a hidden ReLU readout head still allows the readout head to form arbitrary cross-property conjunctions ReLU(W_1 z_1 + W_2 z_2 + b). "
                "Only summing the per-property branch outputs restricts the function class to zero cross-derivatives."
            ),
        ],
    },
    {
        "module_id": "Phase 12",
        "title": "Identifiable Non-Additive Interactions (W3), Additive-Residual Architecture & Optimiser Sensitivity",
        "code_files": [
            "models/compositional.py",
            "experiments/phase12_experiments.py",
            "experiments/analyze_phase12.py",
            "docs/phase12/EXPERIMENT_REGISTRY.md",
            "tests/test_phase12.py",
        ],
        "first_principles": (
            "What happens when the world genuinely contains a non-additive interaction Y = 6A - 4B + 2C + 5AB that IS visible in training "
            "(W3: train combos 000, 100, 010, 110, 001, 111; held-out 101, 011)? "
            "1. The strictly additive model f_A(A) + f_B(B) + f_C(C) cannot fit the training set because no additive function satisfies "
            "f(1,1,0) - f(1,0,0) - f(0,1,0) + f(0,0,0) = 5 != 0 (least-squares projection onto H_add distributes the +5AB residual as bias "
            "across main effects, giving in-dist nRMSE approx 0.08 and held-out nRMSE approx 0.19).\n"
            "2. AdditivePlusResidual ('additive_residual', 1,434 params) decomposes prediction into an additive main-effect trunk plus a "
            "concatenated factorised residual branch: y_hat = f_add(p) + f_res(p). Under Adam with weight decay (1e-4), main effects are "
            "absorbed by the low-complexity additive trunk while f_res only needs to fit the sparse interaction residual +5AB."
        ),
        "empirical_link": (
            "Report CSP-P12-E001 (disjoint seeds 11-20, 720 sensitivity trainings): On W3, additive_residual achieved held-out nRMSE = 0.037 "
            "vs additive 0.192 (P12-H2a MET, 10/10 seeds) and mlp_small 0.179, while retaining the additive advantage on W1 (P12-H2b MET). "
            "Across all 6 (lr in {0.003, 0.01, 0.03} x epochs in {200, 400}) settings on W1 and W2, architecture rankings remained invariant (P12-H4 & H5 MET)."
        ),
        "defense_qa": [
            (
                "Why did P12-H1 ('mlp_small beats additive on W3 held-out') fail even though additive could not fit W3?",
                "Because although the strictly additive model suffered a deterministic approximation bias (held-out nRMSE = 0.192), the unconstrained "
                "mlp_small suffered an equally large variance/extrapolation error (0.179 +/- 0.080) on held-out combinations 101 and 011. "
                "Both failed for opposite bias-variance reasons, whereas additive_residual (0.037) solved both."
            ),
        ],
    },
    {
        "module_id": "Phase 13",
        "title": "Autonomous Property Discovery & Representation Learning from Raw Entangled Sensors",
        "code_files": [
            "models/discovery.py",
            "experiments/phase13_experiments.py",
            "experiments/analyze_phase13.py",
            "experiments/investigation.py",
            "docs/phase13/EXPERIMENT_REGISTRY.md",
            "docs/phase13/INVESTIGATION.md",
            "tests/test_phase13.py",
            "tests/test_investigation.py",
        ],
        "first_principles": (
            "Phase 13 removed pre-separated property channels. The agent receives a 12D raw sensory vector x = z_scaled M in R^12, where "
            "z_scaled = [p_1, p_2, p_3, 2 u_1, 2 u_2, 2 u_3] mixes 3 task-relevant properties (amplitude 1.0) and 3 nuisance factors "
            "(amplitude 2.0 -> 4x variance, so 80% of raw sensor variance is task-irrelevant nuisance!) via a dense 6x12 row-orthonormal "
            "matrix M (M M^T = I_6).\n"
            "1. Why Variance-Driven PCA ('pca_statistical') Fails: Top-3 PCA extracts the 3 eigenvectors of maximum sensor variance, which are "
            "100% pure nuisance factors (2 u_1, 2 u_2, 2 u_3), discarding the task-relevant properties (slot_match_r2 = 0.006)!\n"
            "2. Why Raw-Sensor Additive ('raw_naive_additive') Fails on Nonlinear Worlds: In W2_raw (Y = 3u - 4v + w^2), each raw sensor x_j "
            "mixes w with u and nuisance factors. Applying nonlinear univariate MLPs f_j(x_j) directly to raw sensors creates spurious cross-terms "
            "(alpha w + beta u_1)^2 = alpha^2 w^2 + 2 alpha beta w u_1 + beta^2 u_1^2, inflating held-out error and distractor sensitivity.\n"
            "3. First-Principles Solution ('OrthogonalUnmixingLayer' + Correlation Non-Duplication Guard): Let V_sub in R^{12x6} be the top-6 "
            "right singular vectors of uncentered X_train. Because M M^T = I_6 and V_sub spans rowspan(M), Q* = M V_sub is an exact 6x6 orthogonal "
            "matrix in O(6) regardless of missing training corners! Since physical properties and nuisances are bounded sub-Gaussian variables "
            "(kurtosis kappa = E[z^4]/(E[z^2])^2 in {1.0, 1.5} < 3.0), by Jensen's inequality and Comon's ICA theorem, minimizing scale-invariant "
            "fourth-moment kurtosis L_kurt(W) = (1/4K) sum_m E[Z_m^4]/(E[Z_m^2])^2 over W in O(6)—augmented with the non-duplication correlation "
            "guard sum_{m!=m'} max(0, |Corr(Z_m, Z_{m'})| - 0.52)^2 derived in Stage C Investigation—uniquely recovers the 6 pure underlying "
            "factors without labels! The downstream additive/residual head then zeroes out the 3 nuisance slots via supervised consequence gradients."
        ),
        "empirical_link": (
            "Report CSP-P13-E001 & INVESTIGATION.md (disjoint seeds 21-30, 540 trainings): Forensic audit verified LEGIT status across all 5 checks. "
            "latent_additive matched the privileged clean-channel oracle on W1_raw (0.00012 vs raw_mlp_small 0.1081 and pca_statistical 0.2568, 10/10 seeds) "
            "and W2_raw (0.0036 vs raw_mlp_small 0.0412 and raw_naive_additive 0.0478), while latent_residual recovered W3_raw interactions (0.0409 vs "
            "latent_additive 0.1581). Stage C First-Principles Investigation diagnosed and resolved the 2-seed variance-asymmetric nuisance duplication "
            "in W2_raw and 45-degree polarization saddle in W3_raw, bringing P13-H4 to 10/10 seeds (mean R^2 = 0.9950), P13-H5 CKA to 0.9849, and P13-H7 to 9/10."
        ),
        "defense_qa": [
            (
                "Why did you use uncentered SVD for V_sub instead of centering and whitening X_train by Cov(X_train)^{-1/2}?",
                "Because on a compositional split where a corner like (A=1, B=1) is held out from training, A and B have empirical correlation "
                "Corr_train(A, B) = -0.50 and non-zero sample mean (-1/3). Centering and whitening by Cov_train(X)^{-1/2} shears the Euclidean "
                "angle between physical sensor axes m_A and m_B so they are no longer orthogonal! Keeping the natural sensor origin and projecting "
                "via orthonormal V_sub preserves Q* = M V_sub in O(6) exactly."
            ),
            (
                "What caused the 2-seed anomaly in W2_raw during the initial confirmatory run, and how did first-principles math solve it?",
                "In W2_raw, the 3-level property u has variance 0.57 and kurtosis 1.65, whereas binary nuisance 2u_j has 7x larger variance (4.0) "
                "and lower kurtosis (1.00). In unwhitened Euclidean coordinates, a 36.8-degree rotation Z_1 = 0.8u + 0.6(2u_j), Z_2 = -0.6u + 0.8(2u_j) "
                "allowed the loud binary nuisance 2u_j to dominate BOTH slots simultaneously (|Corr(Z_1, Z_2)| = 0.74), giving a slightly lower "
                "unweighted kurtosis sum (0.2957 < 0.2967) than the true unmixing W*. Adding the correlation guard penalizing |Corr(Z_i, Z_j)| > 0.52 "
                "eliminated factor duplication and raised W2_raw single-slot recovery to 10/10 seeds (mean R^2 = 0.9950)."
            ),
        ],
    },
    {
        "module_id": "Phase 14",
        "title": "Interaction Discovery & Structured Relationships — Functional ANOVA, Mobius Contrasts & Sparse Recovery",
        "code_files": [
            "models/interactions.py",
            "experiments/phase14_experiments.py",
            "experiments/analyze_phase14.py",
            "experiments/investigation.py",
            "docs/phase14/EXPERIMENT_REGISTRY.md",
            "docs/phase14/INVESTIGATION.md",
            "tests/test_phase14.py",
        ],
        "first_principles": (
            "As the number of environmental properties grows from K = 3 (Phase 12) to K = 5 (Phase 14: 2^5 = 32 combinations, C(5,2) = 10 pairs, "
            "C(5,3) = 10 triplets), why does Phase 12's dense 'additive_residual' degrade while 'sparse_interaction' succeeds?\n"
            "1. Why Dense Residual Networks Hallucinate False Interactions at K = 5: In 'additive_residual', the residual branch concatenates "
            "all K = 5 property embeddings into a 40D vector passed into a dense 16-unit ReLU layer. When an interaction +6 p_0 p_1 is observed "
            "in training only when p_3 = 0 (because (p_0=1, p_1=1, p_3=1) is held out), a dense 5-variable ReLU network cannot distinguish the "
            "2-way interaction +6 p_0 p_1 from the 3-way conjunction +6 p_0 p_1 (1 - p_3), inventing 3 to 5 false interactions on held-out contexts!\n"
            "2. First-Principles Functional-ANOVA / Mobius Decomposition ('SparseInteractionDiscoveryModel'): Any function on {0, 1}^K decomposes "
            "uniquely into orthogonal orders of interaction: f(A) = f_0 + sum_i f_i(A_i) + sum_{i<j} w_{ij} A_i A_j + sum_{i<j<k} w_{ijk} A_i A_j A_k. "
            "By the Frisch-Waugh-Lovell Theorem, projecting both the candidate interaction dictionary Psi(A) = [A_i A_j, A_i A_j A_k] in R^{N x 20} "
            "and target Y onto the orthogonal complement of the main-effect space M_main = [1, A] (via QR projection I - Q_m Q_m^T) isolates the "
            "pure non-additive residual without penalizing main effects. Iterative Soft-Thresholding (ISTA) followed by Gauss-Lasso debiasing "
            "recovers the exact sparse interaction support S* with zero shrinkage bias.\n"
            "3. Model-Agnostic Discrete Mixed Partial Derivatives (Mobius Probing): Any black-box model f_hat can be audited for learned interactions "
            "without inspecting internal weights by evaluating its discrete 2nd-order mixed difference Delta_{ij} f_hat = f_hat(1,1) - f_hat(1,0) - "
            "f_hat(0,1) + f_hat(0,0) and 3rd-order mixed difference Delta_{ijk} f_hat across background contexts."
        ),
        "empirical_link": (
            "Report CSP-P14-E001 & docs/phase14/INVESTIGATION.md (disjoint seeds 31-40, 5 worlds x 6 architectures x 2 levels = 600 trainings): "
            "sparse_interaction achieved F1 = 1.000 and 0 false interactions in 10/10 seeds across all 5 worlds (W14_indep, W14_sparse_pair, "
            "W14_multi_pair, W14_higher_order, W14_noisy_pair), while mlp_small/large hallucinated ~10.8 false interactions and additive_residual "
            "hallucinated ~3.9 false interactions (P14-H6 MET, 10/10 seeds). Stage C Investigation proved 100% LEGIT status, explained the "
            "Zero-Error Floor Effect (RER >= 92% in 10/10 seeds on W14_indep and >= 99.7% in 10/10 seeds on W14_multi_pair), and derived the "
            "Midpoint-Split 2-Mode Conditional Mean Estimator to eliminate tail-percentile noise attenuation in W14_noisy_pair (cutting noisy "
            "held-out nRMSE in half from 0.01037 to 0.00537, beating additive_residual in 9/10 seeds)."
        ),
        "defense_qa": [
            (
                "Why did you use Frisch-Waugh-Lovell projection before ISTA instead of running L1 Lasso on [A, Psi(A)] jointly?",
                "Because if main effects A_i are L1-penalized alongside product terms A_i A_j (which are positively correlated with A_i on {0,1}^K), "
                "Lasso shrinkage on A_i leaks main-effect variance into the interaction coefficients w_{ij} (violating the statistical principle "
                "of strong heredity). Projecting out M_main = [1, A] via QR ensures main effects are 100% unpenalized and w_int fits only "
                "variance orthogonal to all additive functions."
            ),
            (
                "In Stage C of Phase 14, why did splitting at the sample median fail on p_3 in W14_noisy_pair whereas splitting at the range midpoint succeeded?",
                "Because 10 of the 16 combinations with p_3 = 1 were held out from the 22 training combinations, leaving 72.7% of training "
                "samples at p_3 = 0 (0.2) and only 27.3% at p_3 = 1 (0.8). When > 50% of samples belong to the lower mode, the sample median "
                "falls inside the lower mode (0.20) rather than between the two modes! Splitting at the 5th/95th percentile midpoint (~0.50) "
                "is invariant to class-ratio imbalance."
            ),
        ],
    },
]

FUTURE_PHASES_ROADMAP = [
    ("Phase 15", "Causal Discovery & Intervention Learning", "research/ontology",
     "Separating observational confounding and hidden common causes from direct causal effects via active do(X) interventions."),
    ("Phase 16", "Cross-World Transfer & Structural Generalisation", "feature/multiple-worlds, research/ontology",
     "Separating reusable structural rules from world-specific sensor calibrations for zero-shot and few-shot transfer."),
    ("Phase 17", "Continual Learning & Persistent Environmental Memory", "feature/memory, research/learning-mechanisms",
     "Sequential multi-world learning with episodic/parametric memory to prevent catastrophic forgetting."),
    ("Phase 18", "Integrated Predictive World Model & Autonomous Planning", "feature/planning, research/architecture",
     "Combining discovered representations, causal structure, and uncertainty-aware multi-step rollouts for closed-loop planning."),
    ("Phase 19", "Real-World Transfer & Application Prototype", "feature/application-prototype",
     "Transferring the validated Caspian world model to a bounded operational or cybersecurity anomaly/decision task."),
    ("Phase 20", "Integrated System Validation & Research Consolidation", "research/ontology, research/architecture, research/learning-mechanisms",
     "End-to-end integration benchmark, full component ablations, literature synthesis, and final technical research paper."),
]


def generate_curriculum_docx_and_md(
    docx_path: str = "docs/Caspian_First_Principles_Study_Curriculum.docx",
    md_path: str = "docs/curriculum/CASPIAN_STUDY_CURRICULUM.md",
) -> str:
    doc = docx.Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8); sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8); sec.right_margin = Inches(0.8)

    PRIMARY = RGBColor(24, 43, 73)
    SECONDARY = RGBColor(70, 80, 95)
    ACCENT = RGBColor(16, 124, 65)

    def h1(t):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
        r = p.add_run(t); r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = PRIMARY

    def h2(t):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(2)
        r = p.add_run(t); r.font.size = Pt(11.5); r.font.bold = True; r.font.color.rgb = SECONDARY

    def body(t, size=10, bold=False, color=None):
        p = doc.add_paragraph(); r = p.add_run(t); r.font.size = Pt(size); r.font.bold = bold
        if color is not None:
            r.font.color.rgb = color

    def table(headers, rows):
        t = doc.add_table(rows=1 + len(rows), cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(headers):
            c = t.rows[0].cells[j]; r = c.paragraphs[0].add_run(h); r.font.bold = True; r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(255, 255, 255); _set_cell_background(c, "182B49"); _set_cell_margins(c)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.rows[i + 1].cells[j]; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(9)
                _set_cell_background(c, "FAFAFA" if i % 2 == 0 else "F0F4F8"); _set_cell_margins(c)
        doc.add_paragraph()

    # Title Header
    p = doc.add_paragraph(); r = p.add_run("PROJECT CASPIAN — LIVING STUDY & DEFENSE CURRICULUM"); r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = PRIMARY
    p = doc.add_paragraph(); r = p.add_run("First-Principles Mathematical Theory, Codebase Architecture, Empirical Ledger & Viva Defense Guide (Phases 0–13, Extensible through Phase 20)"); r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = SECONDARY

    body("Purpose of this Living Document: This curriculum is designed so the researcher can master and defend every mathematical equation, algorithm, codebase module, positive finding, negative/refuted result, and forensic audit in Project Caspian from first principles. It is automatically updated after every phase completion.")

    h1("Overview Matrix: Completed Study Modules (Module 0 & Phases 0–13)")
    summary_rows = [
        [m["module_id"], m["title"], ", ".join(m["code_files"][:3])]
        for m in CURRICULUM_MODULES
    ]
    table(["Module / Phase", "Core Topic & First-Principles Focus", "Primary Codebase Files"], summary_rows)

    md_lines = [
        "# Project Caspian — Living First-Principles Study & Defense Curriculum",
        "",
        "> **Living Document:** Generated by `experiments/generate_curriculum.py` (`docs/Caspian_First_Principles_Study_Curriculum.docx`). Updated at the end of every phase so you can study and defend every equation, file, experiment, and forensic investigation from first principles.",
        "",
    ]

    for m in CURRICULUM_MODULES:
        header_text = f"{m['module_id']}: {m['title']}"
        h1(header_text)
        md_lines.append(f"## {header_text}")
        md_lines.append("")

        h2("1. Linked Codebase Modules & Artifacts")
        files_str = ", ".join(m["code_files"])
        body(f"Files to read alongside this module: {files_str}", size=9.5, bold=True)
        md_lines.append(f"**Linked Codebase Files:** `{files_str}`")
        md_lines.append("")

        h2("2. First-Principles Mathematical & Algorithmic Derivation")
        for para in m["first_principles"].split("\n"):
            body(para, size=10)
            md_lines.append(para)
            md_lines.append("")

        h2("3. Empirical Outcome & Scientific Record")
        body(m["empirical_link"], size=10)
        md_lines.append(f"**Empirical Evidence:** {m['empirical_link']}")
        md_lines.append("")

        h2("4. Viva / Defense Questions & Model Answers")
        md_lines.append("### Viva / Defense Q&A")
        for q, a in m["defense_qa"]:
            body(f"Q: {q}", size=10, bold=True, color=PRIMARY)
            body(f"A (Defense Answer): {a}", size=9.5, color=ACCENT)
            md_lines.append(f"- **Q: {q}**")
            md_lines.append(f"  - **A:** {a}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    h1("Forward Curriculum Expansion Roadmap (Phases 14–20)")
    body("As each subsequent phase is implemented, tested, and audited via experiments/investigation.py, a new module will be appended to this document:")
    table(["Phase", "Title", "Target Branches", "Planned First-Principles Topic"], FUTURE_PHASES_ROADMAP)

    md_lines.append("## Forward Curriculum Expansion Roadmap (Phases 14–20)")
    md_lines.append("")
    for ph, title, br, desc in FUTURE_PHASES_ROADMAP:
        md_lines.append(f"- **{ph} — {title}** (`{br}`): {desc}")

    doc.save(docx_path)
    md_file = Path(md_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return docx_path


if __name__ == "__main__":
    out = generate_curriculum_docx_and_md()
    print("Generated curriculum DOCX:", out)
