# PROJECT CASPIAN — SCIENTIFIC EXPERIMENT REPORT: CSP-P0-E001

**Phase:** Phase 0 — Research & Safety Foundation  
**Experiment ID:** `CSP-P0-E001`  
**Document Layer:** Layer B Permanent Record  
**Date:** 2026-09-22  
**Determinism Status:** VERIFIED (100% Bitwise Trajectory Reproducibility)  
**Semantic Firewall Status:** AUDITED & ACTIVE (Zero Leakages Detected)  

---

## 1. Executive Summary & Objective

Project Caspian investigates whether an artificial agent can construct structured internal representations of an environment through perception, action, memory, and prediction without being given human semantic labels.

Before building the learning agent, Phase 0 establishes a completely controlled, mathematically specified, and bitwise deterministic 2D discrete environment. This report permanently records the implementation, acceptance testing, Semantic Firewall verification, and baseline evaluations for the initial environment and agents.

---

## 2. Environment Specification & Configuration

The environment is a discrete 2D grid world bounded by $[0, W-1] \times [0, H-1]$. All parameters are fully externalized in [`configs/default_world.json`](file:///home/henshawmike/Projects/Project_Caspian/configs/default_world.json):

| Parameter | Configured Value | Mathematical / Operational Meaning |
| :--- | :--- | :--- |
| `grid_width` ($W$) | `5` | Horizontal grid lattice size |
| `grid_height` ($H$) | `5` | Vertical grid lattice size |
| `initial_agent_pos` | `(0, 0)` | Starting Cartesian coordinates of agent |
| `initial_internal_state` ($E_0$) | `100.0` | Initial scalar measurable state variable |
| `step_penalty` ($\delta_{\text{step}}$) | `1.0` | Fixed state decrement applied per timestep |
| `min_internal_state` ($E_{\min}$) | `0.0` | Minimum clamped state value |
| `max_internal_state` ($E_{\max}$) | `100.0` | Maximum clamped state value |
| `interaction_radius` | `1` | Manhattan distance threshold for valid interaction |
| `entities` | `1` | Entity X at $(2, 2)$ with hidden state modifier $+10.0$ |
| `seed` | `42` | Isolated pseudo-random number generator seed |

---

## 3. Mathematical Foundations & Transition Dynamics

Let the environment state at timestep $t$ be:
$$S_t = \langle p_t, E_t, t, \mathcal{E}, c_t, i_t, a_{t-1} \rangle$$

The state transition function $T(S_t, A_t) \rightarrow S_{t+1}$ is governed by exact deterministic equations:

1. **Positional Translation:**
   $$p' = p_t + \Delta(A_t)$$
   where $\Delta(\text{UP})=(0,1)$, $\Delta(\text{DOWN})=(0,-1)$, $\Delta(\text{LEFT})=(-1,0)$, $\Delta(\text{RIGHT})=(1,0)$, $\Delta(\text{INTERACT/NOOP})=(0,0)$.

2. **Boundary & Obstacle Collision Check:**
   $$p_{t+1} = \begin{cases} p_t & \text{if } p' \notin [0, W-1] \times [0, H-1] \lor \exists e \in \mathcal{E} \text{ s.t. } \text{pos}(e)=p' \land \text{blocking}(e) \\ p' & \text{otherwise} \end{cases}$$
   $$c_{t+1} = (p_{t+1} == p_t \land \Delta(A_t) \ne (0, 0))$$

3. **Interaction & Internal State Update:**
   $$\Delta E_{\text{interaction}} = \begin{cases} \sum_{e \in \mathcal{E}_{\text{active}}(p_{t+1})} \delta_e & \text{if } A_t = \text{INTERACT} \\ 0.0 & \text{otherwise} \end{cases}$$
   $$E_{t+1} = \max(E_{\min}, \min(E_{\max}, E_t - \delta_{\text{step}} + \Delta E_{\text{interaction}}))$$

4. **Timestep Progression:**
   $$t_{t+1} = t_t + 1$$

---

## 4. Action & Neutral Observation Interfaces

The action space is strictly discrete:
$$\mathcal{A} = \{\text{UP (0)}, \text{DOWN (1)}, \text{LEFT (2)}, \text{RIGHT (3)}, \text{INTERACT (4)}, \text{NOOP (5)}\}$$

The observation interface exposes only neutral physical measurements:
- `agent_position`: `(x, y)`
- `entities`: `[(entity_id, entity_type, rel_x, rel_y, distance)]`
- `collision`: `bool`
- `internal_state`: `float`
- `timestep`: `int`
- `previous_action`: `Optional[int]`

---

## 5. Acceptance Testing Results

All 12 acceptance criteria pass across 29 unit tests:

| Criterion | Target Test Module | Status |
| :--- | :--- | :--- |
| 1. Same seed produces same initial state | [`test_determinism.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_determinism.py) | **PASSED** |
| 2. Same actions + seed produce same trajectory | [`test_determinism.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_determinism.py) | **PASSED** |
| 3. Caspian cannot move outside the world | [`test_dynamics.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_dynamics.py) | **PASSED** |
| 4. Valid actions are accepted | [`test_dynamics.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_dynamics.py) | **PASSED** |
| 5. Invalid actions are rejected cleanly | [`test_dynamics.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_dynamics.py) | **PASSED** |
| 6. Reset returns environment to initial deterministic state | [`test_world.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_world.py) | **PASSED** |
| 7. Collision behavior is deterministic | [`test_determinism.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_determinism.py) | **PASSED** |
| 8. Interaction behavior is deterministic | [`test_determinism.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_determinism.py) | **PASSED** |
| 9. Measurable state variable changes according to hidden rule | [`test_dynamics.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_dynamics.py) | **PASSED** |
| 10. Observation contains only permitted neutral information | [`test_observations.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_observations.py) | **PASSED** |
| 11. Hidden semantic interpretation is absent from observation | [`test_observations.py`](file:///home/henshawmike/Projects/Project_Caspian/tests/test_observations.py) | **PASSED** |
| 12. All tests pass | Unit test runner (29/29) | **PASSED** |

---

## 6. Experimental Results & Baselines

### Determinism Verification (Seed 42, 5 Resets, 15 Steps)
- **SHA-256 Trajectory Hash:** `ce150a820737511f8fcaa727c87701e423caa9e70b5e7be6155e85a057957b0b`
- **Result:** $100\%$ Bitwise identical across all 5 resets.

### Baseline Comparison (5 Seeds $\times$ 50 Steps)

| Metric | Random Agent | Oracle Agent |
| :--- | :--- | :--- |
| **Avg Final Energy** | $64.0$ | $100.0$ |
| **Avg Cumulative $\Delta E$** | $-36.0$ | $0.0$ |
| **Avg Successful Interactions** | $1.4$ | $47.0$ |
| **Avg Collisions** | $6.0$ | $0.0$ |

---

## 7. Semantic Firewall Audit

- **Audit Target:** All observation keys, values, vector representations, and agent classes.
- **Forbidden Vocabulary:** `{"food", "hazard", "danger", "reward", "beneficial", "harmful", "resource", "target", "enemy", "good", "bad", "positive", "negative", "expected_outcome", "goal", "threat"}`.
- **Leakages Found:** 0.
- **Runtime Assertion:** `assert_semantic_purity()` actively validates every observation.

---

## 8. Artifacts Generated

- DOCX Permanent Record: [`docs/CSP-P0-E001.docx`](file:///home/henshawmike/Projects/Project_Caspian/docs/CSP-P0-E001.docx)
- JSON Structured Log: [`logs/CSP-P0-E001.json`](file:///home/henshawmike/Projects/Project_Caspian/logs/CSP-P0-E001.json)
- Trajectory Log: [`logs/trajectory_seed_42.json`](file:///home/henshawmike/Projects/Project_Caspian/logs/trajectory_seed_42.json)
- Baseline Benchmark Log: [`logs/baseline_comparison.json`](file:///home/henshawmike/Projects/Project_Caspian/logs/baseline_comparison.json)
