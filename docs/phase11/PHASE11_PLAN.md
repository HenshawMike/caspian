# Phase 11 Plan: Architectural Inductive Bias and Compositional Generalisation

Status: Stage 11.0 complete; Stage 11.1-11.5 registered in EXPERIMENT_REGISTRY.md before confirmatory runs.

## Stage 11.0 integrity gate
- Repo inspected; Phase 10 baseline: `pytest -q tests` = 97 passed at Phase 10 close, 113 passed after adding Phase 11 tests (16 new).
- Phase 10 E002 results reproduce from logs/phase10 (deterministic seeded code, verified bit-for-bit for Phase 7/9 in Stage 10.0).
- Known issues reviewed: see KNOWN_ISSUES.md for which affect Phase 11.
- Gate result: PASS (tests green, configs load, no overwrite hazards: safe_io enforced for Phase 11 outputs).

## Reused / modified
| Item | Where | Use |
|---|---|---|
| Simulator, `WorldConfig`, `_collect_passive` | environment/world.py, experiments/phase7_experiments.py | unchanged; data collection |
| `PredictiveMLP`, `Adam`, `MSELoss`, `FeatureNormalizer` | models/, | unchanged; architecture A and all building blocks |
| `_encode_property_type`, `_fit_mlp`, `_interaction_rows` | phase5/phase10 | architecture A1 = exact Phase 10 pipeline |
| NEW `models/compositional.py` | | factorised / additive models built by composing PredictiveMLP modules |
| NEW `experiments/phase11_experiments.py` | | worlds W1, W2, split, training loop, metrics, adaptation, stats helpers |
| NEW `tests/test_phase11.py` | | gradient checks, strict additivity, split/leakage, negative control, determinism |

Why a new training loop: `PredictiveTrainer` restores best weights by rebuilding PredictiveMLP layers, which breaks composite models and continued fine-tuning. Same optimiser, loss, batch size for all Phase 11 architectures.

## Stages
- 11.1 architectures (ARCHITECTURE_DESIGN.md); 11.2 controlled split, additive and interaction worlds; 11.3 adaptation; 11.4 second world W2; 11.5 mechanism measures. All run in one registered matrix plus one adaptation run.
- Exploratory (seeds 9001-9003, `logs/phase11/exploratory_*`): pipeline smoke test only; no hyperparameter was tuned from it. Confirmatory: seeds in SEEDS.
