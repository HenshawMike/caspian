# Project Caspian — Antigravity Agent Rules (`AGENTS.md`)

This file defines mandatory workspace-wide rules for all AI agents operating in Project Caspian, enforcing the scientific governance in [Caspian_Agent_Driven_Development_Plan.md](file:///home/henshawmike/Projects/Project_Caspian/Caspian_Agent_Driven_Development_Plan.md) and [docs/development/BRANCHING_STRATEGY.md](file:///home/henshawmike/Projects/Project_Caspian/docs/development/BRANCHING_STRATEGY.md).

## 1. Mandatory Branch Selection & GitHub Sync Rule (Per-Phase Governance)

Every phase and capability in Project Caspian must be developed, tested, committed, and pushed to the designated branch(es) on GitHub (`origin`) according to its research domain:

| Phase / Work Type | Primary Research / Feature Branch(es) | Integration Target |
| :--- | :--- | :--- |
| **Phase 13 — Autonomous Property Discovery & Representation Learning** | `research/ontology` (and `research/learning-mechanisms`) | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 14 — Interaction Discovery & Structured Relationships** | `research/ontology` (and `research/architecture`) | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 15 — Causal Discovery & Intervention Learning** | `research/ontology` | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 16 — Cross-World Transfer & Structural Generalisation** | `feature/multiple-worlds`, `research/ontology` | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 17 — Continual Learning & Persistent Environmental Memory** | `feature/memory`, `research/learning-mechanisms` | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 18 — Integrated Predictive World Model & Autonomous Planning** | `feature/planning`, `research/architecture` | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 19 — Real-World Transfer & Application Prototype** | `feature/application-prototype` | Merge `--no-ff` to `main` after Phase Exit Review |
| **Phase 20 — Integrated System Validation & Research Consolidation** | `research/ontology`, `research/architecture`, `research/learning-mechanisms` | Merge `--no-ff` to `main` after Phase Exit Review |

### Required Workflow at Each Phase:
1. **Checkout Target Branch Before Starting:**
   - Ensure `main` is clean and all unit tests pass (`.venv/bin/pytest -q`).
   - Check out (or create if `feature/*`) the designated branch for the phase (e.g., `research/ontology` for Phase 13) and fast-forward/merge `main` so it starts from the latest baseline.
2. **Stage A — Pre-Registration & Unit Testing Commit:**
   - Write `docs/phase<N>/EXPERIMENT_REGISTRY.md` specifying hypotheses, test statistic $T$, margins, exploratory smoke seeds (`9001–9003`), and disjoint confirmatory seeds *before* running confirmatory experiments.
   - Implement models, experiment scripts (`experiments/phase<N>_experiments.py`), and unit/gradient tests (`tests/test_phase<N>.py`).
   - Run `.venv/bin/pytest -q` and verify 0 failures and 0 semantic firewall violations.
   - Commit Stage A on the target branch and record `git rev-parse HEAD`.
3. **Stage B — Exploratory Smoke, Confirmatory Execution, Analysis & Report Commit:**
   - Run exploratory smoke seeds (`9001–9003`) to verify pipeline finiteness and 0 leakage.
   - Execute the locked confirmatory run, save outputs via `safe_write_json` in `logs/phase<N>/`, run `experiments/analyze_phase<N>.py`, generate the Layer B report `docs/CSP-P<N>-E001.docx`, and record commit hashes in `docs/phase<N>/OPERATIONAL_RECORD.md`.
   - Re-run `.venv/bin/pytest -q` to confirm all tests pass.
   - Commit Stage B on the target branch.
4. **Post-Testing Branch Selection & GitHub Push:**
   - After testing passes at the end of a phase, sync the work to the relevant research/feature branch(es), merge validated phase completions into `main` using `git merge --no-ff`, and push the updated target branches (`research/*`, `feature/*`, and `main`) to GitHub (`origin`).

## 2. Scientific & Safety Invariants
- **Semantic Firewall:** Never expose semantic labels (`food`, `hazard`, `reward`, `beneficial`, `harmful`, etc.) in `Observation`. Every `Observation` must pass `assert_semantic_purity()`.
- **Pure-NumPy Neural Stack:** All neural models in `models/` must use analytical forward/backward passes in NumPy, verified against numerical finite-difference gradients in `tests/`.
- **Overwrite Protection:** Never overwrite historical logs (`logs/`) or `.docx` reports (`docs/CSP-P*.docx`) from completed phases.
- **Honest Verdicts:** Report `MET`, `NOT MET`, or `REFUTED` strictly according to pre-registered criteria. Never tune hyperparameters on confirmatory seeds or disguise null/refuted findings.
