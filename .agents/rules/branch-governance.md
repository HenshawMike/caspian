---
trigger: always_on
description: Enforces Project Caspian Git branch selection, two-stage pre-registration commit discipline, and GitHub branch pushing after testing each phase.
---

# Project Caspian Branching & Phase Testing Rule

1. **Select the Right Branch Before & After Each Phase:**
   - Always work on the appropriate `research/*` (`research/ontology`, `research/architecture`, `research/learning-mechanisms`) or `feature/*` (`feature/memory`, `feature/planning`, `feature/multiple-worlds`) branch for the current phase.
   - Never commit unvalidated experimental code directly to `main`.
2. **Two-Stage Commit Discipline:**
   - **Pre-Registration Commit:** Commit `docs/phase<N>/EXPERIMENT_REGISTRY.md`, model code, experiment runner, and `tests/test_phase<N>.py` *before* running confirmatory seeds. Verify `.venv/bin/pytest -q` passes.
   - **Confirmatory Execution Commit:** Run locked disjoint confirmatory seeds, run `experiments/analyze_phase<N>.py`, generate `docs/CSP-P<N>-E001.docx`, document commit hashes in `docs/phase<N>/OPERATIONAL_RECORD.md`, and re-run `.venv/bin/pytest -q`.
3. **Select Target Branches & Push to GitHub After Testing:**
   - Once all unit tests, gradient checks, and semantic firewall audits pass for a phase, ensure the commits are placed on the designated domain branches (`research/ontology`, `research/architecture`, `research/learning-mechanisms`, or `feature/*`), merge validated phases into `main` via `--no-ff`, and push the selected branches to `origin` on GitHub.
