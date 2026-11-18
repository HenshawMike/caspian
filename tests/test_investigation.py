from pathlib import Path
from experiments.investigation import (
    audit_confirmatory_logs,
    audit_live_gradient_and_weight_movement,
    audit_models_ast,
)


def test_models_ast_has_zero_oracle_or_semantic_violations():
    res = audit_models_ast(Path("models"))
    assert res["passed"] and len(res["violations"]) == 0


def test_phase13_confirmatory_logs_legitimate():
    res = audit_confirmatory_logs()
    assert res["passed"]
    assert res["leak_rows_total"] == 0
    assert res["non_finite_rows"] == 0
    assert res["seed_independence_verified"]
    assert res["shuffled_control_passed"]
    assert res["structural_negative_control_passed"]


def test_live_gradient_and_weight_movement():
    res = audit_live_gradient_and_weight_movement()
    assert res["passed"]
    assert res["total_weight_l2_movement"] > 0.5
    assert res["loss_ratio_final_over_init"] < 0.05


def test_saved_phase13_investigation_report():
    import json
    rep = json.load(open("logs/phase13/investigation_report.json"))
    assert rep["forensic_verdict"] == "LEGIT"
    assert rep["pillar_b_first_principles_investigation"]["after_fix"]["all_three_criteria_resolved"] is True

