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


def test_saved_phase14_investigation_report():
    import json
    rep = json.load(open("logs/phase14/investigation_report.json"))
    assert rep["forensic_verdict"] == "LEGIT"
    assert rep["pillar_b_first_principles_investigation"]["after_fix"]["all_three_criteria_resolved"] is True


def test_saved_phase15_investigation_report():
    import json
    rep = json.load(open("logs/phase15/investigation_report.json"))
    assert rep["forensic_verdict"] == "LEGIT"
    assert rep["pillar_b_first_principles_investigation"]["after_fix"]["all_phenomena_explained"] is True


def test_saved_phase16_investigation_report():
    import json
    rep = json.load(open("logs/phase16/investigation_report.json"))
    assert rep["forensic_verdict"] == "LEGIT"
    assert rep["pillar_b_first_principles_investigation"]["after_fix"]["all_three_root_causes_resolved"] is True


def test_saved_phase16_independent_confirmation():
    import json
    ind = json.load(open("logs/phase16/independent_v2_analysis.json"))
    assert ind["leak_rows_total"] == 0
    assert ind["autonomous_shift_detection"] is True
    assert ind["independent_tests"]["P16-IND-1"]["shift_detected_seeds"] == 10
    assert ind["independent_tests"]["P16-IND-2"]["false_shift_triggers"] == 0
    assert ind["independent_tests"]["P16-IND-3"]["met"] is True
    assert ind["independent_tests"]["P16-IND-4"]["met"] is True
    assert ind["independent_tests"]["P16-IND-5"]["met"] is True
    assert ind["original_tests"]["P16-H5"]["met"] is True


def test_saved_phase17_investigation_report():
    import json
    rep = json.load(open("logs/phase17/investigation_report.json"))
    assert rep["forensic_verdict"] == "LEGIT"
    assert rep["pillar_b_first_principles_investigation"]["after_fix"]["root_cause_verified"] is True






