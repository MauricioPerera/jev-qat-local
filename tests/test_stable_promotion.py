import pytest

from jev_qat.stable_rescue_run import conditions


def values(accuracy):
    return {"accuracy": accuracy, "macro_f1": .9, "negative_log_loss": .3,
            "per_class": [{"label": "general", "precision": .98, "f1": .98}]}


def test_new_candidate_accepts_preserved_high_ceiling_f1_with_significant_accuracy_gain():
    gates, p = conditions(values(.90), values(.91), {"fixed_errors": 6, "introduced_errors": 0}, True, True, .025)
    assert p == pytest.approx(.015625) and all(gates.values())


@pytest.mark.parametrize("fixed,broken,stable,validated", [(5,0,True,True), (6,1,True,True), (12,0,False,True), (12,0,True,False)])
def test_significance_stability_and_calibration_are_all_required(fixed, broken, stable, validated):
    gates, _ = conditions(values(.9), values(.91), {"fixed_errors": fixed, "introduced_errors": broken}, stable, validated, .025)
    assert not all(gates.values())


def test_a_precise_pvalue_cannot_override_general_regression():
    candidate = values(.91)
    candidate["per_class"][0]["precision"] = .97
    gates, p = conditions(values(.9), candidate, {"fixed_errors": 12, "introduced_errors": 0}, True, True, .025)
    assert p < .025 and not gates["general_precision_not_lower"] and not all(gates.values())
