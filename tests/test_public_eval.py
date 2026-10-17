"""Promotion must reject small, reversed, unverified or practically worse gains."""
import pytest

import copy

import jev_qat.public_eval_run as public_eval
from jev_qat.public_eval_run import conditions


FROZEN = {"v8_task_fingerprint": "unit-test-fixture",
          "files": {"fixture.json": "sha256-fixture"},
          "temperatures": {"v2": 1.2, "v8": 1.4}}


@pytest.fixture(autouse=True)
def isolated_profiles(monkeypatch):
    # Promotion math must be testable without a local bank, model or calibration.
    monkeypatch.setattr(public_eval, "frozen_profiles", lambda: copy.deepcopy(FROZEN))
    monkeypatch.setattr(public_eval, "read_json", lambda path: {"validation_approved": True})


def fixture():
    base = {"accuracy": .90, "macro_f1": .70, "negative_log_loss": .40,
            "per_class": [{"label": "general", "support": 83, "precision": .80, "f1": .80}]}
    candidate = {"accuracy": .92, "macro_f1": .72, "negative_log_loss": .38,
                 "per_class": [{"label": "general", "support": 83, "precision": .85, "f1": .85}]}
    manifest = {"frozen_profiles": copy.deepcopy(FROZEN),
                "protocol": {"alpha": .025, "labels": "Original annotations by humans, documented"}}
    regressions = {"eight_recovered_preserved": True, "three_domain_errors_corrected": True}
    return base, candidate, manifest, regressions


@pytest.mark.parametrize("fixed,broken,pvalue,passes", [(0, 0, 1., False), (5, 0, .03125, False),
    (6, 0, .015625, True), (6, 1, .0625, False), (0, 6, 1., False)])
def test_exact_paired_direction_and_strict_followup_threshold(fixed, broken, pvalue, passes):
    base, candidate, manifest, regressions = fixture()
    gates, p = conditions(base, candidate, {"fixed_errors": fixed, "introduced_errors": broken}, regressions, manifest)
    assert p == pytest.approx(pvalue)
    assert gates["one_sided_exact_p_less_than_0_025"] == passes
    assert all(gates.values()) == passes


def test_significance_cannot_override_lost_recovery_or_general_precision():
    base, candidate, manifest, regressions = fixture()
    regressions["eight_recovered_preserved"] = False
    candidate["per_class"][0]["precision"] = .79
    gates, p = conditions(base, candidate, {"fixed_errors": 12, "introduced_errors": 0}, regressions, manifest)
    assert p < .025
    assert not gates["eight_recoveries_pass_live"] and not gates["general_precision_not_lower"]
    assert not all(gates.values())


def test_calibration_cannot_be_refitted_under_the_frozen_profile():
    base, candidate, manifest, regressions = fixture()
    manifest["frozen_profiles"]["temperatures"]["v8"] += .01
    gates, _ = conditions(base, candidate, {"fixed_errors": 12, "introduced_errors": 0}, regressions, manifest)
    assert not gates["frozen_profiles_unchanged"]


def test_unvalidated_calibration_blocks_promotion_even_with_significance(monkeypatch):
    base, candidate, manifest, regressions = fixture()
    monkeypatch.setattr(public_eval, "read_json", lambda path: {"validation_approved": False})
    gates, p = conditions(base, candidate, {"fixed_errors": 12, "introduced_errors": 0}, regressions, manifest)
    assert p < .025
    assert not gates["calibration_unchanged_and_previously_validated"]
    assert not all(gates.values())
