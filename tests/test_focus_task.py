import numpy as np
import pytest

from jev_qat.calibration import temperature_scale
from jev_qat.eval_data import CATEGORIES, QUESTION
from jev_qat.focus_experiment import mix_array
from jev_qat.focus_task import FocusedTask, NAME, VERSION, combine_probabilities
from jev_qat.adapter import JevQatAdapter


def distribution(winner, confidence):
    values = [(1-confidence)/17]*18
    values[list(CATEGORIES).index(winner)] = confidence
    return values


def test_focus_mixture_can_correct_a_conflicting_general_decision():
    gemma, lexical = distribution("qa", .95), distribution("general", .98)
    result, active = combine_probabilities(gemma, lexical, .7, 2.058)
    assert active
    assert list(CATEGORIES)[np.argmax(result)] == "general"
    assert sum(result) == pytest.approx(1)


def test_unrelated_categories_keep_the_frozen_gemma_distribution():
    gemma, lexical = distribution("transport", .85), distribution("weather", .95)
    result, active = combine_probabilities(gemma, lexical, .8, 2.058)
    assert not active
    assert result == pytest.approx(temperature_scale(gemma, 2.058))


def test_runtime_math_matches_offline_evaluation_and_preserves_category_order():
    from jev_qat.eval_metrics import scaled_array
    gemma = np.array([distribution("qa", .9), distribution("lists", .6), distribution("weather", .8)])
    lexical = np.array([distribution("general", .95), distribution("email", .9), distribution("news", .7)])
    expected = mix_array(scaled_array(gemma, 2.058), lexical, .4)
    actual = np.array([combine_probabilities(a.tolist(), b.tolist(), .4, 2.058)[0] for a,b in zip(gemma, lexical)])
    assert np.allclose(expected, actual, atol=1e-12)


def test_category_weights_match_offline_and_prioritize_the_gemma_focus_category():
    from jev_qat.eval_metrics import scaled_array
    gemma = np.array([distribution("qa", .9), distribution("news", .6), distribution("weather", .8)])
    lexical = np.array([distribution("general", .95), distribution("lists", .9), distribution("email", .7)])
    weights = {"general": .8, "lists": .8, "news": .2}
    calibrated = scaled_array(gemma, 2.058)
    expected = mix_array(calibrated, lexical, weights)
    actual = np.array([combine_probabilities(a.tolist(), b.tolist(), weights, 2.058)[0] for a,b in zip(gemma, lexical)])
    assert np.allclose(expected, actual, atol=1e-12)
    assert actual[0] == pytest.approx(.2*calibrated[0]+.8*lexical[0])
    assert actual[1] == pytest.approx(.8*calibrated[1]+.2*lexical[1])
    assert actual[2] == pytest.approx(calibrated[2])


def test_incomplete_category_weights_are_rejected():
    with pytest.raises(ValueError, match="three focus categories"):
        combine_probabilities(distribution("general", .9), distribution("lists", .9), {"general": .8}, 2)


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_mixture_weights_are_rejected(weight):
    with pytest.raises(ValueError):
        combine_probabilities(distribution("general", .9), distribution("lists", .9), weight, 2)


def test_adapter_exposes_hybrid_method_and_remaps_final_scores(monkeypatch):
    task = {"options": list(CATEGORIES.values()), "version": VERSION, "fingerprint": "fixture", "examples": [],
            "lexical_probabilities": distribution("lists", .99), "mixture_weight": .8, "gemma_component_temperature": 2.058}
    monkeypatch.setattr(FocusedTask, "prepare", lambda *args, **kwargs: task)
    with JevQatAdapter() as adapter:
        adapter._verified_labels = 256
        raw = distribution("email", .98)
        monkeypatch.setattr(adapter, "_post", lambda *args: {"system_fingerprint": "fixture", "choices": [{"logprobs": {"content": [{"top_probs": [
            {"id": label["id"], "prob": value} for label, value in zip(adapter.labels, raw)]}]}}]})
        options = list(reversed(list(CATEGORIES.values())))
        result = adapter.predict(state="añade ana a mi lista", question=QUESTION, options=options, task_profile=NAME)
        assert result["prediction"] == CATEGORIES["lists"]
        assert result["prediction_index"] == options.index(CATEGORIES["lists"])
        assert result["metadata"]["method"] == "gemma_tfidf_gated_mixture"
        assert result["metadata"]["prompt_version"] == VERSION
        assert result["component_probabilities"]["gate_active"]
        assert result["component_probabilities"]["class_order"] == list(CATEGORIES)
