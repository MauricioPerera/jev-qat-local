import numpy as np
import pytest

from jev_qat.calibration import temperature_scale
from jev_qat.eval_data import CATEGORIES
from jev_qat.eval_metrics import scaled_array
from jev_qat.selective_experiment import mix_array
from jev_qat.selective_task import combine_probabilities


def distribution(label, confidence):
    result = [(1-confidence)/17]*18
    result[list(CATEGORIES).index(label)] = confidence
    return result


def test_weak_auxiliary_evidence_cannot_displace_gemma():
    gemma = distribution("weather",.9)
    result,gate = combine_probabilities(gemma,distribution("general",.49),.4,2.058,.5)
    assert not gate and result == pytest.approx(temperature_scale(gemma,2.058))


def test_nonfocus_auxiliary_cannot_displace_a_correct_list_decision():
    gemma = distribution("lists",.9)
    result,gate = combine_probabilities(gemma,distribution("calendar",.99),.4,2.058,.5)
    assert not gate and result == pytest.approx(temperature_scale(gemma,2.058))


def test_confident_focus_auxiliary_uses_exactly_the_selected_weight():
    gemma,lexical = distribution("qa",.7),distribution("general",.8)
    result,gate = combine_probabilities(gemma,lexical,.4,2.058,.5)
    assert gate
    assert result == pytest.approx(.6*np.array(temperature_scale(gemma,2.058))+.4*np.array(lexical))


def test_runtime_selective_gate_matches_vectorized_evaluation():
    gemma = np.array([distribution("qa",.8),distribution("lists",.95),distribution("qa",.8)])
    lexical = np.array([distribution("general",.8),distribution("play",.95),distribution("news",.49)])
    expected = mix_array(scaled_array(gemma,2.058),lexical,.4,.5)
    actual = [combine_probabilities(a.tolist(),b.tolist(),.4,2.058,.5)[0] for a,b in zip(gemma,lexical)]
    assert np.allclose(expected,actual,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("threshold",[-.1,1.1,float("nan")])
def test_invalid_expert_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        combine_probabilities(distribution("general",.8),distribution("news",.9),.4,2,threshold)
