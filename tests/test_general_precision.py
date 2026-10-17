import numpy as np
import pytest

from jev_qat.eval_data import CATEGORIES
from jev_qat.general_precision_math import CONTEXTUAL_TOOLS, combine


def distribution(label, probability):
    result = [(1-probability)/17]*18
    result[list(CATEGORIES).index(label)] = probability
    return result


@pytest.fixture
def rule():
    return {"threshold":.95,"component_temperature":2.058,"weight":.7,"minimum_probability":.95,
            "minimum_general_similarity":.15,"protected_domains":list(CONTEXTUAL_TOOLS)}


def test_tool_rescue_requires_training_support_even_with_confident_focus(rule):
    base = distribution("iot",.7)
    assert combine(base,distribution("general",.999),rule,.10) == base
    result = combine(base,distribution("general",.999),rule,.30)
    assert list(CATEGORIES)[int(np.argmax(result))] == "general"
    assert sum(result) == pytest.approx(1.)


def test_information_and_past_event_rescues_keep_original_behavior(rule):
    for category in ("qa","calendar"):
        result = combine(distribution(category,.7),distribution("general",.99),rule,.01)
        assert list(CATEGORIES)[int(np.argmax(result))] == "general"


def test_base_general_and_non_general_ratios_are_preserved(rule):
    base = distribution("general",.8)
    assert combine(base,distribution("qa",.99),rule,.01) == base
    base = distribution("qa",.7)
    result = combine(base,distribution("general",.99),rule,.30)
    assert result[12]/result[8] == pytest.approx(base[12]/base[8])


@pytest.mark.parametrize("score",[float("nan"),float("inf"),-.1,1.1])
def test_invalid_training_support_fails_closed(rule,score):
    with pytest.raises(ValueError):
        combine(distribution("qa",.7),distribution("general",.99),rule,score)


@pytest.mark.parametrize("category,support,expected,calls",[("iot",.10,"iot",1),("qa",.01,"general",2),("calendar",.42,"general",2)])
def test_live_path_skips_vetoed_prompt_and_remaps_options(monkeypatch,rule,category,support,expected,calls):
    from jev_qat.adapter import JevQatAdapter
    from jev_qat.eval_data import QUESTION
    from jev_qat.general_precision_task import PrecisionTask
    from jev_qat.general_task import GeneralTask, VERSION as FOCUSED_VERSION
    from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION
    monkeypatch.setattr(PrecisionTask,"_load",lambda self:(rule,"fixture-v7"))
    monkeypatch.setattr("jev_qat.general_precision_task.general_evidence",lambda state:{"general_similarity":support,"nearest_general_id":"training-only"})
    task = {"options":list(CATEGORIES.values()),"fingerprint":"fixture-source","examples":[]}
    monkeypatch.setattr(SpanishTask,"prepare",lambda *a,**kw:task | {"version":BASE_VERSION})
    monkeypatch.setattr(GeneralTask,"prepare",lambda *a,**kw:task | {"version":FOCUSED_VERSION,"general_prompt":True})
    with JevQatAdapter() as adapter:
        adapter._verified_labels = 256
        sources = iter([distribution(category,.7),distribution("general",.99)])
        seen = []
        def post(*args):
            values = next(sources); seen.append(1)
            return {"system_fingerprint":"fixture","usage":{"completion_tokens":1,"prompt_tokens":20,"total_tokens":21},
                    "choices":[{"logprobs":{"content":[{"top_probs":[{"id":label["id"],"prob":p} for label,p in zip(adapter.labels,values)]}]}}]}
        monkeypatch.setattr(adapter,"_post",post)
        options = list(reversed(list(CATEGORIES.values())))
        result = adapter.predict(state="un mensaje sin etiqueta en la petición",question=QUESTION,options=options,task_profile="massive_es_v7")
        assert result["prediction"] == CATEGORIES[expected]
        assert result["prediction_index"] == options.index(CATEGORIES[expected])
        assert len(seen) == result["metadata"]["llm_calls"] == result["metadata"]["usage"]["completion_tokens"] == calls
        assert result["metadata"]["task_profile_fingerprint"] == "fixture-v7"
        assert result["component_probabilities"]["reviewed"] == (calls == 2)
