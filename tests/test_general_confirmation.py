import numpy as np
import pytest

from jev_qat.eval_data import CATEGORIES
from jev_qat.general_confirm_math import combine


def distribution(category,p):
    values = [(1-p)/17]*18
    values[list(CATEGORIES).index(category)] = p
    return values


def test_knowledge_requires_stronger_confirmation_than_destination():
    proposed = distribution("general",.7)
    thresholds = {"default":.6,"qa":.95}
    assert combine(proposed,distribution("qa",.8),.7,thresholds) == proposed
    assert np.argmax(combine(proposed,distribution("qa",.98),.7,thresholds)) == list(CATEGORIES).index("qa")
    result = combine(proposed,distribution("transport",.66),.7,thresholds)
    assert np.argmax(result) == list(CATEGORIES).index("transport") and sum(result) == pytest.approx(1)


def test_specialized_proposals_skip_confirmation_changes():
    proposed = distribution("weather",.7)
    assert combine(proposed,distribution("general",.99),.7,{"default":.6,"qa":.95}) == proposed


@pytest.mark.parametrize("invalid",[float("nan"),-.1,1.1])
def test_bad_confirmation_scores_fail_closed(invalid):
    values = distribution("general",.9); values[0] = invalid
    with pytest.raises(ValueError):
        combine(distribution("general",.7),values,.7,.6)


def test_adapter_executes_three_actual_stages_and_preserves_option_mapping(monkeypatch):
    from jev_qat.adapter import JevQatAdapter
    from jev_qat.eval_data import QUESTION
    from jev_qat.general_confirm_adapter import ConfirmedTask
    from jev_qat.general_confirm_task import ConfirmationTask, VERSION as CONFIRM_VERSION
    from jev_qat.general_precision_task import PrecisionTask
    from jev_qat.general_task import GeneralTask, VERSION as FOCUSED_VERSION
    from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION
    rule = {"threshold":.95,"component_temperature":2.058,"weight":.7,"minimum_probability":.95,
            "minimum_general_similarity":.15,"protected_domains":["iot"]}
    monkeypatch.setattr(PrecisionTask,"_load",lambda self:(rule,"fixture-v7"))
    monkeypatch.setattr(ConfirmedTask,"_load",lambda self:({"weight":.7,"minimum_confirmation_probability":{"default":.6,"qa":.95}},"fixture-v8"))
    monkeypatch.setattr("jev_qat.general_precision_task.general_evidence",lambda state:{"general_similarity":.8,"nearest_general_id":"training-only"})
    task = {"options":list(CATEGORIES.values()),"fingerprint":"fixture","examples":[]}
    monkeypatch.setattr(SpanishTask,"prepare",lambda *a,**kw:task | {"version":BASE_VERSION})
    monkeypatch.setattr(GeneralTask,"prepare",lambda *a,**kw:task | {"version":FOCUSED_VERSION,"general_prompt":True})
    monkeypatch.setattr(ConfirmationTask,"prepare",lambda *a,**kw:task | {"version":CONFIRM_VERSION,"confirmation_prompt":True})
    with JevQatAdapter() as adapter:
        adapter._verified_labels = 256
        sources = iter([distribution("qa",.7),distribution("general",.99),distribution("qa",.98)])
        calls = []
        def post(*args):
            values = next(sources);calls.append(1)
            return {"system_fingerprint":"fixture","usage":{"completion_tokens":1,"prompt_tokens":20,"total_tokens":21},
                    "choices":[{"logprobs":{"content":[{"top_probs":[{"id":label["id"],"prob":p} for label,p in zip(adapter.labels,values)]}]}}]}
        monkeypatch.setattr(adapter,"_post",post)
        options = list(reversed(list(CATEGORIES.values())))
        result = adapter.predict(state="mensaje sin etiqueta de referencia",question=QUESTION,options=options,task_profile="massive_es_v8")
        assert result["prediction"] == CATEGORIES["qa"] and result["prediction_index"] == options.index(CATEGORIES["qa"])
        assert result["metadata"]["llm_calls"] == result["metadata"]["usage"]["completion_tokens"] == len(calls) == 3
        assert result["metadata"]["task_profile_fingerprint"] == "fixture-v8" and result["component_probabilities"]["confirmed"]
