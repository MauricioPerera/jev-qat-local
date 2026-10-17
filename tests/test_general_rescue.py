import numpy as np
import pytest

from jev_qat.eval_data import CATEGORIES
from jev_qat.general_rescue_math import GENERAL,combine,needs_review


def distribution(label,p):
    values=[(1-p)/17]*18
    values[list(CATEGORIES).index(label)]=p
    return values


def test_existing_general_winners_are_preserved():
    base=distribution("general",.8)
    assert not needs_review(base,.95,2.058)
    assert combine(base,distribution("qa",.99),.7,.95)==base


def test_confident_specialized_decisions_skip_second_stage():
    assert not needs_review(distribution("weather",.99999999),.95,2.058)


def test_weak_general_evidence_cannot_override():
    base=distribution("qa",.7)
    assert combine(base,distribution("general",.94),.7,.95)==base


def test_general_rescue_preserves_all_other_probability_ratios():
    base=distribution("qa",.7)
    focused=distribution("general",.99)
    result=combine(base,focused,.7,.95)
    assert sum(result)==pytest.approx(1)
    assert result[GENERAL]>base[GENERAL]
    assert np.argmax(result)==GENERAL
    assert result[12]/result[8]==pytest.approx(base[12]/base[8])


def test_non_general_second_stage_does_not_change_specialized_category():
    base=distribution("email",.7)
    assert combine(base,distribution("lists",.99),.7,.95)==base


def test_adapter_executes_nested_stages_without_deadlock_and_remaps_options(monkeypatch):
    from jev_qat.adapter import JevQatAdapter
    from jev_qat.eval_data import QUESTION
    from jev_qat.general_rescue_task import RescueTask
    from jev_qat.general_task import GeneralTask,VERSION as FOCUSED_VERSION
    from jev_qat.spanish_task import SpanishTask,VERSION as BASE_VERSION
    rule={"threshold":.95,"weight":.7,"minimum_probability":.95,"component_temperature":2.058}
    monkeypatch.setattr(RescueTask,"_load",lambda self:(rule,"fixture-rescue"))
    task={"options":list(CATEGORIES.values()),"fingerprint":"fixture-source","examples":[]}
    monkeypatch.setattr(SpanishTask,"prepare",lambda *a,**kw:task|{"version":BASE_VERSION})
    monkeypatch.setattr(GeneralTask,"prepare",lambda *a,**kw:task|{"version":FOCUSED_VERSION,"general_prompt":True})
    with JevQatAdapter() as adapter:
        adapter._verified_labels=256
        sources=iter([distribution("qa",.7),distribution("general",.99)])
        def post(*args):
            values=next(sources)
            return {"system_fingerprint":"fixture","usage":{"completion_tokens":1,"prompt_tokens":20,"total_tokens":21},
                    "choices":[{"logprobs":{"content":[{"top_probs":[{"id":label["id"],"prob":p} for label,p in zip(adapter.labels,values)]}]}}]}
        monkeypatch.setattr(adapter,"_post",post)
        options=list(reversed(list(CATEGORIES.values())))
        result=adapter.predict(state="hola",question=QUESTION,options=options,task_profile="massive_es_v6")
        assert result["prediction"]==CATEGORIES["general"]
        assert result["prediction_index"]==options.index(CATEGORIES["general"])
        assert result["metadata"]["usage"]["completion_tokens"]==2
        assert result["metadata"]["llm_calls"]==2
        assert result["metadata"]["task_profile_fingerprint"]=="fixture-rescue"
        assert sum(result["probabilities"].values())==pytest.approx(1)
