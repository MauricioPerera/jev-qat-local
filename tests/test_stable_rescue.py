import copy

import numpy as np
import pytest

from jev_qat.eval_data import CATEGORIES
from jev_qat.stable_rescue_math import combine, strength


def distribution(label, confidence):
    values = [(1-confidence)/17]*18
    values[list(CATEGORIES).index(label)] = confidence
    return values


@pytest.fixture
def cfg():
    return {"ramp_low": .8, "ramp_high": .9, "rescue_weight": .7, "confirmation_weight": .7,
        "confirmation_low": .5, "confirmation_high": .7, "qa_confirmation_low": .85, "qa_confirmation_high": .95,
        "protected_domains": ["iot", "transport"], "minimum_general_similarity": .15}


def test_meeting_rescue_has_no_old_point95_cliff(cfg):
    base = distribution("calendar", .999)
    a = combine(base, distribution("general", .946), distribution("general", .99), cfg, .4)
    b = combine(base, distribution("general", .964), distribution("general", .99), cfg, .4)
    assert np.argmax(a) == np.argmax(b) == list(CATEGORIES).index("general")
    assert np.abs(np.array(a)-b).sum() < .03


def test_uncertain_attribute_confirmation_preserves_rescue(cfg):
    result = combine(distribution("qa", .98), distribution("general", .96), distribution("qa", .76), cfg, .4)
    assert np.argmax(result) == list(CATEGORIES).index("general")
    result = combine(distribution("qa", .98), distribution("general", .96), distribution("qa", .99), cfg, .4)
    assert np.argmax(result) == list(CATEGORIES).index("qa")


def test_knowledge_confirmation_is_continuous_at_old_cutoff(cfg):
    base, focused = distribution("qa", .98), distribution("general", .96)
    a = combine(base, focused, distribution("qa", .95-1e-7), cfg, .4)
    b = combine(base, focused, distribution("qa", .95+1e-7), cfg, .4)
    assert np.abs(np.array(a)-b).sum() < 1e-5


def test_unsupported_tool_has_no_rescue(cfg):
    base = distribution("iot", .9)
    assert strength(base, distribution("general", .99), cfg, .02) == 0
    assert combine(base, distribution("general", .99), None, cfg, .02) == pytest.approx(base)


def test_positive_strength_requires_actual_confirmation(cfg):
    with pytest.raises(ValueError):
        combine(distribution("qa", .9), distribution("general", .99), None, cfg, .8)


@pytest.mark.parametrize("value", [float("nan"), -1, 2])
def test_bad_evidence_rejected(cfg, value):
    with pytest.raises(ValueError):
        strength(distribution("calendar", .9), distribution("general", .99), cfg, value)


def test_stable_native_alone_disables_prefix_cache(monkeypatch):
    from jev_qat.adapter import JevQatAdapter
    from jev_qat.eval_data import QUESTION
    from jev_qat.stable_rescue_task import StableNativeTask, FOCUS_VERSION
    task = {"options": list(CATEGORIES.values()), "fingerprint": "fixture-stable", "version": FOCUS_VERSION,
            "examples": [], "general_prompt": True, "cache_prompt": False}
    monkeypatch.setattr(StableNativeTask, "prepare", lambda *a, **kw: task)
    with JevQatAdapter() as a:
        a._verified_labels = 256
        requests = []
        def post(route, payload):
            requests.append(payload)
            return {"system_fingerprint": "fixture", "usage": {"completion_tokens": 1},
                    "choices": [{"logprobs": {"content": [{"top_probs": [{"id": e["id"], "prob": p}
                    for e, p in zip(a.labels, distribution("general", .99))]}]}}]}
        monkeypatch.setattr(a, "_post", post)
        result = a.predict(state="un comentario personal", question=QUESTION, options=list(CATEGORIES.values()), task_profile="massive_es_v9_focus")
        assert requests[0]["cache_prompt"] is False and result["metadata"]["cache_prompt"] is False
        a.predict(state="hola", question="pregunta sencilla", options=["uno", "dos"])
        assert "cache_prompt" not in requests[-1]
