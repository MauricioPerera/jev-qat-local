import json

import httpx
import pytest
from pydantic import ValidationError

from jev_qat.adapter import BackendError
from jev_qat.config import settings as e4b_settings
from jev_qat.e2b import ALIAS, E2BPredictionRequest, E2BQatAdapter, register_prediction_routes, settings
from jev_qat.e2b_budget import COUNT_ROUTE, ContextBudgetError


def mocked_adapter(alias=ALIAS, input_tokens=10, backend_context=4096, count_status=200):
    adapter = E2BQatAdapter()
    adapter.client.close()
    calls = []
    labels = {entry["label"]: entry["id"] for entry in adapter.labels}

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": alias}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": backend_context}})
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [labels[payload["content"]]]})
        if request.url.path == COUNT_ROUTE:
            return httpx.Response(count_status, json={"input_tokens": input_tokens})
        assert request.url.path == "/v1/chat/completions"
        entries = [{"id": int(k), "prob": .2 if i == 0 else .8 / (len(payload["logit_bias"]) - 1)}
                   for i, k in enumerate(payload["logit_bias"])]
        return httpx.Response(200, json={
            "choices": [{"logprobs": {"content": [{"top_probs": entries}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "timings": {"cache_n": 0}, "system_fingerprint": "b11669-620d2225c",
        })

    adapter.client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return adapter, calls


def test_e2b_scores_keep_true_model_identity_and_disable_native_cache():
    adapter, calls = mocked_adapter()
    with adapter:
        result = adapter.predict(state="hola", question="¿Saluda?", options=["no", "sí"], cache_prompt=False)
    assert result["prediction"] == "sí" and result["prediction_index"] == 1
    assert result["metadata"]["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf"
    assert result["metadata"]["calibrated"] is False
    assert result["metadata"]["cache_prompt"] is False
    assert calls[-1][1]["cache_prompt"] is False
    budget = result["metadata"]["context_budget"]
    assert budget["input_tokens"] == 10 and budget["total_tokens"] == 11
    assert budget["remaining_tokens"] == 4085 and budget["checked_before_inference"]
    assert calls[-2] == (COUNT_ROUTE, calls[-1][1])


def test_e2b_rejects_calibration_and_e4b_rescue_before_inference():
    adapter, calls = mocked_adapter()
    with adapter:
        with pytest.raises(ValueError, match="E4B calibration cannot be reused"):
            adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"], calibration_profile="massive_scenario_es_v2")
        with pytest.raises(ValueError, match="rescue profiles"):
            adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"], task_profile="massive_es_v9")
    assert not calls


def test_e2b_cannot_silently_connect_to_e4b():
    adapter, calls = mocked_adapter(alias="gemma-e4b-qat-q4-local")
    with adapter:
        with pytest.raises(BackendError, match="requires its own"):
            adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert not calls


def test_e2b_http_request_keeps_strict_cache_and_media_limits():
    with pytest.raises(ValidationError):
        E2BPredictionRequest(state="", question="¿Qué?", options=["a", "b"], cache_prompt="false")
    with pytest.raises(ValidationError):
        E2BPredictionRequest(state="", question="¿Qué?", options=["a", "b"], video_frames=17)
    with pytest.raises(ValidationError):
        E2BPredictionRequest(state="", question="¿Qué?", options=["a", "b"], arbitrary=True)


def test_e2b_configuration_uses_distinct_files_and_ports():
    small, large = settings(), e4b_settings()
    assert small["model"] != large["model"] and small["projector"] != large["projector"]
    assert {small["api_port"], small["backend_port"]}.isdisjoint({large["api_port"], large["backend_port"]})
    assert all(small[k] == large[k] for k in ("runtime", "device", "threads", "context_tokens", "audio_max_seconds", "max_video_frames"))


@pytest.mark.parametrize("inputs,accepted", [(4094, True), (4095, True), (4096, False), (5000, False)])
def test_e2b_reserves_output_at_the_context_boundary(inputs, accepted):
    adapter, calls = mocked_adapter(input_tokens=inputs)
    with adapter:
        if accepted:
            result = adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
            assert result["metadata"]["context_budget"]["total_tokens"] == inputs + 1
        else:
            with pytest.raises(ContextBudgetError) as error:
                adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
            assert error.value.detail["code"] == "context_limit_exceeded"
            assert error.value.detail["input_tokens"] == inputs
            assert error.value.detail["context_limit_tokens"] == 4096
    assert any(route == "/v1/chat/completions" for route, _ in calls) is accepted


@pytest.mark.parametrize("count", [None, True, -1, 0, 9.5, "100"])
def test_e2b_invalid_counts_fail_closed(count):
    adapter, calls = mocked_adapter(input_tokens=count)
    with adapter, pytest.raises(BackendError, match="invalid input token count"):
        adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert not any(route == "/v1/chat/completions" for route, _ in calls)


def test_e2b_unavailable_counter_never_falls_back_to_inference():
    adapter, calls = mocked_adapter(count_status=404)
    with adapter, pytest.raises(BackendError, match="HTTP 404"):
        adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert not any(route == "/v1/chat/completions" for route, _ in calls)


@pytest.mark.parametrize("context", [None, True, 0, -1, "4096", 4096.0])
def test_e2b_invalid_backend_context_never_runs_inference(context):
    adapter, calls = mocked_adapter(backend_context=context)
    with adapter, pytest.raises(BackendError, match="invalid backend context"):
        adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert not any(route in (COUNT_ROUTE, "/v1/chat/completions") for route, _ in calls)


@pytest.mark.parametrize("backend_context,configured,expected", [(2048, 4096, 2048), (8192, 4096, 4096), (8192, 8192, 4096), (4096, 1024, 1024)])
def test_e2b_uses_the_smallest_actual_configured_or_hard_limit(backend_context, configured, expected):
    adapter, _ = mocked_adapter(backend_context=backend_context)
    adapter.config["context_tokens"] = configured
    with adapter:
        result = adapter.predict(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert result["metadata"]["context_budget"]["context_limit_tokens"] == expected


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("modality,parts", [
    ("image", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]),
    ("audio", [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]),
    ("video", [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}"}} for i in range(2)]),
])
def test_e2b_checks_exact_prepared_media_payload_before_completion(monkeypatch, modality, parts, overflow):
    from contextlib import contextmanager

    @contextmanager
    def prepared_content(*_args):
        yield list(parts)

    adapter, calls = mocked_adapter(input_tokens=4200 if overflow else 10)
    monkeypatch.setattr(adapter, "_content", prepared_content)
    with adapter:
        kwargs = dict(state="contexto", question="¿Qué hay?", options=["sí", "no"], modality=modality, media="fixture", video_frames=2)
        if overflow:
            with pytest.raises(ContextBudgetError):
                adapter.predict(**kwargs)
        else:
            adapter.predict(**kwargs)
    count_request = next(payload for route, payload in calls if route == COUNT_ROUTE)
    if overflow:
        assert not any(route == "/v1/chat/completions" for route, _ in calls)
    else:
        assert count_request == calls[-1][1]
    content = count_request["messages"][0]["content"]
    assert content[:-1] == parts
    assert all(text in content[-1]["text"] for text in ("contexto", "¿Qué hay?", "sí", "no"))


@pytest.fixture
def retrieval_bank(tmp_path, monkeypatch):
    """A real, tiny retrieval bank instead of workstation evaluation artifacts."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    import jev_qat.spanish_task as spanish_task
    from jev_qat.calibration import sha256_file
    from jev_qat.eval_data import text_group

    state = "Pon leche en la lista de compras"
    rows = [{"id": "fixture:lists", "state": state, "target": "lists", "group_id": text_group(state)}]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([state])
    bank = tmp_path / "bank.joblib"
    joblib.dump({"version": spanish_task.VERSION, "vectorizer": vectorizer,
                 "matrix": matrix, "records": rows}, bank)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"version": spanish_task.VERSION, "bank_file": str(bank),
                                   "bank_sha256": sha256_file(bank), "neighbors": 6}), encoding="utf8")
    monkeypatch.setattr(spanish_task, "PROFILE_PATH", profile)


def test_e2b_checks_retrieved_examples_after_prompt_preparation(retrieval_bank):
    from jev_qat.eval_data import CATEGORIES, QUESTION

    adapter, calls = mocked_adapter()
    # These 18 labels have already been checked at service startup.
    adapter._verified_labels = 256
    with adapter:
        result = adapter.predict(state="Añade pan a la lista", question=QUESTION,
                                 options=list(CATEGORIES.values()), task_profile="massive_es_v2")
    assert result["metadata"]["retrieved_example_ids"]
    assert calls[-2] == (COUNT_ROUTE, calls[-1][1])
    assert "Pon leche en la lista de compras" in calls[-2][1]["messages"][0]["content"][-1]["text"]
    assert len(calls[-2][1]["messages"][0]["content"][-1]["text"]) > 1000


@pytest.mark.parametrize("inputs, fits", [(10, True), (4095, True), (4096, False), (5000, False)])
def test_e2b_preview_measures_fitting_and_oversized_requests_without_inference(inputs, fits):
    adapter, calls = mocked_adapter(input_tokens=inputs)
    with adapter:
        result = adapter.count_tokens(state="hola", question="¿Saluda?", options=["sí", "no"])
    assert result["input_tokens"] == inputs and result["total_tokens"] == inputs + 1
    assert result["fits_context"] is fits and result["inference_executed"] is False
    assert result["remaining_tokens"] == 4095 - inputs
    assert result["available_tokens"] == max(0, 4095 - inputs)
    assert result["overflow_tokens"] == max(0, inputs - 4095)
    assert result["media_items"] == 0 and result["option_count"] == 2
    assert not adapter._count_only
    assert [route for route, _ in calls if route != "/tokenize"] == [COUNT_ROUTE]


def test_e2b_preview_does_not_change_subsequent_prediction_or_its_payload():
    adapter, calls = mocked_adapter()
    payload = dict(state="hola", question="¿Saluda?", options=["sí", "no"], cache_prompt=False)
    with adapter:
        adapter.count_tokens(**payload)
        result = adapter.predict(**payload)
    assert result["prediction"] == "no"
    calls = [(route, body) for route, body in calls if route != "/tokenize"]
    assert [route for route, _ in calls] == [COUNT_ROUTE, COUNT_ROUTE, "/v1/chat/completions"]
    assert calls[0][1] == calls[1][1] == calls[2][1]


@pytest.mark.parametrize("modality,item", [
    ("image", {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}),
    ("audio", {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}),
    ("video", {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}),
])
def test_e2b_preview_counts_prepared_media_and_cleans_up(monkeypatch, modality, item):
    from contextlib import contextmanager
    cleaned = []

    @contextmanager
    def prepared(*_args):
        try:
            yield [item] * (2 if modality == "video" else 1)
        finally:
            cleaned.append(True)

    adapter, calls = mocked_adapter()
    monkeypatch.setattr(adapter, "_content", prepared)
    with adapter:
        result = adapter.count_tokens(state="contexto", question="¿Qué hay?", options=["sí", "no"],
                                      modality=modality, media="fixture", video_frames=2)
    assert cleaned == [True] and not adapter._count_only
    assert result["media_items"] == (2 if modality == "video" else 1)
    assert result["modality"] == modality
    body = next(body for route, body in calls if route == COUNT_ROUTE)
    assert body["messages"][0]["content"][:-1] == [item] * result["media_items"]
    assert not any(route == "/v1/chat/completions" for route, _ in calls)


def route_client(adapter):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.state.adapter = adapter
    register_prediction_routes(app)
    return TestClient(app)


def test_e2b_count_endpoint_reports_overflow_while_predict_still_rejects_it():
    adapter, calls = mocked_adapter(input_tokens=5000)
    payload = dict(state="hola", question="¿Saluda?", options=["sí", "no"])
    with adapter, route_client(adapter) as client:
        response = client.post("/count-tokens", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["fits_context"] is False and result["overflow_tokens"] == 905
        assert result["inference_executed"] is False
        rejected = client.post("/predict", json=payload)
        assert rejected.status_code == 422 and rejected.json()["detail"]["code"] == "context_limit_exceeded"
        schema = client.get("/openapi.json").json()
        assert "/count-tokens" in schema["paths"]
        assert "E2BTokenBudgetResponse" in schema["components"]["schemas"]
    assert not any(route == "/v1/chat/completions" for route, _ in calls)


@pytest.mark.parametrize("extra", [
    {"cache_prompt": "false"}, {"options": ["sí", "sí"]}, {"options": ["solo"]},
    {"task_profile": "massive_es_v9"}, {"calibration_profile": "massive_scenario_es_v2"},
    {"video_frames": 17}, {"unexpected": True},
])
def test_e2b_count_endpoint_uses_predict_validation_and_resets_mode(extra):
    adapter, calls = mocked_adapter()
    with adapter, route_client(adapter) as client:
        response = client.post("/count-tokens", json={"state": "hola", "question": "¿Saluda?", "options": ["sí", "no"], **extra})
        assert response.status_code == 422
        assert not adapter._count_only
    assert not calls


def test_e2b_count_endpoint_backend_failure_is_closed_and_does_not_leak_mode():
    adapter, calls = mocked_adapter(count_status=404)
    with adapter, route_client(adapter) as client:
        response = client.post("/count-tokens", json={"state": "hola", "question": "¿Saluda?", "options": ["sí", "no"]})
        assert response.status_code == 502
        assert not adapter._count_only
    assert not any(route == "/v1/chat/completions" for route, _ in calls)


def test_e2b_preview_includes_retrieved_examples_from_the_shared_prompt(retrieval_bank):
    from jev_qat.eval_data import CATEGORIES, QUESTION
    adapter, calls = mocked_adapter()
    adapter._verified_labels = 256
    payload = dict(state="Añade pan a la lista", question=QUESTION, options=list(CATEGORIES.values()), task_profile="massive_es_v2")
    with adapter:
        adapter.count_tokens(**payload)
        adapter.predict(**payload)
    assert calls[0] == calls[1] == (COUNT_ROUTE, calls[2][1])
    assert "Pon leche en la lista de compras" in calls[0][1]["messages"][0]["content"][-1]["text"]
    assert len(calls[0][1]["messages"][0]["content"][-1]["text"]) > 1000


def test_e2b_concurrent_prediction_waits_for_preview_and_still_runs_inference():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered, release, prediction_started = threading.Event(), threading.Event(), threading.Event()
    adapter, calls = mocked_adapter()
    payload = dict(state="hola", question="¿Saluda?", options=["sí", "no"])

    def block_first_count(request):
        if request.url.path == COUNT_ROUTE and not entered.is_set():
            entered.set()
            assert release.wait(5)

    def predict():
        prediction_started.set()
        return adapter.predict(**payload)

    adapter.client.event_hooks["request"].append(block_first_count)
    with adapter, ThreadPoolExecutor(max_workers=2) as pool:
        try:
            preview = pool.submit(adapter.count_tokens, **payload)
            assert entered.wait(5)
            prediction = pool.submit(predict)
            assert prediction_started.wait(5) and not prediction.done()
        finally:
            release.set()
        assert preview.result(timeout=5)["inference_executed"] is False
        assert prediction.result(timeout=5)["prediction"] == "no"
    assert sum(route == "/v1/chat/completions" for route, _ in calls) == 1
