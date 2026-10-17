"""Regression tests for media reuse, freshness and per-question token protection."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from test_e2b import mocked_adapter
from jev_qat.e2b import E2BPredictionRequest
from jev_qat.e2b_budget import COUNT_ROUTE, ContextBudgetError
from jev_qat.e2b_web import register_web_routes


def fixture_adapter(tmp_path, monkeypatch, modality, input_tokens=10):
    adapter, calls = mocked_adapter(input_tokens=input_tokens)
    adapter.config["media_roots"] = [str(tmp_path)]
    source = tmp_path / {"image": "photo.png", "audio": "voice.wav", "video": "movie.mp4"}[modality]
    source.write_bytes(b"original media")
    conversions = []

    def convert(args):
        conversions.append(args)
        if "-show_entries" in args:
            return SimpleNamespace(stdout=json.dumps({"format": {"duration": 60}}).encode())
        output = Path(args[-1])
        if modality == "video":
            for i in range(1, int(args[args.index("-frames:v") + 1]) + 1):
                (output.parent / f"frame-{i:03}.jpg").write_bytes(f"frame {i}".encode())
        else:
            output.write_bytes(b"converted audio")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(adapter, "_run_media", convert)
    request = dict(state="shared state", question="Question A?", options=["yes", "no"],
                   media=str(source), modality=modality, video_frames=2)
    return adapter, calls, conversions, source, request


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_prepare_once_for_three_counts_and_three_independent_predictions(tmp_path, monkeypatch, modality):
    adapter, calls, conversions, _, request = fixture_adapter(tmp_path, monkeypatch, modality)
    with adapter:
        for q in ("Question A?", "Question B?", "Question C?"):
            adapter.count_tokens(**{**request, "question": q})
        for q in ("Question A?", "Question B?", "Question C?"):
            result = adapter.predict(**{**request, "question": q})
            assert result["metadata"]["media_preparation"]["hit"] is True
            assert set(result["probabilities"]) == {"yes", "no"}
        assert adapter._media_cache.preparations == 1 and adapter._media_cache.hits == 5
        assert len(conversions) == {"image": 0, "audio": 1, "video": 2}[modality]
    prompts = [body["messages"][0]["content"] for route, body in calls if route == COUNT_ROUTE]
    assert len(prompts) == 6
    assert all(len([p for p in content if p["type"] == "text"]) == 1 for content in prompts)
    assert all(sum(q in content[-1]["text"] for q in ("Question A?", "Question B?", "Question C?")) == 1 for content in prompts)
    assert adapter._media_cache.bytes == 0


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_switch_off_matches_old_repeated_preparation(tmp_path, monkeypatch, modality):
    adapter, _, conversions, _, request = fixture_adapter(tmp_path, monkeypatch, modality)
    with adapter:
        for i in range(3):
            adapter.count_tokens(**request, reuse_media=False)
            adapter.predict(**request, reuse_media=False, cache_prompt=False)
        assert adapter._media_cache.bytes == 0 and not adapter._media_cache.entries
        assert len(conversions) == (6 if modality == "audio" else 12)


def test_cache_does_not_bypass_overflow_or_reuse_another_question_budget(tmp_path, monkeypatch):
    adapter, calls, _, _, request = fixture_adapter(tmp_path, monkeypatch, "image", input_tokens=4096)
    with adapter:
        assert adapter.count_tokens(**request)["fits_context"] is False
        with pytest.raises(ContextBudgetError):
            adapter.predict(**{**request, "question": "A different very long question?"})
        assert adapter._media_cache.hits == 1
    assert len([r for r, _ in calls if r == COUNT_ROUTE]) == 2
    assert not any(r == "/v1/chat/completions" for r, _ in calls)


def test_file_hash_invalidates_even_when_name_length_and_mtime_are_reused(tmp_path, monkeypatch):
    adapter, _, _, source, request = fixture_adapter(tmp_path, monkeypatch, "image")
    with adapter:
        first = adapter.count_tokens(**request)
        stat = source.stat()
        source.write_bytes(b"modified media")  # Same length.
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = adapter.count_tokens(**request)
        assert second["media_preparation"]["hit"] is False
        assert second["media_preparation"]["sha256"] != first["media_preparation"]["sha256"]
        assert len(adapter._media_cache.entries) == 1


def test_removed_or_disallowed_file_cannot_use_a_cached_copy(tmp_path, monkeypatch):
    adapter, _, _, source, request = fixture_adapter(tmp_path, monkeypatch, "image")
    with adapter:
        adapter.count_tokens(**request)
        source.unlink()
        with pytest.raises(ValueError, match="does not exist"):
            adapter.predict(**request)
        source.write_bytes(b"original media")
        adapter.config["media_roots"] = [str(tmp_path / "other")]
        with pytest.raises(ValueError, match="media_roots"):
            adapter.predict(**request)


def test_frame_setting_and_audio_duration_invalidate_preparation(tmp_path, monkeypatch):
    for modality in ("video", "audio"):
        adapter, _, _, _, request = fixture_adapter(tmp_path, monkeypatch, modality)
        with adapter:
            adapter.count_tokens(**request)
            if modality == "video":
                request["video_frames"] = 1
            else:
                adapter.config["audio_max_seconds"] = 15
            second = adapter.count_tokens(**request)
            assert second["media_preparation"]["hit"] is False
            assert second["media_items"] == 1


def test_expiry_and_size_bound_never_serve_stale_or_over_limit_cache(tmp_path, monkeypatch):
    adapter, _, _, _, request = fixture_adapter(tmp_path, monkeypatch, "image")
    with adapter:
        adapter.count_tokens(**request)
        adapter._media_cache.ttl_seconds = 0
        assert adapter.count_tokens(**request)["media_preparation"]["hit"] is False
        adapter._media_cache.max_bytes = 1
        assert adapter.count_tokens(**request)["media_preparation"]["hit"] is False
        assert adapter._media_cache.bytes == 0


def test_file_change_during_conversion_is_rejected_not_cached(tmp_path, monkeypatch):
    adapter, _, _, source, request = fixture_adapter(tmp_path, monkeypatch, "audio")
    convert = adapter._run_media

    def mutate(args):
        result = convert(args)
        source.write_bytes(b"a changed audio")
        return result

    monkeypatch.setattr(adapter, "_run_media", mutate)
    with adapter, pytest.raises(ValueError, match="changed during preparation"):
        adapter.count_tokens(**request)
    assert adapter._media_cache.bytes == 0


def test_upload_removal_releases_cached_content(tmp_path, monkeypatch):
    adapter, _, _, _, request = fixture_adapter(tmp_path, monkeypatch, "image")
    app = FastAPI()
    register_web_routes(app, {"api_url": "http://127.0.0.1:18167", "api_port": 18167,
                              "max_media_bytes": 1024}, tmp_path, on_remove=adapter.release_media)
    with adapter, TestClient(app) as client:
        upload = client.post("/uploads?filename=photo.png", content=b"image bytes").json()
        adapter.count_tokens(**{**request, "media": upload["media"]})
        assert adapter._media_cache.bytes > 0
        assert client.delete("/uploads/" + upload["upload_id"]).status_code == 204
        assert adapter._media_cache.bytes == 0


def test_reuse_switch_is_strict():
    with pytest.raises(ValidationError):
        E2BPredictionRequest(state="", question="Q?", options=["a", "b"], reuse_media="false")
    adapter, _ = mocked_adapter()
    with adapter, pytest.raises(ValueError, match="reuse_media must be a boolean"):
        adapter.predict(state="", question="Q?", options=["a", "b"], reuse_media="false")
