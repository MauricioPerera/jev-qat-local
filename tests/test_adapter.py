import math

import pytest

from jev_qat.adapter import BackendError, JevQatAdapter, normalize_probabilities
from jev_qat.config import settings
from jev_qat.service import PredictionRequest
from pydantic import ValidationError


def test_normalization_uses_token_ids_and_preserves_relative_scores():
    probabilities = normalize_probabilities(
        [{"id": 22, "prob": 0.6}, {"id": 11, "prob": 0.2}, {"id": 99, "prob": 0.2}],
        [11, 22],
    )
    assert probabilities == pytest.approx([0.25, 0.75])
    assert math.fsum(probabilities) == pytest.approx(1)


@pytest.mark.parametrize("entries", [
    [{"id": 1, "prob": 0.9}],
    [{"id": 1, "prob": 0}, {"id": 2, "prob": 0}],
    [{"id": 1, "prob": float("nan")}, {"id": 2, "prob": 0.5}],
    [{"id": 1, "prob": -0.1}, {"id": 2, "prob": 0.5}],
])
def test_invalid_backend_scores_are_rejected_instead_of_fabricated(entries):
    with pytest.raises(BackendError):
        normalize_probabilities(entries, [1, 2])


@pytest.mark.parametrize("options", [["only one"], ["same", "same"], ["", "valid"], ["a", 2], [str(i) for i in range(257)]])
def test_invalid_options_fail_before_inference(options):
    with JevQatAdapter() as adapter:
        with pytest.raises(ValueError):
            adapter.predict(state="test", question="Which?", options=options)


def test_media_outside_workspace_is_rejected(tmp_path):
    config = settings()
    config["media_roots"] = [str(tmp_path / "allowed")]
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not an image")
    with JevQatAdapter(config=config) as adapter:
        with pytest.raises(ValueError, match="media_roots"):
            adapter.predict(state="", question="Which?", options=["yes", "no"], media=outside, modality="image")


def test_invalid_api_types_are_not_silently_coerced():
    with pytest.raises(ValidationError):
        PredictionRequest(state="", question="Which?", options=["yes", "no"], video_frames=True)


def test_label_inventory_supports_256_distinct_options():
    with JevQatAdapter() as adapter:
        assert len(adapter.labels) == 256
        assert len({entry["id"] for entry in adapter.labels}) == 256
        assert len({entry["label"] for entry in adapter.labels}) == 256
