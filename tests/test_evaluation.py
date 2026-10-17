import json
import math

import numpy as np
import pytest

import jev_qat.calibration as calibration
from jev_qat.calibration import CalibrationProfiles, PROMPT_VERSION, sha256_file, temperature_scale
from jev_qat.eval_data import CATEGORIES, disjoint_sources, stratified_sample, text_group
from jev_qat.eval_metrics import fit_temperature, metrics, negative_log_loss, scaled_array
from jev_qat.evaluate import request_for


def test_temperature_fitting_reduces_overconfidence_without_changing_decisions():
    probabilities = np.tile([.99, .01], (100, 1))
    targets = np.array([0]*60+[1]*40)
    temperature = fit_temperature(probabilities, targets)
    calibrated = scaled_array(probabilities, temperature)
    assert temperature == pytest.approx(math.log(99)/math.log(1.5), rel=1e-5)
    assert calibrated[0, 0] == pytest.approx(.6, abs=1e-5)
    assert negative_log_loss(calibrated, targets) < negative_log_loss(probabilities, targets)
    assert np.array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))
    assert temperature_scale([.99, .01], temperature) == pytest.approx(calibrated[0])


def test_metrics_match_hand_computed_binary_example():
    result = metrics([[.8, .2], [.4, .6]], [0, 0], ["a", "b"])
    assert result["accuracy"] == .5
    assert result["macro_f1"] == pytest.approx(1/3)
    assert result["negative_log_loss"] == pytest.approx(-math.log(.32)/2)
    assert result["brier_multiclass_sum"] == pytest.approx(.4)
    assert result["ece_10_bins"] == pytest.approx(.4)
    assert sum(row["count"] for row in result["reliability"]) == 2
    assert result["confusion_matrix"] == [[1, 1], [0, 0]]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        temperature_scale([.4, .6], temperature)


def test_normalized_text_dedup_and_conflicts_across_source_splits():
    def row(index, text, label="alarm"):
        return {"id": str(index), "text": text, "label": label, "lang": "es"}
    sources = {"test": [row(1, "HOLA   mundo"), row(2, "ambiguo")],
               "validation": [row(3, "hola mundo"), row(4, "único")],
               "train": [row(5, "ambiguo", "audio"), row(6, "otro")]}
    cleaned, audit = disjoint_sources(sources)
    assert [row["id"] for row in cleaned["test"]] == ["1"]
    assert [row["id"] for row in cleaned["validation"]] == ["4"]
    assert [row["id"] for row in cleaned["train"]] == ["6"]
    assert audit["ambiguous_text_groups"] == 1
    assert text_group("  HOLA\tmundo") == text_group("hola mundo")


def test_stratified_sampling_is_reproducible_and_inference_request_has_no_target():
    rows = [{"id": str(index), "label": label} for label in CATEGORIES for index in range(20)]
    selected = stratified_sample(rows, 90, 123)
    assert selected == stratified_sample(list(reversed(rows)), 90, 123)
    assert len(selected) == 90
    assert {row["label"] for row in selected} == set(CATEGORIES)
    record = {"state": "hola", "question": "¿cuál?", "options": ["a", "b"], "modality": "text", "target": "a", "source_id": "1"}
    assert set(request_for(record)) == {"state", "question", "options", "modality"}


def test_profile_is_scoped_and_validates_loaded_model_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "PROFILES", tmp_path)
    config = {field: tmp_path / field for field in ("model", "projector")}
    for path in config.values():
        path.write_bytes(b"checkpoint")
    scope = {"question": "question", "options": ["a", "b"], "modality": "text",
             "prompt_version": PROMPT_VERSION, "runtime_fingerprint": "build1",
             **{field+"_sha256": sha256_file(path) for field, path in config.items()}}
    profile = {"schema_version": 1, "validation_approved": True, "temperature": 2,
               "scope": scope, "dataset": "fixture"}
    (tmp_path / "valid.json").write_text(json.dumps(profile))
    profiles = CalibrationProfiles(config)
    request = {"question": "question", "options": ["b", "a"], "modality": "text", "runtime_fingerprint": "build1"}
    values, metadata = profiles.apply("valid", [.1, .9], **request)
    assert values == pytest.approx([.25, .75])
    assert metadata["temperature"] == 2
    for changes in ({"question": "different"}, {"options": ["a", "c"]}, {"modality": "image"}, {"runtime_fingerprint": "build2"}):
        with pytest.raises(ValueError, match="does not match"):
            profiles.apply("valid", [.1, .9], **(request | changes))
    with pytest.raises(ValueError, match="Invalid calibration"):
        profiles.apply("../valid", [.1, .9], **request)
    config["model"].write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="loaded model"):
        profiles.apply("valid", [.1, .9], **request)


def test_unvalidated_profile_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "PROFILES", tmp_path)
    (tmp_path / "unvalidated.json").write_text(json.dumps({"schema_version": 1, "validation_approved": False}))
    with pytest.raises(ValueError, match="independent validation"):
        CalibrationProfiles({}).apply("unvalidated", [.5, .5], question="q", options=["a", "b"], modality="text", runtime_fingerprint="b")
