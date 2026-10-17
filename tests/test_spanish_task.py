import json

import pytest

from jev_qat.adapter import JevQatAdapter
from jev_qat.calibration import CalibrationProfiles, sha256_file
from jev_qat.eval_data import CATEGORIES, QUESTION, load_records, text_group
from jev_qat.spanish_task import NAME, VERSION, SpanishTask, build_prompt


def test_spanish_task_rejects_wrong_scope_before_loading_bank():
    task = SpanishTask()
    arguments = {"state": "hola", "question": QUESTION, "options": list(CATEGORIES.values()), "modality": "text"}
    for changes in ({"question": "otra"}, {"options": ["sí", "no"]}, {"modality": "image"}):
        with pytest.raises(ValueError, match="requires"):
            task.prepare(NAME, **(arguments | changes))
    with pytest.raises(ValueError, match="Unknown"):
        task.prepare("unknown", **arguments)


def test_canonical_label_scores_are_remapped_to_callers_options(monkeypatch):
    examples = [{"id": "training:1", "state": "¿qué canción es?", "target": "music"}]
    task = {"options": list(CATEGORIES.values()), "version": VERSION, "fingerprint": "fixture", "examples": examples}
    monkeypatch.setattr(SpanishTask, "prepare", lambda *args, **kwargs: task)
    with JevQatAdapter() as adapter:
        adapter._verified_labels = 256
        payloads = []
        def post(route, payload):
            payloads.append(payload)
            return {"system_fingerprint": "fixture", "choices": [{"logprobs": {"content": [{"top_probs": [
                {"id": entry["id"], "prob": .9 if index == 9 else .1/17} for index, entry in enumerate(adapter.labels[:18])]}]}}]}
        monkeypatch.setattr(adapter, "_post", post)
        options = list(reversed(list(CATEGORIES.values())))
        result = adapter.predict(state="qué tema suena", question=QUESTION, options=options, task_profile=NAME)
        assert result["prediction"] == CATEGORIES["music"]
        assert result["prediction_index"] == options.index(CATEGORIES["music"])
        assert result["probabilities"][CATEGORIES["music"]] == pytest.approx(.9)
        assert result["metadata"]["prompt_version"] == VERSION
        assert result["metadata"]["retrieved_example_ids"] == ["training:1"]
        reversed_result = adapter.predict(state="qué tema suena", question=QUESTION, options=list(reversed(options)), task_profile=NAME)
        assert reversed_result["probabilities"] == result["probabilities"]
        assert payloads[0] == payloads[1]


def test_prompt_uses_supplied_training_examples_and_separates_target_message():
    labels = [{"label": chr(65+i)} for i in range(18)]
    task = {"examples": [{"state": "dame el teléfono de carlos", "target": "email"}]}
    prompt = build_prompt(task, "mensaje final", labels)
    assert '"etiqueta": "F"' in prompt
    assert prompt.index("EJEMPLOS ETIQUETADOS") < prompt.index("MENSAJE A CLASIFICAR")
    assert prompt.count("mensaje final") == 1


def test_calibration_cannot_cross_between_generic_and_retrieval_profiles(tmp_path, monkeypatch):
    import jev_qat.calibration as calibration
    monkeypatch.setattr(calibration, "PROFILES", tmp_path)
    config = {key: tmp_path / key for key in ("model", "projector")}
    for path in config.values():
        path.write_bytes(b"gguf fixture")
    scope = {"question": QUESTION, "options": sorted(CATEGORIES.values()), "modality": "text", "prompt_version": VERSION,
             "runtime_fingerprint": "runtime", "task_profile": NAME, "task_profile_fingerprint": "bank1",
             **{key+"_sha256": sha256_file(path) for key, path in config.items()}}
    (tmp_path / "v2.json").write_text(json.dumps({"schema_version": 1, "validation_approved": True, "scope": scope, "temperature": 2, "dataset": "fixture"}))
    profiles = CalibrationProfiles(config)
    args = {"question": QUESTION, "options": list(CATEGORIES.values()), "modality": "text", "runtime_fingerprint": "runtime"}
    with pytest.raises(ValueError, match="does not match"):
        profiles.apply("v2", [.5, .5], **args)
    _, metadata = profiles.apply("v2", [.5, .5], **args, prompt_version=VERSION, task_profile=NAME, task_profile_fingerprint="bank1")
    assert metadata["temperature"] == 2
    with pytest.raises(ValueError, match="does not match"):
        profiles.apply("v2", [.5, .5], **args, prompt_version=VERSION, task_profile=NAME, task_profile_fingerprint="otherbank")
