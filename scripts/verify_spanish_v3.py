"""Recompute v3 evidence and optionally verify the deployed mixture and calibration."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx
import joblib
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, DIRECTORY as V1, SOURCE_HASHES, digest, load_records, text_group
from jev_qat.eval_metrics import fit_temperature, metrics, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import DIRECTORY, FOCUS, HEADS, V2, lexical_scores, mix_array, read_json
from jev_qat.focus_run import CALIBRATION_NAME, paired_comparison
from jev_qat.focus_task import NAME, PROFILE_PATH, VERSION, FocusedTask, combine_probabilities
from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION, build_prompt


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify(live=False):
    manifest = read_json(DIRECTORY / "manifest.json")
    original = dict(manifest)
    fingerprint = original.pop("manifest_fingerprint")
    assert digest(original) == fingerprint
    run_dir = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    summary = read_json(run_dir / "summary.json")
    selection = read_json(run_dir / "selection.json")
    env = read_json(run_dir / "environment.json")
    assert run_dir.name == digest({"manifest": fingerprint, "environment": env})[:16]
    records, checks = {}, []
    source_rows = {}
    for split, expected in SOURCE_HASHES.items():
        path = V1 / "source" / f"{split}-es.json.gz"
        assert sha256_file(path) == expected
        source_rows[split] = {str(row["id"]): row for row in
                              (json.loads(line) for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines())}
    for split, info in manifest["splits"].items():
        assert sha256_file(DIRECTORY / info["file"]) == info["sha256"]
        records[split] = load_records(DIRECTORY, split)
        assert len(records[split]) == info["count"]
        assert Counter(row["target"] for row in records[split]) == info["class_counts"]
        assert all(row["group_id"] == text_group(row["state"]) for row in records[split])
        for row in records[split]:
            original_row = source_rows[row["source_split"]][row["source_id"]]
            assert row["state"] == original_row["text"] and row["target"] == original_row["label"]
    old = [row for directory in (V1,V2) for split in ("calibration","validation","test") for row in load_records(directory, split)]
    bank_path = ROOT / manifest["training_bank_file"]
    assert sha256_file(bank_path) == manifest["training_bank_sha256"]
    bank = joblib.load(bank_path)
    assert len(bank["records"]) == manifest["training_count"]
    for row in bank["records"]:
        original_row = source_rows["train"][row["source_id"]]
        assert row["state"] == original_row["text"] and row["target"] == original_row["label"]
    partitions = list(records.items())+[("training", bank["records"]), ("previous_evaluations", old)]
    for index, (a, first) in enumerate(partitions):
        if a not in records:
            continue
        for b, second in partitions[index+1:]:
            for key in ("source_id", "group_id"):
                assert not {row[key] for row in first} & {row[key] for row in second}, (a,b,key)
    checks.append("frozen_splits_no_overlap_with_training_or_previous_evaluations")
    for name, expected in manifest["prior_manifests_sha256"].items():
        assert sha256_file(ROOT / "evaluation" / name / "manifest.json") == expected
    for name, expected in (env["code"] | env["v3_code"]).items():
        assert sha256_file(run_dir / "source" / name) == expected
        assert sha256_file(ROOT / "jev_qat" / name) == expected
    profile = read_json(PROFILE_PATH)
    assert sha256_file(PROFILE_PATH) == env["task_profile_sha256"]
    assert profile == read_json(run_dir / "selected-task-profile.json")
    assert sha256_file(HEADS) == env["heads_sha256"]
    assert sha256_file(ROOT / profile["head_file"]) == profile["head_sha256"]
    head_artifact = joblib.load(ROOT / profile["head_file"])
    assert head_artifact["source_bank_sha256"] == manifest["training_bank_sha256"]
    model = head_artifact["model"]
    assert model.class_weight == {label: (3.0 if label == "general" else 1.5 if label == "news" else 1.2 if label == "lists" else 1.0) for label in CATEGORIES}
    assert max(model.n_iter_) < model.max_iter
    checks.append("archived_scoring_code_training_bank_and_selected_head_checksums")
    development = read_json(run_dir / "development-selection.json")
    assert sha256_file(DIRECTORY / "development-selection.json") == profile["development_selection_sha256"]
    assert development == read_json(DIRECTORY / "development-selection.json")
    assert development["test_used_for_selection"] is False
    extension = read_json(DIRECTORY / "development-plan-extension.json")
    assert extension["test_used"] is False and extension["heads_sha256"] == env["heads_sha256"]
    chosen = development["selected"]
    assert chosen["approved"] and chosen["head_id"] == profile["head_id"] and chosen["weight"] == profile["weight"]
    assert chosen["mean_focus_f1_gain"] >= .02
    assert chosen["metrics"]["accuracy"] >= development["baseline"]["accuracy"]-.005
    assert chosen["metrics"]["macro_f1"] >= development["baseline"]["macro_f1"]-.005
    old_f1 = {row["label"]: row["f1"] for row in development["baseline"]["per_class"]}
    new_f1 = {row["label"]: row["f1"] for row in chosen["metrics"]["per_class"]}
    assert all(new_f1[label] >= old_f1[label]-.01 for label in FOCUS)
    training_ids = {row["id"] for row in bank["records"]}
    base_task = SpanishTask()
    focus_task = FocusedTask(base_task)
    _, _, _, task_fingerprint = focus_task._load()
    gemma, predictions = {}, {}
    for split in records:
        source_path = DIRECTORY / "development-gemma.jsonl" if split == "development" else run_dir / f"{split}-gemma.jsonl"
        gemma[split] = read_lines(source_path)
        if split == "development":
            assert sha256_file(source_path) == development["gemma_predictions_sha256"]
        else:
            assert sha256_file(source_path) == summary["artifacts_sha256"][source_path.name]
        assert len(gemma[split]) == len(records[split])
        for record, prediction in zip(records[split], gemma[split]):
            assert prediction["id"] == record["id"] and prediction["target"] == record["target"]
            assert prediction["request_sha256"] == digest(request_for(record | {"task_profile": "massive_es_v2"}))
            assert prediction["metadata"]["prompt_version"] == BASE_VERSION
            assert prediction["metadata"]["task_profile_fingerprint"] == profile["base_task_fingerprint"]
            assert prediction["metadata"]["runtime_fingerprint"] == "b11669-620d2225c"
            assert set(prediction["metadata"]["retrieved_example_ids"]) <= training_ids
            assert prediction["metadata"]["calibrated"] is False
        raw, targets = arrays(gemma[split])
        lexical = lexical_scores(model, records[split], bank)
        expected = mix_array(scaled_array(raw, profile["gemma_component_temperature"]), lexical, profile["weight"])
        if split == "development":
            assert metrics(expected, targets, list(CATEGORIES)) == chosen["metrics"]
        else:
            path = run_dir / f"{split}-predictions.jsonl"
            assert sha256_file(path) == summary["artifacts_sha256"][path.name]
            predictions[split] = read_lines(path)
            actual, actual_targets = arrays(predictions[split])
            assert np.array_equal(actual_targets, targets)
            assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
            assert np.allclose([row["lexical_probabilities"] for row in predictions[split]], lexical, atol=1e-12)
            for source, result in zip(gemma[split], predictions[split]):
                assert result["derived"] and result["source_request_sha256"] == source["request_sha256"]
                assert result["metadata"]["task_profile_fingerprint"] == task_fingerprint
    # Verify prompt equivalence, which justifies sharing Gemma inference evidence.
    example = records["test"][0]
    arguments = {key: example[key] for key in ("state", "question", "options", "modality")}
    base = base_task.prepare("massive_es_v2", **arguments)
    focused = focus_task.prepare(NAME, **arguments)
    labels = read_json(ROOT / "jev_qat" / "option_labels.json")
    # Label file is a list; only the first 18 labels enter either prompt.
    assert build_prompt(base, example["state"], labels[:18]) == build_prompt(focused, example["state"], labels[:18])
    checks.append("real_gemma_requests_prompt_equivalence_and_recomputed_auxiliary_mixture")
    cal_p, cal_y = arrays(predictions["calibration"])
    assert abs(fit_temperature(cal_p, cal_y)-selection["temperature"]) < 1e-9
    val_p, val_y = arrays(predictions["validation"])
    val_raw = metrics(val_p, val_y, list(CATEGORIES))
    val_cal = metrics(scaled_array(val_p, selection["temperature"]), val_y, list(CATEGORIES))
    assert val_raw == selection["raw_validation"] and val_cal == selection["candidate_validation"]
    assert selection["validation_approved"] == (val_cal["negative_log_loss"] < val_raw["negative_log_loss"]-.0001)
    assert not selection["test_used_for_selection"] and not summary["test_used_for_selection"]
    raw, targets = arrays(predictions["test"])
    calibrated = scaled_array(raw, selection["selected_temperature"])
    base_raw, _ = arrays(gemma["test"])
    baseline = scaled_array(base_raw, profile["gemma_component_temperature"])
    assert metrics(raw, targets, list(CATEGORIES)) == summary["test"]["raw"]
    assert metrics(calibrated, targets, list(CATEGORIES)) == summary["test"]["calibrated"]
    assert metrics(baseline, targets, list(CATEGORIES)) == summary["baseline"]["calibrated"]
    assert np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1))
    assert paired_comparison(baseline, calibrated, targets) == summary["paired_comparison"]
    checks.append("fresh_paired_metrics_bootstrap_temperature_fit_and_validation_gate")
    suites = ET.parse(run_dir / "unit-tests.xml").getroot().findall("testsuite")
    assert suites and all(int(suite.attrib["failures"]) == 0 and int(suite.attrib["errors"]) == 0 for suite in suites)
    test_count = sum(int(suite.attrib["tests"]) for suite in suites)
    if summary["profile_name"]:
        calibration_profile = read_json(PROFILES / f"{CALIBRATION_NAME}.json")
        assert calibration_profile == read_json(run_dir / "calibration-profile.json")
        assert calibration_profile["scope"]["task_profile_fingerprint"] == task_fingerprint
        assert calibration_profile["scope"]["prompt_version"] == VERSION
        assert calibration_profile["temperature"] == selection["selected_temperature"]
        assert calibration_profile["provenance"]["selection_sha256"] == sha256_file(run_dir / "selection.json")
    live_result = {}
    if live:
        config = settings()
        with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
            assert client.get("/evaluation").content == (run_dir / "report.html").read_bytes()
            assert client.get("/evaluation/summary").json() == summary
            payload = request_for(example) | {"task_profile": NAME}
            if summary["profile_name"]:
                payload["calibration_profile"] = CALIBRATION_NAME
            def post(body):
                return client.post("/predict", json=body).raise_for_status().json()
            vector = lambda value: np.array([value["probabilities"][option] for option in CATEGORIES.values()])
            initial, warm = post(payload), post(payload)
            live_result["initial_to_warm_max_probability_delta"] = float(np.max(abs(vector(initial)-vector(warm))))
            saved = []
            # Three focus messages plus the first untouched test item, each with warm adjacent A/B calls.
            focus_examples = [next(row for row in records["test"] if row["target"] == category) for category in FOCUS]
            for record in [example]+focus_examples:
                body = request_for(record) | {"task_profile": NAME}
                if summary["profile_name"]:
                    body["calibration_profile"] = CALIBRATION_NAME
                post(body)
                result = post(body)
                components = result["component_probabilities"]
                mixture, gate = combine_probabilities(components["gemma_raw"], components["lexical"], profile["weight"], profile["gemma_component_temperature"])
                assert gate == components["gate_active"]
                assert components["class_order"] == list(CATEGORIES)
                expected = temperature_scale(mixture, selection["selected_temperature"]) if summary["profile_name"] else mixture
                assert np.allclose(vector(result), expected, atol=1e-12, rtol=1e-12)
                assert result["metadata"]["method"] == "gemma_tfidf_gated_mixture"
                assert result["metadata"]["task_profile_fingerprint"] == task_fingerprint
                changed = post(body | {"options": list(reversed(body["options"]))})
                assert changed["prediction"] == result["prediction"]
                assert np.allclose(vector(result), vector(changed), atol=1e-6, rtol=0)
                saved.append({"request": body, "response": result, "order_max_probability_delta": float(np.max(abs(vector(result)-vector(changed))))})
            save_json(run_dir / "live-examples.json", saved)
            assert client.post("/predict", json=payload | {"calibration_profile": "massive_scenario_es_v2"}).status_code == 422
            if summary["profile_name"]:
                assert client.post("/predict", json=payload | {"task_profile": "massive_es_v2"}).status_code == 422
            previous = post(request_for(example) | {"task_profile": "massive_es_v2", "calibration_profile": "massive_scenario_es_v2"})
            assert previous["metadata"]["calibrated"] and previous["metadata"]["task_profile"] == "massive_es_v2"
            generic = post(request_for(example))
            assert generic["metadata"]["task_profile"] is None and not generic["metadata"]["calibrated"]
            live_result |= {"passed": True, "order_audit_messages": len(saved), "order_invariant": True,
                            "mixture_formula_verified": True, "incompatible_calibrations_rejected": True,
                            "generic_and_v2_preserved": True}
            checks.append("live_report_mixture_calibration_scopes_and_option_order")
    result = {"passed": True, "run_id": run_dir.name, "checks": checks, "unit_tests_passed": test_count, "live": live_result,
              "package_versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "joblib", "numpy", "scipy", "httpx")}}
    save_json(run_dir / "verification.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    verify(parser.parse_args().live)
