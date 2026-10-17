"""Verify v2 train/test separation, paired results, scoped calibration and live API."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, DIRECTORY as OLD_DIRECTORY, digest, load_records
from jev_qat.eval_metrics import metrics, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.improve_spanish import CALIBRATION_NAME, DIRECTORY
from jev_qat.spanish_task import NAME, PROFILE_PATH, VERSION


def verify(live=False):
    manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    copy = dict(manifest)
    expected = copy.pop("manifest_fingerprint")
    assert digest(copy) == expected
    run_dir = Path(json.loads((DIRECTORY / "latest-run.json").read_text(encoding="utf-8"))["directory"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert run_dir.name == digest({"manifest": expected, "environment": environment})[:16]
    checks, records, predictions = [], {}, {}
    for split, info in manifest["splits"].items():
        assert sha256_file(DIRECTORY / info["file"]) == info["sha256"]
        records[split] = [json.loads(line) for line in (DIRECTORY / info["file"]).read_text(encoding="utf-8").splitlines()]
        assert len(records[split]) == info["count"]
    splits = list(records)
    for index, a in enumerate(splits):
        for b in splits[index+1:]:
            for key in ("source_id", "group_id"):
                assert not {row[key] for row in records[a]} & {row[key] for row in records[b]}, f"Overlap: {a}/{b}/{key}"
    old = [row for split in ("calibration", "validation", "test") for row in load_records(OLD_DIRECTORY, split)]
    for split in ("calibration", "validation", "test", "retrieval_training"):
        for key in ("source_id", "group_id"):
            assert not {row[key] for row in records[split]} & {row[key] for row in old}, f"Old data in fresh {split}"
    checks.append("frozen_hashes_and_disjoint_train_development_calibration_validation_fresh_test")
    training_ids = {row["id"] for row in records["retrieval_training"]}
    for split in ("development", "calibration", "validation", "test", "baseline"):
        path = run_dir / f"{split}-predictions.jsonl"
        assert sha256_file(path) == summary["artifacts_sha256"][path.name]
        predictions[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        source = records["test" if split == "baseline" else split]
        assert len(source) == len(predictions[split])
        for record, prediction in zip(source, predictions[split]):
            request = record if split == "baseline" else record | {"task_profile": NAME}
            assert prediction["request_sha256"] == digest(request_for(request))
            assert prediction["target"] == record["target"] and prediction["id"] == record["id"]
            if split != "baseline":
                retrieved = prediction["metadata"]["retrieved_example_ids"]
                assert len(retrieved) <= 6 and set(retrieved) <= training_ids and record["id"] not in retrieved
                assert prediction["metadata"]["prompt_version"] == VERSION
    checks.append("prediction_requests_and_training_only_retrieval_evidence")
    development = json.loads((run_dir / "development-selection.json").read_text(encoding="utf-8"))
    assert development["approved"] and development["fresh_test_used"] is False
    assert development["candidate"]["accuracy"] >= development["baseline"]["accuracy"]+.01
    assert development["candidate"]["macro_f1"] >= development["baseline"]["macro_f1"]
    val_p, val_y = arrays(predictions["validation"])
    val = metrics(val_p, val_y, list(CATEGORIES))
    val_cal = metrics(scaled_array(val_p, selection["temperature"]), val_y, list(CATEGORIES))
    assert val == selection["raw_validation"] and val_cal == selection["candidate_validation"]
    assert selection["validation_approved"] == (val_cal["negative_log_loss"] < val["negative_log_loss"]-.0001)
    assert selection["test_used_for_selection"] is False and summary["test_used_for_selection"] is False
    checks.append("development_selection_and_fresh_validation_calibration_gate")
    raw, targets = arrays(predictions["test"])
    calibrated = scaled_array(raw, selection["selected_temperature"])
    base, base_targets = arrays(predictions["baseline"])
    assert np.array_equal(targets, base_targets)
    assert metrics(raw, targets, list(CATEGORIES)) == summary["test"]["raw"]
    assert metrics(calibrated, targets, list(CATEGORIES)) == summary["test"]["calibrated"]
    assert metrics(base, targets, list(CATEGORIES)) == summary["baseline"]["raw"]
    assert metrics(scaled_array(base, summary["baseline"]["temperature"]), targets, list(CATEGORIES)) == summary["baseline"]["legacy_calibrated"]
    assert np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1))
    baseline_correct, new_correct = base.argmax(axis=1) == targets, raw.argmax(axis=1) == targets
    assert int((~baseline_correct & new_correct).sum()) == summary["paired_comparison"]["fixed_errors"]
    assert int((baseline_correct & ~new_correct).sum()) == summary["paired_comparison"]["introduced_errors"]
    checks.append("fresh_paired_metrics_recomputed_and_temperature_preserves_predictions")
    audit_path = run_dir / "order-audit-predictions.jsonl"
    assert sha256_file(audit_path) == summary["artifacts_sha256"][audit_path.name]
    audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(audit) == summary["order_audit"]["n"]
    for source, prediction in zip(records["test"], audit):
        changed = source | {"options": list(reversed(source["options"])), "task_profile": NAME}
        assert prediction["request_sha256"] == digest(request_for(changed))
        assert prediction["id"] == source["id"]+":audit"
    audit_p, _ = arrays(audit)
    assert int((raw[:len(audit)].argmax(axis=1) != audit_p.argmax(axis=1)).sum()) == summary["order_audit"]["prediction_disagreements"]
    checks.append("order_audit_requests_and_results")
    for name, expected in (environment["code"] | environment["v2_code"]).items():
        assert sha256_file(run_dir / "source" / name) == expected
        if name != "service.py":
            assert sha256_file(ROOT / "jev_qat" / name) == expected
    assert sha256_file(ROOT / manifest["bank_file"]) == manifest["bank_sha256"]
    for name, expected in json.loads((run_dir / "source-manifest.json").read_text(encoding="utf-8")).items():
        assert sha256_file(run_dir / "source" / name) == expected
    checks.append("archived_runtime_source_and_frozen_training_bank")
    suites = ET.parse(run_dir / "unit-tests.xml").getroot().findall("testsuite")
    assert suites and all(int(suite.attrib["failures"]) == 0 and int(suite.attrib["errors"]) == 0 for suite in suites)
    unit_tests_count = sum(int(suite.attrib["tests"]) for suite in suites)
    checks.append("unit_test_evidence")
    if summary["profile_name"]:
        profile = json.loads((PROFILES / f"{CALIBRATION_NAME}.json").read_text(encoding="utf-8"))
        assert profile == json.loads((run_dir / "calibration-profile.json").read_text(encoding="utf-8"))
        assert profile["scope"]["task_profile"] == NAME
        assert profile["scope"]["prompt_version"] == VERSION
        assert profile["provenance"]["selection_sha256"] == sha256_file(run_dir / "selection.json")
        assert profile["temperature"] == selection["selected_temperature"]
        checks.append("v2_calibration_profile_matches_frozen_selection")
    live_results = {}
    if live:
        config = settings()
        with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
            response = client.get("/evaluation")
            assert response.status_code == 200 and response.content == (run_dir / "report.html").read_bytes()
            assert client.get("/evaluation/summary").json() == summary
            payload = request_for(records["test"][0] | {"task_profile": NAME})
            if summary["profile_name"]:
                payload["calibration_profile"] = CALIBRATION_NAME
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            initial = response.json()
            # Warm the identical canonical prefix before the A/B option-order
            # comparison. Cold/full-prefill versus reused Vulkan KV caches can
            # have different rounding; measure that separately below.
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            vector = lambda value: np.array([value["probabilities"][option] for option in payload["options"]])
            live_results["initial_to_warm_max_probability_delta"] = float(np.max(abs(vector(initial)-vector(result))))
            assert result["metadata"]["task_profile"] == NAME
            assert result["metadata"]["prompt_version"] == VERSION
            if summary["profile_name"]:
                expected = temperature_scale([result["raw_probabilities"][option] for option in payload["options"]], selection["selected_temperature"])
                assert np.allclose(expected, [result["probabilities"][option] for option in payload["options"]], atol=1e-12)
            reversed_payload = payload | {"options": list(reversed(payload["options"]))}
            reversed_result = client.post("/predict", json=reversed_payload).raise_for_status().json()
            assert reversed_result["prediction"] == result["prediction"]
            assert np.allclose([result["probabilities"][option] for option in payload["options"]],
                               [reversed_result["probabilities"][option] for option in payload["options"]], atol=1e-6)
            live_results["order_invariant"] = True
            live_results["calibrated"] = result["metadata"]["calibrated"]
            save_json(run_dir / "live-calibrated-example.json", {"request": payload, "response": result})
            if summary["profile_name"]:
                wrong = payload | {"calibration_profile": "massive_scenario_es_v1"}
                assert client.post("/predict", json=wrong).status_code == 422
                wrong = payload | {"task_profile": None}
                assert client.post("/predict", json=wrong).status_code == 422
                live_results["incompatible_calibrations_rejected"] = True
            generic = request_for(records["test"][0])
            default = client.post("/predict", json=generic).raise_for_status().json()
            assert default["metadata"]["task_profile"] is None and default["metadata"]["calibrated"] is False
            live_results["generic_interface_preserved"] = True
            legacy = client.post("/predict", json=generic | {"calibration_profile": "massive_scenario_es_v1"}).raise_for_status().json()
            assert legacy["metadata"]["calibrated"] is True and legacy["metadata"]["task_profile"] is None
            live_results["legacy_calibration_still_valid_for_generic_prompt"] = True
            after_generic = client.post("/predict", json=payload).raise_for_status().json()
            live_results["after_generic_max_probability_delta"] = float(np.max(abs(vector(after_generic)-vector(result))))
            live_results["after_generic_prediction_unchanged"] = after_generic["prediction"] == result["prediction"]
            checks.append("live_report_calibration_order_invariance_and_scope_rejection")
    result = {"passed": True, "run_id": run_dir.name, "checks": checks, "live": live_results,
              "unit_tests_passed": unit_tests_count,
              "retrieval_package_versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "joblib")}}
    save_json(run_dir / "verification.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    verify(args.live)
