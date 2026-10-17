"""Audit saved evaluation evidence, optionally checking the live calibrated API."""
import argparse
from collections import Counter
import json
from pathlib import Path

import httpx
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, DIRECTORY, digest, load_records
from jev_qat.eval_metrics import metrics, scaled_array
from jev_qat.evaluate import arrays, request_for


def verify(directory, live=False):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = manifest.pop("manifest_fingerprint")
    assert digest(manifest) == expected_manifest, "Manifest fingerprint changed"
    latest = json.loads((directory / "latest-run.json").read_text(encoding="utf-8"))
    run_dir = directory / "runs" / latest["run_id"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    records, predictions = {}, {}
    checks = []
    for split, info in manifest["splits"].items():
        assert sha256_file(directory / info["file"]) == info["sha256"], f"Data changed: {split}"
        records[split] = load_records(directory, split)
        assert len(records[split]) == info["count"]
        assert dict(sorted(Counter(row["target"] for row in records[split]).items())) == info["class_counts"]
        prediction_path = run_dir / f"{split}-predictions.jsonl"
        assert sha256_file(prediction_path) == summary["artifacts_sha256"][prediction_path.name]
        predictions[split] = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
        assert len(predictions[split]) == len(records[split])
        for record, prediction in zip(records[split], predictions[split]):
            assert prediction["id"] == record["id"] and prediction["target"] == record["target"]
            assert prediction["request_sha256"] == digest(request_for(record))
    for a, b in (("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
        for key in ("group_id", "source_id"):
            assert not {row[key] for row in records[a]} & {row[key] for row in records[b]}
    checks.append("frozen_data_hashes_counts_requests_and_disjoint_splits")
    assert summary["test_used_for_selection"] is False and selection["test_used_for_selection"] is False
    val_p, val_y = arrays(predictions["validation"])
    val_raw = metrics(val_p, val_y, list(CATEGORIES))
    val_candidate = metrics(scaled_array(val_p, selection["temperature"]), val_y, list(CATEGORIES))
    assert val_raw == selection["raw_validation"] and val_candidate == selection["candidate_validation"]
    assert selection["validation_approved"] == (val_candidate["negative_log_loss"] < val_raw["negative_log_loss"]-0.0001)
    checks.append("independent_validation_gate")
    test_p, test_y = arrays(predictions["test"])
    calibrated = scaled_array(test_p, selection["selected_temperature"])
    assert metrics(test_p, test_y, list(CATEGORIES)) == summary["test"]["raw"]
    assert metrics(calibrated, test_y, list(CATEGORIES)) == summary["test"]["calibrated"]
    assert np.array_equal(test_p.argmax(axis=1), calibrated.argmax(axis=1))
    checks.append("test_metrics_recomputed_and_argmax_preserved")
    for name, expected in env["code"].items():
        assert sha256_file(ROOT / "jev_qat" / name) == expected, f"Evaluation code changed: {name}"
    for artifact in ("report.html", "report.md", "reliability.png", "reliability.svg", "confusion.png", "confusion.svg"):
        assert (run_dir / artifact).stat().st_size > 100
    checks.append("code_fingerprints_and_report_artifacts")
    if summary["profile_name"]:
        profile_path = PROFILES / (summary["profile_name"]+".json")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert profile == json.loads((run_dir / "calibration-profile.json").read_text(encoding="utf-8"))
        assert profile["provenance"]["selection_sha256"] == sha256_file(run_dir / "selection.json")
        assert profile["temperature"] == selection["selected_temperature"]
        checks.append("scoped_profile_matches_frozen_selection")
    live_results = {}
    if live:
        config = settings()
        with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
            response = client.get("/evaluation")
            assert response.status_code == 200 and response.content == (run_dir / "report.html").read_bytes()
            assert client.get("/evaluation/summary").json() == summary
            payload = request_for(records["test"][0])
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            default = response.json()
            assert default["metadata"]["calibrated"] is False and "raw_probabilities" not in default
            live_results["default_uncalibrated"] = True
            if summary["profile_name"]:
                payload["calibration_profile"] = summary["profile_name"]
                response = client.post("/predict", json=payload)
                response.raise_for_status()
                result = response.json()
                assert result["metadata"]["calibrated"] is True
                expected = temperature_scale([result["raw_probabilities"][option] for option in payload["options"]], selection["selected_temperature"])
                assert np.allclose(expected, [result["probabilities"][option] for option in payload["options"]], atol=1e-12)
                assert result["prediction"] == max(result["raw_probabilities"], key=result["raw_probabilities"].get)
                live_results["calibrated_confidence"] = result["confidence"]
                live_results["raw_confidence_same_request"] = max(result["raw_probabilities"].values())
                save_json(run_dir / "live-calibrated-example.json", {"request": payload, "response": result})
                changed = payload | {"question": payload["question"]+" Otra pregunta."}
                assert client.post("/predict", json=changed).status_code == 422
                image = payload | {"modality": "image", "media": str(ROOT / "inference" / "qat-q4-vulkan" / "test-image.png")}
                assert client.post("/predict", json=image).status_code == 422
                live_results["wrong_question_and_image_calibration_rejected"] = True
            checks.append("live_report_default_calibrated_math_and_scope_rejection")
    result = {"passed": True, "run_id": run_dir.name, "checks": checks, "live": live_results}
    save_json(run_dir / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    verify(args.directory, args.live)
