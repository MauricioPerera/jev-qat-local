"""Reproducible local evaluation: python -m jev_qat.evaluate prepare|run|report."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import time

import httpx
import numpy as np

from .calibration import PROMPT_VERSION, PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, DATASET, DIRECTORY, SEED, digest, load_records, prepare
from .eval_metrics import fit_temperature, metrics, paired_nll_difference_ci, scaled_array

PROFILE_NAME = "massive_scenario_es_v1"


def environment(config):
    return {
        "files": {field: sha256_file(config[field]) for field in ("model", "projector", "runtime")},
        "code": {name: sha256_file(Path(__file__).with_name(name)) for name in
                 ("adapter.py", "option_labels.json", "eval_data.py", "evaluate.py", "eval_metrics.py")},
        "settings": {key: config[key] for key in ("device", "context_tokens", "threads")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "httpx")},
        "prompt_version": PROMPT_VERSION,
    }


def request_for(record):
    # The human target and source labels never enter the inference request.
    result = {key: record[key] for key in ("state", "question", "options", "modality")}
    if record.get("task_profile") is not None:
        result["task_profile"] = record["task_profile"]
    return result


def collect(client, records, path, run_dir, phase, expected_prompt_version=PROMPT_VERSION):
    cached = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in cached:
                raise ValueError("Duplicate inference record in cache")
            cached[row["id"]] = row
    collected = []
    fingerprints = set()
    with path.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            payload = request_for(record)
            request_hash = digest(payload)
            result = cached.get(record["id"])
            if result is not None and result["request_sha256"] != request_hash:
                raise ValueError("Cache request differs from the frozen record")
            if result is None:
                for attempt in range(3):
                    try:
                        response = client.post("/predict", json=payload)
                        response.raise_for_status()
                        raw = response.json()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        if attempt == 2:
                            raise
                        time.sleep(2)
                if raw["metadata"].get("calibrated") is not False or raw["metadata"].get("prompt_version") != expected_prompt_version:
                    raise ValueError("Restart the adapter with the current code before evaluating")
                probabilities = raw["probabilities"]
                if set(probabilities) != set(CATEGORIES.values()):
                    raise ValueError("Unexpected option scores from adapter")
                values = [probabilities[CATEGORIES[label]] for label in CATEGORIES]
                if not np.isfinite(values).all() or min(values) < 0 or abs(sum(values)-1) > 1e-6:
                    raise ValueError("Invalid probability vector; evaluation stopped")
                result = {"id": record["id"], "request_sha256": request_hash, "target": record["target"],
                          "probabilities": values, "metadata": raw["metadata"]}
                if "component_probabilities" in raw:
                    result["component_probabilities"] = raw["component_probabilities"]
                stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                stream.flush()
            fingerprints.add(result["metadata"].get("runtime_fingerprint"))
            if len(fingerprints) != 1 or None in fingerprints:
                raise ValueError("Inference runtime fingerprint is missing or changed during evaluation")
            collected.append(result)
            save_json(run_dir / "progress.json", {"phase": phase, "completed": index+1, "total": len(records),
                      "updated_at": datetime.now(timezone.utc).isoformat()})
            if (index+1) % 25 == 0 or index+1 == len(records):
                print(f"{phase}: {index+1}/{len(records)}", flush=True)
    return collected


def arrays(rows):
    classes = list(CATEGORIES)
    return np.array([row["probabilities"] for row in rows]), np.array([classes.index(row["target"]) for row in rows])


def write_profile(run_dir, manifest, env, selection, fingerprint):
    if not selection["validation_approved"]:
        return None
    profile = {
        "schema_version": 1, "method": "temperature_scaling", "temperature": selection["temperature"],
        "validation_approved": True, "dataset": DATASET+":es", "run_id": run_dir.name,
        "scope": {"question": manifest["protocol"]["question"], "options": sorted(CATEGORIES.values()),
                  "modality": "text", "prompt_version": PROMPT_VERSION, "runtime_fingerprint": fingerprint,
                  "model_sha256": env["files"]["model"], "projector_sha256": env["files"]["projector"]},
        "provenance": {"manifest_fingerprint": manifest["manifest_fingerprint"], "revision": manifest["revision"],
                       "selection_sha256": sha256_file(run_dir / "selection.json"),
                       "calibration_predictions_sha256": sha256_file(run_dir / "calibration-predictions.jsonl"),
                       "validation_predictions_sha256": sha256_file(run_dir / "validation-predictions.jsonl")},
        "limitations": ["Only Spanish assistant-request classification with these 18 categories and this prompt.",
                        "Not calibrated for arbitrary text, new categories, images, audio or video.",
                        "Temperature preserves argmax; it does not improve classification accuracy.",
                        "Public data may have appeared in model training; unseen production performance is unknown."],
    }
    save_json(run_dir / "calibration-profile.json", profile)
    save_json(PROFILES / f"{PROFILE_NAME}.json", profile)
    example = request_for(load_records(run_dir.parents[1], "test")[0])
    example["calibration_profile"] = PROFILE_NAME
    save_json(ROOT / "examples" / "decision-massive-calibrated.json", example)
    return PROFILE_NAME


def run(directory=DIRECTORY):
    directory = Path(directory).resolve()
    manifest = prepare(directory)
    config = settings()
    with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
        response = client.get("/health")
        response.raise_for_status()
        print("Fingerprinting local model, projector and runtime...", flush=True)
        env = environment(config)
        run_id = digest({"manifest": manifest["manifest_fingerprint"], "environment": env})[:16]
        run_dir = directory / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = run_dir / "RUNNING.lock"
        try:
            handle = lock.open("x")
        except FileExistsError as error:
            raise RuntimeError(f"An evaluation owns {lock}; inspect its process before removing a stale lock") from error
        try:
            handle.close()
            save_json(run_dir / "environment.json", env)
            save_json(directory / "latest-run.json", {"run_id": run_id, "directory": str(run_dir)})
            cal = collect(client, load_records(directory, "calibration"), run_dir / "calibration-predictions.jsonl", run_dir, "calibration")
            val = collect(client, load_records(directory, "validation"), run_dir / "validation-predictions.jsonl", run_dir, "validation")
            if cal[0]["metadata"]["runtime_fingerprint"] != val[0]["metadata"]["runtime_fingerprint"]:
                raise ValueError("Runtime differs across calibration and validation")
            cal_p, cal_y = arrays(cal)
            val_p, val_y = arrays(val)
            temperature = fit_temperature(cal_p, cal_y)
            raw_val = metrics(val_p, val_y, list(CATEGORIES))
            candidate_val = metrics(scaled_array(val_p, temperature), val_y, list(CATEGORIES))
            approved = candidate_val["negative_log_loss"] < raw_val["negative_log_loss"]-0.0001
            selection = {"temperature": temperature, "selected_temperature": temperature if approved else 1.0,
                         "validation_approved": approved, "raw_validation": raw_val, "candidate_validation": candidate_val,
                         "gate": manifest["protocol"]["activation_gate"], "test_used_for_selection": False}
            selection_path = run_dir / "selection.json"
            if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
                raise ValueError("Previously frozen selection changed; test cannot be used to refit")
            save_json(selection_path, selection)
            print(f"Frozen before test: T={temperature:.6f}; validation_approved={approved}", flush=True)
            fingerprint = cal[0]["metadata"]["runtime_fingerprint"]
            profile_name = write_profile(run_dir, manifest, env, selection, fingerprint)
            test_records = load_records(directory, "test")
            test = collect(client, test_records, run_dir / "test-predictions.jsonl", run_dir, "test")
            if test[0]["metadata"]["runtime_fingerprint"] != fingerprint:
                raise ValueError("Runtime differs between fit and test")
            audit_records = []
            for record in test_records[:manifest["protocol"]["order_audit_count"]]:
                changed = dict(record)
                changed["id"] += ":order-audit"
                order = list(range(len(record["options"])))
                random.Random(f"{SEED}:audit:{record['id']}").shuffle(order)
                changed["options"] = [record["options"][i] for i in order]
                changed["option_labels"] = [record["option_labels"][i] for i in order]
                audit_records.append(changed)
            audit = collect(client, audit_records, run_dir / "order-audit-predictions.jsonl", run_dir, "order-audit")
            if audit[0]["metadata"]["runtime_fingerprint"] != fingerprint:
                raise ValueError("Runtime differs in option-order audit")
            summary = summarize(run_dir, manifest, selection, cal, val, test, audit, profile_name)
            save_json(run_dir / "summary.json", summary)
            save_json(run_dir / "progress.json", {"phase": "complete", "completed": 936, "total": 936})
        finally:
            lock.unlink(missing_ok=True)
    from .eval_report import render
    render(run_dir)
    print(json.dumps({"run_directory": str(run_dir), "temperature": selection["selected_temperature"],
                      "raw_test": summary["test"]["raw"]["accuracy"], "report": str(run_dir / "report.html")}, ensure_ascii=False), flush=True)
    return run_dir


def summarize(run_dir, manifest, selection, cal, val, test, audit, profile_name):
    raw, targets = arrays(test)
    selected = scaled_array(raw, selection["selected_temperature"])
    candidate = scaled_array(raw, selection["temperature"])
    audit_p, audit_y = arrays(audit)
    original = raw[:len(audit)]
    records = {record["id"]: record for record in load_records(run_dir.parents[1], "test")}
    errors = []
    for index, row in enumerate(test):
        predicted = int(raw[index].argmax())
        if predicted != targets[index]:
            errors.append({"id": row["id"], "text": records[row["id"]]["state"], "target": row["target"],
                           "prediction": list(CATEGORIES)[predicted], "raw_confidence": float(raw[index, predicted]),
                           "calibrated_confidence": float(selected[index, predicted])})
    save_json(run_dir / "test-errors.json", sorted(errors, key=lambda row: -row["raw_confidence"]))
    times = np.array([row["metadata"]["elapsed_seconds"] for row in cal+val+test+audit])
    return {
        "schema_version": 1, "dataset": DATASET, "revision": manifest["revision"], "language": "es", "run_id": run_dir.name,
        "classes": list(CATEGORIES), "sample_counts": {name: len(rows) for name, rows in (("calibration", cal), ("validation", val), ("test", test), ("order_audit", audit))},
        "temperature": selection["temperature"], "selected_temperature": selection["selected_temperature"],
        "validation_approved": selection["validation_approved"], "profile_name": profile_name,
        "test_used_for_selection": False,
        "test": {"raw": metrics(raw, targets, list(CATEGORIES)), "calibrated": metrics(selected, targets, list(CATEGORIES)),
                 "candidate": metrics(candidate, targets, list(CATEGORIES)), "nll_change": paired_nll_difference_ci(raw, selected, targets)},
        "order_audit": {"n": len(audit), "prediction_disagreements": int((original.argmax(axis=1) != audit_p.argmax(axis=1)).sum()),
                        "mean_total_variation": float((abs(original-audit_p).sum(axis=1)/2).mean()),
                        "original_accuracy": float((original.argmax(axis=1) == audit_y).mean()),
                        "reordered_accuracy": float((audit_p.argmax(axis=1) == audit_y).mean())},
        "latency_seconds": {"mean": float(times.mean()), "median": float(np.median(times)), "p95": float(np.quantile(times, .95)), "sum": float(times.sum())},
        "artifacts_sha256": {path.name: sha256_file(path) for path in run_dir.glob("*-predictions.jsonl")},
        "limitations": ["Public benchmark of assistant requests, not general social messages or production traffic.",
                        "Pretraining contamination cannot be ruled out.", "450 test messages limit per-category estimates.",
                        "ECE depends on binning; NLL and Brier are also reported.",
                        "Calibration applies only to this taxonomy, Spanish text, question, model and runtime.",
                        "Scalar temperature preserves predictions and does not correct category or ordering errors."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.directory)
        print(json.dumps({"directory": str(args.directory), "splits": manifest["splits"]}, indent=2, ensure_ascii=False))
    elif args.command == "run":
        run(args.directory)
    else:
        from .eval_report import render
        latest = json.loads((args.directory / "latest-run.json").read_text(encoding="utf-8"))
        render(Path(latest["directory"]))


if __name__ == "__main__":
    main()
