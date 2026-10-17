"""Train-only retrieval, development selection and fresh paired test evaluation."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random

import httpx
import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from .calibration import PROMPT_VERSION, PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, DATASET, DIRECTORY as OLD_DIRECTORY, QUESTION, SOURCE_HASHES, SEED, digest, disjoint_sources, load_records, stratified_sample, text_group
from .eval_metrics import fit_temperature, metrics, paired_nll_difference_ci, scaled_array
from .evaluate import arrays, collect, environment, request_for
from .spanish_task import NAME, PROFILE_PATH, VERSION

DIRECTORY = ROOT / "evaluation" / "massive-es-v2"
CALIBRATION_NAME = "massive_scenario_es_v2"


def write_records(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records), encoding="utf-8")
    return {"file": path.name, "sha256": sha256_file(path), "count": len(records),
            "class_counts": dict(sorted(Counter(row["target"] for row in records).items()))}


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for info in manifest["splits"].values():
            if sha256_file(DIRECTORY / info["file"]) != info["sha256"]:
                raise ValueError("Frozen v2 data checksum changed")
        if sha256_file(ROOT / manifest["bank_file"]) != manifest["bank_sha256"]:
            raise ValueError("Frozen retrieval bank changed")
        return manifest
    old = {split: load_records(OLD_DIRECTORY, split) for split in ("calibration", "validation", "test")}
    excluded_ids = {row["source_id"] for rows in old.values() for row in rows}
    excluded_groups = {row["group_id"] for rows in old.values() for row in rows}
    source = {}
    for split, expected in SOURCE_HASHES.items():
        source_path = OLD_DIRECTORY / "source" / f"{split}-es.json.gz"
        if sha256_file(source_path) != expected:
            raise ValueError("Pinned source checksum changed")
        source[split] = [json.loads(line) for line in gzip.decompress(source_path.read_bytes()).decode().splitlines()]
    cleaned, deduplication = disjoint_sources(source)
    eligible = {key: [row for row in values if str(row["id"]) not in excluded_ids and text_group(row["text"]) not in excluded_groups]
                for key, values in cleaned.items()}
    splits, prepared = {}, {}
    splits["development"] = write_records(DIRECTORY / "development.jsonl", old["validation"])
    for index, (split, official, count) in enumerate((("calibration", "train", 300), ("validation", "validation", 150), ("test", "test", 450))):
        selected = stratified_sample(eligible[official], count, SEED+100+index)
        records = []
        for row in selected:
            labels = list(CATEGORIES)
            random.Random(f"v2:{SEED}:{official}:{row['id']}").shuffle(labels)
            records.append({"id": f"massive:es:{official}:{row['id']}", "source_id": str(row["id"]),
                            "source_split": official, "group_id": text_group(row["text"]), "state": row["text"],
                            "question": QUESTION, "options": [CATEGORIES[label] for label in labels],
                            "option_labels": labels, "target": row["label"], "modality": "text"})
        prepared[split] = records
        splits[split] = write_records(DIRECTORY / f"{split}.jsonl", records)
    calibration_ids = {row["source_id"] for row in prepared["calibration"]}
    training = [{"id": f"massive:es:train:{row['id']}", "source_id": str(row["id"]), "state": row["text"],
                 "group_id": text_group(row["text"]), "target": row["label"]}
                for row in eligible["train"] if str(row["id"]) not in calibration_ids]
    splits["retrieval_training"] = write_records(DIRECTORY / "retrieval-training.jsonl", training)
    print(f"Building retrieval bank from {len(training)} official train messages...", flush=True)
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", dtype=np.float32, max_features=70000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, strip_accents="unicode", dtype=np.float32, max_features=90000)),
    ], transformer_weights={"word": .65, "char": .35})
    matrix = normalize(vectorizer.fit_transform([row["state"] for row in training]))
    bank_path = ROOT / "classification" / "massive-es-v2-bank.joblib"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": VERSION, "vectorizer": vectorizer, "matrix": matrix, "records": training}, bank_path, compress=3)
    profile = {"name": NAME, "version": VERSION, "bank_file": str(bank_path.relative_to(ROOT)),
               "bank_sha256": sha256_file(bank_path), "neighbors": 6, "max_examples_per_class": 2,
               "training_count": len(training), "training_sha256": splits["retrieval_training"]["sha256"],
               "dataset": DATASET, "source_revision": json.loads((OLD_DIRECTORY / "manifest.json").read_text())["revision"]}
    save_json(PROFILE_PATH, profile)
    manifest = {"schema_version": 2, "dataset": DATASET, "language": "es", "splits": splits,
                "deduplication": deduplication, "old_v1_records_excluded_from_fresh_splits": len(excluded_ids),
                "bank_file": profile["bank_file"], "bank_sha256": profile["bank_sha256"],
                "protocol": {"seed": SEED+100, "development": "The old 150 validation records; only development may select changes.",
                             "selection_gate": "Development accuracy improves by at least 1 percentage point and macro F1 does not decrease.",
                             "calibration_gate": "Fresh validation NLL improves by more than 0.0001.",
                             "fresh_test": "450 previously unqueried official test records excluding all v1 records; paired v1/v2 predictions.",
                             "retrieval": "6 TF-IDF word/character neighbors, maximum 2 per category, official train only; no normalized exact query matches.",
                             "order": "v2 uses canonical internal answer labels and remaps probabilities to the caller's option order.",
                             "question": QUESTION, "options": CATEGORIES, "prompt_version": VERSION,
                             "test_used_for_selection": False, "order_audit_count": 36},
                "prior_manifest_sha256": sha256_file(OLD_DIRECTORY / "manifest.json")}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(path, manifest)
    analyze_old_errors()
    return manifest


def analyze_old_errors():
    old_run = Path(json.loads((OLD_DIRECTORY / "latest-run.json").read_text())["directory"])
    errors = json.loads((old_run / "test-errors.json").read_text(encoding="utf-8"))
    pairs = Counter((row["target"], row["prediction"]) for row in errors)
    summary = json.loads((old_run / "summary.json").read_text())
    save_json(DIRECTORY / "error-analysis.json", {
        "diagnostic_only": True, "source_run": old_run.name, "errors": len(errors),
        "confusion_pairs": [{"target": a, "prediction": b, "count": n} for (a,b), n in pairs.most_common()],
        "weakest_categories": sorted(summary["test"]["raw"]["per_class"], key=lambda row: row["f1"]),
        "changes": ["Define general narrowly rather than as a catch-all.", "Include contacts in email, coffee machines in iot, sports and trends in news.",
                    "Use training-only nearby examples to clarify corpus conventions and ambiguous Spanish messages.",
                    "Canonical internal labels remove option-order mapping variability.",
                    "Keep the old test as diagnostic material; use fresh paired test data."],
        "label_caution": "Some gold labels are ambiguous, e.g. a recipe request marked general. No relabeling or test deletion is performed."})


def scoped_records(split):
    return [row | {"task_profile": NAME} for row in load_records(DIRECTORY, split)]


def run():
    manifest = prepare()
    config = settings()
    with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
        client.get("/health").raise_for_status()
        env = environment(config)
        env["v2_code"] = {name: sha256_file(Path(__file__).with_name(name)) for name in ("spanish_task.py", "improve_spanish.py", "calibration.py", "service.py")}
        env["task_profile_sha256"] = sha256_file(PROFILE_PATH)
        run_id = digest({"manifest": manifest["manifest_fingerprint"], "environment": env})[:16]
        run_dir = DIRECTORY / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = run_dir / "RUNNING.lock"
        with lock.open("x"):
            pass
        try:
            save_json(run_dir / "environment.json", env)
            save_json(DIRECTORY / "latest-run.json", {"run_id": run_id, "directory": str(run_dir)})
            dev = collect(client, scoped_records("development"), run_dir / "development-predictions.jsonl", run_dir, "development", VERSION)
            old_run = Path(json.loads((OLD_DIRECTORY / "latest-run.json").read_text())["directory"])
            old_dev = [json.loads(line) for line in (old_run / "validation-predictions.jsonl").read_text().splitlines()]
            dev_p, dev_y = arrays(dev)
            base_dev_p, base_dev_y = arrays(old_dev)
            assert np.array_equal(dev_y, base_dev_y)
            dev_metrics = metrics(dev_p, dev_y, list(CATEGORIES))
            base_dev_metrics = metrics(base_dev_p, base_dev_y, list(CATEGORIES))
            approved = dev_metrics["accuracy"] >= base_dev_metrics["accuracy"]+.01 and dev_metrics["macro_f1"] >= base_dev_metrics["macro_f1"]
            development = {"approved": approved, "baseline": base_dev_metrics, "candidate": dev_metrics,
                           "gate": manifest["protocol"]["selection_gate"], "fresh_test_used": False}
            save_json(run_dir / "development-selection.json", development)
            print(f"Development accuracy {base_dev_metrics['accuracy']:.4f} -> {dev_metrics['accuracy']:.4f}; approved={approved}", flush=True)
            if not approved:
                print("Candidate failed development; fresh test remains unopened.", flush=True)
                return run_dir
            cal = collect(client, scoped_records("calibration"), run_dir / "calibration-predictions.jsonl", run_dir, "calibration", VERSION)
            val = collect(client, scoped_records("validation"), run_dir / "validation-predictions.jsonl", run_dir, "validation", VERSION)
            cal_p, cal_y = arrays(cal)
            val_p, val_y = arrays(val)
            temperature = fit_temperature(cal_p, cal_y)
            raw_val = metrics(val_p, val_y, list(CATEGORIES))
            candidate_val = metrics(scaled_array(val_p, temperature), val_y, list(CATEGORIES))
            calibrated_approved = candidate_val["negative_log_loss"] < raw_val["negative_log_loss"]-.0001
            selection = {"temperature": temperature, "selected_temperature": temperature if calibrated_approved else 1.0,
                         "validation_approved": calibrated_approved, "raw_validation": raw_val, "candidate_validation": candidate_val,
                         "test_used_for_selection": False}
            selection_path = run_dir / "selection.json"
            if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
                raise ValueError("The frozen pre-test calibration changed")
            save_json(selection_path, selection)
            print(f"Frozen calibration T={temperature:.5f}; approved={calibrated_approved}", flush=True)
            test = collect(client, scoped_records("test"), run_dir / "test-predictions.jsonl", run_dir, "test-v2", VERSION)
            baseline = collect(client, load_records(DIRECTORY, "test"), run_dir / "baseline-predictions.jsonl", run_dir, "test-v1", PROMPT_VERSION)
            audit_records = []
            for record in scoped_records("test")[:36]:
                changed = dict(record)
                changed["id"] += ":audit"
                changed["options"] = list(reversed(record["options"]))
                audit_records.append(changed)
            audit = collect(client, audit_records, run_dir / "order-audit-predictions.jsonl", run_dir, "order-audit", VERSION)
            result = summarize(run_dir, manifest, development, selection, test, baseline, audit)
            save_json(run_dir / "summary.json", result)
            publish_profile(run_dir, manifest, env, selection, test[0]["metadata"], result)
            save_json(run_dir / "progress.json", {"phase": "complete", "completed": 1536, "total": 1536})
        finally:
            lock.unlink(missing_ok=True)
    from .spanish_report import render
    render(run_dir)
    save_json(ROOT / "evaluation" / "current.json", {"version": 2, "run_id": run_id, "directory": str(run_dir.relative_to(ROOT))})
    print(json.dumps({"run_directory": str(run_dir), "baseline_accuracy": result["baseline"]["legacy_calibrated"]["accuracy"],
                      "v2_accuracy": result["test"]["calibrated"]["accuracy"], "temperature": selection["selected_temperature"]}), flush=True)
    return run_dir


def summarize(run_dir, manifest, development, selection, test, baseline, audit):
    raw, targets = arrays(test)
    base_p, base_y = arrays(baseline)
    assert np.array_equal(targets, base_y)
    scaled = scaled_array(raw, selection["selected_temperature"])
    old_profile = json.loads((PROFILES / "massive_scenario_es_v1.json").read_text())
    old_scaled = scaled_array(base_p, old_profile["temperature"])
    before, after = base_p.argmax(axis=1) == targets, raw.argmax(axis=1) == targets
    fixed, broken = int((~before & after).sum()), int((before & ~after).sum())
    rng = np.random.default_rng(SEED)
    changes = after.astype(float)-before.astype(float)
    differences = [float(changes[rng.integers(0, len(changes), len(changes))].mean()) for _ in range(5000)]
    audit_p, _ = arrays(audit)
    records = load_records(DIRECTORY, "test")
    cases = []
    for index, row in enumerate(records):
        cases.append({"id": row["id"], "text": row["state"], "target": row["target"],
                      "baseline_prediction": list(CATEGORIES)[int(base_p[index].argmax())],
                      "v2_prediction": list(CATEGORIES)[int(raw[index].argmax())],
                      "baseline_correct": bool(before[index]), "v2_correct": bool(after[index]),
                      "v2_raw_confidence": float(raw[index].max()), "v2_calibrated_confidence": float(scaled[index].max())})
    save_json(run_dir / "test-cases.json", cases)
    return {"schema_version": 2, "dataset": DATASET, "run_id": run_dir.name, "classes": list(CATEGORIES),
            "sample_counts": {key: info["count"] for key, info in manifest["splits"].items()},
            "task_profile": NAME, "profile_name": CALIBRATION_NAME if selection["validation_approved"] else None,
            "temperature": selection["temperature"], "selected_temperature": selection["selected_temperature"],
            "validation_approved": selection["validation_approved"], "test_used_for_selection": False,
            "test": {"raw": metrics(raw, targets, list(CATEGORIES)), "calibrated": metrics(scaled, targets, list(CATEGORIES)),
                     "nll_change": paired_nll_difference_ci(raw, scaled, targets)},
            "baseline": {"raw": metrics(base_p, targets, list(CATEGORIES)), "legacy_calibrated": metrics(old_scaled, targets, list(CATEGORIES)),
                         "temperature": old_profile["temperature"]},
            "paired_comparison": {"fixed_errors": fixed, "introduced_errors": broken, "accuracy_gain": float(changes.mean()),
                                  "paired_bootstrap_95ci": np.quantile(differences, [.025, .975]).tolist(),
                                  "mcnemar_exact_pvalue": float(binomtest(fixed, fixed+broken, .5).pvalue) if fixed+broken else 1.0},
            "order_audit": {"n": len(audit), "prediction_disagreements": int((raw[:len(audit)].argmax(axis=1) != audit_p.argmax(axis=1)).sum()),
                            "mean_total_variation": float((abs(raw[:len(audit)]-audit_p).sum(axis=1)/2).mean())},
            "latency_seconds": {"v2_median": float(np.median([row["metadata"]["elapsed_seconds"] for row in test])),
                                "baseline_median": float(np.median([row["metadata"]["elapsed_seconds"] for row in baseline]))},
            "artifacts_sha256": {path.name: sha256_file(path) for path in run_dir.glob("*-predictions.jsonl")}}


def publish_profile(run_dir, manifest, env, selection, metadata, summary):
    if not selection["validation_approved"]:
        return
    profile = {"schema_version": 1, "method": "temperature_scaling", "temperature": selection["selected_temperature"],
               "validation_approved": True, "dataset": DATASET+":es", "run_id": run_dir.name,
               "scope": {"question": QUESTION, "options": sorted(CATEGORIES.values()), "modality": "text", "prompt_version": VERSION,
                         "task_profile": NAME, "task_profile_fingerprint": metadata["task_profile_fingerprint"],
                         "runtime_fingerprint": metadata["runtime_fingerprint"], "model_sha256": env["files"]["model"],
                         "projector_sha256": env["files"]["projector"]},
               "provenance": {"manifest_fingerprint": manifest["manifest_fingerprint"],
                              "selection_sha256": sha256_file(run_dir / "selection.json"), "fresh_test_used_for_fitting": False,
                              "training_bank_sha256": manifest["bank_sha256"]}}
    save_json(run_dir / "calibration-profile.json", profile)
    save_json(PROFILES / f"{CALIBRATION_NAME}.json", profile)
    example = request_for(scoped_records("test")[0]) | {"calibration_profile": CALIBRATION_NAME}
    save_json(ROOT / "examples" / "decision-massive-v2-calibrated.json", example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare()
        print(json.dumps({"splits": {key: info["count"] for key, info in manifest["splits"].items()}, "bank_sha256": manifest["bank_sha256"]}, indent=2))
    elif args.command == "run":
        run()
    else:
        from .spanish_report import render
        render(Path(json.loads((DIRECTORY / "latest-run.json").read_text())["directory"]))


if __name__ == "__main__":
    main()
