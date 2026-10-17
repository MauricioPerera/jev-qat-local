"""Evaluate deterministic focus support on shared, independently collected Gemma evidence."""
from pathlib import Path
import json
import shutil

import httpx
import joblib
import numpy as np
from scipy.stats import binomtest

from .calibration import PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, QUESTION, SEED, digest, load_records
from .eval_metrics import fit_temperature, metrics, paired_nll_difference_ci, scaled_array
from .evaluate import arrays, collect, environment
from .focus_experiment import DIRECTORY, FOCUS, HEADS, development, lexical_scores, mix_array, prepare, read_json
from .focus_task import NAME, PROFILE_PATH, VERSION
from .spanish_task import VERSION as BASE_VERSION

CALIBRATION_NAME = "massive_scenario_es_v3"


def select_profile(selection, manifest):
    trial = selection["selected"]
    if trial is None:
        raise RuntimeError("No candidate passed development; the fresh test remains unopened")
    model = joblib.load(HEADS)["models"][trial["head_id"]]
    head_file = ROOT / "classification" / "massive-es-v3-selected.joblib"
    joblib.dump({"model": model, "source_bank_sha256": manifest["training_bank_sha256"]}, head_file, compress=3)
    first = json.loads((DIRECTORY / "development-gemma.jsonl").read_text(encoding="utf-8").splitlines()[0])
    profile = {"name": NAME, "version": VERSION, "head_file": str(head_file.relative_to(ROOT)), "head_sha256": sha256_file(head_file),
               "weight": trial["weight"], "C": trial["C"], "head_id": trial["head_id"], "focus": list(FOCUS),
               "gemma_component_temperature": selection["gemma_component_temperature"],
               "base_task_fingerprint": first["metadata"]["task_profile_fingerprint"],
               "training_bank_sha256": manifest["training_bank_sha256"], "training_count": manifest["training_count"],
               "development_selection_sha256": sha256_file(DIRECTORY / "development-selection.json")}
    save_json(PROFILE_PATH, profile)
    fingerprint = digest({"profile": sha256_file(PROFILE_PATH), "implementation": sha256_file(Path(__file__).with_name("focus_task.py")),
                          "base": profile["base_task_fingerprint"]})
    return profile, model, fingerprint


def derive(source_rows, records, bank, model, profile, fingerprint, path):
    raw, _ = arrays(source_rows)
    gemma_calibrated = scaled_array(raw, profile["gemma_component_temperature"])
    lexical = lexical_scores(model, records, bank)
    mixed = mix_array(gemma_calibrated, lexical, profile["weight"])
    rows = []
    for index, (record, source) in enumerate(zip(records, source_rows)):
        if source["id"] != record["id"]:
            raise ValueError("Gemma evidence does not align with the evaluation records")
        rows.append({"id": source["id"], "target": source["target"], "probabilities": mixed[index].tolist(),
                     "derived": True, "source_request_sha256": source["request_sha256"],
                     "lexical_probabilities": lexical[index].tolist(),
                     "metadata": source["metadata"] | {"method": "gemma_tfidf_gated_mixture", "task_profile": NAME,
                         "prompt_version": VERSION, "task_profile_fingerprint": fingerprint,
                         "calibrated": False, "calibration": None, "shared_gemma_component": True}})
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    return rows


def paired_comparison(before_p, after_p, targets):
    before = before_p.argmax(axis=1) == targets
    after = after_p.argmax(axis=1) == targets
    fixed, broken = int((~before & after).sum()), int((before & ~after).sum())
    rng = np.random.default_rng(SEED+300)
    gains, focus_deltas = [], []
    classes = len(CATEGORIES)
    focus_indices = [list(CATEGORIES).index(label) for label in FOCUS]
    p0, p1 = before_p.argmax(axis=1), after_p.argmax(axis=1)
    for _ in range(3000):
        selected = rng.integers(0, len(targets), len(targets))
        gains.append(float((after[selected].astype(float)-before[selected]).mean()))
        values = []
        for predictions in (p0, p1):
            matrix = np.bincount(targets[selected]*classes+predictions[selected], minlength=classes*classes).reshape(classes, classes)
            denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
            values.append(np.divide(2*np.diag(matrix), denominator, out=np.zeros(classes, dtype=float), where=denominator!=0)[focus_indices])
        focus_deltas.append(values[1]-values[0])
    intervals = np.quantile(focus_deltas, [.025, .975], axis=0)
    return {"fixed_errors": fixed, "introduced_errors": broken, "accuracy_gain": float(after.mean()-before.mean()),
            "paired_bootstrap_95ci": np.quantile(gains, [.025, .975]).tolist(),
            "focus_f1_change_95ci": {label: intervals[:, i].tolist() for i,label in enumerate(FOCUS)},
            "mcnemar_exact_pvalue": float(binomtest(fixed, fixed+broken, .5).pvalue) if fixed+broken else 1.0,
            "bootstrap_repetitions": 3000}


def run():
    manifest = prepare()
    selection_dev = read_json(DIRECTORY / "development-selection.json") if (DIRECTORY / "development-selection.json").exists() else development()
    profile, model, fingerprint = select_profile(selection_dev, manifest)
    config = settings()
    env = environment(config)
    env["v3_code"] = {name: sha256_file(Path(__file__).with_name(name)) for name in ("focus_task.py", "focus_experiment.py", "focus_run.py", "spanish_task.py", "calibration.py")}
    env["task_profile_sha256"] = sha256_file(PROFILE_PATH)
    env["heads_sha256"] = sha256_file(HEADS)
    run_id = digest({"manifest": manifest["manifest_fingerprint"], "environment": env})[:16]
    run_dir = DIRECTORY / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(run_dir / "environment.json", env)
        save_json(run_dir / "selected-task-profile.json", profile)
        shutil.copy2(DIRECTORY / "development-selection.json", run_dir / "development-selection.json")
        snapshot = run_dir / "source"
        snapshot.mkdir(exist_ok=True)
        for path in (ROOT / "jev_qat").glob("*.py"):
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(ROOT / "jev_qat" / "option_labels.json", snapshot / "option_labels.json")
        save_json(DIRECTORY / "latest-run.json", {"run_id": run_id, "directory": str(run_dir)})
        bank = joblib.load(ROOT / manifest["training_bank_file"])
        derived, gemma = {}, {}
        with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
            for split in ("calibration", "validation"):
                records = load_records(DIRECTORY, split)
                gemma[split] = collect(client, [row | {"task_profile": "massive_es_v2"} for row in records],
                                       run_dir / f"{split}-gemma.jsonl", run_dir, split, BASE_VERSION)
                derived[split] = derive(gemma[split], records, bank, model, profile, fingerprint, run_dir / f"{split}-predictions.jsonl")
            cal_p, cal_y = arrays(derived["calibration"])
            val_p, val_y = arrays(derived["validation"])
            temperature = fit_temperature(cal_p, cal_y)
            raw_validation = metrics(val_p, val_y, list(CATEGORIES))
            candidate_validation = metrics(scaled_array(val_p, temperature), val_y, list(CATEGORIES))
            approved = candidate_validation["negative_log_loss"] < raw_validation["negative_log_loss"]-.0001
            selection = {"temperature": temperature, "selected_temperature": temperature if approved else 1.0,
                         "validation_approved": approved, "raw_validation": raw_validation, "candidate_validation": candidate_validation,
                         "test_used_for_selection": False}
            if (run_dir / "selection.json").exists() and read_json(run_dir / "selection.json") != selection:
                raise ValueError("The frozen pre-test temperature changed")
            save_json(run_dir / "selection.json", selection)
            print(f"Frozen before fresh test: T={temperature:.6f}, approved={approved}", flush=True)
            records = load_records(DIRECTORY, "test")
            gemma["test"] = collect(client, [row | {"task_profile": "massive_es_v2"} for row in records],
                                    run_dir / "test-gemma.jsonl", run_dir, "fresh-test-shared-component", BASE_VERSION)
            derived["test"] = derive(gemma["test"], records, bank, model, profile, fingerprint, run_dir / "test-predictions.jsonl")
        new_raw, targets = arrays(derived["test"])
        baseline_raw, _ = arrays(gemma["test"])
        baseline = scaled_array(baseline_raw, profile["gemma_component_temperature"])
        calibrated = scaled_array(new_raw, selection["selected_temperature"])
        summary = {"schema_version": 3, "run_id": run_id, "dataset": manifest["dataset"], "classes": list(CATEGORIES),
                   "sample_counts": {name: item["count"] for name,item in manifest["splits"].items()}, "focus": list(FOCUS),
                   "task_profile": NAME, "profile_name": CALIBRATION_NAME if approved else None,
                   "temperature": temperature, "selected_temperature": selection["selected_temperature"], "validation_approved": approved,
                   "test_used_for_selection": False, "sampling": manifest["protocol"]["test"],
                   "selected_C": profile["C"], "selected_weight": profile["weight"],
                   "baseline": {"calibrated": metrics(baseline, targets, list(CATEGORIES)), "temperature": profile["gemma_component_temperature"]},
                   "test": {"raw": metrics(new_raw, targets, list(CATEGORIES)), "calibrated": metrics(calibrated, targets, list(CATEGORIES)),
                            "nll_change": paired_nll_difference_ci(new_raw, calibrated, targets)},
                   "paired_comparison": paired_comparison(baseline, calibrated, targets),
                   "artifacts_sha256": {path.name: sha256_file(path) for path in run_dir.glob("*.jsonl")}}
        save_json(run_dir / "summary.json", summary)
        cases = [{"id": row["id"], "text": row["state"], "target": row["target"],
                  "v2_prediction": list(CATEGORIES)[int(baseline[i].argmax())], "v3_prediction": list(CATEGORIES)[int(calibrated[i].argmax())],
                  "v2_correct": bool(baseline[i].argmax()==targets[i]), "v3_correct": bool(calibrated[i].argmax()==targets[i]),
                  "v3_confidence": float(calibrated[i].max())} for i,row in enumerate(records)]
        save_json(run_dir / "test-cases.json", cases)
        if approved:
            scope = {"question": QUESTION, "options": sorted(CATEGORIES.values()), "modality": "text", "prompt_version": VERSION,
                     "task_profile": NAME, "task_profile_fingerprint": fingerprint,
                     "runtime_fingerprint": gemma["test"][0]["metadata"]["runtime_fingerprint"],
                     "model_sha256": env["files"]["model"], "projector_sha256": env["files"]["projector"]}
            calibration = {"schema_version": 1, "method": "temperature_scaling", "temperature": selection["selected_temperature"],
                           "validation_approved": True, "dataset": manifest["dataset"]+":es", "run_id": run_id, "scope": scope,
                           "provenance": {"manifest_fingerprint": manifest["manifest_fingerprint"], "selection_sha256": sha256_file(run_dir / "selection.json"),
                                          "fresh_test_used_for_fitting": False, "head_sha256": profile["head_sha256"]}}
            save_json(PROFILES / f"{CALIBRATION_NAME}.json", calibration)
            save_json(run_dir / "calibration-profile.json", calibration)
            from .evaluate import request_for
            example = request_for(records[0]) | {"task_profile": NAME, "calibration_profile": CALIBRATION_NAME}
            save_json(ROOT / "examples" / "decision-massive-v3-calibrated.json", example)
        save_json(run_dir / "progress.json", {"phase": "complete", "completed": 1080, "total": 1080})
    finally:
        lock.unlink(missing_ok=True)
    from .focus_report import render
    render(run_dir)
    save_json(ROOT / "evaluation" / "current.json", {"version": 3, "run_id": run_id, "directory": str(run_dir.relative_to(ROOT))})
    print(json.dumps({"run_directory": str(run_dir), "v2_accuracy": summary["baseline"]["calibrated"]["accuracy"],
                      "v3_accuracy": summary["test"]["calibrated"]["accuracy"], "temperature": selection["selected_temperature"]}), flush=True)
    return run_dir


def report():
    from .focus_report import render
    render(Path(read_json(DIRECTORY / "latest-run.json")["directory"]))
