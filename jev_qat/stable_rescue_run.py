"""Calibrate on retired public data, then evaluate one sealed new v9 cohort."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import httpx
import numpy as np
from scipy.stats import binomtest

from .calibration import PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, QUESTION, digest, load_records
from .eval_metrics import fit_temperature, metrics, scaled_array
from .evaluate import arrays, environment
from .focus_experiment import read_json
from .focus_run import paired_comparison
from .stable_rescue_adapter import StableRescueTask, NAME, VERSION, PROFILE
from .stable_rescue_data import DIRECTORY, prepare, lines
from .stable_rescue_evidence import evidence

CALIBRATION = "massive_scenario_es_v9"


def general(m):
    return next(r for r in m["per_class"] if r["label"] == "general")


def conditions(base, candidate, paired, stable, validated, alpha):
    fixed, broken = paired["fixed_errors"], paired["introduced_errors"]
    p = float(binomtest(fixed, fixed+broken, .5, alternative="greater").pvalue) if fixed+broken else 1.
    return {"eight_recovered_all_three_orders": stable,
        "three_domain_corrections_all_three_orders": stable,
        "calibration_validated": validated,
        "general_precision_not_lower": general(candidate)["precision"] >= general(base)["precision"],
        "general_f1_not_lower": general(candidate)["f1"] >= general(base)["f1"],
        "accuracy_higher": candidate["accuracy"] > base["accuracy"],
        "macro_f1_not_lower": candidate["macro_f1"] >= base["macro_f1"],
        "one_sided_exact_p_below_alpha": p < alpha,
        "test_nll_not_higher": candidate["negative_log_loss"] <= base["negative_log_loss"]}, p


def run():
    m = prepare()
    stability = read_json(DIRECTORY / "stability-final.json")
    if len(stability["results"]) != 33 or not stability["all_passed"] or not all(r["passed"] for r in stability["results"]):
        raise RuntimeError("All predeclared live stability checks must pass before opening the new test")
    cfg, fp = StableRescueTask()._load()
    env = environment(settings())
    code = ("stable_rescue_adapter.py", "stable_rescue_math.py", "stable_rescue_task.py", "stable_rescue_data.py",
            "stable_rescue_evidence.py", "stable_rescue_run.py", "stable_rescue_report.py", "general_precision_math.py")
    env["v9_code"] = {n: sha256_file(ROOT / "jev_qat" / n) for n in code}
    env["task_fingerprint"] = fp
    env["profile_sha256"] = sha256_file(PROFILE)
    env["stability_sha256"] = sha256_file(DIRECTORY / "stability-final.json")
    runtime_directory = Path(settings()["runtime"]).parent
    env["runtime_libraries"] = {n: sha256_file(runtime_directory / n) for n in (
        "llama-server-impl.dll", "llama.dll", "llama-common.dll", "ggml-vulkan.dll", "ggml-base.dll", "mtmd.dll")}
    run_id = digest({"manifest": m["manifest_fingerprint"], "environment": env})[:16]
    run_dir = DIRECTORY / "runs" / run_id
    run_dir.mkdir(exist_ok=True, parents=True)
    if (run_dir / "summary.json").exists():
        raise RuntimeError("Fixed new test already complete; do not run another trial")
    lock = run_dir / "RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(DIRECTORY / "latest-run.json", {"run_id": run_id, "directory": str(run_dir)})
        save_json(run_dir / "environment.json", env)
        save_json(run_dir / "task-profile.json", cfg)
        shutil.copy2(DIRECTORY / "stability-final.json", run_dir / "stability-final.json")
        snapshot = run_dir / "source"
        snapshot.mkdir(exist_ok=True)
        for n in set(code) | set(env["code"]):
            shutil.copy2(ROOT / "jev_qat" / n, snapshot / n)
        with httpx.Client(base_url=settings()["api_url"], timeout=settings()["request_timeout_seconds"], trust_env=False) as client:
            cal, _ = evidence(client, load_records(DIRECTORY, "calibration"), run_dir, "calibration")
            val, _ = evidence(client, load_records(DIRECTORY, "validation"), run_dir, "validation")
            cp, cy = arrays(cal); vp, vy = arrays(val)
            t = fit_temperature(cp, cy)
            vr, vc = metrics(vp, vy, list(CATEGORIES)), metrics(scaled_array(vp, t), vy, list(CATEGORIES))
            validated = vc["negative_log_loss"] < vr["negative_log_loss"] - .0001
            selected_t = t if validated else 1.
            selection = {"temperature": t, "selected_temperature": selected_t, "validation_approved": validated,
                "raw_validation": vr, "candidate_validation": vc, "test_used_for_selection": False}
            selected_path = run_dir / "selection.json"
            if selected_path.exists() and read_json(selected_path) != selection:
                raise ValueError("Pre-test temperature selection changed")
            save_json(selected_path, selection)
            if validated:
                profile = {"schema_version": 1, "method": "temperature_scaling", "temperature": t, "validation_approved": True,
                    "dataset": "Retired public human-label MTOP + Multi3NLU", "run_id": run_id,
                    "scope": {"question": QUESTION, "options": sorted(CATEGORIES.values()), "modality": "text",
                        "prompt_version": VERSION, "task_profile": NAME, "task_profile_fingerprint": fp,
                        "runtime_fingerprint": cal[0]["metadata"]["runtime_fingerprint"],
                        "model_sha256": env["files"]["model"], "projector_sha256": env["files"]["projector"]},
                    "provenance": {"manifest_fingerprint": m["manifest_fingerprint"], "test_used_for_fitting": False,
                        "selection_sha256": sha256_file(selected_path), "retired_data_only": True}}
                save_json(PROFILES / f"{CALIBRATION}.json", profile)
                save_json(run_dir / "calibration-profile.json", profile)
            freeze_path = run_dir / "pre-test-freeze.json"
            if not freeze_path.exists():
                save_json(freeze_path, {"frozen_at": datetime.now(timezone.utc).isoformat(),
                    "manifest_sha256": sha256_file(DIRECTORY / "manifest.json"), "environment_sha256": sha256_file(run_dir / "environment.json"),
                    "selection_sha256": sha256_file(selected_path), "test_predictions_existed": any(run_dir.glob("test-*.jsonl")),
                    "test_n": m["splits"]["test"]["count"], "alpha": m["protocol"]["alpha"]})
            freeze = read_json(freeze_path)
            if freeze["test_predictions_existed"] or freeze["selection_sha256"] != sha256_file(selected_path):
                raise ValueError("Invalid confirmatory freeze")
            print(json.dumps({"frozen_before_new_test": True, "temperature": selected_t, "validation_approved": validated}), flush=True)
            candidate, baseline = evidence(client, load_records(DIRECTORY, "test"), run_dir, "test")
        raw, y = arrays(candidate); bp, by = arrays(baseline)
        assert np.array_equal(y, by)
        before = scaled_array(bp, read_json(PROFILES / "massive_scenario_es_v2.json")["temperature"])
        after = scaled_array(raw, selected_t)
        bm, cm = metrics(before, y, list(CATEGORIES)), metrics(after, y, list(CATEGORIES))
        paired = paired_comparison(before, after, y)
        gates, p = conditions(bm, cm, paired, stability["all_passed"], validated, m["protocol"]["alpha"])
        activation = {"conditions": gates, "pvalue": p, "alpha": m["protocol"]["alpha"],
            "statistical_and_quality_gates_passed": all(gates.values()), "approved": False, "activated": False, "verification_required": True}
        save_json(run_dir / "activation.json", activation)
        rows = load_records(DIRECTORY, "test")
        per_source = {}
        for source in sorted({r["dataset"] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r["dataset"] == source]
            per_source[source] = {"baseline": metrics(before[ix], y[ix], list(CATEGORIES)), "candidate": metrics(after[ix], y[ix], list(CATEGORIES))}
        summary = {"schema_version": 9, "run_id": run_id, "sample_counts": {s: v["count"] for s, v in m["splits"].items()},
            "classes": list(CATEGORIES), "temperature": t, "selected_temperature": selected_t, "validation_approved": validated,
            "baseline": {"calibrated": bm, "temperature": read_json(PROFILES / "massive_scenario_es_v2.json")["temperature"]},
            "test": {"raw": metrics(raw, y, list(CATEGORIES)), "calibrated": cm}, "paired_comparison": paired,
            "per_source": per_source, "activation": activation, "previous_tests_in_significance": False,
            "artifacts_sha256": {p.name: sha256_file(p) for p in run_dir.glob("*.jsonl")}}
        save_json(run_dir / "summary.json", summary)
        save_json(run_dir / "test-cases.json", [{"id": r["id"], "text": r["state"], "target": r["target"], "dataset": r["dataset"],
            "v2_prediction": list(CATEGORIES)[int(before[i].argmax())], "v9_prediction": list(CATEGORIES)[int(after[i].argmax())],
            "v2_correct": bool(before[i].argmax() == y[i]), "v9_correct": bool(after[i].argmax() == y[i]),
            "v9_confidence": float(after[i].max())} for i, r in enumerate(rows)])
        save_json(run_dir / "progress.json", {"phase": "complete", "new_test_messages": len(rows)})
    finally:
        lock.unlink(missing_ok=True)
    from .stable_rescue_report import render
    render(run_dir)
    print(json.dumps({"run": str(run_dir), "v2_accuracy": bm["accuracy"], "v9_accuracy": cm["accuracy"], "p": p}), flush=True)
    return run_dir


def activate():
    run = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    s, a, v = read_json(run / "summary.json"), read_json(run / "activation.json"), read_json(run / "verification.json")
    if not v["passed"] or not v["live_passed"]:
        raise ValueError("Independent evidence and live checks required")
    m = prepare()
    stable = read_json(run / "stability-final.json")
    gates, p = conditions(s["baseline"]["calibrated"], s["test"]["calibrated"], s["paired_comparison"], stable["all_passed"], s["validation_approved"], m["protocol"]["alpha"])
    if gates != a["conditions"] or p != a["pvalue"]:
        raise ValueError("Decision changed after evaluation")
    a.update(approved=all(gates.values()), activated=all(gates.values()), verification_required=False)
    state = {"version": "v9", "run_id": run.name} if a["activated"] else {
        "version": "v2", "latest_candidate": "v9", "candidate_run_id": run.name, "candidate_status": "not_activated",
        "candidate_calibration_approved": s["validation_approved"], "reason": "Stable v9 failed at least one preregistered new-cohort superiority gate"}
    state["activation_decision"] = str((run / "activation.json").relative_to(ROOT))
    save_json(ROOT / "classification/current.json", state)
    save_json(run / "activation.json", a)
    s["activation"] = a
    save_json(run / "summary.json", s)
    from .stable_rescue_report import render
    render(run)
    save_json(ROOT / "evaluation/current.json", {"version": 9, "run_id": run.name, "directory": str(run.relative_to(ROOT))})
    print(json.dumps({"active": state["version"], "activated": a["activated"], "p": p}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "activate", "report"))
    command = parser.parse_args().command
    if command == "report":
        from .stable_rescue_report import render
        render(Path(read_json(DIRECTORY / "latest-run.json")["directory"]))
    else:
        {"prepare": prepare, "run": run, "activate": activate}[command]()
