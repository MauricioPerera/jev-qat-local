"""One fixed confirmatory comparison: python -m jev_qat.public_eval_run run|activate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import httpx
import numpy as np
from scipy.stats import binomtest

from .config import ROOT, save_json, settings
from .calibration import sha256_file
from .eval_data import CATEGORIES, digest, load_records
from .eval_metrics import metrics, scaled_array
from .evaluate import arrays, environment, request_for
from .focus_experiment import read_json
from .focus_run import paired_comparison
from .general_confirm_adapter import NAME
from .public_eval_data import DIRECTORY, ORIGINAL, frozen_profiles, lines, prepare
from .public_eval_evidence import evidence


def general(m):
    return next(r for r in m["per_class"] if r["label"] == "general")


def conditions(base, candidate, paired, regressions, manifest):
    fixed, broken = paired["fixed_errors"], paired["introduced_errors"]
    p = float(binomtest(fixed, fixed + broken, .5, alternative="greater").pvalue) if fixed + broken else 1.
    gates = {
        "eight_recoveries_pass_live": regressions["eight_recovered_preserved"],
        "three_domain_corrections_pass_live": regressions["three_domain_errors_corrected"],
        "original_human_labels_and_frozen_crosswalk": manifest["protocol"]["labels"].startswith("Original annotations by humans"),
        "general_support_at_least_50": general(candidate)["support"] >= 50,
        "general_f1_gain_at_least_3_points": general(candidate)["f1"] >= general(base)["f1"] + .03,
        "general_precision_not_lower": general(candidate)["precision"] >= general(base)["precision"],
        "accuracy_higher": candidate["accuracy"] > base["accuracy"],
        "macro_f1_not_lower": candidate["macro_f1"] >= base["macro_f1"],
        "one_sided_exact_p_less_than_0_025": p < manifest["protocol"]["alpha"],
        "test_nll_not_higher": candidate["negative_log_loss"] <= base["negative_log_loss"],
        "calibration_unchanged_and_previously_validated": all(read_json(ROOT / f"calibration/profiles/massive_scenario_es_{v}.json")["validation_approved"] for v in ("v2", "v8")),
        "frozen_profiles_unchanged": frozen_profiles() == manifest["frozen_profiles"],
    }
    return gates, p


def regressions(client, run_dir):
    selection = read_json(ORIGINAL / "development-source-scores.json")
    dev = {r["id"]: r for r in lines(ORIGINAL / "development.jsonl")}
    protected = selection["protected_ids"]
    # These three reference labels were fixed before the previous v8 test.
    corrections = ["massive:es:test:14683", "massive:es:test:12264", "massive:es:test:4361"]
    results = []
    for identifier in protected + corrections:
        row = dev[identifier]
        payload = request_for(row) | {"task_profile": NAME, "calibration_profile": "massive_scenario_es_v8"}
        response = client.post("/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        results.append({"id": identifier, "target": row["target"], "passed": result["prediction"] == CATEGORIES[row["target"]],
                        "request_sha256": digest(payload), "result": result})
        print(json.dumps({"protected_check": len(results), "total": 11, "passed": results[-1]["passed"]}), flush=True)
    value = {"eight_recovered_preserved": all(r["passed"] for r in results[:8]),
             "three_domain_errors_corrected": all(r["passed"] for r in results[8:]), "cases": results}
    save_json(run_dir / "regressions.json", value)
    return value


def run():
    manifest = prepare()
    env = environment(settings())
    names = ("public_eval_data.py", "public_eval_evidence.py", "public_eval_run.py", "public_eval_report.py")
    env["follow_up_code"] = {n: sha256_file(ROOT / "jev_qat" / n) for n in names}
    env["frozen_profiles"] = frozen_profiles()
    preflight = DIRECTORY / "preflight-regressions.json"
    if preflight.exists():
        env["preflight_regressions_sha256"] = sha256_file(preflight)
    run_id = digest({"manifest": manifest["manifest_fingerprint"], "environment": env})[:16]
    run_dir = DIRECTORY / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        raise RuntimeError("Fixed cohort already complete; verify/report it without another inference run")
    lock = run_dir / "RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(DIRECTORY / "latest-run.json", {"run_id": run_id, "directory": str(run_dir)})
        save_json(run_dir / "environment.json", env)
        freeze_path = run_dir / "pre-test-freeze.json"
        if not freeze_path.exists():
            save_json(freeze_path, {"frozen_at": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": sha256_file(DIRECTORY / "manifest.json"), "environment_sha256": sha256_file(run_dir / "environment.json"),
                "test_predictions_existed": any(run_dir.glob("test-*.jsonl")),
                "fixed_n": manifest["test"]["count"], "alpha": manifest["protocol"]["alpha"]})
        freeze = read_json(freeze_path)
        if freeze["test_predictions_existed"] or freeze["manifest_sha256"] != sha256_file(DIRECTORY / "manifest.json") or freeze["environment_sha256"] != sha256_file(run_dir / "environment.json"):
            raise ValueError("Preregistered test freeze is invalid or changed")
        snapshot = run_dir / "source"
        snapshot.mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(ROOT / "jev_qat" / name, snapshot / name)
        rows = load_records(DIRECTORY, "test")
        with httpx.Client(base_url=settings()["api_url"], timeout=settings()["request_timeout_seconds"], trust_env=False) as client:
            if (run_dir / "regressions.json").exists():
                checks = read_json(run_dir / "regressions.json")
            elif preflight.exists():
                checks = read_json(preflight)
                shutil.copy2(preflight, run_dir / "regressions.json")
            else:
                checks = regressions(client, run_dir)
            # A failed preservation gate forbids promotion, but must not suppress
            # the requested diagnostic evaluation or be retried until it passes.
            print(json.dumps({"preflight_eight_preserved": checks["eight_recovered_preserved"],
                "promotion_blocked_by_regression": not all(r["passed"] for r in checks["cases"])}), flush=True)
            candidate, baseline = evidence(client, rows, run_dir)
        raw, targets = arrays(candidate)
        bp, by = arrays(baseline)
        assert np.array_equal(targets, by)
        if frozen_profiles() != manifest["frozen_profiles"]:
            raise ValueError("Classifier/calibration changed during the fixed test")
        temps = manifest["frozen_profiles"]["temperatures"]
        base = scaled_array(bp, temps["v2"])
        after = scaled_array(raw, temps["v8"])
        bm, cm = metrics(base, targets, list(CATEGORIES)), metrics(after, targets, list(CATEGORIES))
        paired = paired_comparison(base, after, targets)
        gates, p = conditions(bm, cm, paired, checks, manifest)
        activation = {"statistical_and_quality_gates_passed": all(gates.values()), "conditions": gates,
            "pvalue": p, "alpha": manifest["protocol"]["alpha"], "approved": False, "activated": False,
            "verification_required": True, "reason": "Independent evidence verification required before any promotion"}
        save_json(run_dir / "activation.json", activation)
        source_metrics = {}
        for source in sorted({r["dataset"] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r["dataset"] == source]
            source_metrics[source] = {"baseline": metrics(base[ix], targets[ix], list(CATEGORIES)),
                "candidate": metrics(after[ix], targets[ix], list(CATEGORIES)),
                "paired": paired_comparison(base[ix], after[ix], targets[ix])}
        summary = {"schema_version": 1, "run_id": run_id, "experiment": manifest["experiment"],
            "dataset": "DIHANA human-reviewed closings + fresh original-label MTOP", "sample_counts": {"test": len(rows)},
            "classes": list(CATEGORIES), "frozen_temperatures": temps, "new_calibration_fitted": False,
            "baseline": {"calibrated": bm, "temperature": temps["v2"]},
            "test": {"raw": metrics(raw, targets, list(CATEGORIES)), "calibrated": cm},
            "paired_comparison": paired, "per_source": source_metrics, "activation": activation,
            "protected_eight": checks["eight_recovered_preserved"], "previous_test_in_primary_analysis": False,
            "artifacts_sha256": {p.name: sha256_file(p) for p in run_dir.glob("*.jsonl")}}
        save_json(run_dir / "summary.json", summary)
        save_json(run_dir / "test-cases.json", [{"id": r["id"], "text": r["state"], "target": r["target"], "dataset": r["dataset"],
            "v2_prediction": list(CATEGORIES)[int(base[i].argmax())], "v8_prediction": list(CATEGORIES)[int(after[i].argmax())],
            "v2_correct": bool(base[i].argmax() == targets[i]), "v8_correct": bool(after[i].argmax() == targets[i]),
            "v8_confidence": float(after[i].max())} for i, r in enumerate(rows)])
        save_json(run_dir / "progress.json", {"phase": "complete", "logical_messages": len(rows),
            "native_source_calls": sum(len(lines(run_dir / n)) for n in ("test-v2.jsonl", "test-focus.jsonl", "test-confirmation.jsonl"))})
    finally:
        lock.unlink(missing_ok=True)
    from .public_eval_report import render
    render(run_dir)
    print(json.dumps({"run": str(run_dir), "v2_accuracy": bm["accuracy"], "v8_accuracy": cm["accuracy"], "p": p, "gates_pass": all(gates.values())}), flush=True)
    return run_dir


def activate():
    run_dir = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    manifest = prepare()
    verification = read_json(run_dir / "verification.json")
    summary = read_json(run_dir / "summary.json")
    checks = read_json(run_dir / "regressions.json")
    gates, p = conditions(summary["baseline"]["calibrated"], summary["test"]["calibrated"], summary["paired_comparison"], checks, manifest)
    if not verification["passed"] or not verification["live_composition_passed"]:
        raise RuntimeError("Evidence and live composition must pass independent verification")
    activation = read_json(run_dir / "activation.json")
    if activation["conditions"] != gates or activation["pvalue"] != p:
        raise ValueError("Activation criteria changed after the fixed test")
    activation.update(approved=all(gates.values()), verification_required=False,
        activated=all(gates.values()), reason="All frozen gates and verification passed" if all(gates.values()) else "Frozen statistical/quality gate failed; keep v2")
    if activation["activated"]:
        state = {"version": "v8", "run_id": run_dir.name, "activation_decision": str((run_dir / "activation.json").relative_to(ROOT))}
    else:
        state = {"version": "v2", "latest_candidate": "v8", "candidate_run_id": run_dir.name,
            "candidate_status": "not_activated", "candidate_calibration_approved": True,
            "activation_decision": str((run_dir / "activation.json").relative_to(ROOT)),
            "reason": "Frozen v8 did not satisfy every confirmatory gate on the new public human-label cohort."}
    save_json(run_dir / "activation.json", activation)
    summary["activation"] = activation
    save_json(run_dir / "summary.json", summary)
    save_json(ROOT / "classification/current.json", state)
    from .public_eval_report import render
    render(run_dir)
    save_json(ROOT / "evaluation/current.json", {"version": 8, "run_id": run_dir.name, "directory": str(run_dir.relative_to(ROOT))})
    print(json.dumps({"active": state["version"], "activated": activation["activated"], "p": p}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "activate", "report"))
    command = parser.parse_args().command
    if command == "report":
        from .public_eval_report import render
        render(Path(read_json(DIRECTORY / "latest-run.json")["directory"]))
    else:
        {"prepare": prepare, "run": run, "activate": activate}[command]()
