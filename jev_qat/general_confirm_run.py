"""Paired fresh-cohort evaluation; calibration and promotion are separate decisions."""
import argparse
import json
from pathlib import Path
import shutil

import httpx
import numpy as np
from scipy.stats import binomtest

from .calibration import PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, QUESTION, digest, load_records
from .eval_metrics import fit_temperature, metrics, paired_nll_difference_ci, scaled_array
from .evaluate import arrays, collect, environment, request_for
from .focus_experiment import read_json
from .focus_run import paired_comparison
from .general_confirm_data import DIRECTORY, prepare, lines
from .general_precision_math import combine, general_evidence, needs_review
from .general_confirm_adapter import NAME, VERSION, PROFILE_PATH, ConfirmedTask
from .general_confirm_evidence import evidence
from .general_task import VERSION as FOCUSED_VERSION
from .spanish_task import VERSION as BASE_VERSION

CALIBRATION_NAME = "massive_scenario_es_v8"


def general(value):
    return next(r for r in value["per_class"] if r["label"] == "general")


def conditions(base, candidate, paired, calibration_validated, regressions_preserved):
    fixed,broken = paired["fixed_errors"],paired["introduced_errors"]
    pvalue = float(binomtest(fixed,fixed+broken,.5,alternative="greater").pvalue) if fixed+broken else 1.
    return {"new_correct_regressions_preserved":regressions_preserved,"calibration_validated":calibration_validated,
            "general_f1_gain_at_least_3_points":general(candidate)["f1"] >= general(base)["f1"]+.03,
            "general_precision_not_lower":general(candidate)["precision"] >= general(base)["precision"],
            "overall_accuracy_higher":candidate["accuracy"] > base["accuracy"],
            "macro_f1_not_lower":candidate["macro_f1"] >= base["macro_f1"],
            "paired_one_sided_p_below_0_05":pvalue < .05,
            "test_nll_not_higher":candidate["negative_log_loss"] <= base["negative_log_loss"]},pvalue


def run():
    manifest = prepare()
    env = environment(settings())
    env["v8_code"] = {name:sha256_file(Path(__file__).with_name(name)) for name in
                      ("general_confirm_run.py","general_confirm_data.py","general_confirm_development.py","general_confirm_adapter.py","general_confirm_evidence.py",
                       "general_confirm_task.py","general_confirm_math.py","general_precision_task.py","general_precision_math.py",
                       "general_rescue_math.py","general_task.py","spanish_task.py","calibration.py","focus_run.py")}

    env["task_profile_sha256"] = sha256_file(PROFILE_PATH)
    run_id = digest({"manifest":manifest["manifest_fingerprint"],"environment":env})[:16]
    run_dir = DIRECTORY/"runs"/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = run_dir/"RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(DIRECTORY/"latest-run.json",{"run_id":run_id,"directory":str(run_dir)})
        save_json(run_dir/"environment.json",env)
        save_json(run_dir/"task-profile.json",read_json(PROFILE_PATH))
        shutil.copy2(DIRECTORY/"development-selection.json",run_dir/"development-selection.json")
        snapshot = run_dir/"source"
        snapshot.mkdir(exist_ok=True)
        for source in (ROOT/"jev_qat").glob("*.py"):
            shutil.copy2(source,snapshot/source.name)
        shutil.copy2(ROOT/"jev_qat/option_labels.json",snapshot/"option_labels.json")
        baseline_temperature = read_json(PROFILES/"massive_scenario_es_v2.json")["temperature"]
        with httpx.Client(base_url=settings()["api_url"],timeout=settings()["request_timeout_seconds"],trust_env=False) as client:
            cal,_ = evidence(client,"calibration",run_dir)
            val,_ = evidence(client,"validation",run_dir)
            cp,cy = arrays(cal)
            vp,vy = arrays(val)
            temperature = fit_temperature(cp,cy)
            vr,vc = metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,temperature),vy,list(CATEGORIES))
            approved = vc["negative_log_loss"] < vr["negative_log_loss"]-.0001
            selection = {"temperature":temperature,"selected_temperature":temperature if approved else 1.,
                         "validation_approved":approved,"raw_validation":vr,"candidate_validation":vc,"test_used_for_selection":False}
            if (run_dir/"selection.json").exists() and read_json(run_dir/"selection.json") != selection:
                raise ValueError("Pre-test calibration changed")
            save_json(run_dir/"selection.json",selection)
            if not (run_dir/"pre-test-freeze.json").exists():
                save_json(run_dir/"pre-test-freeze.json",{"selection_sha256":sha256_file(run_dir/"selection.json"),
                          "manifest_sha256":sha256_file(DIRECTORY/"manifest.json"),"test_predictions_existed":(run_dir/"test-v2.jsonl").exists()})
            print(json.dumps({"frozen_before_test":True,"temperature":temperature,"calibration_validated":approved}),flush=True)
            test,baseline = evidence(client,"test",run_dir)
        raw,targets = arrays(test)
        base_raw,base_y = arrays(baseline)
        assert np.array_equal(targets,base_y)
        base = scaled_array(base_raw,baseline_temperature)
        candidate = scaled_array(raw,selection["selected_temperature"])
        bm,cm = metrics(base,targets,list(CATEGORIES)),metrics(candidate,targets,list(CATEGORIES))
        paired = paired_comparison(base,candidate,targets)
        development = read_json(DIRECTORY/"development-selection.json")
        selected = development["selected"]
        gates,pvalue = conditions(bm,cm,paired,approved,selected["all_eight_preserved"])
        gates.update(verified_original_labels=manifest["original_label_verification"], general_test_support_at_least_50=general(cm)["support"]>=50, three_development_domain_corrections=selected["fully_correct_specialized_cases"]==3)
        activation = {"approved":all(gates.values()),"conditions":gates,"mcnemar_one_sided_pvalue":pvalue,
                      "criterion_frozen_in_manifest":True,"activated":False}
        save_json(run_dir/"activation.json",activation)
        records = load_records(DIRECTORY,"test")
        source_metrics = {}
        for source in sorted({r["dataset"] for r in records}):
            ix = [i for i,r in enumerate(records) if r["dataset"] == source]
            source_metrics[source] = {"baseline":metrics(base[ix],targets[ix],list(CATEGORIES)),
                                      "candidate":metrics(candidate[ix],targets[ix],list(CATEGORIES)),
                                      "paired":paired_comparison(base[ix],candidate[ix],targets[ix])}
        summary = {"schema_version":8,"run_id":run_id,"dataset":manifest["dataset"],"classes":list(CATEGORIES),"focus":["general"],
                   "sample_counts":{s:i["count"] for s,i in manifest["splits"].items()},"task_profile":NAME,
                   "profile_name":CALIBRATION_NAME if approved else None,"temperature":temperature,
                   "selected_temperature":selection["selected_temperature"],"validation_approved":approved,"test_used_for_selection":False,
                   "baseline":{"calibrated":bm,"temperature":baseline_temperature},
                   "test":{"raw":metrics(raw,targets,list(CATEGORIES)),"calibrated":cm,"nll_change":paired_nll_difference_ci(raw,candidate,targets)},
                   "paired_comparison":paired,"per_source":source_metrics,"activation":activation,
                   "artifacts_sha256":{p.name:sha256_file(p) for p in run_dir.glob("*.jsonl")}}
        save_json(run_dir/"summary.json",summary)
        save_json(run_dir/"test-cases.json",[{"id":r["id"],"text":r["state"],"target":r["target"],"dataset":r["dataset"],
                  "annotation_source":r["annotation_source"],"v2_prediction":list(CATEGORIES)[int(base[i].argmax())],
                  "v8_prediction":list(CATEGORIES)[int(candidate[i].argmax())],"v2_correct":bool(base[i].argmax()==targets[i]),
                  "v8_correct":bool(candidate[i].argmax()==targets[i]),"v8_confidence":float(candidate[i].max())} for i,r in enumerate(records)])
        _,fingerprint = ConfirmedTask()._load()
        if approved:
            profile = {"schema_version":1,"method":"temperature_scaling","temperature":temperature,"validation_approved":True,
                       "dataset":manifest["dataset"],"run_id":run_id,
                       "scope":{"question":QUESTION,"options":sorted(CATEGORIES.values()),"modality":"text","prompt_version":VERSION,
                                "task_profile":NAME,"task_profile_fingerprint":fingerprint,"runtime_fingerprint":test[0]["metadata"]["runtime_fingerprint"],
                                "model_sha256":env["files"]["model"],"projector_sha256":env["files"]["projector"]},
                       "provenance":{"manifest_fingerprint":manifest["manifest_fingerprint"],"selection_sha256":sha256_file(run_dir/"selection.json"),
                                     "test_used_for_fitting":False,"annotation_caveat":"Original human intent labels with bounded deterministic ontology mapping; MTOP cross-checked against official archive"}}
            save_json(PROFILES/f"{CALIBRATION_NAME}.json",profile)
            save_json(run_dir/"calibration-profile.json",profile)
        example = request_for(records[0]) | {"task_profile":NAME}
        if approved:
            example["calibration_profile"] = CALIBRATION_NAME
        save_json(ROOT/"examples/decision-massive-v8.json",example)
        save_json(run_dir/"progress.json",{"phase":"complete","logical_messages":sum(summary["sample_counts"].values()),
                  "native_source_calls":sum(len(lines(p)) for p in run_dir.glob("*.jsonl") if p.name.endswith(("-v2.jsonl","-focus.jsonl","-confirmation.jsonl")))})
    finally:
        lock.unlink(missing_ok=True)
    from .general_confirm_report import render
    render(run_dir)
    save_json(ROOT/"evaluation/current.json",{"version":8,"run_id":run_id,"directory":str(run_dir.relative_to(ROOT))})
    print(json.dumps({"run":str(run_dir),"v2_accuracy":bm["accuracy"],"v8_accuracy":cm["accuracy"],"activation_approved":activation["approved"]}),flush=True)
    return run_dir


def activate():
    run_dir = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    activation = read_json(run_dir/"activation.json")
    if not activation["approved"]:
        save_json(ROOT/"classification/current.json",{"version":"v2","latest_candidate":"v8","candidate_run_id":run_dir.name,
                  "candidate_status":"not_activated","candidate_calibration_approved":read_json(run_dir/"selection.json")["validation_approved"],
                  "activation_decision":str((run_dir/"activation.json").relative_to(ROOT)),"reason":"V8 did not beat v2 under all predeclared gates on the fresh challenge cohort."})
        print("Not activated: v8 failed frozen comparison gates; calibrated v2 remains the default.")
        return
    verification = read_json(run_dir/"verification.json")
    if not verification["passed"] or not verification["live"]["passed"] or not read_json(run_dir/"integration-results.json")["passed"]:
        raise RuntimeError("Activation requires successful live and integration checks")
    save_json(ROOT/"classification/current.json",{"version":"v8","run_id":run_dir.name,"activation_decision":str((run_dir/"activation.json").relative_to(ROOT))})
    activation["activated"] = True
    save_json(run_dir/"activation.json",activation)
    summary = read_json(run_dir/"summary.json")
    summary["activation"] = activation
    save_json(run_dir/"summary.json",summary)
    from .general_confirm_report import render
    render(run_dir)
    print("Activated v8 after comparison, calibration and live verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","report","activate"))
    command = parser.parse_args().command
    if command == "report":
        from .general_confirm_report import render
        render(Path(read_json(DIRECTORY/"latest-run.json")["directory"]))
    else:
        {"prepare":prepare,"run":run,"activate":activate}[command]()
