"""Evaluate expanded general examples; deployment requires beating frozen v2."""
import argparse
import json
from pathlib import Path
import shutil

import httpx
import numpy as np
from scipy.stats import binomtest

from .calibration import PROFILES,sha256_file
from .config import ROOT,save_json,settings
from .eval_data import CATEGORIES,QUESTION,digest,load_records
from .eval_metrics import fit_temperature,metrics,paired_nll_difference_ci,scaled_array
from .evaluate import arrays,collect,environment,request_for
from .focus_experiment import DIRECTORY as V3,read_json
from .focus_run import paired_comparison
from .selective_experiment import DIRECTORY as V4
from .spanish_task import VERSION as V2_VERSION
from .general_data import DIRECTORY,prepare
from .general_task import NAME,VERSION,PROFILE_PATH,GeneralTask

CALIBRATION_NAME="massive_scenario_es_v5"


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def general(metrics_value):
    return next(row for row in metrics_value["per_class"] if row["label"]=="general")


def run():
    manifest=prepare()
    config=settings()
    env=environment(config)
    env["v5_code"]={name:sha256_file(Path(__file__).with_name(name)) for name in
                    ("general_task.py","general_data.py","general_learning.py","spanish_task.py","calibration.py","focus_run.py")}
    env["task_profile_sha256"]=sha256_file(PROFILE_PATH)
    run_id=digest({"manifest":manifest["manifest_fingerprint"],"environment":env})[:16]
    run_dir=DIRECTORY/"runs"/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    lock=run_dir/"RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(DIRECTORY/"latest-run.json",{"run_id":run_id,"directory":str(run_dir)})
        save_json(run_dir/"environment.json",env)
        save_json(run_dir/"task-profile.json",read_json(PROFILE_PATH))
        snapshot=run_dir/"source"
        snapshot.mkdir(exist_ok=True)
        for path in (ROOT/"jev_qat").glob("*.py"):
            shutil.copy2(path,snapshot/path.name)
        shutil.copy2(ROOT/"jev_qat/option_labels.json",snapshot/"option_labels.json")
        baseline_temperature=read_json(PROFILES/"massive_scenario_es_v2.json")["temperature"]
        prior_v4=Path(read_json(V4/"latest-run.json")["directory"])
        dev_base=read_lines(V3/"development-gemma.jsonl")+read_lines(prior_v4/"validation-gemma.jsonl")
        (run_dir/"development-v2.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dev_base),encoding="utf-8")
        with httpx.Client(base_url=config["api_url"],timeout=config["request_timeout_seconds"],trust_env=False) as client:
            def candidate(split):
                return collect(client,[r|{"task_profile":NAME} for r in load_records(DIRECTORY,split)],
                               run_dir/f"{split}-predictions.jsonl",run_dir,split,VERSION)
            dev=candidate("development")
            dp,dy=arrays(dev)
            bp,by=arrays(dev_base)
            assert np.array_equal(dy,by)
            assert [r["id"] for r in dev]==[r["id"] for r in dev_base]
            dm,bm=metrics(dp,dy,list(CATEGORIES)),metrics(scaled_array(bp,baseline_temperature),dy,list(CATEGORIES))
            approved=(general(dm)["f1"]>=general(bm)["f1"]+.03 and general(dm)["precision"]>=general(bm)["precision"]-.03
                      and dm["accuracy"]>=bm["accuracy"] and dm["macro_f1"]>=bm["macro_f1"])
            development={"approved":approved,"baseline":bm,"candidate":dm,"test_used_for_selection":False}
            save_json(run_dir/"development-selection.json",development)
            print(json.dumps({"development_approved":approved,"general_f1":[general(bm)["f1"],general(dm)["f1"]],
                              "accuracy":[bm["accuracy"],dm["accuracy"]]}),flush=True)
            if not approved:
                save_json(run_dir/"progress.json",{"phase":"development_rejected","completed":450,"total":450,"test_opened":False})
                print("Development rejected: new calibration/validation/test remain unopened.",flush=True)
                return run_dir
            cal,val=candidate("calibration"),candidate("validation")
            cp,cy=arrays(cal)
            vp,vy=arrays(val)
            temperature=fit_temperature(cp,cy)
            vr,vc=metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,temperature),vy,list(CATEGORIES))
            calibrated_approved=vc["negative_log_loss"]<vr["negative_log_loss"]-.0001
            selection={"temperature":temperature,"selected_temperature":temperature if calibrated_approved else 1.,
                       "validation_approved":calibrated_approved,"raw_validation":vr,"candidate_validation":vc,"test_used_for_selection":False}
            if (run_dir/"selection.json").exists() and read_json(run_dir/"selection.json")!=selection:
                raise ValueError("Frozen pre-test temperature changed")
            save_json(run_dir/"selection.json",selection)
            print(f"Frozen before new test: T={temperature:.6f}, validation_approved={calibrated_approved}",flush=True)
            records=load_records(DIRECTORY,"test")
            baseline=collect(client,[r|{"task_profile":"massive_es_v2"} for r in records],run_dir/"test-v2.jsonl",run_dir,"fresh-test-v2",V2_VERSION)
            test=candidate("test")
        raw,targets=arrays(test)
        base_raw,by=arrays(baseline)
        assert np.array_equal(targets,by)
        base=scaled_array(base_raw,baseline_temperature)
        calibrated=scaled_array(raw,selection["selected_temperature"])
        base_metrics,new_metrics=metrics(base,targets,list(CATEGORIES)),metrics(calibrated,targets,list(CATEGORIES))
        paired=paired_comparison(base,calibrated,targets)
        fixed,broken=paired["fixed_errors"],paired["introduced_errors"]
        one_sided=float(binomtest(fixed,fixed+broken,.5,alternative="greater").pvalue) if fixed+broken else 1.
        conditions={"calibration_validated":calibrated_approved,
                    "general_f1_gain_at_least_3_points":general(new_metrics)["f1"]>=general(base_metrics)["f1"]+.03,
                    "general_precision_not_lower":general(new_metrics)["precision"]>=general(base_metrics)["precision"],
                    "overall_accuracy_higher":new_metrics["accuracy"]>base_metrics["accuracy"],
                    "macro_f1_not_lower":new_metrics["macro_f1"]>=base_metrics["macro_f1"],
                    "paired_one_sided_p_below_0_05":one_sided<.05,
                    "test_nll_not_higher":new_metrics["negative_log_loss"]<=base_metrics["negative_log_loss"]}
        activation={"approved":all(conditions.values()),"conditions":conditions,"mcnemar_one_sided_pvalue":one_sided,
                    "criterion_frozen_in_manifest":True,"activated":False}
        save_json(run_dir/"activation.json",activation)
        summary={"schema_version":5,"run_id":run_id,"dataset":manifest["dataset"],"classes":list(CATEGORIES),"focus":["general"],
                 "sample_counts":{s:i["count"] for s,i in manifest["splits"].items()},"task_profile":NAME,
                 "profile_name":CALIBRATION_NAME if calibrated_approved else None,"temperature":temperature,
                 "selected_temperature":selection["selected_temperature"],"validation_approved":calibrated_approved,"test_used_for_selection":False,
                 "baseline":{"calibrated":base_metrics,"temperature":baseline_temperature},
                 "test":{"raw":metrics(raw,targets,list(CATEGORIES)),"calibrated":new_metrics,"nll_change":paired_nll_difference_ci(raw,calibrated,targets)},
                 "paired_comparison":paired,"activation":activation,
                 "artifacts_sha256":{p.name:sha256_file(p) for p in run_dir.glob("*.jsonl")}}
        save_json(run_dir/"summary.json",summary)
        save_json(run_dir/"test-cases.json",[{"id":r["id"],"text":r["state"],"target":r["target"],
                  "v2_prediction":list(CATEGORIES)[int(base[i].argmax())],"v5_prediction":list(CATEGORIES)[int(calibrated[i].argmax())],
                  "v2_correct":bool(base[i].argmax()==targets[i]),"v5_correct":bool(calibrated[i].argmax()==targets[i]),
                  "v5_confidence":float(calibrated[i].max())} for i,r in enumerate(records)])
        _,_,task_fingerprint=GeneralTask()._load()
        if calibrated_approved:
            profile={"schema_version":1,"method":"temperature_scaling","temperature":selection["selected_temperature"],
                     "validation_approved":True,"dataset":manifest["dataset"]+":es","run_id":run_id,
                     "scope":{"question":QUESTION,"options":sorted(CATEGORIES.values()),"modality":"text","prompt_version":VERSION,
                              "task_profile":NAME,"task_profile_fingerprint":task_fingerprint,
                              "runtime_fingerprint":test[0]["metadata"]["runtime_fingerprint"],"model_sha256":env["files"]["model"],
                              "projector_sha256":env["files"]["projector"]},
                     "provenance":{"manifest_fingerprint":manifest["manifest_fingerprint"],"selection_sha256":sha256_file(run_dir/"selection.json"),"test_used_for_fitting":False}}
            save_json(PROFILES/f"{CALIBRATION_NAME}.json",profile)
            save_json(run_dir/"calibration-profile.json",profile)
        example=request_for(records[0])|{"task_profile":NAME}
        if calibrated_approved:
            example["calibration_profile"]=CALIBRATION_NAME
        save_json(ROOT/"examples/decision-massive-v5.json",example)
        save_json(run_dir/"progress.json",{"phase":"complete","completed":1850,"total":1850})
    finally:
        lock.unlink(missing_ok=True)
    report(run_dir)
    save_json(ROOT/"evaluation/current.json",{"version":5,"run_id":run_id,"directory":str(run_dir.relative_to(ROOT))})
    print(json.dumps({"run":str(run_dir),"v2_accuracy":base_metrics["accuracy"],"v5_accuracy":new_metrics["accuracy"],"activation_approved":activation["approved"]}),flush=True)
    return run_dir


def report(run_dir=None):
    from .general_report import render
    render(run_dir or Path(read_json(DIRECTORY/"latest-run.json")["directory"]))


def activate():
    run_dir=Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    activation=read_json(run_dir/"activation.json")
    if not activation["approved"]:
        print("Not activated: v5 failed one or more frozen criteria; v2 remains the default.")
        return
    verification=read_json(run_dir/"verification.json")
    integration=read_json(ROOT/"integration/jev-qat/integration-results.json")
    if not verification["passed"] or not verification["live"]["passed"] or not integration["passed"]:
        raise RuntimeError("Activation requires successful live and integration checks")
    save_json(ROOT/"classification/current.json",{"version":"v5","run_id":run_dir.name,"activation_decision":str((run_dir/"activation.json").relative_to(ROOT))})
    activation["activated"]=True
    save_json(run_dir/"activation.json",activation)
    summary=read_json(run_dir/"summary.json")
    summary["activation"]=activation
    save_json(run_dir/"summary.json",summary)
    report(run_dir)
    print("Activated v5: all frozen comparison, calibration and live checks passed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","report","activate"))
    command=parser.parse_args().command
    {"prepare":prepare,"run":run,"report":report,"activate":activate}[command]()
