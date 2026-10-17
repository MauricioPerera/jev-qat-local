"""Fresh independent evaluation after rejecting the broad v3 mixture."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil

import httpx
import joblib
import numpy as np

from .calibration import PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, SEED, digest, disjoint_sources, load_records, stratified_sample, text_group
from .eval_metrics import fit_temperature, metrics, paired_nll_difference_ci, scaled_array
from .evaluate import arrays, collect, environment, request_for
from .focus_experiment import DIRECTORY as V3, V1, V2, FOCUS, HEADS, focused_sample, lexical_scores, read_json, write_records
from .focus_run import paired_comparison
from .selective_task import NAME, VERSION, PROFILE_PATH
from .spanish_task import VERSION as BASE_VERSION

DIRECTORY = ROOT / "evaluation/massive-es-v4"
CALIBRATION_NAME = "massive_scenario_es_v4"


def mix_array(gemma, lexical, weight, threshold):
    active = np.isin(lexical.argmax(1), [list(CATEGORIES).index(c) for c in FOCUS]) & (lexical.max(1) >= threshold)
    return np.where(active[:,None], (1-weight)*gemma+weight*lexical, gemma)


def prepare():
    path = DIRECTORY / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        for info in manifest["splits"].values():
            assert sha256_file(DIRECTORY/info["file"]) == info["sha256"]
        return manifest
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    selection = read_json(DIRECTORY/"development-selection.json")
    if not selection["selected"]:
        raise RuntimeError("No conservative candidate passed development")
    previous = [r for d in (V1,V2,V3) for s in ("calibration","validation","test") for r in load_records(d,s)]
    previous += load_records(V3,"development")
    ids,groups = {r["source_id"] for r in previous},{r["group_id"] for r in previous}
    sources = {}
    for split, expected in SOURCE_HASHES.items():
        source = V1/"source"/f"{split}-es.json.gz"
        assert sha256_file(source) == expected
        sources[split] = [json.loads(line) for line in gzip.decompress(source.read_bytes()).decode("utf-8").splitlines()]
    clean,_ = disjoint_sources(sources)
    available = {s:[r for r in rows if str(r["id"]) not in ids and text_group(r["text"]) not in groups] for s,rows in clean.items()}
    cal = stratified_sample(available["validation"],300,SEED+401)
    used = {r["id"] for r in cal}
    val = stratified_sample([r for r in available["validation"] if r["id"] not in used],150,SEED+402)
    test = focused_sample(available["test"],600,30,SEED+403)
    development = load_records(V3,"development")+load_records(V3,"validation")
    splits = {"development":write_records(DIRECTORY/"development.jsonl",development)}
    for split, official, rows in (("calibration","validation",cal),("validation","validation",val),("test","test",test)):
        records = [{"id":f"massive:es:{official}:{r['id']}","source_id":str(r["id"]),"source_split":official,
                    "group_id":text_group(r["text"]),"state":r["text"],"question":QUESTION,"options":list(CATEGORIES.values()),
                    "option_labels":list(CATEGORIES),"target":r["label"],"modality":"text"} for r in rows]
        splits[split] = write_records(DIRECTORY/f"{split}.jsonl",records)
    prior = read_json(V3/"manifest.json")
    manifest = {"schema_version":4,"dataset":prior["dataset"],"splits":splits,"focus":list(FOCUS),
                "training_bank_file":prior["training_bank_file"],"training_bank_sha256":prior["training_bank_sha256"],"training_count":prior["training_count"],
                "selection_sha256":sha256_file(DIRECTORY/"development-selection.json"),
                "prior_manifests_sha256":{d.name:sha256_file(d/"manifest.json") for d in (V1,V2,V3)},
                "protocol":{"test":"600 untouched official test messages, 30 per focus category and 510 proportional from the other 15.",
                            "development":"480 prior validation records (v3 development 300 + v3 validation 180), now explicitly development only.",
                            "selection":"48 candidates: three existing train-only heads, threshold .3/.5/.7/.9, mixture weight .4/.6/.8/1. Highest accuracy, then macro F1, then threshold. Mean focus F1 gain >=.01, no focus F1 decrease, overall accuracy and macro F1 improve, accuracy nondecreasing in each source development block.",
                            "calibration":"300 fresh calibration and 150 fresh validation records, proportional sampling from unused official validation. Approve only if validation NLL improves >.0001.",
                            "diagnosis":"Failed v3 test is diagnostic only and excluded from the new test; no training labels changed.","test_used_for_selection":False}}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(path,manifest)
    return manifest


def derive(source, records, bank, model, profile, fingerprint, path):
    raw,_ = arrays(source)
    lexical = lexical_scores(model,records,bank)
    mixed = mix_array(scaled_array(raw,profile["gemma_component_temperature"]),lexical,profile["weight"],profile["threshold"])
    rows = []
    for i,(record,evidence) in enumerate(zip(records,source)):
        assert record["id"] == evidence["id"]
        rows.append({"id":record["id"],"target":record["target"],"probabilities":mixed[i].tolist(),"lexical_probabilities":lexical[i].tolist(),
                     "derived":True,"source_request_sha256":evidence["request_sha256"],"metadata":evidence["metadata"]|{
                         "method":"gemma_tfidf_gated_mixture","prompt_version":VERSION,"task_profile":NAME,
                         "task_profile_fingerprint":fingerprint,"calibrated":False,"shared_gemma_component":True}})
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    return rows


def run():
    manifest = prepare()
    selection_dev = read_json(DIRECTORY/"development-selection.json")
    chosen = selection_dev["selected"]
    model = joblib.load(HEADS)["models"][chosen["head_id"]]
    head = ROOT/"classification/massive-es-v4-selected.joblib"
    joblib.dump({"model":model,"source_bank_sha256":manifest["training_bank_sha256"]},head,compress=3)
    base = read_json(ROOT/"classification/profiles/massive_es_v3.json")
    profile = {"name":NAME,"version":VERSION,"head_file":str(head.relative_to(ROOT)),"head_sha256":sha256_file(head),
               "head_id":chosen["head_id"],"C":float(model.C),"weight":chosen["weight"],"threshold":chosen["threshold"],
               "gemma_component_temperature":base["gemma_component_temperature"],"base_task_fingerprint":base["base_task_fingerprint"],
               "training_bank_sha256":manifest["training_bank_sha256"],"training_count":manifest["training_count"]}
    save_json(PROFILE_PATH,profile)
    fingerprint = digest({"profile":sha256_file(PROFILE_PATH),"implementation":sha256_file(Path(__file__).with_name("selective_task.py")),"base":base["base_task_fingerprint"]})
    config = settings()
    env = environment(config)
    env["v4_code"] = {name:sha256_file(Path(__file__).with_name(name)) for name in ("selective_task.py","selective_experiment.py","spanish_task.py","calibration.py","focus_experiment.py","focus_run.py")}
    env["task_profile_sha256"] = sha256_file(PROFILE_PATH)
    env["selection_search_sha256"] = sha256_file(ROOT/"scripts/search_selective_spanish.py")
    run_id = digest({"manifest":manifest["manifest_fingerprint"],"environment":env})[:16]
    run_dir = DIRECTORY/"runs"/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = run_dir/"RUNNING.lock"
    with lock.open("x"):
        pass
    try:
        save_json(DIRECTORY/"latest-run.json",{"run_id":run_id,"directory":str(run_dir)})
        save_json(run_dir/"environment.json",env)
        save_json(run_dir/"selected-task-profile.json",profile)
        shutil.copy2(DIRECTORY/"development-selection.json",run_dir/"development-selection.json")
        snapshot = run_dir/"source"
        snapshot.mkdir(exist_ok=True)
        for path in (ROOT/"jev_qat").glob("*.py"):
            shutil.copy2(path,snapshot/path.name)
        shutil.copy2(ROOT/"jev_qat/option_labels.json",snapshot/"option_labels.json")
        shutil.copy2(ROOT/"scripts/search_selective_spanish.py",snapshot/"search_selective_spanish.py")
        bank = joblib.load(ROOT/manifest["training_bank_file"])
        results,gemma = {},{}
        with httpx.Client(base_url=config["api_url"],timeout=config["request_timeout_seconds"],trust_env=False) as client:
            for split in ("calibration","validation"):
                records = load_records(DIRECTORY,split)
                gemma[split] = collect(client,[r|{"task_profile":"massive_es_v2"} for r in records],run_dir/f"{split}-gemma.jsonl",run_dir,split,BASE_VERSION)
                results[split] = derive(gemma[split],records,bank,model,profile,fingerprint,run_dir/f"{split}-predictions.jsonl")
            p,y = arrays(results["calibration"])
            vp,vy = arrays(results["validation"])
            temperature = fit_temperature(p,y)
            raw_val,cal_val = metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,temperature),vy,list(CATEGORIES))
            approved = cal_val["negative_log_loss"] < raw_val["negative_log_loss"]-.0001
            selection = {"temperature":temperature,"selected_temperature":temperature if approved else 1.,"validation_approved":approved,
                         "raw_validation":raw_val,"candidate_validation":cal_val,"test_used_for_selection":False}
            if (run_dir/"selection.json").exists() and read_json(run_dir/"selection.json") != selection:
                raise ValueError("Frozen pre-test selection changed")
            save_json(run_dir/"selection.json",selection)
            print(f"Frozen before independent test: T={temperature:.6f}, approved={approved}",flush=True)
            records = load_records(DIRECTORY,"test")
            gemma["test"] = collect(client,[r|{"task_profile":"massive_es_v2"} for r in records],run_dir/"test-gemma.jsonl",run_dir,"independent-test",BASE_VERSION)
            results["test"] = derive(gemma["test"],records,bank,model,profile,fingerprint,run_dir/"test-predictions.jsonl")
        raw,targets = arrays(results["test"])
        base_raw,_ = arrays(gemma["test"])
        baseline = scaled_array(base_raw,profile["gemma_component_temperature"])
        calibrated = scaled_array(raw,selection["selected_temperature"])
        summary = {"schema_version":4,"run_id":run_id,"dataset":manifest["dataset"],"classes":list(CATEGORIES),"focus":list(FOCUS),
                   "sample_counts":{s:i["count"] for s,i in manifest["splits"].items()},"task_profile":NAME,
                   "profile_name":CALIBRATION_NAME if approved else None,"temperature":temperature,"selected_temperature":selection["selected_temperature"],
                   "validation_approved":approved,"test_used_for_selection":False,"sampling":manifest["protocol"]["test"],
                   "selected_C":profile["C"],"selected_weight":profile["weight"],"selected_threshold":profile["threshold"],
                   "baseline":{"calibrated":metrics(baseline,targets,list(CATEGORIES)),"temperature":profile["gemma_component_temperature"]},
                   "test":{"raw":metrics(raw,targets,list(CATEGORIES)),"calibrated":metrics(calibrated,targets,list(CATEGORIES)),
                           "nll_change":paired_nll_difference_ci(raw,calibrated,targets)},"paired_comparison":paired_comparison(baseline,calibrated,targets),
                   "artifacts_sha256":{p.name:sha256_file(p) for p in run_dir.glob("*.jsonl")}}
        save_json(run_dir/"summary.json",summary)
        save_json(run_dir/"test-cases.json",[{"id":r["id"],"text":r["state"],"target":r["target"],"v2_prediction":list(CATEGORIES)[int(baseline[i].argmax())],
                  "v3_prediction":list(CATEGORIES)[int(calibrated[i].argmax())],"v2_correct":bool(baseline[i].argmax()==targets[i]),
                  "v3_correct":bool(calibrated[i].argmax()==targets[i]),"v3_confidence":float(calibrated[i].max())} for i,r in enumerate(records)])
        if approved:
            calibration = {"schema_version":1,"method":"temperature_scaling","temperature":selection["selected_temperature"],"validation_approved":True,
                "dataset":manifest["dataset"]+":es","run_id":run_id,"scope":{"question":QUESTION,"options":sorted(CATEGORIES.values()),"modality":"text",
                "prompt_version":VERSION,"task_profile":NAME,"task_profile_fingerprint":fingerprint,"runtime_fingerprint":gemma["test"][0]["metadata"]["runtime_fingerprint"],
                "model_sha256":env["files"]["model"],"projector_sha256":env["files"]["projector"]},
                "provenance":{"manifest_fingerprint":manifest["manifest_fingerprint"],"selection_sha256":sha256_file(run_dir/"selection.json"),"fresh_test_used_for_fitting":False}}
            save_json(PROFILES/f"{CALIBRATION_NAME}.json",calibration)
            save_json(run_dir/"calibration-profile.json",calibration)
            save_json(ROOT/"examples/decision-massive-v4-calibrated.json",request_for(records[0])|{"task_profile":NAME,"calibration_profile":CALIBRATION_NAME})
        save_json(run_dir/"progress.json",{"phase":"complete","completed":1050,"total":1050})
    finally:
        lock.unlink(missing_ok=True)
    report(run_dir)
    save_json(ROOT/"evaluation/current.json",{"version":4,"run_id":run_id,"directory":str(run_dir.relative_to(ROOT))})
    print(json.dumps({"run_directory":str(run_dir),"baseline_accuracy":summary["baseline"]["calibrated"]["accuracy"],"candidate_accuracy":summary["test"]["calibrated"]["accuracy"],"temperature":selection["selected_temperature"]}),flush=True)


def report(run_dir=None):
    from .focus_report import render
    render(run_dir or Path(read_json(DIRECTORY/"latest-run.json")["directory"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","report"))
    command = parser.parse_args().command
    {"prepare":prepare,"run":run,"report":report}[command]()
