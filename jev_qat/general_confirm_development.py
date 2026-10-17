"""Retire old tests to development and guard every recovered correct message."""
from pathlib import Path
import json

import httpx
import numpy as np

from .calibration import sha256_file
from .config import ROOT, settings, save_json
from .eval_data import CATEGORIES
from .eval_metrics import metrics
from .evaluate import collect
from .focus_experiment import read_json, write_records
from .general_confirm_math import combine
from .general_confirm_task import NAME, VERSION, ConfirmationTask
from .general_precision_math import combine as v7_combine, general_evidence
from .general_precision_task import PrecisionTask

DIRECTORY = ROOT/"evaluation/general-confirm-v8"
PROFILE = ROOT/"classification/profiles/massive_es_v8.json"
OUTER_VERSION = "jev-qat-massive-es-general-confirmed-v8"


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def prepare():
    DIRECTORY.mkdir(exist_ok=True,parents=True)
    if (DIRECTORY/"development-selection.json").exists():
        return read_json(DIRECTORY/"development-selection.json")
    v6 = ROOT/"evaluation/massive-es-v6"
    v6run = v6/"runs/34d37f30edeb13d6"
    v7 = ROOT/"evaluation/general-rescue-v7"
    v7run = v7/"runs/e4fdd93688d16bd0"
    cfg,v7_fp = PrecisionTask()._load()
    records,proposed,baseline,guards = [],[],[],[]
    source_hashes = {}
    for split in ("development","test"):
        rr = lines(v6/f"{split}.jsonl");ee = lines(v6run/f"{split}-predictions.jsonl")
        source_hashes[str((v6run/f"{split}-predictions.jsonl").relative_to(ROOT))] = sha256_file(v6run/f"{split}-predictions.jsonl")
        for r,e in zip(rr,ee):
            c = e["component_probabilities"];b,f = c["base_raw"],c["focused_raw"]
            p = v7_combine(b,f,cfg,general_evidence(r["state"])["general_similarity"]) if f else b
            records.append(r);baseline.append(b);proposed.append(p)
            if np.argmax(e["probabilities"]) == list(CATEGORIES).index(r["target"]) and np.argmax(b) != np.argmax(e["probabilities"]):
                guards.append(r["id"])
    old7 = {r["id"]:r for r in lines(v7run/"test-predictions.jsonl")}
    for r in lines(v7/"test.jsonl"):
        if r["dataset"] != "MASSIVE":
            continue
        e = old7[r["id"]]
        records.append(r);baseline.append(e["component_probabilities"]["base_raw"]);proposed.append(e["probabilities"])
    source_hashes[str((v7run/"test-predictions.jsonl").relative_to(ROOT))] = sha256_file(v7run/"test-predictions.jsonl")
    assert len(guards) == 8 and len({r["id"] for r in records}) == len(records) == 1132
    write_records(DIRECTORY/"development.jsonl",records)
    save_json(DIRECTORY/"development-source-scores.json",{"ids":[r["id"] for r in records],"baseline_raw":baseline,"v7_proposed":proposed,"protected_ids":guards,"sources_sha256":source_hashes})
    reviews = [r | {"task_profile":NAME} for r,p in zip(records,proposed) if np.argmax(p) == list(CATEGORIES).index("general")]
    with httpx.Client(base_url=settings()["api_url"],timeout=180,trust_env=False) as client:
        confirmation = collect(client,reviews,DIRECTORY/"development-confirmation.jsonl",DIRECTORY,"old-data-confirmation",VERSION)
    by_id = {r["id"]:r for r in confirmation}
    targets = np.array([list(CATEGORIES).index(r["target"]) for r in records])
    required_fp = {"massive:es:test:14683","massive:es:test:12264","massive:es:test:4361"}
    trials = []
    for weight in (.5,.7,1.):
      for floor in (.5,.6,.7,.8,.9):
        for qa_floor in (.8,.9,.95,.99):
            minimum = {"default":floor,"qa":qa_floor}
            p = np.array([combine(p,by_id[r["id"]]["probabilities"],weight,minimum) if r["id"] in by_id else p for r,p in zip(records,proposed)])
            ix = {r["id"]:i for i,r in enumerate(records)}
            preserved = all(np.argmax(p[ix[k]]) == targets[ix[k]] for k in guards)
            false_positives_removed = all(np.argmax(p[ix[k]]) != list(CATEGORIES).index("general") for k in required_fp)
            fully_correct = sum(np.argmax(p[ix[k]]) == targets[ix[k]] for k in required_fp)
            trials.append({"weight":weight,"minimum_probability":minimum,"all_eight_preserved":preserved,
                           "all_three_new_false_positives_removed":false_positives_removed,"fully_correct_specialized_cases":int(fully_correct),
                           "metrics":metrics(p,targets,list(CATEGORIES))})
    eligible = [t for t in trials if t["all_eight_preserved"] and t["all_three_new_false_positives_removed"]]
    if not eligible:
        save_json(DIRECTORY/"development-rejected.json",{"trials":trials,"new_holdouts_used":False})
        raise RuntimeError("Confirmation failed the eight-correct / three-false-positive development gate")
    selected = max(eligible,key=lambda t:(t["fully_correct_specialized_cases"],t["metrics"]["accuracy"],-t["weight"],
                                        t["minimum_probability"]["qa"],t["minimum_probability"]["default"]))
    result = {"trials":trials,"selected":selected,"protected_ids":guards,"required_false_positive_ids":sorted(required_fp),
              "new_holdouts_used":False,"previous_tests_explicitly_retired":True,"provisional_oasst_labels_used":False,
              "source_sha256":source_hashes,"confirmation_sha256":sha256_file(DIRECTORY/"development-confirmation.jsonl")}
    save_json(DIRECTORY/"development-selection.json",result)
    example = records[0]
    confirm_fp = ConfirmationTask().prepare(NAME,state=example["state"],question=example["question"],options=example["options"],modality="text")["fingerprint"]
    save_json(PROFILE,{"name":"massive_es_v8","version":OUTER_VERSION,"v7_fingerprint":v7_fp,"confirmation_fingerprint":confirm_fp,
                       "weight":selected["weight"],"minimum_confirmation_probability":selected["minimum_probability"],
                       "development_selection_sha256":sha256_file(DIRECTORY/"development-selection.json")})
    print(json.dumps({"selected":selected,"guard_count":len(guards)}),flush=True)
    return result


if __name__ == "__main__":
    prepare()
