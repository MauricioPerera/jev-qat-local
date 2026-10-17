"""Native source components and explicitly derived v8 distributions."""
import json
import numpy as np
from .config import settings
from .eval_data import CATEGORIES,load_records
from .evaluate import collect
from .general_confirm_data import DIRECTORY
from .general_precision_task import PrecisionTask,NAME as PROPOSED_NAME,VERSION as PROPOSED_VERSION
from .general_precision_math import combine,general_evidence,needs_review
from .spanish_task import VERSION as BASE_VERSION
from .general_task import VERSION as FOCUSED_VERSION

def proposal_evidence(client, split, run_dir):
    config, fingerprint = PrecisionTask()._load()
    records = load_records(DIRECTORY, split)
    baseline = collect(client,[r | {"task_profile":"massive_es_v2"} for r in records],
                       run_dir/f"{split}-v2.jsonl",run_dir,f"{split}-v2",BASE_VERSION)
    support, reviews = {}, []
    for r,b in zip(records,baseline):
        if not needs_review(b["probabilities"],config["threshold"],config["component_temperature"]):
            continue
        s = general_evidence(r["state"])
        support[r["id"]] = s
        winner = list(CATEGORIES)[int(np.argmax(b["probabilities"]))]
        if winner not in config["protected_domains"] or s["general_similarity"] >= config["minimum_general_similarity"]:
            reviews.append(r)
    focused = collect(client,[r | {"task_profile":"massive_es_v5"} for r in reviews],
                      run_dir/f"{split}-focus.jsonl",run_dir,f"{split}-focused",FOCUSED_VERSION)
    by_id = {r["id"]:r for r in focused}
    output = []
    for r,b in zip(records,baseline):
        f = by_id.get(r["id"])
        values = combine(b["probabilities"],f["probabilities"],config,support[r["id"]]["general_similarity"]) if f else b["probabilities"]
        sources = [b] + ([f] if f else [])
        output.append({"id":r["id"],"target":r["target"],"probabilities":values,"derived":True,
                       "source_requests":{"base":b["request_sha256"],"focused":f["request_sha256"] if f else None},
                       "component_probabilities":{"base_raw":b["probabilities"],"focused_raw":f["probabilities"] if f else None,
                                                  "training_evidence":support.get(r["id"]),"reviewed":bool(f)},
                       "metadata":b["metadata"] | {"method":"gemma_general_precision_rescue","prompt_version":PROPOSED_VERSION,
                           "task_profile":PROPOSED_NAME,"task_profile_fingerprint":fingerprint,"calibrated":False,"llm_calls":len(sources),
                           "source_metadata":[x["metadata"] for x in sources],"derived_timings":True,
                           "elapsed_seconds":sum(x["metadata"]["elapsed_seconds"] for x in sources)}})
    (run_dir/f"{split}-proposal.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in output),encoding="utf-8")
    return output, baseline


def evidence(client,split,run_dir):
    from .general_confirm_adapter import ConfirmedTask,NAME,VERSION
    from .general_confirm_task import NAME as CONFIRM_NAME,VERSION as CONFIRM_VERSION
    from .general_confirm_math import combine as confirm_combine,GENERAL
    config,fingerprint=ConfirmedTask()._load()
    proposed,baseline=proposal_evidence(client,split,run_dir)
    records=load_records(DIRECTORY,split)
    reviews=[r|{"task_profile":CONFIRM_NAME} for r,p in zip(records,proposed) if np.argmax(p["probabilities"])==GENERAL]
    confirmations=collect(client,reviews,run_dir/f"{split}-confirmation.jsonl",run_dir,f"{split}-confirmation",CONFIRM_VERSION)
    by_id={r["id"]:r for r in confirmations}
    output=[]
    for r,p in zip(records,proposed):
        c=by_id.get(r["id"])
        values=confirm_combine(p["probabilities"],c["probabilities"],config["weight"],config["minimum_confirmation_probability"]) if c else p["probabilities"]
        output.append({"id":r["id"],"target":r["target"],"probabilities":values,"derived":True,
            "source_requests":{"confirmation":c["request_sha256"] if c else None},
            "component_probabilities":{"v7_raw":p["probabilities"],"confirmation_raw":c["probabilities"] if c else None,
                                       "source_components":p["component_probabilities"],"confirmed":bool(c)},
            "metadata":p["metadata"]|{"method":"gemma_general_contrastive_confirmation","prompt_version":VERSION,"task_profile":NAME,
                "task_profile_fingerprint":fingerprint,"llm_calls":p["metadata"]["llm_calls"]+bool(c),
                "elapsed_seconds":p["metadata"]["elapsed_seconds"]+(c["metadata"]["elapsed_seconds"] if c else 0),
                "source_metadata":[p["metadata"]]+([c["metadata"]] if c else [])}})
    (run_dir/f"{split}-predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in output),encoding="utf-8")
    return output,baseline
