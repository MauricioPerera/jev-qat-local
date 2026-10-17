"""Collect each real source score once; explicitly derive the rescue distribution."""
import json
from pathlib import Path

from .eval_data import load_records
from .evaluate import collect
from .focus_experiment import read_json
from .general_learning import read_lines
from .general_rescue_data import DIRECTORY
from .general_rescue_math import combine,needs_review
from .general_rescue_task import NAME,VERSION,RescueTask
from .general_task import VERSION as FOCUSED_VERSION
from .spanish_task import VERSION as BASE_VERSION


def component_predictions(client,split,run_dir):
    config,fingerprint=RescueTask()._load()
    records=load_records(DIRECTORY,split)
    if split=="development":
        selection=read_json(DIRECTORY/"development-selection.json")
        source=Path(selection["source_v5_run"])
        baseline=read_lines(source/"development-v2.jsonl")
        all_focused={r["id"]:r for r in read_lines(source/"development-predictions.jsonl")}
    else:
        baseline=collect(client,[r|{"task_profile":"massive_es_v2"} for r in records],run_dir/f"{split}-v2.jsonl",run_dir,f"{split}-v2",BASE_VERSION)
    reviews=[r for r,e in zip(records,baseline) if needs_review(e["probabilities"],config["threshold"],config["component_temperature"])]
    if split=="development":
        focused=[all_focused[r["id"]] for r in reviews]
        for name,rows in (("development-v2",baseline),("development-focus",focused)):
            (run_dir/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    else:
        focused=collect(client,[r|{"task_profile":"massive_es_v5"} for r in reviews],run_dir/f"{split}-focus.jsonl",run_dir,f"{split}-expanded-general",FOCUSED_VERSION)
    by_id={r["id"]:r for r in focused}
    output=[]
    for record,base in zip(records,baseline):
        assert record["id"]==base["id"]
        review=by_id.get(base["id"])
        values=combine(base["probabilities"],review["probabilities"],config["weight"],config["minimum_probability"]) if review else base["probabilities"]
        output.append({"id":base["id"],"target":base["target"],"probabilities":values,"derived":True,
            "source_requests":{"base":base["request_sha256"],"focused":review["request_sha256"] if review else None},
            "component_probabilities":{"base_raw":base["probabilities"],"focused_raw":review["probabilities"] if review else None,"reviewed":bool(review)},
            "metadata":base["metadata"]|{"method":"gemma_general_rescue","prompt_version":VERSION,"task_profile":NAME,
                "task_profile_fingerprint":fingerprint,"calibrated":False,"llm_calls":2 if review else 1,
                "source_metadata":[base["metadata"]]+([review["metadata"]] if review else [])}})
    (run_dir/f"{split}-predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in output),encoding="utf-8")
    return output
