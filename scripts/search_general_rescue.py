"""Dev-only refinement after expanded retrieval failed the global gate."""
import json
from pathlib import Path

import numpy as np

from jev_qat.calibration import PROFILES,sha256_file
from jev_qat.config import ROOT,save_json
from jev_qat.eval_data import CATEGORIES
from jev_qat.eval_metrics import metrics,scaled_array
from jev_qat.evaluate import arrays
from jev_qat.focus_experiment import read_json
from jev_qat.general_data import DIRECTORY as V5
from jev_qat.general_learning import general,read_lines
from jev_qat.general_rescue_math import combine,needs_review


def main():
    run=Path(read_json(V5/"latest-run.json")["directory"])
    base,targets=arrays(read_lines(run/"development-v2.jsonl"))
    focused,fy=arrays(read_lines(run/"development-predictions.jsonl"))
    assert np.array_equal(targets,fy)
    temperature=read_json(PROFILES/"massive_scenario_es_v2.json")["temperature"]
    baseline=metrics(scaled_array(base,temperature),targets,list(CATEGORIES))
    trials=[]
    for threshold in (.65,.8,.95,1.):
        review=[needs_review(p.tolist(),threshold,temperature) for p in base]
        for weight in (.3,.5,.7):
            for minimum in (0.,.95,.99):
                result=np.array([combine(a.tolist(),b.tolist(),weight,minimum) if active else a for a,b,active in zip(base,focused,review)])
                candidate=metrics(result,targets,list(CATEGORIES))
                approved=(general(candidate)["f1"]>=general(baseline)["f1"]+.03
                          and general(candidate)["precision"]>=general(baseline)["precision"]-.03
                          and candidate["accuracy"]>baseline["accuracy"] and candidate["macro_f1"]>=baseline["macro_f1"])
                trials.append({"threshold":threshold,"weight":weight,"minimum_probability":minimum,"review_count":sum(review),"approved":approved,"metrics":candidate})
    eligible=[r for r in trials if r["approved"]]
    selected=max(eligible,key=lambda r:(r["metrics"]["accuracy"],general(r["metrics"])["f1"],-r["review_count"],-r["weight"])) if eligible else None
    output={"baseline":baseline,"trials":trials,"selected":selected,"fresh_holdouts_used":False,
            "component_temperature":temperature,"source_v5_run":str(run),
            "source_evidence_sha256":{name:sha256_file(run/name) for name in ("development-v2.jsonl","development-predictions.jsonl")},
            "reason":"Full expanded retrieval improved general F1 but reduced overall accuracy by one example; restrict it to general rescue and preserve v2 elsewhere."}
    save_json(ROOT/"evaluation/massive-es-v6/development-selection.json",output)
    print(json.dumps({"baseline_accuracy":baseline["accuracy"],"selected":{k:v for k,v in selected.items() if k!="metrics"} if selected else None,
                      "accuracy":selected["metrics"]["accuracy"] if selected else None,
                      "general":general(selected["metrics"]) if selected else None}))


if __name__=="__main__":
    main()
