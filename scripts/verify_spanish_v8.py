"""Verify original labels, frozen crosswalk, all score stages and live regressions."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import httpx
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, settings, save_json
from jev_qat.eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, digest
from jev_qat.eval_metrics import metrics, fit_temperature, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.general_confirm_adapter import ConfirmedTask, NAME, VERSION, PROFILE_PATH
from jev_qat.general_confirm_data import DIRECTORY, ACTS, category, family, lines
from jev_qat.general_confirm_run import CALIBRATION_NAME, conditions, general
from jev_qat.general_confirm_task import NAME as CONFIRM_NAME, VERSION as CONFIRM_VERSION
from jev_qat.general_precision_task import PrecisionTask


def equal(a,b):
    if isinstance(b,dict):
        assert a.keys() == b.keys()
        for k in b:
            equal(a[k],b[k])
    elif isinstance(b,list):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            equal(x,y)
    elif isinstance(b,float):
        assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
    else:
        assert a == b


def compose(proposed,confirmation,cfg):
    if confirmation is None:
        return np.array(proposed)
    winner = list(CATEGORIES)[int(np.argmax(confirmation))]
    floor = cfg["minimum_confirmation_probability"].get(winner,cfg["minimum_confirmation_probability"]["default"])
    if np.argmax(proposed) != list(CATEGORIES).index("general") or max(confirmation) < floor:
        return np.array(proposed)
    result = (1-cfg["weight"])*np.array(proposed)+cfg["weight"]*np.array(confirmation)
    return result/result.sum()


def proposal_compose(base,focused,cfg,support):
    result = np.array(base,dtype=float)
    gi = list(CATEGORIES).index("general");winner = list(CATEGORIES)[int(np.argmax(base))]
    if (focused is None or winner == "general" or np.argmax(focused) != gi or focused[gi] < cfg["minimum_probability"]
            or winner in cfg["protected_domains"] and support < cfg["minimum_general_similarity"]):
        return result
    new = max(base[gi],(1-cfg["weight"])*base[gi]+cfg["weight"]*focused[gi])
    result *= (1-new)/(1-base[gi]);result[gi] = new
    return result/result.sum()


def verify(live=False):
    m = read_json(DIRECTORY/"manifest.json");copy = dict(m);fp = copy.pop("manifest_fingerprint");assert digest(copy) == fp
    run = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    env = read_json(run/"environment.json");s = read_json(run/"summary.json");sel = read_json(run/"selection.json")
    assert run.name == digest({"manifest":fp,"environment":env})[:16]
    for group in ("code","v8_code"):
        for name,sha in env[group].items():
            assert sha256_file(ROOT/"jev_qat"/name) == sha == sha256_file(run/"source"/name)
    for field,sha in env["files"].items():
        assert sha256_file(settings()[field]) == sha
    assert sha256_file(PROFILE_PATH) == env["task_profile_sha256"]
    for name,sha in m["sources_sha256"].items():
        assert sha256_file(DIRECTORY/name) == sha
    assert sha256_file(DIRECTORY/"label-crosswalk.json") == m["crosswalk_sha256"]
    assert m["original_label_verification"] and not m["new_llm_labels_used"]
    original_m = {}
    for path in (DIRECTORY/"source/multi3nlu").rglob("fold*.json"):
        for row in read_json(path):
            assert row["uid"] not in original_m
            original_m[row["uid"]] = row
    z = zipfile.ZipFile(DIRECTORY/"source/mtop/official.zip")
    original_t = {split:{str(r[0]):r for r in (line.split("\t") for line in z.read(f"mtop/es/{origin}.txt").decode().splitlines())}
                  for split,origin in (("test","test"),("validation","eval"))}
    original_a = {}
    for split,sha in SOURCE_HASHES.items():
        path = ROOT/"evaluation/massive-es-v1/source"/f"{split}-es.json.gz";assert sha256_file(path) == sha
        original_a[split] = {str(r["id"]):r for r in (json.loads(x) for x in gzip.open(path,"rt",encoding="utf-8"))}
    old = set()
    for name,sha in m["prior_exposure_files"].items():
        path = ROOT/name;assert sha256_file(path) == sha
        old.update(family(r["state"]) for r in lines(path) if isinstance(r,dict) and r.get("state"))
    seen = set(old);records = {}
    for split,info in m["splits"].items():
        path = DIRECTORY/info["file"];assert sha256_file(path) == info["sha256"]
        records[split] = lines(path);assert len(records[split]) == info["count"]
        assert dict(Counter(r["target"] for r in records[split])) == info["class_counts"] == info["labels"]
        for r in records[split]:
            assert r["verified"] and r["family_id"] == digest(family(r["state"])) and family(r["state"]) not in seen
            seen.add(family(r["state"]))
            if r["dataset"] == "Multi3NLU++":
                original = original_m[r["source_id"]]
                assert original["text"] == r["state"] and original["intents"] == r["source_intents"] and original["lang"] == "es"
                assert set(original["intents"]) <= set(ACTS) and original["intents"] and not original.get("slots") and r["target"] == "general"
            elif r["dataset"] == "MTOP":
                original = original_t[r["source_split"]][r["source_id"]]
                assert original[3] == r["state"] and [original[1].removeprefix("IN:")] == r["source_intents"]
                assert category(r["source_intents"][0]) == r["target"]
            else:
                original = original_a[r["source_split"]][r["source_id"]]
                assert original["text"] == r["state"] and original["label"] == r["target"]
    assert sum(map(len,records.values())) == 673
    checks = ["Every original label and bounded crosswalk checked; MTOP verified against official archive, Multi3NLU pure source dialogue acts with no slots, MASSIVE unchanged; all 673 families fresh and disjoint"]
    cfg,task_fp = ConfirmedTask()._load()
    dev = read_json(DIRECTORY/"development-selection.json")
    assert sha256_file(DIRECTORY/"development-selection.json") == m["development_selection_sha256"] == cfg["development_selection_sha256"]
    for name,sha in dev["source_sha256"].items():
        assert sha256_file(ROOT/name) == sha
    assert sha256_file(DIRECTORY/"development-confirmation.jsonl") == dev["confirmation_sha256"]
    rr = lines(DIRECTORY/"development.jsonl");source = read_json(DIRECTORY/"development-source-scores.json")
    assert [r["id"] for r in rr] == source["ids"] and len(rr) == 1132 and len(dev["trials"]) == 60
    confirmations = {r["id"]:r for r in lines(DIRECTORY/"development-confirmation.jsonl")}
    ix = {r["id"]:i for i,r in enumerate(rr)};y = np.array([list(CATEGORIES).index(r["target"]) for r in rr])
    for trial in dev["trials"]:
        c = cfg | {"weight":trial["weight"],"minimum_confirmation_probability":trial["minimum_probability"]}
        p = np.array([compose(v,confirmations[r["id"]]["probabilities"] if r["id"] in confirmations else None,c) for r,v in zip(rr,source["v7_proposed"])])
        equal(metrics(p,y,list(CATEGORIES)),trial["metrics"])
        assert trial["all_eight_preserved"] == all(np.argmax(p[ix[k]]) == y[ix[k]] for k in dev["protected_ids"])
        assert trial["all_three_new_false_positives_removed"] == all(np.argmax(p[ix[k]]) != list(CATEGORIES).index("general") for k in dev["required_false_positive_ids"])
        assert trial["fully_correct_specialized_cases"] == sum(np.argmax(p[ix[k]]) == y[ix[k]] for k in dev["required_false_positive_ids"])
    eligible = [t for t in dev["trials"] if t["all_eight_preserved"] and t["all_three_new_false_positives_removed"]]
    selected = max(eligible,key=lambda t:(t["fully_correct_specialized_cases"],t["metrics"]["accuracy"],-t["weight"],t["minimum_probability"]["qa"],t["minimum_probability"]["default"]))
    equal(selected,dev["selected"])
    assert cfg["weight"] == selected["weight"] and cfg["minimum_confirmation_probability"] == selected["minimum_probability"]
    assert selected["fully_correct_specialized_cases"] == 3 and selected["all_eight_preserved"]
    checks.append("All 60 old-data candidates independently recomputed; all eight recoveries and three full specialized corrections preserved; no OASST provisional labels or new scores in selection")
    from jev_qat.general_precision_math import general_evidence, needs_review
    from jev_qat.general_task import GeneralTask
    from jev_qat.spanish_task import SpanishTask
    _,_,base_fp = SpanishTask()._load();_,_,focus_fp = GeneralTask()._load()
    proposal_cfg,proposal_fp = PrecisionTask()._load()
    output = {};base_output = {}
    for split,rr in records.items():
        base = lines(run/f"{split}-v2.jsonl");focused = {r["id"]:r for r in lines(run/f"{split}-focus.jsonl")}
        proposal = lines(run/f"{split}-proposal.jsonl");confirmed = {r["id"]:r for r in lines(run/f"{split}-confirmation.jsonl")}
        out = lines(run/f"{split}-predictions.jsonl");assert len(base) == len(proposal) == len(out) == len(rr)
        focus_ids,confirm_ids = set(),set()
        for r,b,p,o in zip(rr,base,proposal,out):
            assert r["id"] == b["id"] == p["id"] == o["id"] and r["target"] == b["target"] == p["target"] == o["target"]
            assert b["request_sha256"] == digest(request_for(r) | {"task_profile":"massive_es_v2"}) and b["metadata"]["task_profile_fingerprint"] == base_fp
            support = general_evidence(r["state"])["general_similarity"];winner = list(CATEGORIES)[int(np.argmax(b["probabilities"]))]
            need = needs_review(b["probabilities"],proposal_cfg["threshold"],proposal_cfg["component_temperature"])
            need &= winner not in proposal_cfg["protected_domains"] or support >= proposal_cfg["minimum_general_similarity"]
            if need:
                focus_ids.add(r["id"])
            f = focused.get(r["id"])
            if f:
                assert f["request_sha256"] == digest(request_for(r) | {"task_profile":"massive_es_v5"}) and f["metadata"]["task_profile_fingerprint"] == focus_fp
            expected = proposal_compose(b["probabilities"],f["probabilities"] if f else None,proposal_cfg,support)
            assert np.allclose(expected,p["probabilities"],rtol=0,atol=1e-12) and p["metadata"]["task_profile_fingerprint"] == proposal_fp
            c = confirmed.get(r["id"])
            if np.argmax(p["probabilities"]) == list(CATEGORIES).index("general"):
                confirm_ids.add(r["id"])
            if c:
                assert c["request_sha256"] == digest(request_for(r) | {"task_profile":CONFIRM_NAME})
                assert c["metadata"]["task_profile_fingerprint"] == cfg["confirmation_fingerprint"] and c["metadata"]["prompt_version"] == CONFIRM_VERSION
            assert np.allclose(compose(p["probabilities"],c["probabilities"] if c else None,cfg),o["probabilities"],rtol=0,atol=1e-12)
            assert o["metadata"]["task_profile_fingerprint"] == task_fp and o["derived"] and o["metadata"]["derived_timings"]
            assert o["metadata"]["llm_calls"] == 1+bool(f)+bool(c)
            for e in [b]+([f] if f else [])+([c] if c else []):
                assert not e["metadata"]["calibrated"] and len(e["probabilities"]) == 18 and np.isfinite(e["probabilities"]).all()
                assert min(e["probabilities"]) >= 0 and abs(sum(e["probabilities"])-1) < 1e-6
        assert set(focused) == focus_ids and set(confirmed) == confirm_ids
        output[split],base_output[split] = out,base
    checks.append("Every native request, full score vector, retrieval gate, confirmation decision and both probability compositions independently verified")
    cp,cy = arrays(output["calibration"]);vp,vy = arrays(output["validation"])
    temperature = fit_temperature(cp,cy);assert abs(temperature-sel["temperature"]) < 1e-9
    vr,vc = metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,temperature),vy,list(CATEGORIES))
    approved = vc["negative_log_loss"] < vr["negative_log_loss"]-.0001
    assert sel["validation_approved"] == approved and sel["raw_validation"] == vr and sel["candidate_validation"] == vc
    assert sel["selected_temperature"] == (temperature if approved else 1.)
    freeze = read_json(run/"pre-test-freeze.json")
    assert not freeze["test_predictions_existed"] and freeze["selection_sha256"] == sha256_file(run/"selection.json")
    assert freeze["manifest_sha256"] == sha256_file(DIRECTORY/"manifest.json")
    assert (run/"pre-test-freeze.json").stat().st_mtime <= (run/"test-v2.jsonl").stat().st_birthtime
    raw,y = arrays(output["test"]);br,_ = arrays(base_output["test"])
    before = scaled_array(br,s["baseline"]["temperature"]);after = scaled_array(raw,sel["selected_temperature"])
    bm,cm = metrics(before,y,list(CATEGORIES)),metrics(after,y,list(CATEGORIES))
    assert bm == s["baseline"]["calibrated"] and cm == s["test"]["calibrated"] and metrics(raw,y,list(CATEGORIES)) == s["test"]["raw"]
    paired = paired_comparison(before,after,y);assert paired == s["paired_comparison"]
    gates,pvalue = conditions(bm,cm,paired,approved,selected["all_eight_preserved"])
    gates.update(verified_original_labels=True,general_test_support_at_least_50=general(cm)["support"]>=50,three_development_domain_corrections=selected["fully_correct_specialized_cases"]==3)
    a = read_json(run/"activation.json");assert a["conditions"] == gates and a["approved"] == all(gates.values()) and a["mcnemar_one_sided_pvalue"] == pvalue
    for source,value in s["per_source"].items():
        indices = [i for i,r in enumerate(records["test"]) if r["dataset"] == source]
        assert value["baseline"] == metrics(before[indices],y[indices],list(CATEGORIES)) and value["candidate"] == metrics(after[indices],y[indices],list(CATEGORIES))
    if approved:
        p = read_json(PROFILES/f"{CALIBRATION_NAME}.json")
        assert p == read_json(run/"calibration-profile.json") and p["scope"]["task_profile_fingerprint"] == task_fp
    suite = ET.parse(DIRECTORY/"unit-tests.xml").getroot()
    assert all(int(x.get("failures",0)) == int(x.get("errors",0)) == 0 for x in suite.iter("testsuite"))
    checks.append("Fresh calibration refit, validation and pre-test freeze verified; every final metric, paired interval and promotion gate reproduced")
    live_result = {"passed":False,"executed":live}
    if live:
        devrows = {r["id"]:r for r in lines(DIRECTORY/"development.jsonl")}
        results = []
        with httpx.Client(base_url=settings()["api_url"],timeout=180,trust_env=False) as client:
            for identifier in dev["protected_ids"]+dev["required_false_positive_ids"]:
                r = devrows[identifier];payload = request_for(r) | {"task_profile":NAME}
                if approved:
                    payload["calibration_profile"] = CALIBRATION_NAME
                response = client.post("/predict",json=payload);response.raise_for_status();result = response.json()
                components = result["component_probabilities"]
                expected_raw = compose(components["v7_raw"],components["confirmation_raw"],cfg)
                expected = temperature_scale(expected_raw.tolist(),temperature) if approved else expected_raw
                assert np.allclose([result["probabilities"][v] for v in CATEGORIES.values()],expected,atol=1e-10,rtol=0)
                assert result["prediction"] == CATEGORIES[r["target"]] and result["metadata"]["task_profile_fingerprint"] == task_fp
                assert result["metadata"]["calibrated"] == approved and result["metadata"]["llm_calls"] == result["metadata"]["usage"]["completion_tokens"]
                results.append({"id":identifier,"expected":r["target"],"result":result})
                print(json.dumps({"live_regression":len(results),"total":11,"passed":True}),flush=True)
            payload = request_for(devrows[dev["required_false_positive_ids"][0]]) | {"task_profile":NAME}
            if approved:
                payload["calibration_profile"] = CALIBRATION_NAME
            payload["options"] = list(reversed(payload["options"]))
            response = client.post("/predict",json=payload);response.raise_for_status();reversed_result = response.json()
            assert reversed_result["prediction"] == CATEGORIES[devrows[dev["required_false_positive_ids"][0]]["target"]]
            assert payload["options"][reversed_result["prediction_index"]] == reversed_result["prediction"]
            payload.update(question="Otra pregunta")
            assert client.post("/predict",json=payload).status_code == 422
            payload.update(question=QUESTION,modality="image")
            assert client.post("/predict",json=payload).status_code == 422
            if approved:
                payload.update(modality="text",task_profile="massive_es_v2")
                assert client.post("/predict",json=payload).status_code == 422
        live_result = {"passed":True,"executed":True,"eight_recovered_preserved":True,"three_domain_errors_corrected":True,"cases":results,"reordered":reversed_result,"scope_rejections_passed":True}
        save_json(run/"live-api-results.json",live_result)
        checks.append("Live API preserves all eight recoveries and corrects all three domains, matches final calibration and option mapping, rejects wrong scopes")
    result = {"passed":True,"checks":checks,"live":live_result,"original_human_labels_verified":True,
              "new_human_adjudication_of_crosswalk":False,"new_llm_gold_labels_used":False}
    save_json(run/"verification.json",result)
    print(json.dumps({"passed":True,"checks":len(checks),"live":live_result["passed"]}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--live",action="store_true")
    verify(parser.parse_args().live)
