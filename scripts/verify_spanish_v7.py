"""Audit source provenance, fresh groups, every score composition and live scope."""
import argparse
from collections import Counter
import gzip
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import httpx
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, digest, text_group
from jev_qat.eval_metrics import fit_temperature, metrics, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.general_precision_data import DIRECTORY, PREVIOUS, lines
from jev_qat.general_precision_math import general_evidence, needs_review
from jev_qat.general_precision_run import CALIBRATION_NAME, conditions
from jev_qat.general_precision_task import NAME, VERSION, PROFILE_PATH, PrecisionTask
from jev_qat.general_task import GeneralTask, VERSION as FOCUSED_VERSION
from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION


def compose(base,focused,config,support):
    """Independent expression of the mathematical protocol, not the runtime call."""
    result = np.array(base,dtype=float)
    gi = list(CATEGORIES).index("general")
    winner = list(CATEGORIES)[int(np.argmax(base))]
    if (focused is None or winner == "general" or np.argmax(focused) != gi or focused[gi] < config["minimum_probability"]
            or winner in config["protected_domains"] and support < config["minimum_general_similarity"]):
        return result
    new = max(base[gi],(1-config["weight"])*base[gi]+config["weight"]*focused[gi])
    result *= (1-new)/(1-base[gi]);result[gi] = new
    return result/result.sum()


def assert_equivalent(actual,expected):
    if isinstance(expected,dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equivalent(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            assert_equivalent(a,b)
    elif isinstance(expected,float):
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(actual,expected)
    else:
        assert actual == expected


def verify(live=False):
    cached_evidence = lru_cache(maxsize=None)(general_evidence)
    manifest = read_json(DIRECTORY/"manifest.json")
    copy = dict(manifest); fingerprint = copy.pop("manifest_fingerprint")
    assert digest(copy) == fingerprint
    run = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    env = read_json(run/"environment.json")
    summary = read_json(run/"summary.json")
    selection = read_json(run/"selection.json")
    assert run.name == digest({"manifest":fingerprint,"environment":env})[:16]
    checks = []
    for group in ("code","v7_code"):
        for name,sha in env[group].items():
            assert sha256_file(ROOT/"jev_qat"/name) == sha
            assert sha256_file(run/"source"/name) == sha
    for name,sha in env["files"].items():
        assert sha256_file(settings()[name]) == sha
    assert sha256_file(PROFILE_PATH) == env["task_profile_sha256"]
    assert sha256_file(DIRECTORY/"source/oasst-prompts.jsonl.gz") == manifest["sources_sha256"]["oasst-prompts"]
    assert sha256_file(DIRECTORY/"source/annotation-input.json") == manifest["sources_sha256"]["annotation-input"]
    assert sha256_file(DIRECTORY/"annotations.json") == manifest["sources_sha256"]["annotations"]
    annotations = {r["source_id"]:r for r in read_json(DIRECTORY/"annotations.json")}
    original = [json.loads(x) for x in gzip.open(DIRECTORY/"source/oasst-prompts.jsonl.gz","rt",encoding="utf-8")]
    eligible = [r for r in original if r["lang"] == "es" and r["parent_id"] is None and not r.get("synthetic")
                and not r.get("deleted") and 12 <= len(r["text"]) <= 220]
    random.Random(2026101617).shuffle(eligible)
    assert read_json(DIRECTORY/"source/annotation-input.json") == eligible[:180]
    original = {r["message_id"]:r for r in original}
    gold = {}
    for split,sha in SOURCE_HASHES.items():
        path = ROOT/"evaluation/massive-es-v1/source"/f"{split}-es.json.gz"
        assert sha256_file(path) == sha
        gold[split] = {str(r["id"]):r for r in (json.loads(x) for x in gzip.open(path,"rt",encoding="utf-8"))}
    old_ids,old_groups = set(),set()
    for filename,sha in manifest["prior_exposure_files"].items():
        path = ROOT/filename; assert sha256_file(path) == sha
        for r in lines(path):
            if isinstance(r,dict):
                old_ids.add(r.get("id"))
                if r.get("state"):
                    old_groups.add(text_group(r["state"]))
    records = {}
    for split,info in manifest["splits"].items():
        path = DIRECTORY/info["file"];assert sha256_file(path) == info["sha256"]
        records[split] = lines(path)
        assert len(records[split]) == info["count"]
        assert dict(Counter(r["target"] for r in records[split])) == info["labels"] == info["class_counts"]
        assert len({r["id"] for r in records[split]}) == info["count"]
        assert len({r["group_id"] for r in records[split]}) == info["count"]
        assert not {r["id"] for r in records[split]} & old_ids
        assert not {r["group_id"] for r in records[split]} & old_groups
        for r in records[split]:
            assert r["group_id"] == text_group(r["state"])
            if r["dataset"] == "MASSIVE":
                source = gold[r["source_split"]][r["source_id"]]
                assert (r["state"],r["target"]) == (source["text"],source["label"])
                assert r["target"] != "general"
            else:
                source = original[r["source_id"]]
                assert source["role"] == "prompter" and source["parent_id"] is None and source["lang"] == "es"
                assert source["synthetic"] is False and source["deleted"] is False
                assert r["state"] == source["text"] == annotations[r["source_id"]]["text"]
                assert r["target"] == annotations[r["source_id"]]["target"] != "excluded"
                assert "provisional" in r["annotation_source"]
    for i,(a,first) in enumerate(records.items()):
        for b,second in list(records.items())[i+1:]:
            assert not {r["id"] for r in first} & {r["id"] for r in second}
            assert not {r["group_id"] for r in first} & {r["group_id"] for r in second}
    assert sum(map(len,records.values())) == 507
    checks.append("507 fresh source texts; roles and all previous training/evaluation groups disjoint; original MASSIVE labels verified; OASST human text provenance verified, new intent labels explicitly provisional")
    config,task_fp = PrecisionTask()._load()
    development = read_json(DIRECTORY/"development-selection.json")
    assert sha256_file(DIRECTORY/"development-selection.json") == config["selection_sha256"] == manifest["development_selection_sha256"]
    for split,sha in development["source_sha256"].items():
        assert sha256_file(PREVIOUS/f"{split}-predictions.jsonl") == sha
    for trial in development["trials"]:
        cfg = config | {"minimum_general_similarity":trial["minimum_general_similarity"]}
        kept,removed = True,False
        for split in ("development","test"):
            previous_records = lines(ROOT/"evaluation/massive-es-v6"/f"{split}.jsonl")
            previous_evidence = lines(PREVIOUS/f"{split}-predictions.jsonl")
            values,targets,changed,lost = [],[],[],[]
            for r,e in zip(previous_records,previous_evidence):
                c = e["component_probabilities"];s = cached_evidence(r["state"])
                result = compose(c["base_raw"],c["focused_raw"],cfg,s["general_similarity"])
                before,old,new = (int(np.argmax(v)) for v in (c["base_raw"],e["probabilities"],result))
                y = list(CATEGORIES).index(r["target"])
                values.append(result);targets.append(y)
                if old == y and before != y and new != y:
                    lost.append(r["id"])
                if old != new:
                    changed.append({"id":r["id"],"state":r["state"],"target":r["target"],"previous":list(CATEGORIES)[old],
                                    "new":list(CATEGORIES)[new],"training_evidence":s})
                    if split == "test" and old == list(CATEGORIES).index("general") and new == y and r["target"] != "general":
                        removed = True
            detail = trial["details"][split]
            assert_equivalent(metrics(np.array(values),np.array(targets),list(CATEGORIES)),detail["metrics"])
            assert lost == detail["lost_new_correct_ids"] and changed == detail["changed"]
            kept &= not lost
        assert kept == trial["preserved_all_new_correct"] and removed == trial["removed_introduced_general_false_positive"]
    eligible = [r for r in development["trials"] if r["preserved_all_new_correct"] and r["removed_introduced_general_false_positive"]]
    chosen = max(eligible,key=lambda r:(r["details"]["development"]["metrics"]["accuracy"],-r["minimum_general_similarity"]))
    assert chosen["minimum_general_similarity"] == config["minimum_general_similarity"] == development["selected"]
    assert len(chosen["details"]["test"]["changed"]) == 1
    checks.append("All six old-data threshold trials independently replayed; all 8 newly correct v6 rescues preserved; diagnosed added General false positive removed")
    _,_,base_fp = SpanishTask()._load();_,_,focus_fp = GeneralTask()._load()
    outputs,baselines = {},{}
    for split,rr in records.items():
        baseline = lines(run/f"{split}-v2.jsonl")
        focused = {r["id"]:r for r in lines(run/f"{split}-focus.jsonl")}
        output = lines(run/f"{split}-predictions.jsonl")
        assert len(baseline) == len(output) == len(rr)
        expected_reviews = set()
        for r,b,o in zip(rr,baseline,output):
            assert r["id"] == b["id"] == o["id"] and r["target"] == b["target"] == o["target"]
            assert b["request_sha256"] == digest(request_for(r) | {"task_profile":"massive_es_v2"})
            assert b["metadata"]["task_profile_fingerprint"] == base_fp and b["metadata"]["prompt_version"] == BASE_VERSION
            support = cached_evidence(r["state"])
            winner = list(CATEGORIES)[int(np.argmax(b["probabilities"]))]
            review = needs_review(b["probabilities"],config["threshold"],config["component_temperature"])
            review &= winner not in config["protected_domains"] or support["general_similarity"] >= config["minimum_general_similarity"]
            if review:
                expected_reviews.add(r["id"])
            f = focused.get(r["id"])
            if f:
                assert f["request_sha256"] == digest(request_for(r) | {"task_profile":"massive_es_v5"})
                assert f["metadata"]["task_profile_fingerprint"] == focus_fp and f["metadata"]["prompt_version"] == FOCUSED_VERSION
            values = compose(b["probabilities"],f["probabilities"] if f else None,config,support["general_similarity"])
            assert np.allclose(values,o["probabilities"],atol=1e-12,rtol=0)
            assert o["derived"] and o["metadata"]["derived_timings"] and o["metadata"]["task_profile_fingerprint"] == task_fp
            assert o["metadata"]["llm_calls"] == 1+bool(f) and o["metadata"]["calibrated"] is False
            assert o["component_probabilities"]["base_raw"] == b["probabilities"]
            assert o["component_probabilities"]["focused_raw"] == (f["probabilities"] if f else None)
            assert o["component_probabilities"]["reviewed"] == review
            for source in [b]+([f] if f else []):
                assert source["metadata"]["calibrated"] is False
                assert len(source["probabilities"]) == 18 and np.isfinite(source["probabilities"]).all()
                assert min(source["probabilities"]) >= 0 and abs(sum(source["probabilities"])-1) < 1e-6
        assert set(focused) == expected_reviews
        outputs[split],baselines[split] = output,baseline
    checks.append("Every native request hash, retrieval gate and derived 18-way probability independently verified; no reference labels enter inference payloads")
    cp,cy = arrays(outputs["calibration"]); vp,vy = arrays(outputs["validation"])
    temperature = fit_temperature(cp,cy)
    assert abs(temperature-selection["temperature"]) < 1e-9
    vr,vc = metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,temperature),vy,list(CATEGORIES))
    approved = vc["negative_log_loss"] < vr["negative_log_loss"]-.0001
    assert selection["validation_approved"] == approved and selection["raw_validation"] == vr and selection["candidate_validation"] == vc
    assert selection["selected_temperature"] == (temperature if approved else 1.)
    freeze = read_json(run/"pre-test-freeze.json")
    assert freeze["test_predictions_existed"] is False
    assert freeze["selection_sha256"] == sha256_file(run/"selection.json") and freeze["manifest_sha256"] == sha256_file(DIRECTORY/"manifest.json")
    assert (run/"pre-test-freeze.json").stat().st_mtime <= (run/"test-v2.jsonl").stat().st_birthtime
    raw,y = arrays(outputs["test"]);base_raw,_ = arrays(baselines["test"])
    base = scaled_array(base_raw,summary["baseline"]["temperature"]);candidate = scaled_array(raw,selection["selected_temperature"])
    bm,cm = metrics(base,y,list(CATEGORIES)),metrics(candidate,y,list(CATEGORIES))
    assert summary["baseline"]["calibrated"] == bm and summary["test"]["calibrated"] == cm and summary["test"]["raw"] == metrics(raw,y,list(CATEGORIES))
    paired = paired_comparison(base,candidate,y)
    assert summary["paired_comparison"] == paired
    gates,pvalue = conditions(bm,cm,paired,approved,chosen["preserved_all_new_correct"])
    activation = read_json(run/"activation.json")
    assert activation["conditions"] == gates and activation["approved"] == all(gates.values()) and activation["mcnemar_one_sided_pvalue"] == pvalue
    for source,data in summary["per_source"].items():
        ix = [i for i,r in enumerate(records["test"]) if r["dataset"] == source]
        assert data["baseline"] == metrics(base[ix],y[ix],list(CATEGORIES)) and data["candidate"] == metrics(candidate[ix],y[ix],list(CATEGORIES))
    if approved:
        profile = read_json(PROFILES/f"{CALIBRATION_NAME}.json")
        assert profile == read_json(run/"calibration-profile.json") and profile["temperature"] == temperature
        assert profile["scope"]["task_profile_fingerprint"] == task_fp and not profile["provenance"]["test_used_for_fitting"]
    checks.append("Calibration refit on calibration only; validation decision and temperature frozen before test creation; paired bootstrap, every metric and promotion gate reproduced")
    suite = ET.parse(DIRECTORY/"unit-tests.xml").getroot()
    assert all(int(s.get("failures",0)) == int(s.get("errors",0)) == 0 for s in suite.iter("testsuite"))
    live_result = {"passed":False,"executed":live}
    if live:
        results = []
        with httpx.Client(base_url=settings()["api_url"],timeout=180,trust_env=False) as client:
            for message,expected in (("dame algunos consejos de salud","general"),("hoy he tenido una reunión con jorge","general"),("y la oscuridad ha caído","iot")):
                options = list(CATEGORIES.values())
                def predict(options):
                    payload = {"state":message,"question":QUESTION,"options":options,"task_profile":NAME}
                    if approved:
                        payload["calibration_profile"] = CALIBRATION_NAME
                    response = client.post("/predict",json=payload);response.raise_for_status()
                    result = response.json()
                    c = result["component_probabilities"]
                    s = c["training_evidence"]
                    expected_raw = compose(c["base_raw"],c["focused_raw"],config,s["general_similarity"] if s else 0.)
                    expected_prob = temperature_scale(expected_raw.tolist(),temperature) if approved else expected_raw
                    actual = [result["probabilities"][v] for v in CATEGORIES.values()]
                    assert np.allclose(actual,expected_prob,atol=1e-10,rtol=0)
                    assert result["metadata"]["task_profile_fingerprint"] == task_fp and result["metadata"]["calibrated"] == approved
                    assert result["prediction"] == CATEGORIES[expected]
                    assert result["prediction_index"] == options.index(CATEGORIES[expected])
                    return result
                cold = predict(options)
                warm = predict(options)
                reversed_result = predict(list(reversed(options)))
                assert max(abs(warm["probabilities"][v]-reversed_result["probabilities"][v]) for v in options) < 1e-8
                results.append({"message":message,"expected":expected,"initial":cold,"repeat":warm,"reordered":reversed_result,
                                "initial_repeat_max_delta":max(abs(cold["probabilities"][v]-warm["probabilities"][v]) for v in options)})
            payload = {"state":"hola","question":"Otra pregunta","options":list(CATEGORIES.values()),"task_profile":NAME}
            assert client.post("/predict",json=payload).status_code == 422
            payload.update(question=QUESTION,modality="image")
            assert client.post("/predict",json=payload).status_code == 422
            if approved:
                payload.update(modality="text",task_profile="massive_es_v2",calibration_profile=CALIBRATION_NAME)
                assert client.post("/predict",json=payload).status_code == 422
        live_result = {"passed":True,"executed":True,"cases":results,"scope_rejections_passed":True}
        save_json(run/"live-api-results.json",live_result)
        checks.append("Actual API preserves both newly corrected messages, rejects the lighting false positive, matches probability composition/calibration, supports reordered options and rejects wrong scopes")
    result = {"passed":True,"checks":checks,"live":live_result,"human_adjudicated_oasst_intent_labels":False,
              "note":"Verified reproducibility and text provenance; this does not certify the new single-annotator OASST intent labels as human gold."}
    save_json(run/"verification.json",result)
    print(json.dumps({"passed":True,"checks":len(checks),"live":live_result["passed"]}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    verify(parser.parse_args().live)
