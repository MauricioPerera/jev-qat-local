"""Verify the selective experiment, including every probability and live inference."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx
import joblib
import numpy as np

from jev_qat.calibration import PROFILES, sha256_file, temperature_scale
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, SOURCE_HASHES, digest, load_records, text_group
from jev_qat.eval_metrics import fit_temperature, metrics, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import V1,V2,DIRECTORY as V3, FOCUS, HEADS, lexical_scores, read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.selective_experiment import DIRECTORY, CALIBRATION_NAME, mix_array
from jev_qat.selective_task import NAME,VERSION,PROFILE_PATH,SelectiveTask,combine_probabilities
from jev_qat.spanish_task import SpanishTask,VERSION as BASE_VERSION,build_prompt


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify(live=False):
    manifest = read_json(DIRECTORY/"manifest.json")
    original = dict(manifest)
    fingerprint = original.pop("manifest_fingerprint")
    assert digest(original) == fingerprint
    run = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    env,summary,selection = [read_json(run/f"{name}.json") for name in ("environment","summary","selection")]
    assert run.name == digest({"manifest":fingerprint,"environment":env})[:16]
    records = {s:load_records(DIRECTORY,s) for s in manifest["splits"]}
    checks = []
    sources = {}
    for split,sha in SOURCE_HASHES.items():
        path = V1/"source"/f"{split}-es.json.gz"
        assert sha256_file(path) == sha
        sources[split] = {str(r["id"]):r for r in (json.loads(line) for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines())}
    for split,info in manifest["splits"].items():
        assert sha256_file(DIRECTORY/info["file"]) == info["sha256"]
        assert len(records[split]) == info["count"] and Counter(r["target"] for r in records[split]) == info["class_counts"]
        for r in records[split]:
            gold = sources[r["source_split"]][r["source_id"]]
            assert (r["state"],r["target"]) == (gold["text"],gold["label"])
            assert r["group_id"] == text_group(r["state"])
    assert records["development"] == load_records(V3,"development")+load_records(V3,"validation")
    previous = [r for d in (V1,V2,V3) for s in ("calibration","validation","test") for r in load_records(d,s)]+load_records(V3,"development")
    bank_path = ROOT/manifest["training_bank_file"]
    assert sha256_file(bank_path) == manifest["training_bank_sha256"]
    bank = joblib.load(bank_path)
    assert len(bank["records"]) == manifest["training_count"]
    for split in ("calibration","validation","test"):
        for other in [previous,bank["records"]]+[rows for name,rows in records.items() if name!=split]:
            for key in ("source_id","group_id"):
                assert not {r[key] for r in records[split]} & {r[key] for r in other}
    for name,sha in manifest["prior_manifests_sha256"].items():
        assert sha256_file(ROOT/"evaluation"/name/"manifest.json") == sha
    checks.append("original_labels_pinned_sources_and_fresh_disjoint_calibration_validation_test")
    for name,sha in (env["code"]|env["v4_code"]).items():
        assert sha256_file(run/"source"/name) == sha and sha256_file(ROOT/"jev_qat"/name) == sha
    assert sha256_file(run/"source/search_selective_spanish.py") == env["selection_search_sha256"]
    profile = read_json(PROFILE_PATH)
    assert profile == read_json(run/"selected-task-profile.json")
    assert sha256_file(PROFILE_PATH) == env["task_profile_sha256"]
    assert sha256_file(ROOT/profile["head_file"]) == profile["head_sha256"]
    artifact = joblib.load(ROOT/profile["head_file"])
    assert artifact["source_bank_sha256"] == manifest["training_bank_sha256"]
    model = artifact["model"]
    assert model.class_weight is None and model.C==16 and max(model.n_iter_) < model.max_iter
    trained = joblib.load(HEADS)["models"][profile["head_id"]]
    assert np.array_equal(model.coef_,trained.coef_) and np.array_equal(model.intercept_,trained.intercept_)
    base_task = SpanishTask()
    focused = SelectiveTask(base_task)
    _,_,_,task_fingerprint = focused._load()
    arguments = {key:records["test"][0][key] for key in ("state","question","options","modality")}
    labels = read_json(ROOT/"jev_qat/option_labels.json")[:18]
    assert build_prompt(base_task.prepare("massive_es_v2",**arguments),arguments["state"],labels)==build_prompt(focused.prepare(NAME,**arguments),arguments["state"],labels)
    checks.append("archived_scoring_source_and_train_only_head_checksums")
    development = read_json(run/"development-selection.json")
    assert sha256_file(DIRECTORY/"development-selection.json") == manifest["selection_sha256"]
    assert development == read_json(DIRECTORY/"development-selection.json")
    assert len(development["trials"]) == 48 and not development["fresh_test_used"]
    selected = development["selected"]
    assert selected["approved"] and selected["head_id"]==profile["head_id"]
    assert selected["weight"]==profile["weight"] and selected["threshold"]==profile["threshold"]
    old_run = Path(read_json(V3/"latest-run.json")["directory"])
    dev_evidence = lines(V3/"development-gemma.jsonl")+lines(old_run/"validation-gemma.jsonl")
    gemma = {"development":dev_evidence}
    predictions = {}
    for split,rows in records.items():
        if split!="development":
            gemma[split] = lines(run/f"{split}-gemma.jsonl")
        assert len(gemma[split]) == len(rows)
        for r,evidence in zip(rows,gemma[split]):
            assert (r["id"],r["target"]) == (evidence["id"],evidence["target"])
            assert evidence["request_sha256"] == digest(request_for(r|{"task_profile":"massive_es_v2"}))
            assert evidence["metadata"]["prompt_version"] == BASE_VERSION
            assert evidence["metadata"]["task_profile_fingerprint"] == profile["base_task_fingerprint"]
            assert not evidence["metadata"]["calibrated"]
            assert set(evidence["metadata"]["retrieved_example_ids"]) <= {r["id"] for r in bank["records"]}
        raw,targets = arrays(gemma[split])
        base = scaled_array(raw,profile["gemma_component_temperature"])
        lexical = lexical_scores(model,rows,bank)
        mixture = mix_array(base,lexical,profile["weight"],profile["threshold"])
        if split=="development":
            assert metrics(base,targets,list(CATEGORIES)) == development["baseline"]
            assert metrics(mixture,targets,list(CATEGORIES)) == selected["metrics"]
            old = {r["label"]:r["f1"] for r in development["baseline"]["per_class"]}
            new = {r["label"]:r["f1"] for r in selected["metrics"]["per_class"]}
            assert all(new[c]>=old[c]-1e-12 for c in FOCUS) and selected["mean_focus_f1_gain"]>=.01
            assert selected["metrics"]["macro_f1"]>=development["baseline"]["macro_f1"]
            assert all((mixture[s].argmax(1)==targets[s]).mean()>=(base[s].argmax(1)==targets[s]).mean() for s in (slice(0,300),slice(300,None)))
        else:
            predictions[split] = lines(run/f"{split}-predictions.jsonl")
            actual,actual_y = arrays(predictions[split])
            assert np.array_equal(actual_y,targets) and np.allclose(actual,mixture,atol=1e-12,rtol=1e-12)
            for e,r in zip(gemma[split],predictions[split]):
                assert r["derived"] and r["source_request_sha256"]==e["request_sha256"]
                assert r["metadata"]["task_profile_fingerprint"]==task_fingerprint
    for name,sha in summary["artifacts_sha256"].items():
        assert sha256_file(run/name)==sha
    checks.append("development_gate_and_every_shared_gemma_and_derived_probability")
    p,y = arrays(predictions["calibration"])
    assert abs(fit_temperature(p,y)-selection["temperature"])<1e-9
    vp,vy = arrays(predictions["validation"])
    vr,vc = metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,selection["temperature"]),vy,list(CATEGORIES))
    assert vr==selection["raw_validation"] and vc==selection["candidate_validation"]
    assert selection["validation_approved"]==(vc["negative_log_loss"]<vr["negative_log_loss"]-.0001)
    assert not selection["test_used_for_selection"] and not summary["test_used_for_selection"]
    raw,targets = arrays(predictions["test"])
    calibrated = scaled_array(raw,selection["selected_temperature"])
    baseline = scaled_array(arrays(gemma["test"])[0],profile["gemma_component_temperature"])
    assert metrics(raw,targets,list(CATEGORIES))==summary["test"]["raw"]
    assert metrics(calibrated,targets,list(CATEGORIES))==summary["test"]["calibrated"]
    assert metrics(baseline,targets,list(CATEGORIES))==summary["baseline"]["calibrated"]
    assert paired_comparison(baseline,calibrated,targets)==summary["paired_comparison"]
    assert np.array_equal(raw.argmax(1),calibrated.argmax(1))
    if summary["profile_name"]:
        cal = read_json(PROFILES/f"{CALIBRATION_NAME}.json")
        assert cal==read_json(run/"calibration-profile.json")
        assert cal["scope"]["task_profile_fingerprint"]==task_fingerprint and cal["temperature"]==selection["selected_temperature"]
        assert cal["provenance"]["selection_sha256"]==sha256_file(run/"selection.json")
    suites = ET.parse(run/"unit-tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.attrib["failures"])+int(s.attrib["errors"])==0 for s in suites)
    test_count = sum(int(s.attrib["tests"]) for s in suites)
    checks.append("independent_test_metrics_bootstrap_and_pretest_calibration")
    delivery_manifest = run/"delivery-source-manifest.json"
    if delivery_manifest.exists():
        for relative,sha in read_json(delivery_manifest).items():
            assert sha256_file(run/"delivery-source"/relative)==sha
            assert sha256_file(ROOT/relative)==sha
        checks.append("delivered_helper_report_verifier_documentation_and_tests")
    live_result = {}
    if live:
        config = settings()
        with httpx.Client(base_url=config["api_url"],timeout=config["request_timeout_seconds"],trust_env=False) as client:
            assert client.get("/evaluation").content==(run/"report.html").read_bytes()
            assert client.get("/evaluation/summary").json()==summary
            post = lambda body:client.post("/predict",json=body).raise_for_status().json()
            vector = lambda result:np.array([result["probabilities"][v] for v in CATEGORIES.values()])
            chosen_rows = [records["test"][0]]+[next(r for r in records["test"] if r["target"]==c) for c in FOCUS]
            saved = []
            for record in chosen_rows:
                body = request_for(record)|{"task_profile":NAME}
                if summary["profile_name"]:
                    body["calibration_profile"] = CALIBRATION_NAME
                initial,result = post(body),post(body)
                c = result["component_probabilities"]
                mixture,gate = combine_probabilities(c["gemma_raw"],c["lexical"],profile["weight"],profile["gemma_component_temperature"],profile["threshold"])
                assert c["gate_active"]==gate and c["threshold"]==profile["threshold"]
                expected = temperature_scale(mixture,selection["selected_temperature"]) if summary["profile_name"] else mixture
                assert np.allclose(vector(result),expected,atol=1e-12,rtol=1e-12)
                assert result["metadata"]["task_profile_fingerprint"]==task_fingerprint
                changed = post(body|{"options":list(reversed(body["options"]))})
                assert result["prediction"]==changed["prediction"] and np.allclose(vector(result),vector(changed),atol=1e-6,rtol=0)
                saved.append({"request":body,"response":result,"order_max_delta":float(np.max(abs(vector(result)-vector(changed)))),
                              "initial_to_warm_max_probability_delta":float(np.max(abs(vector(result)-vector(initial))))})
            save_json(run/"live-examples.json",saved)
            assert client.post("/predict",json=body|{"calibration_profile":"massive_scenario_es_v2"}).status_code==422
            if summary["profile_name"]:
                assert client.post("/predict",json=body|{"task_profile":"massive_es_v2"}).status_code==422
            else:
                assert client.post("/predict",json=body|{"calibration_profile":CALIBRATION_NAME}).status_code==422
            generic = request_for(records["test"][0])
            assert post(generic)["metadata"]["task_profile"] is None
            assert post(generic|{"task_profile":"massive_es_v2","calibration_profile":"massive_scenario_es_v2"})["metadata"]["calibrated"]
            live_result = {"passed":True,"order_audit_messages":len(saved),"mixture_formula_verified":True,"incompatible_profiles_rejected":True,
                           "initial_to_warm_max_probability_delta":max(r["initial_to_warm_max_probability_delta"] for r in saved)}
            checks.append("live_report_selective_mixture_calibration_scope_and_option_order")
    result = {"passed":True,"run_id":run.name,"checks":checks,"unit_tests_passed":test_count,"live":live_result,
              "package_versions":{name:importlib.metadata.version(name) for name in ("scikit-learn","joblib","numpy","scipy","httpx")}}
    save_json(run/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    verify(parser.parse_args().live)
