"""Independently verify retired training, new holdouts, calibration and activation."""
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
from scipy.stats import binomtest

from jev_qat.calibration import PROFILES,sha256_file,temperature_scale
from jev_qat.config import ROOT,save_json,settings
from jev_qat.eval_data import CATEGORIES,SOURCE_HASHES,digest,load_records,text_group
from jev_qat.eval_metrics import fit_temperature,metrics,scaled_array
from jev_qat.evaluate import arrays,request_for
from jev_qat.focus_experiment import V1,V2,DIRECTORY as V3,read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.selective_experiment import DIRECTORY as V4
from jev_qat.general_rescue_data import DIRECTORY
from jev_qat.general_rescue_run import CALIBRATION_NAME,general,read_lines
from jev_qat.general_rescue_task import NAME,VERSION,PROFILE_PATH,RescueTask
from jev_qat.general_rescue_math import combine,needs_review
from jev_qat.general_task import GeneralTask,VERSION as FOCUSED_VERSION
from jev_qat.general_data import DIRECTORY as V5
from jev_qat.spanish_task import VERSION as V2_VERSION


def verify(live=False):
    manifest=read_json(DIRECTORY/"manifest.json")
    copy=dict(manifest)
    fingerprint=copy.pop("manifest_fingerprint")
    assert digest(copy)==fingerprint
    run=Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    env=read_json(run/"environment.json")
    summary=read_json(run/"summary.json")
    selection=read_json(run/"selection.json")
    activation=read_json(run/"activation.json")
    assert run.name==digest({"manifest":fingerprint,"environment":env})[:16]
    records={s:load_records(DIRECTORY,s) for s in manifest["splits"]}
    checks=[]
    gold={}
    for split,sha in SOURCE_HASHES.items():
        path=V1/"source"/f"{split}-es.json.gz"
        assert sha256_file(path)==sha
        gold[split]={str(r["id"]):r for r in (json.loads(line) for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines())}
    for split,info in manifest["splits"].items():
        assert sha256_file(DIRECTORY/info["file"])==info["sha256"]
        assert len(records[split])==info["count"]
        assert len({r["id"] for r in records[split]})==len(records[split])
        assert len({r["group_id"] for r in records[split]})==len(records[split])
        assert Counter(r["target"] for r in records[split])==info["class_counts"]
        for r in records[split]:
            source=gold[r["source_split"]][r["source_id"]]
            assert (r["state"],r["target"],r["group_id"])==(source["text"],source["label"],text_group(source["text"]))
    partitions=list(records.items())
    for i,(a,first) in enumerate(partitions):
        for b,second in partitions[i+1:]:
            for key in ("source_id","group_id"):
                assert not {r[key] for r in first}&{r[key] for r in second},(a,b,key)
    previous={r["id"]:r for d in (V1,V2,V3,V4) for s in ("calibration","validation","test") for r in load_records(d,s)}
    previous.update({r["id"]:r for r in load_records(V3,"development")})
    for split in ("calibration","validation","test"):
        for key in ("source_id","group_id"):
            assert not {r[key] for r in records[split]}&{r[key] for r in previous.values()}
    assert records["development"]==load_records(V3,"development")+load_records(V4,"validation")
    assert sha256_file(V5/"manifest.json")==manifest["source_v5_manifest_sha256"]
    for split,info in manifest["splits"].items():
        assert sha256_file(V5/info["file"])==info["sha256"]
    retired=[r for r in records["training"] if r["source_split"]=="test"]
    expected_retired={r["id"] for r in previous.values() if r["source_split"]=="test" and r["target"]=="general"}
    assert {r["id"] for r in retired}==expected_retired==set(manifest["retired_test_general_ids"])
    assert len(retired)==manifest["retired_test_general_count"]==138
    assert all(r["target"]=="general" for r in retired)
    assert sum(r["target"]=="general" for r in records["training"])==manifest["general_training_after"]==818
    for name,sha in manifest["prior_manifests_sha256"].items():
        assert sha256_file(ROOT/"evaluation"/name/"manifest.json")==sha
    checks.append("original_labels_explicit_test_retirement_and_genuinely_new_disjoint_holdouts")
    for name,sha in (env["code"]|env["v6_code"]).items():
        assert sha256_file(run/"source"/name)==sha
        assert sha256_file(ROOT/"jev_qat"/name)==sha
    profile=read_json(PROFILE_PATH)
    assert profile==read_json(run/"task-profile.json")
    assert sha256_file(PROFILE_PATH)==env["task_profile_sha256"]
    assert sha256_file(DIRECTORY/"development-selection.json")==manifest["development_selection_sha256"]==profile["selection_sha256"]
    assert sha256_file(ROOT/manifest["bank_file"])==manifest["bank_sha256"]
    bank=joblib.load(ROOT/manifest["bank_file"])
    assert bank["records"]==records["training"] and bank["version"]==FOCUSED_VERSION
    assert bank["matrix"].shape[0]==len(records["training"])
    norms=np.sqrt(bank["matrix"].multiply(bank["matrix"]).sum(1)).A.ravel()
    # A source containing only tokens below min_df legitimately has no features.
    assert np.all((norms==0)|np.isclose(norms,1,atol=1e-6))
    task=GeneralTask()
    _,_,focused_fingerprint=task._load()
    _,task_fingerprint=RescueTask()._load()
    predictions={}
    components={}
    training_ids={r["id"] for r in records["training"]}
    for split in ("development","calibration","validation","test"):
        path=run/f"{split}-predictions.jsonl"
        assert sha256_file(path)==summary["artifacts_sha256"][path.name]
        predictions[split]=read_lines(path)
        assert len(predictions[split])==len(records[split])
        native_base=read_lines(run/f"{split}-v2.jsonl")
        native_focus=read_lines(run/f"{split}-focus.jsonl")
        components[split]=native_base,native_focus
        assert len(native_base)==len(records[split])
        review_ids={r["id"] for r,b in zip(records[split],native_base) if needs_review(b["probabilities"],profile["threshold"],profile["component_temperature"])}
        assert {r["id"] for r in native_focus}==review_ids
        focused_by_id={r["id"]:r for r in native_focus}
        for i,(r,prediction,base) in enumerate(zip(records[split],predictions[split],native_base)):
            assert (prediction["id"],prediction["target"])==(r["id"],r["target"])
            assert prediction["derived"] and prediction["source_requests"]["base"]==base["request_sha256"]
            assert base["id"]==r["id"] and base["target"]==r["target"]
            assert base["request_sha256"]==digest(request_for(r|{"task_profile":"massive_es_v2"}))
            assert base["metadata"]["prompt_version"]==V2_VERSION
            focused=focused_by_id.get(r["id"])
            expected=combine(base["probabilities"],focused["probabilities"],profile["weight"],profile["minimum_probability"]) if focused else base["probabilities"]
            assert np.allclose(expected,prediction["probabilities"],atol=1e-12,rtol=1e-12)
            assert prediction["component_probabilities"]["reviewed"]==bool(focused)
            metadata=prediction["metadata"]
            assert metadata["prompt_version"]==VERSION and metadata["task_profile_fingerprint"]==task_fingerprint
            assert not metadata["calibrated"] and metadata["method"]=="gemma_general_rescue"
            assert set(base["metadata"]["retrieved_example_ids"])<=training_ids
            if focused:
                assert prediction["source_requests"]["focused"]==focused["request_sha256"]
                assert focused["request_sha256"]==digest(request_for(r|{"task_profile":"massive_es_v5"}))
                assert focused["metadata"]["prompt_version"]==FOCUSED_VERSION
                assert focused["metadata"]["task_profile_fingerprint"]==focused_fingerprint
                ids=focused["metadata"]["retrieved_example_ids"]
                assert len(ids)<=8 and len(ids)==len(set(ids)) and set(ids)<=training_ids
            else:
                assert prediction["source_requests"]["focused"] is None
            if focused and i%31==0:
                arguments={k:r[k] for k in ("state","question","options","modality")}
                retrieved=task.prepare("massive_es_v5",**arguments)
                assert [e["id"] for e in retrieved["examples"]]==ids
        for name in (f"{split}-v2.jsonl",f"{split}-focus.jsonl"):
            assert sha256_file(run/name)==summary["artifacts_sha256"][name]
    checks.append("frozen_expanded_bank_scoring_source_and_training_only_retrieval")
    v4run=Path(read_json(V4/"latest-run.json")["directory"])
    dev_base=read_lines(run/"development-v2.jsonl")
    assert dev_base==read_lines(V3/"development-gemma.jsonl")+read_lines(v4run/"validation-gemma.jsonl")
    assert sha256_file(run/"development-v2.jsonl")==summary["artifacts_sha256"]["development-v2.jsonl"]
    t2=summary["baseline"]["temperature"]
    dp,dy=arrays(predictions["development"])
    bp,by=arrays(dev_base)
    assert np.array_equal(dy,by)
    dm,bm=metrics(dp,dy,list(CATEGORIES)),metrics(scaled_array(bp,t2),dy,list(CATEGORIES))
    development=read_json(run/"development-selection.json")
    assert dm==development["candidate"] and bm==development["baseline"] and development["approved"]
    assert general(dm)["f1"]>=general(bm)["f1"]+.03 and general(dm)["precision"]>=general(bm)["precision"]-.03
    assert dm["accuracy"]>=bm["accuracy"] and dm["macro_f1"]>=bm["macro_f1"]
    search=read_json(DIRECTORY/"development-selection.json")
    assert not search["fresh_holdouts_used"] and len(search["trials"])==36
    source=Path(search["source_v5_run"])
    assert read_json(source/"progress.json")["test_opened"] is False
    for name,sha in search["source_evidence_sha256"].items():
        assert sha256_file(source/name)==sha
    focused_full=read_lines(source/"development-predictions.jsonl")
    fp,fy=arrays(focused_full)
    assert np.array_equal(fy,dy)
    for trial in search["trials"]:
        review=[needs_review(p.tolist(),trial["threshold"],search["component_temperature"]) for p in bp]
        candidate=np.array([combine(a.tolist(),b.tolist(),trial["weight"],trial["minimum_probability"]) if active else a for a,b,active in zip(bp,fp,review)])
        score=metrics(candidate,dy,list(CATEGORIES))
        assert score==trial["metrics"] and trial["review_count"]==sum(review)
        expected=(general(score)["f1"]>=general(bm)["f1"]+.03 and general(score)["precision"]>=general(bm)["precision"]-.03
                  and score["accuracy"]>bm["accuracy"] and score["macro_f1"]>=bm["macro_f1"])
        assert trial["approved"]==expected
    eligible=[r for r in search["trials"] if r["approved"]]
    best=max(eligible,key=lambda r:(r["metrics"]["accuracy"],general(r["metrics"])["f1"],-r["review_count"],-r["weight"]))
    assert best==search["selected"]
    assert all(profile[k]==best[k] for k in ("threshold","weight","minimum_probability"))
    cp,cy=arrays(predictions["calibration"])
    assert abs(fit_temperature(cp,cy)-selection["temperature"])<1e-9
    vp,vy=arrays(predictions["validation"])
    vr,vc=metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,selection["temperature"]),vy,list(CATEGORIES))
    assert vr==selection["raw_validation"] and vc==selection["candidate_validation"]
    assert selection["validation_approved"]==(vc["negative_log_loss"]<vr["negative_log_loss"]-.0001)
    assert not selection["test_used_for_selection"] and not development["test_used_for_selection"] and not summary["test_used_for_selection"]
    baseline_rows=read_lines(run/"test-v2.jsonl")
    assert sha256_file(run/"test-v2.jsonl")==summary["artifacts_sha256"]["test-v2.jsonl"]
    for r,prediction in zip(records["test"],baseline_rows):
        assert (r["id"],r["target"])==(prediction["id"],prediction["target"])
        assert prediction["request_sha256"]==digest(request_for(r|{"task_profile":"massive_es_v2"}))
        assert prediction["metadata"]["prompt_version"]==V2_VERSION and not prediction["metadata"]["calibrated"]
    raw,targets=arrays(predictions["test"])
    base=scaled_array(arrays(baseline_rows)[0],t2)
    calibrated=scaled_array(raw,selection["selected_temperature"])
    before,after=metrics(base,targets,list(CATEGORIES)),metrics(calibrated,targets,list(CATEGORIES))
    assert before==summary["baseline"]["calibrated"] and after==summary["test"]["calibrated"]
    assert metrics(raw,targets,list(CATEGORIES))==summary["test"]["raw"]
    paired=paired_comparison(base,calibrated,targets)
    assert paired==summary["paired_comparison"]
    assert np.array_equal(raw.argmax(1),calibrated.argmax(1))
    fixed,broken=paired["fixed_errors"],paired["introduced_errors"]
    pvalue=float(binomtest(fixed,fixed+broken,.5,alternative="greater").pvalue) if fixed+broken else 1.
    conditions={"calibration_validated":selection["validation_approved"],"general_f1_gain_at_least_3_points":general(after)["f1"]>=general(before)["f1"]+.03,
                "general_precision_not_lower":general(after)["precision"]>=general(before)["precision"],"overall_accuracy_higher":after["accuracy"]>before["accuracy"],
                "macro_f1_not_lower":after["macro_f1"]>=before["macro_f1"],"paired_one_sided_p_below_0_05":pvalue<.05,
                "test_nll_not_higher":after["negative_log_loss"]<=before["negative_log_loss"]}
    assert activation["conditions"]==conditions and activation["approved"]==all(conditions.values())
    assert activation["mcnemar_one_sided_pvalue"]==pvalue
    assert summary["activation"]==activation
    if activation["activated"]:
        assert activation["approved"] and read_json(ROOT/"classification/current.json")["version"]=="v6"
    if summary["profile_name"]:
        cal=read_json(PROFILES/f"{CALIBRATION_NAME}.json")
        assert cal==read_json(run/"calibration-profile.json")
        assert cal["scope"]["task_profile_fingerprint"]==task_fingerprint and cal["temperature"]==selection["selected_temperature"]
        assert cal["provenance"]["selection_sha256"]==sha256_file(run/"selection.json")
    checks.append("development_new_calibration_new_paired_test_and_strict_activation_gate")
    suites=ET.parse(run/"unit-tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.attrib["failures"])+int(s.attrib["errors"])==0 for s in suites)
    test_count=sum(int(s.attrib["tests"]) for s in suites)
    if (run/"delivery-source-manifest.json").exists():
        for relative,sha in read_json(run/"delivery-source-manifest.json").items():
            assert sha256_file(ROOT/relative)==sha and sha256_file(run/"delivery-source"/relative)==sha
        checks.append("delivered_source_checksums")
    live_result={}
    if live:
        config=settings()
        with httpx.Client(base_url=config["api_url"],timeout=config["request_timeout_seconds"],trust_env=False) as client:
            assert client.get("/evaluation").content==(run/"report.html").read_bytes()
            assert client.get("/evaluation/summary").json()==summary
            post=lambda body:client.post("/predict",json=body).raise_for_status().json()
            vector=lambda result:np.array([result["probabilities"][option] for option in CATEGORIES.values()])
            examples=[records["test"][0]]+[r for r in records["test"] if r["target"]=="general"][:5]
            saved=[]
            for r in examples:
                body=request_for(r)|{"task_profile":NAME}
                if summary["profile_name"]:
                    body["calibration_profile"]=CALIBRATION_NAME
                initial,result=post(body),post(body)
                assert result["metadata"]["task_profile_fingerprint"]==task_fingerprint
                assert result["metadata"]["calibrated"]==bool(summary["profile_name"])
                c=result["component_probabilities"]
                assert c["reviewed"]==needs_review(c["base_raw"],profile["threshold"],profile["component_temperature"])
                expected_raw=combine(c["base_raw"],c["focused_raw"],profile["weight"],profile["minimum_probability"]) if c["reviewed"] else c["base_raw"]
                actual_raw=[result.get("raw_probabilities",result["probabilities"])[option] for option in CATEGORIES.values()]
                assert np.allclose(expected_raw,actual_raw,atol=1e-12,rtol=1e-12)
                assert result["metadata"]["usage"]["completion_tokens"]==(2 if c["reviewed"] else 1)
                if summary["profile_name"]:
                    expected=temperature_scale([result["raw_probabilities"][option] for option in CATEGORIES.values()],selection["selected_temperature"])
                    assert np.allclose(expected,vector(result),atol=1e-12,rtol=1e-12)
                changed=post(body|{"options":list(reversed(body["options"]))})
                assert changed["prediction"]==result["prediction"] and np.allclose(vector(changed),vector(result),atol=1e-6,rtol=0)
                saved.append({"request":body,"response":result,"order_max_delta":float(np.max(abs(vector(result)-vector(changed)))),
                              "initial_to_warm_max_probability_delta":float(np.max(abs(vector(result)-vector(initial))))})
            assert client.post("/predict",json=body|{"calibration_profile":"massive_scenario_es_v2"}).status_code==422
            if summary["profile_name"]:
                assert client.post("/predict",json=body|{"task_profile":"massive_es_v2"}).status_code==422
            generic=request_for(records["test"][0])
            assert post(generic)["metadata"]["task_profile"] is None
            assert post(generic|{"task_profile":"massive_es_v2","calibration_profile":"massive_scenario_es_v2"})["metadata"]["calibrated"]
            save_json(run/"live-examples.json",saved)
            live_result={"passed":True,"order_audit_messages":len(saved),"scoped_calibration_verified":True,"v2_and_generic_preserved":True,
                         "initial_to_warm_max_probability_delta":max(r["initial_to_warm_max_probability_delta"] for r in saved)}
            checks.append("live_calibration_scope_option_order_and_v2_preservation")
    result={"passed":True,"run_id":run.name,"checks":checks,"unit_tests_passed":test_count,"live":live_result,
            "package_versions":{name:importlib.metadata.version(name) for name in ("scikit-learn","joblib","numpy","scipy","httpx")}}
    save_json(run/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    verify(parser.parse_args().live)
