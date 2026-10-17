"""Independently verify human-source labels, uncached components and v9 gates."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import httpx
import numpy as np
from scipy.stats import binomtest

from jev_qat.calibration import sha256_file, PROFILES
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, QUESTION, digest
from jev_qat.eval_metrics import metrics, fit_temperature, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.general_precision_math import general_evidence
from jev_qat.stable_rescue_adapter import StableRescueTask, NAME, VERSION, PROFILE
from jev_qat.stable_rescue_data import DIRECTORY, V8, EXPANDED, lines
from jev_qat.stable_rescue_task import StableNativeTask, FOCUS_NAME, CONFIRM_NAME, FOCUS_VERSION, CONFIRM_VERSION
from jev_qat.stable_rescue_run import conditions, CALIBRATION
from jev_qat.general_confirm_data import family
from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION


def equal(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            equal(a[k], b[k])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(b, float):
        assert math.isclose(a, b, abs_tol=1e-11, rel_tol=1e-11), (a, b)
    else:
        assert a == b


def independent(base, focused, confirmed, cfg, similarity):
    def ramp(v, a, b):
        x = min(1., max(0., (v-a)/(b-a)))
        return 3*x*x-2*x*x*x
    g = list(CATEGORIES).index("general"); q = list(CATEGORIES).index("qa")
    p = np.array(base, dtype=float)
    if p.argmax() == g:
        r = 1.
    elif focused is None or (list(CATEGORIES)[int(p.argmax())] in cfg["protected_domains"] and similarity < cfg["minimum_general_similarity"]):
        r = 0.
    else:
        r = ramp(focused[g], cfg["ramp_low"], cfg["ramp_high"])
        old = p[g]
        new = max(old, old + cfg["rescue_weight"]*r*(focused[g]-old))
        p *= (1-new)/(1-old); p[g] = new; p /= p.sum()
    if confirmed is not None:
        reliability = sum(ramp(v, cfg["qa_confirmation_low"] if i == q else cfg["confirmation_low"],
            cfg["qa_confirmation_high"] if i == q else cfg["confirmation_high"]) for i, v in enumerate(confirmed))
        w = cfg["confirmation_weight"]*r*min(1., reliability)
        p = (1-w)*p + w*np.array(confirmed); p /= p.sum()
    assert (confirmed is not None) == (r > 0)
    return p, r


def verify(live=False):
    m = read_json(DIRECTORY / "manifest.json")
    copy = dict(m); mf = copy.pop("manifest_fingerprint"); assert digest(copy) == mf
    run = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    env, s = read_json(run / "environment.json"), read_json(run / "summary.json")
    assert run.name == digest({"manifest": mf, "environment": env})[:16]
    for group in ("code", "v9_code"):
        for name, sha in env[group].items():
            assert sha256_file(ROOT / "jev_qat" / name) == sha == sha256_file(run / "source" / name)
    for name, sha in env["files"].items():
        assert sha256_file(settings()[name]) == sha
    for name, sha in env["runtime_libraries"].items():
        assert sha256_file(Path(settings()["runtime"]).parent / name) == sha
    assert sha256_file(PROFILE) == env["profile_sha256"]
    assert sha256_file(run / "stability-final.json") == sha256_file(DIRECTORY / "stability-final.json") == env["stability_sha256"]
    for name, sha in m["sources"].items():
        assert sha256_file(ROOT / name) == sha
    assert sha256_file(DIRECTORY / "label-crosswalk.json") == m["crosswalk_sha256"]
    crosswalk = read_json(DIRECTORY / "label-crosswalk.json")
    old_families, old_ids = set(), set()
    for name, sha in m["prior_exposure_files"].items():
        path = ROOT / name; assert sha256_file(path) == sha
        for r in lines(path):
            if isinstance(r, dict):
                old_ids.add(r.get("id"))
                if r.get("state"):
                    old_families.add(family(r["state"]))
    records = {}
    for split, info in m["splits"].items():
        assert sha256_file(DIRECTORY / info["file"]) == info["sha256"]
        records[split] = lines(DIRECTORY / info["file"])
        assert len(records[split]) == info["count"] and dict(Counter(r["target"] for r in records[split])) == info["class_counts"]
    test = records["test"]
    assert len(test) == 849 and len({family(r["state"]) for r in test}) == len(test)
    assert not {r["id"] for r in test} & old_ids and not {family(r["state"]) for r in test} & old_families
    assert not {family(r["state"]) for r in records["calibration"]} & {family(r["state"]) for r in records["validation"]}
    for split in ("calibration", "validation"):
        assert not {family(r["state"]) for r in records[split]} & {family(r["state"]) for r in test}
    original_multi = {}
    for p in (V8 / "source/multi3nlu").rglob("fold*.json"):
        original_multi.update({r["uid"]: r for r in read_json(p)})
    original_qa = {r["id"]: r for r in read_json(DIRECTORY / "source/mintaka/mintaka_test.json")}
    with zipfile.ZipFile(V8 / "source/mtop/official.zip") as archive:
        original_mtop = {sp: {r[0]: r for r in (x.split("\t") for x in archive.read(f"mtop/es/{origin}.txt").decode().splitlines())}
            for sp, origin in (("test", "test"), ("validation", "eval"))}
    groups = {}
    with tarfile.open(EXPANDED / "source/dihana/official.tgz") as archive:
        for split, rows in records.items():
            groups[split] = set()
            for r in rows:
                if r["dataset"] == "MTOP":
                    g = original_mtop[r["source_split"]][r["source_id"]]
                    assert g[3] == r["state"] and [g[1].removeprefix("IN:")] == r["source_intents"]
                    assert crosswalk["mtop"][r["source_intents"][0]] == r["target"]
                elif r["dataset"] == "Multi3NLU++":
                    g = original_multi[r["source_id"]]
                    assert g["text"] == r["state"] and g["intents"] == r["source_intents"] and g["lang"] == "es" and not g.get("slots")
                    assert set(g["intents"]) <= set(crosswalk["multi3nlu"]["general_acts"]) and r["target"] == "general"
                    groups[split].add(r["source_id"].rsplit("_", 1)[0])
                elif r["dataset"] == "Mintaka":
                    g = original_qa[r["source_id"]]
                    assert g["translations"]["es"] == r["state"] and g["category"] == r["source_category"]
                    assert g["category"] in crosswalk["mintaka"]["qa_topics"] and r["target"] == "qa"
                else:
                    assert r["dataset"] == "DIHANA" and split == "test" and r["target"] == "general"
                    a = r["original_annotation"]
                    line = archive.extractfile(a["archive_member"]).read().decode("latin1").splitlines()[a["annotation_line"]-1]
                    expected = r["state"] + " (" + a["annotation"] + ") pal: " + " - ".join(map(str, a["word_range"]))
                    assert " ".join(line.split()) == " ".join(expected.split())
                    assert a["annotation"].endswith(":Nil:Nil") and r["source_intents"][0] in crosswalk["dihana"]["general_acts"]
    assert not groups["calibration"] & groups["validation"] and not groups["test"] & (groups["calibration"] | groups["validation"])
    dg = [r for r in test if r["dataset"] == "DIHANA"]
    mg = [r for r in test if r["dataset"] == "Multi3NLU++"]
    assert len({r["source_speaker"] for r in dg}) == len(dg) == 24
    assert len({r["source_group"] for r in mg}) == len(mg) == 25
    assert not crosswalk["new_human_adjudication_of_crosswalk"]
    checks = ["All 849 fresh references matched to original human-labelled public sources; 600 MTOP, 200 Spanish Mintaka QA, 25 Multi3 fold groups and 24 DIHANA participants; calibration/test group separation and family disjointness verified"]
    cfg, fp = StableRescueTask()._load(); assert fp == env["task_fingerprint"]
    _, _, base_fp = SpanishTask()._load()
    kwargs = dict(state="", question=QUESTION, options=list(CATEGORIES.values()), modality="text")
    handler = StableNativeTask()
    native_fp = {name: handler.prepare(name, **kwargs)["fingerprint"] for name in (FOCUS_NAME, CONFIRM_NAME)}
    outputs, baselines = {}, {}
    runtime = set()
    for split, rows in records.items():
        base = lines(run / f"{split}-v2.jsonl"); out = lines(run / f"{split}-predictions.jsonl")
        focused = {r["id"]: r for r in lines(run / f"{split}-focus.jsonl")}
        confirmed = {r["id"]: r for r in lines(run / f"{split}-confirmation.jsonl")}
        assert len(base) == len(out) == len(rows)
        needs_focus, needs_confirm = set(), set()
        for r, b, result in zip(rows, base, out):
            assert r["id"] == b["id"] == result["id"] and r["target"] == b["target"] == result["target"]
            assert b["request_sha256"] == digest(request_for(r) | {"task_profile": "massive_es_v2"})
            assert b["metadata"]["task_profile_fingerprint"] == base_fp and b["metadata"]["prompt_version"] == BASE_VERSION
            if "imported_native_evidence" in b:
                assert split != "test" and sha256_file(b["imported_native_evidence"]["file"]) == b["imported_native_evidence"]["sha256"]
            f, c = focused.get(r["id"]), confirmed.get(r["id"])
            sim = general_evidence(r["state"])["general_similarity"]
            bv = np.array(b["probabilities"]); z = np.log(np.clip(bv,1e-300,1))/cfg["component_temperature"]; z=np.exp(z-z.max());z/=z.sum()
            winner = list(CATEGORIES)[int(bv.argmax())]
            review = winner != "general" and z.max() <= cfg["review_threshold"] and (winner not in cfg["protected_domains"] or sim >= cfg["minimum_general_similarity"])
            if review:
                needs_focus.add(r["id"])
            for comp, name, version in ((f, FOCUS_NAME, FOCUS_VERSION), (c, CONFIRM_NAME, CONFIRM_VERSION)):
                if comp:
                    assert comp["request_sha256"] == digest(request_for(r) | {"task_profile": name})
                    assert comp["metadata"]["task_profile_fingerprint"] == native_fp[name] and comp["metadata"]["prompt_version"] == version
                    assert comp["metadata"]["cache_prompt"] is False
            expected, strength = independent(b["probabilities"], f["probabilities"] if f else None, c["probabilities"] if c else None, cfg, sim)
            if strength > 0:
                needs_confirm.add(r["id"])
            assert np.allclose(expected, result["probabilities"], atol=1e-12, rtol=0)
            assert result["derived"] and result["metadata"]["task_profile_fingerprint"] == fp
            assert result["metadata"]["llm_calls"] == 1+bool(f)+bool(c)
            for native in [b] + ([f] if f else []) + ([c] if c else []):
                v=np.array(native["probabilities"]); assert v.shape == (18,) and np.isfinite(v).all() and v.min() >= 0 and abs(v.sum()-1)<1e-6
                assert native["metadata"]["calibrated"] is False
                runtime.add(native["metadata"]["runtime_fingerprint"])
        assert set(focused) == needs_focus and set(confirmed) == needs_confirm
        outputs[split], baselines[split] = out, base
    assert len(runtime) == 1
    checks.append("All native requests, fingerprints and explicit cache_prompt=false validated; independently recomputed every continuous rescue/confirmation vector and exact required call set")
    cp,cy=arrays(outputs["calibration"]); vp,vy=arrays(outputs["validation"])
    t=fit_temperature(cp,cy); sel=read_json(run/"selection.json")
    assert abs(t-sel["temperature"])<1e-9
    vr,vc=metrics(vp,vy,list(CATEGORIES)),metrics(scaled_array(vp,t),vy,list(CATEGORIES))
    equal(vr,sel["raw_validation"]);equal(vc,sel["candidate_validation"])
    validated=vc["negative_log_loss"]<vr["negative_log_loss"]-.0001
    assert validated == sel["validation_approved"] and sel["selected_temperature"] == (t if validated else 1.)
    freeze=read_json(run/"pre-test-freeze.json")
    assert not freeze["test_predictions_existed"] and freeze["test_n"]==849 and freeze["alpha"]==.025
    assert freeze["selection_sha256"]==sha256_file(run/"selection.json") and freeze["manifest_sha256"]==sha256_file(DIRECTORY/"manifest.json")
    assert freeze["environment_sha256"]==sha256_file(run/"environment.json")
    assert datetime.fromisoformat(freeze["frozen_at"]).timestamp() <= (run/"test-v2.jsonl").stat().st_birthtime
    raw,y=arrays(outputs["test"]);bp,by=arrays(baselines["test"]);assert np.array_equal(y,by)
    before=scaled_array(bp,s["baseline"]["temperature"]);after=scaled_array(raw,sel["selected_temperature"])
    bm,cm=metrics(before,y,list(CATEGORIES)),metrics(after,y,list(CATEGORIES))
    equal(bm,s["baseline"]["calibrated"]);equal(cm,s["test"]["calibrated"]);equal(metrics(raw,y,list(CATEGORIES)),s["test"]["raw"])
    paired=paired_comparison(before,after,y);equal(paired,s["paired_comparison"])
    stability=read_json(run/"stability-final.json");assert len(stability["results"])==33 and stability["all_passed"]
    plan=read_json(DIRECTORY/"stability-plan.json")["orders"]
    old_dev={r["id"]:r for r in lines(V8/"development.jsonl")}
    protected=read_json(V8/"development-source-scores.json")["protected_ids"]
    assert plan[0][:8]==protected and len(set(plan[0]))==11
    for order in (1,2,3):
        part=[r for r in stability["results"] if r["pass"]==order]
        assert [r["id"] for r in part]==plan[order-1] and all(r["passed"] for r in part)
        for r in part:
            assert r["target"]==old_dev[r["id"]]["target"]
            v=r["result"];comp=v["component_probabilities"]
            expected,_=independent(comp["base_raw"],comp["focused_raw"],comp["confirmation_raw"],cfg,(comp["training_evidence"] or {}).get("general_similarity",0.))
            assert v["prediction"]==CATEGORIES[r["target"]] and v["metadata"]["task_profile_fingerprint"]==fp
            assert np.allclose(expected,[v["probabilities"][o] for o in CATEGORIES.values()],atol=1e-12,rtol=0)
    gates,p=conditions(bm,cm,paired,True,validated,.025);a=read_json(run/"activation.json")
    assert a["conditions"]==gates and a["pvalue"]==p and a["statistical_and_quality_gates_passed"]==all(gates.values())
    fixed,broken=paired["fixed_errors"],paired["introduced_errors"]
    assert p==(float(binomtest(fixed,fixed+broken,.5,alternative="greater").pvalue) if fixed+broken else 1.)
    assert not s["previous_tests_in_significance"]
    for source,value in s["per_source"].items():
        ix=[i for i,r in enumerate(test) if r["dataset"]==source]
        equal(value["baseline"],metrics(before[ix],y[ix],list(CATEGORIES)));equal(value["candidate"],metrics(after[ix],y[ix],list(CATEGORIES)))
    for name,sha in s["artifacts_sha256"].items():assert sha256_file(run/name)==sha
    if validated:
        profile=read_json(PROFILES/f"{CALIBRATION}.json")
        assert profile==read_json(run/"calibration-profile.json") and profile["scope"]["task_profile_fingerprint"]==fp
    suite=ET.parse(DIRECTORY/"unit-tests.xml").getroot()
    assert all(int(x.get("errors",0))==int(x.get("failures",0))==0 for x in suite.iter("testsuite"))
    checks.append("Temperature refit and held-out validation, pre-test freeze, metrics/bootstrap/exact p reproduced; 24/24 protected and 9/9 domain checks passed across three fixed orders")
    live_results=[]
    if live:
        if not (run/"integration-results.json").exists():
            source=ROOT/"integration/jev-qat/integration-results.json"
            assert source.stat().st_mtime >= (run/"environment.json").stat().st_mtime
            shutil.copy2(source,run/"integration-results.json")
        integration=read_json(run/"integration-results.json")
        assert integration["passed"]
        chosen=[next(r for r in test if r["target"]=="general"),next(r for r in test if r["dataset"]=="Mintaka")]
        with httpx.Client(base_url=settings()["api_url"],timeout=180,trust_env=False) as client:
            for r in chosen:
                payload=request_for(r)|{"task_profile":NAME}
                if validated:payload["calibration_profile"]=CALIBRATION
                response=client.post("/predict",json=payload);response.raise_for_status();v=response.json();comp=v["component_probabilities"]
                expected,_=independent(comp["base_raw"],comp["focused_raw"],comp["confirmation_raw"],cfg,(comp["training_evidence"] or {}).get("general_similarity",0.))
                expected=scaled_array(np.array([expected]),sel["selected_temperature"])[0]
                assert np.allclose(expected,[v["probabilities"][o] for o in CATEGORIES.values()],atol=1e-10,rtol=0)
                assert v["metadata"]["task_profile_fingerprint"]==fp and v["metadata"]["calibrated"]==validated
                assert v["metadata"]["llm_calls"]==v["metadata"]["usage"]["completion_tokens"]
                live_results.append({"id":r["id"],"result":v})
            payload["options"]=list(reversed(payload["options"]));response=client.post("/predict",json=payload);response.raise_for_status();v=response.json()
            assert payload["options"][v["prediction_index"]]==v["prediction"]
            payload["question"]="Otra pregunta";assert client.post("/predict",json=payload).status_code==422
    save_json(run/"verification.json",{"passed":True,"checks":checks,"live_passed":bool(live),"live_examples":live_results,
        "integration_passed": bool(live),
        "original_human_labels":True,"new_human_crosswalk_adjudication":False,"eight_recoveries_all_three_orders":True})
    print(json.dumps({"passed":True,"checks":len(checks),"live":bool(live)}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--live",action="store_true")
    verify(parser.parse_args().live)
