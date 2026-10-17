"""Independent source, score-composition, probability and promotion audit."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import httpx
import numpy as np
from scipy.stats import binomtest

from jev_qat.calibration import sha256_file
from jev_qat.config import ROOT, save_json, settings
from jev_qat.eval_data import CATEGORIES, digest
from jev_qat.eval_metrics import metrics, scaled_array
from jev_qat.evaluate import arrays, request_for
from jev_qat.focus_experiment import read_json
from jev_qat.focus_run import paired_comparison
from jev_qat.general_confirm_adapter import ConfirmedTask, NAME, VERSION
from jev_qat.general_confirm_task import NAME as CONFIRM_NAME, VERSION as CONFIRM_VERSION
from jev_qat.general_precision_math import general_evidence
from jev_qat.general_precision_task import PrecisionTask
from jev_qat.general_task import GeneralTask, VERSION as FOCUSED_VERSION
from jev_qat.spanish_task import SpanishTask, VERSION as BASE_VERSION
from jev_qat.public_eval_data import DIRECTORY, ORIGINAL, family, frozen_profiles, lines, prepare
from jev_qat.public_eval_run import conditions


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
        assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (a, b)
    else:
        assert a == b


def independent_proposal(base, focused, cfg, similarity):
    p = np.array(base, dtype=float)
    g = list(CATEGORIES).index("general")
    scores = np.log(np.clip(p, 1e-300, 1)) / cfg["component_temperature"]
    scores = np.exp(scores - scores.max()); scores /= scores.sum()
    review = np.argmax(p) != g and scores.max() <= cfg["threshold"]
    domain = list(CATEGORIES)[int(np.argmax(p))]
    review = review and (domain not in cfg["protected_domains"] or similarity >= cfg["minimum_general_similarity"])
    if not review or focused is None or np.argmax(focused) != g or focused[g] < cfg["minimum_probability"]:
        return p, review
    new = max(p[g], (1 - cfg["weight"]) * p[g] + cfg["weight"] * focused[g])
    p *= (1 - new) / (1 - p[g]); p[g] = new
    return p / p.sum(), review


def independent_confirm(proposed, confirmed, cfg):
    p = np.array(proposed, dtype=float)
    if confirmed is None:
        return p
    winner = list(CATEGORIES)[int(np.argmax(confirmed))]
    threshold = cfg["minimum_confirmation_probability"].get(winner, cfg["minimum_confirmation_probability"]["default"])
    if np.argmax(p) != list(CATEGORIES).index("general") or max(confirmed) < threshold:
        return p
    p = (1 - cfg["weight"]) * p + cfg["weight"] * np.array(confirmed)
    return p / p.sum()


def audit_api(result, cfg, fingerprint, temperature):
    assert result["metadata"]["task_profile_fingerprint"] == fingerprint
    assert result["metadata"]["calibrated"]
    comp = result["component_probabilities"]
    raw = independent_confirm(comp["v7_raw"], comp["confirmation_raw"], cfg)
    expected = scaled_array(np.array([raw]), temperature)[0]
    got = [result["probabilities"][v] for v in CATEGORIES.values()]
    assert np.allclose(expected, got, rtol=0, atol=1e-10)
    assert result["prediction"] == list(CATEGORIES.values())[int(expected.argmax())]
    assert result["metadata"]["llm_calls"] == result["metadata"]["usage"]["completion_tokens"]


def verify(live=False):
    m = prepare()
    copy = dict(m); fp = copy.pop("manifest_fingerprint"); assert digest(copy) == fp
    run = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    env = read_json(run / "environment.json")
    assert run.name == digest({"manifest": fp, "environment": env})[:16]
    assert frozen_profiles() == m["frozen_profiles"] == env["frozen_profiles"]
    if "preflight_regressions_sha256" in env:
        assert sha256_file(DIRECTORY / "preflight-regressions.json") == env["preflight_regressions_sha256"] == sha256_file(run / "regressions.json")
    for group in ("code", "follow_up_code"):
        for name, sha in env[group].items():
            assert sha256_file(ROOT / "jev_qat" / name) == sha
            if group == "follow_up_code":
                assert sha256_file(run / "source" / name) == sha
    for name, sha in env["files"].items():
        assert sha256_file(settings()[name]) == sha
    for name, sha in m["source_files"].items():
        assert sha256_file(ROOT / name) == sha
    for filename, key in (("label-crosswalk.json", "crosswalk_sha256"), ("label-provenance.json", "label_provenance_sha256")):
        assert sha256_file(DIRECTORY / filename) == m[key]
    assert hashlib.file_digest((DIRECTORY / "source/dihana/official.tgz").open("rb"), "md5").hexdigest() == "a9ca9324bccf04093f3115b7ac10ea12"
    old_families, old_ids = set(), set()
    for name, sha in m["prior_exposure_files"].items():
        p = ROOT / name; assert sha256_file(p) == sha
        for row in lines(p):
            if isinstance(row, dict):
                old_ids.add(row.get("id"))
                if row.get("state"):
                    old_families.add(family(row["state"]))
    rows = lines(DIRECTORY / "test.jsonl")
    assert len(rows) == m["test"]["count"] == 1283
    assert dict(Counter(r["target"] for r in rows)) == m["counts"]["labels"]
    assert len({r["family_id"] for r in rows}) == len(rows)
    assert not {r["id"] for r in rows}.intersection(old_ids)
    assert not {family(r["state"]) for r in rows}.intersection(old_families)
    dihana = [r for r in rows if r["dataset"] == "DIHANA"]
    assert len(dihana) == len({r["source_speaker"] for r in dihana}) == 83
    with tarfile.open(DIRECTORY / "source/dihana/official.tgz") as archive:
        for row in dihana:
            ann = row["original_annotation"]
            data = archive.extractfile(ann["archive_member"]).read().decode("latin1").splitlines()
            line = data[ann["annotation_line"] - 1]
            assert " ".join(line.split()) == " ".join(row["state"].split()) + " (U:Cierre:Nil:Nil) pal: " + " - ".join(str(x) for x in ann["word_range"])
            assert row["target"] == "general" and row["source_intents"] == ["Cierre"]
    crosswalk = read_json(DIRECTORY / "label-crosswalk.json")["mtop"]
    with zipfile.ZipFile(ORIGINAL / "source/mtop/official.zip") as archive:
        gold = {split: {r[0]: r for r in (x.split("\t") for x in archive.read(f"mtop/es/{original}.txt").decode().splitlines())}
                for split, original in (("test", "test"), ("validation", "eval"))}
    for row in rows:
        assert row["original_human_annotation"] and not row["crosswalk_new_human_adjudication"]
        assert row["family_id"] == digest(family(row["state"]))
        if row["dataset"] == "MTOP":
            g = gold[row["source_split"]][row["source_id"]]
            assert g[3] == row["state"] and [g[1].removeprefix("IN:")] == row["source_intents"]
            assert crosswalk[row["source_intents"][0]] == row["target"]
    checks = ["All 1283 IDs and punctuation-insensitive families fresh; 83 distinct DIHANA participants; every selected original human label/text checked against original archive"]
    freeze = read_json(run / "pre-test-freeze.json")
    assert not freeze["test_predictions_existed"] and freeze["fixed_n"] == len(rows) and freeze["alpha"] == .025
    assert freeze["manifest_sha256"] == sha256_file(DIRECTORY / "manifest.json")
    assert freeze["environment_sha256"] == sha256_file(run / "environment.json")
    first = (run / "test-v2.jsonl").stat().st_birthtime
    assert datetime.fromisoformat(freeze["frozen_at"]).timestamp() <= first
    assert datetime.fromisoformat(m["created_at"]).timestamp() <= first
    assert not m["protocol"]["previous_test_used_in_significance"] and m["protocol"]["fixed_mtop_n"] == 1200
    cfg, fingerprint = ConfirmedTask()._load(); proposal_cfg, proposal_fp = PrecisionTask()._load()
    _, _, base_fp = SpanishTask()._load(); _, _, focus_fp = GeneralTask()._load()
    sources = {n: lines(run / f"test-{n}.jsonl") for n in ("v2", "focus", "confirmation", "proposal", "predictions")}
    assert len(sources["v2"]) == len(sources["proposal"]) == len(sources["predictions"]) == len(rows)
    focused = {r["id"]: r for r in sources["focus"]}; confirmed = {r["id"]: r for r in sources["confirmation"]}
    need_focus, need_confirm = set(), set()
    runtime = set()
    for row, b, p, out in zip(rows, sources["v2"], sources["proposal"], sources["predictions"]):
        assert row["id"] == b["id"] == p["id"] == out["id"] and row["target"] == b["target"] == p["target"] == out["target"]
        assert b["request_sha256"] == digest(request_for(row) | {"task_profile": "massive_es_v2"})
        assert b["metadata"]["prompt_version"] == BASE_VERSION and b["metadata"]["task_profile_fingerprint"] == base_fp
        f, c = focused.get(row["id"]), confirmed.get(row["id"])
        similarity = general_evidence(row["state"])["general_similarity"]
        proposal, required = independent_proposal(b["probabilities"], f["probabilities"] if f else None, proposal_cfg, similarity)
        if required:
            need_focus.add(row["id"])
        if f:
            assert f["request_sha256"] == digest(request_for(row) | {"task_profile": "massive_es_v5"})
            assert f["metadata"]["prompt_version"] == FOCUSED_VERSION and f["metadata"]["task_profile_fingerprint"] == focus_fp
        assert np.allclose(proposal, p["probabilities"], atol=1e-12, rtol=0) and p["derived"]
        assert p["metadata"]["task_profile_fingerprint"] == proposal_fp
        if proposal.argmax() == list(CATEGORIES).index("general"):
            need_confirm.add(row["id"])
        if c:
            assert c["request_sha256"] == digest(request_for(row) | {"task_profile": CONFIRM_NAME})
            assert c["metadata"]["prompt_version"] == CONFIRM_VERSION and c["metadata"]["task_profile_fingerprint"] == cfg["confirmation_fingerprint"]
        expected = independent_confirm(proposal, c["probabilities"] if c else None, cfg)
        assert np.allclose(expected, out["probabilities"], atol=1e-12, rtol=0)
        assert out["derived"] and out["metadata"]["task_profile_fingerprint"] == fingerprint
        assert out["metadata"]["llm_calls"] == 1 + bool(f) + bool(c)
        for native in [b] + ([f] if f else []) + ([c] if c else []):
            vector = np.array(native["probabilities"])
            assert vector.shape == (18,) and np.isfinite(vector).all() and (vector >= 0).all() and abs(vector.sum() - 1) < 1e-6
            assert native["metadata"]["calibrated"] is False
            runtime.add(native["metadata"]["runtime_fingerprint"])
    assert need_focus == set(focused) and need_confirm == set(confirmed) and len(runtime) == 1
    checks.append("Every native request, runtime and prompt fingerprint checked; independently recomputed all v7/v8 compositions and frozen temperatures; no extra/missing component calls")
    s = read_json(run / "summary.json")
    raw, y = arrays(sources["predictions"]); bp, by = arrays(sources["v2"]); assert np.array_equal(y, by)
    before = scaled_array(bp, m["frozen_profiles"]["temperatures"]["v2"])
    after = scaled_array(raw, m["frozen_profiles"]["temperatures"]["v8"])
    bm, cm = metrics(before, y, list(CATEGORIES)), metrics(after, y, list(CATEGORIES))
    equal(bm, s["baseline"]["calibrated"]); equal(cm, s["test"]["calibrated"]); equal(metrics(raw, y, list(CATEGORIES)), s["test"]["raw"])
    paired = paired_comparison(before, after, y); equal(paired, s["paired_comparison"])
    reg = read_json(run / "regressions.json")
    old = {r["id"]: r for r in lines(ORIGINAL / "development.jsonl")}
    protected_ids = read_json(ORIGINAL / "development-source-scores.json")["protected_ids"]
    assert [r["id"] for r in reg["cases"][:8]] == protected_ids and len(reg["cases"]) == 11
    for row in reg["cases"]:
        audit_api(row["result"], cfg, fingerprint, m["frozen_profiles"]["temperatures"]["v8"])
        assert row["passed"] == (row["result"]["prediction"] == CATEGORIES[old[row["id"]]["target"]])
        assert row["request_sha256"] == digest(request_for(old[row["id"]]) | {"task_profile": NAME, "calibration_profile": "massive_scenario_es_v8"})
    gates, p = conditions(bm, cm, paired, reg, m)
    a = read_json(run / "activation.json")
    assert a["conditions"] == gates and a["pvalue"] == p and a["alpha"] == .025
    fixed, broken = paired["fixed_errors"], paired["introduced_errors"]
    exact = float(binomtest(fixed, fixed + broken, .5, alternative="greater").pvalue) if fixed + broken else 1.
    assert exact == p and a["statistical_and_quality_gates_passed"] == all(gates.values())
    assert not s["previous_test_in_primary_analysis"] and not s["new_calibration_fitted"]
    for name, value in s["per_source"].items():
        ix = [i for i, row in enumerate(rows) if row["dataset"] == name]
        equal(value["baseline"], metrics(before[ix], y[ix], list(CATEGORIES)))
        equal(value["candidate"], metrics(after[ix], y[ix], list(CATEGORIES)))
    for name, sha in s["artifacts_sha256"].items():
        assert sha256_file(run / name) == sha
    suite = ET.parse(DIRECTORY / "unit-tests.xml").getroot()
    assert all(int(x.get("failures", 0)) == int(x.get("errors", 0)) == 0 for x in suite.iter("testsuite"))
    assert reg["eight_recovered_preserved"] == all(r["passed"] for r in reg["cases"][:8])
    assert reg["three_domain_errors_corrected"] == all(r["passed"] for r in reg["cases"][8:])
    checks.append("All metrics/intervals and one-sided exact p value reproduced; fixed N, alpha and freeze predate first score; original regression failures retained without retry")
    live_results = []
    if live:
        chosen = [next(row for row in rows if row["dataset"] == "DIHANA"), next(row for row in rows if row["dataset"] == "MTOP")]
        with httpx.Client(base_url=settings()["api_url"], timeout=settings()["request_timeout_seconds"], trust_env=False) as client:
            for row in chosen:
                response = client.post("/predict", json=request_for(row) | {"task_profile": NAME, "calibration_profile": "massive_scenario_es_v8"})
                response.raise_for_status(); result = response.json()
                audit_api(result, cfg, fingerprint, m["frozen_profiles"]["temperatures"]["v8"])
                live_results.append({"id": row["id"], "result": result})
    save_json(run / "verification.json", {"passed": True, "checks": checks,
        "live_composition_passed": bool(live), "live_examples": live_results,
        "regression_cases_checked": 11, "eight_recoveries_preserved": reg["eight_recovered_preserved"],
        "three_domain_corrections_preserved": reg["three_domain_errors_corrected"],
        "original_human_annotations": True, "new_human_crosswalk_adjudication": False})
    print(json.dumps({"passed": True, "groups": len(checks), "live": bool(live)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    verify(parser.parse_args().live)
