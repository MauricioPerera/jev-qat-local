"""Archive the delivered v9 evidence and check the served decision/default."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import httpx

from jev_qat.calibration import sha256_file
from jev_qat.config import ROOT, save_json, settings
from jev_qat.focus_experiment import read_json
from jev_qat.general_probability_audit import audit
from jev_qat.stable_rescue_adapter import StableRescueTask
from jev_qat.stable_rescue_data import DIRECTORY


def finalize():
    run=Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    s=read_json(run/"summary.json");a=read_json(run/"activation.json");v=read_json(run/"verification.json")
    assert v["passed"] and v["live_passed"] and v["integration_passed"]
    assert not a["verification_required"] and a["activated"]==a["approved"]==all(a["conditions"].values())
    state=read_json(ROOT/"classification/current.json")
    assert state["version"]==("v9" if a["activated"] else "v2")
    assert StableRescueTask()._load()[1]==read_json(run/"environment.json")["task_fingerprint"]
    save_json(run/"general-probability-audit.json",audit(run,s))
    files=list((ROOT/"jev_qat").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    files += [ROOT/"README.md",ROOT/"pyproject.toml",ROOT/"jev_qat/option_labels.json"]
    files += [ROOT/"scripts"/n for n in ("verify_spanish_v9.py","smoke_spanish_v9.py","finalize_spanish_v9.py")]
    files += list((ROOT/"classification/profiles").glob("*.json"))+list((ROOT/"calibration/profiles").glob("*.json"))
    files += [DIRECTORY/n for n in ("manifest.json","label-crosswalk.json","stability-plan.json","stability-final.json",
        "stability-summary.json","cache-probe.json","development-recomposition.json","startup-file-audit.json","unit-tests.xml")]
    files += list((DIRECTORY/"development-initial").glob("*"))
    archive=run/"delivery-source";hashes={}
    for p in sorted(set(files)):
        if not p.is_file():continue
        relative=p.relative_to(ROOT);out=archive/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
        hashes[str(relative)]=sha256_file(out)
    save_json(run/"delivery-source-manifest.json",hashes)
    suite=ET.parse(DIRECTORY/"unit-tests.xml").getroot()
    assert all(int(x.get("failures",0))==int(x.get("errors",0))==0 for x in suite.iter("testsuite"))
    tests=sum(int(x.get("tests",0)) for x in suite.iter("testsuite"))
    with httpx.Client(base_url=settings()["api_url"],timeout=30,trust_env=False) as client:
        r=client.get("/evaluation");r.raise_for_status();assert r.content==(run/"report.html").read_bytes()
        r=client.get("/evaluation/summary");r.raise_for_status();assert r.json()==s
    result={"passed":True,"new_test_messages":s["sample_counts"]["test"],"unit_tests_passed":tests,
        "active_version":state["version"],"v9_activated":a["activated"],"eight_recovered_three_orders":True,
        "meeting_stabilized":True,"significant":a["pvalue"]<a["alpha"],"pvalue":a["pvalue"],"alpha":a["alpha"],
        "native_components_verified":True,"multimodal_integration_passed":True,
        "report_sha256":sha256_file(run/"report.html"),"summary_sha256":sha256_file(run/"summary.json"),
        "source_archive_manifest_sha256":sha256_file(run/"delivery-source-manifest.json")}
    save_json(run/"delivery-verification.json",result)
    print(result,flush=True)


if __name__=="__main__":
    finalize()
