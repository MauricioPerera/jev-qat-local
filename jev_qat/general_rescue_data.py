"""Reuse unopened v5 holdouts for the development-selected rescue variant."""
import shutil
from pathlib import Path

from .calibration import sha256_file
from .config import ROOT,save_json
from .eval_data import digest
from .focus_experiment import read_json
from .general_data import DIRECTORY as V5
from .general_task import GeneralTask
from .general_rescue_task import NAME,VERSION,PROFILE_PATH
from .spanish_task import SpanishTask

DIRECTORY=ROOT/"evaluation/massive-es-v6"


def prepare():
    if (DIRECTORY/"manifest.json").exists():
        result=read_json(DIRECTORY/"manifest.json")
        for info in result["splits"].values():
            assert sha256_file(DIRECTORY/info["file"])==info["sha256"]
        return result
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    original=read_json(V5/"manifest.json")
    selection=read_json(DIRECTORY/"development-selection.json")
    chosen=selection["selected"]
    if not chosen:
        raise RuntimeError("No rescue candidate passed development")
    source=read_json(V5/"latest-run.json")
    assert read_json(Path(source["directory"])/"progress.json")["test_opened"] is False
    for info in original["splits"].values():
        shutil.copy2(V5/info["file"],DIRECTORY/info["file"])
    _,_,base_fp=SpanishTask()._load()
    _,_,focused_fp=GeneralTask()._load()
    profile={"name":NAME,"version":VERSION,"base_fingerprint":base_fp,"focused_fingerprint":focused_fp,
             "threshold":chosen["threshold"],"weight":chosen["weight"],"minimum_probability":chosen["minimum_probability"],
             "component_temperature":selection["component_temperature"],"selection_sha256":sha256_file(DIRECTORY/"development-selection.json")}
    save_json(PROFILE_PATH,profile)
    protocol=original["protocol"]|{
        "development":"450 old validation messages; raw expanded retrieval improved general but lost one overall correct answer. Rescue search uses only those cached development scores.",
        "development_gate":"36 rescue candidates: baseline confidence threshold .65/.8/.95/1, weight .3/.5/.7, focused probability floor 0/.95/.99. General F1 gain >=.03, precision >=v2-.03, accuracy >v2, macro F1 >=v2. Max accuracy, then general F1, fewer reviews, smaller weight.",
        "composition":"First use v2 raw Gemma scores. Review only non-general winners with calibrated v2 confidence below the selected threshold. Query the expanded general prompt; only a sufficiently confident general answer may raise the general probability. Preserve all other v2 probability ratios. New outer temperature calibrates the raw composition.",
        "shared_evidence":"Evaluation collects native v2 scores for all messages and native v5 scores only for review messages, then deterministically derives v6. Live API executes those same stages; derived records are explicitly marked.",
        "test":"Same 500 reserved v5 test records; v5 stopped after development and never queried calibration, validation or test. All 50 general remain untouched before this experiment."}
    manifest={k:v for k,v in original.items() if k not in ("manifest_fingerprint","schema_version","protocol")}
    manifest.update({"schema_version":6,"protocol":protocol,"source_v5_manifest_sha256":sha256_file(V5/"manifest.json"),
                     "development_selection_sha256":sha256_file(DIRECTORY/"development-selection.json")})
    manifest["manifest_fingerprint"]=digest(manifest)
    save_json(DIRECTORY/"manifest.json",manifest)
    return manifest
