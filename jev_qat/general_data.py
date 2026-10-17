"""Freeze new holdouts before expanding general training, with explicit retirement."""
from collections import Counter
import gzip
import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from .calibration import sha256_file
from .config import ROOT, save_json
from .eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, SEED, digest, disjoint_sources, load_records, stratified_sample, text_group
from .focus_experiment import V1,V2,DIRECTORY as V3,read_json,write_records
from .selective_experiment import DIRECTORY as V4
from .general_task import NAME,VERSION,PROFILE_PATH

DIRECTORY = ROOT / "evaluation/massive-es-v5"
BANK_PATH = ROOT / "classification/massive-es-v5-bank.joblib"


def record(row, official):
    return {"id":f"massive:es:{official}:{row['id']}","source_id":str(row["id"]),"source_split":official,
            "group_id":text_group(row["text"]),"state":row["text"],"question":QUESTION,"options":list(CATEGORIES.values()),
            "option_labels":list(CATEGORIES),"target":row["label"],"modality":"text"}


def prepare():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    manifest_path = DIRECTORY/"manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for info in manifest["splits"].values():
            assert sha256_file(DIRECTORY/info["file"])==info["sha256"]
        assert sha256_file(BANK_PATH)==manifest["bank_sha256"]
        return manifest
    sources = {}
    for split,sha in SOURCE_HASHES.items():
        path = V1/"source"/f"{split}-es.json.gz"
        assert sha256_file(path)==sha
        sources[split]=[json.loads(line) for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()]
    clean,deduplication = disjoint_sources(sources)
    previous = {r["id"]:r for d in (V1,V2,V3,V4) for s in ("calibration","validation","test") for r in load_records(d,s)}
    previous.update({r["id"]:r for r in load_records(V3,"development")})
    used_ids = {r["source_id"] for r in previous.values()}
    used_groups = {r["group_id"] for r in previous.values()}
    development = load_records(V3,"development")+load_records(V4,"validation")
    dev_ids = {r["source_id"] for r in development}
    old_validation = {r["source_id"] for r in previous.values() if r["source_split"]=="validation"}
    training = [record(r,"train") for r in clean["train"]]
    training += [record(r,"validation") for r in clean["validation"] if str(r["id"]) in old_validation-dev_ids]
    retired_general = [record(r,"test") for r in clean["test"] if str(r["id"]) in used_ids and r["label"]=="general"]
    training += retired_general
    unused = {s:[r for r in rows if str(r["id"]) not in used_ids and text_group(r["text"]) not in used_groups] for s,rows in clean.items()}
    calibration = stratified_sample(unused["validation"],250,SEED+501)
    cal_ids={r["id"] for r in calibration}
    validation = stratified_sample([r for r in unused["validation"] if r["id"] not in cal_ids],150,SEED+502)
    general = [r for r in unused["test"] if r["label"]=="general"]
    if len(general)<50:
        raise ValueError("Insufficient untouched general test messages")
    test = stratified_sample(general,50,SEED+503)+stratified_sample([r for r in unused["test"] if r["label"]!="general"],450,SEED+504)
    import random
    random.Random(SEED+505).shuffle(test)
    partitions = {"training":training,"development":development,"calibration":[record(r,"validation") for r in calibration],
                  "validation":[record(r,"validation") for r in validation],"test":[record(r,"test") for r in test]}
    for i,(a,first) in enumerate(partitions.items()):
        for b,second in list(partitions.items())[i+1:]:
            for key in ("source_id","group_id"):
                assert not {r[key] for r in first}&{r[key] for r in second},(a,b,key)
    splits={s:write_records(DIRECTORY/f"{s}.jsonl",rows) for s,rows in partitions.items()}
    vectorizer = FeatureUnion([
        ("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents="unicode",dtype=np.float32,max_features=70000)),
        ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,strip_accents="unicode",dtype=np.float32,max_features=90000)),
    ],transformer_weights={"word":.65,"char":.35})
    matrix=normalize(vectorizer.fit_transform([r["state"] for r in training]))
    joblib.dump({"version":VERSION,"records":training,"vectorizer":vectorizer,"matrix":matrix},BANK_PATH,compress=3)
    profile={"name":NAME,"version":VERSION,"bank_file":str(BANK_PATH.relative_to(ROOT)),"bank_sha256":sha256_file(BANK_PATH),
             "training_count":len(training),"general_count":sum(r["target"]=="general" for r in training),"neighbors":6,"general_minimum":2}
    save_json(PROFILE_PATH,profile)
    manifest={"schema_version":5,"dataset":read_json(V1/"manifest.json")["dataset"],"splits":splits,"deduplication":deduplication,
              "bank_file":profile["bank_file"],"bank_sha256":profile["bank_sha256"],"general_training_before":585,"general_training_after":profile["general_count"],
              "retired_test_general_count":len(retired_general),"retired_test_general_ids":[r["id"] for r in retired_general],
              "prior_manifests_sha256":{d.name:sha256_file(d/"manifest.json") for d in (V1,V2,V3,V4)},
              "protocol":{"seed":SEED+500,"training":"All clean official train + previously used validation outside development + 138 retired general test examples. Past tests are explicitly retired and may not measure v5.",
                          "development":"450 old official validation messages, outside training; v3 development 300 + v4 validation 150. Baseline v2 evidence is already available.",
                          "retrieval":"Six nearest diverse training examples (max two per category), plus nearest general examples until two are present, at most eight total. Exact normalized query matches excluded.",
                          "development_gate":"General F1 gain >=.03; general precision >=v2-.03; overall accuracy and macro F1 >=v2. One predeclared retrieval candidate.",
                          "calibration":"250 untouched proportional validation messages fit scalar temperature; 150 other untouched validation messages must improve NLL by >.0001. Temperature frozen before test inference.",
                          "test":"500 untouched official test messages: all 50 remaining general examples, 450 proportional from other 17 categories. General enriched to 10%; some other classes have very small remaining support.",
                          "activation_gate":"Calibration validation approved; general F1 gain >=.03 and general precision >=v2; overall accuracy >v2; macro F1 >=v2; one-sided exact McNemar p<.05; test NLL <=v2. No retuning after test.",
                          "test_used_for_selection":False}}
    manifest["manifest_fingerprint"]=digest(manifest)
    save_json(manifest_path,manifest)
    print(json.dumps({"training":len(training),"general":profile["general_count"],"retired_general":len(retired_general),"holdouts":{s:len(v) for s,v in partitions.items() if s!="training"}}),flush=True)
    return manifest
