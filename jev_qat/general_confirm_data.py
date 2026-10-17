"""Verified original labels, explicit ontology mappings and fresh grouped holdouts."""
from collections import Counter
import gzip
import json
import random
import unicodedata
import zipfile

from .calibration import sha256_file
from .config import ROOT, save_json
from .eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, digest, disjoint_sources, text_group, stratified_sample
from .focus_experiment import read_json, write_records
from .general_confirm_development import DIRECTORY, prepare as development, lines
from .general_data import record

MULTI_REV = "cde8e2537d7ac06212ef4b25df2cf3084f26ec0f"
MTOP_REV = "9a200bf94b01afa0438a39c54f6a3aceaf78856f"
ACTS = ("affirm","deny","dont_know","acknowledge","greet","end_call","thank")


def family(text):
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
                           for c in unicodedata.normalize("NFKC",text).casefold()).split())


def category(intent):
    if intent.endswith("_ALARM"):
        return "alarm"
    if intent in ("PLAY_MUSIC","PLAY_MEDIA","STOP_MUSIC","PREVIOUS_TRACK_MUSIC","SKIP_TRACK_MUSIC","REWIND_MUSIC","REPLAY_MUSIC","PAUSE_MUSIC","RESUME_MUSIC","FAST_FORWARD_MUSIC"):
        return "play"
    if intent.endswith("_MUSIC"):
        return "music"
    if intent.endswith("_NEWS"):
        return "news"
    if intent == "GET_WEATHER":
        return "weather"
    if intent.endswith("_REMINDER") or intent.startswith(("UPDATE_REMINDER_","GET_REMINDER_")):
        return "calendar"
    if intent.endswith("_RECIPES"):
        return "cooking"
    # Calling, messaging, event discovery and timers have ambiguous crosswalks.
    return None


def prepare():
    path = DIRECTORY/"manifest.json"
    if path.exists():
        m = read_json(path)
        for info in m["splits"].values():
            assert sha256_file(DIRECTORY/info["file"]) == info["sha256"]
        return m
    selection = development()
    exposure = sorted(p for p in (ROOT/"evaluation").rglob("*.jsonl")
                      if DIRECTORY not in p.parents and "source" not in p.parts and "delivery-source" not in p.parts)
    used_ids,used_groups,used_families = set(),set(),set()
    for p in exposure:
        for r in lines(p):
            if isinstance(r,dict):
                used_ids.add(r.get("id"))
                if r.get("state"):
                    used_groups.add(text_group(r["state"]));used_families.add(family(r["state"]))
    for r in lines(DIRECTORY/"development.jsonl"):
        used_groups.add(r["group_id"]);used_families.add(family(r["state"]))
    original = zipfile.ZipFile(DIRECTORY/"source/mtop/official.zip")
    mtop,source_hashes,mappings = {},{},{}
    seen = set(used_families)
    for split,original_split in (("test","test"),("validation","eval")):
        p = DIRECTORY/"source/mtop"/f"{split}.jsonl"
        source_hashes[str(p.relative_to(DIRECTORY))] = sha256_file(p)
        gold = {str(row[0]):row for row in (line.split("\t") for line in original.read(f"mtop/es/{original_split}.txt").decode().splitlines())}
        mtop[split] = []
        for r in lines(p):
            g = gold[str(r["id"])]
            assert (r["text"],r["label_text"]) == (g[3],g[1].removeprefix("IN:"))
            label = category(r["label_text"])
            if not label or family(r["text"]) in seen:
                continue
            seen.add(family(r["text"]))
            mappings[r["label_text"]] = label
            mtop[split].append({"id":f"mtop:es:{split}:{r['id']}","source_id":str(r["id"]),"source_split":split,
                "state":r["text"],"group_id":text_group(r["text"]),"family_id":digest(family(r["text"])),"label":label,"target":label,
                "source_intents":[r["label_text"]],"question":QUESTION,"options":list(CATEGORIES.values()),"option_labels":list(CATEGORIES),
                "modality":"text","dataset":"MTOP","annotation_source":"Original MTOP human intent, text/intent/id checked against official Meta archive; deterministic bounded crosswalk","verified":True})
    general = []
    for p in sorted((DIRECTORY/"source/multi3nlu").rglob("fold*.json")):
        source_hashes[str(p.relative_to(DIRECTORY))] = sha256_file(p)
        for r in read_json(p):
            if not r.get("intents") or not set(r["intents"]) <= set(ACTS) or r.get("slots") or family(r["text"]) in seen:
                continue
            assert r["lang"] == "es"
            seen.add(family(r["text"]))
            general.append({"id":"multi3nlu:es:"+r["uid"],"source_id":r["uid"],"source_split":"published_folds",
                "state":r["text"],"group_id":text_group(r["text"]),"family_id":digest(family(r["text"])),"label":"general","target":"general",
                "source_intents":r["intents"],"question":QUESTION,"options":list(CATEGORIES.values()),"option_labels":list(CATEGORIES),"modality":"text",
                "dataset":"Multi3NLU++","annotation_source":"Original human dialogue-act labels, professional Spanish translation and documented QA; only pure conversational acts without task slots","verified":True})
    if len(general) < 140:
        raise ValueError(f"Need >=140 fresh original-label General messages, found {len(general)}")
    random.Random(2026101801).shuffle(general)
    partitions = {"calibration":general[:30],"validation":general[30:60],"test":general[60:140]}
    pool = mtop["validation"]
    for offset,split in enumerate(("calibration","validation")):
        chosen = stratified_sample(pool,120,2026101810+offset)
        ids = {r["id"] for r in chosen};pool = [r for r in pool if r["id"] not in ids]
        partitions[split] += chosen
    partitions["test"] += stratified_sample(mtop["test"],250,2026101812)
    # The final untouched MASSIVE validation examples retain the original labels.
    sources = {}
    for split,sha in SOURCE_HASHES.items():
        p = ROOT/"evaluation/massive-es-v1/source"/f"{split}-es.json.gz"
        assert sha256_file(p) == sha
        sources[split] = [json.loads(x) for x in gzip.decompress(p.read_bytes()).decode("utf-8").splitlines()]
    clean,_ = disjoint_sources(sources)
    unused = [record(r,"validation") for r in clean["validation"]
              if f"massive:es:validation:{r['id']}" not in used_ids and text_group(r["text"]) not in used_groups and family(r["text"]) not in seen]
    for r in unused:
        r.update(family_id=digest(family(r["state"])),dataset="MASSIVE",annotation_source="Unchanged original human scenario label",verified=True)
    partitions["test"] += unused
    for offset,(split,rows) in enumerate(partitions.items()):
        random.Random(2026101820+offset).shuffle(rows)
        assert len({r["id"] for r in rows}) == len(rows) and len({r["family_id"] for r in rows}) == len(rows)
        for other in list(partitions)[offset+1:]:
            assert not {r["family_id"] for r in rows} & {r["family_id"] for r in partitions[other]}
    splits = {s:write_records(DIRECTORY/f"{s}.jsonl",r) for s,r in partitions.items()}
    for s,rr in partitions.items():
        splits[s]["datasets"] = dict(Counter(r["dataset"] for r in rr));splits[s]["labels"] = dict(Counter(r["target"] for r in rr))
    crosswalk = {"mtop":mappings,"multi3nlu_general_acts":list(ACTS),"multi3nlu_requires_no_slots":True,
                 "excluded_multi3nlu":"Any task intent, missing labels, slots, or repeat (ambiguous with audio)",
                 "excluded_mtop":"Calling, messages, event discovery, timers and knowledge about contacts lack an unambiguous MASSIVE crosswalk"}
    save_json(DIRECTORY/"label-crosswalk.json",crosswalk)
    manifest = {"schema_version":8,"dataset":"Spanish original-label challenge: Multi3NLU++ conversational acts + MTOP specialized requests + untouched MASSIVE",
        "splits":splits,"sources_sha256":source_hashes | {"source/mtop/official.zip":sha256_file(DIRECTORY/"source/mtop/official.zip")},
        "revisions":{"multi3nlu":MULTI_REV,"mtop_mirror":MTOP_REV},"original_label_verification":True,"new_llm_labels_used":False,
        "crosswalk_sha256":sha256_file(DIRECTORY/"label-crosswalk.json"),"available_fresh_general":len(general),"new_massive_count":len(unused),
        "prior_exposure_files":{str(p.relative_to(ROOT)):sha256_file(p) for p in exposure},
        "development_selection_sha256":sha256_file(DIRECTORY/"development-selection.json"),
        "protocol":{"seed":2026101801,"test_used_for_selection":False,
            "labels":"Original human labels retained; deterministic and bounded 18-category crosswalk frozen before scores. No model-generated references. MTOP rows individually checked against official archive. Multi3NLU professional translation and verification documented by authors; not a new human adjudication of our crosswalk.",
            "mapping_scope":"General test is enriched with pure conversational dialogue acts, not the entire distribution of open General requests. These examples do not establish performance for advice or creative tasks; eight old recovered cases remain mandatory regressions.",
            "sources":"https://arxiv.org/html/2212.10455 ; https://aclanthology.org/2021.eacl-main.257/ ; https://dl.fbaipublicfiles.com/mtop/mtop.zip",
            "independence":"Fresh IDs, normalized texts and punctuation-insensitive families outside previous banks/evaluations and all development. Independence from model pretraining unknown. Public Spanish translations and human-authored examples are not native production traffic.",
            "development":"1132 old original-human-label MASSIVE records; old v6 and v7 tests explicitly retired. No provisional OASST labels used. 63 native confirmations, 60 composition candidates; choose three corrected domain cases, eight preserved, max development accuracy then lower weight and higher thresholds.",
            "calibration":"150 new messages fit scalar temperature; 150 other new messages validate NLL improvement >.0001. Freeze before test inference.",
            "activation_gate":"Eight recovered cases and all three false-positive domain corrections pass live; verified original labels and frozen crosswalk; >=50 General test support; calibration validated; General F1 gain >=.03, precision >=v2, accuracy >v2, macro F1 >=v2, one-sided exact McNemar p<.05, test NLL <=v2; no tuning on new test."}}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(path,manifest)
    print(json.dumps({"new_counts":{s:i["count"] for s,i in splits.items()},"general_available":len(general),"massive":len(unused)}),flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
