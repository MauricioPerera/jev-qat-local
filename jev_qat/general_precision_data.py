"""Freeze new text groups and an explicit, provisionally annotated OASST cohort."""
from collections import Counter
import gzip
import json
import random
from pathlib import Path

import numpy as np

from .calibration import sha256_file
from .config import ROOT, save_json
from .eval_data import CATEGORIES, QUESTION, SOURCE_HASHES, digest, disjoint_sources, stratified_sample, text_group
from .general_data import record
from .general_precision_math import CONTEXTUAL_TOOLS, combine, general_evidence
from .general_precision_task import NAME, VERSION, PROFILE_PATH
from .general_rescue_task import RescueTask
from .eval_metrics import metrics
from .focus_experiment import read_json, write_records

DIRECTORY = ROOT / "evaluation/general-rescue-v7"
PREVIOUS = ROOT / "evaluation/massive-es-v6/runs/34d37f30edeb13d6"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def select():
    """Old v6 test is retired to development; every new v6 true rescue is guarded."""
    original, _ = RescueTask()._load()
    config = {k:v for k,v in original.items() if k not in ("name", "version", "selection_sha256")}
    config.update(name=NAME, version=VERSION, protected_domains=list(CONTEXTUAL_TOOLS))
    sets = {}
    for split in ("development", "test"):
        records = lines(ROOT / "evaluation/massive-es-v6" / f"{split}.jsonl")
        evidence = lines(PREVIOUS / f"{split}-predictions.jsonl")
        support = [general_evidence(r["state"]) for r in records]
        targets = np.array([list(CATEGORIES).index(r["target"]) for r in records])
        sets[split] = records, evidence, support, targets
    trials = []
    for minimum in (.10, .15, .20, .25, .30, .40):
        cfg = config | {"minimum_general_similarity":minimum}
        trial = {"minimum_general_similarity":minimum, "preserved_all_new_correct":True, "details":{}}
        for split, (records, evidence, support, targets) in sets.items():
            values, lost, changes = [], [], []
            for r, e, s in zip(records, evidence, support):
                c = e["component_probabilities"]
                base, focus = c["base_raw"], c["focused_raw"]
                value = combine(base, focus, cfg, s["general_similarity"]) if focus else base
                values.append(value)
                before, old, new = (int(np.argmax(v)) for v in (base, e["probabilities"], value))
                target = list(CATEGORIES).index(r["target"])
                if old == target and before != target and new != target:
                    lost.append(r["id"])
                if new != old:
                    changes.append({"id":r["id"], "state":r["state"], "target":r["target"],
                                    "previous":list(CATEGORIES)[old], "new":list(CATEGORIES)[new], "training_evidence":s})
            trial["preserved_all_new_correct"] &= not lost
            trial["details"][split] = {"metrics":metrics(np.array(values), targets, list(CATEGORIES)),
                                        "lost_new_correct_ids":lost,"changed":changes}
        trial["removed_introduced_general_false_positive"] = any(
            r["target"] != "general" and r["previous"] == "general" and r["new"] == r["target"]
            for r in trial["details"]["test"]["changed"])
        trials.append(trial)
    eligible = [r for r in trials if r["preserved_all_new_correct"] and r["removed_introduced_general_false_positive"]]
    if not eligible:
        raise ValueError("No candidate preserves new correct rescues and fixes the diagnosed false positive")
    chosen = max(eligible, key=lambda r:(r["details"]["development"]["metrics"]["accuracy"],
                                       -r["minimum_general_similarity"]))
    selection = {"source_v6_run":str(PREVIOUS), "previous_test_explicitly_retired":True,
                 "new_test_used_for_selection":False, "trials":trials,
                 "rule":"Protect contextual tool domains with minimum training support; preserve ALL new correct v6 decisions in old development and retired test. Max old development accuracy, then smallest threshold.",
                 "selected":chosen["minimum_general_similarity"],
                 "source_sha256":{s:sha256_file(PREVIOUS/f"{s}-predictions.jsonl") for s in sets}}
    save_json(DIRECTORY / "development-selection.json",selection)
    config.update(minimum_general_similarity=chosen["minimum_general_similarity"],
                  selection_sha256=sha256_file(DIRECTORY/"development-selection.json"))
    save_json(PROFILE_PATH,config)
    return selection


def annotate():
    """Labels assigned by Codex BEFORE any new model predictions, not corpus gold."""
    selected = read_json(DIRECTORY / "source/annotation-input.json")
    general = {23,26,27,30,36,45,50,51,57,59,61,63,72,76,80,82,91,100,112,124,129,135,138,140,143,156,157,162,163,169,171}
    excluded = {22,52,53,73,81,89,98,105,110,118,125,147,153,164,175}
    other = {7:"play",21:"cooking",79:"datetime",106:"recommendation",113:"calendar",116:"recommendation",
             134:"music",145:"cooking",152:"music",155:"cooking",172:"recommendation",173:"music"}
    rows = []
    for i, r in enumerate(selected):
        label = "excluded" if i in excluded else "general" if i in general else other.get(i,"qa")
        rows.append({"source_id":r["message_id"],"index":i,"text":r["text"],"target":label,
                     "annotation_source":"Codex, one annotator, before inference; provisional, no independent human adjudication",
                     "reason":("Ambiguous taxonomy, context or multiple intents" if label=="excluded" else
                               "Conversation, personal advice, opinion, creative or text transformation request" if label=="general" else
                               "Factual or procedural knowledge request; a requested output list is not a stored personal list" if label=="qa" else
                               "Explicit request in the named specialized domain")})
    save_json(DIRECTORY/"annotations.json",rows)
    return rows


def prepare():
    path = DIRECTORY/"manifest.json"
    if path.exists():
        result = read_json(path)
        for info in result["splits"].values():
            assert sha256_file(DIRECTORY/info["file"]) == info["sha256"]
        return result
    selection = select()
    annotations = annotate()
    prior_paths = sorted(p for p in (ROOT/"evaluation").rglob("*.jsonl")
                         if DIRECTORY not in p.parents and "source" not in p.parts and "delivery-source" not in p.parts)
    used_ids, used_groups = set(), set()
    for source in prior_paths:
        for r in lines(source):
            if isinstance(r,dict):
                used_ids.add(r.get("id"))
                if r.get("state"):
                    used_groups.add(text_group(r["state"]))
    sources = {}
    for split, sha in SOURCE_HASHES.items():
        source = ROOT/"evaluation/massive-es-v1/source"/f"{split}-es.json.gz"
        assert sha256_file(source) == sha
        sources[split] = [json.loads(x) for x in gzip.decompress(source.read_bytes()).decode("utf-8").splitlines()]
    clean, _ = disjoint_sources(sources)
    unused = [record(r,"test") for r in clean["test"]
              if f"massive:es:test:{r['id']}" not in used_ids and text_group(r["text"]) not in used_groups]
    if any(r["target"] == "general" for r in unused):
        raise ValueError("Unexpected unused General source; review the exhaustion audit")
    remaining = [r | {"label":r["target"]} for r in unused]
    partitions = {}
    for offset,(split,count) in enumerate((("test",182),("calibration",80),("validation",80))):
        chosen = stratified_sample(remaining,count,2026101701+offset)
        identifiers = {r["id"] for r in chosen}
        remaining = [r for r in remaining if r["id"] not in identifiers]
        partitions[split] = [r | {"dataset":"MASSIVE","annotation_source":"Original human scenario label"} for r in chosen]
    # Split only after all labels are fixed. No request predictions influence roles.
    fresh, duplicates = [], []
    seen = set(used_groups)
    for r in annotations:
        if r["target"] == "excluded":
            continue
        group = text_group(r["text"])
        if group in seen:
            duplicates.append(r["source_id"])
            continue
        seen.add(group)
        fresh.append({"id":"oasst1:es:"+r["source_id"],"source_id":r["source_id"],"source_split":"public_prompts",
                      "group_id":group,"state":r["text"],"question":QUESTION,"options":list(CATEGORIES.values()),
                      "option_labels":list(CATEGORIES),"target":r["target"],"label":r["target"],"modality":"text",
                      "dataset":"OpenAssistant/oasst1","annotation_source":r["annotation_source"]})
    for offset,(split,count) in enumerate((("calibration",40),("validation",40),("test",len(fresh)-80))):
        chosen = stratified_sample(fresh,count,2026101711+offset)
        identifiers = {r["id"] for r in chosen}
        fresh = [r for r in fresh if r["id"] not in identifiers]
        partitions[split] += chosen
        random.Random(2026101720+offset).shuffle(partitions[split])
    roles = list(partitions.items())
    for i,(a, first) in enumerate(roles):
        assert len({r["group_id"] for r in first}) == len(first)
        for b, second in roles[i+1:]:
            assert not {r["id"] for r in first} & {r["id"] for r in second}
            assert not {r["group_id"] for r in first} & {r["group_id"] for r in second}
    splits = {s:write_records(DIRECTORY/f"{s}.jsonl",r) for s,r in partitions.items()}
    for s,r in partitions.items():
        splits[s]["datasets"] = dict(Counter(x["dataset"] for x in r))
        splits[s]["labels"] = dict(Counter(x["target"] for x in r))
    manifest = {"schema_version":7,"dataset":"Independent Spanish challenge cohort: untouched MASSIVE negatives + human OASST prompts with provisional new labels",
                "splits":splits,"oasst_revision":OASST_REVISION,
                "sources_sha256":{"oasst-prompts":sha256_file(DIRECTORY/"source/oasst-prompts.jsonl.gz"),
                                  "annotation-input":sha256_file(DIRECTORY/"source/annotation-input.json"),
                                  "annotations":sha256_file(DIRECTORY/"annotations.json")},
                "prior_exposure_files":{str(p.relative_to(ROOT)):sha256_file(p) for p in prior_paths},
                "untouched_massive_test_count_before_sampling":len(unused),"untouched_massive_general_test_count":0,
                "dropped_oasst_duplicates":duplicates,"development_selection_sha256":sha256_file(DIRECTORY/"development-selection.json"),
                "protocol":{"test_used_for_selection":False,"previous_test":"V6 test 500 explicitly retired to development/regression. Its scores are never presented as new evaluation.",
                            "new_data":"342 unseen MASSIVE non-general texts plus a deterministic random sample of 180 Spanish OASST human root prompts, 12-220 characters. Exclude ambiguous intents before inference. Exact normalized dedup against every previous evaluation/training record.",
                            "annotation":"OASST contains human messages, not gold MASSIVE intent labels. New 18-way labels were assigned by Codex before predictions, one annotator, without human adjudication. Provisional challenge evaluation, not a replacement for independent human-gold production data.",
                            "calibration":"120 new messages fit T, 120 other new messages must improve NLL >.0001; freeze before test.",
                            "activation_gate":"Preserve every newly correct v6 development/regression decision; calibration validated; General F1 gain >=.03, General precision >=v2, overall accuracy >v2, macro F1 >=v2, paired exact one-sided McNemar p<.05, test NLL <=v2. No tuning on new test.",
                            "limitation":"General support comes only from provisionally annotated OASST; challenge prevalence differs from MASSIVE. Report separate source results. Human adjudication required to claim independent human-gold accuracy.",
                            "training":"Unchanged frozen v2/v5 banks. No new holdout text enters prompts as training; exact query excluded from retrieval."}}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(path,manifest)
    print(json.dumps({"splits":{s:i["count"] for s,i in splits.items()},"labels":{s:i["labels"] for s,i in splits.items()},"selected_floor":selection["selected"]}),flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
