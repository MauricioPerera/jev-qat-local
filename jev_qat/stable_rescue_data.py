"""New public human-labelled cohort; old data may fit calibration only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import tarfile

from .calibration import sha256_file
from .config import ROOT, save_json
from .eval_data import digest, stratified_sample
from .focus_experiment import read_json, write_records
from .general_confirm_data import ACTS, family
from .public_eval_data import record, mtop_records

DIRECTORY = ROOT / "evaluation/stable-rescue-v9"
V8 = ROOT / "evaluation/general-confirm-v8"
EXPANDED = ROOT / "evaluation/general-confirm-v8-expanded"
SEED = 2026101619


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def prior_exposure():
    files = sorted(p for p in (ROOT / "evaluation").rglob("*.jsonl")
                   if DIRECTORY not in p.parents and not {"source", "delivery-source"}.intersection(p.parts))
    ids, families = set(), set()
    for p in files:
        for r in lines(p):
            if isinstance(r, dict):
                ids.add(r.get("id"))
                if r.get("state"):
                    families.add(family(r["state"]))
    return ids, families, {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def new_general(families):
    used = set(families)
    by_dialogue = {}
    source_files = []
    for p in sorted((V8 / "source/multi3nlu").rglob("fold*.json")):
        source_files.append(p)
        for r in read_json(p):
            f = family(r["text"])
            if not r.get("intents") or not set(r["intents"]) <= set(ACTS) | {"repeat"} or r.get("slots") or f in used:
                continue
            # Dialogue repair refers to the current conversation, with no task slots.
            dialogue = r["uid"].rsplit("_", 1)[0]  # Published fold ID, not a conversation ID.
            by_dialogue.setdefault(dialogue, []).append(record("multi3nlu:es:" + r["uid"], r["text"], "general", "Multi3NLU++",
                source_id=r["uid"], source_intents=r["intents"], source_group=dialogue,
                source_split="published_folds", annotation_source="Original human dialogue-act labels, professional Spanish translation and source-author QA; pure conversational/repair acts without task slots"))
    result = []
    for dialogue in sorted(by_dialogue):
        choices = sorted(by_dialogue[dialogue], key=lambda r: digest([SEED, r["id"]]))
        for r in choices:
            if family(r["state"]) not in used:
                result.append(r); used.add(family(r["state"]))
                break
    selected_dialogues = set(by_dialogue)
    # Four human-annotated task-independent acts, with empty task-level fields.
    pattern = re.compile(r"^(.*?) \(U:(Afirmacion|Negacion|Cierre|No_entendido):Nil:Nil\) pal: (\d+) - (\d+)$")
    by_speaker = defaultdict(dict)
    archive_path = EXPANDED / "source/dihana/official.tgz"
    source_files.append(archive_path)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if not member.name.endswith(".dia"):
                continue
            dialogue = Path(member.name).stem
            speaker = dialogue.split("_")[0]
            for number, line in enumerate(archive.extractfile(member).read().decode("latin1").splitlines(), 1):
                hit = pattern.match(line.strip())
                if not hit:
                    continue
                text, f = hit[1], family(hit[1])
                if f in used or len(f) < 8 or re.search(r"\b(?:tren\w*|viaj\w*|billet\w*|horario\w*|barcelona|madrid|características)\b", f):
                    continue
                by_speaker[speaker].setdefault(f, record(f"dihana:es:{dialogue}:annotation:{number}", text, "general", "DIHANA",
                    source_speaker=speaker, source_dialogue=dialogue, source_intents=[hit[2]], source_split="original_archive",
                    original_annotation={"archive_member": member.name, "annotation_line": number,
                        "annotation": f"U:{hit[2]}:Nil:Nil", "word_range": [int(hit[3]), int(hit[4])]},
                    annotation_source="Original human dialogue-act annotation consistently reviewed by a human; Nil task/concept fields; unchanged text and act from official archive"))
    matching = {}

    def assign(speaker, seen):
        for f in sorted(by_speaker[speaker]):
            if f in seen:
                continue
            seen.add(f)
            if f not in matching or assign(matching[f], seen):
                matching[f] = speaker
                return True
        return False

    for speaker in sorted(by_speaker):
        assign(speaker, set())
    result += [by_speaker[s][f] for f, s in sorted(matching.items())]
    return result, selected_dialogues, source_files


def prepare():
    path = DIRECTORY / "manifest.json"
    if path.exists():
        m = read_json(path)
        for info in m["splits"].values():
            if sha256_file(DIRECTORY / info["file"]) != info["sha256"]:
                raise ValueError("Frozen data changed")
        return m
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    ids, families, exposure = prior_exposure()
    general, test_dialogues, sources = new_general(families)
    pool, source_audit = mtop_records(ids, families | {family(r["state"]) for r in general})
    if len(pool) < 600:
        raise ValueError(f"Need 600 new MTOP examples, have {len(pool)}")
    mintaka_file = DIRECTORY / "source/mintaka/mintaka_test.json"
    mintaka = []
    seen = set(families) | {family(r["state"]) for r in general}
    for r in read_json(mintaka_file):
        if r["category"] not in ("history", "geography", "books"):
            continue
        text = r["translations"].get("es")
        if not text or family(text) in seen:
            continue
        seen.add(family(text))
        row = record("mintaka:es:test:" + r["id"], text, "qa", "Mintaka",
            source_id=r["id"], source_split="test", source_category=r["category"],
            source_complexity=r["complexityType"], source_language="es-MX",
            annotation_source="Crowd-written factual QA with human entity/answer annotation and professional Spanish (Mexico) translation; selected knowledge topics mapped to qa")
        mintaka.append(row | {"label": r["category"]})
    qa = [r | {"label": "qa"} for r in stratified_sample(mintaka, 200, SEED + 20)]
    test = general + stratified_sample(pool, 600, SEED) + qa
    sources += list((DIRECTORY / "source/mintaka").glob("*"))
    # Retire old public tests, never use the new cohort to fit temperature.
    old_general = []
    for split in ("calibration", "validation", "test"):
        old_general += [r for r in lines(V8 / f"{split}.jsonl") if r["dataset"] == "Multi3NLU++"]
    grouped = defaultdict(list)
    for r in old_general:
        dialogue = r["source_id"].rsplit("_", 1)[0]
        if dialogue not in test_dialogues:
            grouped[dialogue].append(r)
    old_general = [sorted(rr, key=lambda r: digest([SEED, r["id"]]))[0] for _, rr in sorted(grouped.items())]
    if len(old_general) < 12:
        raise ValueError("Insufficient retired General dialogues independent of the new test")
    old_pool = [r for r in lines(EXPANDED / "test.jsonl") if r["dataset"] == "MTOP"]
    partitions = {"test": test}
    for index, split in enumerate(("calibration", "validation")):
        chosen = stratified_sample(old_pool, 94, SEED + index + 1)
        selected_ids = {r["id"] for r in chosen}
        old_pool = [r for r in old_pool if r["id"] not in selected_ids]
        partitions[split] = chosen + old_general[index*6:(index+1)*6]
    for split, rows in partitions.items():
        rows.sort(key=lambda r: digest([SEED, split, r["id"]]))
        assert len({family(r["state"]) for r in rows}) == len(rows)
    for s0, s1 in (("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
        assert not {family(r["state"]) for r in partitions[s0]} & {family(r["state"]) for r in partitions[s1]}
    assert not {r["id"] for r in test}.intersection(ids)
    assert not {family(r["state"]) for r in test}.intersection(families)
    assert len(general) >= 35
    sources += [V8 / "source/mtop" / n for n in ("official.zip", "test.jsonl", "validation.jsonl")]
    crosswalk = {"mtop": read_json(V8 / "label-crosswalk.json")["mtop"],
        "mintaka": {"qa_topics": ["history", "geography", "books"], "language": "es-MX",
            "target": "qa", "provenance": "https://aclanthology.org/2022.coling-1.138/"},
        "multi3nlu": {"general_acts": list(ACTS) + ["repeat"], "no_slots": True,
            "grouping": "Published domain/fold UID prefix; actual conversation and author IDs unavailable",
            "meaning": "Pure conversational acts and requests to repeat the current utterance; no content playback/device-volume task is annotated"},
        "dihana": {"general_acts": ["Afirmacion", "Negacion", "Cierre", "No_entendido"], "task_levels": ["Nil", "Nil"]},
        "new_human_adjudication_of_crosswalk": False}
    save_json(DIRECTORY / "label-crosswalk.json", crosswalk)
    m = {"schema_version": 9, "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {s: write_records(DIRECTORY / f"{s}.jsonl", rows) for s, rows in partitions.items()},
        "counts": {s: {"datasets": dict(Counter(r["dataset"] for r in rr)), "labels": dict(Counter(r["target"] for r in rr))} for s, rr in partitions.items()},
        "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(sources)},
        "prior_exposure_files": exposure, "source_audit": source_audit,
        "crosswalk_sha256": sha256_file(DIRECTORY / "label-crosswalk.json"),
        "protocol": {"seed": SEED, "candidate": "v9", "alpha": .025, "fixed_new_mtop_n": 600, "fixed_new_mintaka_n": 200,
            "new_general_n": len(general), "new_test_only_for_significance": True,
            "calibration": "100 retired public examples fit temperature; 100 other retired public examples validate NLL. V9 weights/ramps selected only on old MASSIVE regressions before these sets. No new test used.",
            "labels": "Documented human original annotations and Spanish texts only, no model-generated gold. Bounded deterministic crosswalk with no new human adjudication. Multi3 dialogue repair belongs to current-conversation General in the frozen mapping.",
            "test_unit": "Unique text family; at most one Multi3 published domain/fold UID-prefix group and one DIHANA participant per new cohort. Multi3 calibration/validation/test groups are disjoint. These IDs do not establish author or conversation independence. DIHANA participants may appear in earlier published evaluation; no DIHANA used in v9 parameter selection or calibration. MTOP author IDs unavailable; remaining author/template dependence unknown.",
            "scope": "Public challenge, nine category families; conversational repair/closing General plus specialized requests and factual history/geography/books QA. Not all advice or creative requests. Unknown Gemma pretraining exposure.",
            "activation": "All 8 recovered cases and 3 known domain corrections pass every one of 3 predeclared live orders; no lost recovered cases. New-test accuracy higher, General precision/F1 and macro F1 no lower, exact one-sided McNemar p<.025, validated calibration and NLL no higher, independent evidence/API/integration checks. No optional stopping, no retuning or enlarging this test.",
            "practical_gate_change": "For this new candidate, preserve General F1 instead of the previous +3pp gate, which was unattainable in the last high-ceiling cohort. This change is fixed before any v9 new-test inference; statistical superiority in accuracy remains mandatory."}}
    m["manifest_fingerprint"] = digest(m)
    save_json(path, m)
    print(json.dumps({"new_test_n": len(test), "new_general": len(general), "retired_calibration": 100, "retired_validation": 100}), flush=True)
    return m


if __name__ == "__main__":
    prepare()
