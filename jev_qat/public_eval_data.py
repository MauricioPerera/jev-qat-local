"""Frozen v8 follow-up: public human-annotated data, no new model training."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tarfile
import zipfile

from .calibration import sha256_file
from .config import ROOT, save_json
from .eval_data import CATEGORIES, QUESTION, digest, stratified_sample, text_group
from .focus_experiment import read_json, write_records
from .general_confirm_adapter import ConfirmedTask
from .general_confirm_data import category, family

DIRECTORY = ROOT / "evaluation/general-confirm-v8-expanded"
ORIGINAL = ROOT / "evaluation/general-confirm-v8"
MIAM_REVISION = "cfd51154604b900f4cbd70d4c1f6214845f7f837"
SEED = 2026101608
MTOP_COUNT = 1200
# Closing segments with explicit travel content cannot be mapped to General safely.
TASK_WORDS = r"\b(?:tren\w*|viaj\w*|billet\w*|horario\w*|salir|volver|barcelona|madrid|características)\b"


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def exposure():
    paths = sorted(p for p in (ROOT / "evaluation").rglob("*.jsonl")
                   if DIRECTORY not in p.parents and not {"source", "delivery-source"}.intersection(p.parts))
    ids, families = set(), set()
    for path in paths:
        for row in lines(path):
            if isinstance(row, dict):
                ids.add(row.get("id"))
                if row.get("state"):
                    families.add(family(row["state"]))
    return ids, families, {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def official_closings():
    index = {}
    pattern = r"^(.*?) \(U:Cierre:Nil:Nil\) pal: (\d+) - (\d+)$"
    with tarfile.open(DIRECTORY / "source/dihana/official.tgz", "r:gz") as archive:
        for member in archive:
            if not member.name.endswith(".dia"):
                continue
            for number, line in enumerate(archive.extractfile(member).read().decode("latin1").splitlines(), 1):
                hit = re.match(pattern, line.strip())
                if hit:
                    index[(Path(member.name).stem, " ".join(hit[1].split()))] = {
                        "archive_member": member.name, "annotation_line": number,
                        "annotation": "U:Cierre:Nil:Nil", "word_range": [int(hit[2]), int(hit[3])]}
    return index


def record(identifier, text, target, dataset, **extra):
    return {"id": identifier, "state": text, "target": target, "label": target,
            "group_id": text_group(text), "family_id": digest(family(text)),
            "dataset": dataset, "question": QUESTION, "options": list(CATEGORIES.values()),
            "option_labels": list(CATEGORIES), "modality": "text",
            "original_human_annotation": True, "crosswalk_new_human_adjudication": False, **extra}


def dihana_records(used_families):
    original = official_closings()
    # Maximum matching retains unique text families AND unique actual participants.
    by_speaker = defaultdict(dict)
    checked = 0
    for path in sorted((DIRECTORY / "source/miam").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream), 2):
                if row["Speaker"] != "U" or row["Dialogue_Act"] != "Cierre":
                    continue
                key = row["File_ID"], " ".join(row["Utterance"].split())
                if key not in original:
                    raise ValueError("DIHANA mirror text/act does not match the original human annotation")
                checked += 1
                f = family(row["Utterance"])
                if f in used_families or len(f) < 8 or re.search(TASK_WORDS, f):
                    continue
                speaker = row["File_ID"].split("_")[0]
                by_speaker[speaker].setdefault(f, record(
                    f"dihana:es:{row['File_ID']}:{path.stem}:{number}", row["Utterance"], "general", "DIHANA",
                    source_speaker=speaker, source_dialogue=row["File_ID"], source_split=path.stem,
                    source_row=number, source_intents=["Cierre"], original_annotation=original[key],
                    annotation_source="Human dialog-act annotation, consistently reviewed by a human; exact original U:Cierre:Nil:Nil segment match"))
    matching = {}

    def assign(speaker, visited):
        for f in sorted(by_speaker[speaker]):
            if f in visited:
                continue
            visited.add(f)
            if f not in matching or assign(matching[f], visited):
                matching[f] = speaker
                return True
        return False

    for speaker in sorted(by_speaker):
        assign(speaker, set())
    result = [by_speaker[s][f] for f, s in sorted(matching.items())]
    assert len({r["source_speaker"] for r in result}) == len(result)
    return result, {"mirror_closing_rows_checked": checked,
                    "eligible_families": len({f for rr in by_speaker.values() for f in rr}),
                    "selected_independent_speakers": len(result)}


def mtop_records(used_ids, used_families):
    output, seen, checked = [], set(used_families), 0
    with zipfile.ZipFile(ORIGINAL / "source/mtop/official.zip") as archive:
        for split, original_split in (("test", "test"), ("validation", "eval")):
            gold = {r[0]: r for r in (line.split("\t") for line in
                    archive.read(f"mtop/es/{original_split}.txt").decode("utf-8").splitlines())}
            for row in lines(ORIGINAL / f"source/mtop/{split}.jsonl"):
                g = gold[str(row["id"])]
                if (row["text"], row["label_text"]) != (g[3], g[1].removeprefix("IN:")):
                    raise ValueError("MTOP mirror ID/text/intent differs from the official archive")
                checked += 1
                target, f = category(row["label_text"]), family(row["text"])
                identifier = f"mtop:es:{split}:{row['id']}"
                if target and f not in seen and identifier not in used_ids:
                    seen.add(f)
                    output.append(record(identifier, row["text"], target, "MTOP",
                        source_id=str(row["id"]), source_split=split, source_intents=[row["label_text"]],
                        annotation_source="Original human MTOP intent and professional Spanish translation; ID/full text/intent match official Meta archive"))
    # Remove any family with contradictory bounded source-intent mappings, including unselected rows.
    labels = defaultdict(set)
    for split in ("test", "validation"):
        for row in lines(ORIGINAL / f"source/mtop/{split}.jsonl"):
            target = category(row["label_text"])
            labels[family(row["text"])].add(target)
    output = [r for r in output if labels[family(r["state"])] == {r["target"]}]
    return output, {"official_mirror_rows_checked": checked, "fresh_unambiguous_pool": len(output)}


def frozen_profiles():
    _, fingerprint = ConfirmedTask()._load()
    paths = [ROOT / f"classification/profiles/massive_es_{v}.json" for v in ("v2", "v8")]
    paths += [ROOT / f"calibration/profiles/massive_scenario_es_{v}.json" for v in ("v2", "v8")]
    paths += [ROOT / "jev_qat" / name for name in (
        "adapter.py", "calibration.py", "spanish_task.py", "general_task.py", "general_rescue_task.py",
        "general_rescue_math.py", "general_precision_task.py", "general_precision_math.py",
        "general_confirm_adapter.py", "general_confirm_task.py", "general_confirm_math.py")]
    return {"v8_task_fingerprint": fingerprint,
            "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
            "temperatures": {v: read_json(ROOT / f"calibration/profiles/massive_scenario_es_{v}.json")["temperature"]
                             for v in ("v2", "v8")}}


def prepare():
    path = DIRECTORY / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if sha256_file(DIRECTORY / "test.jsonl") != manifest["test"]["sha256"]:
            raise ValueError("Frozen evaluation records changed")
        if frozen_profiles() != manifest["frozen_profiles"]:
            raise ValueError("v8/v2 or their calibrations changed after preregistration")
        return manifest
    ids, families, exposed = exposure()
    general, dihana_audit = dihana_records(families)
    pool, mtop_audit = mtop_records(ids, families | {family(r["state"]) for r in general})
    selected = stratified_sample(pool, MTOP_COUNT, SEED)
    records = sorted(general + selected, key=lambda r: digest([SEED, r["id"]]))
    assert len({r["family_id"] for r in records}) == len(records)
    assert len(general) >= 50 and not {r["id"] for r in records}.intersection(ids)
    assert not {family(r["state"]) for r in records}.intersection(families)
    sources = list((DIRECTORY / "source").rglob("*"))
    sources = [p for p in sources if p.is_file() and p.suffix != ".part"]
    sources += [ORIGINAL / "source/mtop" / n for n in ("official.zip", "test.jsonl", "validation.jsonl")]
    crosswalk = read_json(ORIGINAL / "label-crosswalk.json")
    crosswalk["dihana"] = {"Cierre": "general", "speaker": "U", "original_lower_levels": ["Nil", "Nil"],
                           "exclude_task_words_regex": TASK_WORDS, "minimum_family_length": 8,
                           "sampling": "Maximum deterministic matching, one family per participant and one participant per family"}
    crosswalk.pop("multi3nlu_general_acts", None)
    save_json(DIRECTORY / "label-crosswalk.json", crosswalk)
    info = write_records(DIRECTORY / "test.jsonl", records)
    save_json(DIRECTORY / "label-provenance.json", {
        "human_original_labels": True, "new_human_review_of_18_category_crosswalk": False,
        "model_generated_gold_labels": False, "dihana": dihana_audit, "mtop": mtop_audit,
        "sources": {
            "dihana_manual_annotation_and_single_reviewer": "https://aclanthology.org/L10-1209/",
            "dihana_original_release": "https://doi.org/10.5281/zenodo.5572470",
            "miam_revision": MIAM_REVISION,
            "mtop_professional_translation": "https://aclanthology.org/2021.eacl-main.257/"}})
    manifest = {"schema_version": 1, "experiment": "v8-frozen-public-human-follow-up",
        "created_at": datetime.now(timezone.utc).isoformat(), "test": info,
        "counts": {"datasets": dict(Counter(r["dataset"] for r in records)),
                   "labels": dict(Counter(r["target"] for r in records))},
        "source_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(sources)},
        "prior_exposure_files": exposed, "frozen_profiles": frozen_profiles(),
        "crosswalk_sha256": sha256_file(DIRECTORY / "label-crosswalk.json"),
        "label_provenance_sha256": sha256_file(DIRECTORY / "label-provenance.json"),
        "protocol": {"seed": SEED, "fixed_mtop_n": MTOP_COUNT, "new_dihana_general": len(general),
            "new_cohort_only": True, "previous_test_used_in_significance": False,
            "primary_test": "One-sided exact paired McNemar on accuracy, v8 greater than v2",
            "alpha": .025, "strict_p_less_than_alpha": True,
            "multiple_looks": "Conservative .025 threshold for this second v8 cohort. The earlier nominal .05 test had p=.5. No retrospective claim that the historical decision procedure controlled a .05 familywise rate.",
            "no_optional_stopping": "One fixed cohort, one final promotion decision after all records. Interrupted transport requests may resume identical cached requests. No enlarging/replacing/retuning after scores.",
            "no_calibration_fit": "Use previously frozen v2/v8 temperatures. Assess NLL, Brier and reliability descriptively; fit no parameters on this confirmatory cohort.",
            "activation_gates": ["all eight protected recoveries pass live", "three prior domain corrections pass live",
                "unchanged model/task/calibration fingerprints", "documented original human labels and frozen crosswalk",
                "at least 50 General targets", "General F1 gain >= .03", "General precision >= v2",
                "accuracy > v2", "macro F1 >= v2", "one-sided exact McNemar p < .025", "NLL <= v2",
                "independent verification and local API composition checks pass"],
            "scope": "Fixed public challenge mixture of seven MTOP specialized categories and DIHANA conversational closings. Not all 18 categories, open advice/creative General, or Spanish production traffic. Unknown pretraining exposure.",
            "independence": "Exclude all previously exposed IDs and punctuation-insensitive text families. DIHANA uses one segment per actual participant. MTOP writer IDs unavailable; unique utterance families are units, remaining author/template dependence cannot be excluded.",
            "labels": "Original annotations by humans and documented Spanish provenance only. Both corpora cross-checked against original archives. Bounded deterministic ontology mapping, with no new human adjudication of that mapping."}}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(path, manifest)
    print(json.dumps({"frozen_n": len(records), "datasets": manifest["counts"]["datasets"],
                      "alpha": .025, "general_support": len(general)}), flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
