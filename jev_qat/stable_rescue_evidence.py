"""Native shared v2 baseline and actual uncached v9 source components."""
import json
from pathlib import Path

import numpy as np

from .calibration import sha256_file
from .eval_data import CATEGORIES
from .evaluate import collect
from .general_precision_math import general_evidence, needs_review
from .spanish_task import VERSION as BASE_VERSION
from .stable_rescue_adapter import StableRescueTask, NAME, VERSION
from .stable_rescue_data import lines, DIRECTORY, V8, EXPANDED
from .stable_rescue_math import combine, proposal
from .stable_rescue_task import FOCUS_NAME, CONFIRM_NAME, FOCUS_VERSION, CONFIRM_VERSION


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def import_retired_baseline(records, path):
    """Reuse real, unchanged v2 native scores; mark imported evidence explicitly."""
    if path.exists():
        return
    sources = [V8 / "runs/f91c19811cbf27ff" / f"{s}-v2.jsonl" for s in ("calibration", "validation", "test")]
    sources.append(EXPANDED / "runs/5632184e6623ffd0/test-v2.jsonl")
    needed = {r["id"] for r in records}
    found = {}
    for source in sources:
        for row in lines(source):
            if row["id"] in needed:
                found[row["id"]] = row | {"imported_native_evidence": {"file": str(source), "sha256": sha256_file(source)}}
    if set(found) != needed:
        raise ValueError("Retired v2 baseline evidence missing")
    write(path, [found[r["id"]] for r in records])


def evidence(client, records, run_dir, split):
    cfg, fingerprint = StableRescueTask()._load()
    base_path = run_dir / f"{split}-v2.jsonl"
    if split in ("calibration", "validation"):
        import_retired_baseline(records, base_path)
    baseline = collect(client, [r | {"task_profile": "massive_es_v2"} for r in records], base_path, run_dir, f"{split}-v2", BASE_VERSION)
    supports, review_rows = {}, []
    for r, b in zip(records, baseline):
        if not needs_review(b["probabilities"], cfg["review_threshold"], cfg["component_temperature"]):
            continue
        supports[r["id"]] = general_evidence(r["state"])
        winner = list(CATEGORIES)[int(np.argmax(b["probabilities"]))]
        if winner not in cfg["protected_domains"] or supports[r["id"]]["general_similarity"] >= cfg["minimum_general_similarity"]:
            review_rows.append(r)
    focused = collect(client, [r | {"task_profile": FOCUS_NAME} for r in review_rows],
        run_dir / f"{split}-focus.jsonl", run_dir, f"{split}-focus", FOCUS_VERSION)
    by_id = {r["id"]: r for r in focused}
    confirmation_rows = []
    for r, b in zip(records, baseline):
        f = by_id.get(r["id"])
        _, strength = proposal(b["probabilities"], f["probabilities"] if f else None, cfg,
            supports.get(r["id"], {}).get("general_similarity", 0.))
        if strength > 0:
            confirmation_rows.append(r)
    confirmed = collect(client, [r | {"task_profile": CONFIRM_NAME} for r in confirmation_rows],
        run_dir / f"{split}-confirmation.jsonl", run_dir, f"{split}-confirmation", CONFIRM_VERSION)
    confirmed_by_id = {r["id"]: r for r in confirmed}
    output = []
    for r, b in zip(records, baseline):
        f, c = by_id.get(r["id"]), confirmed_by_id.get(r["id"])
        similarity = supports.get(r["id"], {}).get("general_similarity", 0.)
        fv, cv = f["probabilities"] if f else None, c["probabilities"] if c else None
        values = combine(b["probabilities"], fv, cv, cfg, similarity)
        p, strength = proposal(b["probabilities"], fv, cfg, similarity)
        sources = [b] + ([f] if f else []) + ([c] if c else [])
        if len({s["metadata"]["runtime_fingerprint"] for s in sources}) != 1:
            raise ValueError("Runtime changed across stable-rescue stages")
        output.append({"id": r["id"], "target": r["target"], "probabilities": values, "derived": True,
            "component_probabilities": {"base_raw": b["probabilities"], "focused_raw": fv, "confirmation_raw": cv,
                "proposal_raw": p, "rescue_strength": strength, "training_evidence": supports.get(r["id"]),
                "native_prefix_cache": False, "base_prefix_cache": "v2_default", "class_order": list(CATEGORIES)},
            "metadata": b["metadata"] | {"task_profile": NAME, "prompt_version": VERSION,
                "task_profile_fingerprint": fingerprint, "llm_calls": len(sources), "calibrated": False,
                "source_metadata": [s["metadata"] for s in sources], "derived_timings": True,
                "elapsed_seconds": sum(s["metadata"]["elapsed_seconds"] for s in sources)}})
    write(run_dir / f"{split}-predictions.jsonl", output)
    return output, baseline
