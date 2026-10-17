"""Collect native components once; compose the unchanged v8 deterministically."""
import json
import numpy as np

from .eval_data import CATEGORIES
from .evaluate import collect
from .general_confirm_adapter import ConfirmedTask, NAME, VERSION
from .general_confirm_math import combine as confirm_combine, GENERAL
from .general_confirm_task import NAME as CONFIRM_NAME, VERSION as CONFIRM_VERSION
from .general_precision_math import combine, general_evidence, needs_review
from .general_precision_task import PrecisionTask, NAME as PROPOSED_NAME, VERSION as PROPOSED_VERSION
from .general_task import VERSION as FOCUSED_VERSION
from .spanish_task import VERSION as BASE_VERSION


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def evidence(client, records, run_dir):
    cfg, proposal_fp = PrecisionTask()._load()
    outer_cfg, fingerprint = ConfirmedTask()._load()
    baseline = collect(client, [r | {"task_profile": "massive_es_v2"} for r in records],
                       run_dir / "test-v2.jsonl", run_dir, "test-v2", BASE_VERSION)
    supports, reviews = {}, []
    for r, b in zip(records, baseline):
        if not needs_review(b["probabilities"], cfg["threshold"], cfg["component_temperature"]):
            continue
        supports[r["id"]] = general_evidence(r["state"])
        winner = list(CATEGORIES)[int(np.argmax(b["probabilities"]))]
        if winner not in cfg["protected_domains"] or supports[r["id"]]["general_similarity"] >= cfg["minimum_general_similarity"]:
            reviews.append(r)
    focused = collect(client, [r | {"task_profile": "massive_es_v5"} for r in reviews],
                      run_dir / "test-focus.jsonl", run_dir, "test-focused", FOCUSED_VERSION)
    by_id = {r["id"]: r for r in focused}
    proposed = []
    for r, b in zip(records, baseline):
        f = by_id.get(r["id"])
        values = combine(b["probabilities"], f["probabilities"], cfg, supports[r["id"]]["general_similarity"]) if f else b["probabilities"]
        sources = [b] + ([f] if f else [])
        proposed.append({"id": r["id"], "target": r["target"], "probabilities": values, "derived": True,
            "component_probabilities": {"base_raw": b["probabilities"], "focused_raw": f["probabilities"] if f else None,
                                        "training_evidence": supports.get(r["id"]), "reviewed": bool(f)},
            "metadata": b["metadata"] | {"task_profile": PROPOSED_NAME, "prompt_version": PROPOSED_VERSION,
                "task_profile_fingerprint": proposal_fp, "llm_calls": len(sources), "calibrated": False,
                "derived_timings": True, "elapsed_seconds": sum(s["metadata"]["elapsed_seconds"] for s in sources),
                "source_metadata": [s["metadata"] for s in sources]}})
    write(run_dir / "test-proposal.jsonl", proposed)
    reviews = [r | {"task_profile": CONFIRM_NAME} for r, p in zip(records, proposed) if np.argmax(p["probabilities"]) == GENERAL]
    confirmed = collect(client, reviews, run_dir / "test-confirmation.jsonl", run_dir, "test-confirmation", CONFIRM_VERSION)
    by_id = {r["id"]: r for r in confirmed}
    output = []
    for r, p in zip(records, proposed):
        c = by_id.get(r["id"])
        values = confirm_combine(p["probabilities"], c["probabilities"], outer_cfg["weight"],
                                  outer_cfg["minimum_confirmation_probability"]) if c else p["probabilities"]
        output.append({"id": r["id"], "target": r["target"], "probabilities": values, "derived": True,
            "component_probabilities": {"v7_raw": p["probabilities"], "confirmation_raw": c["probabilities"] if c else None,
                                        "source_components": p["component_probabilities"], "confirmed": bool(c)},
            "metadata": p["metadata"] | {"task_profile": NAME, "prompt_version": VERSION,
                "task_profile_fingerprint": fingerprint, "llm_calls": p["metadata"]["llm_calls"] + bool(c),
                "source_metadata": [p["metadata"]] + ([c["metadata"]] if c else []),
                "elapsed_seconds": p["metadata"]["elapsed_seconds"] + (c["metadata"]["elapsed_seconds"] if c else 0)}})
    write(run_dir / "test-predictions.jsonl", output)
    return output, baseline
