"""v9: cache-independent rescue components and a smooth, confirmed mixture."""
import json
import time

from .calibration import sha256_file
from .config import ROOT
from .eval_data import CATEGORIES, QUESTION, digest
from .general_precision_math import general_evidence, needs_review
from .stable_rescue_math import combine, proposal
from .stable_rescue_task import StableNativeTask, FOCUS_NAME, CONFIRM_NAME
from .spanish_task import SpanishTask

NAME = "massive_es_v9"
VERSION = "jev-qat-massive-es-stable-rescue-v9"
PROFILE = ROOT / "classification/profiles/massive_es_v9.json"


class StableRescueTask:
    def __init__(self):
        self._loaded = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        cfg = json.loads(PROFILE.read_text(encoding="utf-8"))
        _, _, base_fp = SpanishTask()._load()
        handler = StableNativeTask()
        kwargs = dict(state="", question=QUESTION, options=list(CATEGORIES.values()), modality="text")
        fps = {name: handler.prepare(name, **kwargs)["fingerprint"] for name in (FOCUS_NAME, CONFIRM_NAME)}
        if cfg["version"] != VERSION or cfg["base_fingerprint"] != base_fp or cfg["native_fingerprints"] != fps:
            raise ValueError("Stable rescue source or profile changed")
        fp = digest({"profile": sha256_file(PROFILE), "implementation": sha256_file(__file__),
                     "math": sha256_file(ROOT / "jev_qat/stable_rescue_math.py"),
                     "native_transport": sha256_file(ROOT / "jev_qat/adapter.py"),
                     "base": base_fp, "native": fps})
        self._loaded = cfg, fp
        return self._loaded


def predict(adapter, *, state, question, options, media=None, modality="text", video_frames=16, calibration_profile=None):
    started = time.monotonic()
    with adapter._lock:
        if modality != "text" or media is not None:
            raise ValueError("Stable Spanish rescue requires text with no media")
        if not hasattr(adapter, "_stable_rescue_task"):
            adapter._stable_rescue_task = StableRescueTask()
        cfg, fp = adapter._stable_rescue_task._load()
        args = dict(state=state, question=question, options=options, modality="text", video_frames=video_frames)
        base = adapter.predict(**args, task_profile="massive_es_v2")
        b = [base["probabilities"][v] for v in CATEGORIES.values()]
        evidence = None
        focus = confirm = None
        if needs_review(b, cfg["review_threshold"], cfg["component_temperature"]):
            evidence = general_evidence(state)
            winner = list(CATEGORIES)[max(range(18), key=b.__getitem__)]
            if winner not in cfg["protected_domains"] or evidence["general_similarity"] >= cfg["minimum_general_similarity"]:
                focus = adapter.predict(**args, task_profile=FOCUS_NAME)
        f = [focus["probabilities"][v] for v in CATEGORIES.values()] if focus else None
        similarity = evidence["general_similarity"] if evidence else 0.
        p, r = proposal(b, f, cfg, similarity)
        if r > 0:
            confirm = adapter.predict(**args, task_profile=CONFIRM_NAME)
        c = [confirm["probabilities"][v] for v in CATEGORIES.values()] if confirm else None
        values = combine(b, f, c, cfg, similarity)
        sources = [base] + ([focus] if focus else []) + ([confirm] if confirm else [])
        if len({s["metadata"]["runtime_fingerprint"] for s in sources}) != 1:
            raise ValueError("Runtime changed during stable rescue")
        raw = dict(zip(CATEGORIES.values(), values))
        ordered = [raw[o] for o in options]
        calibration = None
        if calibration_profile is not None:
            ordered, calibration = adapter._calibration_profiles.apply(calibration_profile, ordered, question=question,
                options=options, modality="text", runtime_fingerprint=base["metadata"]["runtime_fingerprint"],
                prompt_version=VERSION, task_profile=NAME, task_profile_fingerprint=fp)
        index = max(range(18), key=ordered.__getitem__)
        usage = {key: sum((s["metadata"].get("usage") or {}).get(key, 0) for s in sources)
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        result = {"prediction": options[index], "prediction_index": index, "confidence": ordered[index],
            "probabilities": dict(zip(options, ordered)),
            "component_probabilities": {"base_raw": b, "focused_raw": f, "confirmation_raw": c,
                "proposal_raw": p, "rescue_strength": r, "training_evidence": evidence,
                "native_prefix_cache": False, "base_prefix_cache": "v2_default", "class_order": list(CATEGORIES)},
            "metadata": base["metadata"] | {"method": "gemma_stable_smooth_confirmed_rescue", "prompt_version": VERSION,
                "task_profile": NAME, "task_profile_fingerprint": fp, "calibrated": calibration is not None,
                "calibration": calibration, "llm_calls": len(sources), "usage": usage,
                "source_metadata": [s["metadata"] for s in sources],
                "retrieved_example_ids": list(dict.fromkeys(i for s in sources for i in s["metadata"]["retrieved_example_ids"])),
                "elapsed_seconds": round(time.monotonic()-started, 4)}}
        if calibration_profile is not None:
            result["raw_probabilities"] = dict(zip(options, [raw[o] for o in options]))
        return result
