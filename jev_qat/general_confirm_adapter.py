"""Runtime v8: independently confirm every proposed General decision."""
import json
import time

from .calibration import sha256_file
from .config import ROOT
from .eval_data import CATEGORIES, QUESTION, digest
from .general_confirm_development import PROFILE as PROFILE_PATH, OUTER_VERSION as VERSION
from .general_confirm_math import GENERAL, combine
from .general_confirm_task import NAME as CONFIRM_NAME, ConfirmationTask
from .general_precision_task import PrecisionTask

NAME = "massive_es_v8"


class ConfirmedTask:
    def __init__(self):
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            config = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            _,source_fp = PrecisionTask()._load()
            confirm_fp = ConfirmationTask().prepare(CONFIRM_NAME,state="",question=QUESTION,options=list(CATEGORIES.values()),modality="text")["fingerprint"]
            if config["version"] != VERSION or config["v7_fingerprint"] != source_fp or config["confirmation_fingerprint"] != confirm_fp:
                raise ValueError("Confirmed task or frozen source fingerprint changed")
            fp = digest({"profile":sha256_file(PROFILE_PATH),"implementation":sha256_file(__file__),
                         "math":sha256_file(ROOT/"jev_qat/general_confirm_math.py"),"source":source_fp,"confirmation":confirm_fp})
            self._loaded = config,fp
        return self._loaded


def predict(adapter,*,state,question,options,media=None,modality="text",video_frames=16,calibration_profile=None):
    started = time.monotonic()
    with adapter._lock:
        if modality != "text" or media is not None:
            raise ValueError("Confirmed Spanish classifier requires text and no media")
        if not hasattr(adapter,"_confirmed_task"):
            adapter._confirmed_task = ConfirmedTask()
        cfg,fp = adapter._confirmed_task._load()
        args = dict(state=state,question=question,options=options,modality="text",video_frames=video_frames)
        source = adapter.predict(**args,task_profile="massive_es_v7")
        proposed = [source["probabilities"][v] for v in CATEGORIES.values()]
        confirm = None
        values = proposed
        if max(range(18),key=proposed.__getitem__) == GENERAL:
            confirm = adapter.predict(**args,task_profile=CONFIRM_NAME)
            if confirm["metadata"]["runtime_fingerprint"] != source["metadata"]["runtime_fingerprint"]:
                raise ValueError("Inference runtime changed during confirmation")
            values = combine(proposed,[confirm["probabilities"][v] for v in CATEGORIES.values()],cfg["weight"],cfg["minimum_confirmation_probability"])
        canonical = dict(zip(CATEGORIES.values(),values))
        values = [canonical[o] for o in options]
        raw = dict(zip(options,values))
        calibration = None
        if calibration_profile is not None:
            values,calibration = adapter._calibration_profiles.apply(calibration_profile,values,question=question,options=options,
                modality="text",runtime_fingerprint=source["metadata"]["runtime_fingerprint"],prompt_version=VERSION,task_profile=NAME,task_profile_fingerprint=fp)
        index = max(range(18),key=values.__getitem__)
        parts = [source]+([confirm] if confirm else [])
        usage = {k:sum((r["metadata"].get("usage") or {}).get(k,0) for r in parts) for k in ("prompt_tokens","completion_tokens","total_tokens")}
        result = {"prediction":options[index],"prediction_index":index,"confidence":values[index],"probabilities":dict(zip(options,values)),
                  "component_probabilities":{"v7_raw":proposed,"confirmation_raw":[confirm["probabilities"][v] for v in CATEGORIES.values()] if confirm else None,
                      "source_components":source["component_probabilities"],"confirmed":bool(confirm),"weight":cfg["weight"],
                      "minimum_confirmation_probability":cfg["minimum_confirmation_probability"],"class_order":list(CATEGORIES)},
                  "metadata":source["metadata"] | {"method":"gemma_general_contrastive_confirmation","prompt_version":VERSION,"task_profile":NAME,
                      "task_profile_fingerprint":fp,"calibrated":calibration is not None,"calibration":calibration,
                      "source_metadata":[r["metadata"] for r in parts],"llm_calls":source["metadata"]["llm_calls"]+bool(confirm),
                      "usage":usage,"elapsed_seconds":round(time.monotonic()-started,4),
                      "retrieved_example_ids":list(dict.fromkeys(i for r in parts for i in r["metadata"]["retrieved_example_ids"]))}}
        if calibration_profile is not None:
            result["raw_probabilities"] = raw
        return result
