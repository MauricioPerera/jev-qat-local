"""General rescue with frozen training support for contextual tool domains."""
import json
import time

from .calibration import sha256_file
from .config import ROOT
from .eval_data import CATEGORIES,digest
from .general_precision_math import combine,needs_review,general_evidence
from .general_task import GeneralTask
from .spanish_task import SpanishTask

NAME="massive_es_v7"
VERSION="jev-qat-massive-es-general-precision-v7"
PROFILE_PATH=ROOT/"classification/profiles/massive_es_v7.json"


class PrecisionTask:
    def __init__(self):
        self._loaded=None

    def _load(self):
        if self._loaded is None:
            config=json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            if config["version"]!=VERSION:
                raise ValueError("General rescue version mismatch")
            _,_,base_fp=SpanishTask()._load()
            _,_,focused_fp=GeneralTask()._load()
            if (base_fp,focused_fp)!=(config["base_fingerprint"],config["focused_fingerprint"]):
                raise ValueError("General rescue source components changed")
            fingerprint=digest({"profile":sha256_file(PROFILE_PATH),"implementation":sha256_file(__file__),
                                "math":sha256_file(__file__.replace("general_precision_task.py","general_precision_math.py")),
                                "base":base_fp,"focused":focused_fp,"composition":sha256_file(__file__.replace("general_precision_task.py","general_rescue_math.py"))})
            self._loaded=config,fingerprint
        return self._loaded


def predict(adapter,*,state,question,options,media=None,modality="text",video_frames=16,calibration_profile=None):
    started=time.monotonic()
    with adapter._lock:
        if modality!="text" or media is not None:
            raise ValueError("General rescue requires text with no media")
        if not hasattr(adapter,"_precision_task"):
            adapter._precision_task=PrecisionTask()
        config,fingerprint=adapter._precision_task._load()
        # Child requests retain the existing strict validation and canonical order.
        arguments=dict(state=state,question=question,options=options,modality="text",video_frames=video_frames)
        base=adapter.predict(**arguments,task_profile="massive_es_v2")
        values=[base["probabilities"][v] for v in CATEGORIES.values()]
        active=needs_review(values,config["threshold"],config["component_temperature"])
        evidence=general_evidence(state) if active else None
        winner=list(CATEGORIES)[max(range(18),key=values.__getitem__)]
        if active and winner in config["protected_domains"] and evidence["general_similarity"] < config["minimum_general_similarity"]:
            active=False
        focused=None
        probabilities=list(values)
        if active:
            focused=adapter.predict(**arguments,task_profile="massive_es_v5")
            if focused["metadata"]["runtime_fingerprint"]!=base["metadata"]["runtime_fingerprint"]:
                raise ValueError("Inference runtime changed between rescue stages")
            probabilities=combine(values,[focused["probabilities"][v] for v in CATEGORIES.values()],config,evidence["general_similarity"])
        canonical=dict(zip(CATEGORIES.values(),probabilities))
        probabilities=[canonical[option] for option in options]
        original=dict(zip(options,probabilities))
        calibration=None
        if calibration_profile is not None:
            probabilities,calibration=adapter._calibration_profiles.apply(calibration_profile,probabilities,question=question,options=options,
                modality="text",runtime_fingerprint=base["metadata"]["runtime_fingerprint"],prompt_version=VERSION,
                task_profile=NAME,task_profile_fingerprint=fingerprint)
        index=max(range(len(probabilities)),key=probabilities.__getitem__)
        sources=[base]+([focused] if focused else [])
        usage={key:sum((r["metadata"].get("usage") or {}).get(key,0) for r in sources)
               for key in ("prompt_tokens","completion_tokens","total_tokens")}
        result={"prediction":options[index],"prediction_index":index,"confidence":probabilities[index],"probabilities":dict(zip(options,probabilities)),
                "component_probabilities":{"base_raw":values,"focused_raw":[focused["probabilities"][v] for v in CATEGORIES.values()] if focused else None,
                    "reviewed":active,"training_evidence":evidence,"weight":config["weight"],"threshold":config["threshold"],"minimum_probability":config["minimum_probability"],
                    "component_temperature":config["component_temperature"],"protected_domains":config["protected_domains"],"minimum_general_similarity":config["minimum_general_similarity"],"class_order":list(CATEGORIES)},
                "metadata":base["metadata"]|{"method":"gemma_general_precision_rescue","prompt_version":VERSION,"task_profile":NAME,
                    "task_profile_fingerprint":fingerprint,"calibrated":calibration is not None,"calibration":calibration,
                    "retrieved_example_ids":list(dict.fromkeys(e for r in sources for e in r["metadata"]["retrieved_example_ids"])),
                    "source_metadata":[r["metadata"] for r in sources],"llm_calls":len(sources),"usage":usage,"elapsed_seconds":round(time.monotonic()-started,4)}}
        if calibration_profile is not None:
            result["raw_probabilities"]=original
        return result
