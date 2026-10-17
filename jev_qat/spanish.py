"""Convenience API for the evaluated Spanish message classifier."""
import json

import httpx

from .config import ROOT, settings
from .eval_data import CATEGORIES, QUESTION
from .spanish_task import DEFINITIONS


def classify_spanish_message(message, *, calibrated=True, api_url=None, version=None):
    """Classify using the active evaluated profile, or explicitly select v2-v9.

    Returns the normal adapter response plus category IDs and their probabilities.
    The local service must be running; no remote inference or fallback is used.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty Spanish string")
    if not isinstance(calibrated, bool):
        raise ValueError("calibrated must be a boolean")
    if version is None:
        current = ROOT / "classification" / "current.json"
        version = json.loads(current.read_text(encoding="utf-8"))["version"] if current.exists() else "v2"
    if version not in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        raise ValueError("version must be v2, v3, v4, v5, v6, v7, v8 or v9")
    config = settings()
    payload = {"state": message, "question": QUESTION, "options": list(CATEGORIES.values()),
               "modality": "text", "task_profile": f"massive_es_{version}"}
    if calibrated:
        payload["calibration_profile"] = f"massive_scenario_es_{version}"
    with httpx.Client(base_url=api_url or config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
        response = client.post("/predict", json=payload)
        response.raise_for_status()
        result = response.json()
    result["category"] = next(key for key, description in CATEGORIES.items() if description == result["prediction"])
    result["category_description"] = DEFINITIONS[result["category"]]
    result["category_probabilities"] = {key: result["probabilities"][description] for key, description in CATEGORIES.items()}
    return result
