"""Gated supervised support for general, lists and news; Gemma remains a component."""
import json
import math

from .calibration import sha256_file, temperature_scale
from .config import ROOT
from .eval_data import CATEGORIES, digest

NAME = "massive_es_v3"
VERSION = "jev-qat-massive-es-focus-v3"
PROFILE_PATH = ROOT / "classification" / "profiles" / f"{NAME}.json"
FOCUS = ("general", "lists", "news")


def combine_probabilities(gemma, lexical, weight, component_temperature):
    if len(gemma) != len(CATEGORIES) or len(lexical) != len(CATEGORIES):
        raise ValueError("Both classifier components must score all 18 categories")
    weights = list(weight.values()) if isinstance(weight, dict) else [weight]
    if isinstance(weight, dict) and set(weight) != set(FOCUS):
        raise ValueError("Per-category weights must cover the three focus categories")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in weights):
        raise ValueError("Mixture weight must be in [0,1]")
    if any(not math.isfinite(value) or value < 0 for value in lexical) or not math.isclose(math.fsum(lexical), 1, abs_tol=1e-6):
        raise ValueError("Invalid lexical classifier probabilities")
    calibrated_gemma = temperature_scale(gemma, component_temperature)
    focus_indices = {list(CATEGORIES).index(label) for label in FOCUS}
    gate = max(range(18), key=calibrated_gemma.__getitem__) in focus_indices or max(range(18), key=lexical.__getitem__) in focus_indices
    gemma_index = max(range(18), key=calibrated_gemma.__getitem__)
    lexical_index = max(range(18), key=lexical.__getitem__)
    dominant = list(CATEGORIES)[gemma_index if gemma_index in focus_indices else lexical_index]
    effective = (weight.get(dominant, 0.0) if isinstance(weight, dict) else weight) if gate else 0.0
    values = [(1-effective)*a+effective*b for a,b in zip(calibrated_gemma, lexical)]
    total = math.fsum(values)
    return [value/total for value in values], gate


class FocusedTask:
    def __init__(self, base_task):
        self.base_task = base_task
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import joblib
            config = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            if config["version"] != VERSION:
                raise ValueError("Focused classifier version mismatch")
            head = ROOT / config["head_file"]
            if sha256_file(head) != config["head_sha256"]:
                raise ValueError("Focused classifier checksum mismatch")
            _, bank, base_fingerprint = self.base_task._load()
            if base_fingerprint != config["base_task_fingerprint"]:
                raise ValueError("Frozen Gemma task fingerprint mismatch")
            model = joblib.load(head)["model"]
            if set(model.classes_) != set(CATEGORIES):
                raise ValueError("Focused classifier has the wrong categories")
            fingerprint = digest({"profile": sha256_file(PROFILE_PATH), "implementation": sha256_file(__file__), "base": base_fingerprint})
            self._loaded = (config, bank, model, fingerprint)
        return self._loaded

    def prepare(self, name, **arguments):
        if name != NAME:
            raise ValueError("Unknown focused task_profile")
        # Reuse byte-identical v2 evidence and prompt construction. Its bank has
        # no development/calibration/validation/test examples from iteration 3.
        base = self.base_task.prepare("massive_es_v2", **arguments)
        config, bank, model, fingerprint = self._load()
        from sklearn.preprocessing import normalize
        features = normalize(bank["vectorizer"].transform([arguments["state"]]))
        values = model.predict_proba(features)[0]
        lexical = [float(values[list(model.classes_).index(label)]) for label in CATEGORIES]
        return base | {"name": NAME, "version": VERSION, "fingerprint": fingerprint,
                       "lexical_probabilities": lexical, "mixture_weight": config["weight"],
                       "gemma_component_temperature": config["gemma_component_temperature"]}
