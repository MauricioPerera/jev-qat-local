"""Conservative focus expert: require positive, sufficiently confident evidence."""
import json
import math

from .calibration import sha256_file, temperature_scale
from .config import ROOT
from .eval_data import CATEGORIES, digest
from .focus_task import FOCUS

NAME = "massive_es_v4"
VERSION = "jev-qat-massive-es-selective-v4"
PROFILE_PATH = ROOT / "classification/profiles/massive_es_v4.json"


def combine_probabilities(gemma, lexical, weight, component_temperature, threshold):
    if len(gemma) != 18 or len(lexical) != 18:
        raise ValueError("Both components must score all 18 categories")
    if any(not math.isfinite(x) or x < 0 for x in lexical) or not math.isclose(math.fsum(lexical),1,abs_tol=1e-6):
        raise ValueError("Invalid lexical probabilities")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (weight,threshold)):
        raise ValueError("Weight and threshold must be in [0,1]")
    calibrated = temperature_scale(gemma, component_temperature)
    index = max(range(18), key=lexical.__getitem__)
    active = list(CATEGORIES)[index] in FOCUS and lexical[index] >= threshold
    values = [(1-weight)*a+weight*b for a,b in zip(calibrated,lexical)] if active else calibrated
    total = math.fsum(values)
    return [x/total for x in values], active


class SelectiveTask:
    def __init__(self, base_task):
        self.base_task = base_task
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import joblib
            config = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            if config["version"] != VERSION or sha256_file(ROOT/config["head_file"]) != config["head_sha256"]:
                raise ValueError("Selective classifier version/checksum mismatch")
            _, bank, base_fingerprint = self.base_task._load()
            if base_fingerprint != config["base_task_fingerprint"]:
                raise ValueError("Frozen Gemma component changed")
            model = joblib.load(ROOT/config["head_file"])["model"]
            if set(model.classes_) != set(CATEGORIES):
                raise ValueError("Selective classifier category mismatch")
            fingerprint = digest({"profile":sha256_file(PROFILE_PATH),"implementation":sha256_file(__file__),"base":base_fingerprint})
            self._loaded = config,bank,model,fingerprint
        return self._loaded

    def prepare(self, name, **arguments):
        if name != NAME:
            raise ValueError("Unknown selective task_profile")
        base = self.base_task.prepare("massive_es_v2", **arguments)
        config,bank,model,fingerprint = self._load()
        from sklearn.preprocessing import normalize
        values = model.predict_proba(normalize(bank["vectorizer"].transform([arguments["state"]])))[0]
        lexical = [float(values[list(model.classes_).index(label)]) for label in CATEGORIES]
        return base | {"name":NAME,"version":VERSION,"fingerprint":fingerprint,"lexical_probabilities":lexical,
                       "mixture_weight":config["weight"],"gemma_component_temperature":config["gemma_component_temperature"],
                       "selective_threshold":config["threshold"]}
