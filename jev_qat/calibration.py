"""Scoped temperature scaling; no change to the default uncalibrated adapter."""
import hashlib
import json
import math
from pathlib import Path
import re

from .config import ROOT

PROMPT_VERSION = "jev-qat-labels-v1"
PROFILES = ROOT / "calibration" / "profiles"


def temperature_scale(probabilities, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be positive and finite")
    if not probabilities or any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("Invalid probabilities for calibration")
    if math.fsum(probabilities) <= 0:
        raise ValueError("Calibration probabilities cannot all be zero")
    scores = [math.log(max(value, 1e-300)) / temperature for value in probabilities]
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CalibrationProfiles:
    def __init__(self, config):
        self.config = config
        self._fingerprints = {}

    def _fingerprint(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._fingerprints:
            self._fingerprints[key] = sha256_file(path)
        return self._fingerprints[key]

    def apply(self, name, probabilities, *, question, options, modality, runtime_fingerprint,
              prompt_version=None, task_profile=None, task_profile_fingerprint=None):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("Invalid calibration profile name")
        path = PROFILES / f"{name}.json"
        if not path.is_file():
            raise ValueError("Calibration profile does not exist")
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile.get("schema_version") != 1 or profile.get("validation_approved") is not True:
            raise ValueError("Calibration profile has not passed independent validation")
        scope = profile["scope"]
        expected = {
            "question": question, "options": sorted(options), "modality": modality,
            "prompt_version": prompt_version or PROMPT_VERSION, "runtime_fingerprint": runtime_fingerprint,
            "task_profile": task_profile, "task_profile_fingerprint": task_profile_fingerprint,
        }
        if any(scope.get(key) != value for key, value in expected.items()):
            raise ValueError("Calibration profile does not match this task, options, modality, prompt or runtime")
        for field in ("model", "projector"):
            if scope.get(field + "_sha256") != self._fingerprint(self.config[field]):
                raise ValueError("Calibration profile does not match the loaded model files")
        scaled = temperature_scale(probabilities, profile["temperature"])
        return scaled, {
            "profile": name, "method": "temperature_scaling", "temperature": profile["temperature"],
            "dataset": profile["dataset"], "validation_approved": True,
            "scope": "Spanish MASSIVE assistant-request categories; not a universal calibration",
        }
