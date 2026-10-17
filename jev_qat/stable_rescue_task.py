"""Stable v9 native components explicitly disable prefix-cache reuse."""
from .calibration import sha256_file
from .eval_data import digest
from .general_task import GeneralTask
from .general_confirm_task import ConfirmationTask

FOCUS_NAME = "massive_es_v9_focus"
CONFIRM_NAME = "massive_es_v9_confirm"
FOCUS_VERSION = "jev-qat-stable-general-focus-v9"
CONFIRM_VERSION = "jev-qat-stable-general-confirm-v9"


class StableNativeTask:
    def __init__(self):
        self.focus = GeneralTask()
        self.confirm = ConfirmationTask()

    def prepare(self, name, **kwargs):
        if name == FOCUS_NAME:
            task = self.focus.prepare("massive_es_v5", **kwargs)
            version = FOCUS_VERSION
        elif name == CONFIRM_NAME:
            task = self.confirm.prepare("massive_es_v8_confirm", **kwargs)
            version = CONFIRM_VERSION
        else:
            raise ValueError("Unknown stable native component")
        fp = digest({"source": task["fingerprint"], "cache_prompt": False,
                     "implementation": sha256_file(__file__), "version": version})
        return task | {"name": name, "version": version, "fingerprint": fp, "cache_prompt": False}
