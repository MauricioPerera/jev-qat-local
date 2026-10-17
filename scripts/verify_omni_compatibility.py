"""Check Jev-Omni's published head against Gemma E4B without loading model weights."""

import hashlib
import json
from pathlib import Path

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "gemma-4-E4B-it"
UPSTREAM = ROOT / "compatibility" / "upstream-jev-omni"
HEAD_SHA256 = "8c81edecd733f7db327604803c14cf9ecbee53d2af055eeb09f78b64bcbf2638"


def main():
    target_config = json.loads((TARGET / "config.json").read_text(encoding="utf-8"))
    decision = json.loads((UPSTREAM / "decision_config.json").read_text(encoding="utf-8"))
    head_path = UPSTREAM / "head.pt"
    actual_sha256 = hashlib.sha256(head_path.read_bytes()).hexdigest()
    if actual_sha256 != HEAD_SHA256:
        raise ValueError("Published head checksum mismatch")
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    target_hidden = target_config["text_config"]["hidden_size"]
    target_head = torch.nn.Module()
    target_head.register_buffer("mu", torch.zeros(1, target_hidden))
    target_head.register_buffer("sd", torch.ones(1, target_hidden))
    target_head.linear = torch.nn.Linear(target_hidden, decision["output_classes"])
    try:
        target_head.load_state_dict(state, strict=True)
        load_error = None
    except RuntimeError as error:
        load_error = str(error)
    try:
        config = transformers.AutoConfig.from_pretrained(
            TARGET, local_files_only=True, trust_remote_code=False
        )
        config_class = type(config).__name__
        config_error = None
    except (ImportError, OSError, ValueError) as error:
        config_class = None
        config_error = str(error)
    cuda_available = torch.cuda.is_available()
    report = {
        "target_model": "google/gemma-4-E4B-it",
        "target_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "jev_model": "akhilaaa3/Jev-Omni",
        "jev_revision": "fee89b076e2263bfa8bce8d34142caa4ca733a48",
        "target_architecture": target_config["architectures"][0],
        "jev_base_model": decision["base_model"],
        "target_hidden_size": target_hidden,
        "jev_hidden_size": decision["hidden_size"],
        "published_head_shapes": {key: list(value.shape) for key, value in state.items()},
        "head_sha256_verified": True,
        "strict_head_load_succeeded": load_error is None,
        "strict_head_load_error": load_error,
        "target_has_jev_decision_config": (TARGET / "decision_config.json").exists(),
        "target_has_jev_head": (TARGET / "head.pt").exists(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "target_config_class": config_class,
        "target_config_error": config_error,
        "cuda_available": cuda_available,
        "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "reference_loader_requires_cuda": True,
        "reference_loader_hardware_supported": cuda_available,
        "full_model_inference_tested": False,
        "verdict": "not_drop_in_compatible" if load_error else "head_shapes_compatible_only",
        "integration_path": "Train an E4B-specific decision head and adapt the CUDA-only loader for the selected backend.",
    }
    report_path = ROOT / "compatibility" / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
