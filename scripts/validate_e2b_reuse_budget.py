"""Live check: prepared media never bypasses the full per-question context limit."""
import hashlib
import json
from pathlib import Path
import re
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_multimodal_resources import cases

OUT = ROOT / "inference/e2b-prepare-reuse"
source = next(payload["media"] for name, _, payload, _ in cases() if name == "image")
source = Path(source)
data = source.read_bytes()
before_hash = hashlib.sha256(data).hexdigest()
log = ROOT / "integration/jev-qat-e2b/backend.log"
with httpx.Client(base_url="http://127.0.0.1:18167", timeout=180, trust_env=False) as client:
    upload = client.post("/uploads", params={"filename": source.name}, content=data)
    upload.raise_for_status()
    receipt = upload.json()
    try:
        body = {"state": "", "question": "¿Qué código hay en la imagen?", "options": ["5831", "1385"],
                "modality": "image", "media": receipt["media"], "cache_prompt": True, "reuse_media": True}

        def count(request):
            response = client.post("/count-tokens", json=request)
            response.raise_for_status()
            return response.json()

        small = count(body)
        assert small["fits_context"] and not small["inference_executed"] and not small["media_preparation"]["hit"]
        too_large = {**body, "state": "palabra " * 5000}
        preview = count(too_large)
        assert preview["media_preparation"]["hit"] and not preview["fits_context"]
        # The native runtime logs slot execution, but does not log token-count HTTP routes.
        # First demonstrate that a valid inference produces a slot-launch log entry.
        positive_offset = log.stat().st_size
        positive = client.post("/predict", json={**body, "cache_prompt": False})
        positive.raise_for_status()
        assert positive.json()["metadata"]["usage"]["completion_tokens"] == 1
        with log.open("rb") as stream:
            stream.seek(positive_offset)
            positive_log = stream.read().decode("utf8", errors="replace")
        assert re.search(r"launch_slot_.*processing task", positive_log)
        offset = log.stat().st_size
        rejected = client.post("/predict", json=too_large)
        assert rejected.status_code == 422
        detail = rejected.json()["detail"]
        assert detail["code"] == "context_limit_exceeded" and detail["total_tokens"] == preview["total_tokens"] > 4096
        with log.open("rb") as stream:
            stream.seek(offset)
            tail = stream.read().decode("utf8", errors="replace")
        slot_launches = len(re.findall(r"launch_slot_.*processing task", tail))
        assert slot_launches == 0 and not tail.strip()
        after = count(body)
        assert after["media_preparation"]["hit"] and after["fits_context"] and after["input_tokens"] == small["input_tokens"]
        ui = client.get("/ui/e2b.js")
        assert ui.status_code == 200 and "cache_prompt: Boolean(shared.media)" in ui.text and "reuse_media: true" in ui.text
        report = {"passed": True, "small_budget": small, "oversized_budget_using_same_prepared_file": preview,
                  "predict_status": rejected.status_code, "rejection": detail, "small_budget_after_rejection": after,
                  "native_inference_slots_started_for_overflow": slot_launches,
                  "native_token_counter": detail["counter"], "native_logging_positive_control": True,
                  "ui_uses_preparation_and_prefix_reuse": True, "source_sha256": before_hash}
    finally:
        client.delete("/uploads/"+receipt["upload_id"]).raise_for_status()
    assert not Path(receipt["media"]).exists() and hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
    assert all(client.get(f"http://127.0.0.1:{port}/health").status_code == 200 for port in (18165, 18167))
    (OUT / "native-budget-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"passed": True, "oversized_input_tokens": preview["input_tokens"], "status": rejected.status_code,
                      "no_native_inference_started": slot_launches == 0, "ui_reuse_enabled": True}))
