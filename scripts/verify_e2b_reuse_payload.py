"""Prove the disabled optimization emits the archived adapter's exact requests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_e2b import mocked_adapter

OUT = ROOT / "inference/e2b-prepare-reuse"
spec = importlib.util.spec_from_file_location("jev_qat._archived_e2b", OUT / "baseline/jev_qat/e2b.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

current, current_calls = mocked_adapter()
old = module.E2BQatAdapter()
old.client.close()
old_calls = []
transport = current.client._transport


def handler(request):
    response = transport.handle_request(request)
    if request.method == "POST":
        old_calls.append((request.url.path, json.loads(request.content)))
        current_calls.pop()  # Same mock tokenizer/score generator, independent call lists.
    return response


old.client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
with tempfile.TemporaryDirectory(prefix="e2b-payload-check-") as tmp:
    path = Path(tmp) / "photo.png"
    path.write_bytes(b"same image bytes")
    current.config["media_roots"] = old.config["media_roots"] = [tmp]
    with current, old:
        for mode in ("text", "image"):
            for question, options in (("Question A?", ["yes", "no"]), ("Question B?", ["alpha", "beta", "gamma"])):
                request = {"state": "same state", "question": question, "options": options,
                           "modality": mode, "cache_prompt": False}
                if mode == "image":
                    request["media"] = str(path)
                old.count_tokens(**request)
                current.count_tokens(**request, reuse_media=False)
                a = old.predict(**request)
                b = current.predict(**request, reuse_media=False)
                assert a["probabilities"] == b["probabilities"]
assert old_calls == current_calls
result = {"passed": True, "cases": 4, "all_native_post_payloads_identical": True, "native_post_requests_compared": len(old_calls),
          "network_inference_executed": False, "requests": current_calls}
(OUT / "baseline-payload-equivalence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps({k: v for k, v in result.items() if k != "requests"}))
