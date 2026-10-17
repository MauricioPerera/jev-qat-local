"""Live upload -> count -> predict -> remove checks for the local E2B page."""
import hashlib
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.config import save_json
from jev_qat.e2b import settings
from benchmark_multimodal_resources import cases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = settings()
    report = {"checks": {}, "tests": []}
    uploads = (ROOT / "integration" / "jev-qat-e2b" / "uploads").resolve()
    with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
        page = client.get("/")
        report["checks"]["ui_served"] = page.status_code == 200 and "Varias decisiones." in page.text
        report["checks"]["assets_served"] = all(client.get(path).status_code == 200 for path in ("/ui/e2b.css", "/ui/e2b.js"))
        for name, _, payload, expected in cases():
            if name not in ("image", "audio", "video_2"):
                continue
            source = Path(payload["media"])
            before = sha(source)
            response = client.post("/uploads", params={"filename": source.name}, content=source.read_bytes(),
                                   headers={"Content-Type": "application/octet-stream", "Origin": config["api_url"]})
            response.raise_for_status()
            receipt = response.json()
            copied = Path(receipt["media"])
            try:
                assert copied.parent == uploads and sha(copied) == before
                assert receipt["modality"] == payload["modality"]
                request = {**payload, "media": receipt["media"], "cache_prompt": False}
                count = client.post("/count-tokens", json=request)
                count.raise_for_status()
                budget = count.json()
                assert budget["fits_context"] is True and budget["inference_executed"] is False
                inferred = client.post("/predict", json=request)
                inferred.raise_for_status()
                result = inferred.json()
                assert result["metadata"]["usage"]["prompt_tokens"] == budget["input_tokens"]
                assert result["prediction"] == expected
                report["tests"].append({"case": name, "original_sha256": before, "upload_bytes": receipt["bytes"],
                                        "budget": budget, "prediction": result})
            finally:
                removed = client.delete("/uploads/" + receipt["upload_id"], headers={"Origin": config["api_url"]})
                assert removed.status_code == 204 and not copied.exists()
                assert sha(source) == before
        oversized = {"state": " dato" * 5000, "question": "Elige una opción.", "options": ["a", "b"]}
        budget = client.post("/count-tokens", json=oversized).json()
        report["checks"]["over_limit_still_blocked_by_server"] = budget["fits_context"] is False and client.post("/predict", json=oversized).status_code == 422
        report["checks"]["original_files_preserved_and_copies_removed"] = len(report["tests"]) == 3
        report["checks"]["both_models_ready"] = all(client.get(f"http://127.0.0.1:{port}/health").status_code == 200 for port in (18165, 18167))
    report["passed"] = all(report["checks"].values())
    save_json(ROOT / "inference" / "e2b-web-ui" / "api-results.json", report)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "modalities": [t["case"] for t in report["tests"]]}), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
