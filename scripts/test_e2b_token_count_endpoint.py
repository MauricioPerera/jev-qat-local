"""Exercise the live /count-tokens contract without triggering inference."""
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.config import save_json
from jev_qat.e2b import E2BQatAdapter, settings
from benchmark_multimodal_resources import cases

OUT = ROOT / "inference" / "e2b-token-count-endpoint"


def fill_to(adapter, payload, target):
    n = 0
    for _ in range(6):
        request = {**payload, "state": " dato" * n}
        budget = adapter.count_tokens(**request)
        if budget["input_tokens"] == target:
            return request, budget
        n += target - budget["input_tokens"]
        assert n >= 0
    raise AssertionError("Could not construct exact token boundary")


def log_after(log, offset):
    return log.read_bytes()[offset:].decode("utf8", errors="replace")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = settings()
    report = {"checks": {}, "counted": [], "oversized": [], "inferred_controls": []}
    controls = {}
    test_cases = list(cases())
    plan = ROOT / "inference" / "gemma-e2b-vs-e4b" / "plan.json"
    if plan.is_file():
        for case in json.loads(plan.read_text(encoding="utf8"))["cases"]:
            if case["id"] in ("image_alps", "audio_30", "video_60_16"):
                test_cases.append((case["id"], case["label"], case["payload"], None))
    baseline_path = ROOT / "inference" / "e2b-context-budget" / "integration-results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf8"))
    report["baseline_native_inference_evidence_sha256"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    native_counts = {r["case"]: r["usage"]["prompt_tokens"] for r in baseline["accepted"]}
    log = ROOT / "integration" / "jev-qat-e2b" / "backend.log"
    offset = log.stat().st_size
    with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client, E2BQatAdapter() as adapter:
        schema = client.get("/openapi.json").json()
        assert "/count-tokens" in schema["paths"], "Restart E2B with the new endpoint first"
        paths = schema["paths"]
        report["checks"]["same_request_schema_as_predict"] = paths["/count-tokens"]["post"]["requestBody"] == paths["/predict"]["post"]["requestBody"]
        report["checks"]["documented_response_schema"] = "E2BTokenBudgetResponse" in schema["components"]["schemas"]
        for name, _, payload, _ in test_cases:
            payload = {k: v for k, v in payload.items() if k != "calibration_profile"}
            payload["cache_prompt"] = False
            controls[name] = payload
            started = time.monotonic()
            response = client.post("/count-tokens", json=payload)
            response.raise_for_status()
            budget = response.json()
            assert budget["inference_executed"] is False and budget["fits_context"] is True
            assert budget["input_tokens"] == native_counts[name]
            assert budget["total_tokens"] == budget["input_tokens"] + 1
            assert budget["available_tokens"] == 4096 - budget["total_tokens"] == budget["remaining_tokens"]
            assert budget["overflow_tokens"] == 0 and budget["option_count"] == len(payload["options"])
            assert budget["modality"] == payload.get("modality", "text")
            report["counted"].append({"case": name, "status": response.status_code, "budget": budget,
                                      "seconds": time.monotonic() - started, "matches_previously_verified_native_count": True})
            print(json.dumps({"case": name, "input_tokens": budget["input_tokens"], "available_tokens": budget["available_tokens"]}), flush=True)

        base = dict(state="", question="Elige una respuesta.", options=["a", "b"], cache_prompt=False)
        boundary, _ = fill_to(adapter, base, 4095)
        response = client.post("/count-tokens", json=boundary)
        response.raise_for_status()
        budget = response.json()
        report["checks"]["exact_boundary_fits_with_no_space_left"] = budget["fits_context"] is True and budget["total_tokens"] == 4096 and budget["available_tokens"] == 0
        too_large, _ = fill_to(adapter, base, 4096)
        overflow = {"text_boundary": too_large,
                    "long_options": {**base, "options": [" dato" * 2200 + " a", " dato" * 2200 + " b"]}}
        for name in ("image", "audio", "video_2"):
            media = controls[name]
            text = {k: v for k, v in media.items() if k not in ("media", "modality", "video_frames")}
            padded, _ = fill_to(adapter, text, 4090)
            overflow[name] = {**media, "state": padded["state"]}
        for name, payload in overflow.items():
            response = client.post("/count-tokens", json=payload)
            assert response.status_code == 200
            budget = response.json()
            assert budget["fits_context"] is False and budget["available_tokens"] == 0
            assert budget["overflow_tokens"] == -budget["remaining_tokens"] > 0
            assert budget["inference_executed"] is False
            rejected = client.post("/predict", json=payload)
            assert rejected.status_code == 422
            assert rejected.json()["detail"]["input_tokens"] == budget["input_tokens"]
            report["oversized"].append({"case": name, "count_http_status": 200, "predict_http_status": 422, "budget": budget})

        invalid = [
            {"cache_prompt": "false"}, {"video_frames": 17}, {"options": ["a", "a"]},
            {"task_profile": "massive_es_v9"}, {"calibration_profile": "massive_scenario_es_v2"},
            {"modality": "image", "media": str(ROOT.parent / "outside.png")},
        ]
        statuses = [client.post("/count-tokens", json={**base, **extra}).status_code for extra in invalid]
        report["checks"]["invalid_requests_return_422"] = statuses == [422] * len(invalid)
        untouched_log = log_after(log, offset)
        report["checks"]["no_native_inference_for_counts_and_rejections"] = "slot launch_slot_" not in untouched_log
        (OUT / "count-only-backend.log").write_text(untouched_log, encoding="utf8")

        # Confirm count-only mode did not leak, and compare with fresh inference.
        for name in ("text_short", "image", "audio", "video_2"):
            response = client.post("/predict", json=controls[name])
            response.raise_for_status()
            result = response.json()
            previous = next(row["budget"] for row in report["counted"] if row["case"] == name)
            assert result["metadata"]["usage"]["prompt_tokens"] == previous["input_tokens"]
            assert result["metadata"]["context_budget"]["total_tokens"] == previous["total_tokens"]
            assert result["metadata"]["usage"]["completion_tokens"] == 1
            report["inferred_controls"].append({"case": name, "result": result, "count_matches_inference": True})
        report["checks"]["only_four_explicit_controls_performed_inference"] = log_after(log, offset).count("slot launch_slot_") == 4
        report["checks"]["all_modalities_counted"] = {r["budget"]["modality"] for r in report["counted"]} == {"text", "image", "audio", "video"}
        report["checks"]["e2b_ready"] = client.get("/health").status_code == 200
        report["checks"]["e4b_ready"] = client.get("http://127.0.0.1:18165/health").status_code == 200
    report["passed"] = all(report["checks"].values())
    save_json(OUT / "integration-results.json", report)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "counted_cases": len(report["counted"]),
                      "oversized_cases": len(report["oversized"]), "fresh_inferred_controls": len(report["inferred_controls"])}), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
