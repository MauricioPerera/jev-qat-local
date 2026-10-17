"""Live checks of native multimodal counts, context boundaries and HTTP rejection."""
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
from jev_qat.e2b_budget import COUNT_ROUTE, ContextBudgetError
from benchmark_multimodal_resources import cases

OUT = ROOT / "inference" / "e2b-context-budget"


class PreviewComplete(Exception):
    """The transport hook stopped an accepted request before sending inference."""


class Trace:
    def __init__(self):
        self.preview = False
        self.requests = []

    def request(self, request):
        route = request.url.path
        if route not in (COUNT_ROUTE, "/v1/chat/completions"):
            return
        if route == "/v1/chat/completions" and self.preview:
            raise PreviewComplete()
        self.requests.append({"route": route, "body_sha256": hashlib.sha256(request.content).hexdigest()})


def preview(adapter, trace, payload):
    trace.preview = True
    try:
        adapter.predict(**payload)
        raise AssertionError("Preview unexpectedly performed inference")
    except PreviewComplete:
        return dict(adapter._context_budget)
    except ContextBudgetError as error:
        return {k: v for k, v in error.detail.items() if k not in ("code", "message")}
    finally:
        trace.preview = False


def fill_to(adapter, trace, payload, target):
    n = 0
    for _ in range(6):
        request = {**payload, "state": " dato" * n}
        budget = preview(adapter, trace, request)
        if budget["input_tokens"] == target:
            return request, budget
        n += target - budget["input_tokens"]
        assert n >= 0
    raise AssertionError("Could not construct an exact native token boundary")


def assert_rejected(adapter, trace, payload):
    before = len(trace.requests)
    try:
        adapter.predict(**payload)
    except ContextBudgetError as error:
        calls = trace.requests[before:]
        assert [r["route"] for r in calls] == [COUNT_ROUTE]
        assert error.detail["total_tokens"] > error.detail["context_limit_tokens"]
        return error.detail
    raise AssertionError("Oversized request was not rejected")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = settings()
    report = {"checks": {}, "accepted": [], "rejected": []}
    trace = Trace()
    controls = {}
    test_cases = list(cases())
    plan = ROOT / "inference" / "gemma-e2b-vs-e4b" / "plan.json"
    if plan.is_file():
        for case in json.loads(plan.read_text(encoding="utf8"))["cases"]:
            if case["id"] in ("image_alps", "audio_30", "video_60_16"):
                test_cases.append((case["id"], case["label"], case["payload"], None))
    with E2BQatAdapter() as adapter:
        adapter.client.event_hooks["request"].append(trace.request)
        for name, _, payload, _ in test_cases:
            payload = {k: v for k, v in payload.items() if k != "calibration_profile"}
            payload["cache_prompt"] = False
            controls[name] = payload
            before = len(trace.requests)
            started = time.monotonic()
            result = adapter.predict(**payload)
            budget, usage = result["metadata"]["context_budget"], result["metadata"]["usage"]
            calls = trace.requests[before:]
            assert [c["route"] for c in calls] == [COUNT_ROUTE, "/v1/chat/completions"]
            assert calls[0]["body_sha256"] == calls[1]["body_sha256"]
            assert budget["input_tokens"] == usage["prompt_tokens"]
            assert usage["completion_tokens"] == 1 and usage["prompt_tokens_details"]["cached_tokens"] == 0
            report["accepted"].append({"case": name, "budget": budget, "usage": usage,
                                       "seconds": time.monotonic() - started, "same_count_and_inference_payload": True})
            print(json.dumps({"case": name, "preflight_tokens": budget["input_tokens"], "inference_tokens": usage["prompt_tokens"]}), flush=True)

        base = {"state": "", "question": "Elige una respuesta.", "options": ["a", "b"], "cache_prompt": False}
        boundary, _ = fill_to(adapter, trace, base, 4095)
        result = adapter.predict(**boundary)
        assert result["metadata"]["usage"]["prompt_tokens"] == 4095
        assert result["metadata"]["usage"]["completion_tokens"] == 1
        report["checks"]["exact_4095_input_plus_1_output_accepted"] = result["metadata"]["context_budget"]["remaining_tokens"] == 0

        too_large, _ = fill_to(adapter, trace, base, 4096)
        overflow = {"text_boundary": too_large,
                    "long_options": {**base, "options": [" dato" * 2200 + " a", " dato" * 2200 + " b"]}}
        # The text-only version fits; adding real prepared media crosses 4096.
        for name in ("image", "audio", "video_2"):
            media = controls[name]
            text_only = {k: v for k, v in media.items() if k not in ("media", "modality", "video_frames")}
            padded, text_budget = fill_to(adapter, trace, text_only, 4090)
            overflow[name] = {**media, "state": padded["state"]}
            report.setdefault("media_overflow_text_only_budgets", {})[name] = text_budget

        log = ROOT / "integration" / "jev-qat-e2b" / "backend.log"
        offset = log.stat().st_size
        for name, payload in overflow.items():
            detail = assert_rejected(adapter, trace, payload)
            report["rejected"].append({"case": name, "python": detail, "completion_requests_sent": 0})
        with httpx.Client(base_url=config["api_url"], trust_env=False, timeout=180) as client:
            health = client.get("/health").json()
            report["checks"]["live_service_has_budget_guard"] = health.get("context_budget_check") is True and health.get("context_limit_tokens") == 4096
            assert report["checks"]["live_service_has_budget_guard"], "Restart E2B with the updated adapter first"
            for entry in report["rejected"]:
                response = client.post("/predict", json=overflow[entry["case"]])
                assert response.status_code == 422
                detail = response.json()["detail"]
                assert detail["code"] == "context_limit_exceeded"
                assert detail["input_tokens"] == entry["python"]["input_tokens"]
                entry["http_status"], entry["http_detail"] = response.status_code, detail
            response = client.post("/predict", json=controls["text_short"])
            response.raise_for_status()
            result = response.json()
            report["checks"]["valid_http_request_includes_budget"] = result["metadata"]["context_budget"]["input_tokens"] == result["metadata"]["usage"]["prompt_tokens"]
        # Exactly one native task above belongs to the final valid HTTP control.
        new_log = log.read_bytes()[offset:].decode("utf8", errors="replace")
        (OUT / "rejection-and-http-control-backend.log").write_text(new_log, encoding="utf8")
        report["checks"]["no_native_tasks_for_rejections"] = new_log.count("slot launch_slot_") == 1
        report["checks"]["all_modalities_counted_exactly"] = {r["case"] for r in report["accepted"]} == set(controls)
        report["checks"]["all_oversized_modalities_and_options_blocked"] = len(report["rejected"]) == 5
    with httpx.Client(trust_env=False, timeout=5) as client:
        report["checks"]["e4b_and_e2b_ready"] = all(client.get(f"http://127.0.0.1:{p}/health").status_code == 200 for p in (18165, 18167))
    report["transport_trace"] = trace.requests
    report["passed"] = all(report["checks"].values())
    save_json(OUT / "integration-results.json", report)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "counts": {r["case"]: r["budget"]["input_tokens"] for r in report["accepted"]},
                      "rejected": {r["case"]: r["python"]["total_tokens"] for r in report["rejected"]}}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
