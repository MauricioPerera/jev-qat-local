"""Live E2B HTTP/Python contract checks; content answers recorded separately."""
import json
import math
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multimodal_resources import cases
from jev_qat.config import save_json
from jev_qat.e2b import classify_spanish_e2b, load_jev_omni_e2b_qat, settings


def main():
    config = settings()
    report = {"checks": {}, "tests": [], "content_checks_are_not_accuracy_benchmark": True}
    with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
        health = client.get("/health")
        report["checks"]["ready_and_correct_model"] = health.status_code == 200 and health.json()["backend"] == "gemma-e2b-qat-q4-local"
        for identifier, label, payload, expected in cases():
            if identifier == "video_16":
                continue
            payload = dict(payload)
            payload.pop("calibration_profile", None)
            payload["cache_prompt"] = False
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            probabilities = result["probabilities"]
            metadata = result["metadata"]
            checks = {
                "all_option_scores": set(probabilities) == set(payload["options"]),
                "finite_normalized": all(math.isfinite(p) and p >= 0 for p in probabilities.values()) and abs(sum(probabilities.values())-1) < 1e-6,
                "valid_prediction_index": payload["options"][result["prediction_index"]] == result["prediction"],
                "true_e2b_identity": metadata["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf",
                "uncalibrated": metadata["calibrated"] is False and metadata["calibration"] is None,
                "one_output_token": metadata["usage"]["completion_tokens"] == 1,
                "zero_cached_tokens": metadata["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
                "media_item_count": metadata["media_items"] == (payload.get("video_frames", 1) if payload.get("modality") in ("image", "audio", "video") else 0),
            }
            report["tests"].append({"case": identifier, "checks": checks, "contract_passed": all(checks.values()),
                                    "expected_source_content": expected, "content_matches": result["prediction"] == expected,
                                    "seconds": time.perf_counter()-started, "result": result})
        request = {"state": "El identificador correcto es item_173.", "question": "Selecciona el identificador correcto exactamente como aparece en el estado.",
                   "options": [f"item_{i:03}" for i in range(256)], "cache_prompt": False}
        response = client.post("/predict", json=request)
        response.raise_for_status()
        large = response.json()
        report["checks"]["256_option_scores"] = set(large["probabilities"]) == set(request["options"]) and abs(sum(large["probabilities"].values())-1) < 1e-6
        report["256_options_result"] = large
        base = {"state": "hola", "question": "¿Es un saludo?", "options": ["sí", "no"]}
        for name, extra in (("e4b_calibration_rejected", {"calibration_profile": "massive_scenario_es_v2"}),
                            ("e4b_rescue_rejected", {"task_profile": "massive_es_v9"}),
                            ("invalid_cache_type_rejected", {"cache_prompt": "false"}),
                            ("duplicate_options_rejected", {"options": ["sí", "sí"]})):
            response = client.post("/predict", json={**base, **extra})
            report["checks"][name] = response.status_code == 422
        report["checks"]["stop_requires_management_token"] = client.post("/admin/stop").status_code == 403
    with load_jev_omni_e2b_qat() as adapter:
        result = adapter.predict(state="La reunión empieza a las diez. Ahora son las nueve.", question="¿Ya empezó la reunión?", options=["sí", "no"], cache_prompt=False)
        report["checks"]["python_interface_identity"] = result["metadata"]["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf"
    report["spanish_helper_result"] = classify_spanish_e2b("Añade pan a mi lista de compras", cache_prompt=False)
    report["checks"]["spanish_helper_uncalibrated"] = report["spanish_helper_result"]["metadata"]["calibrated"] is False
    report["passed"] = all(report["checks"].values()) and all(t["contract_passed"] for t in report["tests"])
    save_json(ROOT / "integration" / "jev-qat-e2b" / "integration-results.json", report)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "content_matches": [(t["case"],t["content_matches"]) for t in report["tests"]]}, ensure_ascii=True), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
