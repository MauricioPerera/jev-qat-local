"""Exercise the running HTTP bridge and Python adapter on local fixtures."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from jev_qat import load_jev_omni_qat
from jev_qat.config import STATE, save_json, settings


def main():
    config = settings()
    fixture_dir = ROOT / "inference" / "qat-q4-vulkan"
    video = STATE / "test-video.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1",
        "-i", str(fixture_dir / "test-image.png"), "-t", "2", "-r", "2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    cases = [
        ("text", {"state": "La reunión empieza a las 10:00. Ahora son las 09:00.",
                  "question": "¿Ya empezó la reunión?", "options": ["Sí", "No"]}, "No"),
        ("text_reordered", {"state": "La reunión empieza a las 10:00. Ahora son las 09:00.",
                            "question": "¿Ya empezó la reunión?", "options": ["No", "Sí"]}, "No"),
        ("image", {"state": "", "question": "¿Qué código numérico aparece arriba en la imagen?",
                   "options": ["8513", "5831", "1385"], "modality": "image", "media": str(fixture_dir / "test-image.png")}, "5831"),
        ("audio", {"state": "", "question": "¿Qué número se menciona en el audio?",
                   "options": ["17", "42", "81"], "modality": "audio", "media": str(fixture_dir / "test-audio.wav")}, "42"),
        ("video", {"state": "", "question": "¿Qué figura aparece en el centro del video?",
                   "options": ["círculo azul", "triángulo amarillo", "cuadrado rojo"],
                   "modality": "video", "media": str(video), "video_frames": 2}, "triángulo amarillo"),
        ("256_options", {"state": "El identificador correcto es item_173.",
                         "question": "Selecciona el identificador correcto, exactamente como aparece en el estado.",
                         "options": [f"item_{index:03}" for index in range(256)]}, "item_173"),
    ]
    report = {"service": "jev-qat-bridge", "tests": [], "passed": False,
              "calibrated": False, "original_jev_omni_head_used": False}
    with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
        assert client.get("/health").json()["service"] == "jev-qat-bridge"
        for name, payload, expected in cases:
            started = time.monotonic()
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            probabilities = result["probabilities"]
            checks = {
                "prediction": result["prediction"] == expected,
                "index": payload["options"][result["prediction_index"]] == result["prediction"],
                "complete_scores": set(probabilities) == set(payload["options"]),
                "normalized": abs(math.fsum(probabilities.values()) - 1) < 1e-6,
                "valid_probabilities": all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()),
                "uncalibrated_explicit": result["metadata"]["calibrated"] is False,
                "no_original_head_claim": result["metadata"]["original_jev_omni_head_used"] is False,
                "one_output_token": result["metadata"]["usage"]["completion_tokens"] == 1,
            }
            if payload.get("modality") == "video":
                checks["two_frames"] = result["metadata"]["media_items"] == 2
            item = {"case": name, "expected": expected, "passed": all(checks.values()), "checks": checks,
                    "wall_seconds": round(time.monotonic() - started, 3), "result": result}
            report["tests"].append(item)
            save_json(STATE / "integration-results.json", report)
            print(json.dumps({"case": name, "passed": item["passed"], "prediction": result["prediction"],
                              "confidence": result["confidence"], "seconds": item["wall_seconds"]}, ensure_ascii=True), flush=True)
        bad = client.post("/predict", json={"state": "", "question": "Which?", "options": ["same", "same"]})
        report["duplicate_options_rejected"] = bad.status_code == 422
        unauthorized = client.post("/admin/stop")
        report["management_token_required"] = unauthorized.status_code == 403
    with load_jev_omni_qat() as adapter:
        result = adapter.predict(**cases[0][1])
        report["python_interface_passed"] = result["prediction"] == cases[0][2]
    report["passed"] = (all(item["passed"] for item in report["tests"])
                        and report["duplicate_options_rejected"] and report["management_token_required"]
                        and report["python_interface_passed"])
    save_json(STATE / "integration-results.json", report)
    print(f"All integration checks passed: {report['passed']}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
