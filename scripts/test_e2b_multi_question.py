"""Verify independent question requests over one shared text or uploaded file."""
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


def main():
    config = settings()
    report = {"cases": [], "checks": {}}
    text = {"state": "Añade pan a mi lista de compras para hoy. No es urgente.", "question": "¿Qué tipo de solicitud es?",
            "options": ["Listas", "Noticias", "General"], "cache_prompt": False}
    tests = [("text", text, "Listas", "¿El mensaje indica que es urgente?")]
    for name, _, payload, expected in cases():
        if name in ("image", "audio", "video_2"):
            tests.append((name, {**payload, "cache_prompt": False}, expected, {
                "image": "¿Aparece el código 5831 en la imagen?", "audio": "¿Se menciona el número cuarenta y dos en el audio?",
                "video_2": "¿La figura central es un triángulo amarillo?"}[name]))
    with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
        for name, first, expected, second_prompt in tests:
            receipt = source = original_hash = None
            if first.get("media"):
                source = Path(first["media"])
                original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                response = client.post("/uploads", params={"filename": source.name}, content=source.read_bytes())
                response.raise_for_status(); receipt = response.json()
                first = {**first, "media": receipt["media"]}
            try:
                second = {**first, "question": second_prompt, "options": ["Sí", "No"]}
                rows = []
                for request in (first, second):
                    response = client.post("/count-tokens", json=request)
                    response.raise_for_status(); budget = response.json()
                    assert budget["fits_context"] is True
                    response = client.post("/predict", json=request)
                    response.raise_for_status(); result = response.json()
                    assert set(result["probabilities"]) == set(request["options"])
                    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
                    assert result["metadata"]["usage"]["prompt_tokens"] == budget["input_tokens"]
                    assert result["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                    rows.append({"question": request["question"], "options": request["options"], "budget": budget, "result": result})
                assert rows[0]["result"]["prediction"] == expected
                excessive = {**first, "question": " dato" * 5000}
                response = client.post("/count-tokens", json=excessive)
                response.raise_for_status(); blocked = response.json()
                assert blocked["fits_context"] is False
                assert client.post("/predict", json=excessive).status_code == 422
                # One oversized question must not alter another question's count.
                after = client.post("/count-tokens", json=second).json()
                assert after["fits_context"] is True and after["input_tokens"] == rows[1]["budget"]["input_tokens"]
                report["cases"].append({"modality": name, "shared_upload_id": receipt["upload_id"] if receipt else None,
                                        "original_sha256": original_hash, "independent_questions": rows, "blocked_question_budget": blocked})
            finally:
                if receipt:
                    assert client.delete("/uploads/" + receipt["upload_id"]).status_code == 204
                    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
        report["checks"]["four_modalities_two_independent_vectors_each"] = len(report["cases"]) == 4
        report["checks"]["oversized_questions_isolated"] = all(not r["blocked_question_budget"]["fits_context"] for r in report["cases"])
        report["checks"]["e4b_and_e2b_ready"] = all(client.get(f"http://127.0.0.1:{port}/health").status_code == 200 for port in (18165, 18167))
    report["passed"] = all(report["checks"].values())
    save_json(ROOT / "inference" / "e2b-multi-question" / "api-results.json", report)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "native_inferences": 8}), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
