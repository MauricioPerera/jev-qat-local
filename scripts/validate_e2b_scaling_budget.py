"""Live boundary/overflow/isolation tests after the resource measurement has ended."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.config import save_json
from jev_qat.e2b import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "inference/e2b-scaling-5-10-20")
    out = parser.parse_args().output.resolve()
    measured = json.loads((out / "results.json").read_text(encoding="utf8"))
    report = {"cases": [], "outside_performance_intervals": True, "checks": {}}
    log = ROOT / "integration/jev-qat-e2b/backend.log"
    with httpx.Client(base_url="http://127.0.0.1:18167", timeout=180, trust_env=False) as client:
        for case in measured["cases"]:
            source = Path(case["source"])
            data = source.read_bytes()
            uploaded = client.post("/uploads", params={"filename": source.name}, content=data)
            uploaded.raise_for_status()
            receipt = uploaded.json()
            row = {"case": case["id"], "upload_id": receipt["upload_id"], "fit_attempts": [], "groups": []}
            report["cases"].append(row)

            def body(q, state="", cache=True):
                return {"state": state, "question": q["question"], "options": q["options"],
                        "media": receipt["media"], "modality": case["modality"], "video_frames": case["video_frames"],
                        "reuse_media": True, "cache_prompt": cache}

            def count(request):
                response = client.post("/count-tokens", json=request)
                response.raise_for_status()
                value = response.json()
                assert not value["inference_executed"] and value["context_limit_tokens"] == 4096
                return value

            try:
                q = case["questions"][0]
                small = count(body(q))
                assert small["fits_context"] and not small["media_preparation"]["hit"]
                filler = max(1, 4095-small["input_tokens"])
                for _ in range(6):
                    request = body(q, " x"*filler, cache=False)
                    fit = count(request)
                    row["fit_attempts"].append({"filler_repeats": filler, "input_tokens": fit["input_tokens"]})
                    if fit["input_tokens"] == 4095:
                        break
                    filler += 4095-fit["input_tokens"]
                assert fit["input_tokens"] == 4095 and fit["total_tokens"] == 4096
                assert fit["fits_context"] and fit["available_tokens"] == 0 and fit["media_preparation"]["hit"]
                offset = log.stat().st_size
                accepted = client.post("/predict", json=request)
                accepted.raise_for_status()
                result = accepted.json()
                assert result["metadata"]["usage"]["prompt_tokens"] == 4095
                assert result["metadata"]["context_budget"]["total_tokens"] == 4096
                with log.open("rb") as stream:
                    stream.seek(offset)
                    positive_log = stream.read().decode("utf8", errors="replace")
                assert re.search(r"launch_slot_.*processing task", positive_log)
                row["boundary_budget"] = fit
                row["boundary_result"] = result
                row["native_logging_positive_control"] = True
                oversized = body(q, " x"*(filler+1))
                overflow = count(oversized)
                assert overflow["input_tokens"] == 4096 and overflow["total_tokens"] == 4097
                assert not overflow["fits_context"] and overflow["overflow_tokens"] == 1
                row["overflow_budget"] = overflow
                for n in (5, 10, 20):
                    offset = log.stat().st_size
                    rejected = client.post("/predict", json=oversized)
                    assert rejected.status_code == 422
                    detail = rejected.json()["detail"]
                    assert detail["code"] == "context_limit_exceeded" and detail["total_tokens"] == 4097
                    with log.open("rb") as stream:
                        stream.seek(offset)
                        delta = stream.read().decode("utf8", errors="replace")
                    assert not re.search(r"launch_slot_.*processing task", delta)
                    # Continue with a real question from the requested size after the rejected request.
                    good = body(case["questions"][n-1])
                    good_budget = count(good)
                    assert good_budget["fits_context"] and good_budget["media_preparation"]["hit"]
                    response = client.post("/predict", json=good)
                    response.raise_for_status()
                    answer = response.json()
                    assert answer["metadata"]["usage"]["prompt_tokens"] == good_budget["input_tokens"]
                    baseline_trial = next(t for t in measured["trials"] if t["purpose"] == "main" and t["case"] == case["id"] and t["count"] == n and t["mode"] == "baseline")
                    reference = next(a for a in baseline_trial["answers"] if a["question_id"] == n-1)
                    probability_delta = max(abs(v-reference["result"]["probabilities"][k]) for k, v in answer["probabilities"].items())
                    row["groups"].append({"logical_group_size": n, "rejected_status": rejected.status_code,
                                           "rejection": detail, "new_native_inferences_for_rejected_query": 0,
                                           "next_valid_question_id": n-1, "next_valid_budget": good_budget,
                                           "next_valid_result": answer, "max_probability_delta_from_isolated_control": probability_delta})
                    save_json(out / "budget-controls-progress.json", report)
                    print(json.dumps({"event": "budget_control", "case": case["id"], "size": n,
                                      "boundary_input": 4095, "overflow_input": 4096, "rejected_status": 422,
                                      "next_question_ok": True, "probability_delta": probability_delta}), flush=True)
            finally:
                client.delete("/uploads/"+receipt["upload_id"]).raise_for_status()
            assert not Path(receipt["media"]).exists() and hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        report["checks"] = {
            "three_modalities_accept_4095_input_plus_one_output": len(report["cases"]) == 3 and all(c["boundary_result"]["metadata"]["context_budget"]["total_tokens"] == 4096 for c in report["cases"]),
            "all_nine_overflows_rejected_before_inference": sum(len(c["groups"]) for c in report["cases"]) == 9 and all(g["rejected_status"] == 422 and g["new_native_inferences_for_rejected_query"] == 0 for c in report["cases"] for g in c["groups"]),
            "rejected_questions_do_not_poison_following_questions": all(g["next_valid_budget"]["fits_context"] and g["max_probability_delta_from_isolated_control"] < .01 for c in report["cases"] for g in c["groups"]),
            "e2b_service_ready": client.get(settings()["api_url"] + "/health").status_code == 200,
        }
    report["passed"] = all(report["checks"].values())
    save_json(out / "budget-controls.json", report)
    print(json.dumps({"event": "done", "passed": report["passed"], "checks": report["checks"]}), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
