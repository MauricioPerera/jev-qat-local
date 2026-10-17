"""Paired old/optimized HTTP flows with one shared upload and three questions."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_e2b_shared_file import Cpu
from benchmark_multimodal_resources import Monitor
from jev_qat.config import save_json
from jev_qat.e2b import RUNTIME_STATE, settings

OUT = ROOT / "inference/e2b-prepare-reuse/benchmark"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workflow(client, case, mode, round_index, cpu):
    before = cpu.read()
    start = time.perf_counter()
    t = {"case": case["id"], "mode": mode, "round": round_index, "start": start, "answers": [],
         "phases": [], "budgets": [], "error": None, "upload_id": None,
         "question_order": [(i+round_index) % 3 for i in range(3)]}

    def phase(kind, operation, q=None):
        a = time.perf_counter()
        value = operation()
        b = time.perf_counter()
        t["phases"].append({"kind": kind, "question": q, "start": a, "end": b, "seconds": b-a})
        return value

    def upload():
        data = Path(case["source"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == case["sha256"]
        response = client.post("/uploads", params={"filename": Path(case["source"]).name},
                               headers={"Content-Type": "application/octet-stream"}, content=data)
        response.raise_for_status()
        receipt = response.json()
        assert receipt["bytes"] == case["bytes"]
        t["upload_id"] = receipt["upload_id"]
        return receipt

    def remove(receipt):
        response = client.delete("/uploads/" + receipt["upload_id"])
        response.raise_for_status()
        assert not Path(receipt["media"]).exists()

    def post(route, body):
        response = client.post(route, json=body)
        response.raise_for_status()
        return response.json()

    receipt = None
    try:
        receipt = phase("read_and_upload", upload)
        requests = []
        for index, qid in enumerate(t["question_order"]):
            q = case["questions"][qid]
            body = {"state": "", "question": q["question"], "options": q["options"], "media": receipt["media"],
                    "modality": case["modality"], "video_frames": case["video_frames"],
                    "reuse_media": mode == "optimized", "cache_prompt": mode == "optimized" and index > 0}
            budget = phase("count_tokens", lambda: post("/count-tokens", body), qid)
            assert budget["fits_context"] is True and budget["inference_executed"] is False
            prep = budget["media_preparation"]
            assert prep["enabled"] is (mode == "optimized")
            assert prep["hit"] is (mode == "optimized" and index > 0)
            t["budgets"].append(budget)
            requests.append((qid, body, budget))
        for index, (qid, body, budget) in enumerate(requests):
            result = phase("predict", lambda: post("/predict", body), qid)
            meta = result["metadata"]
            usage, timings = meta["usage"], meta["timings"]
            assert usage["prompt_tokens"] == budget["input_tokens"] and usage["completion_tokens"] == 1
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            assert cached == timings["cache_n"]
            assert (cached > 0) is (mode == "optimized" and index > 0)
            assert meta["media_preparation"]["hit"] is (mode == "optimized")
            assert meta["context_budget"]["total_tokens"] <= 4096
            assert set(result["probabilities"]) == set(body["options"])
            assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
            t["answers"].append({"question_id": qid, "budget": budget, "result": result,
                                  "response_at": time.perf_counter()-start})
            print(json.dumps({"event": "answer", "case": case["id"], "mode": mode, "round": round_index,
                              "q": qid, "seconds": round(t["phases"][-1]["seconds"], 3), "cached_tokens": cached}), flush=True)
    except Exception as error:
        t["error"] = repr(error)
    finally:
        if receipt:
            phase("remove_copy", lambda: remove(receipt))
    end = time.perf_counter()
    after = cpu.read()
    t.update({"end": end, "wall_seconds": end-start, "cpu_before": before, "cpu_after": after,
              "cpu_seconds": after["seconds"] - before["seconds"],
              "cpu_mean_percent": 100 * (after["seconds"]-before["seconds"]) / (end-start) / psutil.cpu_count(),
              "first_response_seconds": t["answers"][0]["response_at"] if t["answers"] else None,
              "last_response_seconds": t["answers"][-1]["response_at"] if t["answers"] else None,
              "upload_bytes": case["bytes"]})
    return t


def main():
    OUT.mkdir(exist_ok=False)
    config = settings()
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf8"))
    service, backend = state["pid"], state["backend_pid"]
    original = json.loads((ROOT / "inference/e2b-shared-file-benchmark/plan.json").read_text(encoding="utf8"))
    cases = original["cases"]
    frozen = {p: sha(ROOT/p) for p in ("jev_qat/e2b.py", "jev_qat/e2b_media.py", "jev_qat/e2b_budget.py", "jev_qat/e2b_web.py", "jev_qat/web/e2b.js", "jev_qat/adapter.py", "jev_qat_e2b_config.json", "jev_qat_config.json", "classification/current.json")}
    for rel in frozen:
        target = OUT / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for name in ("benchmark_e2b_media_reuse.py", "benchmark_e2b_shared_file.py", "benchmark_multimodal_resources.py"):
        shutil.copy2(ROOT / "scripts" / name, OUT / name)
    cpu = Cpu(service, backend)
    monitor = Monitor(service, backend, .1)
    monitor.roots.append(psutil.Process())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases, "rounds": 4,
              "first_round_excluded_from_steady_summary": True, "questions_per_workflow": 3,
              "modes": {"baseline": "current shared-upload workflow; prepare on every count/predict; all cache_prompt=false",
                        "optimized": "same workflow with bounded preparation cache; first inference cold, next two reuse native common prefix"},
              "scope": "E2B service + native backend + codec descendants + benchmark client/monitor; no browser rendering, no E4B",
              "logical_processors": psutil.cpu_count(), "service_pid": service, "backend_pid": backend,
              "client_pid": psutil.Process().pid, "frozen_sources": frozen, "config": config,
              "backend_in_service_cpu_job": cpu.backend_in_job,
              "native_cache_reset_before_each_workflow": True, "os_file_cache_cleared": False, "trials": []}
    save_json(OUT / "plan.json", report)
    monitor.start()
    try:
        with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
            client.get("/health").raise_for_status()
            time.sleep(1.5)
            report["idle_interval"] = [monitor.samples[0]["start"], time.perf_counter()]
            for r in range(4):
                for case in (cases if r % 2 == 0 else list(reversed(cases))):
                    modes = ("baseline", "optimized") if (r+cases.index(case)) % 2 == 0 else ("optimized", "baseline")
                    for mode in modes:
                        time.sleep(.3)
                        t = workflow(client, case, mode, r, cpu)
                        report["trials"].append(t)
                        save_json(OUT / "progress.json", report)
                        print(json.dumps({"event": "workflow", "completed": len(report["trials"]), "expected": 24,
                                          "case": t["case"], "mode": mode, "round": r,
                                          "seconds": round(t["wall_seconds"], 3), "error": t["error"]}), flush=True)
                        if t["error"]:
                            raise RuntimeError(t["error"])
            time.sleep(1)
    finally:
        monitor.close()
        with (OUT / "samples.jsonl").open("w", encoding="utf8") as stream:
            for sample in monitor.samples:
                stream.write(json.dumps(sample) + "\n")
        cpu.close()
        save_json(OUT / "progress.json", report)
    for t in report["trials"]:
        t["resources"] = monitor.summary(t["start"], t["end"])
        for p in t["phases"]:
            p["resources"] = monitor.summary(p["start"], p["end"])
    report["idle_resources"] = monitor.summary(*report["idle_interval"])
    intervals = [s["end"]-s["start"] for s in monitor.samples]
    report["sample_intervals"] = {"median_seconds": statistics.median(intervals), "max_seconds": max(intervals), "count": len(intervals)}
    report["checks"] = {
        "all_24_workflows_72_predictions_succeeded": len(report["trials"]) == 24 and all(t["error"] is None and len(t["answers"]) == 3 for t in report["trials"]),
        "complete_monitor_coverage": all(abs(t["resources"]["covered_seconds"] - t["wall_seconds"]) < 1e-6 for t in report["trials"]),
        "frozen_sources_unchanged": all(sha(ROOT/p) == digest for p, digest in frozen.items()),
        "same_real_media": all(sha(c["source"]) == c["sha256"] for c in cases), "monitor_valid": monitor.error is None,
    }
    save_json(OUT / "results.json", report)
    print(json.dumps({"event": "done", "checks": report["checks"]}), flush=True)
    assert all(report["checks"].values())


if __name__ == "__main__":
    main()
