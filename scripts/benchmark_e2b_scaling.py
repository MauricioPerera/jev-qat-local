"""Measure actual 5/10/20-question workflows, plus a reverse-order independence audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_e2b_shared_file import Cpu
from benchmark_multimodal_resources import Monitor
from e2b_scaling_questions import cases
from jev_qat.config import save_json
from jev_qat.e2b import RUNTIME_STATE, settings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workflow(client, case, count, mode, purpose, order, cpu, out, report):
    before = cpu.read()
    start = time.perf_counter()
    t = {"case": case["id"], "count": count, "mode": mode, "purpose": purpose,
         "order": order, "start": start, "answers": [], "budgets": [], "phases": [],
         "error": None, "upload_id": None, "upload_bytes": case["bytes"]}
    report["trials"].append(t)
    receipt = None

    def phase(kind, operation, q=None):
        a = time.perf_counter()
        value = operation()
        b = time.perf_counter()
        t["phases"].append({"kind": kind, "question": q, "start": a, "end": b, "seconds": b-a})
        return value

    def post(route, body):
        response = client.post(route, json=body)
        response.raise_for_status()
        return response.json()

    def upload():
        data = Path(case["source"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == case["sha256"]
        response = client.post("/uploads", params={"filename": Path(case["source"]).name},
                               headers={"Content-Type": "application/octet-stream"}, content=data)
        response.raise_for_status()
        r = response.json()
        assert r["bytes"] == case["bytes"]
        t["upload_id"] = r["upload_id"]
        return r

    def remove():
        client.delete("/uploads/"+receipt["upload_id"]).raise_for_status()
        assert not Path(receipt["media"]).exists()

    try:
        receipt = phase("read_and_upload", upload)
        requests = []
        for index, qid in enumerate(order):
            q = case["questions"][qid]
            body = {"state": "", "question": q["question"], "options": q["options"],
                    "media": receipt["media"], "modality": case["modality"], "video_frames": case["video_frames"],
                    "reuse_media": mode == "optimized", "cache_prompt": mode == "optimized" and index > 0}
            budget = phase("count_tokens", lambda: post("/count-tokens", body), qid)
            assert budget["fits_context"] and not budget["inference_executed"]
            assert budget["total_tokens"] == budget["input_tokens"] + 1 <= 4096
            assert budget["media_preparation"]["hit"] is (mode == "optimized" and index > 0)
            t["budgets"].append({"question_id": qid, "budget": budget})
            requests.append((qid, body, budget))
        for index, (qid, body, budget) in enumerate(requests):
            result = phase("predict", lambda: post("/predict", body), qid)
            m = result["metadata"]
            usage = m["usage"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            assert usage["prompt_tokens"] == budget["input_tokens"] and usage["completion_tokens"] == 1
            assert m["context_budget"]["total_tokens"] == budget["total_tokens"] <= 4096
            assert cached == m["timings"]["cache_n"]
            assert (cached > 0) is (mode == "optimized" and index > 0)
            assert m["media_preparation"]["hit"] is (mode == "optimized")
            assert set(result["probabilities"]) == set(body["options"])
            assert abs(sum(result["probabilities"].values())-1) < 1e-6
            t["answers"].append({"question_id": qid, "budget": budget, "result": result,
                                  "response_at": time.perf_counter()-start})
            if (index+1) % 5 == 0 or index+1 == count:
                save_json(out / "progress.json", report)
                print(json.dumps({"event": "answers", "case": case["id"], "mode": mode, "purpose": purpose,
                                  "count": count, "completed": index+1, "seconds_so_far": round(time.perf_counter()-start, 3)}), flush=True)
    except Exception as error:
        t["error"] = repr(error)
    finally:
        if receipt:
            phase("remove_copy", remove)
    end = time.perf_counter()
    after = cpu.read()
    t.update({"end": end, "wall_seconds": end-start, "cpu_before": before, "cpu_after": after,
              "cpu_seconds": after["seconds"]-before["seconds"],
              "cpu_mean_percent": 100*(after["seconds"]-before["seconds"])/(end-start)/psutil.cpu_count(),
              "first_response_seconds": t["answers"][0]["response_at"] if t["answers"] else None,
              "last_response_seconds": t["answers"][-1]["response_at"] if t["answers"] else None})
    save_json(out / "progress.json", report)
    print(json.dumps({"event": "workflow", "case": case["id"], "mode": mode, "purpose": purpose, "count": count,
                      "seconds": round(t["wall_seconds"], 3), "error": t["error"],
                      "finished_workflows": len(report["trials"]), "expected_workflows": 24}), flush=True)
    if t["error"]:
        raise RuntimeError(t["error"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "inference/e2b-scaling-5-10-20")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    config = settings()
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf8"))
    service, backend = state["pid"], state["backend_pid"]
    data = cases()
    for case in data:
        path = Path(case["source"])
        if not path.is_file():
            raise FileNotFoundError("Prepare public media first: python scripts/prepare_real_multimodal_benchmark.py")
        # FFmpeg builds can produce different encoded bytes for the same excerpt.
        # Record both the historical reference and the exact files of this new run.
        case["reference_bytes"], case["reference_sha256"] = case["bytes"], case["sha256"]
        case["bytes"], case["sha256"] = path.stat().st_size, sha(path)
    frozen = {p: sha(ROOT/p) for p in ("jev_qat/e2b.py", "jev_qat/e2b_media.py", "jev_qat/e2b_budget.py", "jev_qat/e2b_web.py", "jev_qat/web/e2b.js", "jev_qat/adapter.py", "jev_qat_e2b_config.json", "jev_qat_config.json", "benchmarks/scaling-cases.json")}
    if (ROOT / "classification/current.json").is_file():
        frozen["classification/current.json"] = sha(ROOT / "classification/current.json")
    for p in (*frozen, "scripts/benchmark_e2b_scaling.py", "scripts/e2b_scaling_questions.py", "scripts/benchmark_e2b_shared_file.py", "scripts/benchmark_multimodal_resources.py"):
        target = out / "source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/p, target)
    cpu = Cpu(service, backend)
    monitor = Monitor(service, backend, .1)
    monitor.roots.append(psutil.Process())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": data, "sizes": [5, 10, 20],
              "repetitions_main": 1, "warmup_questions_per_case": 3, "reverse_audit_questions_per_case": 20,
              "expected_workflows": 24, "expected_predictions": 279, "expected_main_predictions": 210,
              "first_inference_cold_in_every_workflow": True, "model_already_loaded": True, "os_file_cache_cleared": False,
              "modes": {"baseline": "reuse_media=false; cache_prompt=false for every question",
                        "optimized": "reuse_media=true; first cache_prompt=false; subsequent cache_prompt=true"},
              "protocol": "One upload; count all N questions; infer N sequentially with fresh independent prompts; remove owned upload. Same question order in each paired mode. Separate warmup and reverse-order audit.",
              "scope": "E2B bridge + native backend + codec descendants + test client/monitor; excludes browser rendering and E4B",
              "logical_processors": psutil.cpu_count(), "physical_ram_bytes": psutil.virtual_memory().total,
              "service_pid": service, "backend_pid": backend, "client_pid": psutil.Process().pid,
              "backend_in_service_cpu_job": cpu.backend_in_job, "config": config, "frozen_sources": frozen, "trials": []}
    save_json(out / "plan.json", report)
    monitor.start()
    try:
        with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
            client.get("/health").raise_for_status()
            time.sleep(1.5)
            report["idle_interval"] = [monitor.samples[0]["start"], time.perf_counter()]
            for case in data:
                workflow(client, case, 3, "optimized", "warmup", [0, 1, 2], cpu, out, report)
            schedule = {"image_alps": [5, 10, 20], "audio_30": [20, 10, 5], "video_60_16": [10, 5, 20]}
            for step in range(3):
                for case in (data if step % 2 == 0 else list(reversed(data))):
                    n = schedule[case["id"]][step]
                    offset = (data.index(case)+step) % n
                    order = list(range(n))[offset:] + list(range(n))[:offset]
                    modes = ("baseline", "optimized") if (step+data.index(case)) % 2 == 0 else ("optimized", "baseline")
                    for mode in modes:
                        time.sleep(.3)
                        workflow(client, case, n, mode, "main", order, cpu, out, report)
            for case in reversed(data):
                time.sleep(.3)
                workflow(client, case, 20, "optimized", "reverse_audit", list(reversed(range(20))), cpu, out, report)
            time.sleep(1)
    finally:
        monitor.close()
        with (out / "samples.jsonl").open("w", encoding="utf8") as stream:
            for sample in monitor.samples:
                stream.write(json.dumps(sample)+"\n")
        cpu.close()
        save_json(out / "progress.json", report)
    for t in report["trials"]:
        t["resources"] = monitor.summary(t["start"], t["end"])
        for phase in t["phases"]:
            phase["resources"] = monitor.summary(phase["start"], phase["end"])
    intervals = [s["end"]-s["start"] for s in monitor.samples]
    report["sample_intervals"] = {"requested_seconds": .1, "median_seconds": statistics.median(intervals), "max_seconds": max(intervals), "count": len(intervals)}
    report["idle_resources"] = monitor.summary(*report["idle_interval"])
    report["checks"] = {
        "24_workflows_279_predictions": len(report["trials"]) == 24 and sum(len(t["answers"]) for t in report["trials"]) == 279,
        "18_main_workflows_210_predictions": sum(t["purpose"] == "main" for t in report["trials"]) == 18 and sum(len(t["answers"]) for t in report["trials"] if t["purpose"] == "main") == 210,
        "all_resource_intervals_covered": all(abs(t["resources"]["covered_seconds"]-t["wall_seconds"]) < 1e-6 for t in report["trials"]),
        "all_requests_succeeded": all(t["error"] is None and len(t["answers"]) == t["count"] for t in report["trials"]),
        "frozen_sources_unchanged": all(sha(ROOT/p) == digest for p, digest in frozen.items()),
        "real_files_unchanged": all(sha(c["source"]) == c["sha256"] for c in data), "monitor_valid": monitor.error is None,
    }
    save_json(out / "results.json", report)
    print(json.dumps({"event": "done", "checks": report["checks"], "output": str(out)}), flush=True)
    assert all(report["checks"].values())


if __name__ == "__main__":
    main()
