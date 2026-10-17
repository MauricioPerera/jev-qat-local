"""Paired HTTP workflows: three questions sharing one upload vs three uploads."""
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
import win32api
import win32con
import win32job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_multimodal_resources import Monitor
from jev_qat.config import save_json
from jev_qat.e2b import settings, RUNTIME_STATE

OUT = ROOT / "inference" / "e2b-shared-file-benchmark"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Cpu:
    """Unlimited job for service/future ffmpeg children; add backend and client."""
    def __init__(self, service_pid, backend_pid):
        self.job = win32job.CreateJobObject(None, "")
        handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, service_pid)
        try:
            win32job.AssignProcessToJobObject(self.job, handle)
        finally:
            handle.Close()
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION, False, backend_pid)
        try:
            self.backend_in_job = bool(win32job.IsProcessInJob(handle, self.job))
        finally:
            handle.Close()
        self.extra = [psutil.Process()]
        if not self.backend_in_job:
            self.extra.append(psutil.Process(backend_pid))

    def read(self):
        info = win32job.QueryInformationJobObject(self.job, win32job.JobObjectBasicAccountingInformation)
        value = (info["TotalUserTime"] + info["TotalKernelTime"]) / 1e7
        for process in self.extra:
            cpu = process.cpu_times()
            value += cpu.user + cpu.system
        return {"seconds": value, "job_processes": info["TotalProcesses"], "active_job_processes": info["ActiveProcesses"]}

    def close(self):
        self.job.Close()


def cases():
    plan = json.loads((ROOT / "inference/gemma-e2b-vs-e4b/plan.json").read_text(encoding="utf8"))
    originals = {row["id"]: row for row in plan["cases"]}
    extras = {
        "image_alps": [
            ("¿Se observan montañas en la imagen?", ["sí", "no"]),
            ("¿La fotografía fue tomada en un entorno natural o en un interior?", ["entorno natural", "interior de un edificio"]),
        ],
        "audio_30": [
            ("¿Qué tipo de contenido se escucha?", ["una voz humana hablando", "solo música instrumental", "silencio"]),
            ("¿Se escucha habla humana en el audio?", ["sí", "no"]),
        ],
        "video_60_16": [
            ("¿El vídeo muestra una ciudad junto al mar?", ["sí", "no"]),
            ("¿Desde qué perspectiva se observa principalmente la escena?", ["vista aérea", "desde el interior de un coche", "primer plano de una persona"]),
        ],
    }
    result = []
    for name, additional in extras.items():
        row = originals[name]
        payload = row["payload"]
        path = Path(payload["media"])
        result.append({"id": name, "label": row["label"], "source": str(path), "bytes": path.stat().st_size,
                       "sha256": sha(path), "modality": payload["modality"], "video_frames": payload.get("video_frames", 4),
                       "questions": [{"id": i, "question": question, "options": options} for i, (question, options) in enumerate(
                           [(payload["question"], payload["options"]), *additional])]})
    return result


def workflow(client, case, mode, round_index, cpu, trace):
    before = cpu.read()
    start = time.perf_counter()
    trial = {"case": case["id"], "mode": mode, "round": round_index, "start": start,
             "question_order": [(i + round_index) % 3 for i in range(3)], "phases": [], "answers": [], "uploads": [], "error": None}
    owned = []

    def phase(kind, operation, question=None):
        a = time.perf_counter()
        value = operation()
        b = time.perf_counter()
        trial["phases"].append({"kind": kind, "question": question, "start": a, "end": b, "seconds": b - a})
        return value

    def upload(question=None):
        def operation():
            data = Path(case["source"]).read_bytes()
            response = client.post("/uploads", params={"filename": Path(case["source"]).name},
                                   headers={"Content-Type": "application/octet-stream"}, content=data)
            response.raise_for_status()
            receipt = response.json()
            owned.append(receipt)
            assert receipt["bytes"] == len(data) == case["bytes"] and receipt["modality"] == case["modality"]
            trial["uploads"].append({"upload_id": receipt["upload_id"], "bytes": len(data), "modality": receipt["modality"]})
            return receipt
        return phase("read_and_upload", operation, question)

    def remove(receipt, question=None):
        def operation():
            response = client.delete("/uploads/" + receipt["upload_id"])
            response.raise_for_status()
            assert not Path(receipt["media"]).exists()
            owned.remove(receipt)
        phase("remove_copy", operation, question)

    def body(question, receipt):
        return {"state": "", "question": question["question"], "options": question["options"],
                "media": receipt["media"], "modality": case["modality"], "video_frames": case["video_frames"], "cache_prompt": False}

    def count(question, request):
        def operation():
            response = client.post("/count-tokens", json=request)
            response.raise_for_status()
            budget = response.json()
            assert budget["fits_context"] is True and budget["inference_executed"] is False
            return budget
        return phase("count_tokens", operation, question["id"])

    def infer(question, request, budget):
        def operation():
            response = client.post("/predict", json=request)
            response.raise_for_status()
            result = response.json()
            meta, usage = result["metadata"], result["metadata"]["usage"]
            assert usage["prompt_tokens_details"]["cached_tokens"] == meta["timings"]["cache_n"] == 0
            assert usage["completion_tokens"] == 1 and usage["prompt_tokens"] == budget["input_tokens"]
            assert meta["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf" and meta["cache_prompt"] is False
            assert set(result["probabilities"]) == set(question["options"])
            assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
            return result
        result = phase("predict", operation, question["id"])
        trial["answers"].append({"question_id": question["id"], "budget": budget, "result": result,
                                  "response_at": time.perf_counter() - start})
        print(json.dumps({"event": "answer", "case": case["id"], "mode": mode, "round": round_index,
                          "question": question["id"], "seconds": round(trial["phases"][-1]["seconds"], 3)}, ensure_ascii=True), flush=True)

    try:
        order = [case["questions"][i] for i in trial["question_order"]]
        if mode == "shared":
            receipt = upload()
            requests = [(q, body(q, receipt)) for q in order]
            budgets = [count(q, request) for q, request in requests]
            for (question, request), budget in zip(requests, budgets):
                infer(question, request, budget)
            remove(receipt)
        else:
            for question in order:
                receipt = upload(question["id"])
                request = body(question, receipt)
                budget = count(question, request)
                infer(question, request, budget)
                remove(receipt, question["id"])
    except Exception as error:
        trial["error"] = repr(error)
    finally:
        for receipt in list(owned):
            try:
                remove(receipt)
            except Exception as error:
                trial["cleanup_error"] = repr(error)
    end = time.perf_counter()
    after = cpu.read()
    trial.update({"end": end, "wall_seconds": end - start, "cpu_before": before, "cpu_after": after,
                  "cpu_seconds": after["seconds"] - before["seconds"], "cpu_mean_percent": 100 * (after["seconds"] - before["seconds"]) / (end-start) / psutil.cpu_count(),
                  "first_response_seconds": trial["answers"][0]["response_at"] if trial["answers"] else None,
                  "last_response_seconds": trial["answers"][-1]["response_at"] if trial["answers"] else None,
                  "upload_bytes": sum(x["bytes"] for x in trial["uploads"])})
    trace.append(trial)
    return trial


def main():
    OUT.mkdir(exist_ok=False)
    config = settings()
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf8"))
    service, backend = state["pid"], state["backend_pid"]
    assert config["model"] in psutil.Process(backend).cmdline()
    samples = cases()
    source_paths = [ROOT / p for p in ("jev_qat/e2b.py", "jev_qat/adapter.py", "jev_qat/e2b_budget.py", "jev_qat/e2b_web.py", "jev_qat/web/e2b.js", "jev_qat_e2b_config.json", "jev_qat_config.json", "classification/current.json")]
    frozen = {str(path): sha(path) for path in source_paths}
    for name in ("benchmark_e2b_shared_file.py", "benchmark_multimodal_resources.py"):
        shutil.copy2(ROOT / "scripts" / name, OUT / name)
    cpu = Cpu(service, backend)
    monitor = Monitor(service, backend, .1)
    monitor.roots.append(psutil.Process())
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": samples, "rounds": 4,
            "first_round_excluded_from_steady_summary": True, "questions_per_workflow": 3,
            "modes": {"shared": "one upload; count all questions; infer all sequentially; remove one copy",
                      "separate": "for each question: read/upload, count, infer, remove its copy"},
            "cache_prompt": False, "os_file_cache_cleared": False, "parallel_inferences": 1,
            "measured_scope": "E2B HTTP bridge + native backend + ffmpeg/ffprobe descendants + benchmark client/monitor; excludes browser rendering and E4B",
            "service_pid": service, "backend_pid": backend, "client_pid": psutil.Process().pid,
            "backend_in_service_cpu_job": cpu.backend_in_job, "logical_processors": psutil.cpu_count(),
            "config": config, "frozen_sources": frozen}
    save_json(OUT / "plan.json", plan)
    report = {**plan, "trials": []}
    monitor.start()
    try:
        with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
            assert client.get("/health").status_code == 200
            time.sleep(1.5)
            report["idle_interval"] = [monitor.samples[0]["start"], time.perf_counter()]
            for round_index in range(4):
                order = samples if round_index % 2 == 0 else list(reversed(samples))
                for case in order:
                    case_position = samples.index(case)
                    modes = ("shared", "separate") if (round_index + case_position) % 2 == 0 else ("separate", "shared")
                    for mode in modes:
                        time.sleep(.3)
                        trial = workflow(client, case, mode, round_index, cpu, report["trials"])
                        save_json(OUT / "progress.json", report)
                        print(json.dumps({"event": "workflow", "case": case["id"], "mode": mode, "round": round_index,
                                          "seconds": round(trial["wall_seconds"], 3), "completed": len(report["trials"]), "expected": 24, "error": trial["error"]}), flush=True)
                        if trial["error"]:
                            raise RuntimeError(trial["error"])
            time.sleep(.8)
    finally:
        monitor.close()
        with (OUT / "samples.jsonl").open("w", encoding="utf8") as stream:
            for sample in monitor.samples:
                stream.write(json.dumps(sample) + "\n")
        cpu.close()
        save_json(OUT / "progress.json", report)
    for trial in report["trials"]:
        trial["resources"] = monitor.summary(trial["start"], trial["end"])
        for phase in trial["phases"]:
            phase["resources"] = monitor.summary(phase["start"], phase["end"])
    report["idle_resources"] = monitor.summary(*report["idle_interval"])
    intervals = [s["end"] - s["start"] for s in monitor.samples]
    report["sample_intervals"] = {"requested_seconds": .1, "median_seconds": statistics.median(intervals), "max_seconds": max(intervals), "count": len(intervals)}
    report["checks"] = {
        "all_24_workflows_72_predictions_succeeded": len(report["trials"]) == 24 and all(not t["error"] and len(t["answers"]) == 3 for t in report["trials"]),
        "all_resources_cover_entire_workflow": all(abs(t["resources"]["covered_seconds"] - t["wall_seconds"]) < 1e-6 for t in report["trials"]),
        "sources_and_model_configuration_unchanged": all(sha(Path(path)) == digest for path, digest in frozen.items()),
        "real_media_unchanged": all(sha(Path(case["source"])) == case["sha256"] for case in samples),
        "same_service_and_backend_pids": all(psutil.pid_exists(pid) for pid in (service, backend)),
        "monitor_valid": monitor.error is None,
    }
    save_json(OUT / "results.json", report)
    print(json.dumps({"event": "done", "checks": report["checks"], "output": str(OUT)}), flush=True)
    assert all(report["checks"].values())


if __name__ == "__main__":
    main()
