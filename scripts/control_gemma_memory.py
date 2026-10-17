"""Fresh-process memory control; original E4B service remains available."""
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multimodal_resources import Monitor
from jev_qat.adapter import JevQatAdapter
from jev_qat.config import save_json, settings as e4b_settings
from jev_qat import e2b
from jev_qat.__main__ import health as e4b_health

OUT = ROOT / "inference" / "gemma-e2b-vs-e4b" / "memory-control"


def worker(model, url):
    adapter = e2b.E2BQatAdapter(base_url=url) if model == "e2b" else JevQatAdapter(base_url=url)
    native = adapter._post
    if model == "e4b":
        def post(route, payload):
            return native(route, {**payload, "cache_prompt": False} if route == "/v1/chat/completions" else payload)
        adapter._post = post
    with adapter:
        print('{"ready":true}', flush=True)
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if model == "e2b":
                    payload["cache_prompt"] = False
                result = adapter.predict(**payload)
                print(json.dumps({"result": result}, ensure_ascii=True), flush=True)
            except Exception as error:
                print(json.dumps({"error": repr(error)}), flush=True)


def measure(model, url, backend_pid, cases):
    with (OUT / f"{model}-worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", model, url], cwd=ROOT,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        if json.loads(process.stdout.readline()) != {"ready": True}:
            raise RuntimeError("Memory control worker failed")

        def predict(payload):
            process.stdin.write(json.dumps(payload, ensure_ascii=True)+"\n")
            process.stdin.flush()
            result = json.loads(process.stdout.readline())
            if "error" in result:
                raise RuntimeError(result["error"])
            return result["result"]

        predict(cases[0]["payload"])
        monitor = Monitor(process.pid, backend_pid, .1)
        monitor.start()
        trials = []
        try:
            time.sleep(1)
            for case in cases:
                start = time.perf_counter()
                result = predict(case["payload"])
                end = time.perf_counter()
                assert result["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                trials.append({"case": case["id"], "start": start, "end": end, "result": result})
                print(json.dumps({"memory_control": model, "case": case["id"], "seconds": round(end-start, 3)}), flush=True)
                time.sleep(.3)
            time.sleep(.8)
        finally:
            monitor.close()
            process.stdin.close()
            process.wait(timeout=15)
        for trial in trials:
            trial["resources"] = monitor.summary(trial["start"], trial["end"])
        with (OUT / f"samples-{model}.jsonl").open("w", encoding="utf-8") as f:
            for sample in monitor.samples:
                f.write(json.dumps(sample)+"\n")
        native_memory = psutil.Process(backend_pid).memory_info()
        return {"backend_pid": backend_pid, "trials": trials,
                "rss_peak_gib": max(t["resources"]["rss_peak_gib"] for t in trials),
                "commit_peak_gib": max(t["resources"]["committed_peak_gib"] for t in trials),
                "gpu_shared_peak_gib": max(t["resources"]["gpu_shared_peak_gib"] for t in trials),
                "backend_lifetime_peak_wset_gib": native_memory.peak_wset/1024**3,
                "interval_coverage_complete": all(abs(t["resources"]["covered_seconds"]-(t["end"]-t["start"]))<1e-6 for t in trials)}


def main():
    OUT.mkdir(exist_ok=False)
    source = json.loads((OUT.parent / "results.json").read_text(encoding="utf-8"))
    ids = ("text_v2", "image_alps", "audio_30", "video_60_16")
    cases = [next(c for c in source["cases"] if c["id"] == name) for name in ids]
    save_json(OUT / "plan.json", {"cases": cases, "fresh_native_processes": True,
                                  "original_e4b_stays_loaded": True, "one_test_backend_loaded_at_a_time": True})
    assert e4b_health()["status"] == "ready"
    e2b.stop()
    report = {"background": "original E4B remains loaded and idle for both conditions", "models": {}}
    clone = None
    try:
        config = e4b_settings()
        port = 18168
        from jev_qat.service import check_port
        check_port(port)
        args = [config["runtime"], "--model", config["model"], "--mmproj", config["projector"],
                "--device", config["device"], "--gpu-layers", "all", "--ctx-size", "4096", "--parallel", "1",
                "--threads", "8", "--threads-batch", "8", "--host", "127.0.0.1", "--port", str(port),
                "--reasoning", "off", "--alias", "gemma-e4b-qat-q4-local"]
        with (OUT / "e4b-backend.log").open("w", encoding="utf-8") as log:
            clone = subprocess.Popen(args, cwd=Path(config["runtime"]).parent, stdout=log, stderr=log,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        url = f"http://127.0.0.1:{port}"
        started = time.perf_counter()
        with httpx.Client(base_url=url, timeout=2, trust_env=False) as client:
            while time.perf_counter()-started < 180:
                if clone.poll() is not None:
                    raise RuntimeError("Fresh E4B control backend exited")
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("Fresh E4B control startup failed")
        report["models"]["e4b"] = measure("e4b", url, clone.pid, cases)
    finally:
        if clone is not None and clone.poll() is None:
            clone.terminate()
            try:
                clone.wait(timeout=10)
            except subprocess.TimeoutExpired:
                clone.kill()
                clone.wait(timeout=5)
        e2b.start()
    state = json.loads(e2b.RUNTIME_STATE.read_text(encoding="utf-8"))
    report["models"]["e2b"] = measure("e2b", e2b.settings()["backend_url"], state["backend_pid"], cases)
    report["passed"] = all(m["interval_coverage_complete"] and len(m["trials"])==4 for m in report["models"].values())
    report["e4b_default_healthy"] = e4b_health()["status"] == "ready"
    report["e2b_healthy"] = e2b.health()["status"] == "ready"
    save_json(OUT / "results.json", report)
    print(json.dumps({"passed": report["passed"], "models": {m:{k:v for k,v in d.items() if k!='trials'} for m,d in report["models"].items()}}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv)==4 and sys.argv[1]=="--worker":
        worker(sys.argv[2],sys.argv[3])
    else:
        main()
