import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
import psutil

from .config import ROOT, RUNTIME_STATE, STATE, settings


def health():
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            response = client.get(settings()["api_url"] + "/health")
            result = response.json()
            if response.status_code == 200 and result.get("service") == "jev-qat-bridge":
                return result
    except (httpx.HTTPError, ValueError):
        pass
    return None


def start():
    if health():
        return {"status": "already_running", "api_url": settings()["api_url"]}
    from .service import check_port
    config = settings()
    check_port(config["api_port"])
    check_port(config["backend_port"])
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "service.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "jev_qat", "serve"], cwd=ROOT,
                                   stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.monotonic()
    while time.monotonic() - started < 320:
        if process.poll() is not None:
            raise RuntimeError("Service startup failed; see integration/jev-qat/service.log")
        if health():
            return {"status": "ready", "pid": process.pid, "api_url": config["api_url"],
                    "docs": config["api_url"] + "/docs"}
        time.sleep(0.5)
    # The service has its own 300s startup deadline and cleans up its backend.
    raise TimeoutError("Service startup timed out; inspect the local service logs")


def stop():
    if not RUNTIME_STATE.exists():
        return {"status": "not_running"}
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
    if state.get("workspace") != str(ROOT):
        raise RuntimeError("Service state belongs to a different workspace")
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.post(state["api_url"] + "/admin/stop",
                                   headers={"X-Management-Token": state["management_token"]})
            response.raise_for_status()
    except httpx.HTTPError:
        pass
    for _ in range(30):
        alive = []
        for pid_field, created_field in (("pid", "created"), ("backend_pid", "backend_created")):
            try:
                process = psutil.Process(state[pid_field])
                if abs(process.create_time() - state[created_field]) < 0.01:
                    alive.append(process)
            except (psutil.NoSuchProcess, KeyError):
                pass
        if not alive:
            return {"status": "stopped"}
        time.sleep(0.5)
    # Fallback only targets recorded PIDs with matching creation time and command.
    config = settings()
    for process in alive:
        try:
            command = process.cmdline()
            is_service = "jev_qat" in command and "serve" in command
            is_backend = config["model"] in command and Path(command[0]).resolve() == Path(config["runtime"]).resolve()
            if not (is_service or is_backend):
                raise RuntimeError("Recorded process identity changed; refusing to terminate it")
            process.terminate()
            process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
    state["status"] = "stopped"
    from .config import save_json
    save_json(RUNTIME_STATE, state)
    return {"status": "stopped"}


def main():
    parser = argparse.ArgumentParser(description="Local E4B QAT adapter with Jev-Omni-shaped decisions")
    parser.add_argument("command", choices=["start", "stop", "status", "serve", "predict"])
    parser.add_argument("--input", type=Path, help="UTF-8 JSON request for predict")
    args = parser.parse_args()
    if args.command == "serve":
        from .service import run_service
        run_service()
        return
    if args.command == "start":
        result = start()
    elif args.command == "stop":
        result = stop()
    elif args.command == "status":
        result = health() or {"status": "not_running"}
    else:
        if args.input is None:
            parser.error("predict requires --input <request.json>")
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        with httpx.Client(timeout=settings()["request_timeout_seconds"], trust_env=False) as client:
            response = client.post(settings()["api_url"] + "/predict", json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Prediction HTTP {response.status_code}: {response.text}")
            result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
