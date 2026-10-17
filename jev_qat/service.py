from contextlib import asynccontextmanager
import hmac
import json
import re
import secrets
import socket
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
import httpx
import psutil
from pydantic import BaseModel, ConfigDict, Field, StrictStr
import uvicorn

from .adapter import BackendError, JevQatAdapter
from .config import ROOT, RUNTIME_STATE, STATE, save_json, settings


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: StrictStr
    question: StrictStr = Field(min_length=1)
    options: list[StrictStr] = Field(min_length=2, max_length=256)
    media: StrictStr | None = None
    modality: Literal["text", "image", "audio", "video"] = "text"
    video_frames: int = Field(default=16, ge=1, le=16, strict=True)
    calibration_profile: StrictStr | None = None
    task_profile: StrictStr | None = None


def check_port(port):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Local port {port} is already in use")


def run_service():
    config = settings()
    STATE.mkdir(parents=True, exist_ok=True)
    check_port(config["api_port"])
    check_port(config["backend_port"])
    for field in ("runtime", "model", "projector"):
        if not Path(config[field]).is_file():
            raise RuntimeError(f"Missing configured file: {config[field]}")
    for field in ("model", "projector"):
        with Path(config[field]).open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise RuntimeError(f"Invalid GGUF file: {config[field]}")
    arguments = [
        config["runtime"], "--model", config["model"], "--mmproj", config["projector"],
        "--device", config["device"], "--gpu-layers", "all", "--ctx-size", str(config["context_tokens"]),
        "--parallel", "1", "--threads", str(config["threads"]), "--threads-batch", str(config["threads"]),
        "--host", "127.0.0.1", "--port", str(config["backend_port"]), "--reasoning", "off",
        "--alias", "gemma-e4b-qat-q4-local",
    ]
    token = secrets.token_urlsafe(32)
    state = {
        "service": "jev-qat-bridge", "status": "starting", "api_url": config["api_url"],
        "backend_url": config["backend_url"], "management_token": token,
        "pid": psutil.Process().pid, "created": psutil.Process().create_time(),
        "workspace": str(ROOT),
    }

    @asynccontextmanager
    async def lifespan(app):
        backend = None
        adapter = None
        with (STATE / "backend.log").open("a", encoding="utf-8") as log:
            try:
                backend = subprocess.Popen(arguments, cwd=Path(config["runtime"]).parent,
                                           stdout=log, stderr=log,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state.update({"backend_pid": backend.pid, "backend_created": psutil.Process(backend.pid).create_time()})
                save_json(RUNTIME_STATE, state)
                started = time.monotonic()
                with httpx.Client(base_url=config["backend_url"], timeout=2, trust_env=False) as client:
                    while time.monotonic() - started < 300:
                        if backend.poll() is not None:
                            raise RuntimeError("llama.cpp exited during startup; see integration/jev-qat/backend.log")
                        try:
                            if client.get("/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.5)
                    else:
                        raise TimeoutError("llama.cpp startup timed out")
                adapter = JevQatAdapter(config=config)
                adapter._verify_labels(256)
                app.state.adapter = adapter
                app.state.backend = backend
                state["status"] = "ready"
                save_json(RUNTIME_STATE, state)
                yield
            except Exception as error:
                state["status"] = "failed"
                state["error"] = str(error)
                raise
            finally:
                if adapter is not None:
                    adapter.close()
                if backend is not None and backend.poll() is None:
                    backend.terminate()
                    try:
                        backend.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        backend.kill()
                        backend.wait(timeout=5)
                if state["status"] != "failed":
                    state["status"] = "stopped"
                save_json(RUNTIME_STATE, state)

    app = FastAPI(
        title="Gemma E4B QAT local decision adapter",
        description="Jev-Omni-shaped predict API using E4B token scores. Default probabilities are uncalibrated; a validated task-specific calibration_profile can be requested. Local evaluation report: /evaluation.",
        version="0.1.0", lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        if app.state.backend.poll() is not None:
            raise HTTPException(503, "Local inference backend stopped")
        return {"service": "jev-qat-bridge", "status": "ready", "backend": "gemma-e4b-qat-q4-local",
                "device": config["device"], "modalities": ["text", "image", "audio", "video"],
                "max_options": 256, "calibrated": False, "original_jev_omni_head_used": False}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        try:
            return app.state.adapter.predict(**request.model_dump())
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except BackendError as error:
            raise HTTPException(502, str(error)) from error

    def evaluation_artifact(filename):
        directory = ROOT / "evaluation" / "massive-es-v1"
        latest = directory / "latest-run.json"
        current = ROOT / "evaluation" / "current.json"
        if current.is_file():
            selection = json.loads(current.read_text(encoding="utf-8"))
            run_id = selection.get("run_id", "")
            run_dir = (ROOT / selection.get("directory", "")).resolve()
            if not re.fullmatch(r"[a-f0-9]{16}", run_id) or not run_dir.is_relative_to(ROOT / "evaluation") or run_dir.name != run_id:
                raise HTTPException(500, "Invalid current evaluation path")
            path = run_dir / filename
            if not path.is_file():
                raise HTTPException(404, "The selected evaluation report is unavailable")
            return path
        if not latest.is_file():
            raise HTTPException(404, "Run python -m jev_qat.evaluate run first")
        run_id = json.loads(latest.read_text(encoding="utf-8")).get("run_id", "")
        if not re.fullmatch(r"[a-f0-9]{16}", run_id):
            raise HTTPException(500, "Invalid evaluation run identifier")
        path = directory / "runs" / run_id / filename
        if not path.is_file():
            raise HTTPException(404, "The evaluation is still running or has no completed report")
        return path

    @app.get("/evaluation", response_class=FileResponse)
    def evaluation_report():
        return FileResponse(evaluation_artifact("report.html"), media_type="text/html")

    @app.get("/evaluation/summary")
    def evaluation_summary():
        return json.loads(evaluation_artifact("summary.json").read_text(encoding="utf-8"))

    @app.post("/admin/stop", include_in_schema=False)
    def stop(x_management_token: str = Header(default="")):
        if not hmac.compare_digest(x_management_token, token):
            raise HTTPException(403, "Management token required")
        app.state.uvicorn.should_exit = True
        return {"status": "stopping"}

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=config["api_port"], log_level="info"))
    app.state.uvicorn = server
    server.run()
