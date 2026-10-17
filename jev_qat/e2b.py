"""Independent E2B QAT Q4 adapter and local service; E4B stays unchanged."""
import argparse
from contextlib import asynccontextmanager, contextmanager
import hmac
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
import httpx
import psutil
from pydantic import BaseModel, Field
import uvicorn

from .adapter import BackendError, JevQatAdapter
from .config import ROOT, save_json
from .e2b_budget import CONTEXT_LIMIT, ContextBudgetError, check_context_budget, measure_context_budget
from .e2b_web import register_web_routes
from .e2b_media import PreparedMediaCache
from .service import PredictionRequest, check_port

ALIAS = "gemma-e2b-qat-q4-local"
REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
STATE = ROOT / "integration" / "jev-qat-e2b"
RUNTIME_STATE = STATE / "service.json"
COMPARISON = ROOT / "inference" / "gemma-e2b-vs-e4b"


class _TokenBudgetReady(Exception):
    """Unwind shared prompt/media preparation before reaching completion I/O."""
    def __init__(self, budget, media_items):
        self.budget, self.media_items = budget, media_items
        super().__init__("Token count completed without inference")


def settings():
    config = json.loads((ROOT / "jev_qat_e2b_config.json").read_text(encoding="utf-8"))
    for field in ("runtime", "model", "projector"):
        config[field] = str((ROOT / config[field]).resolve())
    config["media_roots"] = [str((ROOT / root).resolve()) for root in config["media_roots"]]
    config["backend_url"] = f"http://127.0.0.1:{config['backend_port']}"
    config["api_url"] = f"http://127.0.0.1:{config['api_port']}"
    return config


class E2BQatAdapter(JevQatAdapter):
    """Same native option scoring and preprocessing, with E2B model identity."""
    def __init__(self, base_url=None, config=None):
        super().__init__(base_url=base_url, config=config or settings())
        self._identity_verified = False
        self._cache_prompt = True
        self._context_budget = None
        self._count_only = False
        self._reuse_media = True
        self._media_cache = PreparedMediaCache()
        self._last_media_preparation = None

    def close(self):
        with self._lock:
            self._media_cache.release()
            super().close()

    def release_media(self, path):
        with self._lock:
            self._media_cache.release(path)

    @contextmanager
    def _content(self, media, modality, video_frames):
        loader = lambda: super(E2BQatAdapter, self)._content(media, modality, video_frames)
        with self._media_cache.content(media=media, modality=modality, frames=video_frames,
                                      adapter=self, loader=loader, enabled=self._reuse_media) as (parts, receipt):
            self._last_media_preparation = receipt
            yield parts

    def _verify_identity(self):
        if not self._identity_verified:
            try:
                response = self.client.get("/v1/models")
                response.raise_for_status()
                identifiers = {entry["id"] for entry in response.json()["data"]}
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                raise BackendError("Could not verify the local E2B backend identity") from error
            if identifiers != {ALIAS}:
                raise BackendError("E2B adapter requires its own gemma-e2b-qat-q4-local backend")
            self._identity_verified = True

    def _verify_labels(self, count):
        self._verify_identity()
        try:
            super()._verify_labels(count)
        except BackendError as error:
            raise BackendError("E2B tokenizer did not return the required single-token option labels") from error

    def _post(self, route, payload):
        if route == "/v1/chat/completions":
            payload = {**payload, "cache_prompt": self._cache_prompt}
            if self._count_only:
                budget = measure_context_budget(self, payload)
                media_items = sum(item["type"] != "text" for message in payload["messages"] for item in message["content"])
                raise _TokenBudgetReady(budget, media_items)
            self._context_budget = check_context_budget(self, payload)
        return super()._post(route, payload)

    def count_tokens(self, **kwargs):
        """Prepare the identical predict() request and measure it, even if too large.

        The private signal exits the base predict() before completion transport or
        score parsing; its media context manager still cleans up temporary files.
        The lock prevents this mode from affecting concurrent predict() callers.
        """
        with self._lock:
            self._count_only = True
            try:
                self.predict(**kwargs)
            except _TokenBudgetReady as counted:
                budget = counted.budget
                return {
                    **budget, "fits_context": budget["remaining_tokens"] >= 0,
                    "available_tokens": max(0, budget["remaining_tokens"]),
                    "overflow_tokens": max(0, -budget["remaining_tokens"]),
                    "inference_executed": False, "backend": ALIAS,
                    "modality": kwargs.get("modality", "text"), "option_count": len(kwargs["options"]),
                    "media_items": counted.media_items,
                    "media_preparation": self._last_media_preparation,
                }
            finally:
                self._count_only = False
        raise BackendError("E2B token counting did not reach the prepared request")

    def predict(self, *, cache_prompt=True, reuse_media=True, **kwargs):
        if not isinstance(cache_prompt, bool):
            raise ValueError("cache_prompt must be a boolean")
        if not isinstance(reuse_media, bool):
            raise ValueError("reuse_media must be a boolean")
        if kwargs.get("calibration_profile") is not None:
            raise ValueError("E2B has no validated calibration profile; E4B calibration cannot be reused")
        if kwargs.get("task_profile") not in (None, "massive_es_v2"):
            raise ValueError("E2B supports the shared massive_es_v2 retrieval prompt; E4B rescue profiles are not approved for E2B")
        with self._lock:
            self._cache_prompt = cache_prompt
            self._reuse_media = reuse_media
            self._last_media_preparation = None
            self._context_budget = None
            result = super().predict(**kwargs)
            result["metadata"].update({
                "backend": "gemma-4-E2B-it-qat-q4_0-gguf", "model_repository": REPO,
                "model_revision": REVISION, "cache_prompt": cache_prompt,
                "calibrated": False, "calibration": None,
                "context_budget": self._context_budget,
                "media_preparation": self._last_media_preparation,
            })
            return result


def load_jev_omni_e2b_qat(base_url=None):
    return E2BQatAdapter(base_url=base_url)


def classify_spanish_e2b(message, *, cache_prompt=True, api_url=None):
    from .eval_data import CATEGORIES, QUESTION
    from .spanish_task import DEFINITIONS
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty Spanish string")
    if not isinstance(cache_prompt, bool):
        raise ValueError("cache_prompt must be a boolean")
    payload = {"state": message, "question": QUESTION, "options": list(CATEGORIES.values()),
               "task_profile": "massive_es_v2", "modality": "text", "cache_prompt": cache_prompt}
    with httpx.Client(base_url=api_url or settings()["api_url"], timeout=180, trust_env=False) as client:
        response = client.post("/predict", json=payload)
        response.raise_for_status()
        result = response.json()
    result["category"] = next(k for k, v in CATEGORIES.items() if v == result["prediction"])
    result["category_description"] = DEFINITIONS[result["category"]]
    result["category_probabilities"] = {k: result["probabilities"][v] for k, v in CATEGORIES.items()}
    return result


class E2BPredictionRequest(PredictionRequest):
    cache_prompt: bool = Field(default=True, strict=True)
    reuse_media: bool = Field(default=True, strict=True,
        description="Reuse bounded, content-verified audio/video/image preparation. Token counting still runs for each question.")


class E2BTokenBudgetResponse(BaseModel):
    input_tokens: int = Field(description="Complete input including chat template, instructions, options, examples and prepared media.")
    reserved_output_tokens: int
    total_tokens: int = Field(description="Input plus reserved output tokens.")
    context_limit_tokens: int
    remaining_tokens: int = Field(description="Signed capacity after input and reserved output; negative if too large.")
    available_tokens: int = Field(description="Remaining capacity, clamped to zero.")
    overflow_tokens: int = Field(description="Tokens exceeding the context limit, or zero when it fits.")
    fits_context: bool
    inference_executed: bool
    backend: str
    modality: str
    option_count: int
    media_items: int = Field(description="Actual images, converted audio clips or extracted video frames counted.")
    counter: str
    includes_multimodal: bool
    checked_before_inference: bool
    media_preparation: dict | None = None


def register_prediction_routes(app):
    def invoke(operation, request):
        try:
            return operation(**request.model_dump())
        except ContextBudgetError as error:
            raise HTTPException(422, error.detail) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except BackendError as error:
            raise HTTPException(502, str(error)) from error

    @app.post("/predict")
    def predict(request: E2BPredictionRequest):
        return invoke(app.state.adapter.predict, request)

    @app.post("/count-tokens", response_model=E2BTokenBudgetResponse,
              summary="Count the complete request without inference",
              description="Same input as /predict. Counts text, options, instructions, retrieved examples and prepared image/audio/video content. Returns HTTP 200 with fits_context=false for oversized requests; does not generate a response. Audio is limited to 30 seconds; video uses the requested sampled frames.",
              responses={422: {"description": "Invalid request or unsupported profile."},
                         502: {"description": "Backend identity, context or token count could not be verified."}})
    def count_tokens(request: E2BPredictionRequest):
        return invoke(app.state.adapter.count_tokens, request)


def health():
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            response = client.get(settings()["api_url"] + "/health")
            result = response.json()
            if response.status_code == 200 and result.get("service") == "jev-qat-e2b-bridge":
                return result
    except (httpx.HTTPError, ValueError):
        pass
    return None


def serve():
    config = settings()
    for port in (config["api_port"], config["backend_port"]):
        check_port(port)
    for field in ("model", "projector"):
        with Path(config[field]).open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise RuntimeError(f"Invalid E2B {field} file")
    STATE.mkdir(parents=True, exist_ok=True)
    arguments = [config["runtime"], "--model", config["model"], "--mmproj", config["projector"],
                 "--device", config["device"], "--gpu-layers", "all", "--ctx-size", str(config["context_tokens"]),
                 "--parallel", "1", "--threads", str(config["threads"]), "--threads-batch", str(config["threads"]),
                 "--host", "127.0.0.1", "--port", str(config["backend_port"]), "--reasoning", "off", "--alias", ALIAS]
    token = secrets.token_urlsafe(32)
    state = {"service": "jev-qat-e2b-bridge", "status": "starting", "workspace": str(ROOT),
             "pid": psutil.Process().pid, "created": psutil.Process().create_time(),
             "api_url": config["api_url"], "backend_url": config["backend_url"], "management_token": token}

    @asynccontextmanager
    async def lifespan(app):
        backend = adapter = None
        try:
            with (STATE / "backend.log").open("a", encoding="utf-8") as log:
                backend = subprocess.Popen(arguments, cwd=Path(config["runtime"]).parent, stdout=log, stderr=log,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
            state.update({"backend_pid": backend.pid, "backend_created": psutil.Process(backend.pid).create_time()})
            save_json(RUNTIME_STATE, state)
            started = time.monotonic()
            with httpx.Client(base_url=config["backend_url"], timeout=2, trust_env=False) as client:
                while time.monotonic() - started < 300:
                    if backend.poll() is not None:
                        raise RuntimeError("E2B backend exited during startup; inspect integration/jev-qat-e2b/backend.log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.5)
                else:
                    raise TimeoutError("E2B startup exceeded 300 seconds")
            adapter = E2BQatAdapter(config=config)
            adapter._verify_labels(256)
            app.state.backend, app.state.adapter = backend, adapter
            state["status"] = "ready"
            save_json(RUNTIME_STATE, state)
            yield
        except Exception as error:
            state.update({"status": "failed", "error": str(error)})
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

    app = FastAPI(title="Gemma E2B QAT local decision adapter", version="0.1.0", lifespan=lifespan,
                  description="Text, image, audio and sampled-video option scoring. E2B probabilities are uncalibrated.")

    @app.get("/health")
    def status():
        if app.state.backend.poll() is not None:
            raise HTTPException(503, "E2B backend stopped")
        return {"service": "jev-qat-e2b-bridge", "status": "ready", "backend": ALIAS,
                "device": config["device"], "modalities": ["text", "image", "audio", "video"],
                "max_options": 256, "calibrated": False, "original_jev_omni_head_used": False,
                "context_limit_tokens": min(CONTEXT_LIMIT, config["context_tokens"]), "context_budget_check": True}

    register_prediction_routes(app)
    register_web_routes(app, config, STATE, on_remove=lambda path: app.state.adapter.release_media(path))

    @app.get("/comparison")
    def comparison():
        path = COMPARISON / "report.html"
        if not path.is_file():
            raise HTTPException(404, "Comparison is not finished yet")
        return FileResponse(path, media_type="text/html")

    @app.get("/comparison/summary")
    def comparison_summary():
        path = COMPARISON / "summary.json"
        if not path.is_file():
            raise HTTPException(404, "Comparison is not finished yet")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/comparison/figure")
    def comparison_figure():
        path = COMPARISON / "comparison.png"
        if not path.is_file():
            raise HTTPException(404, "Comparison figure is not finished yet")
        return FileResponse(path, media_type="image/png")

    @app.post("/admin/stop", include_in_schema=False)
    def shutdown(x_management_token: str = Header(default="")):
        if not hmac.compare_digest(x_management_token, token):
            raise HTTPException(403, "Management token required")
        app.state.server.should_exit = True
        return {"status": "stopping"}

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=config["api_port"], log_level="info"))
    app.state.server = server
    server.run()


def start():
    if health():
        return {"status": "already_running", "api_url": settings()["api_url"]}
    config = settings()
    check_port(config["api_port"])
    check_port(config["backend_port"])
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "service.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "jev_qat.e2b", "serve"], cwd=ROOT, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    started = time.monotonic()
    while time.monotonic() - started < 320:
        if process.poll() is not None:
            raise RuntimeError("E2B startup failed; inspect integration/jev-qat-e2b/service.log")
        if health():
            return {"status": "ready", "pid": process.pid, "api_url": config["api_url"], "docs": config["api_url"] + "/docs"}
        time.sleep(.5)
    raise TimeoutError("E2B startup timed out; inspect local logs")


def stop():
    if not RUNTIME_STATE.is_file():
        return {"status": "not_running"}
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
    if state.get("workspace") != str(ROOT):
        raise RuntimeError("E2B state belongs to another workspace")
    with httpx.Client(timeout=5, trust_env=False) as client:
        try:
            client.post(state["api_url"] + "/admin/stop", headers={"X-Management-Token": state["management_token"]}).raise_for_status()
        except httpx.HTTPError:
            pass
    config = settings()
    for _ in range(30):
        alive = []
        for pid_key, created_key in (("pid", "created"), ("backend_pid", "backend_created")):
            try:
                process = psutil.Process(state[pid_key])
                if abs(process.create_time() - state[created_key]) < .01:
                    alive.append(process)
            except (psutil.NoSuchProcess, KeyError):
                pass
        if not alive:
            return {"status": "stopped"}
        time.sleep(.5)
    for process in alive:
        try:
            command = process.cmdline()
            is_service = "jev_qat.e2b" in command and "serve" in command
            is_backend = config["model"] in command and Path(command[0]).resolve() == Path(config["runtime"]).resolve()
            if not (is_service or is_backend):
                raise RuntimeError("E2B process identity changed; refusing to terminate it")
            process.terminate()
            process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
    state["status"] = "stopped"
    save_json(RUNTIME_STATE, state)
    return {"status": "stopped"}


def main():
    parser = argparse.ArgumentParser(description="Independent Gemma E2B QAT Q4 multimodal adapter")
    parser.add_argument("command", choices=("start", "stop", "status", "serve", "predict"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        serve()
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
        with httpx.Client(base_url=settings()["api_url"], timeout=180, trust_env=False) as client:
            response = client.post("/predict", json=json.loads(args.input.read_text(encoding="utf-8-sig")))
            response.raise_for_status()
            result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
