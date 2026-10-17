"""Check the installer's private runtime without downloading the model or GPU use."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", type=Path, required=True)
    base = parser.parse_args().install_dir.resolve()
    assert Path(sys.executable).is_relative_to(base / "python")
    import fastapi
    import httpx
    import pydantic
    import psutil
    import uvicorn
    from fastapi.testclient import TestClient
    from jev_qat.e2b import E2BPredictionRequest
    from jev_qat.config import ROOT
    assert ROOT == base / "app"
    for package in (fastapi, httpx, pydantic, psutil, uvicorn):
        assert Path(package.__file__).resolve().is_relative_to(base / "python/Lib/site-packages")
    request = E2BPredictionRequest(state="texto", question="¿Es texto?", options=["sí", "no"])
    assert request.options == ["sí", "no"]
    app = fastapi.FastAPI()
    @app.get("/health")
    def health():
        return {"status": "ok"}
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
    record = json.loads((base / "install.json").read_text())
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        subprocess.run([str(Path(record["ffmpeg_bin"]) / name), "-version"], check=True,
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    assert record["status"] == "dependencies_ready"
    assert not (base / "app/integration/jev-qat-e2b/service.json").exists()
    assert not list((base / "app/models").rglob("*.gguf"))
    print(json.dumps({"private_python": True, "private_packages": True, "unicode_install_path": True,
                      "fastapi_request": True, "ffmpeg_ffprobe": True,
                      "model_not_downloaded": True, "service_not_started": True, "device": record["device"]}))


if __name__ == "__main__":
    main()
