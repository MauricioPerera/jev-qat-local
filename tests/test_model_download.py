"""Exercise the installer's shared range downloader with small GGUF fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def downloader(tmp_path, monkeypatch, fail=False):
    spec = importlib.util.spec_from_file_location("fixture_downloader", ROOT / "scripts/download_gemma_e4b_qat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STATE = tmp_path / "state"
    module.DESTINATION = tmp_path / "models"
    module.PROGRESS = module.STATE / "progress.json"
    module.RECEIPT = module.STATE / "receipt.json"
    module.CHUNK = 64
    blobs = {"model.gguf": b"GGUF" + bytes(range(252)) * 2,
             "projector.gguf": b"GGUF" + bytes(range(100))}
    module.FILES = {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in blobs.items()}
    requests = []
    state = {"fail": fail}
    def transport(request):
        name = request.url.path.rsplit("/", 1)[1]
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        requests.append((name, start, end))
        if state["fail"] and name == "model.gguf" and start == 64:
            return httpx.Response(503)
        return httpx.Response(206, headers={"Content-Range": f"bytes {start}-{end}/{len(blobs[name])}"},
                              content=blobs[name][start:end + 1])
    client_type = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(transport), **kwargs))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module, blobs, requests, state


def test_downloads_both_verified_files_and_reuses_complete_models(tmp_path, monkeypatch):
    module, blobs, calls, _ = downloader(tmp_path, monkeypatch)
    module.main()
    for name, data in blobs.items():
        assert (module.DESTINATION / name).read_bytes() == data
    assert json.loads(module.RECEIPT.read_text())["status"] == "complete"
    count = len(calls)
    module.main()
    assert len(calls) == count


def test_interrupted_ranges_resume_and_verify_full_sha256(tmp_path, monkeypatch):
    module, blobs, calls, state = downloader(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="rerun to resume"):
        module.main()
    checkpoint = json.loads(module.PROGRESS.read_text())
    completed = {(name, start * module.CHUNK) for name, item in checkpoint["files"].items()
                 for start in item["completed_ranges"]}
    state["fail"] = False
    before = len(calls)
    module.main()
    assert not completed.intersection((name, start) for name, start, end in calls[before:])
    for name, data in blobs.items():
        assert (module.DESTINATION / name).read_bytes() == data
    assert all(item["sha256_verified"] for item in json.loads(module.RECEIPT.read_text())["files"].values())
