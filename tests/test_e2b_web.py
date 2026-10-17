from pathlib import Path
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from jev_qat.e2b_web import register_web_routes


def client_for(tmp_path, limit=32 * 1024 * 1024):
    app = FastAPI()
    register_web_routes(app, {"api_url": "http://127.0.0.1:18167", "api_port": 18167,
                              "max_media_bytes": limit}, tmp_path)
    return TestClient(app)


def test_ui_and_assets_are_served_locally_with_the_inference_button_locked(tmp_path):
    with client_for(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert '<html lang="es">' in page.text and 'id="infer"' in page.text
        assert re.search(r'<button[^>]+id="infer"[^>]+disabled', page.text)
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        for path, kind in (("/ui/e2b.css", "text/css"), ("/ui/e2b.js", "text/javascript")):
            response = client.get(path)
            assert response.status_code == 200 and kind in response.headers["content-type"]
            assert response.headers["x-content-type-options"] == "nosniff"
        assert client.get("/ui/../e2b.py").status_code == 404
        assert client.get("/uploads").status_code == 405


@pytest.mark.parametrize("filename,modality,extension", [
    ("foto.PNG", "image", ".png"), ("voz.mp3", "audio", ".mp3"), ("clip.webm", "video", ".webm"),
])
def test_upload_uses_its_own_random_path_and_can_remove_only_that_copy(tmp_path, filename, modality, extension):
    original = tmp_path / filename
    original.write_bytes(b"original")
    with client_for(tmp_path) as client:
        response = client.post("/uploads", params={"filename": filename}, content=b"uploaded bytes")
        assert response.status_code == 200
        receipt = response.json()
        path = Path(receipt["media"])
        assert path.parent == tmp_path / "uploads"
        assert re.fullmatch(r"[0-9a-f]{32}" + re.escape(extension), path.name)
        assert receipt["filename"] == filename and receipt["modality"] == modality
        assert path.read_bytes() == b"uploaded bytes" and receipt["bytes"] == 14
        assert client.delete("/uploads/" + receipt["upload_id"]).status_code == 204
        assert not path.exists() and original.read_bytes() == b"original"
        assert client.delete("/uploads/" + receipt["upload_id"]).status_code == 204


@pytest.mark.parametrize("filename,status", [("page.html", 415), ("../foto.png", 422),
                                            ("C:\\foto.png", 422), ("folder/foto.png", 422)])
def test_upload_rejects_unsupported_names_before_writing(tmp_path, filename, status):
    with client_for(tmp_path) as client:
        assert client.post("/uploads", params={"filename": filename}, content=b"test").status_code == status
    assert not (tmp_path / "uploads").exists()


def test_upload_enforces_actual_stream_size_and_cleans_up_partial_files(tmp_path):
    with client_for(tmp_path, limit=6) as client:
        assert client.post("/uploads?filename=empty.png", content=b"").status_code == 400
        assert client.post("/uploads?filename=large.png", content=b"0123456").status_code == 413
        response = client.post("/uploads?filename=chunked.png", content=iter([b"0123", b"4567"]))
        assert response.status_code == 413
        response = client.post("/uploads?filename=wrong-length.png", headers={"Content-Length": "1"}, content=b"0123456")
        assert response.status_code == 413
        assert not list((tmp_path / "uploads").iterdir())
        response = client.post("/uploads?filename=exact.png", content=b"012345")
        assert response.status_code == 200 and response.json()["bytes"] == 6


def test_upload_and_removal_reject_unrelated_browser_origins(tmp_path):
    with client_for(tmp_path) as client:
        bad = {"Origin": "https://unrelated.example"}
        assert client.post("/uploads?filename=foto.png", content=b"test", headers=bad).status_code == 403
        good = {"Origin": "http://127.0.0.1:18167"}
        receipt = client.post("/uploads?filename=foto.png", content=b"test", headers=good).json()
        assert client.delete("/uploads/" + receipt["upload_id"], headers=bad).status_code == 403
        assert Path(receipt["media"]).exists()
        assert client.delete("/uploads/" + receipt["upload_id"], headers=good).status_code == 204
        assert client.delete("/uploads/not-an-upload").status_code == 404
