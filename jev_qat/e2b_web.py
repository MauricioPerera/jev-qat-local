"""Local E2B form and bounded uploads; prediction and counting keep their API."""
from pathlib import Path
import re
import uuid

from fastapi import HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

WEB = Path(__file__).with_name("web")
EXTENSIONS = {
    **dict.fromkeys((".jpg", ".jpeg", ".png", ".webp"), "image"),
    **dict.fromkeys((".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"), "audio"),
    **dict.fromkeys((".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"), "video"),
}
HEADERS = {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; script-src 'self'; style-src 'self'; frame-ancestors 'none'; form-action 'self'",
}


def register_web_routes(app, config, state_directory, on_remove=None):
    uploads = (Path(state_directory) / "uploads").resolve()
    origins = {config["api_url"], f"http://localhost:{config['api_port']}"}

    def check_origin(request):
        if request.headers.get("origin") not in (None, *origins):
            raise HTTPException(403, "Use the local E2B page to upload or remove files")

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(WEB / "e2b.html", headers=HEADERS)

    @app.get("/ui/e2b.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(WEB / "e2b.css", media_type="text/css", headers=HEADERS)

    @app.get("/ui/e2b.js", include_in_schema=False)
    def javascript():
        return FileResponse(WEB / "e2b.js", media_type="text/javascript", headers=HEADERS)

    @app.post("/uploads", summary="Upload one local image, audio or video file")
    async def upload(request: Request, filename: str = Query(min_length=1, max_length=255)):
        check_origin(request)
        if any(char in filename for char in ("/", "\\", ":", "\x00")):
            raise HTTPException(422, "Supply a filename without a directory")
        extension = Path(filename).suffix.lower()
        if extension not in EXTENSIONS:
            raise HTTPException(415, "Unsupported image, audio or video extension")
        limit = config["max_media_bytes"]
        if "content-length" in request.headers:
            try:
                length = int(request.headers["content-length"])
            except ValueError as error:
                raise HTTPException(400, "Invalid Content-Length") from error
            if length < 0 or length > limit:
                raise HTTPException(413, f"The file must be at most {limit} bytes")
        uploads.mkdir(parents=True, exist_ok=True)
        identifier = uuid.uuid4().hex
        target = uploads / (identifier + extension)
        partial = uploads / (identifier + ".part")
        size = 0
        try:
            with partial.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"The file must be at most {limit} bytes")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "The file is empty")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return {"upload_id": identifier, "filename": filename, "media": str(target),
                "modality": EXTENSIONS[extension], "bytes": size}

    @app.delete("/uploads/{upload_id}", status_code=204, summary="Remove a file uploaded by this interface")
    def remove(upload_id: str, request: Request):
        check_origin(request)
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise HTTPException(404, "Upload not found")
        for extension in EXTENSIONS:
            target = uploads / (upload_id + extension)
            if target.is_file() and target.resolve().parent == uploads:
                target.unlink()
            if on_remove is not None:
                on_remove(target)
        return Response(status_code=204)
