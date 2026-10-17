"""Fetch attributed public recordings and prepare explicit bounded excerpts."""
import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multimodal_resources import probe

OUT = ROOT / "inference" / "real-media-benchmark-20261017"
SOURCES = {
    "alps.jpg": "File:Fronalpstock big.jpg",
    "paris.jpg": "File:Paris - Eiffelturm und Marsfeld2.jpg",
    "speech-blade-runner.ogg": "File:Es-Blade Runner 2.ogg",
    "speech-tierra.ogg": "File:Es-tierra 3of4-article.ogg",
    "bus.webm": "File:Waiting for a Bus.webm",
    "marseille.webm": "File:Marseille par drone.webm",
}
HEADERS = {"User-Agent": "LocalGemmaResourceBenchmark/1.0 (public media performance measurement)"}


def plain(value):
    return html.unescape(re.sub("<[^>]+>", " ", value)).strip()


def acquire(item):
    name, title = item
    with httpx.Client(timeout=90, follow_redirects=True, trust_env=False, headers=HEADERS) as client:
        response = client.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "titles": title,
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iilimit": 1,
        })
        response.raise_for_status()
        metadata = response.json()
        page = next(iter(metadata["query"]["pages"].values()))
        info = page["imageinfo"][0]
        url = urlsplit(info["url"])
        assert url.hostname == "upload.wikimedia.org"
        media_url = info["url"]
        path = OUT / "source" / name
        if not path.exists() or path.stat().st_size != info["size"]:
            temporary = path.with_suffix(path.suffix + ".part")
            with client.stream("GET", media_url) as download:
                download.raise_for_status()
                count = 0
                with temporary.open("wb") as stream:
                    for chunk in download.iter_bytes():
                        count += len(chunk)
                        if count > 100 * 1024**2:
                            raise ValueError("Unexpected source size")
                        stream.write(chunk)
            assert count == info["size"]
            temporary.replace(path)
        ext = info["extmetadata"]
        record = {"filename": name, "title": title, "source_page": info["descriptionurl"],
                  "download_url": media_url, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "author": plain(ext.get("Artist", {}).get("value", "")),
                  "license": ext.get("LicenseShortName", {}).get("value"),
                  "license_url": ext.get("LicenseUrl", {}).get("value"),
                  "description": plain(ext.get("ImageDescription", {}).get("value", ""))}
        (OUT / "source" / (name + ".metadata.json")).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"downloaded": name, "bytes": record["bytes"], "license": record["license"]}), flush=True)
        return record


def ffmpeg(arguments):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *arguments], check=True,
                   timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    (OUT / "fixtures").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        sources = list(executor.map(acquire, SOURCES.items()))
    derived = []
    for name in ("alps.jpg", "paris.jpg"):
        shutil.copy2(OUT / "source" / name, OUT / "fixtures" / name)
        derived.append({"fixture": name, "source": name, "transformation": "original image bytes unchanged"})
    for source, target, duration in (("speech-blade-runner.ogg", "speech-15s.wav", 15),
                                     ("speech-tierra.ogg", "speech-30s.wav", 30)):
        args = ["-ss", "10", "-i", str(OUT / "source" / source), "-t", str(duration),
                "-map", "0:a:0", "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le",
                str(OUT / "fixtures" / target)]
        ffmpeg(args)
        derived.append({"fixture": target, "source": source, "start_seconds": 10,
                        "duration_seconds": duration, "transformation": "explicit excerpt; mono PCM 22050 Hz", "ffmpeg": args})
    for source, target, duration in (("bus.webm", "bus-30s.mp4", 30), ("marseille.webm", "marseille-60s.mp4", 60)):
        args = ["-i", str(OUT / "source" / source), "-t", str(duration), "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                "-r", "25", "-movflags", "+faststart", str(OUT / "fixtures" / target)]
        ffmpeg(args)
        derived.append({"fixture": target, "source": source, "start_seconds": 0, "duration_seconds": duration,
                        "transformation": "explicit excerpt; H264 25 fps; original dimensions; video-only", "ffmpeg": args})
    files = [probe(OUT / "fixtures" / record["fixture"]) for record in derived]
    for file in files:
        assert Path(file["path"]).stat().st_size < 33554432
    for name in ("bus-30s.mp4", "marseille-60s.mp4"):
        ffmpeg(["-i", str(OUT / "fixtures" / name), "-vf", "fps=1/10,scale=400:-2,tile=3x2",
                "-frames:v", "1", str(OUT / (name + ".contact.png"))])
    manifest = {"sources": sources, "derivations": derived, "fixtures": files,
                "download_and_clip_preparation_excluded_from_inference_latency": True}
    (OUT / "media-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    credits = ["# Procedencia de medios reales", "",
               "Originales descargados para medir inferencia local. Fotos sin alterar; audio y vídeo recortados explícitamente. "
               "Los recortes se mantienen bajo la licencia indicada de su fuente. No se utiliza generación sintética.", ""]
    for source in sources:
        credits += [f"- [{source['title']}]({source['source_page']}): {source['author']}. "
                    f"[{source['license']}]({source['license_url']}). Archivo: {source['filename']}."]
    (OUT / "ATTRIBUTION.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUT / "media-manifest.json"), "fixtures": files}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
