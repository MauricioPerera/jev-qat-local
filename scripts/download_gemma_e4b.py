"""Resume a pinned Gemma E4B download using verified HTTP byte ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import httpx


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "gemma-4-E4B-it" / "model.safetensors"
PARTIAL = DESTINATION.with_suffix(".safetensors.http-partial")
PROGRESS = ROOT / "compatibility" / "download-progress.json"
RECEIPT = ROOT / "compatibility" / "download-receipt.json"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SHA256 = "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
SIZE = 15992595884
CHUNK = 32 * 1024 * 1024
URL = f"https://huggingface.co/google/gemma-4-E4B-it/resolve/{REVISION}/model.safetensors"
ESSENTIAL_FILES = {
    "config.json": 5145, "generation_config.json": 208,
    "processor_config.json": 1689, "chat_template.jinja": 18569,
    "tokenizer.json": 32169626, "tokenizer_config.json": 3082,
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    started = time.monotonic()
    for name, expected_size in ESSENTIAL_FILES.items():
        candidate = DESTINATION.parent / name
        if not candidate.exists() or candidate.stat().st_size != expected_size:
            raise ValueError(f"Missing or incomplete auxiliary file: {name}")
    completed = set()
    if PROGRESS.exists() and PARTIAL.exists():
        checkpoint = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if checkpoint["sha256"] != SHA256 or checkpoint["chunk_size"] != CHUNK:
            raise ValueError("Download checkpoint belongs to a different file")
        completed = set(checkpoint["completed_chunks"])
    if not DESTINATION.exists():
        if not PARTIAL.exists():
            PARTIAL.touch(exist_ok=False)
        elif PARTIAL.stat().st_size > SIZE:
            raise ValueError("Partial file exceeds expected size")

        def save_progress():
            temporary = PROGRESS.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "model": "google/gemma-4-E4B-it", "revision": REVISION,
                "status": "downloading", "process_id": os.getpid(),
                "sha256": SHA256, "total_bytes": SIZE, "chunk_size": CHUNK,
                "completed_chunks": sorted(completed),
                "completed_bytes": sum(min(CHUNK, SIZE - index * CHUNK) for index in completed),
            }, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, PROGRESS)

        save_progress()
        with httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(60, connect=20),
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=12),
            headers={"Accept-Encoding": "identity"},
        ) as client:
            def fetch(index):
                first = index * CHUNK
                last = min(SIZE - 1, first + CHUNK - 1)
                expected_range = f"bytes {first}-{last}/{SIZE}"
                for attempt in range(5):
                    try:
                        with client.stream("GET", URL, headers={"Range": f"bytes={first}-{last}"}) as response:
                            if response.status_code != 206 or response.headers.get("content-range") != expected_range:
                                raise ValueError(f"Unexpected byte range response (HTTP {response.status_code})")
                            received = 0
                            with PARTIAL.open("r+b") as stream:
                                stream.seek(first)
                                for block in response.iter_bytes(chunk_size=1024 * 1024):
                                    if received + len(block) > last - first + 1:
                                        raise ValueError("Range exceeded expected size")
                                    stream.write(block)
                                    received += len(block)
                                stream.flush()
                                os.fsync(stream.fileno())
                            if received != last - first + 1:
                                raise ValueError("Truncated byte range")
                        return index
                    except (httpx.HTTPError, ValueError):
                        if attempt == 4:
                            raise RuntimeError(f"Chunk {index} failed after five attempts") from None
                        time.sleep(min(2 ** attempt, 10))

            indexes = range((SIZE + CHUNK - 1) // CHUNK)
            remaining = [index for index in indexes if index not in completed]
            baseline = sum(min(CHUNK, SIZE - index * CHUNK) for index in completed)
            last_report = 0
            print(f"Downloading {len(remaining)} ranges; {baseline / 1e9:.2f} GB already complete.", flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(fetch, index) for index in remaining]
                for future in as_completed(futures):
                    completed.add(future.result())
                    save_progress()
                    now = time.monotonic()
                    if now - last_report >= 25 or len(completed) == len(indexes):
                        count = sum(min(CHUNK, SIZE - index * CHUNK) for index in completed)
                        rate = (count - baseline) / max(now - started, 1) / 1e6
                        print(f"{count / 1e9:.2f}/{SIZE / 1e9:.2f} GB ({100 * count / SIZE:.1f}%), average {rate:.1f} MB/s", flush=True)
                        last_report = now
        print("Verifying complete file SHA-256...", flush=True)
        if digest(PARTIAL) != SHA256:
            raise ValueError("Complete file checksum mismatch; partial retained for inspection")
        os.replace(PARTIAL, DESTINATION)
    else:
        if DESTINATION.stat().st_size != SIZE or digest(DESTINATION) != SHA256:
            raise ValueError("Existing model file does not match the pinned checkpoint")

    receipt = {
        "model": "google/gemma-4-E4B-it", "revision": REVISION,
        "path": str(DESTINATION), "bytes": SIZE, "sha256": SHA256,
        "sha256_verified": True, "status": "complete",
        "auxiliary_files_present": sorted(ESSENTIAL_FILES),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        RECEIPT.write_text(json.dumps({
            "model": "google/gemma-4-E4B-it", "revision": REVISION,
            "status": "failed", "error": str(error),
        }, indent=2) + "\n", encoding="utf-8")
        raise
