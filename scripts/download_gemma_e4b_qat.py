"""Download the official QAT Q4 model and projector with resumable ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import httpx


ROOT = Path(__file__).resolve().parents[1]
REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
REVISION = "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"
DESTINATION = ROOT / "models" / "gemma-4-E4B-it-QAT-Q4_0"
STATE = ROOT / "compatibility" / "qat-q4"
PROGRESS = STATE / "download-progress.json"
RECEIPT = STATE / "download-receipt.json"
CHUNK = 16 * 1024 * 1024
FILES = {
    "gemma-4-E4B-it-mmproj.gguf": {
        "size": 991552256,
        "sha256": "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",
    },
    "gemma-4-E4B_q4_0-it.gguf": {
        "size": 5154941280,
        "sha256": "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
    },
}


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    completed = {name: set() for name in FILES}
    if PROGRESS.exists():
        checkpoint = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if checkpoint["revision"] != REVISION or checkpoint["chunk_size"] != CHUNK:
            raise ValueError("Download checkpoint does not match this revision")
        completed = {name: set(checkpoint["files"][name]["completed_ranges"]) for name in FILES}
    partials = {name: DESTINATION / (name + ".http-partial") for name in FILES}
    verified = set()
    for name, spec in FILES.items():
        final = DESTINATION / name
        if final.exists():
            if final.stat().st_size != spec["size"] or digest(final) != spec["sha256"]:
                raise ValueError(f"Existing file checksum mismatch: {name}")
            verified.add(name)
            completed[name] = set(range((spec["size"] + CHUNK - 1) // CHUNK))
        elif not partials[name].exists():
            partials[name].touch(exist_ok=False)
            completed[name] = set()
        elif partials[name].stat().st_size > spec["size"]:
            raise ValueError(f"Partial file exceeds expected size: {name}")

    total = sum(spec["size"] for spec in FILES.values())

    def save_progress(status):
        data = {name: {
            **spec, "completed_ranges": sorted(completed[name]),
            "completed_bytes": sum(min(CHUNK, spec["size"] - index * CHUNK) for index in completed[name]),
            "sha256_verified": name in verified,
        } for name, spec in FILES.items()}
        count = sum(item["completed_bytes"] for item in data.values())
        write_json(PROGRESS, {
            "model": REPO, "revision": REVISION, "status": status,
            "process_id": os.getpid(), "chunk_size": CHUNK,
            "completed_bytes": count, "total_bytes": total, "files": data,
        })
        return count

    baseline = save_progress("downloading")
    started = time.monotonic()
    last_report = 0
    with httpx.Client(
        follow_redirects=True, timeout=httpx.Timeout(60, connect=20),
        limits=httpx.Limits(max_connections=12, max_keepalive_connections=12),
        headers={"Accept-Encoding": "identity"},
    ) as client:
        def fetch(name, index):
            spec = FILES[name]
            first = index * CHUNK
            last = min(spec["size"] - 1, first + CHUNK - 1)
            expected = f"bytes {first}-{last}/{spec['size']}"
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            for attempt in range(6):
                try:
                    with client.stream("GET", url, headers={"Range": f"bytes={first}-{last}"}) as response:
                        if response.status_code != 206 or response.headers.get("content-range") != expected:
                            raise ValueError(f"Unexpected range response (HTTP {response.status_code})")
                        received = 0
                        with partials[name].open("r+b") as stream:
                            stream.seek(first)
                            for block in response.iter_bytes(chunk_size=1024 * 1024):
                                if received + len(block) > last - first + 1:
                                    raise ValueError("Range exceeds expected size")
                                stream.write(block)
                                received += len(block)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if received != last - first + 1:
                            raise ValueError("Truncated byte range")
                    return name, index
                except (httpx.HTTPError, ValueError):
                    if attempt == 5:
                        raise RuntimeError(f"Range {index} failed for {name}; rerun to resume") from None
                    time.sleep(min(2 ** attempt, 10))

        tasks = [(name, index) for name, spec in FILES.items() if name not in verified
                 for index in range((spec["size"] + CHUNK - 1) // CHUNK)
                 if index not in completed[name]]
        print(f"Official QAT Q4 + multimodal projector: {total / 1e9:.3f} GB total.", flush=True)
        print(f"Resuming at {baseline / 1e9:.3f} GB; {len(tasks)} ranges remaining.", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, name, index) for name, index in tasks]
            for future in as_completed(futures):
                name, index = future.result()
                completed[name].add(index)
                count = save_progress("downloading")
                now = time.monotonic()
                if now - last_report >= 25 or count == total:
                    rate = (count - baseline) / max(now - started, 1) / 1e6
                    print(f"{count / 1e9:.3f}/{total / 1e9:.3f} GB ({100 * count / total:.1f}%), average {rate:.1f} MB/s", flush=True)
                    last_report = now

    save_progress("verifying")
    for name, spec in FILES.items():
        if name in verified:
            continue
        print(f"Verifying SHA-256: {name}", flush=True)
        if partials[name].stat().st_size != spec["size"] or digest(partials[name]) != spec["sha256"]:
            raise ValueError(f"Complete file checksum mismatch: {name}")
        with partials[name].open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError(f"Missing GGUF file header: {name}")
        os.replace(partials[name], DESTINATION / name)
        verified.add(name)
        save_progress("verifying")
    save_progress("complete")
    receipt = {
        "model": REPO, "revision": REVISION, "status": "complete",
        "path": str(DESTINATION), "total_bytes": total,
        "files": {name: {**spec, "sha256_verified": True} for name, spec in FILES.items()},
    }
    write_json(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        STATE.mkdir(parents=True, exist_ok=True)
        write_json(RECEIPT, {"model": REPO, "revision": REVISION, "status": "failed", "error": str(error)})
        if PROGRESS.exists():
            failed_progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
            failed_progress["status"] = "failed"
            write_json(PROGRESS, failed_progress)
        raise
