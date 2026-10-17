"""Independently reproduce paired summaries and validate model/source hashes."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "gemma-e2b-vs-e4b"


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    report = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert len(report["trials"]) == 80 and len(report["cases"]) == 10
    assert all(report["checks"].values())
    case_lookup = {c["id"]: c for c in report["cases"]}
    traces = {m: [json.loads(line) for line in (OUT / f"samples-{m}.jsonl").read_text(encoding="utf-8").splitlines()] for m in ("e4b", "e2b")}
    previous_end = 0
    for trial in report["trials"]:
        assert trial["start"] >= previous_end
        previous_end = trial["end"]
        result = trial["result"]
        metadata = result["metadata"]
        assert metadata["backend"] == f"gemma-4-{trial['model'].upper()}-it-qat-q4_0-gguf"
        assert metadata["runtime_fingerprint"] == "b11669-620d2225c"
        assert metadata["calibrated"] is False
        assert metadata["usage"]["completion_tokens"] == 1
        assert metadata["usage"]["prompt_tokens_details"]["cached_tokens"] == metadata["timings"]["cache_n"] == 0
        payload = case_lookup[trial["case"]]["payload"]
        assert set(result["probabilities"]) == set(payload["options"])
        assert abs(math.fsum(result["probabilities"].values()) - 1) < 1e-6
        assert payload["options"][result["prediction_index"]] == result["prediction"]
        if payload.get("modality") == "video":
            assert metadata["media_items"] == payload["video_frames"]
        selected = [s for s in traces[trial["model"]] if s["end"] > trial["start"] and s["start"] < trial["end"]]
        weights = [min(s["end"], trial["end"]) - max(s["start"], trial["start"]) for s in selected]
        duration = trial["end"] - trial["start"]
        assert abs(sum(weights) - duration) < 1e-6
        expected_gpu = sum(s["gpu_busy_percent"] * w for s, w in zip(selected, weights)) / duration
        assert abs(expected_gpu - trial["resources"]["gpu_mean_percent"]) < 1e-6
        expected_cpu = 100 * (trial["cpu_job_after"]["seconds"] - trial["cpu_job_before"]["seconds"]) / duration / 16
        assert abs(expected_cpu - trial["cpu_total_mean_percent"]) < 1e-8
        assert abs(max(s["rss_bytes"] for s in selected)/1024**3 - trial["resources"]["rss_peak_gib"]) < 1e-9
    for row in report["comparison"]:
        for model in ("e4b", "e2b"):
            trials = [t for t in report["trials"] if t["case"] == row["case"] and t["model"] == model]
            assert len(trials) == 4
            warm = [t for t in trials if t["round"] > 0]
            expected = statistics.median(t["wall_seconds"] for t in warm)
            assert abs(expected - row[model]["median_seconds"]) < 1e-8
        expected_ratio = row["e4b"]["median_seconds"] / row["e2b"]["median_seconds"]
        assert abs(expected_ratio - row["e2b_speed_ratio"]) < 1e-8
    verified = []
    for model, files in report["model_files"].items():
        for file in files:
            path = Path(file["path"])
            print(f"Verifying {model.upper()} {path.name}", flush=True)
            assert path.stat().st_size == file["bytes"]
            assert file_sha(path) == file["sha256"]
            verified.append({"path": str(path), "sha256": file["sha256"]})
    for path, expected in report["frozen_files"].items():
        assert file_sha(Path(path)) == expected
    media_manifest = json.loads((ROOT / "inference" / "real-media-benchmark-20261017" / "media-manifest.json").read_text(encoding="utf-8"))
    for fixture in media_manifest["fixtures"]:
        assert file_sha(Path(fixture["path"])) == fixture["sha256"]
    assert json.loads((ROOT / "classification" / "current.json").read_text(encoding="utf-8"))["version"] == "v2"
    integration = json.loads((ROOT / "integration" / "jev-qat-e2b" / "integration-results.json").read_text(encoding="utf-8"))
    assert integration["passed"]
    verification = {"passed": True, "requests": 80, "unique_cases": 10, "model_file_hashes": verified,
                    "same_runtime": True, "no_overlapping_inferences": True, "native_cached_tokens_all_zero": True,
                    "resource_summaries_reproduced": True, "cpu_children_accounted": True,
                    "same_real_media_hashes_verified": True,
                    "e2b_live_integration_passed": True, "e4b_default_version_preserved": "v2"}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in verification.items() if k != "model_file_hashes"}), flush=True)


if __name__ == "__main__":
    main()
