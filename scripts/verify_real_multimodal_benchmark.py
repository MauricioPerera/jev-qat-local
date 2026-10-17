"""Audit recorded traces, original files, frame diversity and comparisons."""
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.adapter import JevQatAdapter

OUT = ROOT / "inference" / "real-media-benchmark-20261017"


def main():
    report = json.loads((OUT / "run-01" / "results.json").read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in (OUT / "run-01" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(report["trials"]) == 56 and all(report["checks"].values())
    cases = {c["id"]: c for c in report["cases"]}
    for fixture in report["media_manifest"]["fixtures"]:
        path = Path(fixture["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
        if path.suffix == ".wav":
            duration = float(fixture["metadata"]["format"]["duration"])
            assert duration in (15, 30) and duration <= 30
        if path.suffix == ".mp4":
            duration = float(fixture["metadata"]["format"]["duration"])
            assert duration in (30, 60)
    for trial in report["trials"]:
        response = trial["result"]
        assert response is not None and trial["error"] is None
        metadata = response["metadata"]
        assert metadata["benchmark_cache_prompt"] is False
        assert metadata["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert metadata["timings"]["cache_n"] == 0
        assert metadata["usage"]["completion_tokens"] == 1
        assert metadata["runtime_fingerprint"] == "b11669-620d2225c"
        assert metadata["usage"]["prompt_tokens"] < report["configuration"]["context_tokens"]
        payload = cases[trial["case"]]["payload"]
        assert set(response["probabilities"]) == set(payload["options"])
        assert abs(math.fsum(response["probabilities"].values()) - 1) < 1e-6
        if payload.get("modality") == "video":
            assert metadata["media_items"] == payload["video_frames"]
        selected = [s for s in trace if s["end"] > trial["start"] and s["start"] < trial["end"]]
        weights = [min(s["end"], trial["end"]) - max(s["start"], trial["start"]) for s in selected]
        duration = trial["end"] - trial["start"]
        assert abs(sum(weights) - duration) < 1e-6
        gpu = sum(s["gpu_busy_percent"] * w for s, w in zip(selected, weights)) / duration
        assert abs(gpu - trial["resources"]["gpu_mean_percent"]) < 1e-6
        for key, source in (("rss_peak_gib", "rss_bytes"), ("gpu_shared_peak_gib", "gpu_shared_bytes"),
                            ("committed_peak_gib", "private_committed_bytes")):
            assert abs(max(s[source] for s in selected) / 1024**3 - trial["resources"][key]) < 1e-9
        total_delta = trial["cpu_job_after"]["seconds"] - trial["cpu_job_before"]["seconds"]
        expected_cpu = 100 * total_delta / duration / 16
        assert abs(expected_cpu - trial["cpu_total_mean_percent"]) < 1e-8
    for row in report["comparison"]:
        trials = [t for t in report["trials"] if t["case"] == row["case"] and t["round"] > 0]
        assert len(trials) == 3
        assert abs(statistics.median(t["wall_seconds"] for t in trials) - row["median_seconds"]) < 1e-9
    # Exercise only decoding, never another model request: real frames must differ.
    video_checks = []
    with JevQatAdapter() as adapter:
        for identifier in ("video_30_2", "video_30_16", "video_60_2", "video_60_16"):
            payload = cases[identifier]["payload"]
            with adapter._content(payload["media"], "video", payload["video_frames"]) as content:
                hashes = [hashlib.sha256(base64.b64decode(c["image_url"]["url"].split(",", 1)[1])).hexdigest() for c in content]
            assert len(hashes) == payload["video_frames"] and len(set(hashes)) > 1
            video_checks.append({"case": identifier, "frames": len(hashes), "unique_frames": len(set(hashes)), "sha256": hashes})
    content_failures = [{"case": t["case"], "round": t["round"], "prediction": t["result"]["prediction"],
                         "confidence": t["result"]["confidence"]} for t in report["trials"] if not t["content_check_matches"]]
    verification = {"passed": True, "successful_inferences": 56, "every_native_cache_count_zero": True,
                    "request_interval_coverage": "complete", "resource_aggregates_reproduced": True,
                    "job_cpu_totals_reproduced": True, "all_frame_counts_correct": True,
                    "source_hashes_verified": True, "real_video_frames": video_checks,
                    "source_based_content_check_failures": content_failures,
                    "content_checks_are_not_independent_human_gold": True}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in verification.items() if k != "real_video_frames"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
