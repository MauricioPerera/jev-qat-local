"""Probe native multimodal prefix reuse without changing prompts or the service."""
import json
from pathlib import Path
import shutil
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.e2b import E2BQatAdapter

OUT = ROOT / "inference/e2b-prepare-reuse"
OUT.mkdir(exist_ok=True)
baseline = OUT / "baseline"
baseline.mkdir(exist_ok=True)
for rel in ("jev_qat/e2b.py", "jev_qat/e2b_web.py", "jev_qat/e2b_budget.py", "jev_qat/web/e2b.js", "jev_qat_e2b_config.json"):
    target = baseline / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(ROOT / rel, target)

old = json.loads((ROOT / "inference/e2b-shared-file-benchmark/plan.json").read_text(encoding="utf8"))
results = []
with E2BQatAdapter() as adapter:
    adapter._verify_labels(256)
    for case in old["cases"]:
        for index, q in enumerate(case["questions"]):
            started = time.perf_counter()
            result = adapter.predict(state="", question=q["question"], options=q["options"],
                                     media=case["source"], modality=case["modality"],
                                     video_frames=case["video_frames"], cache_prompt=index > 0)
            row = {"case": case["id"], "question_id": index, "seconds": time.perf_counter()-started, "result": result}
            results.append(row)
            (OUT / "native-probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps({"case": row["case"], "q": index, "seconds": round(row["seconds"], 3),
                              "usage": result["metadata"]["usage"], "timings": result["metadata"]["timings"]}), flush=True)
