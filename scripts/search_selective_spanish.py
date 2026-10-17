"""Development-only conservative rescue search; never reads test predictions."""
import json
from pathlib import Path

import joblib
import numpy as np

from jev_qat.config import ROOT, save_json
from jev_qat.eval_data import CATEGORIES, load_records
from jev_qat.eval_metrics import metrics, scaled_array
from jev_qat.evaluate import arrays
from jev_qat.focus_experiment import DIRECTORY, FOCUS, HEADS, lexical_scores, read_json


def selective(gemma, lexical, weight, threshold):
    prediction = lexical.argmax(axis=1)
    active = np.isin(prediction, [list(CATEGORIES).index(c) for c in FOCUS]) & (lexical.max(axis=1) >= threshold)
    return np.where(active[:, None], (1-weight)*gemma+weight*lexical, gemma)


def main():
    run_dir = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    config = read_json(ROOT / "classification/profiles/massive_es_v3.json")
    bank = joblib.load(ROOT / read_json(DIRECTORY / "manifest.json")["training_bank_file"])
    data = {}
    for split in ("development", "validation"):
        path = DIRECTORY / "development-gemma.jsonl" if split == "development" else run_dir / "validation-gemma.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        raw, targets = arrays(rows)
        data[split] = (scaled_array(raw, config["gemma_component_temperature"]), targets, load_records(DIRECTORY, split))
    joined = np.concatenate([item[0] for item in data.values()])
    targets = np.concatenate([item[1] for item in data.values()])
    baseline = metrics(joined, targets, list(CATEGORIES))
    old = {r["label"]: r["f1"] for r in baseline["per_class"]}
    trials = []
    for head_id, model in joblib.load(HEADS)["models"].items():
        lexical = np.concatenate([lexical_scores(model, item[2], bank) for item in data.values()])
        for threshold in (.3, .5, .7, .9):
            for weight in (.4, .6, .8, 1.):
                mixed = selective(joined, lexical, weight, threshold)
                candidate = metrics(mixed, targets, list(CATEGORIES))
                new = {r["label"]: r["f1"] for r in candidate["per_class"]}
                gain = np.mean([new[c]-old[c] for c in FOCUS])
                accs = [float((mixed[:300].argmax(1)==targets[:300]).mean()), float((mixed[300:].argmax(1)==targets[300:]).mean())]
                approved = (gain >= .01 and all(new[c] >= old[c]-1e-12 for c in FOCUS)
                            and candidate["accuracy"] > baseline["accuracy"]
                            and candidate["macro_f1"] >= baseline["macro_f1"]
                            and all(acc >= float((p.argmax(1)==y).mean()) for acc,(p,y,_) in zip(accs,data.values())))
                trials.append({"head_id": head_id, "threshold": threshold, "weight": weight, "mean_focus_f1_gain": float(gain), "approved": bool(approved), "metrics": candidate})
    approved = [row for row in trials if row["approved"]]
    selected = max(approved, key=lambda row:(row["metrics"]["accuracy"],row["metrics"]["macro_f1"],row["threshold"])) if approved else None
    result = {"baseline": baseline, "trials": trials, "selected": selected, "selection_records": 480,
              "previous_test_used_for_diagnosis_only": True, "fresh_test_used": False}
    save_json(ROOT / "evaluation/massive-es-v4/development-selection.json", result)
    print(json.dumps({"baseline": baseline["accuracy"], "selected": {k:v for k,v in selected.items() if k!='metrics'} if selected else None,
                      "selected_accuracy": selected["metrics"]["accuracy"] if selected else None,
                      "focus": [(r['label'],r['f1']) for r in selected['metrics']['per_class'] if r['label'] in FOCUS] if selected else None}))


if __name__ == "__main__":
    main()
