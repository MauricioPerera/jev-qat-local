"""Third iteration: target-category development, fresh test and hybrid calibration."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
from itertools import product

import httpx
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .calibration import PROFILES, sha256_file
from .config import ROOT, save_json, settings
from .eval_data import CATEGORIES, DATASET, DIRECTORY as V1, QUESTION, SOURCE_HASHES, SEED, digest, disjoint_sources, load_records, stratified_sample, text_group
from .eval_metrics import metrics, scaled_array
from .evaluate import arrays, collect
from .spanish_task import VERSION as V2_VERSION

DIRECTORY = ROOT / "evaluation" / "massive-es-v3"
V2 = ROOT / "evaluation" / "massive-es-v2"
FOCUS = ("general", "lists", "news")
CS = (4.0, 16.0)
WEIGHTS = (.2, .4, .6, .8)
HEADS = ROOT / "classification" / "massive-es-v3-heads.joblib"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_records(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records), encoding="utf-8")
    return {"file": path.name, "sha256": sha256_file(path), "count": len(records),
            "class_counts": dict(sorted(Counter(row["target"] for row in records).items()))}


def focused_sample(rows, total, per_focus, seed):
    chosen = []
    for index, category in enumerate(FOCUS):
        pool = [row for row in rows if row["label"] == category]
        if len(pool) < per_focus:
            raise ValueError(f"Insufficient untouched {category} examples: {len(pool)}")
        random.Random(seed+index).shuffle(pool)
        chosen.extend(pool[:per_focus])
    rest = [row for row in rows if row["label"] not in FOCUS]
    chosen.extend(stratified_sample(rest, total-len(chosen), seed+10))
    random.Random(seed).shuffle(chosen)
    return chosen


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path = DIRECTORY / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for info in manifest["splits"].values():
            if sha256_file(DIRECTORY / info["file"]) != info["sha256"]:
                raise ValueError("Frozen v3 partition changed")
        return manifest
    prior = [row for directory in (V1, V2) for split in ("calibration", "validation", "test") for row in load_records(directory, split)]
    excluded_ids = {row["source_id"] for row in prior}
    excluded_groups = {row["group_id"] for row in prior}
    sources = {}
    for split, expected in SOURCE_HASHES.items():
        path = V1 / "source" / f"{split}-es.json.gz"
        if sha256_file(path) != expected:
            raise ValueError("Pinned source changed")
        sources[split] = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    cleaned, deduplication = disjoint_sources(sources)
    eligible = {key: [row for row in values if str(row["id"]) not in excluded_ids and text_group(row["text"]) not in excluded_groups]
                for key, values in cleaned.items()}
    # All v3 development/calibration/validation come from unused official
    # validation, which is entirely outside the frozen v2 training bank.
    dev = focused_sample(eligible["validation"], 300, 25, SEED+301)
    used = {str(row["id"]) for row in dev}
    pool = [row for row in eligible["validation"] if str(row["id"]) not in used]
    calibration = stratified_sample(pool, 300, SEED+302)
    used.update(str(row["id"]) for row in calibration)
    validation = stratified_sample([row for row in pool if str(row["id"]) not in used], 180, SEED+303)
    test = focused_sample(eligible["test"], 600, 50, SEED+304)
    splits = {}
    for split, official, selected in (("development", "validation", dev), ("calibration", "validation", calibration),
                                       ("validation", "validation", validation), ("test", "test", test)):
        records = []
        for row in selected:
            order = list(CATEGORIES)
            random.Random(f"v3:{row['id']}").shuffle(order)
            records.append({"id": f"massive:es:{official}:{row['id']}", "source_id": str(row["id"]),
                            "source_split": official, "group_id": text_group(row["text"]), "state": row["text"],
                            "question": QUESTION, "options": [CATEGORIES[key] for key in order], "option_labels": order,
                            "target": row["label"], "modality": "text"})
        splits[split] = write_records(DIRECTORY / f"{split}.jsonl", records)
    bank_profile = read_json(ROOT / "classification" / "profiles" / "massive_es_v2.json")
    manifest = {"schema_version": 3, "dataset": DATASET, "language": "es", "splits": splits,
                "deduplication": deduplication, "excluded_prior_record_ids": len(excluded_ids),
                "training_bank_file": bank_profile["bank_file"], "training_bank_sha256": bank_profile["bank_sha256"],
                "training_count": bank_profile["training_count"],
                "protocol": {"seed": SEED+300, "focus": list(FOCUS), "C_grid": list(CS), "weight_grid": list(WEIGHTS),
                             "development": "300 untouched official validation, 25 examples per focus category; remaining categories proportional.",
                             "test": "600 untouched official test: 50 per focus category and 450 proportional examples from the other 15 categories.",
                             "sampling_caution": "Focus categories are deliberately enriched; paired comparisons refer to this distribution, not historical natural samples.",
                             "selection_gate": "Mean focus F1 improves >=0.02; every focus F1 >=baseline-0.01; overall accuracy and macro F1 >=baseline-0.005.",
                             "mixture": "Convex average of frozen calibrated v2 Gemma probabilities and supervised TF-IDF probabilities, only if either argmax is a focus class.",
                             "calibration": "300 natural-proportion unused validation records fit scalar temperature; another 180 gate validation NLL improvement >0.0001.",
                             "shared_component": "Both versions reuse the same frozen v2 Gemma evidence per message; no second LLM inference is necessary for deterministic postprocessing.",
                             "test_used_for_selection": False},
                "prior_manifests_sha256": {directory.name: sha256_file(directory / "manifest.json") for directory in (V1, V2)}}
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(manifest_path, manifest)
    analyze_errors()
    return manifest


def analyze_errors():
    run = Path(read_json(V2 / "latest-run.json")["directory"])
    cases = read_json(run / "test-cases.json")
    relevant = [row for row in cases if not row["v2_correct"] and (row["target"] in FOCUS or row["v2_prediction"] in FOCUS)]
    pairs = Counter((row["target"], row["v2_prediction"]) for row in relevant)
    save_json(DIRECTORY / "error-analysis.json", {"diagnostic_only": True, "prior_run": run.name,
              "relevant_errors": relevant, "confusions": [{"target": a, "prediction": b, "count": n} for (a,b), n in pairs.most_common()],
              "hypotheses": ["General misses conversational statements and open-ended requests.",
                             "Lists competes with contacts and topic words instead of the requested collection operation.",
                             "News competes with recommendations, factual QA and ambiguous short phrases."],
              "label_policy": "Preserve original labels, including ambiguous examples; do not train on prior tests."})


def train_heads(manifest):
    if HEADS.exists():
        return
    bank_path = ROOT / manifest["training_bank_file"]
    if sha256_file(bank_path) != manifest["training_bank_sha256"]:
        raise ValueError("Training bank checksum mismatch")
    bank = joblib.load(bank_path)
    labels = np.array([row["target"] for row in bank["records"]])
    models, convergence = {}, {}
    with threadpool_limits(limits=4):
        for strength in CS:
            print(f"Fitting CPU lexical classifier C={strength:g}, n={len(labels)}...", flush=True)
            model = LogisticRegression(C=strength, max_iter=500, tol=1e-3, solver="lbfgs", random_state=SEED)
            model.fit(bank["matrix"], labels)
            if max(model.n_iter_) >= model.max_iter:
                raise RuntimeError("Lexical classifier did not converge")
            models[str(strength)] = model
            convergence[str(strength)] = model.n_iter_.tolist()
    joblib.dump({"models": models, "source_bank_sha256": manifest["training_bank_sha256"]}, HEADS, compress=3)
    save_json(DIRECTORY / "lexical-training.json", {"heads_sha256": sha256_file(HEADS), "convergence_iterations": convergence,
              "source_bank_sha256": manifest["training_bank_sha256"], "training_count": len(labels)})


def lexical_scores(model, records, bank):
    with threadpool_limits(limits=4):
        features = normalize(bank["vectorizer"].transform([row["state"] for row in records]))
        values = model.predict_proba(features)
    columns = [list(model.classes_).index(label) for label in CATEGORIES]
    return values[:, columns]


def mix_array(gemma_calibrated, lexical, weight):
    focus_indices = [list(CATEGORIES).index(label) for label in FOCUS]
    gate = np.isin(gemma_calibrated.argmax(axis=1), focus_indices) | np.isin(lexical.argmax(axis=1), focus_indices)
    if isinstance(weight, dict):
        names = list(CATEGORIES)
        dominant = [names[a] if names[a] in FOCUS else names[b] for a,b in zip(gemma_calibrated.argmax(axis=1), lexical.argmax(axis=1))]
        values = np.array([weight.get(label, 0.0) for label in dominant])[:, None]
    else:
        values = weight
    mixture = (1-values)*gemma_calibrated+values*lexical
    return np.where(gate[:, None], mixture, gemma_calibrated)


def augment_heads(manifest):
    artifact = joblib.load(HEADS)
    if "4.0_weighted" in artifact["models"]:
        return
    bank = joblib.load(ROOT / manifest["training_bank_file"])
    weights = {label: (3.0 if label == "general" else 1.5 if label == "news" else 1.2 if label == "lists" else 1.0) for label in CATEGORIES}
    with threadpool_limits(limits=4):
        model = LogisticRegression(C=4, class_weight=weights, max_iter=500, tol=1e-3, solver="lbfgs", random_state=SEED)
        model.fit(bank["matrix"], [row["target"] for row in bank["records"]])
    if max(model.n_iter_) >= model.max_iter:
        raise RuntimeError("Weighted classifier did not converge")
    artifact["models"]["4.0_weighted"] = model
    joblib.dump(artifact, HEADS, compress=3)
    save_json(DIRECTORY / "development-plan-extension.json", {
        "reason": "The initial global-mixture trials failed the predeclared development gate; the test has not been queried.",
        "extension": "Per-category support weights selected on development only; dominant Gemma category selects the weight when it is a focus class, otherwise the lexical focus category does.",
        "weight_grid_per_category": list(WEIGHTS), "extra_training_model": {"C": 4, "class_weight": weights},
        "unchanged_gate": manifest["protocol"]["selection_gate"], "test_used": False,
        "heads_sha256": sha256_file(HEADS), "weighted_model_iterations": model.n_iter_.tolist()})


def development():
    manifest = prepare()
    train_heads(manifest)
    augment_heads(manifest)
    config = settings()
    records = load_records(DIRECTORY, "development")
    with httpx.Client(base_url=config["api_url"], timeout=config["request_timeout_seconds"], trust_env=False) as client:
        predictions = collect(client, [row | {"task_profile": "massive_es_v2"} for row in records],
                              DIRECTORY / "development-gemma.jsonl", DIRECTORY, "development-v2-component", V2_VERSION)
    raw, targets = arrays(predictions)
    temperature = read_json(PROFILES / "massive_scenario_es_v2.json")["temperature"]
    calibrated = scaled_array(raw, temperature)
    baseline = metrics(calibrated, targets, list(CATEGORIES))
    bank = joblib.load(ROOT / manifest["training_bank_file"])
    models = joblib.load(HEADS)["models"]
    trials = []
    for name, model in models.items():
        lexical = lexical_scores(model, records, bank)
        for combination in product(WEIGHTS, repeat=3):
            weight = dict(zip(FOCUS, combination))
            candidate = metrics(mix_array(calibrated, lexical, weight), targets, list(CATEGORIES))
            before = {row["label"]: row["f1"] for row in baseline["per_class"]}
            after = {row["label"]: row["f1"] for row in candidate["per_class"]}
            gain = float(np.mean([after[label]-before[label] for label in FOCUS]))
            approved = (gain >= .02 and all(after[label] >= before[label]-.01 for label in FOCUS)
                        and candidate["accuracy"] >= baseline["accuracy"]-.005
                        and candidate["macro_f1"] >= baseline["macro_f1"]-.005)
            trials.append({"head_id": name, "C": float(model.C), "weight": weight, "approved": approved, "mean_focus_f1_gain": gain, "metrics": candidate})
    eligible = [row for row in trials if row["approved"]]
    selected = max(eligible, key=lambda row: (row["mean_focus_f1_gain"], row["metrics"]["macro_f1"], row["metrics"]["accuracy"])) if eligible else None
    result = {"baseline": baseline, "trials": trials, "selected": selected, "test_used_for_selection": False,
              "gemma_component_temperature": temperature, "gemma_predictions_sha256": sha256_file(DIRECTORY / "development-gemma.jsonl")}
    save_json(DIRECTORY / "development-selection.json", result)
    print(json.dumps({"baseline_accuracy": baseline["accuracy"], "selected": {"head_id": selected["head_id"], "C": selected["C"], "weight": selected["weight"],
                     "accuracy": selected["metrics"]["accuracy"], "focus_f1_gain": selected["mean_focus_f1_gain"]} if selected else None}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "development", "run", "report"))
    command = parser.parse_args().command
    if command == "prepare":
        manifest = prepare()
        train_heads(manifest)
        print(json.dumps({key: info["count"] for key, info in manifest["splits"].items()}))
    elif command == "development":
        development()
    else:
        from .focus_run import run, report
        (run if command == "run" else report)()


if __name__ == "__main__":
    main()
