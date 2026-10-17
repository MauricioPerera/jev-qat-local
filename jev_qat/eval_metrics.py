"""Metrics and scalar temperature fitting for multiclass probabilities."""
import math

import numpy as np
from scipy.optimize import minimize_scalar


def scaled_array(probabilities, temperature):
    scores = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-300, 1)) / temperature
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)


def negative_log_loss(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    return float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-300, 1)).mean())


def fit_temperature(probabilities, targets):
    result = minimize_scalar(
        lambda log_t: negative_log_loss(scaled_array(probabilities, np.exp(log_t)), targets),
        bounds=(math.log(0.05), math.log(100)), method="bounded", options={"xatol": 1e-8},
    )
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("Temperature optimization did not converge")
    return float(np.exp(result.x))


def wilson_interval(correct, count):
    if count == 0:
        return [None, None]
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def metrics(probabilities, targets, classes, bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if probabilities.shape != (len(targets), len(classes)) or not len(targets):
        raise ValueError("Probability matrix shape does not match targets and classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probability rows must be finite, nonnegative and sum to one")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    per_class = []
    for index, label in enumerate(classes):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class.append({"label": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    assignments = np.minimum((confidence*bins).astype(int), bins-1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        average = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count/len(targets)*abs(average-accuracy)
        reliability.append({"lower": index/bins, "upper": (index+1)/bins, "count": count,
                            "confidence": average, "accuracy": accuracy})
    one_hot = np.eye(len(classes))[targets]
    coverage = []
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        selected = confidence >= threshold
        count = int(selected.sum())
        successes = int(correct[selected].sum())
        coverage.append({"threshold": threshold, "count": count, "coverage": count/len(targets),
                         "accuracy": successes/count if count else None, "accuracy_95ci": wilson_interval(successes, count)})
    return {
        "n": len(targets), "accuracy": float(correct.mean()),
        "accuracy_95ci": wilson_interval(int(correct.sum()), len(targets)),
        "macro_precision": float(np.mean([item["precision"] for item in per_class])),
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "negative_log_loss": negative_log_loss(probabilities, targets),
        "brier_multiclass_sum": float(((probabilities-one_hot)**2).sum(axis=1).mean()),
        "ece_10_bins": ece, "mean_confidence": float(confidence.mean()),
        "errors_with_confidence_ge_0_9": int(((~correct) & (confidence >= 0.9)).sum()),
        "reliability": reliability, "coverage": coverage, "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def paired_nll_difference_ci(raw, calibrated, targets, seed=20261016, repetitions=2000):
    targets = np.asarray(targets, dtype=int)
    rows = np.arange(len(targets))
    raw_loss = -np.log(np.clip(np.asarray(raw)[rows, targets], 1e-300, 1))
    calibrated_loss = -np.log(np.clip(np.asarray(calibrated)[rows, targets], 1e-300, 1))
    differences = calibrated_loss - raw_loss
    rng = np.random.default_rng(seed)
    means = [float(differences[rng.integers(0, len(targets), len(targets))].mean()) for _ in range(repetitions)]
    return {"mean_calibrated_minus_raw": float(differences.mean()),
            "paired_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
            "repetitions": repetitions, "seed": seed}
