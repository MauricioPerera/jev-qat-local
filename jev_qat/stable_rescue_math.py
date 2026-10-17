"""Continuous General rescue; native scores are never thresholded at .95."""
import math

from .eval_data import CATEGORIES

GENERAL = list(CATEGORIES).index("general")
QA = list(CATEGORIES).index("qa")


def validate(values):
    if len(values) != 18 or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(math.fsum(values), 1., abs_tol=1e-6):
        raise ValueError("Invalid stable-rescue distribution")


def smooth(value, low, high):
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid probability ramp")
    x = min(1., max(0., (value-low)/(high-low)))
    return x*x*(3-2*x)


def strength(base, focused, cfg, similarity):
    validate(base)
    if not math.isfinite(similarity) or not 0 <= similarity <= 1.000001:
        raise ValueError("Invalid training-evidence similarity")
    if max(range(18), key=base.__getitem__) == GENERAL:
        return 1.
    if focused is None:
        return 0.
    validate(focused)
    winner = list(CATEGORIES)[max(range(18), key=base.__getitem__)]
    if winner in cfg["protected_domains"] and similarity < cfg["minimum_general_similarity"]:
        return 0.
    return smooth(focused[GENERAL], cfg["ramp_low"], cfg["ramp_high"])


def proposal(base, focused, cfg, similarity):
    r = strength(base, focused, cfg, similarity)
    result = list(base)
    if focused is None or r == 0 or max(range(18), key=base.__getitem__) == GENERAL:
        return result, r
    alpha = cfg["rescue_weight"] * r
    if not 0 <= alpha <= 1:
        raise ValueError("Invalid rescue weight")
    old = base[GENERAL]
    new = max(old, (1-alpha)*old + alpha*focused[GENERAL])
    factor = (1-new)/(1-old)
    result = [v*factor for v in base]
    result[GENERAL] = new
    return [v/math.fsum(result) for v in result], r


def combine(base, focused, confirmation, cfg, similarity):
    p, r = proposal(base, focused, cfg, similarity)
    if confirmation is None:
        if r > 0:
            raise ValueError("Positive rescue strength requires independent confirmation")
        return p
    validate(confirmation)
    # At most one class can exceed .5. Sum of these continuous ramps avoids
    # an argmax branch and the previous hard knowledge-confidence cutoff.
    reliability = sum(smooth(v, cfg["qa_confirmation_low"] if i == QA else cfg["confirmation_low"],
                             cfg["qa_confirmation_high"] if i == QA else cfg["confirmation_high"])
                      for i, v in enumerate(confirmation))
    if reliability > 1.000001:
        raise ValueError("Confirmation ramps overlap")
    weight = cfg["confirmation_weight"] * r * min(1., reliability)
    if not 0 <= weight <= 1:
        raise ValueError("Invalid confirmation weight")
    result = [(1-weight)*a + weight*b for a, b in zip(p, confirmation)]
    return [v/math.fsum(result) for v in result]
