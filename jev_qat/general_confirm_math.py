"""Combine independent full-label confirmation with a proposed General decision."""
import math

from .eval_data import CATEGORIES

GENERAL = list(CATEGORIES).index("general")


def combine(proposed,confirmation,weight,minimum_probability):
    for values in (proposed,confirmation):
        if len(values) != 18 or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(math.fsum(values),1.,abs_tol=1e-6):
            raise ValueError("Invalid confirmation distribution")
    thresholds = minimum_probability if isinstance(minimum_probability,dict) else {"default":minimum_probability}
    if not 0 <= weight <= 1 or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError("Invalid confirmation parameters")
    winner = list(CATEGORIES)[max(range(18),key=confirmation.__getitem__)]
    minimum = thresholds.get(winner,thresholds["default"])
    if max(range(18),key=proposed.__getitem__) != GENERAL or max(confirmation) < minimum:
        return list(proposed)
    result = [(1-weight)*a+weight*b for a,b in zip(proposed,confirmation)]
    total = math.fsum(result)
    return [p/total for p in result]
