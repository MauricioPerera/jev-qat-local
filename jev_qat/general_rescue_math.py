"""Probability-preserving rescue of general; other categories retain v2 ratios."""
import math

from .calibration import temperature_scale
from .eval_data import CATEGORIES

GENERAL=list(CATEGORIES).index("general")


def needs_review(base, maximum_confidence, component_temperature):
    calibrated=temperature_scale(base,component_temperature)
    return max(range(len(base)),key=base.__getitem__)!=GENERAL and max(calibrated)<=maximum_confidence


def combine(base,focused,weight,minimum_probability=0.):
    if len(base)!=18 or len(focused)!=18 or not math.isfinite(weight) or not 0<=weight<=1 or not 0<=minimum_probability<=1:
        raise ValueError("Invalid rescue distributions/weight")
    for values in (base,focused):
        if any(not math.isfinite(v) or v<0 for v in values) or not math.isclose(math.fsum(values),1,abs_tol=1e-6):
            raise ValueError("Invalid rescue probabilities")
    if (max(range(18),key=base.__getitem__)==GENERAL or max(range(18),key=focused.__getitem__)!=GENERAL
            or focused[GENERAL]<minimum_probability):
        return list(base)
    old=base[GENERAL]
    new=max(old,(1-weight)*old+weight*focused[GENERAL])
    factor=(1-new)/(1-old)
    result=[p*factor for p in base]
    result[GENERAL]=new
    total=math.fsum(result)
    return [p/total for p in result]
