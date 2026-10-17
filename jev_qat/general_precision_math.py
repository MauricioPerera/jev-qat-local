"""Evidence gate for general rescues of contextual device/tool requests."""
import math
from functools import lru_cache

from .eval_data import CATEGORIES
from .general_rescue_math import GENERAL, combine as previous_combine, needs_review

# These domains execute an action or address a channel. Elliptical requests can
# express those actions without an explicit imperative (e.g. lighting requests).
CONTEXTUAL_TOOLS = ("alarm", "audio", "email", "iot", "lists", "play", "social", "takeaway", "transport")


def combine(base, focused, config, general_similarity):
    if not math.isfinite(general_similarity) or not 0 <= general_similarity <= 1.000001:
        raise ValueError("Invalid training-evidence similarity")
    proposed = previous_combine(base, focused, config["weight"], config["minimum_probability"])
    winner = list(CATEGORIES)[max(range(18), key=base.__getitem__)]
    if winner in config["protected_domains"] and general_similarity < config["minimum_general_similarity"]:
        return list(base)
    return proposed


@lru_cache(maxsize=1)
def _evidence_bank():
    from .general_task import GeneralTask
    return GeneralTask()._load()[1]


def general_evidence(state):
    """Retrieve evidence from the frozen training bank, excluding exact queries."""
    from sklearn.preprocessing import normalize
    from .eval_data import text_group
    bank = _evidence_bank()
    query = normalize(bank["vectorizer"].transform([state]))
    scores = (bank["matrix"] @ query.T).toarray().ravel()
    group = text_group(state)
    choices = [(float(scores[i]), r["id"]) for i, r in enumerate(bank["records"])
               if r["target"] == "general" and r["group_id"] != group]
    score, identifier = max(choices, default=(0., None))
    return {"general_similarity": min(1., score), "nearest_general_id": identifier}
