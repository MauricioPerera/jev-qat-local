"""Descriptive general-vs-rest calibration audit; never selects parameters."""
import numpy as np

from .eval_data import CATEGORIES
from .eval_metrics import scaled_array
from .evaluate import arrays
from .general_learning import read_lines


def audit(run_dir,summary):
    baseline,_=arrays(read_lines(run_dir/"test-v2.jsonl"))
    raw,targets=arrays(read_lines(run_dir/"test-predictions.jsonl"))
    before=scaled_array(baseline,summary["baseline"]["temperature"])
    after=scaled_array(raw,summary["selected_temperature"])
    index=list(CATEGORIES).index("general")
    truth=(targets==index).astype(float)
    def measure(probabilities):
        p=np.clip(probabilities[:,index],1e-15,1-1e-15)
        bins=np.minimum((p*10).astype(int),9)
        ece=sum(float((bins==i).mean())*abs(float(p[bins==i].mean()-truth[bins==i].mean())) for i in range(10) if np.any(bins==i))
        return {"positive_count":int(truth.sum()),"negative_count":int((1-truth).sum()),
                "binary_brier":float(np.mean((p-truth)**2)),"binary_nll":float(-np.mean(truth*np.log(p)+(1-truth)*np.log1p(-p))),
                "ece_10_bins":ece,"positive_class_nll":float(-np.log(p[truth==1]).mean())}
    return {"descriptive_only":True,"parameters_selected":False,"v2":measure(before),"candidate":measure(after)}
