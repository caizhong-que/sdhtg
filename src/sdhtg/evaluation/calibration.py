from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def calibrate_threshold(labels, scores, objective="f1", grid_size=2001, minimum_recall=None) -> dict:
    y=np.asarray(labels,dtype=int); s=np.asarray(scores,dtype=float)
    if len(y)!=len(s) or len(y)==0 or not np.isfinite(s).all(): raise ValueError("invalid calibration arrays")
    candidates=np.unique(np.quantile(s,np.linspace(0,1,grid_size)))
    best=None
    for threshold in candidates:
        prediction=s>=threshold; tp=((prediction==1)&(y==1)).sum(); fn=((prediction==0)&(y==1)).sum()
        recall=tp/max(tp+fn,1)
        if minimum_recall is not None and recall<minimum_recall: continue
        value=f1_score(y,prediction,zero_division=0) if objective=="f1" else recall
        candidate=(float(value),-float(threshold),float(recall))
        if best is None or candidate>best[0]: best=(candidate,float(threshold))
    if best is None: raise ValueError("no threshold satisfies calibration constraints")
    return {"threshold":best[1],"objective":objective,"validation_value":best[0][0],"recall":best[0][2]}
