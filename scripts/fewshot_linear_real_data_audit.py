#!/usr/bin/env python3
"""Independent sklearn audit of GPU few-shot prediction metrics on real caches."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

project=Path("/raid1/xwan0900/SSL_proj")
sys.path.insert(0,str(project/"scripts"))
from fewshot_linear_gpu_v2 import normalize,selected_subsets,read_cache
from fewshot_linear_aggregate_v2 import DATASETS

root=project/"outputs/patch_benchmark_v1/features"
out=project/"outputs/fewshot_linear_gpu_v2"
checks=0
differences=[]
examples=[]
for ds in DATASETS:
 for ep in (100,200):
  for method in ("lejepa","simplex_k8"):
   path=root/"full/main"/method/f"epoch_{ep}"/(ds+".npz")
   meta,data=read_cache(path,method,ep,ds)
   tx,ty,ti=data["train"]
   qx,qy,qi=data["test"]
   cached=json.loads((out/"full"/method/f"epoch_{ep}"/(ds+".json")).read_text())
   tx,qx=normalize(tx),normalize(qx)
   supports=selected_subsets(ds,ti,ty)
   classes=np.sort(np.unique(ty))
   idxmap={int(t):i for i,t in enumerate(ti)}
   for shot in (1,5,10):
    for rep in (0,17,49,99):
     pos=[idxmap[int(x)] for x in supports[str(shot)][rep]]
     X=tx[pos];Y=ty[pos]
     model=LogisticRegression(C=1.0,solver="lbfgs",tol=1e-10,max_iter=3000)
     model.fit(X,Y)
     p=model.predict(qx)
     metrics={"accuracy":accuracy_score(qy,p),
              "balanced_accuracy":balanced_accuracy_score(qy,p),
              "macro_f1":f1_score(qy,p,labels=classes,average="macro",zero_division=0)}
     for m,v in metrics.items():
      original=cached["results"][str(shot)]["per_subset"][m][rep]
      diff=abs(v-original)
      differences.append(diff)
      if diff>0:examples.append([ds,ep,method,shot,rep,m,diff])
     checks+=1
report={"cases":checks,"metric_comparisons":len(differences),
        "exact_metric_matches":sum(float(x)==0 for x in differences),
        "differences_gt_1e-6":sum(x>1e-6 for x in differences),
        "max_absolute_metric_difference":max(differences),
        "first_differences":examples[:25],
        "independent_reference":"sklearn LogisticRegression lbfgs tol=1e-10 C=1.0"}
dest=project/"benchmark_reports/fewshot_linear_gpu_v2/real_data_sklearn_audit.json"
dest.write_text(json.dumps(report,indent=2,default=lambda v: v.item() if hasattr(v,'item') else str(v))+"\n")
print(json.dumps(report,indent=2,default=lambda v: v.item() if hasattr(v,'item') else str(v))[:5000])
