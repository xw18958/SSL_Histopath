#!/usr/bin/env python3
"""Verify and aggregate LeJEPA versus K=8 GPU few-shot linear-probe results.

Requires all 44 jobs: 11 datasets x 2 epochs x 2 models; each shot has
100 identically indexed training support subsets shared across methods.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

DATASETS=("bach","breakhis_8subtype","crc_val_he_7k","endometrial_4class",
          "gashissdb_binary","kather_2016","lc25000_5class","mhist",
          "oral_oscc","renalcell_6class","sicapv2_4class")
METRICS=("accuracy","balanced_accuracy","macro_f1")
METHODS=("lejepa","simplex_k8")
EPOCHS=(100,200)
SHOTS=(1,5,10)


def ci(a):
    x=np.asarray(a,dtype=np.float64)
    mean=float(x.mean())
    sd=float(x.std(ddof=1))
    radius=float(student_t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return mean,sd,mean-radius,mean+radius


def audit_and_aggregate(root,out):
    saved={}
    problems=[]
    for ds in DATASETS:
        for epoch in EPOCHS:
            for method in METHODS:
                p=root/"full"/method/f"epoch_{epoch}"/(ds+".json")
                if not p.exists():
                    problems.append("MISSING "+str(p));continue
                r=json.loads(p.read_text())
                expected={"protocol":"fewshot_linear_gpu_v1","mode":"full",
                          "method":method,"epoch":epoch,"dataset":ds,
                          "repeats":100,"encoder_frozen":True,"C":1.0}
                for k,v in expected.items():
                    if r.get(k)!=v:problems.append(f"{p}: {k} != {v}")
                if r.get("shots")!=[1,5,10]:problems.append(f"{p}: shots mismatch")
                if len(r.get("results",{}))!=3:problems.append(f"{p}: missing shots")
                for shot in SHOTS:
                    z=r.get("results",{}).get(str(shot),{})
                    for metric in METRICS:
                        values=z.get("per_subset",{}).get(metric,[])
                        if len(values)!=100:problems.append(f"{p}: {shot} {metric} != 100")
                        if len(values)==100 and (not np.all(np.isfinite(values)) or
                                                   np.min(values)<0 or np.max(values)>1):
                            problems.append(f"{p}: invalid scores")
                        if len(values)==100:
                            reported=z["metrics"][metric]
                            if abs(float(reported["mean"])-float(np.mean(values)))>1e-9:
                                problems.append(f"{p}: means disagree")
                    if not (0<=z.get("cpu_sklearn_rescue_count",-1)<=100):
                        problems.append(f"{p}: missing solver rescue audit")
                saved[(ds,epoch,method)]=r
            a=saved.get((ds,epoch,"lejepa"))
            b=saved.get((ds,epoch,"simplex_k8"))
            if a is None or b is None:continue
            for k in ("manifest_sha256","row_order_sha256","selected_support_sha256",
                      "n_train","n_test","classes"):
                if a.get(k)!=b.get(k):
                    problems.append(f"{ds}/e{epoch}: LeJEPA/Simplex {k} differs")
            for method in METHODS:
                current=saved.get((ds,epoch,method))
                earlier=saved.get((ds,100,method))
                if current and earlier and current["selected_support_sha256"]!=earlier["selected_support_sha256"]:
                    problems.append(f"{ds} {method}: supports differ across epochs")
    if problems:
        raise RuntimeError("AUDIT FAILED:\n"+"\n".join(problems[:50]))
    if len(saved)!=44:
        raise RuntimeError(f"Expected 44 jobs, got {len(saved)}")
    rows=[]
    aggregates=[]
    for epoch in EPOCHS:
        for shot in SHOTS:
            for metric in METRICS:
                deltas=[]
                wins=0
                for ds in DATASETS:
                    a=saved[(ds,epoch,"lejepa")]["results"][str(shot)]["per_subset"][metric]
                    b=saved[(ds,epoch,"simplex_k8")]["results"][str(shot)]["per_subset"][metric]
                    difference=(np.asarray(b)-np.asarray(a))*100
                    dmean,dsd,low,high=ci(difference)
                    deltas.append(dmean)
                    wins+=dmean>0
                    rows.append(dict(dataset=ds,epoch=epoch,shot=shot,metric=metric,
                                    lejepa_pct=100*np.mean(a),simplex_k8_pct=100*np.mean(b),
                                    difference_pp=dmean,paired_resample_sd_pp=dsd,
                                    paired_resample_ci95_low_pp=low,paired_resample_ci95_high_pp=high,
                                    sample_count=100))
                avg,sd,lo,hi=ci(deltas)
                aggregates.append(dict(epoch=epoch,shot=shot,metric=metric,
                                       dataset_count=len(DATASETS),simplex_wins=wins,
                                       lejepa_avg_pct=np.mean([r["lejepa_pct"] for r in rows
                                           if r["epoch"]==epoch and r["shot"]==shot and r["metric"]==metric]),
                                       simplex_k8_avg_pct=np.mean([r["simplex_k8_pct"] for r in rows
                                           if r["epoch"]==epoch and r["shot"]==shot and r["metric"]==metric]),
                                       difference_pp=avg,dataset_level_sd_pp=sd,
                                       descriptive_dataset_ci95_low_pp=lo,
                                       descriptive_dataset_ci95_high_pp=hi))
    rescue_total=sum(saved[key]["results"][str(k)]["cpu_sklearn_rescue_count"]
                     for key in saved for k in SHOTS)
    report={"protocol":"fewshot_linear_gpu_v1","status":"complete",
            "methods":list(METHODS),"epochs":list(EPOCHS),"datasets":list(DATASETS),
            "complete_jobs":len(saved),"trained_classifiers":44*3*100,
            "gpu_batched_fits":44*3*100-rescue_total,
            "cpu_sklearn_rescued_fits":rescue_total,
            "fairness_checks":"manifest, sample-order and support IDs matched; saved scores validated",
            "uncertainty_note":"Support-set CIs describe support-resampling variability on a fixed test set; dataset-level CIs are descriptive, not independent-patient inference.",
            "summary":aggregates}
    out.mkdir(parents=True,exist_ok=True)
    for name,data in (("per_dataset.csv",rows),("summary.csv",aggregates)):
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,data[0].keys());writer.writeheader();writer.writerows(data)
    (out/"verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for r in aggregates:
        if r["metric"]=="accuracy":
            print(f"epoch={r['epoch']} shot={r['shot']} baseline={r['lejepa_avg_pct']:.3f}% "
                  f"simplex={r['simplex_k8_avg_pct']:.3f}% delta={r['difference_pp']:+.3f}pp "
                  f"wins={r['simplex_wins']}/{r['dataset_count']}")
    print("VERIFIED",len(saved),"jobs",44*3*100,"fits",rescue_total,"CPU rescue fits")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    audit_and_aggregate(args.root,args.out)
