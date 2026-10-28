#!/usr/bin/env python3
"""Compare fixed MAE, LeJEPA, and Simplex-LeJEPA K=8 result JSONs.

Audits exact source splits, retrieval identities, and few-shot support samples.
Never uses test data to select checkpoints or hyperparameters.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark_reports/mae_cvpr_20261027"
SRC=OUT/"source"
METHODS=("lejepa","simplex_k8","mae")
EPOCHS=(100,200)
DATASETS={
 "mhist":"MHIST","crc_val_he_7k":"CRC-HE-7K","breakhis_8subtype":"BreaKHis",
 "kather_2016":"Kather","bach":"BACH","sicapv2_4class":"SICAPv2",
 "oral_oscc":"Oral OSCC","endometrial_4class":"Endometrial",
 "gashissdb_binary":"GaHISSDB","renalcell_6class":"RenalCell",
 "lc25000_5class":"LC25000"}
NAMES={**DATASETS,"arch":"ARCH","pathcap":"PathCap",
 "glas":"GlaS (mean Test A/B)","ebhi":"EBHI-SEG",
 "bcss5":"BCSS (5 classes)","glas_instance":"GlaS instance (mean A/B)"}

def read(p):
 if not p.is_file():raise FileNotFoundError(p)
 return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def same(vals,label):
 if any(v!=vals[0] for v in vals[1:]):raise ValueError("Protocol mismatch: "+label+" = "+repr(vals))
def paths(task,ds,ep):
 out=[]
 for m in METHODS:
  b=SRC/m
  if task=="classification":p=b/"classification"/ds/f"epoch_{ep}/test_metrics.json"
  elif task=="retrieval":p=b/"retrieval"/ds/f"epoch_{ep}/test_retrieval_metrics.json"
  elif task=="patch":p=b/"patch"/f"epoch_{ep}/{ds}.json"
  elif task=="fewshot":p=b/"fewshot"/f"epoch_{ep}/{ds}.json"
  elif task=="semantic":
   if m=="mae":p=b/"semantic"/ds/f"epoch_{ep}/result.json"
   elif ds=="bcss5":p=SRC/"refs_bcss5_raw"/m/f"epoch_{ep}/result.json"
   else:p=SRC/"refs_semantic_raw"/ds/m/f"epoch_{ep}/result.json"
  elif task=="instance":
   p=b/f"instance/epoch_{ep}/result.json" if m=="mae" else SRC/"refs_instance_raw"/m/f"epoch_{ep}/result.json"
  else:raise ValueError(task)
  out.append(p)
 return out

def audit(task,ds,ep,docs):
 tag=f"{task}/{ds}/{ep}"
 if task=="classification":
  for m,x in zip(METHODS,docs):
   assert x["dataset"]==ds and x["encoder_epoch"]==ep and x["test_evaluated_once"]
   assert x["method"]==("simplex_sigreg_lejepa" if m=="simplex_k8" else m)
  for key in ("manifest_sha256","split_counts","preprocessing","class_names","split_policy"):
   same([x["dataset_metadata"][key] for x in docs],tag+key)
  for key in ("test_images","num_classes","class_names","feature_dim"):
   same([x[key] for x in docs],tag+key)
 elif task=="retrieval":
  for x in docs:
   assert x["dataset"]==ds and x["encoder_epoch"]==ep and x["test_evaluated_once_within_protocol"]
  for key in ("manifest_sha256","split_record_sha256","protocol_sha256","evaluation_protocol_version","unique_caption_candidates","image_candidates"):
   same([x[key] for x in docs],tag+key)
 elif task=="patch":
  for x in docs:
   assert x["dataset"]==ds and x["epoch"]==ep and x["split_variant"]=="main"
   assert x["smoke_per_class"]==0 and x["gallery_train_only"] and x["query_test_only"] and x["no_test_selection"]
  for key in ("manifest_sha256","row_order_sha256","train_indices_sha256","test_indices_sha256","support_subsets_sha256","feature_dim"):
   same([x[key] for x in docs],tag+key)
 elif task=="fewshot":
  for x in docs:
   assert x["protocol"]=="fewshot_linear_gpu_v2" and x["mode"]=="full"
   assert x["epoch"]==ep and x["dataset"]==ds and x["encoder_frozen"] and x["repeats"]==100 and x["C"]==1
  for key in ("manifest_sha256","row_order_sha256","selected_support_sha256","n_train","n_test","classes","shots"):
   same([x[key] for x in docs],tag+key)
 elif task in ("semantic","instance"):
  h="source_manifest_sha256" if task=="instance" else "manifest_sha256"
  for x in docs:
   assert x["epoch"]==ep and x["test_used_for_tuning"] is False and x["smoke"] is False
  same([x[h] for x in docs],tag+h)
  fields=("source_splits","eval","representation") if task=="instance" else ("test_images","representation","preprocessing","head","protocol")
  for key in fields:same([x[key] for x in docs],tag+key)

rows=[]
provenance=[]
def add(task,ds,ep,metric,values,source):
 assert len(values)==3 and all(isinstance(v,(int,float)) and 0<=v<=1 for v in values),(task,values)
 score={m:round(v*100,9) for m,v in zip(METHODS,values)}
 rows.append(dict(task=task,dataset=ds,label=NAMES[ds],epoch=ep,metric=metric,**score,
    simplex_minus_lejepa_pp=round(score["simplex_k8"]-score["lejepa"],9),
    simplex_minus_mae_pp=round(score["simplex_k8"]-score["mae"],9)))
 provenance.append(dict(task=task,dataset=ds,epoch=ep,metric=metric,
   inputs={m:dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for m,p in zip(METHODS,source)}))

for ep in EPOCHS:
 for ds in DATASETS:
  for task,metric in (("classification","Accuracy"),("patch","mAP")):
   src=paths(task,ds,ep); docs=[read(p) for p in src];audit(task,ds,ep,docs)
   vals=[x["test"]["accuracy"] for x in docs] if task=="classification" else [x["retrieval"]["mAP"] for x in docs]
   add(task,ds,ep,metric,vals,src)
  src=paths("fewshot",ds,ep);docs=[read(p) for p in src];audit("fewshot",ds,ep,docs)
  for shots in (1,5,10):
   assert all(len(x["results"][str(shots)]["per_subset"]["accuracy"])==100 for x in docs)
   add("fewshot",ds,ep,f"{shots}-shot accuracy",
       [x["results"][str(shots)]["metrics"]["accuracy"]["mean"] for x in docs],src)
 for ds in ("arch","pathcap"):
  src=paths("retrieval",ds,ep);docs=[read(p) for p in src];audit("retrieval",ds,ep,docs)
  add("retrieval",ds,ep,"Mean Recall",[x["test"]["overall_mean_recall"] for x in docs],src)
 for ds in ("glas","ebhi","bcss5"):
  src=paths("semantic",ds,ep);docs=[read(p) for p in src];audit("semantic",ds,ep,docs)
  vals=[(x["test"]["testa"]["miou"]+x["test"]["testb"]["miou"])/2 for x in docs] if ds=="glas" else [x["test"]["miou"] for x in docs]
  add("semantic",ds,ep,"mIoU",vals,src)
 src=paths("instance","glas_instance",ep);docs=[read(p) for p in src];audit("instance","glas_instance",ep,docs)
 add("instance","glas_instance",ep,"Object Dice",
    [(x["testA"]["object_dice"]+x["testB"]["object_dice"])/2 for x in docs],src)
 add("instance_f1","glas_instance",ep,"Object F1",
    [(x["testA"]["object_f1"]+x["testB"]["object_f1"])/2 for x in docs],src)
assert len(rows)==124,len(rows)
with (OUT/"comparison_all_metrics.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/"comparison_audit.json").write_text(json.dumps({
 "methods":METHODS,"epochs":EPOCHS,"metric_rows":len(rows),"entries":provenance},indent=2)+"\n")
groups=[("classification","Accuracy"),("retrieval","Mean Recall"),
 ("patch","mAP"),("fewshot","1-shot accuracy"),("fewshot","5-shot accuracy"),
 ("fewshot","10-shot accuracy"),("semantic","mIoU"),("instance","Object Dice")]
summary=[]
for task,metric in groups:
 for ep in EPOCHS:
  selected=[x for x in rows if x["task"]==task and x["metric"]==metric and x["epoch"]==ep]
  assert selected
  summary.append(dict(task=task,metric=metric,epoch=ep,n_datasets=len(selected),
   **{m:round(sum(r[m] for r in selected)/len(selected),6) for m in METHODS},
   k8_wins_over_mae=sum(r["simplex_k8"]>r["mae"] for r in selected),
   k8_wins_over_lejepa=sum(r["simplex_k8"]>r["lejepa"] for r in selected),
   mae_wins_over_lejepa=sum(r["mae"]>r["lejepa"] for r in selected)))
with (OUT/"comparison_summary.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

def table(task,metric):
 chosen=[x for x in rows if x["task"]==task and x["metric"]==metric]
 order=list(dict.fromkeys(x["dataset"] for x in chosen))
 output=["| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
 for ds in order:
  blocks=[next(x for x in chosen if x["dataset"]==ds and x["epoch"]==ep) for ep in EPOCHS]
  nums=[b[m] for b in blocks for m in METHODS]
  output.append("| "+blocks[0]["label"]+" | "+" | ".join(f"{n:.2f}" for n in nums)+" |")
 return "\n".join(output)

lines=[
 "# MAE vs LeJEPA vs Simplex-LeJEPA (K=8)",
 "",
 "Comparison across matched 100- and 200-epoch checkpoints. All scores are percentages (%), higher is better.",
 "All three methods use the same fresh ViT-B/32 encoder and PanNuke pretraining images.",
 "Exact downstream split, preprocessing, protocol and support subset identities were checked before aggregation.",
 "",
 "## Summary — unweighted mean across datasets",
 "",
 "| Task | Epoch | Datasets | LeJEPA | Simplex K=8 | MAE | K8 > MAE | K8 > LeJEPA |",
 "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for s in summary:
 label=s["task"]+(" "+s["metric"] if s["task"]=="fewshot" else "")
 lines.append(f"| {label} | {s['epoch']} | {s['n_datasets']} | {s['lejepa']:.2f} | {s['simplex_k8']:.2f} | {s['mae']:.2f} | {s['k8_wins_over_mae']}/{s['n_datasets']} | {s['k8_wins_over_lejepa']}/{s['n_datasets']} |")
spec=[
 ("classification","Accuracy","Classification","Frozen linear-probe test accuracy."),
 ("retrieval","Mean Recall","Image-text retrieval","Symmetric mean recall of image-text and text-image retrieval."),
 ("patch","mAP","Patch image-to-image retrieval","Train gallery, test query, mAP on frozen embeddings."),
 ("fewshot","1-shot accuracy","Trainable few-shot, 1 shot","100 matched support sets, logistic regression, accuracy."),
 ("fewshot","5-shot accuracy","Trainable few-shot, 5 shots","100 matched support sets, logistic regression, accuracy."),
 ("fewshot","10-shot accuracy","Trainable few-shot, 10 shots","100 matched support sets, logistic regression, accuracy."),
 ("semantic","mIoU","Semantic segmentation","GlaS reports equal mean of official Test A/B. BCSS uses five official classes."),
 ("instance","Object Dice","GlaS gland instance segmentation","Object Dice is the arithmetic mean of official Test A and B results."),
]
for task,metric,title,note in spec:
 lines+=["","## "+title,"",note,"",table(task,metric)]
lines+=["","## Data and protocol integrity","",
 "- Exactly three models: LeJEPA, Simplex-LeJEPA K=8, MAE. No K=64 or other K enters these tables.",
 "- Classification: matched source manifest SHA, sample counts, preprocessing and classes.",
 "- Retrieval: matched source records, caption IDs, protocol SHA and candidate sets.",
 "- Patch retrieval: matched gallery/query IDs, row order and support subsets.",
 "- Few-shot: same 100 fixed support selections per dataset, shot size and solver protocol.",
 "- Segmentation: same source-manifest hashes and head protocols; five-class BCSS and separate gland-instance evaluation.",
 "- Checkpoints at epochs 100 and 200 were selected before seeing test results. No cross-dataset winner-based filtering.",
 "- All task means weight datasets equally, not individual samples.",
 "- MAE uses the shared ViT-B/32 backbone; this is not a native ViT-B/16 MAE replication.",
 "- Equal epochs and dataset exposure are not equal FLOPs: MAE uses masked one-view reconstruction, JEPA uses eight views.",
 "- Selected MAE learning rate was 0.00015 from three sequential 20-epoch development trials.",
 "",
 "## Reproduce",
 "",
 "Run python3 scripts/aggregate_mae_cvpr_comparison.py from the Git repository root.",
 "comparison_all_metrics.csv: all 124 metric rows, including auxiliary gland object F1.",
 "comparison_summary.csv: all task means and dataset win counts.",
 "comparison_audit.json: source JSON paths and SHA-256 hashes for each metric.",
 "source/: exact small original experiment result JSONs, with no images or weight files.",
 "experiment_metadata/: LR tuning outcome, resolved MAE configuration, 250-epoch training summary, and SHA-256 checkpoint manifest.",
 "MAE checkpoints remain on gpu1-358-0; each is approximately 438 MB, so the four checkpoint weights are not stored in this Git repository.",
]
(OUT/"README.md").write_text("\n".join(lines)+"\n")
print("PASS: generated",len(rows),"metrics and",len(summary),"task summaries")
for s in summary:print(s["task"],s["metric"],s["epoch"],round(s["lejepa"],2),round(s["simplex_k8"],2),round(s["mae"],2),s["k8_wins_over_mae"],"/",s["n_datasets"])
