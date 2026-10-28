#!/usr/bin/env python3
"""Audit and aggregate the fixed DINOv3 100/200e comparison; never reruns tests."""
from __future__ import annotations
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--gpu1",type=Path,required=True,help="DINOv3 gpu1 downstream_results JSON snapshot")
    ap.add_argument("--gpu2",type=Path,required=True,help="DINOv3 gpu2 downstream_results JSON snapshot")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    repo=args.repo.resolve()
    base=repo/"benchmark_reports/mae_cvpr_20261027"
    baseline=list(csv.DictReader((base/"comparison_all_metrics.csv").open()))
    audit=json.loads((base/"comparison_audit.json").read_text())
    entries={(x["task"],x["dataset"],int(x["epoch"]),x["metric"]):x for x in audit["entries"]}
    if len(baseline)!=len(entries):raise RuntimeError("Baseline audit/CSV coverage differs")
    expected={"classification":22,"retrieval":4,"patch":22,"fewshot":66,"semantic":6,"instance":2,"instance_f1":2}
    counts=defaultdict(int)
    output=[]; evidence=[]
    checkpoint_hash={}
    for row in baseline:
        task,ds,ep,metric=row["task"],row["dataset"],int(row["epoch"]),row["metric"]
        counts[task]+=1
        inp=entries[task,ds,ep,metric]["inputs"]["mae"]
        original=repo/inp["path"]
        if not original.is_file() or sha(original)!=inp["sha256"]:
            raise RuntimeError("MAE baseline source mismatch "+str(original))
        m=json.loads(original.read_text())
        if task=="classification":
            p=args.gpu1/"classification"/ds/f"epoch_{ep}/test_metrics.json"
        elif task=="retrieval":
            root=args.gpu1 if ds=="arch" else args.gpu2
            p=root/"image_text_retrieval"/ds/f"epoch_{ep}/test_retrieval_metrics.json"
        elif task=="patch":
            p=args.gpu2/"patch_benchmark_v1/results/full/main/dinov3"/f"epoch_{ep}/{ds}.json"
        elif task=="fewshot":
            p=args.gpu2/"fewshot_linear_v2/full/dinov3"/f"epoch_{ep}/{ds}.json"
        elif task=="semantic":
            p=args.gpu2/"segmentation_semantic"/ds/f"epoch_{ep}/result.json"
        elif task in ("instance","instance_f1"):
            p=args.gpu2/"glas_instance"/f"epoch_{ep}/result.json"
        else:raise RuntimeError(task)
        if not p.is_file():raise RuntimeError("Missing DINOv3 result "+str(p))
        d=json.loads(p.read_text())
        if d["method"]!="dinov3" or int(d.get("epoch",d.get("encoder_epoch",-1)))!=ep:
            raise RuntimeError("Incorrect method or epoch: "+str(p))
        ckpt=d.get("checkpoint_sha256",d.get("ckpt_sha256",d.get("ssl_checkpoint_sha256",d.get("feature_checkpoint_sha256"))))
        if ckpt:
            if ep in checkpoint_hash and checkpoint_hash[ep]!=ckpt:
                raise RuntimeError("DINOv3 checkpoint hash inconsistent: "+str(p))
            checkpoint_hash[ep]=ckpt
        if task=="classification":
            if d["dataset_metadata"]["manifest_sha256"]!=m["dataset_metadata"]["manifest_sha256"]:
                raise RuntimeError("Classification manifest mismatch "+str(p))
            if d["test_evaluated_once"] is not True:raise RuntimeError("Classification test reused")
            value=d["test"]["accuracy"]*100
        elif task=="retrieval":
            for k in ("manifest_sha256","protocol_sha256","split_record_sha256","input_image_files_sha256"):
                if d[k]!=m[k]:raise RuntimeError("Retrieval "+k+" mismatch "+str(p))
            if not d["test_evaluated_once_within_protocol"] or d["selection"]["test_used"] is not False:
                raise RuntimeError("Retrieval test isolation failed")
            value=d["test"]["overall_mean_recall"]*100
        elif task=="patch":
            for k in ("manifest_sha256","row_order_sha256","support_subsets_sha256",
                      "train_indices_sha256","test_indices_sha256"):
                if d[k]!=m[k]:raise RuntimeError("Patch "+k+" mismatch "+str(p))
            if not (d["gallery_train_only"] and d["query_test_only"] and d["no_test_selection"]):
                raise RuntimeError("Patch test isolation failed")
            if int(d.get("smoke_per_class",1))!=0:raise RuntimeError("Patch smoke results not full")
            value=d["retrieval"]["mAP"]*100
        elif task=="fewshot":
            for k in ("manifest_sha256","row_order_sha256","selected_support_sha256"):
                if d[k]!=m[k]:raise RuntimeError("Fewshot "+k+" mismatch "+str(p))
            if not (d["encoder_frozen"] and int(d["repeats"])==100):
                raise RuntimeError("Fewshot protocol mismatch")
            shot=metric.split("-")[0]
            if int(shot) not in (1,5,10):raise RuntimeError("Unexpected shots "+metric)
            value=d["results"][shot]["metrics"]["accuracy"]["mean"]*100
        elif task=="semantic":
            if d["manifest_sha256"]!=m["manifest_sha256"] or d["test_used_for_tuning"] is not False:
                raise RuntimeError("Semantic data or test isolation mismatch "+str(p))
            if ds=="glas":
                value=100*statistics.mean([d["test"]["testa"]["miou"],d["test"]["testb"]["miou"]])
            else:value=100*d["test"]["miou"]
        else:
            if d["source_manifest_sha256"]!=m["source_manifest_sha256"] or d["test_used_for_tuning"] is not False:
                raise RuntimeError("Instance data or test isolation mismatch "+str(p))
            key="object_dice" if task=="instance" else "object_f1"
            value=100*statistics.mean([d["testA"][key],d["testB"][key]])
        item={k:row[k] for k in ("task","dataset","label","epoch","metric","lejepa","simplex_k8","mae")}
        item.update(dinov3=f"{value:.9f}",dinov3_minus_simplex_pp=f"{value-float(row['simplex_k8']):.9f}",
                    dinov3_minus_mae_pp=f"{value-float(row['mae']):.9f}")
        output.append(item)
        evidence.append({"task":task,"dataset":ds,"epoch":ep,"metric":metric,
                         "dinov3_source":str(p.relative_to(args.gpu1 if str(p).startswith(str(args.gpu1)) else args.gpu2)),
                         "source_server":"gpu1-358-0" if str(p).startswith(str(args.gpu1)) else "gpu2-358-0",
                         "dinov3_sha256":sha(p),
                         "baseline_mae_sha256":inp["sha256"],
                         "checkpoint_sha256":ckpt})
    if dict(counts)!=expected:raise RuntimeError(f"Comparison metric coverage differs: {dict(counts)}")
    if set(checkpoint_hash)!={100,200}:raise RuntimeError("Missing checkpoint hashes")
    groups=defaultdict(list)
    for r in output:
        if r["task"]=="instance_f1":continue
        groups[(r["task"],int(r["epoch"]))].append(r)
    summary=[]
    for (task,epoch),rows in sorted(groups.items()):
        summary.append({"task":task,"epoch":epoch,"datasets":len(rows),
                        **{method:round(statistics.mean(float(x[method]) for x in rows),5)
                           for method in ("lejepa","simplex_k8","mae","dinov3")},
                        **{f"dinov3_wins_vs_{method}":sum(float(x["dinov3"])>float(x[method]) for x in rows)
                           for method in ("lejepa","simplex_k8","mae")}})
    args.out.mkdir(parents=True,exist_ok=True)
    def csv_write(name,rows):
        with (args.out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    csv_write("comparison_all_metrics.csv",output)
    csv_write("comparison_summary.csv",summary)
    (args.out/"comparison_audit.json").write_text(json.dumps({
        "status":"PASS","rows":len(output),"metric_counts":dict(counts),
        "checkpoint_sha256":checkpoint_hash,"checks":["MAE baseline SHA256","source manifest identity",
        "retrieval protocol and split signatures","patch/fewshot row and support identities",
        "test-isolation flags","GlaS official Test A/B arithmetic mean","fixed epoch and checkpoint identity"],
        "entries":evidence},indent=2)+"\n")
    print("AUDIT PASS",len(output),"rows",dict(counts),"checkpoints",checkpoint_hash)
    for s in summary:print(s)
if __name__=="__main__":main()
