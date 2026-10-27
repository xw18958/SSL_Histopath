#!/usr/bin/env python3
"""Run only APPROVED MAE downstream tasks with the original frozen protocols.

Role based, one run per immutable checkpoint/dataset/protocol. No test-set
model selection. Can restart safely: completed outputs are skipped, partial
test markers are NEVER overwritten.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

DATASETS = (
    "mhist", "crc_val_he_7k", "breakhis_8subtype", "kather_2016", "bach",
    "sicapv2_4class", "oral_oscc", "endometrial_4class",
    "gashissdb_binary", "renalcell_6class", "lc25000_5class",
)
EPOCHS = (100, 200)
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

def checked_checkpoint(ckpt_dir: Path, epoch: int):
    path = ckpt_dir / f"epoch_{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError("Missing approved fixed-epoch MAE checkpoint: "+str(path))
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if (raw.get("framework") != "ssl_standard_v1" or raw.get("method_name") != "mae"
            or raw.get("epoch") != epoch or raw["config"]["method"]["name"] != "mae"):
        raise ValueError("Checkpoint framework/method/epoch mismatch")
    return path, raw

def existing_done(output: Path, name: str) -> bool:
    if (output / name).is_file():
        return True
    if (output / "test_started.json").exists():
        raise RuntimeError(f"Refusing to repeat partially executed test: {output}")
    return False

def run_cmd(args: list[str], *, cuda: int=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    env["SSL_MAE_CKPT_ROOT"] = str(Path(env["MAE_CKPT_ROOT"]).resolve())
    env["PYTHONPATH"] = str(ROOT/"src")
    print("RUN", " ".join(map(str,args)),flush=True)
    subprocess.run([PYTHON, *map(str,args)],env=env,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("classification","arch","pathcap","patch-fewshot","segmentation"))
    parser.add_argument("--ckpt-root",type=Path,required=True)
    parser.add_argument("--out-root",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    assert args.ckpt_root.is_dir()
    args.out_root.mkdir(parents=True,exist_ok=True)
    for epoch in EPOCHS:
        checked_checkpoint(args.ckpt_root, epoch)
    os.environ["MAE_CKPT_ROOT"]=str(args.ckpt_root.resolve())
    os.environ["SSL_MAE_CKPT_ROOT"]=str(args.ckpt_root.resolve())

    if args.task in ("classification","arch","pathcap"):
        from pannuke_ssl.ssl_framework.downstream import run_downstream
        from pannuke_ssl.ssl_framework.image_text_retrieval import run_image_text_retrieval
        for epoch in EPOCHS:
            checkpoint,raw=checked_checkpoint(args.ckpt_root,epoch)
            config=copy.deepcopy(raw["config"])
            config["backbone"]["config_dir"]=str(Path(os.environ["SSL_MODEL_ROOT"])/"plip_model")
            config["manifests"]["root"]=str(ROOT/"manifests/ssl_standard")
            if args.task=="classification":
                for ds in DATASETS:
                    out=args.out_root/"classification"/ds/f"epoch_{epoch}"
                    if existing_done(out,"test_metrics.json"):
                        print("SKIP COMPLETED",ds,epoch,flush=True);continue
                    print("CLASSIFICATION",ds,epoch,flush=True)
                    run_downstream(config,checkpoint,out,dataset=ds)
            else:
                ds=args.task
                out=args.out_root/"image_text_retrieval"/ds/f"epoch_{epoch}"
                if existing_done(out,"test_retrieval_metrics.json"):
                    print("SKIP COMPLETED",ds,epoch,flush=True);continue
                print("IMAGE_TEXT_RETRIEVAL",ds,epoch,flush=True)
                run_image_text_retrieval(config,checkpoint,out,dataset=ds)
    elif args.task=="patch-fewshot":
        for epoch in EPOCHS:
            for ds in DATASETS:
                bench=args.out_root/"patch_benchmark_v1"
                result=bench/"results/full/main/mae"/f"epoch_{epoch}"/f"{ds}.json"
                cache=bench/"features/full/main/mae"/f"epoch_{epoch}"/f"{ds}.npz"
                if not result.exists() or not cache.exists():
                    run_cmd([ROOT/"scripts/patch_benchmark_mae.py", "extract-evaluate",
                             "--dataset",ds,"--epoch",str(epoch),
                             "--out",bench,"--variant","main",
                             "--device","cuda:0"],cuda=args.gpu)
                target=args.out_root/"fewshot_linear_v2/full/mae"/f"epoch_{epoch}"/f"{ds}.json"
                if target.exists():
                    print("SKIP COMPLETED FEWSHOT",ds,epoch,flush=True);continue
                run_cmd([ROOT/"scripts/fewshot_linear_gpu_v2.py",
                         "--cache-root",bench/"features",
                         "--out-root",args.out_root/"fewshot_linear_v2",
                         "--method","mae","--epoch",str(epoch),"--dataset",ds,
                         "--gpu","0","--max-iter","200"],cuda=args.gpu)
    elif args.task=="segmentation":
        manifest_dir=Path(os.environ["MAE_SEG_MANIFEST_ROOT"])
        data_root=Path(os.environ["SSL_DATA_ROOT"])
        for epoch in EPOCHS:
            ckpt,_=checked_checkpoint(args.ckpt_root,epoch)
            # Source manifests contain paths such as GlaS/Warwick_QU_Dataset/
            # and EBHI-SEG/...; all are relative to SSL_DATA_ROOT itself.
            for ds in ("glas", "ebhi"):
                root = data_root
                out=args.out_root/"segmentation_semantic"/ds/f"epoch_{epoch}"
                if existing_done(out,"result.json"):
                    print("SKIP COMPLETED SEMANTIC",ds,epoch,flush=True);continue
                run_cmd([ROOT/"scripts/run_pathology_segmentation.py",
                        "--data-root",root,"--manifest",manifest_dir/f"{ds}.json",
                        "--checkpoint",ckpt,"--model-root",os.environ["SSL_MODEL_ROOT"],
                        "--out-dir",out,"--method","mae","--epoch",str(epoch),
                        "--gpu","0"],cuda=args.gpu)
            out=args.out_root/"segmentation_semantic"/"bcss5"/f"epoch_{epoch}"
            if not existing_done(out,"result.json"):
                run_cmd([ROOT/"scripts/run_bcss5_official.py",
                        "--data-root",data_root,
                        "--manifest",manifest_dir/"bcss5.json",
                        "--checkpoint",ckpt,"--model-root",os.environ["SSL_MODEL_ROOT"],
                        "--out-dir",out,"--method","mae","--epoch",str(epoch),
                        "--gpu","0"],cuda=args.gpu)
            out=args.out_root/"glas_instance"/f"epoch_{epoch}"
            if not existing_done(out,"result.json"):
                run_cmd([ROOT/"scripts/run_glas_instance.py",
                        "--data-root",data_root,
                        "--manifest",manifest_dir/"glas.json",
                        "--checkpoint",ckpt,"--model-root",os.environ["SSL_MODEL_ROOT"],
                        "--out-dir",out,"--method","mae","--epoch",str(epoch),
                        "--gpu","0"],cuda=args.gpu)
    print("ROLE FINISHED",args.task,flush=True)

if __name__=="__main__":
    main()
