from __future__ import annotations
import hashlib, random
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader
from pannuke_ssl.data import PanNukeImageDataset, all_source_rows, loader_kwargs
from pannuke_ssl.monitor import _classification_metrics,_fixed_linear_predictions,feature_diagnostics,weighted_knn_predictions
from pannuke_ssl.parquet import build_source_index,preload_images,read_metadata,verify_records
from pannuke_ssl.utils import write_csv

def module_sha(module:torch.nn.Module)->str:
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        t=value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update((t.view(torch.uint16) if t.dtype==torch.bfloat16 else t).numpy().tobytes())
    return h.hexdigest()

def _base_dataset(c):
    index=build_source_index(Path(c["data"]["root"])); rows=all_source_rows(index); expected=int(c["data"]["expected_ssl_images"])
    if len(rows)!=expected or len({(r["fold"],r["sample_index"]) for r in rows})!=expected: raise ValueError(f"Need exactly {expected} unique SSL sources")
    cache=preload_images(rows,index) if c["data"]["cache_in_ram"] else None
    return PanNukeImageDataset(rows,index,cache,include_label=False,include_key=True)

def build_ssl_loader(c,method,*,batch_size=None,workers=None):
    def seed_worker(_):
        s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s)
    dataset=method.wrap_dataset(_base_dataset(c)); w=int(c["training"]["num_workers"] if workers is None else workers)
    kw=dict(batch_size=int(batch_size or c["training"]["batch_size"]),shuffle=True,num_workers=w,pin_memory=torch.cuda.is_available(),persistent_workers=w>0,drop_last=False,generator=torch.Generator().manual_seed(int(c["seed"])),worker_init_fn=seed_worker)
    if w>0: kw["prefetch_factor"]=2
    return DataLoader(dataset,**kw)

class Validator:
    """Train/validation-only frozen-feature monitor; test images are never decoded."""
    def __init__(self,c:dict[str,Any],out:Path,device:torch.device):
        self.c,self.out,self.device,self.history=c,Path(out),device,[]; rows=read_metadata(Path(c["data"]["metadata_csv"])); expected={"train":int(c["downstream"]["train_count"]),"val":int(c["downstream"]["validation_count"]),"test":int(c["downstream"]["test_count"])}
        if dict(Counter(r["split"] for r in rows))!=expected: raise ValueError("Downstream split sizes changed")
        per=Counter((r["split"],int(r["class_id"])) for r in rows)
        for split,total in expected.items():
            if total%19: raise ValueError("Balanced split size must be divisible by 19")
            n=total//19
            if [per[(split,i)] for i in range(19)]!=[n]*19: raise ValueError("Downstream split not balanced")
        selected=[r for r in rows if r["split"] in ("train","val")]; index=build_source_index(Path(c["data"]["root"])); verify_records(rows,index); cache=preload_images(selected,index)
        self.loaders={s:DataLoader(PanNukeImageDataset([r for r in selected if r["split"]==s],index,cache,include_label=True),**loader_kwargs(int(c["validation"]["batch_size"]),int(c["validation"]["num_workers"]),shuffle=False)) for s in ("train","val")}
    @torch.inference_mode()
    def features(self,encoder,split):
        xs,ys=[],[]
        for images,y in self.loaders[split]:
            x=images.to(self.device,dtype=torch.float32,non_blocking=True).div_(255.)
            with torch.autocast("cuda",dtype=torch.bfloat16): tokens=encoder(x)
            if tokens.ndim!=3: raise RuntimeError("Encoder must return [B,T,D]")
            xs.append(tokens.float().mean(1).cpu()); ys.append(y.long().cpu())
        return torch.cat(xs),torch.cat(ys)
    def evaluate(self,encoder,epoch):
        was=encoder.training; encoder.eval(); tx,ty=self.features(encoder,"train"); vx,vy=self.features(encoder,"val"); vc=self.c["validation"]
        knn=weighted_knn_predictions(tx,ty,vx,classes=19,k=int(vc["knn_k"]),temperature=float(vc["knn_temperature"])); lin=_fixed_linear_predictions(tx,ty,vx,classes=19,seed=int(self.c["seed"]),max_iter=int(vc["linear_lbfgs_max_iter"]),device=self.device)
        row={"epoch":float(epoch)}; row.update({f"knn_val_{k}":v for k,v in _classification_metrics(knn,vy).items()}); row.update({f"linear_val_{k}":v for k,v in _classification_metrics(lin,vy).items()}); row.update(feature_diagnostics(torch.cat((tx,vx)).to(self.device),near_constant_std_threshold=float(vc["near_constant_std_threshold"])))
        self.history.append(row); write_csv(self.history,self.out/"validation_metrics.csv")
        if was: encoder.train()
        return row
