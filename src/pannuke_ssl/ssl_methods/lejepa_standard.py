from __future__ import annotations
import math
from typing import Any
import torch
from torch.utils.data import Dataset
from pannuke_ssl.ijepa_lejepa_fairness_source import FairnessDataset, LeJEPAFair
from pannuke_ssl.models import FreshPLIPVisionEncoder

class Step:
    def __init__(self,loss,metrics): self.loss,self.metrics=loss,metrics

class StandardLeJEPA(torch.nn.Module):
    def __init__(self,c:dict[str,Any],device:torch.device):
        super().__init__(); self.config,self.device=c,device; m=c["method"]; v,a,o=m["views"],m["augmentations"],m["objective"]
        self.compat={"seed":c["seed"],"lejepa":{"global_views":2,"local_views":[4,6],"global_size":int(v["global_size"]),"local_size":int(v["local_size"]),"global_scale":list(v["global_scale"]),"local_scale":list(v["local_scale"]),"lambda":float(o["sigreg_lambda"]),"projector_dim":int(m["projector"]["dim"]),"projector_hidden_dim":int(m["projector"]["hidden_dim"]),"sigreg_slices":int(o["sigreg_slices"]),"sigreg_points":int(o["sigreg_points"]),"sigreg_t_max":float(o["sigreg_t_max"]),"horizontal_flip_p":float(a["horizontal_flip_p"]),"color_jitter_p":float(a["color_jitter_p"]),"grayscale_p":float(a["grayscale_p"]),"gaussian_blur_p":float(a["gaussian_blur_p"]),"solarize_p":float(a["solarize_p"])}}
        base=FreshPLIPVisionEncoder(c["backbone"]["config_dir"],256).to(device)
        if (base.num_patches,base.hidden_size,base.patch_size)!=(64,768,32): raise RuntimeError("Unexpected shared encoder")
        self.model=LeJEPAFair(self.compat,base).to(device)
    @property
    def encoder(self): return self.model.encoder.base
    def wrap_dataset(self,base:Dataset): return FairnessDataset(base,"lejepa_2g6l",self.compat)
    def train_mode(self): self.model.train()
    def optimizer_parameters(self): return [p for p in self.model.parameters() if p.requires_grad]
    def batch_size(self,batch): return int(batch["global_views"].shape[0])
    def training_step(self,batch,*,bf16:bool):
        g=batch["global_views"].to(self.device,non_blocking=True); l=batch["local_views"].to(self.device,non_blocking=True); gs=[g[:,i] for i in range(g.shape[1])]; ls=[l[:,i] for i in range(l.shape[1])]
        if len(gs)!=2 or len(ls)!=6: raise RuntimeError("Standard LeJEPA must remain 2G+6L")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16): loss,inv,sig,_=self.model(gs,ls)
        return Step(loss,{"invariance_loss":float(inv.detach()),"sigreg_loss":float(sig.detach()),"logical_views":8.0})
    def build_optimizer(self):
        o=self.config["method"]["optimizer"]; return torch.optim.AdamW(self.model.parameters(),lr=float(o["peak_lr"]),weight_decay=float(o["weight_decay"]),betas=(.9,.999))
    def schedule(self,step,total):
        o=self.config["method"]["optimizer"]; p=min(max(step/max(1,total-1),0.),1.); warm=float(o["warmup_fraction"]); peak=float(o["peak_lr"]); final=peak*float(o["final_lr_ratio"]); lr=peak*(p/warm) if p<warm else final+.5*(peak-final)*(1+math.cos(math.pi*(p-warm)/(1-warm))); return {"lr":lr,"weight_decay":float(o["weight_decay"])}
    def after_optimizer_step(self,step,total): return {}
