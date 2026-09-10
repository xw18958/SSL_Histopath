from __future__ import annotations
import math
from typing import Any
import torch
from torch.utils.data import Dataset
from pannuke_ssl.ijepa import objective
from pannuke_ssl.ijepa_lejepa_fairness_source import FairnessDataset, IJEPAFairPredictor
from pannuke_ssl.ijepa_lejepa_fairness_source_fixed import IJEPAOfficialMaskSampler
from pannuke_ssl.models import FreshPLIPVisionEncoder, make_teacher, update_ema

class Step:
    def __init__(self,loss,metrics): self.loss,self.metrics=loss,metrics

class StandardIJEPA(torch.nn.Module):
    def __init__(self,c:dict[str,Any],device:torch.device):
        super().__init__(); self.config,self.device=c,device; m=c["method"]
        self.compat={"seed":c["seed"],"ijepa":{"image_size":256,"patch_size":32,"hidden_size":768,"crop_scale":list(m["views"]["crop_scale"]),"predictor_dim":384,"predictor_depth":4,"predictor_heads":12,"predictor_mlp_ratio":4,"target_count":4,"context_scale":list(m["mask"]["context_scale"]),"target_scale":list(m["mask"]["target_scale"]),"target_aspect":list(m["mask"]["target_aspect"]),"minimum_context_tokens":10,"minimum_block_tokens":4,"target_target_overlap":True,"context_target_overlap":False,"ema_momentum":0.996}}
        self.student=FreshPLIPVisionEncoder(c["backbone"]["config_dir"],256).to(device); heads=int(self.student.model.config.num_attention_heads)
        if (self.student.num_patches,self.student.hidden_size,self.student.patch_size,heads)!=(64,768,32,12): raise RuntimeError("Unexpected shared encoder")
        self.teacher=make_teacher(self.student); self.predictor=IJEPAFairPredictor(num_heads=12).to(device); self.masker=IJEPAOfficialMaskSampler(self.compat,int(c["seed"]))
    @property
    def encoder(self): return self.teacher
    def wrap_dataset(self,base:Dataset): return FairnessDataset(base,"ijepa_4layer",self.compat)
    def train_mode(self): self.student.train(); self.predictor.train(); self.teacher.eval()
    def optimizer_parameters(self): return [p for x in (self.student,self.predictor) for p in x.parameters() if p.requires_grad]
    def batch_size(self,batch): return int(batch["image"].shape[0])
    def training_step(self,batch,*,bf16:bool):
        images=batch["image"].to(self.device,dtype=torch.float32,non_blocking=True).div_(255.); context,targets=self.masker.sample(images.shape[0],self.device)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16): loss,_,_=objective(self.student,self.teacher,self.predictor,images,context,targets)
        return Step(loss,{"context_tokens":float(context.shape[1]),"target_tokens_per_block":float(targets.shape[2])})
    def build_optimizer(self):
        decay,no=[] ,[]
        for mod in (self.student,self.predictor):
            for name,p in mod.named_parameters():
                if p.requires_grad: (no if ("bias" in name or p.ndim==1) else decay).append(p)
        o=self.config["method"]["optimizer"]
        return torch.optim.AdamW([{"params":decay,"weight_decay":float(o["weight_decay"])},{"params":no,"weight_decay":0.,"_no_weight_decay":True}],lr=float(o["peak_lr"]))
    def schedule(self,step,total):
        o=self.config["method"]["optimizer"]; p=min(max(step/max(1,total-1),0.),1.); warm=float(o["warmup_fraction"]); peak=float(o["peak_lr"]); start=peak*float(o["start_lr_ratio"]); final=peak*float(o["final_lr_ratio"])
        lr=start+(peak-start)*(p/warm) if p<warm else final+.5*(peak-final)*(1+math.cos(math.pi*(p-warm)/(1-warm))); wd0,wd1=float(o["weight_decay"]),float(o["final_weight_decay"]); wd=wd1-.5*(wd1-wd0)*(1+math.cos(math.pi*p)); return {"lr":lr,"weight_decay":wd}
    def after_optimizer_step(self,step,total):
        e=self.config["method"]["ema"]; p=min(max(step/max(1,total-1),0.),1.); start,end=float(e["start"]),float(e["end"]); m=end-.5*(end-start)*(1+math.cos(math.pi*p)); update_ema(self.student,self.teacher,m); return {"ema_momentum":m}
