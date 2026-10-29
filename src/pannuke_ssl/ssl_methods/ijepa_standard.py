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
        mask, pred, backbone = m["mask"], m["predictor"], c["backbone"]
        self.compat = {"seed": c["seed"], "ijepa": {
            "image_size": int(backbone["image_size"]), "patch_size": int(backbone["patch_size"]),
            "hidden_size": int(backbone["hidden_size"]), "crop_scale": list(m["views"]["crop_scale"]),
            "predictor_dim": int(pred["dim"]), "predictor_depth": int(pred["depth"]),
            "predictor_heads": int(pred["heads"]), "predictor_mlp_ratio": float(pred["mlp_ratio"]),
            "target_count": int(mask["target_count"]), "context_scale": list(mask["context_scale"]),
            "target_scale": list(mask["target_scale"]), "target_aspect": list(mask["target_aspect"]),
            "minimum_context_tokens": int(mask["minimum_context_tokens"]),
            "minimum_block_tokens": int(mask["minimum_block_tokens"]),
            "target_target_overlap": bool(mask["target_target_overlap"]),
            "context_target_overlap": bool(mask["context_target_overlap"]),
            "ema_momentum": float(m["ema"]["start"]),
        }}
        self.student = FreshPLIPVisionEncoder(backbone["config_dir"], int(backbone["image_size"])).to(device)
        heads = int(self.student.model.config.num_attention_heads)
        if (self.student.num_patches, self.student.hidden_size, self.student.patch_size, heads) != (64, 768, 32, 12):
            raise RuntimeError("Unexpected shared encoder")
        if int(pred["heads"]) != heads or int(mask["context_count"]) != 1 or not mask["target_target_overlap"] or mask["context_target_overlap"]:
            raise ValueError("Unsupported I-JEPA mask or predictor architecture for the approved comparison")
        self.teacher = make_teacher(self.student)
        self.predictor = IJEPAFairPredictor(
            num_heads=int(pred["heads"]), predictor_dim=int(pred["dim"]),
            predictor_depth=int(pred["depth"]), mlp_ratio=float(pred["mlp_ratio"]),
        ).to(device)
        self.masker = IJEPAOfficialMaskSampler(self.compat, int(c["seed"]))
    @property
    def encoder(self): return self.teacher
    def wrap_dataset(self,base:Dataset): return FairnessDataset(base,"ijepa_4layer",self.compat)
    def train_mode(self): self.student.train(); self.predictor.train(); self.teacher.eval()
    def optimizer_parameters(self): return [p for x in (self.student,self.predictor) for p in x.parameters() if p.requires_grad]
    def batch_size(self,batch): return int(batch["image"].shape[0])
    def training_step(self,batch,*,bf16:bool):
        images=batch["image"].to(self.device,dtype=torch.float32,non_blocking=True).div_(255.); context,targets=self.masker.sample(images.shape[0],self.device)
        if (context[:, None, :, None] == targets[:, :, None, :]).any().item():
            raise RuntimeError("I-JEPA context/target overlap would leak masked target pixels")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16): loss,_,_=objective(self.student,self.teacher,self.predictor,images,context,targets)
        return Step(loss,{"context_tokens":float(context.shape[1]),"target_tokens_per_block":float(targets.shape[2]),"context_target_overlap":0.0})
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
        e=self.config["method"]["ema"]; p=min(max(step/max(1,total-1),0.),1.); start,end=float(e["start"]),float(e["end"]); m=start+(end-start)*p; update_ema(self.student,self.teacher,m); return {"ema_momentum":m}
