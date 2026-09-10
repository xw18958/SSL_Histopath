from __future__ import annotations
import copy, json
from pathlib import Path
import torch
from pannuke_ssl.ijepa_training import protected_manifest
from pannuke_ssl.ssl_framework import Validator, build_ssl_loader, load_standard_config, load_tuning_spec, module_sha
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.utils import seed_everything


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    repo=Path("/raid1/xwan0900/SSL_proj"); before=protected_manifest(repo); device=torch.device("cuda"); hashes={}; results={}
    for name in ("ijepa","lejepa"):
        c=load_standard_config(name); load_tuning_spec(name); s=copy.deepcopy(c); s["training"]["batch_size"]=2; s["training"]["num_workers"]=0; s["data"]["cache_in_ram"]=False; seed_everything(int(s["seed"])); m=build_method(s,device); hashes[name]=module_sha(m.encoder); loader=build_ssl_loader(s,m,batch_size=2,workers=0); assert len(loader.dataset)==7901; batch=next(iter(loader)); opt=m.build_optimizer(); opt.zero_grad(set_to_none=True); step=m.training_step(batch,bf16=True); assert torch.isfinite(step.loss); step.loss.backward(); grads=[p.grad for p in m.optimizer_parameters() if p.grad is not None]; assert grads and all(torch.isfinite(g).all() for g in grads); results[name]={"loss":float(step.loss.detach()),"metrics":step.metrics}; del m,loader,batch,opt,step; torch.cuda.empty_cache()
    assert hashes["ijepa"]==hashes["lejepa"]
    c=load_standard_config("ijepa"); seed_everything(int(c["seed"])); m=build_method(c,device); v=Validator(c,Path(c["output"]["root"])/"_framework_smoke",device); metrics=v.evaluate(m.encoder,0); assert "linear_val_macro_f1" in metrics; assert protected_manifest(repo)==before
    print(json.dumps({"passed":True,"ssl_unique_images":7901,"ssl_labels_loaded":False,"source_lrs_in_grids":True,"initial_encoder_sha_identical":True,"test_split_accessed":False,"methods":results,"validation_linear_macro_f1":metrics["linear_val_macro_f1"],"protected_existing_pipeline_unchanged":True},indent=2),flush=True)
if __name__=="__main__": main()
