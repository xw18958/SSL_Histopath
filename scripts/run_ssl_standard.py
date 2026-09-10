from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from pannuke_ssl.ssl_framework import apply_lr, load_standard_config, run_downstream, run_tuning, train_ssl


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("tune","pretrain","downstream","pipeline")); p.add_argument("--method",required=True,choices=("ijepa","lejepa")); p.add_argument("--learning-rate",type=float,default=None); p.add_argument("--ignore-tuned",action="store_true"); a=p.parse_args()
    c=load_standard_config(a.method); root=Path(c["output"]["root"])/a.method
    if a.action=="tune": result=run_tuning(c)
    elif a.action=="pretrain":
        lr=a.learning_rate; selected=root/"tuning/best_hyperparameters.yaml"
        if lr is None and selected.exists() and not a.ignore_tuned: lr=float(yaml.safe_load(selected.read_text())["selected"]["learning_rate"])
        if lr is not None: c=apply_lr(c,lr)
        result=train_ssl(c,root/"pretrain")
    elif a.action=="downstream": result=run_downstream(c,root/"pretrain/checkpoints/best.pt",root/"downstream")
    else:
        tune=run_tuning(c); c=apply_lr(c,float(tune["selected_value"])); pre=train_ssl(c,root/"pretrain"); down=run_downstream(c,root/"pretrain/checkpoints/best.pt",root/"downstream"); result={"tuning":tune,"pretrain":pre,"downstream":down}
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
