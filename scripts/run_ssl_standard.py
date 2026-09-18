from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from pannuke_ssl.ssl_framework import apply_lr, apply_tuned_hyperparameters, load_standard_config, run_downstream, run_tuning, train_ssl, write_final_report


METHODS=("ijepa","lejepa","simplex_sigreg_lejepa","dinov3")


def _apply_saved_tuning(c,root:Path,ignore_tuned:bool):
    selected=root/"tuning/best_hyperparameters.yaml"
    if selected.exists() and not ignore_tuned:
        values=yaml.safe_load(selected.read_text())["selected"]
        return apply_tuned_hyperparameters(c,values)
    return c


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("tune","pretrain","downstream","pipeline","report")); p.add_argument("--method",required=True,choices=METHODS); p.add_argument("--learning-rate",type=float,default=None); p.add_argument("--ignore-tuned",action="store_true"); a=p.parse_args()
    c=load_standard_config(a.method); root=Path(c["output"]["root"])/a.method
    if a.action=="tune": result=run_tuning(c)
    elif a.action=="pretrain":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        if a.learning_rate is not None: c=apply_lr(c,a.learning_rate)
        result=train_ssl(c,root/"pretrain")
    elif a.action=="downstream":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=run_downstream(c,root/"pretrain/checkpoints/best.pt",root/"downstream"); write_final_report(c)
    elif a.action=="report":
        c=_apply_saved_tuning(c,root,a.ignore_tuned); result={"report":str(write_final_report(c))}
    else:
        tune=run_tuning(c); c=apply_tuned_hyperparameters(c,tune["selected_parameters"]); pre=train_ssl(c,root/"pretrain"); down=run_downstream(c,root/"pretrain/checkpoints/best.pt",root/"downstream"); report=write_final_report(c); result={"tuning":tune,"pretrain":pre,"downstream":down,"report":str(report)}
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
