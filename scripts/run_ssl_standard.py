from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from pannuke_ssl.ssl_framework import apply_lr, apply_tuned_hyperparameters, load_standard_config, run_downstream, run_tuning, train_ssl, write_final_report
from pannuke_ssl.ssl_framework.external_datasets import EXTERNAL_DATASETS, PANNUKE_DATASET


METHODS=("ijepa","lejepa","simplex_sigreg_lejepa","dinov3")


def _apply_saved_tuning(c,root:Path,ignore_tuned:bool):
    selected=root/"tuning/best_hyperparameters.yaml"
    if selected.exists() and not ignore_tuned:
        values=yaml.safe_load(selected.read_text())["selected"]
        return apply_tuned_hyperparameters(c,values)
    return c


def _validate_action_dataset(action: str, dataset: str) -> None:
    if dataset != PANNUKE_DATASET and action != "downstream":
        raise ValueError(
            f"Optional dataset {dataset!r} is downstream-only; tune/pretrain/pipeline/report remain fixed to {PANNUKE_DATASET}"
        )
    if action == "downstream" and dataset == PANNUKE_DATASET:
        raise ValueError("PanNuke is development/pretraining-only; final downstream evaluation must use an external dataset")

def _downstream_output_path(root: Path, dataset: str) -> Path:
    if dataset == PANNUKE_DATASET:
        return root / "downstream"
    if dataset not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown downstream dataset {dataset!r}")
    return root / "downstream_datasets" / dataset


def _run_downstream_checkpoints(c, root: Path, dataset: str):
    epochs = [int(epoch) for epoch in c["training"]["checkpoint_epochs"]]
    base = _downstream_output_path(root, dataset)
    results = {}
    for epoch in epochs:
        checkpoint = root / "pretrain_full" / "checkpoints" / f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing required SSL checkpoint: {checkpoint}")
        output = base / f"epoch_{epoch}"
        results[str(epoch)] = run_downstream(c, checkpoint, output, dataset=dataset)
    return {"dataset": dataset, "checkpoint_epochs": epochs, "results": results}


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("tune","pretrain","downstream","pipeline","report")); p.add_argument("--method",required=True,choices=METHODS); p.add_argument("--dataset",choices=(PANNUKE_DATASET,*EXTERNAL_DATASETS),default=PANNUKE_DATASET); p.add_argument("--learning-rate",type=float,default=None); p.add_argument("--ignore-tuned",action="store_true"); a=p.parse_args()
    try: _validate_action_dataset(a.action,a.dataset)
    except ValueError as error: p.error(str(error))
    c=load_standard_config(a.method); root=Path(c["output"]["root"])/a.method
    if a.action=="tune": result=run_tuning(c)
    elif a.action=="pretrain":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        if a.learning_rate is not None: c=apply_lr(c,a.learning_rate)
        result=train_ssl(c,root/"pretrain_full",use_all_data=True,validate=False,early_stop=False,checkpoint_epochs=c["training"]["checkpoint_epochs"])
    elif a.action=="downstream":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=_run_downstream_checkpoints(c,root,a.dataset)
    elif a.action=="report":
        c=_apply_saved_tuning(c,root,a.ignore_tuned); result={"report":str(write_final_report(c))}
    else:
        tune=run_tuning(c); c=apply_tuned_hyperparameters(c,tune["selected_parameters"]); pre=train_ssl(c,root/"pretrain_full",use_all_data=True,validate=False,early_stop=False,checkpoint_epochs=c["training"]["checkpoint_epochs"]); result={"tuning":tune,"pretrain":pre}
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
