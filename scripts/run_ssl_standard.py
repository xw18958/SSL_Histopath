from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from pannuke_ssl.ssl_framework import (
    apply_lr,
    apply_tuned_hyperparameters,
    load_standard_config,
    run_downstream,
    run_tuning,
    train_ssl,
    write_final_report,
)
from pannuke_ssl.ssl_framework.external_datasets import (
    DatasetProtocolNotReadyError,
    EXTERNAL_DATASETS,
    PANNUKE_DATASET,
    assert_dataset_ready,
    ready_external_datasets,
)
from pannuke_ssl.ssl_framework.image_retrieval import run_image_retrieval


METHODS=("ijepa","lejepa","simplex_sigreg_lejepa","dinov3")


def _apply_saved_tuning(c,root:Path,ignore_tuned:bool):
    selected=root/"tuning/best_hyperparameters.yaml"
    if selected.exists() and not ignore_tuned:
        values=yaml.safe_load(selected.read_text())["selected"]
        return apply_tuned_hyperparameters(c,values)
    return c


def _validate_action_dataset(action: str, dataset: str) -> None:
    single_dataset_actions = {"downstream", "image-retrieval"}
    if dataset != PANNUKE_DATASET and action not in single_dataset_actions:
        raise ValueError(
            f"Optional dataset {dataset!r} is downstream-only/retrieval-only; "
            f"tune/pretrain/pipeline/report/suite actions remain fixed to {PANNUKE_DATASET}"
        )
    if action in single_dataset_actions:
        if dataset == PANNUKE_DATASET:
            raise ValueError("PanNuke is development/pretraining-only; final evaluation must use an external dataset")
        try:
            assert_dataset_ready(dataset)
        except DatasetProtocolNotReadyError as error:
            raise ValueError(str(error)) from error


def _downstream_output_path(root: Path, dataset: str) -> Path:
    if dataset == PANNUKE_DATASET:
        return root / "downstream"
    if dataset not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown downstream dataset {dataset!r}")
    return root / "downstream_datasets" / dataset


def _run_downstream_checkpoints(c, root: Path, dataset: str):
    assert_dataset_ready(dataset)
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


def _suite_dataset_names(tier: str) -> tuple[str, ...]:
    if tier == "all":
        return ready_external_datasets()
    return ready_external_datasets(tier=tier)


def _run_downstream_suite(c, root: Path, tier: str):
    datasets = _suite_dataset_names(tier)
    if not datasets:
        raise RuntimeError(f"No ready external datasets for tier={tier!r}")
    return {
        "tier": tier,
        "datasets": datasets,
        "results": {dataset: _run_downstream_checkpoints(c, root, dataset) for dataset in datasets},
    }


def _image_retrieval_output_path(root: Path, dataset: str) -> Path:
    if dataset not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown image-retrieval dataset {dataset!r}")
    return root / "image_retrieval_datasets" / dataset


def _run_image_retrieval_checkpoints(c, root: Path, dataset: str, ks: list[int] | tuple[int, ...]):
    assert_dataset_ready(dataset)
    epochs = [int(epoch) for epoch in c["training"]["checkpoint_epochs"]]
    base = _image_retrieval_output_path(root, dataset)
    results = {}
    for epoch in epochs:
        checkpoint = root / "pretrain_full" / "checkpoints" / f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing required SSL checkpoint: {checkpoint}")
        output = base / f"epoch_{epoch}"
        results[str(epoch)] = run_image_retrieval(c, checkpoint, output, dataset=dataset, ks=ks)
    return {"dataset": dataset, "checkpoint_epochs": epochs, "ks": list(ks), "results": results}


def _run_image_retrieval_suite(c, root: Path, tier: str, ks: list[int] | tuple[int, ...]):
    datasets = _suite_dataset_names(tier)
    if not datasets:
        raise RuntimeError(f"No ready external datasets for tier={tier!r}")
    return {
        "tier": tier,
        "datasets": datasets,
        "ks": list(ks),
        "results": {dataset: _run_image_retrieval_checkpoints(c, root, dataset, ks) for dataset in datasets},
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=(
            "tune","pretrain","downstream","downstream-suite",
            "image-retrieval","image-retrieval-suite","pipeline","report"
        ),
    )
    p.add_argument("--method",required=True,choices=METHODS)
    p.add_argument("--dataset",choices=(PANNUKE_DATASET,*EXTERNAL_DATASETS),default=PANNUKE_DATASET)
    p.add_argument("--suite-tier",choices=("main","supplementary","all"),default="main")
    p.add_argument("--retrieval-k",type=int,nargs="+",default=[1,5,10])
    p.add_argument("--learning-rate",type=float,default=None)
    p.add_argument("--ignore-tuned",action="store_true")
    a=p.parse_args()
    try:
        _validate_action_dataset(a.action,a.dataset)
    except ValueError as error:
        p.error(str(error))
    c=load_standard_config(a.method)
    root=Path(c["output"]["root"])/a.method
    if a.action=="tune":
        result=run_tuning(c)
    elif a.action=="pretrain":
        if a.learning_rate is not None: c=apply_lr(c,a.learning_rate)
        result=train_ssl(c,root/"pretrain_full",use_all_data=True,validate=False,early_stop=False,checkpoint_epochs=c["training"]["checkpoint_epochs"])
    elif a.action=="downstream":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=_run_downstream_checkpoints(c,root,a.dataset)
    elif a.action=="downstream-suite":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=_run_downstream_suite(c,root,a.suite_tier)
    elif a.action=="image-retrieval":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=_run_image_retrieval_checkpoints(c,root,a.dataset,a.retrieval_k)
    elif a.action=="image-retrieval-suite":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result=_run_image_retrieval_suite(c,root,a.suite_tier,a.retrieval_k)
    elif a.action=="report":
        c=_apply_saved_tuning(c,root,a.ignore_tuned)
        result={"report":str(write_final_report(c))}
    else:
        tune=run_tuning(c)
        c=apply_tuned_hyperparameters(c,tune["selected_parameters"])
        pre=train_ssl(c,root/"pretrain_full",use_all_data=True,validate=False,early_stop=False,checkpoint_epochs=c["training"]["checkpoint_epochs"])
        result={"tuning":tune,"pretrain":pre}
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
