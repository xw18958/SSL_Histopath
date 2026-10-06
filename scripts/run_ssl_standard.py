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
    run_image_text_retrieval,
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
from pannuke_ssl.ssl_framework.run_management import attach_run_context, write_run_metadata
from pannuke_ssl.ssl_framework.image_text_datasets import IMAGE_TEXT_DATASETS
from pannuke_ssl.ssl_framework.retrieval_protocol import RETRIEVAL_VERSION


METHODS=("ijepa","lejepa","simplex_sigreg_lejepa","dinov3")


def _apply_saved_tuning(c,root:Path,ignore_tuned:bool):
    selected=root/"tuning/best_hyperparameters.yaml"
    if selected.exists() and not ignore_tuned:
        values=yaml.safe_load(selected.read_text())["selected"]
        return apply_tuned_hyperparameters(c,values)
    return c


def _validate_action_dataset(action: str, dataset: str) -> None:
    single_dataset_actions = {"downstream"}
    image_text_actions = {"image-text-retrieval"}
    if action in image_text_actions:
        if dataset not in IMAGE_TEXT_DATASETS:
            raise ValueError(f"Image-text retrieval requires one of {IMAGE_TEXT_DATASETS}, got {dataset!r}")
        return
    if dataset != PANNUKE_DATASET and action not in single_dataset_actions:
        raise ValueError(
            f"Optional dataset {dataset!r} is downstream-only; "
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
        completed = output / "test_metrics.json"
        if completed.is_file():
            results[str(epoch)] = json.loads(completed.read_text())
            continue
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


def _run_image_text_retrieval_checkpoints(c, root: Path, dataset: str):
    if dataset not in IMAGE_TEXT_DATASETS:
        raise ValueError(f"Unknown image-text dataset {dataset!r}")
    epochs=[int(epoch) for epoch in c["training"]["checkpoint_epochs"]]
    results={}
    for epoch in epochs:
        checkpoint=root/"pretrain_full"/"checkpoints"/f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing required SSL checkpoint: {checkpoint}")
        output=root/"image_text_retrieval"/dataset/f"epoch_{epoch}"
        completed = output / "test_retrieval_metrics.json"
        if completed.is_file():
            saved = json.loads(completed.read_text())
            if saved.get("evaluation_protocol_version") != RETRIEVAL_VERSION:
                raise RuntimeError("Superseded retrieval output: preserve it and use run_retrieval_repair.py with a fresh output root")
            results[str(epoch)] = saved
            continue
        results[str(epoch)]=run_image_text_retrieval(c,checkpoint,output,dataset=dataset)
    return {"dataset":dataset,"checkpoint_epochs":epochs,"results":results}


def _run_image_text_retrieval_suite(c, root: Path):
    return {
        "datasets": IMAGE_TEXT_DATASETS,
        "results": {dataset:_run_image_text_retrieval_checkpoints(c,root,dataset) for dataset in IMAGE_TEXT_DATASETS},
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=(
            "tune","pretrain","downstream","downstream-suite","image-text-retrieval","image-text-retrieval-suite","pipeline","report"
        ),
    )
    p.add_argument("--method",required=True,choices=METHODS)
    p.add_argument("--dataset",choices=(PANNUKE_DATASET,*EXTERNAL_DATASETS,*IMAGE_TEXT_DATASETS),default=PANNUKE_DATASET)
    p.add_argument("--suite-tier",choices=("main","supplementary","all"),default="main")
    p.add_argument("--learning-rate",type=float,default=None)
    p.add_argument("--run-id",default=None)
    p.add_argument("--simplex-components",type=int,default=None)
    p.add_argument("--ignore-tuned",action="store_true")
    a=p.parse_args()
    try:
        _validate_action_dataset(a.action,a.dataset)
    except ValueError as error:
        p.error(str(error))
    c=load_standard_config(a.method)
    if a.simplex_components is not None:
        if a.method != "simplex_sigreg_lejepa":
            p.error("--simplex-components is only valid for simplex_sigreg_lejepa")
        if a.simplex_components < 2 or a.simplex_components - 1 > int(c["backbone"]["hidden_size"]):
            p.error("Invalid simplex component count for the shared feature dimension")
        c["method"]["objective"]["simplex_components"] = int(a.simplex_components)
    c,run_id,root=attach_run_context(c,a.run_id)
    write_run_metadata(c,root,run_id=run_id,action=a.action,dataset=None if a.dataset==PANNUKE_DATASET else a.dataset)
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
    elif a.action=="image-text-retrieval":
        result=_run_image_text_retrieval_checkpoints(c,root,a.dataset)
    elif a.action=="image-text-retrieval-suite":
        result=_run_image_text_retrieval_suite(c,root)
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
