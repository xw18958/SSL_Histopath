from __future__ import annotations
import argparse, json
from pathlib import Path
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.image_text_datasets import (
    IMAGE_TEXT_DATASETS,
    inspect_image_text_dataset,
    prepare_image_text_manifest,
)


def main() -> None:
    parser=argparse.ArgumentParser(description="Inspect or freeze leakage-aware pathology image-text splits")
    parser.add_argument("--dataset",required=True,choices=IMAGE_TEXT_DATASETS)
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--val-count",type=int)
    parser.add_argument("--test-count",type=int,default=None)
    parser.add_argument("--train-count",type=int,default=None)
    parser.add_argument("--allow-missing-images",action="store_true")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--output-root",type=Path,default=Path(load_standard_config("lejepa")["manifests"]["root"]))
    args=parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_image_text_dataset(args.dataset),indent=2),flush=True)
        return
    if args.val_count is None:
        parser.error("--val-count is required when freezing a retrieval split; no universal VAL size was predeclared")
    result=prepare_image_text_manifest(
        args.dataset,args.output_root,val_count=args.val_count,test_count=args.test_count,
        train_count=args.train_count,allow_missing_images=args.allow_missing_images,force=args.force,
    )
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
