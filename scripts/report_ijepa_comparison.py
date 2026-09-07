"""Render a comparison from saved evaluations only; never calls an encoder/probe."""
import csv
import json
from pathlib import Path
import numpy as np
from pannuke_ssl.ijepa_training import protected_manifest, sha
from pannuke_ssl.utils import atomic_json_dump, write_csv

ROOT=Path("/raid1/xwan0900/SSL_proj")
SSL=ROOT/"outputs/ijepa_duration_pilot"
IJ=ROOT/"outputs/ijepa_duration_pilot_final_probe"
B0=ROOT/"outputs/b0_duration_pilot_final_probe"
OUT=ROOT/"outputs/ijepa_b0_comparison"


def read(path): return json.loads(path.read_text())
def csvrows(path):
    with path.open() as f: return list(csv.DictReader(f))


def main():
    OUT.mkdir(exist_ok=True)
    b0,ij=read(B0/"linear_probe_summary.json"),read(IJ/"linear_probe_summary.json")
    selection=read(SSL/"duration_selection.json")
    assert b0["encoder_description"]=="epoch 10 of the 300-epoch schedule with 30-epoch warmup"
    assert b0["correct_test_predictions"]==70 and b0["test_evaluations"]==1
    assert ij["test_evaluations"]==1
    marker=read(IJ/"test_started.json")
    assert marker["selection_sha256"]==sha(IJ/"selection.json")==ij["selection_sha256"]
    assert marker["probe_checkpoint_sha256"]==sha(IJ/"best_linear_probe.pt")
    assert marker["encoder_checkpoint_sha256"]==sha(SSL/"checkpoints/best.pt")==selection["checkpoint_sha256"]
    assert sha(IJ/"test_started.json")==ij["test_marker_sha256"]
    assert sha(IJ/"test_predictions.npz")==ij["predictions_sha256"]
    assert protected_manifest(ROOT)==read(SSL/"protected_manifest.json")
    assert len(list((SSL/"checkpoints").glob("*.pt")))==1
    assert len(list(IJ.glob("*.pt")))==1
    leaderboard=csvrows(IJ/"probe_leaderboard.csv")
    assert len(leaderboard)==6
    assert len({r["cache_sha256"] for r in leaderboard})==1
    assert len({r["standardized_tensor_sha256"] for r in leaderboard})==1
    assert {r["cache_sha256"] for r in leaderboard}=={sha(IJ/"train_val_features.npz")}
    matrix=np.loadtxt(IJ/"test_confusion_matrix.csv",delimiter=",",dtype=int)
    assert matrix.shape==(19,19) and np.all(matrix.sum(1)==13)
    # Exercise the real test entrypoint guard, while forbidding any extraction.
    from unittest.mock import patch
    import ijepa_duration_final_probe as final_probe
    c=read(IJ/"resolved_config.json")
    with patch.object(final_probe,"extract",side_effect=AssertionError("Repeated extraction forbidden")) as extract:
        try:
            final_probe.test_once(c)
        except FileExistsError:
            pass
        else:
            raise AssertionError("One-time test entrypoint did not reject repeated access")
        extract.assert_not_called()
    assert sha(IJ/"test_started.json")==ij["test_marker_sha256"]
    metrics=[]
    labels={"accuracy":"Accuracy","balanced_accuracy":"Balanced accuracy","macro_f1":"Macro-F1","weighted_f1":"Weighted-F1"}
    for metric,label in labels.items():
        metrics.append({"metric":label,"b0_percent":100*b0["test"][metric],
                        "ijepa_percent":100*ij["test"][metric],
                        "ijepa_minus_b0_percentage_points":100*(ij["test"][metric]-b0["test"][metric])})
    write_csv(metrics,OUT/"comparison_metrics.csv")
    bclasses=csvrows(B0/"test_per_class_metrics.csv")
    iclasses=csvrows(IJ/"test_per_class_metrics.csv")
    perclass=[]
    for b,i in zip(bclasses,iclasses,strict=True):
        assert b["class_id"]==i["class_id"] and b["class_name"]==i["class_name"]
        row={"class_id":i["class_id"],"class_name":i["class_name"],"support":13}
        for m in ("precision","recall","f1-score"):
            row["b0_"+m]=float(b[m]); row["ijepa_"+m]=float(i[m])
        row["f1_delta_pp"]=100*(float(i["f1-score"])-float(b["f1-score"]))
        perclass.append(row)
    write_csv(perclass,OUT/"per_class_comparison.csv")
    p=ij["selection"]
    lines=["# I-JEPA versus B0: matched downstream comparison", "",
           "Both rows are single-seed, transductive linear-probe results. SSL used all 7,901 PanNuke images unlabeled, including downstream validation/test images.","",
           "| Test metric | B0 | I-JEPA | I-JEPA − B0 (pp) |","|---|---:|---:|---:|"]
    lines += [f'| {r["metric"]} | {r["b0_percent"]:.2f}% | {r["ijepa_percent"]:.2f}% | {r["ijepa_minus_b0_percentage_points"]:+.2f} |' for r in metrics]
    lines += ["",f'B0: 70/247 correct; I-JEPA: {ij["correct_test_predictions"]}/247 correct. Each class has 13 test images, so accuracy equals balanced accuracy and weighted-F1 equals macro-F1.',"",
              "| Selection | B0 | I-JEPA |","|---|---:|---:|",
              f'| Selected SSL epoch | 10 | {selection["selected_epoch"]} |',
              f'| Duration-monitor validation macro-F1 | 33.0250% | {100*selection["selection_score"]:.4f}% |',
              f'| Probe LR | 0.001 | {p["learning_rate"]} |',f'| Probe weight decay | 0 | {p["weight_decay"]} |',
              f'| Selected probe epoch | 17 | {p["best_epoch"]} |',
              f'| Final-probe validation macro-F1 | {100*b0["selection"]["val_macro_f1"]:.4f}% | {100*p["val_macro_f1"]:.4f}% |',"",
              "B0 uses **epoch 10 of the 300-epoch schedule with 30-epoch warmup**. "
              f'I-JEPA uses **epoch {selection["selected_epoch"]} of the 300-epoch schedule with 30-epoch warmup**. Each encoder was independently selected using validation macro-F1; the six-trial final probe was then selected separately.',"",
              "## Matched protocol and implementation", "",
              "Both use fresh random ViT-B/32 encoders from the local architecture configuration, seed 20260903, clean native 256×256 images, 64 patch tokens of width 768, batch size 128, AdamW, BF16, 300 epochs and 30 warmup epochs, LR 1e-4→1e-6, weight decay 0.04→0.40, EMA 0.996→1.0, and gradient clipping at norm 5. B0 is the retained completed run; it was never retrained or retested.","",
              "I-JEPA predicts LayerNorm-normalized, stop-gradient teacher patch targets from a masked student context using Smooth-L1. Patches are removed before encoder attention. The predictor uses width 384, two Transformer blocks, six heads, MLP ratio four, and fixed two-dimensional sine/cosine positions with learned target mask queries. B0 degradation conditioning and variance/covariance regularization are absent.","",
              "This is the requested matched-objective adaptation, not an official-scale I-JEPA reproduction. Its 8×8 mask sampler draws target scales 0.15–0.20, context scales 0.85–1.0, and target aspect ratios 0.75–1.5 before integer rectangle rounding. Rounding realizes 8, 9, or 12 tokens per target. Four targets are mutually disjoint and removed from the context; at least 10 context tokens remain. Contexts are randomly trimmed to the batch-minimum length. Native CLIP CLS is retained alongside visible context tokens; target pixels cannot enter its attention. The official implementation permits target-target overlap; the stricter disjointness here follows the approved comparison plan.","",
              "Duration monitoring uses the same weighted 20-NN and deterministic LBFGS probe math as B0 at epoch 0 and every 10 epochs. A new strict loader decodes only train/validation for monitoring. The selection threshold is ≥0.005 macro-F1 improvement, retaining earlier ties. SSL itself decodes all 7,901 images without using downstream labels. The only saved SSL checkpoint is the selected student.","",
              "Final probing freezes the encoder and mean-pools FP32-converted patch tokens to 768 features after BF16 inference. The 2,052/247 train/validation feature arrays are cached once. Sample mean/std are fitted on training only, with std floor 1e-6. Each Linear(768,19) uses AdamW, batch size 256, constant LR, seed 20260903, at most 50 epochs, patience eight. LR {0.001,0.003,0.01} × weight decay {0,0.0001} are selected by validation macro-F1, with earliest-epoch/first-grid-candidate ties. The 247-image test set is decoded and evaluated once only after selection is frozen.","",
              "## Verification", "",
              "The GPU smoke checked 1,600 mask examples, three real-image optimization steps, finite loss/gradients, frozen teacher and EMA, predictor block ordering, 64×768 shape, and exact invariance of student context outputs to changes in held-out target pixels. Pretrained weight-loading calls were forbidden. Metadata retained 2,052/247/247 totals and 108/13/13 per-class counts. No original image copies were saved.","",
              "Final-probe smoke checked frozen encoder tensor hashes before/after extraction, exact repeated-batch determinism, train-only standardization, a two-epoch trial, and cache/tensor identity across all six real trials. Selected checkpoint, feature cache, probe, selection, saved prediction and exclusive test marker hashes are audited. A guarded second entrypoint invocation correctly raised FileExistsError before image extraction (which was explicitly mocked to fail if reached); the classifier was evaluated only once. Full SHA256/size/mtime manifests confirm B0 outputs, original metadata, and pre-existing source/config/script files remain unchanged.","",
              f'I-JEPA encoder SHA256: `{ij["encoder_checkpoint_sha256"]}`. B0 encoder SHA256: `{b0["encoder_checkpoint_sha256"]}`.',"",
              "## I-JEPA validation grid", "","| LR | Weight decay | Best epoch | Epochs run | Validation macro-F1 |","|---:|---:|---:|---:|---:|"]
    lines += [f'| {r["learning_rate"]} | {r["weight_decay"]} | {r["best_epoch"]} | {r["epochs_run"]} | {100*float(r["val_macro_f1"]):.4f}% |' for r in leaderboard]
    lines += ["","## Per-class held-out metrics","","All precision/recall/F1 entries are percentages; each class has 13 test examples.","",
              "| Class | B0 P | B0 R | B0 F1 | I-JEPA P | I-JEPA R | I-JEPA F1 | ΔF1 pp |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in perclass:
        vals=[100*r[method+"_"+metric] for method in ("b0","ijepa") for metric in ("precision","recall","f1-score")]
        lines.append("| "+r["class_name"]+" | "+" | ".join(f"{v:.2f}" for v in vals)+f' | {r["f1_delta_pp"]:+.2f} |')
    lines += ["","![I-JEPA test confusion matrix](ijepa_test_confusion_matrix.png)","",
              "Rows are true classes; columns are predicted classes. Class order is the per-class table above. Numeric matrices and saved test predictions accompany this report.","",
              "## Interpretation limits", "",
              "These are descriptive results from one SSL seed per method, not evidence of statistical significance or a universal optimal SSL duration. Each checkpoint belongs to its full 300-epoch schedule with 30-epoch warmup. A subsequent reproducibility study should fix warmup steps and use multiple SSL seeds. The comparison is specific to this small transductive PanNuke setting, this coarse 8×8 mask adaptation, and the matched lightweight predictor; it does not establish the performance of published full-scale I-JEPA. Fine-tuning, zero-shot and retrieval metrics were not evaluated in this linear-probe workflow.","",
              "References: [I-JEPA paper](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html); [official configuration](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml); [official masking implementation](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py).", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    import shutil
    for source,name in ((IJ/"test_confusion_matrix.png","ijepa_test_confusion_matrix.png"),
                        (IJ/"test_confusion_matrix.csv","ijepa_test_confusion_matrix.csv"),
                        (B0/"test_confusion_matrix.png","b0_test_confusion_matrix.png"),
                        (B0/"test_confusion_matrix.csv","b0_test_confusion_matrix.csv")):
        shutil.copyfile(source,OUT/name)
    atomic_json_dump({"b0":b0,"ijepa":ij,"comparison":metrics,
                      "protected_b0_unchanged":True,"exclusive_test_marker_rejects_repeat":True},OUT/"comparison_summary.json")
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=="__main__": main()
