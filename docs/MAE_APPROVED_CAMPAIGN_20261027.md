# MAE CVPR experiment campaign: approved 27 October 2026

Branch: feat/mae-matched-baseline-20261027.

Approved tuning: sequential learning-rate candidates 0.0000375, 0.000075
and 0.00015; 20 epochs each, 6305 PanNuke development train patches;
selection by development linear-probe macro-F1. No grid search or repeated
pretraining seeds.

Final pretraining: one fresh random-initialization MAE on the exact shared
CLIP ViT-B/32, 256px, patch32, 75 percent random masking, normalized pixel
reconstruction, decoder 8 layers / width512 / 16 heads, AdamW BF16,
batch128, 250 epochs and checkpoints 100/150/200/250, 7901 unlabeled
PanNuke images.

Main evaluation: frozen 100- and 200-epoch MAE encoder, 11 existing
selected pathology classification datasets, ARCH and PathCap image-text
retrieval, image-to-image patch retrieval (mAP), trainable 1/5/10-shot
few-shot (100 fixed subsets), GlaS and EBHI semantic segmentation,
official five-class BCSS semantic segmentation, and GlaS gland instance
segmentation with official 60 Test A / 20 Test B subsets and object metrics.
The GlaS instance task is not substituted with binary semantic mIoU.
No data labels are loaded during MAE SSL training.

All evaluation adapters retain the existing test-once marker and source
dataset/split/hash checking. The new branch only expands method adapters;
the existing downstream evaluation losses and metrics are not rewritten.

Source/server layout:
- Tuning: xwan0900@gpu1-jinman-2.it.usyd.edu.au;
  source /raid1/xwan0900/mae_review_smoke_20261027;
  run root /raid1/xwan0900/mae_cvpr_20261027;
  tracking logs/tuning.log.
- 250-epoch training, 11-classification-dataset suite and ARCH retrieval:
  xwan0900@gpu1-358-0.research.sydney.edu.au;
  source /home/xwan0900/mae_review_smoke_20261027;
  run root /home/xwan0900/mae_cvpr_20261027;
  tracking logs/pretrain_gate.log and logs/gpu1_downstream.log.
- PathCap retrieval, patch mAP, trainable few-shot, semantic segmentation
  and native GlaS gland instance probing:
  xwan0900@gpu2-358-0.research.sydney.edu.au, dual Blackwell PRO 6000
  Max-Q; corresponding source and run paths in /home/xwan0900;
  tracking logs/gpu2_downstream.log and task-specific logs.

The training dependency gate starts 250-epoch MAE only after the completed
three-candidate tuning result is pulled from gpu1-jinman-2 and validated.
Downstream dependency gates start only after verified 100- and 200-epoch
checkpoints and completed 250-epoch training; the GPU2 copy is SHA-verified.
All gates stop on missing data, failed prerequisites, incompatible
checkpoints or incomplete existing test markers.

These jobs compare same architecture, PanNuke images, epochs and frozen
evaluation. Do not claim equal FLOPs, views, or official native MAE ViT-B/16
training because masked MAE sees only 16 of 64 patches on one crop while
LeJEPA/Simplex process eight augmented views. Report measured compute
and the longer original MAE training recipe as limitations.
