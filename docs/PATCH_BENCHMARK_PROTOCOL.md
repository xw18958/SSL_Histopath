# Frozen Histopathology Patch Benchmark

Original LeJEPA and Regular-Simplex LeJEPA K=64 (epochs 100, 150, 200, 250, 300), plus original Regular-Simplex LeJEPA K=8 (epochs 100, 150, 200, 250); 15 histopathology classification datasets; **no SSL retraining**.

## Representation and evaluation

Both models use the **frozen backbone encoder before the projector**, mean-pooled final patch tokens. RGB preprocessing matches the existing deterministic frozen-probe pipeline: bicubic resize-short-side to 256, centered 256 crop, raw image / 255, BF16 CUDA inference; no stochastic augmentation.

**Retrieval:** all training images are the gallery, all test images are queries, with all same-class gallery records relevant. Features receive per-sample L2 normalization; similarity is cosine. Report mAP, macro-mAP (per query-class equal weight), Acc@1/3/5 (at least one correct class in top-k), and 5-NN majority-vote accuracy, with deterministic tie handling.

**Few-shot:** use exactly 1, 5 and 10 labeled training images per class. Generate 100 predetermined independently drawn support subsets for each dataset/shot setting, shared across all methods and their available checkpoints. Each prototype is the mean of individually L2-normalized support features, L2-normalized again. Classify by largest prototype cosine similarity. Report macro-F1, balanced accuracy, accuracy, mean, sample standard deviation and 95% t confidence interval; store all per-subset scores and paired differences.

All cache records include checkpoint SHA256, versioned manifest SHA256, feature dimension and exact split record indices. The test set is never used for hyperparameter selection, checkpoint selection or cohort/feature normalization.

## Exact-image split leakage repair

The frozen **main** evaluation uses original splits unless decoded-RGB content checks find an exact duplicate crossing train/val/test. Original manifests are retained, never overwritten.

- BreaKHis: **image-level split suffices; patient isolation is not required**. Exact duplicates from the original image-level partition are repaired using deterministic within-class image swaps. The 3,552 images and original per-class split counts are retained. Exact copies carrying conflicting class labels are held in validation, outside the retrieval gallery and evaluation test queries.
- Endometrial: versioned duplicate-content filter excludes only offending records without moving any remaining patient/slide grouping between splits.
- Oral OSCC: versioned duplicate-content filter excludes only offending records; original labels are unchanged.
- All others: original dataset manifests, with cross-split exact content audit. Source-level independence is marked unverified where provenance identifiers are unavailable.

The repaired manifests include parent hashes, removed/swapped record audit trails and zero exact-content cross-split overlap assertions. Near-duplicate content and unknown patient/slide relationships remain scientific caveats, not silently asserted absent.

## Completeness checks

Completed: 210 method/checkpoint/dataset evaluations (75 LeJEPA, 75 K64, 60 K8), including 210 retrieval evaluations and 630 few-shot settings with 100 support subsets each. Unit tests cover independent retrieval AP, query-macro aggregation, shot sampling, few-shot prototypes, confidence intervals, cache validation and manifest invariance.

Every matched method comparison is checked for identical dataset/sample order/labels, matching gallery/query and support IDs, feature dimension and split hash. Independent recomputation produces overall equal-weight 15-dataset means and differences (percentage points). Final reporting is only declared complete when all jobs are present.


## Three-method results and reproducibility

K=8 has four saved checkpoints at epochs 100, 150, 200 and 250. Its epoch 300 does not exist. The three_methods directory contains shared-epoch three-way comparisons, as well as the original epoch-300 LeJEPA/K64 comparison. This benchmark uses cosine nearest-prototype classification, which is not identical to UNI SimpleShot.

Private image-level split manifests are excluded from this public repository; they contain per-image labels and identifiers. Run the included deterministic split-repair scripts to produce corrected manifests locally and audit their exact-image overlap before using the main split variant.

To run the optional multi-server coordinator, populate your own private copy of configs/patch_benchmark/cluster.example.json and set PATCH_BENCHMARK_CLUSTER_CONFIG=/absolute/path/cluster.json. Use --preflight and only then --run --confirm-full-run. Server names, usernames and private storage paths are not committed.

The public results contain only dataset-level summaries, aggregate metrics and verification, not raw features or individual patient/image records.
