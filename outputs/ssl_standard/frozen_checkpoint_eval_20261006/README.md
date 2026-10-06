> **Retrieval results superseded.** LeJEPA's tokenizer was incomplete, and the original scoring did not handle repeated captions for caption-level retrieval. Use [the corrected bundle](../retrieval_repair_20261006/README.md). Classification values are retained. The original checkpoint-primary designation is superseded; the repaired comparison reports all saved checkpoints.

# Frozen-checkpoint downstream evaluation (2026-10-06)

This directory contains the compact, publication-facing outputs for the final frozen-checkpoint comparison between LeJEPA (ssl-lejepa-s20260903) and Simplex-SIGReg-LeJEPA K=64, sigma=1 (ssl-simplex-k64-s20260903).

## Scope

- 15 histopathology classification datasets.
- Frozen encoder checkpoints at epochs 100, 150, 200, 250, and 300.
- ARCH and IPATH image-text retrieval.
- 150 classification checkpoint evaluations + 20 retrieval checkpoint evaluations = **170 total**.
- All saved checkpoints are reported. No checkpoint is designated primary in the repaired comparison.

## Files

- classification_results.csv: all classification metrics.
- retrieval_results.csv: all ARCH/IPATH retrieval metrics.
- checkpoint_summary.csv: cross-dataset checkpoint means.
- all_results.json: full saved test-metric payloads in one portable file.
- all_selections.json: probe/projector selection metadata.
- dataset_provenance.json: canonical split policy, counts, and manifest hashes.
- verification_report.json: integrity/protocol verification result.
- pretraining/: 300-epoch pretraining metrics, run summaries, resolved configs, and run metadata.

Large model checkpoints (.pt) and prediction arrays (.npz) are intentionally not committed.

## Verification

Outputs were checked against the canonical manifests in manifests/ssl_standard/dataset_manifests. Checks require identical dataset provenance across both methods, exact checkpoint epochs, correct test counts, one-time TEST markers, TEST excluded from model/hyperparameter selection, and frozen SSL/text encoders for retrieval.

Verification status: **PASS** (2844/2844 checks passed).