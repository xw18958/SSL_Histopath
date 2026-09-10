# B0-LN encoder-gradient audit

Diagnostic-only, single-batch, FP32 gradient measurement for the existing B0-LN objective. No parameters, optimizer state, or EMA state were updated during measurement.

## Exact scope

- The prediction gradient is `∇θ Lpred`; the regularizer gradient is `∇θ (lambda_reg * Lreg)` for student encoder parameters only.
- The B0-LN teacher target is the existing stateless FP32 final-dimension LayerNorm; student tokens are not normalized.
- The exact saved epoch-20 student state was loaded directly. Historical EMA-teacher and predictor states were unavailable, so this is a saved-representation diagnostic rather than a historical training-state gradient audit.
- The fixed diagnostic batch has 8 unlabeled SSL images and a private transition seed; the balanced downstream/test loader was never constructed.

## Results

| Epoch | Pred loss | Weighted reg loss | ||g_pred|| | ||g_reg|| | Reg/Pred grad ratio | Cosine | Ratio class | Cosine class |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 20 | 0.406596 | 0.156449 | 0.889622 | 2.32824 | 2.61711 | -0.172328 | regularizer-dominant | mostly orthogonal / weak relation |

## Interpretation

This is a valid partial diagnostic for the exact saved epoch-20 student only. Epochs 1, 5, 10, and 50 are unavailable because isolated replay did not reproduce the saved student state; no temporal conclusion is possible.

## Integrity and no-test proof

- Epoch-20 replay verification: `{"exact": false, "reason": "historical_teacher_and_predictor_states_unavailable; isolated_replay_student_hash_mismatch", "replay_max_abs_difference": 0.021803483366966248, "replay_student_hash": "f3b9f77bedaa32638f4088e7486c9bd7f8a8c9d71be9494dfc3658b5b8e6f626", "saved_student_hash": "9bc18dbe18b9a49f671843b33165648dc6c1b20dd3239f072e8193fc94616525"}`.
- Protected B0/B1/I-JEPA reference entries before/after: `205` / `205`; verification passed.
- Saved B0-LN checkpoint SHA-256 before/after: `89b162f899e1e90dd7d02b205de66218db17258fa2c4a288836268cebba63e02` (unchanged).
- The script verified the 7,901-record unlabeled SSL source index and loaded only the fixed 8-image diagnostic batch. It did not construct the full SSL loader, read balanced downstream metadata, construct a test loader, touch a test marker, or read test predictions.

The result is descriptive and single-seed. It does not establish statistical significance or by itself prove that changing `lambda_reg` will improve B0-LN.

## Saved-state limitation

Only the exact saved epoch-20 student checkpoint was available. The historical EMA teacher and predictor were not retained. The row above therefore uses a deterministic fresh B0 predictor and a read-only teacher surrogate initialized from the saved student; it is valid as a diagnostic at the saved representation but is not the historical optimizer-state gradient. Epochs 1, 5, 10, and 50 remain unavailable after the replay hash mismatch, so no temporal conclusion is possible.
