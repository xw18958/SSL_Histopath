# B0-LN encoder-gradient audit

Diagnostic-only, single-batch, FP32 gradient measurement for the existing B0-LN objective. No parameters, optimizer state, or EMA state were updated during measurement.

## Exact scope

- The prediction gradient is `∇θ Lpred`; the regularizer gradient is `∇θ (lambda_reg * Lreg)` for student encoder parameters only.
- The B0-LN teacher target is the existing stateless FP32 final-dimension LayerNorm; student tokens are not normalized.
- The replay used the original 300-epoch schedule controls through epoch 50, then verified the replayed epoch-20 student state against the saved B0-LN epoch-20 checkpoint before accepting missing epoch states.
- The fixed diagnostic batch has 8 unlabeled SSL images and a private transition seed; the balanced downstream/test loader was never constructed.

## Results

| Epoch | Pred loss | Weighted reg loss | ||g_pred|| | ||g_reg|| | Reg/Pred grad ratio | Cosine | Ratio class | Cosine class |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.311128 | 0.0953725 | 0.232676 | 0.0214809 | 0.0923211 | -0.00386071 | weak | mostly orthogonal / weak relation |
| 5 | 0.0705421 | 0.0932062 | 0.131781 | 0.0500527 | 0.379817 | 0.0119925 | moderate | mostly orthogonal / weak relation |
| 10 | 0.052802 | 0.0822947 | 0.3547 | 0.582497 | 1.64222 | -0.195633 | regularizer-dominant | mostly orthogonal / weak relation |
| 20 | 0.100997 | 0.152369 | 0.653708 | 2.4025 | 3.6752 | -0.345898 | regularizer-dominant | meaningfully conflicting |

## Interpretation

The diagnostic is blocked: the isolated replay did not reproduce the saved epoch-20 student state exactly. No unverified missing-epoch gradient rows are reported.
Replay status: `blocked_replay_mismatch`.

## Integrity and no-test proof

- Epoch-20 replay verification: `{"checkpoint_epoch": 20, "different_tensor_values": 87467301, "exact": false, "expected_hash": "9bc18dbe18b9a49f671843b33165648dc6c1b20dd3239f072e8193fc94616525", "max_abs_difference": 0.021803483366966248, "max_relative_difference": 3685884.75, "replayed_hash": "f3b9f77bedaa32638f4088e7486c9bd7f8a8c9d71be9494dfc3658b5b8e6f626"}`.
- Protected B0/B1/I-JEPA reference entries before/after: `205` / `205`; verification passed.
- Saved B0-LN checkpoint SHA-256 before/after: `89b162f899e1e90dd7d02b205de66218db17258fa2c4a288836268cebba63e02` (unchanged).
- The script constructed only the unlabeled 7,901-image SSL loader and the fixed 8-image diagnostic batch. It did not read balanced downstream metadata, construct a test loader, touch a test marker, or read test predictions.

The result is descriptive and single-seed. It does not establish statistical significance or by itself prove that changing `lambda_reg` will improve B0-LN.
