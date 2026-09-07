# B0/B1 clean vs maximum-degradation validation sanity check

| Encoder | Clean validation macro-F1 | Mixed s=1 validation macro-F1 | Degraded − clean |
|---|---:|---:|---:|
| B0 | 34.66% | 34.22% | -0.43 pp |
| B1 | 26.85% | 33.54% | +6.69 pp |

Maximum degradation used calibrated defocus radius 5 or resolution factor 5.33333 at s=1.0, assigned once by deterministic class-stratified alternation.

## Conclusion

The condition effect is not consistently large for both encoders; this diagnostic alone does not establish clean-domain mismatch as the main explanation.

This is a single-seed validation-only diagnostic. Clean values are hash-verified existing final-probe selections; new features and probes use train/validation only. No held-out-test loader, marker, features, predictions, or metrics were created.
