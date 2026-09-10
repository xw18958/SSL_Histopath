#!/usr/bin/env python
"""Partial B0-LN gradient audit using the exact saved epoch-20 student.

The original B0-LN duration run retained only its selected student checkpoint;
its EMA teacher and predictor states were intentionally discarded.  This
script therefore measures the requested gradients at that exact student state
with a deterministic fresh B0 predictor and a teacher surrogate initialized
from the same saved student.  It never claims these are the historical
epoch-20 training-state gradients and cannot infer temporal behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from pannuke_ssl.b0_ln_training import verify_protected_manifest
from pannuke_ssl.config import load_yaml
from pannuke_ssl.training import _endpoints, build_b0_models
from pannuke_ssl.utils import atomic_json_dump, seed_everything

# When executed as a file, the repository root is not on sys.path if callers
# intentionally expose only src/ through PYTHONPATH.  Add this local root only
# for importing the other isolated audit helper; no external path is touched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.b0_ln_gradient_audit import (
    _fixed_diagnostic_batch,
    _gradient_audit,
    _render_report,
    _write_csv,
    sha256_path,
    tensor_state_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact saved B0-LN epoch-20 student gradients")
    parser.add_argument("--config", default="configs/b0_ln_gradient_audit.yaml")
    args = parser.parse_args()
    audit_config = load_yaml(args.config)
    reference = load_yaml(audit_config["reference_config"])
    output = Path(audit_config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
    output.mkdir(parents=True)

    reference_output = Path(reference["output_dir"])
    checkpoint_path = reference_output / "checkpoints/best.pt"
    checkpoint_sha_before = sha256_path(checkpoint_path)
    if checkpoint_sha_before != "89b162f899e1e90dd7d02b205de66218db17258fa2c4a288836268cebba63e02":
        raise RuntimeError("Saved B0-LN checkpoint SHA does not match the recorded epoch-20 artifact")
    protected_before = verify_protected_manifest(reference_output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 20:
        raise RuntimeError("The saved B0-LN checkpoint is not epoch 20")
    saved_student = checkpoint["student"]
    saved_student_hash = tensor_state_hash(saved_student)
    if saved_student_hash != "9bc18dbe18b9a49f671843b33165648dc6c1b20dd3239f072e8193fc94616525":
        raise RuntimeError("Saved B0-LN student state does not match the recorded epoch-20 hash")

    seed = int(reference["seed"])
    seed_everything(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("Saved epoch-20 gradient audit requires CUDA")
    device = torch.device("cuda")
    student, teacher, predictor = build_b0_models(reference, device)
    student.load_state_dict(saved_student)
    # No historical EMA target was retained.  This teacher is a read-only
    # surrogate initialized from the exact saved student, solely to evaluate
    # the B0-LN target path without inventing a training state.
    teacher.load_state_dict(saved_student)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher surrogate unexpectedly requires gradients")
    loaded_hash_before = tensor_state_hash({name: value.detach().cpu() for name, value in student.state_dict().items()})
    if loaded_hash_before != saved_student_hash:
        raise RuntimeError("Loaded student differs from the exact saved epoch-20 state")
    predictor_hash = tensor_state_hash({name: value.detach().cpu() for name, value in predictor.state_dict().items()})

    images, actions, source, target, delta, records = _fixed_diagnostic_batch(
        reference,
        device,
        batch_size=8,
        seed=int(audit_config.get("diagnostic_transition_seed", 20268204)),
    )
    row = _gradient_audit(
        student,
        teacher,
        predictor,
        images,
        actions,
        source,
        target,
        delta,
        _endpoints(reference["endpoints_json"]),
        reference["train"],
        epoch=20,
    )
    loaded_hash_after = tensor_state_hash({name: value.detach().cpu() for name, value in student.state_dict().items()})
    teacher_hash_after = tensor_state_hash({name: value.detach().cpu() for name, value in teacher.state_dict().items()})
    if loaded_hash_after != loaded_hash_before or teacher_hash_after != saved_student_hash:
        raise RuntimeError("Gradient measurement changed the saved-state model surrogates")

    checkpoint_sha_after = sha256_path(checkpoint_path)
    protected_after = verify_protected_manifest(reference_output)
    historical_blocker = {
        "exact": False,
        "reason": "historical_teacher_and_predictor_states_unavailable; isolated_replay_student_hash_mismatch",
        "saved_student_hash": saved_student_hash,
        "replay_student_hash": "f3b9f77bedaa32638f4088e7486c9bd7f8a8c9d71be9494dfc3658b5b8e6f626",
        "replay_max_abs_difference": 0.021803483366966248,
    }
    result = {
        "status": "partial_saved_epoch20_only",
        "measurement_scope": "exact_saved_student_with_fresh_predictor_and_student_initialized_teacher_surrogate",
        "historical_training_state_verified": False,
        "historical_training_state_blocker": "Original B0-LN artifact retains student only; EMA teacher and predictor states are unavailable, and fresh replay did not reproduce the saved student exactly.",
        "epoch20_state_match": historical_blocker,
        "epoch20_saved_checkpoint_epoch": 20,
        "saved_checkpoint_sha256": checkpoint_sha_before,
        "saved_student_state_hash": saved_student_hash,
        "fresh_predictor_state_hash": predictor_hash,
        "teacher_surrogate_state_hash": teacher_hash_after,
        "gradient_row": row,
        "gradient_rows": [row],
        "audit_epochs_requested": [1, 5, 10, 20, 50],
        "audit_epochs_measured": [20],
        "audit_epochs_unavailable": [1, 5, 10, 50],
        "diagnostic_batch": records,
        "diagnostic_batch_size": len(records),
        "diagnostic_transition_seed": int(audit_config.get("diagnostic_transition_seed", 20268204)),
        "ssl_source_count": 7901,
        "optimizer_constructed": False,
        "optimizer_updated": False,
        "ema_updated": False,
        "student_state_unchanged_during_measurement": loaded_hash_after == loaded_hash_before,
        "balanced_downstream_loader_constructed": False,
        "test_loader_constructed": False,
        "test_marker_touched": False,
        "test_predictions_touched": False,
        "protected_manifest_entries_before": len(protected_before["protected_paths"]),
        "protected_manifest_entries_after": len(protected_after["protected_paths"]),
        "protected_references_unchanged": True,
        "checkpoint_sha256_after": checkpoint_sha_after,
    }
    atomic_json_dump(result, output / "gradient_audit.json")
    _write_csv([row], output / "gradient_audit.csv")

    # Reuse the shared report renderer for the main table, then append the
    # saved-state caveat and the unavailable-epoch statement explicitly.
    report_path = output / "REPORT.md"
    _render_report(result, report_path, protected_before, protected_after, reference)
    report_text = report_path.read_text(encoding="utf-8")
    report_text = report_text.replace(
        "The replay used the original 300-epoch schedule controls through epoch 50, then verified the replayed epoch-20 student state against the saved B0-LN epoch-20 checkpoint before accepting missing epoch states.",
        "The exact saved epoch-20 student state was loaded directly. Historical EMA-teacher and predictor states were unavailable, so this is a saved-representation diagnostic rather than a historical training-state gradient audit.",
    )
    report_text = report_text.replace(
        "The diagnostic is blocked: the isolated replay did not reproduce the saved epoch-20 student state exactly. No unverified missing-epoch gradient rows are reported.\nReplay status: `partial_saved_epoch20_only`.",
        "This is a valid partial diagnostic for the exact saved epoch-20 student only. Epochs 1, 5, 10, and 50 are unavailable because isolated replay did not reproduce the saved student state; no temporal conclusion is possible.",
    )
    report_path.write_text(report_text, encoding="utf-8")
    with (output / "REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Saved-state limitation\n\n"
            "Only the exact saved epoch-20 student checkpoint was available. The historical EMA teacher and predictor were not retained. "
            "The row above therefore uses a deterministic fresh B0 predictor and a read-only teacher surrogate initialized from the saved student; it is valid as a diagnostic at the saved representation but is not the historical optimizer-state gradient. "
            "Epochs 1, 5, 10, and 50 remain unavailable after the replay hash mismatch, so no temporal conclusion is possible.\n"
        )
    atomic_json_dump(
        {
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "saved_student_hash": saved_student_hash,
            "protected_manifest_entries_before": len(protected_before["protected_paths"]),
            "protected_manifest_entries_after": len(protected_after["protected_paths"]),
            "protected_references_unchanged": True,
            "optimizer_constructed": False,
            "optimizer_updated": False,
            "ema_updated": False,
            "test_loader_constructed": False,
            "test_marker_touched": False,
            "test_predictions_touched": False,
        },
        output / "integrity.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
