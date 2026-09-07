"""CPU mask tests plus a small real-image GPU optimization/leakage smoke."""
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import CLIPVisionModel
from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa import BlockMasks, build_models, encode_context, objective
from pannuke_ssl.ijepa_training import protected_manifest, verified_metadata, validate_config
from pannuke_ssl.models import update_ema
from pannuke_ssl.parquet import build_source_index, preload_images
from pannuke_ssl.utils import atomic_json_dump, seed_everything


def main():
    started = time.perf_counter()
    c = load_yaml("configs/ijepa_duration_pilot.yaml")
    validate_config(c)
    output = Path(c["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "pretrain_metrics.csv").exists()
    torch.set_num_threads(4)
    protected = protected_manifest(output.parents[1])
    prior = output / "protected_manifest.json"
    if prior.exists():
        assert protected == json.loads(prior.read_text())
    else:
        atomic_json_dump(protected, prior)
    sampler = BlockMasks(c["seed"])
    min_context, lengths = 64, set()
    for _ in range(100):
        ctx, tgt = sampler.sample(16)
        assert tgt.shape[:2] == (16,4) and ctx.shape[1] >= 10
        min_context = min(min_context,ctx.shape[1]); lengths.add(tgt.shape[2])
        for a,b in zip(ctx.tolist(),tgt.tolist()):
            assert len(set(a)) == len(a)
            union = [i for block in b for i in block]
            assert len(set(union)) == len(union)
            assert set(a).isdisjoint(union)
            assert min(a+union)>=0 and max(a+union)<64
            for block in b:
                ys, xs = {i//8 for i in block}, {i%8 for i in block}
                assert len(ys)*len(xs) == len(block)
                assert len(ys)==max(ys)-min(ys)+1 and len(xs)==max(xs)-min(xs)+1
    a1,b1=BlockMasks(5).sample(4); a2,b2=BlockMasks(5).sample(4)
    assert torch.equal(a1,a2) and torch.equal(b1,b2)
    rows = verified_metadata(c)
    selected = [r for r in rows if r["split"]=="train"][:8]
    index = build_source_index(Path(c["data_root"]))
    cache = preload_images(selected,index)
    assert set(cache)=={(r["fold"],r["sample_index"]) for r in selected}
    images = torch.stack([torch.from_numpy(np.array(cache[r["fold"],r["sample_index"]],copy=True)).permute(2,0,1)
                          for r in selected]).cuda().float()/255
    seed_everything(c["seed"])
    # Architecture config loading is allowed; pretrained weight loading must fail.
    with patch.object(CLIPVisionModel,"from_pretrained",side_effect=AssertionError("Pretrained loading forbidden")):
        student,teacher,predictor=build_models(c,torch.device("cuda"))
    assert not any(p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(a,b) for a,b in zip(student.parameters(),teacher.parameters()))
    student.eval(); predictor.eval()
    context,targets = BlockMasks(77).sample(8,"cuda")
    with torch.no_grad():
        full = student(images)
        assert full.shape==(8,64,768)
        equivalent = encode_context(student,images,torch.arange(64,device="cuda").expand(8,-1))
        assert torch.allclose(full,equivalent,atol=1e-6,rtol=1e-6)
        visible = encode_context(student,images,context)
        changed = images.clone()
        for bi in range(8):
            for ti in targets[bi].flatten().tolist():
                y,x=divmod(ti,8)
                changed[bi,:,y*32:(y+1)*32,x*32:(x+1)*32]=.123
        masked_change = encode_context(student,changed,context)
        assert torch.equal(visible,masked_change), "Target pixel leakage through encoder attention"
        predicted = predictor(visible,context,targets)
        permuted = predictor(visible,context,targets[:,[3,1,0,2]])
        assert torch.allclose(predicted[:,[3,1,0,2]],permuted,atol=1e-5,rtol=1e-5)
    student.train(); predictor.train()
    optimizer = torch.optim.AdamW(list(student.parameters())+list(predictor.parameters()),lr=1e-4)
    initial = next(student.parameters()).detach().clone()
    losses=[]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        context,targets=BlockMasks(99+_).sample(8,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,prediction,target=objective(student,teacher,predictor,images,context,targets)
        assert torch.isfinite(loss) and not target.requires_grad
        assert prediction.shape==(*targets.shape,768)
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        assert student.model.vision_model.embeddings.patch_embedding.weight.grad.abs().sum()>0
        assert predictor.mask_token.grad.abs().sum()>0
        assert predictor.input_projection.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
        optimizer.step()
        before=next(teacher.parameters()).detach().clone()
        update_ema(student,teacher,.996)
        expected=before*.996+next(student.parameters()).detach()*.004
        assert torch.allclose(next(teacher.parameters()),expected,atol=1e-7,rtol=1e-6)
        losses.append(float(loss))
    assert not torch.equal(initial,next(student.parameters()))
    assert protected_manifest(output.parents[1])==protected
    result={"passed":True,"mask_examples_checked":1600,"minimum_context_tokens":min_context,
            "realized_target_tokens":sorted(lengths),"gpu_steps":3,"gpu_losses":losses,
            "pretrained_weight_loading_forbidden":True,"student_teacher_initially_identical":True,
            "masked_before_attention":True,"target_pixel_changes_leave_context_exactly_unchanged":True,
            "predictor_block_order_verified":True,"full_encoding_matches_original_encoder":True,
            "teacher_stopgradient_and_ema_verified":True,"metadata_split_counts":[2052,247,247],
            "metadata_perclass_counts":[108,13,13],"smoke_decoded_splits":["train"],
            "test_images_decoded_by_smoke":False,"protected_b0_unchanged":True,
            "seconds":time.perf_counter()-started}
    atomic_json_dump(result,output/"smoke_test.json")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
