import pytest
import torch
import importlib.util
import json
from pathlib import Path

from pannuke_ssl.ssl_framework.image_text_retrieval import _sym_clip_loss, _train_trial
from pannuke_ssl.ssl_framework.retrieval_protocol import (
    caption_groups, caption_aware_metrics, caption_aware_chance,
    multi_positive_clip_loss, validate_plip_assets,
)


def test_missing_tokenizer_is_rejected_before_transformers_fallback(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'placeholder')
    with pytest.raises(RuntimeError, match='Incomplete PLIP assets'):
        validate_plip_assets(tmp_path)


def test_two_token_vocabulary_is_rejected(tmp_path, monkeypatch):
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        (tmp_path / name).write_bytes(b'placeholder')
    class Broken:
        vocab_size = 2
    monkeypatch.setattr('pannuke_ssl.ssl_framework.retrieval_protocol.AutoTokenizer.from_pretrained', lambda *a, **k: Broken())
    with pytest.raises(RuntimeError, match='vocabulary'):
        validate_plip_assets(tmp_path)


def test_unique_caption_loss_equals_original_clip_loss():
    torch.manual_seed(19)
    image, text = torch.randn(6, 8), torch.randn(6, 8)
    torch.testing.assert_close(_sym_clip_loss(image, text, 4.), multi_positive_clip_loss(image, text, torch.arange(6), 4.))


def test_both_images_sharing_a_caption_are_correct():
    image = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    groups = torch.tensor([0, 0, 1])
    metrics = caption_aware_metrics(image, image.clone(), groups)
    assert all(value == 1. for value in metrics.values())
    assert float(multi_positive_clip_loss(image, image.clone(), groups, 20.)) < .2


def test_reordering_does_not_change_scores():
    torch.manual_seed(23)
    image = torch.randn(12, 8)
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6])
    text = torch.randn(7, 8)[groups]
    permutation = torch.randperm(len(image))
    before = caption_aware_metrics(image, text, groups)
    after = caption_aware_metrics(image[permutation], text[permutation], groups[permutation])
    for key in before:
        assert before[key] == pytest.approx(after[key], abs=1e-12)


def test_fully_tied_embeddings_equal_group_aware_chance():
    groups = torch.tensor([0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7])
    same = torch.ones(len(groups), 8)
    actual, expected = caption_aware_metrics(same, same, groups), caption_aware_chance(groups)
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)


def test_caption_identity_is_exact_and_empty_is_rejected():
    assert caption_groups(['same', 'same', 'Same', 'same.']).tolist() == [0, 0, 1, 2]
    with pytest.raises(ValueError, match='Empty'):
        caption_groups(['   '])


def test_multi_positive_head_training_has_finite_updates():
    torch.manual_seed(29)
    groups = torch.arange(16).repeat_interleave(2)
    text, image = torch.randn(16, 8)[groups], torch.randn(32, 12)
    result = _train_trial(image, text, image, text, lr=.001, wd=0., epochs=2,
                         batch_size=16, seed=31, device=torch.device('cpu'), logit_scale=4.,
                         train_caption_ids=groups, val_caption_ids=groups)
    assert list(result['state']) == ['weight']
    assert torch.isfinite(result['state']['weight']).all()
    assert len(result['history']) == 2
    assert 0 <= result['val_overall_mean_recall'] <= 1


def _standard_runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_ssl_standard.py'
    spec = importlib.util.spec_from_file_location('standard_retrieval_resume_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_runner_rejects_retired_retrieval(tmp_path):
    standard = _standard_runner()
    c = standard.load_standard_config('lejepa')
    for epoch in c['training']['checkpoint_epochs']:
        ck = tmp_path / 'pretrain_full/checkpoints' / f'epoch_{epoch}.pt'
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.touch()
        output = tmp_path / 'image_text_retrieval/arch' / f'epoch_{epoch}'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'test_retrieval_metrics.json').write_text(json.dumps({'test': {'overall_mean_recall': .01}}))
    with pytest.raises(RuntimeError, match='Superseded retrieval output'):
        standard._run_image_text_retrieval_checkpoints(c, tmp_path, 'arch')


def test_standard_runner_preserves_completed_corrected_retrieval(tmp_path, monkeypatch):
    from pannuke_ssl.ssl_framework.retrieval_protocol import RETRIEVAL_VERSION
    standard = _standard_runner()
    c = standard.load_standard_config('lejepa')
    for epoch in c['training']['checkpoint_epochs']:
        ck = tmp_path / 'pretrain_full/checkpoints' / f'epoch_{epoch}.pt'
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.touch()
        output = tmp_path / 'image_text_retrieval/arch' / f'epoch_{epoch}'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'test_retrieval_metrics.json').write_text(json.dumps({'evaluation_protocol_version': RETRIEVAL_VERSION}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed corrected experiments must not be rerun')
    monkeypatch.setattr(standard, 'run_image_text_retrieval', forbidden)
    result = standard._run_image_text_retrieval_checkpoints(c, tmp_path, 'arch')
    assert len(result['results']) == 4
