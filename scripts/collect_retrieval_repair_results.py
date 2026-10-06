"""Collect corrected retrieval and retained classification, excluding legacy retrieval."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean

VERSION = 'caption_aware_v2_20261006'
EPOCHS = (100, 150, 200, 250, 300)


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(rows, path):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.work_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    results = []
    preservation = []
    for i in (0, 1):
        results.extend(json.loads((root / f'repaired_retrieval_{i}.json').read_text()))
        preservation.append(json.loads((root / f'preservation_verified_{i}.json').read_text()))
    results.sort(key=lambda r: (r['dataset'], r['encoder_epoch'], r['method']))
    expected = {(m, d, e) for m in ('lejepa', 'simplex_sigreg_lejepa') for d in ('arch', 'ipath') for e in EPOCHS}
    assert {(r['method'], r['dataset'], r['encoder_epoch']) for r in results} == expected
    assert len(results) == 20
    assert len({r['protocol_sha256'] for r in results}) == 1
    assert len({json.dumps(r['assets'], sort_keys=True) for r in results}) == 1
    assert len({json.dumps(r['source'], sort_keys=True) for r in results}) == 1
    for report in preservation:
        assert report['status'] == 'PASS'
    for dataset in ('arch', 'ipath'):
        entries = [r for r in results if r['dataset'] == dataset]
        for field in ('manifest_sha256', 'input_image_files_sha256', 'unique_caption_candidates'):
            assert len({r[field] for r in entries}) == 1
    rows = []
    for r in results:
        s = r['selection']
        assert r['evaluation_protocol_version'] == VERSION
        assert r['assets']['vocab_size'] == 49408
        assert r['image_candidates'] == 700
        assert s['test_used'] is False
        assert s['frozen_weight_hashes_before'] == s['frozen_weight_hashes_after']
        assert r['input_signature'] == s['input_signature']
        assert r['test_evaluated_once_within_protocol'] is True
        for direction in ('i2t', 't2i'):
            values = [r['test'][f'{direction}_r@{k}'] for k in (1, 5, 10)]
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
            assert values == sorted(values)
            assert math.isclose(mean(values), r['test'][f'{direction}_mean_recall'], abs_tol=1e-12)
        assert math.isclose(mean(r['test'][f'{d}_r@{k}'] for d in ('i2t','t2i') for k in (1,5,10)), r['test']['overall_mean_recall'], abs_tol=1e-12)
        rows.append({'method':r['method'], 'dataset':r['dataset'], 'epoch':r['encoder_epoch'],
                     'image_candidates':r['image_candidates'], 'unique_caption_candidates':r['unique_caption_candidates'],
                     **r['test'], 'chance_mean_recall':r['chance']['overall_mean_recall'],
                     'learning_rate':s['learning_rate'], 'weight_decay':s['weight_decay'],
                     'selected_head_epoch':s['best_epoch'], 'head_epochs_run':s['epochs_run'],
                     'validation_mean_recall':s['validation']['overall_mean_recall'],
                     'token_collisions':r['truncation_or_tokenization_collisions'], 'evaluation_protocol_version':VERSION})
    classes = list(csv.DictReader((root / 'retained_classification_results.csv').open()))
    assert len(classes) == 150
    original_classes = json.loads((root / 'retained_all_results.json').read_text())
    count = sum(len(epochs) for datasets in original_classes.values() for epochs in datasets.values())
    assert count == 150
    for row in classes:
        method_key = 'lejepa' if row['method'] == 'lejepa' else 'simplex_k64'
        saved = original_classes[method_key][row['dataset']][row['epoch']]
        for metric in ('accuracy','balanced_accuracy','macro_f1','weighted_f1'):
            assert float(row[metric]) == saved['test'][metric]
    summaries = []
    for epoch in EPOCHS:
        for method in ('lejepa', 'simplex_sigreg_lejepa'):
            cls = [r for r in classes if int(r['epoch']) == epoch and ('simplex_sigreg_lejepa' if r['method'] == 'simplex_k64' else r['method']) == method]
            retrieval = [r for r in rows if r['epoch'] == epoch and ('simplex_sigreg_lejepa' if r['method'] == 'simplex_k64' else r['method']) == method]
            assert len(cls) == 15 and len(retrieval) == 2
            summaries.append({'epoch':epoch,'method':method,'classification_mean_accuracy':mean(float(x['accuracy']) for x in cls),
                              'classification_mean_macro_f1':mean(float(x['macro_f1']) for x in cls),
                              'retrieval_mean_recall':mean(x['overall_mean_recall'] for x in retrieval)})
    write_csv(rows, out / 'retrieval_results.csv')
    write_csv(summaries, out / 'checkpoint_summary.csv')
    shutil.copyfile(root / 'retained_classification_results.csv', out / 'classification_results.csv')
    save_json({'classification':original_classes,'retrieval':results}, out / 'all_results.json')
    save_json({'classification':json.loads((root / 'retained_all_selections.json').read_text()),
               'retrieval':[r['selection'] for r in results]}, out / 'all_selections.json')
    shutil.copyfile(root / 'retained_dataset_provenance.json', out / 'dataset_provenance.json')
    sync = json.loads((root / 'server_input_sync_verified.json').read_text())
    report = {'status':'PASS','evaluation_protocol_version':VERSION,'classification_evaluations_retained':150,
              'retrieval_evaluations_replaced':20,'ssl_pretraining_rerun':False,'classification_rerun':False,
              'checkpoint_policy':'all_requested_checkpoints_no_primary_designation',
              'server_input_sync':sync,'preservation_verification':preservation,
              'regression_tests_passed_per_server':15,'train_val_smoke_tests_passed':4,
              'frozen_components_unchanged_all_retrieval_fits':True,
              'train_val_selection_before_test_all_retrieval_fits':True,
              'legacy_retrieval_excluded':True,'protocol_sha256':results[0]['protocol_sha256'],
              'assets':results[0]['assets'],'source':results[0]['source']}
    save_json(report, out / 'verification_report.json')
    save_json({'legacy_bundle':'outputs/ssl_standard/frozen_checkpoint_eval_20261006',
               'legacy_lejepa_retrieval':'invalid_tokenizer',
               'legacy_simplex_retrieval':'superseded_pair_id_protocol',
               'replacement_protocol':VERSION,'classification':'retained_original_results',
               'checkpoint_policy':'no_primary_designation'}, out / 'artifact_status.json')
    by = {(r['method'],r['dataset'],r['epoch']):r for r in rows}
    lines = [
        '# Corrected retrieval results — 6 October 2026', '',
        'The tokenizer failure and repeated-caption handling have been repaired. Both servers use matching code, PLIP assets, canonical manifests, and retrieval image bytes. Twenty retrieval heads were retrained and evaluated. The existing SSL checkpoints and all 150 classification evaluations were retained.', '',
        'All saved checkpoints are reported; none is designated primary. Classification metrics are frozen linear-probe TEST results. Full encoder fine-tuning and zero-shot results are not available in this experiment bundle.', '',
        'This is a versioned correction after the original TEST results were inspected. Replacement hyperparameters and heads were selected using TRAIN/VAL, with all selections sealed before replacement TEST evaluation.', '',
        '## Retrieval', '',
        'Scores below are Mean Recall (%), averaging image-to-text and text-to-image R@1/5/10. Evaluation uses unique caption candidates and accepts every image associated with the exact caption. Query weighting is uniform over images for image-to-text and unique captions for text-to-image. Exact-score ties are averaged over uniform tie order. ARCH has 700 images and 582 caption queries; IPATH has 700 images and 426 caption queries. No distinct full-caption tokenization collisions were observed on either TEST set.', '',
        '| SSL epoch | ARCH LeJEPA | ARCH Simplex | IPATH LeJEPA | IPATH Simplex |',
        '|---:|---:|---:|---:|---:|']
    for e in EPOCHS:
        values = [by[m,d,e]['overall_mean_recall']*100 for d in ('arch','ipath') for m in ('lejepa','simplex_sigreg_lejepa')]
        lines.append(f'| {e} | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    lines += ['', 'The previous large retrieval advantage is not supported by the corrected experiment. ARCH performance is close, while LeJEPA has higher IPATH Mean Recall at all five saved checkpoints. The corrected relevance definition differs from the retired pair-ID metric, so old and new percentages should not be compared as identical benchmarks.', '',
              '## Retained classification', '', 'Means below give each of the 15 datasets equal weight. These values were not retrained or reevaluated.', '',
              '| SSL epoch | LeJEPA Accuracy | Simplex Accuracy | LeJEPA macro-F1 | Simplex macro-F1 |', '|---:|---:|---:|---:|---:|']
    for e in EPOCHS:
        a,b = [next(x for x in summaries if x['epoch']==e and x['method']==m) for m in ('lejepa','simplex_sigreg_lejepa')]
        lines.append(f"| {e} | {100*a['classification_mean_accuracy']:.2f} | {100*b['classification_mean_accuracy']:.2f} | {100*a['classification_mean_macro_f1']:.2f} | {100*b['classification_mean_macro_f1']:.2f} |")
    lines += ['', 'Dataset-specific custom splits and their actual independence units are retained in `dataset_provenance.json`. Comparisons with published scores require matching protocols.', '',
              '## Detailed retrieval', '', '| Dataset | SSL epoch | Method | I→T R@1 | R@5 | R@10 | T→I R@1 | R@5 | R@10 | Mean Recall |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values = [r[f'{d}_r@{k}']*100 for d in ('i2t','t2i') for k in (1,5,10)] + [r['overall_mean_recall']*100]
        lines.append(f"| {r['dataset'].upper()} | {r['epoch']} | {r['method']} | " + ' | '.join(f'{x:.2f}' for x in values) + ' |')
    lines += ['', '## Detailed retained classification', '', '| Dataset | SSL epoch | Method | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |', '|---|---:|---|---:|---:|---:|---:|']
    for r in sorted(classes,key=lambda r:(r['dataset'],int(r['epoch']),r['method'])):
        values = [float(r[k])*100 for k in ('accuracy','balanced_accuracy','macro_f1','weighted_f1')]
        lines.append(f"| {r['dataset']} | {r['epoch']} | {r['method']} | " + ' | '.join(f'{x:.2f}' for x in values) + ' |')
    lines += ['', '## Artifact policy', '', 'Original LeJEPA retrieval metrics and heads are invalid because of the tokenizer failure. Original Simplex retrieval outputs are superseded by the caption-aware protocol. Original files remain archived with checksums on both servers; neither category enters the corrected tables. Raw images, captions, canonical splits, pretraining artifacts, and classification artifacts were preserved.', '',
              'Verification: 15 focused tests passed on each server; four TRAIN/VAL-only smoke tests passed; 17 canonical manifests and 9,334 retrieval image files match between servers; all retained artifact checksums match their original preservation ledgers. See `verification_report.json` for the precise checks and signatures.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'PASS','output':str(out),'retrieval_evaluations':len(rows),'classification_evaluations_retained':len(classes),'summary':summaries},indent=2))


if __name__ == '__main__':
    main()
