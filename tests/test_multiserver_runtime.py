from __future__ import annotations

import json
from pathlib import Path

from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.run_management import default_run_id, experiment_root, write_run_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_standard_config_paths_are_runtime_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv('SSL_DATA_ROOT', str(tmp_path/'data'))
    monkeypatch.setenv('SSL_MODEL_ROOT', str(tmp_path/'models'))
    monkeypatch.setenv('SSL_RUN_ROOT', str(tmp_path/'runs'))
    monkeypatch.setenv('SSL_PROJECT_ROOT', str(ROOT))
    c=load_standard_config('lejepa')
    assert c['data']['root'] == str(tmp_path/'data/PanNuke/data')
    assert c['backbone']['config_dir'] == str(tmp_path/'models/plip_model')
    assert c['output']['root'] == str(tmp_path/'runs/ssl_standard')
    assert c['manifests']['root'] == str(ROOT/'manifests/ssl_standard')


def test_default_run_ids_separate_simplex_k_values(monkeypatch, tmp_path):
    monkeypatch.setenv('SSL_DATA_ROOT', str(tmp_path/'data'))
    monkeypatch.setenv('SSL_MODEL_ROOT', str(tmp_path/'models'))
    monkeypatch.setenv('SSL_RUN_ROOT', str(tmp_path/'runs'))
    monkeypatch.setenv('SSL_PROJECT_ROOT', str(ROOT))
    le=load_standard_config('lejepa')
    sx=load_standard_config('simplex_sigreg_lejepa')
    assert default_run_id(le) == 'ssl-lejepa-s20260903'
    assert default_run_id(sx) == 'ssl-simplex-k64-s20260903'
    sx['method']['objective']['simplex_components']=16
    assert default_run_id(sx) == 'ssl-simplex-k16-s20260903'
    assert experiment_root(sx).name == 'ssl-simplex-k16-s20260903'


def test_committed_manifests_are_server_path_independent():
    manifest_dir=ROOT/'manifests/ssl_standard/dataset_manifests'
    files=sorted(manifest_dir.glob('*.json'))
    assert files
    for path in files:
        document=json.loads(path.read_text(encoding='utf-8'))
        assert document['schema_version'] == 3
        assert document['dataset_root_spec'].startswith('${SSL_DATA_ROOT}/')
        raw=path.read_text(encoding='utf-8')
        assert '/raid1/xwan0900' not in raw
        assert '/home/xwan0900' not in raw


def test_committed_image_text_manifests_are_server_path_independent():
    manifest_dir=ROOT/'manifests/ssl_standard/image_text_manifests'
    files=sorted(manifest_dir.glob('*.json'))
    assert {path.stem for path in files} == {'arch','ipath','pathcap'}
    for path in files:
        document=json.loads(path.read_text(encoding='utf-8'))
        assert document['schema_version'] == 2
        assert document['dataset_root_spec'].startswith('${SSL_DATA_ROOT}/')
        assert document['retrieval_evaluation_ready'] is True
        raw=path.read_text(encoding='utf-8')
        assert '/raid1/xwan0900' not in raw
        assert '/home/xwan0900' not in raw


def test_run_metadata_keeps_per_invocation_execution_history(monkeypatch, tmp_path):
    monkeypatch.setenv('SSL_DATA_ROOT', str(tmp_path/'data'))
    monkeypatch.setenv('SSL_MODEL_ROOT', str(tmp_path/'models'))
    monkeypatch.setenv('SSL_RUN_ROOT', str(tmp_path/'runs'))
    monkeypatch.setenv('SSL_PROJECT_ROOT', str(ROOT))
    c=load_standard_config('lejepa')
    run_root=tmp_path/'one-run'
    write_run_metadata(c,run_root,run_id='ssl-lejepa-s20260903',action='pretrain')
    write_run_metadata(c,run_root,run_id='ssl-lejepa-s20260903',action='downstream',dataset='mhist')
    assert (run_root/'run_metadata.json').is_file()
    executions=sorted((run_root/'executions').glob('*.json'))
    assert len(executions) == 2
    actions={json.loads(path.read_text())['action'] for path in executions}
    assert actions == {'pretrain','downstream'}
