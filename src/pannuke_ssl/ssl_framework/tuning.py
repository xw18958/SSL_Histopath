from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch,yaml
from pannuke_ssl.utils import atomic_json_dump,write_csv
from .config import apply_tuned_hyperparameters,load_tuning_spec
from .trainer import train_ssl

def run_tuning(c:dict[str,Any]):
    spec=load_tuning_spec(c['method']['name']); root=Path(c['output']['root'])/c['method']['name']/'tuning'; root.mkdir(parents=True,exist_ok=True)
    if (root/'tuning_summary.json').exists(): raise FileExistsError('Tuning already exists')
    p=spec['parameters']['learning_rate']; source=float(p['source_value']); lrs=[float(x) for x in p['candidates']]
    k_spec=spec['parameters'].get('simplex_components'); ks=[int(x) for x in k_spec['candidates']] if k_spec else [None]
    rows=[]
    for k in ks:
        for lr in lrs:
            selected={'learning_rate':lr}
            if k is not None: selected['simplex_components']=int(k)
            trial_config=apply_tuned_hyperparameters(c,selected)
            trial_name=f'lr_{lr:.0e}' if k is None else f'K_{k}_lr_{lr:.0e}'
            try:
                trial=train_ssl(trial_config,root/trial_name,epochs=20,interval=5,early_stop=False)
                row={'learning_rate':lr,'is_source_value':lr==source,'status':'completed','best_epoch':trial['best_epoch'],'best_validation_linear_macro_f1':trial['best_validation_linear_macro_f1'],'error':None}
            except (FloatingPointError,torch.cuda.OutOfMemoryError) as exc:
                torch.cuda.empty_cache(); row={'learning_rate':lr,'is_source_value':lr==source,'status':'failed_safety_stop','best_epoch':None,'best_validation_linear_macro_f1':None,'error':f'{type(exc).__name__}: {exc}'}
            if k is not None: row['simplex_components']=int(k)
            rows.append(row)
    valid=[r for r in rows if r['status']=='completed' and r['best_validation_linear_macro_f1'] is not None]
    if not valid: raise RuntimeError('No tuning candidate completed safely')
    score=max(float(r['best_validation_linear_macro_f1']) for r in valid); tied=[r for r in valid if math.isclose(float(r['best_validation_linear_macro_f1']),score,abs_tol=1e-12)]; source_tie=[r for r in tied if r['is_source_value']]
    pool=source_tie or tied
    chosen=min(pool,key=lambda r:int(r.get('simplex_components',0))) if k_spec else pool[0]
    selected_parameters={'learning_rate':float(chosen['learning_rate'])}
    if k_spec: selected_parameters['simplex_components']=int(chosen['simplex_components'])
    result={'method':c['method']['name'],'selected_value':float(chosen['learning_rate']),'selected_validation_linear_macro_f1':float(chosen['best_validation_linear_macro_f1']),'source_value':source,'source_value_included':True,'source_reference':p['source_reference'],'selected_parameters':selected_parameters,'budget':spec['budget'],'rows':rows,'test_used':False}
    if k_spec: result['selected_simplex_components']=int(chosen['simplex_components'])
    write_csv(rows,root/'tuning_summary.csv'); atomic_json_dump(result,root/'tuning_summary.json'); (root/'best_hyperparameters.yaml').write_text(yaml.safe_dump({'method':c['method']['name'],'selected':selected_parameters,'selection':{'metric':'linear_val_macro_f1','value':result['selected_validation_linear_macro_f1'],'test_used':False},'source_reference':{'learning_rate':source,'included_in_candidates':True,'citation':p['source_reference']}},sort_keys=False),encoding='utf-8'); return result
