from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch,yaml
from pannuke_ssl.utils import atomic_json_dump,write_csv
from .config import apply_lr,load_tuning_spec
from .trainer import train_ssl

def run_tuning(c:dict[str,Any]):
    spec=load_tuning_spec(c['method']['name']); root=Path(c['output']['root'])/c['method']['name']/'tuning'; root.mkdir(parents=True,exist_ok=True)
    if (root/'tuning_summary.json').exists(): raise FileExistsError('Tuning already exists')
    p=spec['parameters']['learning_rate']; source=float(p['source_value']); rows=[]
    for lr in [float(x) for x in p['candidates']]:
        try:
            trial=train_ssl(apply_lr(c,lr),root/f'lr_{lr:.0e}',epochs=20,interval=5,early_stop=False); rows.append({'learning_rate':lr,'is_source_value':lr==source,'status':'completed','best_epoch':trial['best_epoch'],'best_validation_linear_macro_f1':trial['best_validation_linear_macro_f1'],'error':None})
        except (FloatingPointError,torch.cuda.OutOfMemoryError) as exc:
            torch.cuda.empty_cache(); rows.append({'learning_rate':lr,'is_source_value':lr==source,'status':'failed_safety_stop','best_epoch':None,'best_validation_linear_macro_f1':None,'error':f'{type(exc).__name__}: {exc}'})
    valid=[r for r in rows if r['status']=='completed' and r['best_validation_linear_macro_f1'] is not None]
    if not valid: raise RuntimeError('No LR tuning candidate completed safely')
    score=max(float(r['best_validation_linear_macro_f1']) for r in valid); tied=[r for r in valid if math.isclose(float(r['best_validation_linear_macro_f1']),score,abs_tol=1e-12)]; source_tie=[r for r in tied if r['is_source_value']]; chosen=(source_tie or tied)[0]
    result={'method':c['method']['name'],'selected_value':float(chosen['learning_rate']),'selected_validation_linear_macro_f1':float(chosen['best_validation_linear_macro_f1']),'source_value':source,'source_value_included':True,'source_reference':p['source_reference'],'budget':spec['budget'],'rows':rows,'test_used':False}; write_csv(rows,root/'tuning_summary.csv'); atomic_json_dump(result,root/'tuning_summary.json'); (root/'best_hyperparameters.yaml').write_text(yaml.safe_dump({'method':c['method']['name'],'selected':{'learning_rate':result['selected_value']},'selection':{'metric':'linear_val_macro_f1','value':result['selected_validation_linear_macro_f1'],'test_used':False},'source_reference':{'learning_rate':source,'included_in_candidates':True,'citation':p['source_reference']}},sort_keys=False),encoding='utf-8'); return result
