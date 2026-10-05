from __future__ import annotations

import csv
import io
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.fft import dctn
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedGroupKFold

_IMAGE_SUFFIXES={'.bmp','.jpeg','.jpg','.png','.tif','.tiff'}
_SPLITS=('train','val','test')


def _split_counts_8_1_1(total:int)->dict[str,int]:
    exact={'train':0.8*total,'val':0.1*total,'test':0.1*total}
    out={k:int(math.floor(v)) for k,v in exact.items()}
    order=sorted(out,key=lambda k:(-(exact[k]-out[k]),_SPLITS.index(k)))
    for k in order[:total-sum(out.values())]: out[k]+=1
    return out


def _file_record(path:Path,root:Path,class_id:int,class_name:str,**extra)->dict[str,Any]:
    return {'storage':'file','relative_path':path.relative_to(root).as_posix(),'class_id':int(class_id),'class_name':class_name,**extra}


def _parquet_record(path:Path,root:Path,row_group:int,row_in_group:int,class_id:int,class_name:str,**extra)->dict[str,Any]:
    return {'storage':'parquet','parquet_file':path.relative_to(root).as_posix(),'row_group':int(row_group),'row_in_group':int(row_in_group),'class_id':int(class_id),'class_name':class_name,**extra}


def _hdf5_record(path:Path,root:Path,index:int,class_id:int,class_name:str,**extra)->dict[str,Any]:
    return {'storage':'hdf5','hdf5_file':path.relative_to(root).as_posix(),'hdf5_index':int(index),'class_id':int(class_id),'class_name':class_name,**extra}


def _source_counts(rows:list[dict[str,Any]], class_names:tuple[str,...])->dict[str,int]:
    c=Counter(r['class_name'] for r in rows); return {name:int(c[name]) for name in class_names}


def _plain_balanced(rows:list[dict[str,Any]],class_names:tuple[str,...],quota:int,seed:int):
    targets=_split_counts_8_1_1(quota); out=[]
    for cid,name in enumerate(class_names):
        vals=sorted((r for r in rows if r['class_name']==name),key=lambda r:repr(sorted(r.items())))
        random.Random(seed+cid).shuffle(vals); vals=vals[:quota]
        a=targets['train']; b=a+targets['val']
        for split,part in [('train',vals[:a]),('val',vals[a:b]),('test',vals[b:])]:
            out.extend({**r,'split':split} for r in part)
    return out,targets


def _fold_assignment(rows:list[dict[str,Any]],class_names:tuple[str,...],targets:dict[str,int],seed:int)->dict[str,str]:
    y=np.array([class_names.index(r['class_name']) for r in rows],dtype=np.int64)
    groups=np.array([str(r['group_id']) for r in rows],dtype=object)
    unique=np.unique(groups)
    if len(unique)<10: raise ValueError(f'Need >=10 groups for 10-fold group split, found {len(unique)}')
    best=None
    for attempt in range(40):
        sg=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed+attempt)
        fold_of={}
        for fold,(_,te) in enumerate(sg.split(np.zeros(len(y)),y,groups)):
            for g in np.unique(groups[te]): fold_of[str(g)]=fold
        for vf,tf in permutations(range(10),2):
            cap={s:Counter() for s in _SPLITS}
            for r in rows:
                f=fold_of[str(r['group_id'])]; s='val' if f==vf else ('test' if f==tf else 'train'); cap[s][r['class_name']]+=1
            deficit=sum(max(0,targets[s]-cap[s][name]) for s in _SPLITS for name in class_names)
            dev=sum(abs(cap[s][name]-targets[s]) for s in ('val','test') for name in class_names)
            key=(deficit,dev,attempt,vf,tf)
            if best is None or key<best[0]: best=(key,fold_of,vf,tf,cap)
            if deficit==0 and dev==0: break
        if best and best[0][0]==0: break
    if best is not None and best[0][0] == 0:
        _,fold_of,vf,tf,_=best
        return {g:('val' if f==vf else ('test' if f==tf else 'train')) for g,f in fold_of.items()}

    # Exact capacity-safe fallback: mixed-integer assignment of whole groups.
    # This is used when fixed 10-fold boundaries miss a target by a few samples.
    from scipy.optimize import Bounds, LinearConstraint, milp
    gids=sorted(set(str(g) for g in groups)); ng=len(gids); gi={g:i for i,g in enumerate(gids)}
    counts={(g,name):0 for g in gids for name in class_names}; sizes=Counter()
    for r in rows:
        g=str(r['group_id']); counts[(g,r['class_name'])]+=1; sizes[g]+=1
    nvar=ng*3
    c=np.zeros(nvar,dtype=float)
    rng=np.random.default_rng(seed)
    for i,g in enumerate(gids):
        c[i*3+1]=sizes[g]+rng.random()*1e-6
        c[i*3+2]=sizes[g]+rng.random()*1e-6
        c[i*3+0]=rng.random()*1e-9
    A=[]; lb=[]; ub=[]
    for i in range(ng):
        row=np.zeros(nvar); row[i*3:(i+1)*3]=1; A.append(row); lb.append(1); ub.append(1)
    for si,split in enumerate(_SPLITS):
        for name in class_names:
            row=np.zeros(nvar)
            for i,g in enumerate(gids): row[i*3+si]=counts[(g,name)]
            A.append(row); lb.append(targets[split]); ub.append(np.inf)
    res=milp(c,integrality=np.ones(nvar),bounds=Bounds(np.zeros(nvar),np.ones(nvar)),constraints=LinearConstraint(np.asarray(A),np.asarray(lb),np.asarray(ub)),options={'time_limit':30})
    if not res.success:
        raise ValueError(f'Could not make capacity-safe group split; stratified best={None if best is None else best[0]}, MILP={res.message}')
    assign={}
    for i,g in enumerate(gids): assign[g]=_SPLITS[int(np.argmax(res.x[i*3:(i+1)*3]))]
    return assign


def _grouped_balanced(rows:list[dict[str,Any]],class_names:tuple[str,...],quota:int,seed:int):
    targets=_split_counts_8_1_1(quota)
    assignment=_fold_assignment(rows,class_names,targets,seed)
    out=[]
    for cid,name in enumerate(class_names):
        for si,split in enumerate(_SPLITS):
            vals=sorted((r for r in rows if r['class_name']==name and assignment[str(r['group_id'])]==split),key=lambda r:repr(sorted(r.items())))
            random.Random(seed+cid*100+si).shuffle(vals)
            need=targets[split]
            if len(vals)<need: raise ValueError(f'{name}/{split}: need {need}, have {len(vals)}')
            out.extend({**r,'split':split} for r in vals[:need])
    return out,targets,assignment


def _grouped_natural(rows:list[dict[str,Any]],class_names:tuple[str,...],seed:int):
    y=np.array([class_names.index(r['class_name']) for r in rows]); groups=np.array([str(r['group_id']) for r in rows],dtype=object)
    sg=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed)
    folds=[]
    for _,te in sg.split(np.zeros(len(y)),y,groups): folds.append(set(str(g) for g in np.unique(groups[te])))
    target={name:Counter(r['class_name'] for r in rows)[name]*0.1 for name in class_names}
    best=None
    all_groups=set(str(g) for g in groups)
    for vf,tf in permutations(range(10),2):
        split_groups={'val':folds[vf],'test':folds[tf],'train':all_groups-folds[vf]-folds[tf]}
        counts={split:Counter(r['class_name'] for r in rows if str(r['group_id']) in gs) for split,gs in split_groups.items()}
        if any(counts[split][name]==0 for split in _SPLITS for name in class_names):
            continue
        score=sum(abs(counts[split][name]-Counter(r['class_name'] for r in rows)[name]*({'train':.8,'val':.1,'test':.1}[split])) for split in _SPLITS for name in class_names)
        key=(score,vf,tf)
        if best is None or key<best[0]: best=(key,vf,tf)
    if best is None:
        raise ValueError('No grouped 8:1:1-style split preserves every EBHI class in train/val/test')
    _,vf,tf=best
    assign={g:('val' if g in folds[vf] else ('test' if g in folds[tf] else 'train')) for g in set(groups)}
    out=[{**r,'split':assign[str(r['group_id'])]} for r in rows]
    return out,assign


def _natural_stratified_all(rows:list[dict[str,Any]],seed:int,stratum_key='stratum'):
    out=[]
    strata=defaultdict(list)
    for r in rows: strata[str(r.get(stratum_key,r['class_name']))].append(r)
    for i,key in enumerate(sorted(strata)):
        vals=sorted(strata[key],key=lambda r:repr(sorted(r.items()))); random.Random(seed+i).shuffle(vals)
        n=_split_counts_8_1_1(len(vals)); a=n['train'];b=a+n['val']
        for split,part in [('train',vals[:a]),('val',vals[a:b]),('test',vals[b:])]: out.extend({**r,'split':split} for r in part)
    return out


def _parquet_rows(path:Path,root:Path,class_names:tuple[str,...],label_map=None,path_parser=None):
    pf=pq.ParquetFile(path); out=[]
    for rg in range(pf.metadata.num_row_groups):
        t=pf.read_row_group(rg,columns=['label','image.path'])
        for i,row in enumerate(t.to_pylist()):
            raw=int(row['label']); ipath=str(row['image']['path'])
            if path_parser is not None: name=path_parser(ipath)
            else: name=class_names[label_map[raw] if label_map else raw]
            cid=class_names.index(name)
            out.append(_parquet_record(path,root,rg,i,cid,name,source_name=ipath))
    return out


def kather_2016(config,seed):
    p=next(config.root.glob('*.parquet')); rows=_parquet_rows(p,config.root,config.class_names)
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets=_plain_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'min_1000_smallest_class_then_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'oversampling':False}}


def bach(config,seed):
    def parse(name):
        s=Path(name).stem.lower()
        if s.startswith('is'): return 'InSitu'
        if s.startswith('iv'): return 'Invasive'
        if s.startswith('b'): return 'Benign'
        if s.startswith('n'): return 'Normal'
        raise ValueError(name)
    rows=[]
    for p in sorted(config.root.glob('train*.parquet')): rows.extend(_parquet_rows(p,config.root,config.class_names,path_parser=parse))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets=_plain_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'labeled_400_only_min_1000_then_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'unlabeled_public_test_excluded':True}}


def wsss(config,seed):
    p=config.root/'train.parquet'; pf=pq.ParquetFile(p); rows=[]
    for rg in range(pf.metadata.num_row_groups):
        t=pf.read_row_group(rg,columns=['image_id','label_tumor','label_stroma','label_normal'])
        for i,r in enumerate(t.to_pylist()):
            vals=[r['label_tumor'],r['label_stroma'],r['label_normal']]
            if vals.count(1)!=1 or vals.count(0)!=2: continue
            name=('tumor','stroma','normal')[vals.index(1)]; iid=str(r['image_id'])
            parts=iid.split('-'); gid='-'.join(parts[:6]) if iid.startswith('TCGA-') else parts[0]
            rows.append(_parquet_record(p,config.root,rg,i,config.class_names.index(name),name,group_id=gid,image_id=iid))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':pf.metadata.num_rows,'source_class_counts':counts,'balancing':{'policy':'clean_single_label_min1000_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'WSI','groups':len(assign),'official_unlabeled_val_test_excluded':True}}


def _xlsx(path:Path):
    from xml.etree import ElementTree as ET
    nsu='http://schemas.openxmlformats.org/spreadsheetml/2006/main'; ns={'m':nsu}
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(n.text or '' for n in x.iter(f'{{{nsu}}}t')) for x in root.findall('m:si',ns)]
        sheets=sorted(x for x in z.namelist() if x.startswith('xl/worksheets/sheet') and x.endswith('.xml')); root=ET.fromstring(z.read(sheets[0]))
    def col(ref):
        v=0
        for ch in ''.join(c for c in ref if c.isalpha()).upper(): v=v*26+ord(ch)-64
        return v-1
    rr=[]
    for row in root.findall('.//m:sheetData/m:row',ns):
        d={}
        for cell in row.findall('m:c',ns):
            idx=col(cell.attrib.get('r','A1')); typ=cell.attrib.get('t'); node=cell.find('m:v',ns); val='' if node is None or node.text is None else node.text
            if typ=='s' and val: val=shared[int(val)]
            elif typ=='inlineStr':
                ii=cell.find('m:is',ns); val='' if ii is None else ''.join(n.text or '' for n in ii.iter(f'{{{nsu}}}t'))
            d[idx]=val
        rr.append(d)
    m=max(max(x,default=-1) for x in rr); headers=[rr[0].get(i,'') for i in range(m+1)]
    return [{headers[i]:r.get(i,'') for i in range(len(headers)) if headers[i]} for r in rr[1:] if any(r.values())]


def sicap(config,seed):
    def primary(path):
        out={}
        for r in _xlsx(path):
            active=[n for n in config.class_names if int(float(r[n]))==1]
            if len(active)==1: out[str(r['image_name'])]=active[0]
        return out
    labels={**primary(config.root/'partition/Test/Train.xlsx'),**primary(config.root/'partition/Test/Test.xlsx')}
    patient={str(r['slide_id']):str(r['patient_id']) for r in _xlsx(config.root/'wsi_labels.xlsx')}
    rows=[]
    for fn,name in labels.items():
        p=config.root/'images'/fn; slide=fn.split('_Block_')[0]
        if not p.is_file(): raise FileNotFoundError(p)
        rows.append(_file_record(p,config.root,config.class_names.index(name),name,group_id=patient.get(slide,slide),slide_id=slide))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'min1000_patient_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'patient','groups':len(assign),'g4c_used_as_separate_class':False}}


def oral_oscc(config,seed):
    rows=[]
    for p in sorted(config.root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        parent=p.parent.name.lower()
        name='OSCC' if 'oscc' in parent else ('Normal' if 'normal' in parent else None)
        if name is None:
            continue
        mag='100x' if '100x' in parent else ('400x' if '400x' in parent else 'unknown')
        rows.append(_file_record(p,config.root,config.class_names.index(name),name,magnification=mag))
    counts=_source_counts(rows,config.class_names)
    expected={'Normal':290,'OSCC':934}
    if counts!=expected:
        raise ValueError(f'Unexpected combined Oral OSCC counts: {counts}')
    out=[]; split_counts={}
    for cid,name in enumerate(config.class_names):
        items=[r for r in rows if r['class_name']==name]
        random.Random(seed+cid).shuffle(items)
        n=_split_counts_8_1_1(len(items)); split_counts[name]=n
        a=n['train']; b=a+n['val']
        out.extend({**r,'split':'train'} for r in items[:a])
        out.extend({**r,'split':'val'} for r in items[a:b])
        out.extend({**r,'split':'test'} for r in items[b:])
    return sorted(out,key=lambda r:(r['split'],r['class_id'],r['relative_path'])),{
        'source_image_count':len(rows),'source_class_counts':counts,
        'balancing':{'policy':'whole_dataset_natural_stratified_8_1_1','oversampling':False,'subsampling':False,'per_class_split_counts':split_counts,'magnifications':['100x','400x']}}


def endometrial(config,seed):
    rows=[]
    for name in config.class_names:
        for p in sorted((config.root/name).rglob('*')):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                gid=re.split(r'[-_(]',p.stem)[0]
                rows.append(_file_record(p,config.root,config.class_names.index(name),name,group_id=gid))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'min1000_case_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'case_id','groups':len(assign)}}


def osteosarcoma(config,seed):
    rows=[]; source=list(config.root.rglob('*.jpg'))
    def norm(s): return re.sub(r'[-_\s]+','-',Path(s).name.strip()).lower()
    ambiguous=0
    for cp in sorted(config.root.rglob('PathologistValidation.csv')):
        annotations=defaultdict(set)
        with cp.open(errors='replace',newline='') as f:
            for r in csv.reader(f):
                if len(r)<2 or r[0].strip().lower()=='zannotated': continue
                annotations[norm(r[0])].add(r[1].strip())
        files={norm(p.name):p for p in cp.parent.glob('*.jpg')}
        for key,p in sorted(files.items()):
            labels=annotations.get(key,set())
            if not labels: raise ValueError(f'Missing Osteosarcoma annotation for {p}')
            if len(labels)!=1: raise ValueError(f'Conflicting Osteosarcoma annotations for {p}: {labels}')
            raw=next(iter(labels))
            if raw=='viable: non-viable': ambiguous+=1; continue
            label={'Viable':'Viable-Tumor','Non-Tumor':'Non-Tumor','Non-Viable-Tumor':'Non-Viable-Tumor'}.get(raw)
            if label is None: raise ValueError(f'Unexpected Osteosarcoma label {raw!r}')
            n=norm(p.name)
            if n.startswith('case-48-'): gid='Case-48'
            elif n.startswith('case-4-'): gid='Case-4'
            elif n.startswith('case-3-'): gid='Case-3'
            elif n.startswith('p9-'): gid='P9'
            else: raise ValueError(f'Cannot infer Osteosarcoma patient for {p.name}')
            rows.append(_file_record(p,config.root,config.class_names.index(label),label,group_id=gid))
    if len(rows)!=1091 or ambiguous!=53 or len(source)!=1144:
        raise ValueError(f'Unexpected Osteosarcoma counts: source={len(source)} selected={len(rows)} ambiguous={ambiguous}')
    # Four-patient constraint. Case-4 + P9 is the only two-patient training
    # composition with substantial support for all three classes.
    assign={'Case-4':'train','P9':'train','Case-3':'val','Case-48':'test'}
    out=[{**r,'split':assign[r['group_id']]} for r in rows]
    counts={s:Counter(r['class_name'] for r in out if r['split']==s) for s in _SPLITS}
    if any(counts[s][n]==0 for s in _SPLITS for n in config.class_names):
        raise ValueError(f'Osteosarcoma patient split lost a class: {counts}')
    return sorted(out,key=lambda r:(r['split'],r['class_id'],r['relative_path'])),{
        'source_image_count':len(source),'source_class_counts':_source_counts(rows,config.class_names),
        'balancing':{'policy':'fixed_four_patient_disjoint_split_after_deduplicating_annotations','split_unit':'patient','patient_assignment':assign,'ambiguous_viable_nonviable_excluded':ambiguous,'selected_class_split_counts':{s:dict(counts[s]) for s in _SPLITS}}}


def gashis(config,seed):
    rows=[]; base=config.root/'160'
    for name in config.class_names:
        for p in sorted((base/name).glob('*.png')): rows.append(_file_record(p,config.root,config.class_names.index(name),name,resolution='160'))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets=_plain_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'160px_subdatabase_min1000_then_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'image_WSI_ids_not_released','resolution':160}}


def renalcell(config,seed):
    rows=[]
    for name in config.class_names:
        for p in sorted((config.root/name).glob('*.png')):
            gid='-'.join(p.name.split('_')[0].split('-')[:3])
            rows.append(_file_record(p,config.root,config.class_names.index(name),name,group_id=gid))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'min1000_TCGA_case_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'TCGA_case','groups':len(assign),'pannuke_source_overlap':'PanNuke contains 134 Kidney patches but exposes no TCGA case IDs; exact case overlap cannot be identified from released PanNuke metadata.'}}


def ebhi(config,seed):
    rows=[]
    for name in config.class_names:
        for p in sorted((config.root/name/'image').glob('*.png')):
            parts=p.stem.split('-'); gid='-'.join(parts[:-2])
            rows.append(_file_record(p,config.root,config.class_names.index(name),name,group_id=gid))
    out,assign=_grouped_natural(rows,config.class_names,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':_source_counts(rows,config.class_names),'balancing':{'policy':'whole_dataset_natural_specimen_grouped_approx_8_1_1','oversampling':False,'subsampling':False,'split_unit':'specimen_slide_prefix','groups':len(assign)}}


def _canonical_phash_bits(path:Path)->np.ndarray:
    with Image.open(path) as im: a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
    vals=[]
    for k in range(4):
        b=np.rot90(a,k)
        for c in (b,np.fliplr(b)):
            z=dctn(c,type=2,norm='ortho')[:8,:8]; med=np.median(z.ravel()[1:]); bits=(z>med).astype(np.uint8).ravel(); vals.append(bits)
    return min(vals,key=lambda x:bytes(np.packbits(x)))


def lc25000(config,seed):
    mapping={'colon_aca':config.root/'colon_image_sets/colon_aca','colon_n':config.root/'colon_image_sets/colon_n','lung_aca':config.root/'lung_image_sets/lung_aca','lung_n':config.root/'lung_image_sets/lung_n','lung_scc':config.root/'lung_image_sets/lung_scc'}
    rows=[]; family_meta={}
    for ci,name in enumerate(config.class_names):
        files=sorted(mapping[name].glob('*.jpeg')); X=np.stack([_canonical_phash_bits(p) for p in files]).astype(np.float32)
        km=MiniBatchKMeans(n_clusters=250,random_state=seed+ci,n_init=10,batch_size=1024,max_iter=300).fit(X)
        counts=Counter(int(x) for x in km.labels_); family_meta[name]={'families':250,'min_size':min(counts.values()),'max_size':max(counts.values())}
        for p,lab in zip(files,km.labels_): rows.append(_file_record(p,config.root,ci,name,group_id=f'{name}:family{int(lab):03d}'))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'min1000_reconstructed_augmentation_family_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'reconstructed_augmentation_family','family_reconstruction':'250 families/class via deterministic dihedral-invariant perceptual-hash MiniBatchKMeans; release provides no original-family IDs','family_stats':family_meta}}


def pcam(config,seed):
    import h5py
    xfile=config.root/'camelyonpatch_level_2_split_test_x.h5'; yfile=config.root/'camelyonpatch_level_2_split_test_y.h5'; meta=config.root/'camelyonpatch_level_2_split_test_meta.csv'
    with h5py.File(yfile,'r') as h:
        key=next(iter(h.keys())); labels=np.asarray(h[key]).reshape(-1).astype(int)
    meta_rows=list(csv.DictReader(meta.open()))
    if len(meta_rows)!=len(labels): raise ValueError('PCam metadata/label length mismatch')
    rows=[]
    for i,(lab,m) in enumerate(zip(labels,meta_rows)):
        name=config.class_names[int(lab)]; rows.append(_hdf5_record(xfile,config.root,i,int(lab),name,group_id=str(m['wsi']),wsi=str(m['wsi'])))
    counts=_source_counts(rows,config.class_names); quota=min(1000,min(counts.values())); out,targets,assign=_grouped_balanced(rows,config.class_names,quota,seed)
    return out,{'source_image_count':len(rows),'source_class_counts':counts,'balancing':{'policy':'official_test_pool_min1000_WSI_grouped_8_1_1','per_class_quota':quota,'per_class_split_counts':targets,'split_unit':'WSI','groups':len(assign),'note':'Local PCam copy contains the official 32768-image test pool; this pool is deterministically repartitioned for the downstream probe benchmark.'}}
