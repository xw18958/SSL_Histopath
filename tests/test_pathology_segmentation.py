"""Regression checks for the isolated frozen pathology semantic segmentation downstream."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

SRC=Path(__file__).resolve().parents[1]/"scripts"

def module(filename):
    spec=importlib.util.spec_from_file_location(filename,SRC/(filename+".py"))
    obj=importlib.util.module_from_spec(spec)
    sys.modules[filename]=obj
    spec.loader.exec_module(obj)
    return obj

prep=module("prepare_pathology_segmentation")
run=module("run_pathology_segmentation")

def test_split_group_stability_and_disjoint():
    g=["patient"+str(i) for i in range(100)]
    a=prep.grouped_split(g)
    assert a==prep.grouped_split(list(reversed(g)))
    assert set(a.values())=={"train","val","test"}
    assert 60<=sum(x=="train" for x in a.values())<=80

def test_masks_and_rgb_aligned():
    img=Image.fromarray(np.tile(np.arange(224,dtype=np.uint8)[None,:,None],(224,1,3)))
    mask=Image.fromarray(np.full((224,224),255,dtype=np.uint8))
    rgb,lab=run.prep_pair(img,mask,"ebhi")
    assert rgb.shape==(3,256,256) and lab.shape==(256,256)
    assert set(np.unique(lab.numpy()))=={1}
    glas=Image.fromarray(np.zeros((224,224),dtype=np.uint8))
    _,zeros=run.prep_pair(img,glas,"glas")
    assert zeros.max()==0
    invalid=Image.fromarray(np.full((224,224),2,dtype=np.uint8))
    with pytest.raises(ValueError):run.prep_pair(img,invalid,"bcss")

def test_metrics_ignore_and_independent_reference():
    gt=np.array([[1,0,255],[1,0,255]],dtype=np.uint8)
    pr=np.array([[1,0,1],[0,1,0]],dtype=np.uint8)
    cm=run.score(pr,gt)
    assert cm.tolist()==[[1,1],[1,1]]
    s=run.summary(cm)
    assert abs(s["miou"]-1/3)<1e-9
    assert abs(s["foreground_dice"]-.5)<1e-9
    assert s["valid_pixels"]==4

def test_grouping_violation_fails(tmp_path):
    (tmp_path/"im.png").touch()
    (tmp_path/"mask.png").touch()
    r=[{"image":"im.png","mask":"mask.png","group":"g","split":"train"},
       {"image":"mask.png","mask":"im.png","group":"g","split":"test"}]
    with pytest.raises(ValueError,match="group crosses"):prep.validate(r,tmp_path)

def test_manifest_sha_and_mutation_failure(tmp_path):
    records=[{"image":str(i),"mask":str(i),"group":str(i),"split":split}
             for i,split in enumerate(("train","val","test"))]
    content={"version":"pathology_semantic_seg_v1","dataset":"toy","records":records}
    import hashlib,json
    hash=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    content["manifest_sha256"]=hash
    p=tmp_path/"test.json";p.write_text(json.dumps(content))
    assert run.manifest_load(p)["manifest_sha256"]==hash
    content["records"][0]["split"]="val"
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError,match="checksum"):run.manifest_load(p)

def test_linear_head_has_only_two_class_output():
    import torch
    head=torch.nn.Conv2d(768,2,1)
    with torch.no_grad():
        z=run.logits(head,torch.zeros(2,64,768))
    assert z.shape==(2,2,256,256)
    assert sum(p.numel() for p in head.parameters())==1538
