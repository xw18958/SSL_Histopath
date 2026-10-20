"""Publication-protocol checks for official BCSS five-class frozen linear probe."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]/"scripts"

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+".py"))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

prep=load("prepare_bcss5_official")
run=load("run_bcss5_official")

def test_official_ontology():
    assert len(prep.CLASSES)==5
    assert prep.LUT[0]==255 and prep.LUT[7]==255 and prep.LUT[15]==255
    assert [prep.LUT[x] for x in (1,19,20)]==[0,0,0]
    assert prep.LUT[2]==1
    assert [prep.LUT[x] for x in (3,10,11)]==[2,2,2]
    assert prep.LUT[4]==3
    assert all(prep.LUT[x]==4 for x in [5,6,8,9,12,13,14,16,17,18,21])
    assert set(prep.GROUPS)==set(range(1,22))-{7,15}

def test_institutions():
    for p in prep.TEST:assert prep.split(p)=="test"
    for p in prep.VAL:assert prep.split(p)=="val"
    assert prep.TEST.isdisjoint(prep.VAL)
    for p in ("A2","A7","AR","D8"):assert prep.split(p)=="train"

def test_resize_preserves_five_labels():
    a=np.zeros((256,256),np.uint8)
    for c in range(5):
        a[:,c*50:(c+1)*50]=c
    a[:8,:]=255
    image=Image.fromarray(np.zeros((256,256,3),dtype=np.uint8))
    _,lab=run.prep_pair(image,Image.fromarray(a),"bcss5")
    assert set(np.unique(lab.numpy()))=={0,1,2,3,4,255}
    assert lab.shape==(256,256)

def test_multiclass_confusion_and_metrics():
    y=np.array([0,1,2,3,4,255])
    p=np.array([0,1,0,4,3,4])
    cm=run.score(p,y)
    assert cm.shape==(5,5)
    assert cm.sum()==5
    assert cm[0,0]==1 and cm[2,0]==1 and cm[3,4]==1 and cm[4,3]==1
    m=run.summary(cm)
    assert m["evaluated_classes"]==[0,1,2,3,4]
    assert abs(m["miou"]-((1/2+1+0+0+0)/5))<1e-12
    assert m["valid_pixels"]==5

def test_head_really_has_five_channels():
    import torch
    h=torch.nn.Conv2d(768,5,1)
    assert sum(p.numel() for p in h.parameters())==3845
    assert run.logits(h,torch.zeros(2,64,768)).shape==(2,5,256,256)

def test_incomplete_classes_are_not_inflated():
    cm=np.zeros((5,5),np.int64);cm[0,0]=50
    m=run.summary(cm)
    assert m["evaluated_classes"]==[0] and m["miou"]==1
    assert len(m["per_class_iou"])==5
    assert np.all(np.isfinite(np.array([x for x in m["per_class_iou"] if x is not None])))
