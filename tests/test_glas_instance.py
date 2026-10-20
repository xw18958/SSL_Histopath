"""GlaS native instance-mask and metric regression checks; test without GPU."""
import importlib.util,sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"run_glas_instance.py"
spec=importlib.util.spec_from_file_location("run_glas_instance",SCRIPT)
model=importlib.util.module_from_spec(spec);sys.modules["run_glas_instance"]=model;spec.loader.exec_module(model)

def two_instances():
    a=np.zeros((112,144),dtype=np.int32)
    a[12:52,10:60]=1
    a[60:105,72:129]=2
    return a

def test_native_instances_different_ids():
    x=two_instances()
    dist=model.object_distance(x)
    assert dist.shape==x.shape
    assert dist[x==0].max()==0 and dist[x==1].max()==1 and dist[x==2].max()==1
    assert len(np.unique(x[x>0]))==2

def test_identical_perfect_glands():
    x=two_instances()
    m=model.instance_metrics(x,x,True)
    assert m["predicted_glands"]==m["gt_glands"]==2
    assert abs(m["object_f1"]-1)<1e-12
    assert abs(m["object_dice"]-1)<1e-12
    assert abs(m["object_hausdorff"])<1e-12

def test_empty_predictions_penalized():
    x=two_instances()
    y=np.zeros_like(x)
    m=model.instance_metrics(y,x,True)
    assert m["object_f1"]==0 and m["object_dice"]==0
    assert m["object_hausdorff"]>0

def test_instance_merging_is_not_perfect():
    x=two_instances()
    pred=(x>0).astype(np.int32)
    met=model.instance_metrics(pred,x,True)
    assert met["object_f1"]<1
    assert met["object_dice"]<1

def test_small_object_filter():
    x=two_instances()
    x[0:2,0:2]=3
    p=model.relabel_small(x,min_area=8)
    assert set(np.unique(p))=={0,1,2}

def test_watershed_returns_distinct_instances():
    x=two_instances()
    fg=(x>0).astype(np.float32)
    dist=model.object_distance(x)
    got=model.watershed_instances(fg,dist,peak_threshold=.25,min_area=40)
    assert got.dtype==np.int32
    assert np.unique(got).size>=3
    assert np.all(got[~(x>0)]==0)

def test_tiling_covers_native_image():
    h,w=522,775
    ys=model.axis_positions(h)
    xs=model.axis_positions(w)
    cover=np.zeros((h,w),np.int32)
    for y in ys:
        for x in xs:cover[y:y+256,x:x+256]+=1
    assert np.all(cover>=1)
    assert ys[-1]==h-256 and xs[-1]==w-256

def test_linear_head_uses_spatial_features():
    h=torch.nn.Conv2d(768,2,1)
    y=model.head_logits(h,torch.zeros(2,64,768))
    assert y.shape==(2,2,256,256)
    assert sum(p.numel() for p in h.parameters())==1538
