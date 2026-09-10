import pytest
from pannuke_ssl.ssl_framework import EarlyStopper, load_standard_config, load_tuning_spec


def test_safe_early_stop_gate():
    s=EarlyStopper(enabled=True,min_epochs=60,patience=5,delta=.005); assert s.update(10,.5)["improved"]
    for e in (20,30,40,50,60): assert s.update(e,.503)["stale_monitors"]==0
    for i,e in enumerate((70,80,90,100),1):
        u=s.update(e,.503); assert u["stale_monitors"]==i and not u["should_stop"]
    assert s.update(110,.503)["should_stop"]


def test_delta_improvement_resets_patience():
    s=EarlyStopper(enabled=True,min_epochs=60,patience=5,delta=.005); s.update(10,.5); s.update(70,.502); u=s.update(80,.505); assert u["improved"] and u["best_score"]==pytest.approx(.505) and u["stale_monitors"]==0


def test_source_learning_rates_must_be_in_tuning_grid():
    expected={"ijepa":.001,"lejepa":.0005}
    for name,source in expected.items():
        load_standard_config(name); spec=load_tuning_spec(name); p=spec["parameters"]["learning_rate"]; assert float(p["source_value"])==source and source in [float(x) for x in p["candidates"]]


def test_ijepa_source_eval_and_ema_metadata():
    c=load_standard_config("ijepa"); meta=c["method"]["source_metadata"]
    assert meta["evaluation_encoder"]=="target_encoder_ema"
    assert meta["evaluation_pooling"]=="average_patch_tokens"
    assert meta["source_ema"]==[.996,1.0]
    assert meta["source_ema_schedule"]=="linear"


def test_common_protocol_matches_across_methods():
    a,b=load_standard_config("ijepa"),load_standard_config("lejepa")
    paths=[("seed",),("data","expected_ssl_images"),("training","max_epochs"),("training","batch_size"),("validation","selection_metric"),("validation","interval_epochs"),("early_stopping","min_epochs"),("early_stopping","patience_monitors"),("downstream","train_count"),("downstream","validation_count"),("downstream","test_count")]
    for path in paths:
        x,y=a,b
        for k in path: x,y=x[k],y[k]
        assert x==y
