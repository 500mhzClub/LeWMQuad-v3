import copy

import numpy as np
import pytest

from lewm.rgb_body_learning_experiment_development import prediction_metrics
from scripts.audit_go2_rgb_body_learning_comparison_development_v1 import check_metrics,scalar_metrics


def test_scalar_audit_matches_and_rejects_metric_corruption():
    rng=np.random.default_rng(37)
    predictions=rng.normal(size=(10,8,5))
    targets={'motion':rng.normal(size=(10,8,3)),'contact':rng.integers(0,2,size=(10,8)).astype(float),
        'motion_valid':rng.uniform(size=(10,8))>.2,'contact_valid':rng.uniform(size=(10,8))>.1}
    targets['motion'][~targets['motion_valid']]=np.nan
    targets['contact'][~targets['contact_valid']]=np.nan
    metadata=[{'layout_id':f'l{i//5}'} for i in range(10)]
    result=prediction_metrics(predictions,targets,metadata)
    check_metrics(predictions,targets,metadata,result)
    wrong=copy.deepcopy(result); wrong['layout_macro']['contact_brier']+=.01
    with pytest.raises(ValueError,match='macro metric'): check_metrics(predictions,targets,metadata,wrong)


def test_scalar_heading_wrap():
    predictions=np.zeros((1,1,5)); predictions[...,2]=np.sin(-np.pi+.01); predictions[...,3]=np.cos(-np.pi+.01)
    targets={'motion':np.array([[[0.,0.,np.pi-.01]]]),'contact':np.zeros((1,1)),
        'motion_valid':np.ones((1,1),dtype=bool),'contact_valid':np.ones((1,1),dtype=bool)}
    _,macro=scalar_metrics(predictions,targets,[{'layout_id':'a'}])
    assert macro['yaw_error_rad']==pytest.approx(.02)
