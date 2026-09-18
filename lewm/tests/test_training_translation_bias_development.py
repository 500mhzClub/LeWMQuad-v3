from copy import deepcopy
import numpy as np
import pytest
import torch
from lewm.training_translation_bias_development import fit_translation_bias,correct_arrays,TrainingTranslationBiasModel
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.tests.test_observation_horizon_goal_selection_development import history
from lewm.pulse_timed_training_runner_development import state_digest


def fixture():
    rows=[];active=np.zeros((3,8),bool)
    for i,n in enumerate((8,4,2)):
        active[i,:n]=True
        rows.append(dict(source='family' if i<2 else 'switch',available=True,data_role='train',
            targets=[dict(in_plan=h<n,offset_ns=(h+1)*100_000_000 if h<n else 0,
                motion_valid=h<n,motion=[0.,0.,0.] if h<n else None,contact=0.) for h in range(8)]))
    rows.append(dict(source='family',available=True,data_role='geometry_transfer',targets='MUST NOT READ'))
    p=np.zeros((3,8,5),np.float32)
    for i in range(3):p[i,active[i],0]=(2**i)*.001;p[i,active[i],1]=-.003;p[i,active[i],3]=1.
    arrays=dict(indices=np.arange(3,dtype=np.int64),prediction_valid=active,
        target_offsets_ns=np.where(active,np.arange(1,9)*100_000_000,0).astype(np.int64),direct_outcomes=p)
    schedule=dict(batches=[[0,1,2,0,1,2] for _ in range(1200)])
    return rows,arrays,schedule


def test_exact_draw_and_motion_mask_weighting_ignores_transfer():
    rows,a,s=fixture();r=fit_translation_bias(rows,a,s,head='direct_outcomes')
    np.testing.assert_allclose(r['residual_mean_xy_m'][0],[(300*.001+600*.002+1200*.004)/2100,-.003],rtol=1e-6)
    assert r['training_examples']==3 and r['training_draws']==7200 and r['motion_counts']==[3,3,2,2,1,1,1,1]
    rows[-1]['targets']=None
    assert fit_translation_bias(rows,a,s,head='direct_outcomes')==r
    bad=deepcopy(s);bad['batches'][0][0]=3
    with pytest.raises(ValueError,match='training data'):fit_translation_bias(rows,a,bad,head='direct_outcomes')


def test_correction_preserves_original_arrays_unknown_padding_yaw_and_contact():
    rows,a,s=fixture();original=deepcopy(a);r=fit_translation_bias(rows,a,s,head='direct_outcomes')
    corrected=correct_arrays(a,{'direct_outcomes':r})
    for k in a:np.testing.assert_array_equal(a[k],original[k])
    np.testing.assert_array_equal(corrected['direct_outcomes'][...,2:],a['direct_outcomes'][...,2:])
    assert not np.any(corrected['direct_outcomes'][~a['prediction_valid']])
    bad=deepcopy(a);bad['target_offsets_ns'][0,0]=500_000_000
    with pytest.raises(ValueError):fit_translation_bias(rows,bad,s,head='direct_outcomes')


def test_runtime_numpy_and_torch_correction_match_with_unchanged_base():
    rows,a,s=fixture();receipt=fit_translation_bias(rows,a,s,head='direct_outcomes')
    base=ObservationHorizonRGBBodyJEPA(8).eval();before=state_digest(base.state_dict())
    wrapper=TrainingTranslationBiasModel(base,{'direct_outcomes':receipt});inputs=candidate_inputs(history())
    with torch.inference_mode():raw=base(**inputs);out=wrapper(**inputs)
    arrays=dict(indices=np.arange(6,dtype=np.int64),prediction_valid=raw['prediction_valid'].numpy(),
        target_offsets_ns=raw['target_offsets_ns'].numpy(),direct_outcomes=raw['direct_outcomes'].numpy())
    expected=correct_arrays(arrays,{'direct_outcomes':receipt})
    np.testing.assert_array_equal(out['direct_outcomes'].numpy(),expected['direct_outcomes'])
    torch.testing.assert_close(out['rollout_outcomes'],raw['rollout_outcomes'],rtol=0,atol=0)
    assert state_digest(base.state_dict())==before
    with pytest.raises(ValueError,match='evaluation-only'):wrapper.train()
    assert all(p.grad is None for p in wrapper.parameters())
