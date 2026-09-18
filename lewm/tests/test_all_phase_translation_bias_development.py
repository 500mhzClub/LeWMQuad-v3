from copy import deepcopy
import numpy as np
import pytest
import torch
from lewm import all_phase_translation_bias_development as mod
from lewm.training_translation_bias_development import fit_translation_bias as old_fit
from lewm.tests.test_training_translation_bias_development import fixture
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.tests.test_observation_horizon_goal_selection_development import history
from lewm.pulse_timed_training_runner_development import state_digest


def data():
    rows,arrays,schedule = fixture()
    for row in rows[:-1]:
        for target in row['targets']: target['contact_valid']=target['in_plan']
    return rows,arrays,schedule


def test_original_supported_estimator_remains_exact():
    rows,arrays,schedule=data(); old=old_fit(rows,arrays,schedule,head='direct_outcomes')
    actual=mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')
    for key in ('motion_counts','effective_horizon_weights','residual_mean_xy_m','applied_bias_xy_m'):
        assert actual[key]==old[key]
    assert actual['motionless_training_examples']==0 and actual['native_training_targets_used']


def test_motionless_context_is_accounted_without_dropping_it_or_using_transfer_labels():
    rows,arrays,schedule=data()
    for target in rows[2]['targets']:
        target['motion_valid']=False;target['motion']=None
    arrays['direct_outcomes'][2,arrays['prediction_valid'][2],0]=999.
    actual=mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')
    assert actual['training_examples']==3 and actual['training_draws']==7200
    assert actual['motionless_training_indices']==[2] and actual['motionless_training_draws']==2400
    assert actual['motion_contributing_examples']==2
    np.testing.assert_allclose(actual['residual_mean_xy_m'][0],[(300*.001+600*.002)/900,-.003],rtol=1e-6)
    rows[-1]['targets']=object()
    assert mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')==actual
    arrays['target_offsets_ns'][2,0]+=1
    with pytest.raises(ValueError):mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')


@pytest.mark.parametrize('fault',['transfer_draw','omitted_context','contact_motion','unsupported_horizon'])
def test_invalid_training_scope_or_motion_support_is_rejected(fault):
    rows,arrays,schedule=data()
    if fault=='transfer_draw':schedule['batches'][0][0]=3
    elif fault=='omitted_context':schedule['batches']=[[0]*6 for _ in range(1200)]
    elif fault=='contact_motion':rows[0]['targets'][0]['contact']=1.
    elif fault=='unsupported_horizon':
        for row in rows[:-1]:row['targets'][-1]['motion_valid']=False
    with pytest.raises(ValueError):mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')


def test_numpy_and_runtime_corrections_match_and_preserve_base_yaw_contact_and_padding():
    rows,arrays,schedule=data();receipt=mod.fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')
    base=ObservationHorizonRGBBodyJEPA(8).eval();before=state_digest(base.state_dict())
    wrapper=mod.AllPhaseTranslationBiasModel(base,{'direct_outcomes':receipt});inputs=candidate_inputs(history())
    with torch.inference_mode():raw=base(**inputs);out=wrapper(**inputs)
    arrays=dict(indices=np.arange(6,dtype=np.int64),prediction_valid=raw['prediction_valid'].numpy(),
        target_offsets_ns=raw['target_offsets_ns'].numpy(),direct_outcomes=raw['direct_outcomes'].numpy())
    expected=mod.correct_arrays(arrays,{'direct_outcomes':receipt})
    np.testing.assert_array_equal(out['direct_outcomes'].numpy(),expected['direct_outcomes'])
    torch.testing.assert_close(out['direct_outcomes'][...,2:],raw['direct_outcomes'][...,2:],rtol=0,atol=0)
    torch.testing.assert_close(out['rollout_outcomes'],raw['rollout_outcomes'],rtol=0,atol=0)
    assert state_digest(base.state_dict())==before
    with pytest.raises(ValueError):wrapper.train()
