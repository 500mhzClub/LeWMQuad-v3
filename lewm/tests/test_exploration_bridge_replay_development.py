import copy

import pytest

from scripts.check_go2_exploration_bridge_replay_development_v1 import compare_selection


def example():
    return {'input_tensor_sha256':{'rgb':'a'},'member_predictions':[[1.,2.]],'selected_action_index':2,
        'initial_direction_xy':[0.,.8],'direction_current_body_xy':[.1,.7],'candidate_costs':[1.,2.],
        'adapter_ms':1.,'inference_ms':.5}


def test_only_declared_bearing_roundoff_and_timing_variation_allowed():
    a=example(); b=copy.deepcopy(a); b['initial_direction_xy'][0]=5e-17; b['adapter_ms']=2.
    assert compare_selection(a,b)==5e-17


@pytest.mark.parametrize('field',['input_tensor_sha256','member_predictions','selected_action_index','candidate_costs','schema'])
def test_meaningful_prediction_input_action_or_schema_change_rejected(field):
    a=example(); b=copy.deepcopy(a)
    if field=='input_tensor_sha256': b[field]['rgb']='b'
    if field=='member_predictions': b[field][0][0]+=1e-14
    if field=='selected_action_index': b[field]=3
    if field=='candidate_costs': b[field][0]+=1e-8
    if field=='schema': b['extra']=1
    with pytest.raises(ValueError): compare_selection(a,b)
