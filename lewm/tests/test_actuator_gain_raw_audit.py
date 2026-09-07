import copy

import pytest

from scripts.audit_go2_actuator_gain_pair_development_v1 import audit_identity


def identity():
    return {'arm':'checkpoint','dof_indices_rollout_order':list(range(6,18)),
        'before':{'kp':[100.]*12,'kv':[10.]*12},
        'effective':{'kp':[20.]*12,'kv':[.5]*12},'expected_checkpoint':{'kp':20.,'kv':.5},
        'initial_state_before_intervention':{'position':[[0,0,.375]],'quaternion_wxyz':[[1,0,0,0]],
            'velocity':[[0,0,0]],'angular_velocity':[[0,0,0]],'joint_position':[[0]*12],'joint_velocity':[[0]*12]}}


def test_valid_native_identity_readback_passes():
    value=identity()
    assert audit_identity(value,value['effective'],'checkpoint')==value['initial_state_before_intervention']


@pytest.mark.parametrize('fault',['arm','joint','initial_gain','effective','terminal','nonfinite'])
def test_identity_corruption_is_rejected(fault):
    value=identity()
    terminal=copy.deepcopy(value['effective'])
    if fault=='arm':
        value['arm']='default'
    elif fault=='joint':
        value['dof_indices_rollout_order'][1]=6
    elif fault=='initial_gain':
        value['before']['kp'][0]=20.
    elif fault=='effective':
        value['effective']['kv'][0]=10.
    elif fault=='terminal':
        terminal['kv'][0]=10.
    else:
        value['initial_state_before_intervention']['position'][0][0]=float('nan')
    with pytest.raises(ValueError):
        audit_identity(value,terminal,'checkpoint')
