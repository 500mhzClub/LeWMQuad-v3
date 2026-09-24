import numpy as np
import pytest

from lewm.actuator_gain_development import configure_gains
from scripts.run_go2_actuator_gain_pair_development_v1 import gain_spec


class Robot:
    def __init__(self):
        self.kp = np.full(12,100.)
        self.kv = np.full(12,10.)
        self.calls = []

    def get_dofs_kp(self, indices):
        return self.kp

    def get_dofs_kv(self, indices):
        return self.kv

    def set_dofs_kp(self, values, indices):
        self.calls.append(('kp',indices))
        self.kp = np.array(values)

    def set_dofs_kv(self, values, indices):
        self.calls.append(('kv',indices))
        self.kv = np.array(values)


@pytest.mark.parametrize('arm', ['default','checkpoint'])
def test_gains_only_intervention_is_explicit_and_read_back(arm):
    robot = Robot()
    indices = list(range(6,18))
    result = configure_gains(robot,indices,{'kp':20.,'kd':.5},arm)
    assert result['before'] == {'kp':[100.]*12,'kv':[10.]*12}
    if arm == 'default':
        assert robot.calls == []
        assert result['effective'] == result['before']
    else:
        assert robot.calls == [('kp',indices),('kv',indices)]
        assert result['effective'] == {'kp':[20.]*12,'kv':[.5]*12}


@pytest.mark.parametrize('indices', [list(range(11)), [6]*12, list(range(-1,11)), [True]+list(range(1,12))])
def test_bad_dof_identity_rejected_before_any_mutation(indices):
    robot = Robot()
    with pytest.raises(ValueError,match='DOFs'):
        configure_gains(robot,indices,{'kp':20.,'kd':.5},'checkpoint')
    assert robot.calls == []


@pytest.mark.parametrize('cfg', [{'kp':10.,'kd':.5},{'kp':20.,'kd':0.},{'kp':float('nan'),'kd':.5}])
def test_wrong_checkpoint_identity_cannot_silently_change_study(cfg):
    robot = Robot()
    with pytest.raises(ValueError,match='checkpoint gains'):
        configure_gains(robot,list(range(6,18)),cfg,'checkpoint')
    assert robot.calls == []


def test_unexpected_native_defaults_fail_closed():
    robot = Robot()
    robot.kp[3] = 99.
    with pytest.raises(ValueError,match='native initial'):
        configure_gains(robot,list(range(6,18)),{'kp':20.,'kd':.5},'checkpoint')
    assert robot.calls == []


def test_ignored_actuator_setting_is_detected():
    robot = Robot()
    robot.set_dofs_kp = lambda values,indices: None
    with pytest.raises(ValueError,match='readback'):
        configure_gains(robot,list(range(6,18)),{'kp':20.,'kd':.5},'checkpoint')


def test_pair_changes_identity_and_gain_arm_not_geometry_or_controller():
    left,right = [gain_spec('left90',.75,arm) for arm in ('default','checkpoint')]
    assert left['geometry'] == right['geometry']
    assert left['procedural_seed'] == right['procedural_seed'] == 2026090804
    assert left['arm'] == right['arm'] == 'baseline'
    assert left['scene_id'] != right['scene_id']
