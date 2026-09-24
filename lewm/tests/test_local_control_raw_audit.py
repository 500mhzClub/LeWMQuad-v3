import copy

import numpy as np
import pytest

from lewm.local_execution_controller_development import trial_spec
from scripts.audit_go2_local_control_factorial_development_v1 import audit_decisions


def fixture():
    spec = trial_spec('straight', .75, 'baseline')
    count = 800
    arrays = {'timestamp_s': .002*np.arange(1,count+1),
        'base_pose_world':np.tile([0,0,.3,0,0,0,1.],(count,1)),
        'base_twist_world':np.zeros((count,6)),
        'phase':np.array([0]*750+[1]*50), 'edge_index':np.zeros(count,dtype=int),
        'applied_command':np.array([[0,0,0]]*750+[[.25,0,0]]*50),
        'requested_command':np.array([[0,0,0]]*750+[[.25,0,0]]*50)}
    decisions = [{'edge_index':0,'pre_sample_index':749,'timestamp_s':1.5,'stage':'APPROACH',
        'inputs':dict(tick=0,alignment_error=0.,pursuit_error=0.,arrival_error=0.,
            body_forward_velocity=0.,angular_velocity=0.,crossed=False,stable_arrival=False),
        'requested_command':[.25,0.,0.]}]
    edges = [{'geometry':spec['geometry'],'controller_terminal_reason':None,'terminal_global_sample_index':799}]
    return arrays, decisions, spec, edges


def test_audit_accepts_complete_causal_tick():
    audit_decisions(*fixture())


@pytest.mark.parametrize('fault', ['future','skip','command','phase','slew','unexplained'])
def test_audit_rejects_causal_or_execution_corruption(fault):
    arrays, decisions, spec, edges = copy.deepcopy(fixture())
    if fault == 'future':
        decisions[0]['inputs']['crossed'] = True
    elif fault == 'skip':
        decisions[0]['pre_sample_index'] = 750
    elif fault == 'command':
        arrays['requested_command'][799,0] = .2
    elif fault == 'phase':
        arrays['phase'][799] = 2
    elif fault == 'slew':
        arrays['applied_command'][799,0] = -.25
    elif fault == 'unexplained':
        arrays = {key:np.concatenate([value,value[-1:]]) for key,value in arrays.items()}
    with pytest.raises(ValueError):
        audit_decisions(arrays, decisions, spec, edges)
