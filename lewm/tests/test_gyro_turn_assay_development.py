import math
import numpy as np
import pytest
from lewm.gyro_turn_assay_development import timed_decision,reduce_turn
from scripts.run_go2_gyro_turn_assay_development_v1 import trials


def test_trial_population_and_pairs():
    rows=trials(); assert len(rows)==18 and len({r['scene_id'] for r in rows})==18
    for i in range(0,18,2):
        assert rows[i]['geometry']==rows[i+1]['geometry']
        assert rows[i]['procedural_seed']==rows[i+1]['procedural_seed']
    assert rows==trials()


@pytest.mark.parametrize('target',[math.pi/2,-math.pi/2,math.pi])
def test_timed_turn_has_fixed_turn_and_dwell(target):
    count=math.ceil(abs(target)/.035)
    assert timed_decision(count-1,target)['requested_command'][2]==math.copysign(.35,target)
    assert timed_decision(count,target)['status']=='SETTLING_TIMED'
    assert timed_decision(count+3,target)['status']=='COMPLETE_TIMED'


def test_physical_endpoint_not_timed_schedule_is_success():
    raw={'timestamp_s':np.arange(500)*.002,'base_pose_world':np.tile([0.,0.,.3,0.,0.,0.,1.],(500,1)),
        'base_twist_world':np.zeros((500,6)),'phase':np.array([1]*250+[2]*250),'physics_contact':np.zeros(500,dtype=bool)}
    result=reduce_turn(raw,0,[],math.pi/2,'COMPLETE_TIMED',None)
    assert not result['physical_task_success'] and not result['checks']['heading_error_at_most_0p12']
    raw['base_pose_world'][1:,5:]=[math.sin(math.pi/4),math.cos(math.pi/4)]
    assert reduce_turn(raw,0,[],math.pi/2,'COMPLETE_TIMED',None)['physical_task_success']
    raw['physics_contact'][-1]=True
    assert not reduce_turn(raw,0,[],math.pi/2,'COMPLETE_TIMED','DISALLOWED_CONTACT')['physical_task_success']
