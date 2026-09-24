from copy import deepcopy
import math

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemory


def selection(heading, action='right_turn', left_clear=True):
    return dict(action=action, before_memory_filter_action=action,
        waypoint_body_xy_m=[math.cos(-heading), math.sin(-heading)],
        memory_forecast_candidates=[dict(action=a,
            nominal_predicted_path_clear=left_clear if a=='left_turn' else True,
            reserve_recovery_path_clear=False) for a in ACTIONS])


def interrupted():
    memory=InterruptedRouteTurnMemory()
    memory.select(selection(2.),[0.,0.],2.,1,None)
    recovery=selection(1.8,'left_turn');recovery['scan_utilities']=[]
    assert memory.select(recovery,[0.,0.],1.8,1,100) is recovery
    return memory


def test_measured_interruption_tries_and_keeps_other_clear_direction():
    memory=interrupted();original=selection(2.);before=deepcopy(original)
    result=memory.select(original,[0.,0.],2.,1,None)
    assert result['action']=='left_turn' and original==before
    # A later parent choice favoring the shorter right turn cannot undo this
    # alternate attempt before its measured heading is reached.
    assert memory.select(selection(2.2),[0.,0.],2.2,1,None)['action']=='left_turn'
    # Existing clearance can stop it; memory never grants a blocked turn.
    assert memory.select(selection(2.3,left_clear=False),[0.,0.],2.3,1,None)['action']=='hold'
    recovery=selection(2.4,'right_turn');recovery['scan_utilities']=[]
    assert memory.select(recovery,[0.,0.],2.4,1,200) is recovery
    assert {r['direction'] for r in memory.failed}=={-1,1}
    # Both observed directions failing is not evidence for inventing a third
    # clear route or suppressing measured visual recovery.
    result=memory.select(selection(2.),[0.,0.],2.,1,None)
    assert 'visual_route_turn_memory' not in result


def test_scope_motion_and_current_clearance_required():
    memory=InterruptedRouteTurnMemory()
    memory.select(selection(2.),[0.,0.],2.,1,None)
    recovery=selection(2.,'left_turn')
    memory.select(recovery,[0.,0.],2.,1,100)
    assert not memory.failed  # A requested but unmeasured turn is insufficient.
    for position,generation,left_clear in (([.3,0.],1,True),([0.,0.],2,True),([0.,0.],1,False)):
        memory=interrupted();original=selection(2.,left_clear=left_clear)
        assert memory.select(original,position,2.,generation,None) is original


def test_measured_heading_or_translation_releases_alternate_attempt():
    memory=interrupted()
    memory.select(selection(2.),[0.,0.],2.,1,None)
    result=memory.select(selection(.05,'hold'),[0.,0.],.05,1,None)
    assert result['action']=='hold' and memory.active is None
    memory=interrupted()
    memory.select(selection(2.),[0.,0.],2.,1,None)
    result=memory.select(selection(2.1,'forward'),[0.,0.],2.1,1,None)
    assert result['action']=='forward' and memory.active is None
