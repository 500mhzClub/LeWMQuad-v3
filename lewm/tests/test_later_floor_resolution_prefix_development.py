from copy import deepcopy
import pytest
from scripts.replay_go2_later_floor_resolution_maze_prefix_v1 import compare_current


def pair():
    old=dict(failure=None,evidence={'pose':1},original_visual_evidence={'raw':2},memory_receipt={'map':3},
        mission_receipt={'phase':'OUTBOUND'},requested_command=[0.,0.,0.],terminal=None,
        new_selection=dict(prediction=[1,2],surface_checks=[{'possible_intersection':True}],nominal_path_checks=[True]))
    new=deepcopy(old);new['requested_command']=[.2,0.,0.]
    new['new_selection']['surface_checks']=[dict(possible_intersection=False,
        original_contact_check_before_later_floor_resolution=deepcopy(old['new_selection']['surface_checks'][0]))]
    return old,new


def test_declared_contact_change_preserves_original_evidence():
    old,new=pair()
    assert compare_current(old,new)==dict(command_changed=True,terminal_changed=False)


@pytest.mark.parametrize('fault',['failure','pose','map','mission','prediction','contact','nominal'])
def test_undeclared_changes_rejected(fault):
    old,new=pair()
    if fault=='failure':new['failure']='missing evidence'
    elif fault=='pose':new['evidence']['pose']=2
    elif fault=='map':new['memory_receipt']['map']=4
    elif fault=='mission':new['mission_receipt']['phase']='RETURN'
    elif fault=='prediction':new['new_selection']['prediction']=[2,3]
    elif fault=='contact':new['new_selection']['surface_checks'][0]['original_contact_check_before_later_floor_resolution']={}
    else:new['new_selection']['nominal_path_checks']=[False]
    with pytest.raises(ValueError):compare_current(old,new)
