import ast
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from lewm.active_view_waypoint_selection_development import scan_rank, restrict
from lewm.active_view_goal_probe_development import ActiveViewGoalProbe
from lewm.geometry_progress_predictive_selection_development import score_candidates


def fixture():
    p=np.zeros((6,8,5),float);p[:,:,3]=1.;p[:,:,4]=-10.
    for index,yaw in ((4,.2),(5,-.2),(2,.3),(3,-.3)):
        p[index,:,2]=np.sin(yaw);p[index,:,3]=np.cos(yaw)
    result=score_candidates(p,goal_body_xy_m=[0.,0.],contact_penalty_m=1.2)|dict(prediction=p.tolist())
    result['surface_checks']=[dict(possible_intersection=False) for _ in range(6)]
    return result


@pytest.mark.parametrize('heading,expected',[(.785,'left_turn'),(-.785,'right_turn')])
def test_learned_yaw_selects_corresponding_turn_and_scan_excludes_arcs(heading,expected):
    source=fixture();before=deepcopy(source)
    ranked=scan_rank(source,heading)
    result=restrict(ranked,('hold','left_turn','right_turn'))
    assert result['action']==expected and result['requested_command'][0]==0.
    assert source==before and result['prediction']==source['prediction']


def test_surface_conflicts_cannot_be_overridden_by_scan_heading_reward():
    source=fixture()
    for i in (0,4,5):source['surface_checks'][i]['possible_intersection']=True
    row=restrict(scan_rank(source,.785),('hold','left_turn','right_turn'))
    assert row['action'] is None and row['requested_command']==[0.,0.,0.]


def test_bad_sensor_failure_latches_and_keeps_original_mission_goal():
    p=ActiveViewGoalProbe(object(),object(),persistent=True)
    row=p.observe({}, {}, {},now_ns=1)
    assert row['terminal']=='SENSOR_OR_MODEL_FAILURE' and row['requested_command']==[0.,0.,0.]
    assert row['goal_initial_body_xy_m']==[1.2,0.]
    assert p.observe({}, {}, {},now_ns=2)['terminal']==row['terminal']


def test_native_goal_and_actuator_audits_remain_unchanged():
    from scripts import active_view_goal_audit_development as new
    from scripts import surface_memory_goal_audit_development as old
    def dump(module,name):
        t=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert dump(new,name)==dump(old,name)
