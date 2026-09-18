import ast
from pathlib import Path
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.nominal_action_constraint_development import constrain
from lewm.nominal_action_goal_probe_development import NominalActionGoalProbe


def fixture():
    predictions=np.zeros((6,8,5));predictions[:,:,3]=1.
    return dict(prediction=predictions.tolist(),candidates=[dict(action=a,utility_m=float(6-i)) for i,a in enumerate(ACTIONS)],
        surface_checks=[dict(possible_intersection=False) for _ in ACTIONS],phase_allowed_actions=list(ACTIONS),
        phase_admissible_candidates=6,action='hold')


def test_nominal_conflict_rejects_highest_utility_and_preserves_original_forecast():
    s=fixture();s['prediction'][0][0][0]=.11
    result=constrain(s,[0,0,0],np.eye(3),{(11,0)})
    assert result['action']=='forward' and s['action']=='hold'
    assert result['prediction']==s['prediction'] and result['surface_checks']==s['surface_checks']
    assert not result['nominal_action_checks'][0]['nominal_disk_connector_clear']
    assert result['phase_admissible_candidates']==5 and result['all_current_occupied_squares_checked']
    assert not result['model_error_bound_applied']


def test_all_occupied_squares_rotation_and_phase_surface_vetoes_are_applied():
    s=fixture();s['phase_allowed_actions']=['hold','left_turn','right_turn']
    s['surface_checks'][0]['possible_intersection']=True
    s['prediction'][4][0][0]=.2
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    result=constrain(s,[0,0,0],R,{(50,50),(0,10)})
    assert result['action']=='right_turn'
    assert result['nominal_action_checks'][4]['nearest_observed_cell']==[0,10]


def test_infeasible_start_does_not_silently_waive_radius_for_escape():
    result=constrain(fixture(),[.11,0,0],np.eye(3),{(11,0)})
    assert result['action'] is None and result['requested_command']==[0.,0.,0.]
    assert result['phase_admissible_candidates']==0


def test_failure_latch_and_goal_remain_inherited():
    controller=NominalActionGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    row=controller.observe({}, {}, {},now_ns=1)
    assert row['controller']=='nominal_action_goal_probe_v1' and row['terminal']=='SENSOR_OR_MODEL_FAILURE'
    assert row['requested_command']==[0.,0.,0.] and row['goal_initial_body_xy_m']==[1.2,0.]
    assert controller.observe({}, {}, {},now_ns=2)['terminal']==row['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import nominal_action_goal_audit_development as new
    from scripts import continuous_connector_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)
