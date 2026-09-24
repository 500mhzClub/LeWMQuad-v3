import ast
from pathlib import Path
from lewm.commitment_pose_goal_probe_development import apply_waypoint_utility, CommitmentPoseGoalProbe
from lewm.tests.test_commitment_pose_waypoint_utility_development import fixture


def test_original_surface_conflict_veto_survives_new_alignment_reward():
    selection=fixture([0,1])
    selection['surface_checks']=[dict(possible_intersection=i==4) for i in range(6)]
    result=apply_waypoint_utility(selection)
    assert result['action']!='left_turn' and result['original_surface_conflicts_preserved']
    assert result['surface_checks']==selection['surface_checks']
    for c in selection['surface_checks']:c['possible_intersection']=True
    assert apply_waypoint_utility(selection)['action'] is None


def test_failure_latch_and_goal_are_inherited():
    controller=CommitmentPoseGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    row=controller.observe({}, {}, {},now_ns=1)
    assert row['controller']=='commitment_pose_goal_probe_v1'
    assert row['terminal']=='SENSOR_OR_MODEL_FAILURE' and row['requested_command']==[0.,0.,0.]
    assert row['goal_initial_body_xy_m']==[1.2,0.]
    assert controller.observe({}, {}, {},now_ns=2)['terminal']==row['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import commitment_pose_goal_audit_development as new
    from scripts import matched_model_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)
