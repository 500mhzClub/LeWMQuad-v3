from copy import deepcopy
import pytest
from lewm.ensemble_observation_replan_goal_probe_development import combine
from lewm.geometry_progress_pilot_development import ACTIONS


def member():
    return dict(proposal=dict(route_cells=[[0,0]]),mode='WAYPOINT',view_budget_exhausted=False,
        phase_allowed_actions=list(ACTIONS),head='direct_outcomes',input_variant='full',goal_body_xy_m=[1,0],
        prediction=[[[0,0,0,1,0] for _ in range(8)] for _ in ACTIONS],
        candidates=[dict(action=a,utility_m=float(i)) for i,a in enumerate(ACTIONS)],
        surface_checks=[dict(possible_intersection=False) for _ in ACTIONS],
        nominal_action_checks=[dict(action=a,nominal_disk_connector_clear=True) for a in ACTIONS])


def test_one_members_conflict_cannot_be_hidden_by_mean_or_majority():
    rows=[member() for _ in range(3)];original=deepcopy(rows)
    rows[1]['nominal_action_checks'][5]['nominal_disk_connector_clear']=False
    rows[2]['surface_checks'][4]['possible_intersection']=True
    result=combine(rows)
    assert result['action']==ACTIONS[3]
    assert result['ensemble_candidate_checks'][5]['member_feasible']==[True,False,True]
    assert result['phase_admissible_candidates']==4
    assert rows[0]==original[0]
    for i in range(4):rows[0]['surface_checks'][i]['possible_intersection']=True
    assert combine(rows)['action'] is None


def test_mean_utility_is_used_only_after_all_member_admission():
    rows=[member() for _ in range(3)]
    for r in rows:r['candidates'][0]['utility_m']=6.
    rows[2]['candidates'][5]['utility_m']=50.
    assert combine(rows)['action']==ACTIONS[5]
    rows[2]['nominal_action_checks'][5]['nominal_disk_connector_clear']=False
    assert combine(rows)['action']==ACTIONS[0]
    rows[2]['goal_body_xy_m']=[0,1]
    with pytest.raises(ValueError,match='same observed'):combine(rows)
    with pytest.raises(ValueError,match='three'):combine(rows[:2])
