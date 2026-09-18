from copy import deepcopy
import numpy as np
import pytest
from lewm.eight_step_planning_development import plan
from lewm.eight_step_planning_goal_probe_development import EightStepPlanningGoalProbe
from lewm.observation_horizon_goal_probe_development import ObservationHorizonGoalProbe
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.tests.test_observation_horizon_goal_selection_development import selection


def checked(prediction=None,cells=()):
    s=selection();s['mode']='WAYPOINT';s['surface_checks']=[dict(possible_intersection=False) for _ in range(6)]
    if prediction is not None:s['prediction']=prediction.tolist()
    return constrain(s,np.zeros(3),np.eye(3),cells)


def test_later_segment_rejects_action_that_passed_first_and_final_endpoints():
    p=np.asarray(selection()['prediction']);p[1,0,0]=.02;p[1,3,0]=.2;p[1,-1,0]=.03
    s=checked(p,[(11,0)]);s['mode']='VIEW_ACQUISITION'
    s['candidates'][1]['utility_m']=100.
    assert s['nominal_action_checks'][1]['nominal_disk_connector_clear']
    frozen=deepcopy(s);r=plan(s,np.zeros(3),np.eye(3),[(11,0)])
    assert r['action']!='forward' and s==frozen
    path=r['nominal_path_checks'][1]
    assert not path['all_predicted_segments_nominally_clear']
    assert any(not x['nominal_disk_connector_clear'] for x in path['segments'][1:-1])
    assert path['segments'][0]['nominal_disk_connector_clear'] and path['segments'][-1]['nominal_disk_connector_clear']
    assert all(x['radius_m']==.45 for row in r['nominal_path_checks'] for x in row['segments'])


def test_waypoint_scores_terminal_forecast_and_retains_surface_and_phase_vetoes():
    p=np.asarray(selection()['prediction']);p[1,-1,0]=.3;p[2,-1,0]=.2
    s=checked(p);r=plan(s,np.zeros(3),np.eye(3),[])
    assert r['action']=='forward' and r['scored_horizon_ns']==800_000_000
    assert r['actual_commitment_horizon_ns']==100_000_000
    s['surface_checks'][1]['possible_intersection']=True
    assert plan(s,np.zeros(3),np.eye(3),[])['action']=='left_arc'
    s['phase_allowed_actions']=['hold']
    assert plan(s,np.zeros(3),np.eye(3),[])['action']=='hold'
    s['surface_checks'][0]['possible_intersection']=True
    stop=plan(s,np.zeros(3),np.eye(3),[])
    assert stop['action'] is None and stop['requested_command']==[0.,0.,0.]


def test_scan_keeps_existing_scores_and_all_eight_clock_checks_are_required():
    s=checked();s['mode']='VIEW_ACQUISITION'
    r=plan(s,np.zeros(3),np.eye(3),[])
    assert r['candidates']==s['candidates']
    for mutate in (lambda x:x.update(first_prediction_horizon_ns=500_000_000),
            lambda x:x['prediction'][1][7].__setitem__(3,0.),
            lambda x:x['nominal_action_checks'][0].__setitem__('radius_m',.4)):
        bad=deepcopy(s);mutate(bad)
        with pytest.raises(ValueError):plan(bad,np.zeros(3),np.eye(3),[])


def test_cadence_sensor_failure_and_arrival_contract_are_inherited():
    assert EightStepPlanningGoalProbe.advance is ObservationHorizonGoalProbe.advance
    c=EightStepPlanningGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    r=c.observe({}, {}, {},now_ns=1)
    assert r['terminal']=='SENSOR_OR_MODEL_FAILURE' and r['requested_command']==[0.,0.,0.]
    assert r['maximum_open_loop_command_ticks']==1 and r['goal_initial_body_xy_m']==[1.2,0.]
