from copy import deepcopy
import numpy as np
import pytest
import torch
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.observation_horizon_predictive_selection_development import candidate_inputs,select
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_waypoint_utility_development import score_commitment_pose
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.observation_horizon_goal_probe_development import ObservationHorizonGoalProbe
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe


def history():
    return {k:torch.zeros(shape) for k,shape in dict(rgb=(4,3,96,128),body=(4,20,63),control=(4,15,7)).items()}


def selection():
    p=np.zeros((6,8,5));p[:,:,3]=1.;p[:,:,4]=-5.
    p[1,0,0]=.05;p[2,0,0]=.03
    return dict(prediction=p.tolist(),first_prediction_horizon_ns=100_000_000,prediction_horizon_ns=800_000_000,
        target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)),
        candidates=[dict(action=a,utility_m=float(i)) for i,a in enumerate(ACTIONS)],goal_body_xy_m=[1,0],
        action='right_turn',phase_allowed_actions=list(ACTIONS),phase_admissible_candidates=6)


class Memory:
    def footprint(self,geometry,xy,yaw,**kwargs):return dict(possible_intersection=xy[0]>.04)


def test_candidate_forecasts_use_actual_eight_command_prefixes_and_short_clocks():
    inputs=candidate_inputs(history())
    assert inputs['known_action_blocks'].shape==(6,8,1,3)
    for i,a in enumerate(ACTIONS):
        torch.testing.assert_close(inputs['known_action_blocks'][i,:,0],
            torch.tensor(candidate_commands(a)[:8])/torch.tensor([.3,1.,.5]),rtol=0,atol=0)
    result=select(ObservationHorizonRGBBodyJEPA(8).eval(),history(),head='direct_outcomes',
        input_variant='full',goal_body_xy_m=[1,0],contact_penalty_m=1.2)
    assert result['first_prediction_horizon_ns']==100_000_000 and result['prediction_horizon_ns']==800_000_000
    assert result['target_offsets_ns']==list(range(100_000_000,800_000_001,100_000_000))


def test_surface_veto_and_nominal_radius_survive_shorter_scoring():
    s=score_commitment_pose(selection());assert s['action']=='forward' and s['scored_horizon_ns']==100_000_000
    s=filter_selection(s,Memory(),object(),now_ns=1,persistent=True)
    assert s['action']=='left_arc' and s['surface_filter_horizon_ns']==100_000_000
    s=constrain(s,np.zeros(3),np.eye(3),[])
    assert s['action']=='left_arc' and s['nominal_constraint_horizon_ns']==100_000_000
    assert all(c['radius_m']==.45 for c in s['nominal_action_checks'])
    stopped=constrain(s,np.array([.11,0,0]),np.eye(3),[(11,0)])
    assert stopped['action'] is None and stopped['requested_command']==[0.,0.,0.]


def test_old_forecast_cannot_enter_any_new_timed_guard():
    s=selection();s['first_prediction_horizon_ns']=500_000_000
    for call in (lambda:score_commitment_pose(s),lambda:filter_selection(s,Memory(),object(),now_ns=1,persistent=True),
            lambda:constrain(s,np.zeros(3),np.eye(3),[])):
        with pytest.raises(ValueError,match='100-ms'):call()


def test_actual_observation_cadence_and_failure_stop_remain_inherited():
    assert ObservationHorizonGoalProbe.advance is ObservationReplanGoalProbe.advance
    c=ObservationHorizonGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    result=c.observe({}, {}, {},now_ns=1)
    assert result['terminal']=='SENSOR_OR_MODEL_FAILURE' and result['requested_command']==[0.,0.,0.]
    assert result['model_forecast_horizon_ns']==100_000_000 and result['maximum_open_loop_command_ticks']==1
    assert result['goal_initial_body_xy_m']==[1.2,0.]
