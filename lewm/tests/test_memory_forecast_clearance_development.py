import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.memory_forecast_clearance_development import select_clear_prediction


def test_memory_rejects_hidden_obstacle_path_and_respects_view_actions():
    p=np.zeros((6,8,5));p[1,:,0]=np.linspace(.02,.16,8)
    scores=dict(action='forward',candidates=[dict(action=a,utility_m=10-i) for i,a in enumerate(ACTIONS)])
    result=select_clear_prediction(scores,p,{(55,0)},np.zeros(3),np.eye(3))
    assert result['action']=='hold'
    assert not result['memory_forecast_candidates'][1]['nominal_predicted_path_clear']
    assert result['memory_forecast_candidates'][0]['nominal_predicted_path_clear']
    scores['scan_utilities']=[dict(action='hold',utility_m=0),dict(action='left_turn',utility_m=1)]
    assert select_clear_prediction(scores,p,{(55,0)},np.zeros(3),np.eye(3))['action']=='left_turn'
    # A common-prefix collision cannot be hidden by a safe-looking endpoint.
    p[:,0,0]=.2
    result=select_clear_prediction(scores,p,{(55,0)},np.zeros(3),np.eye(3))
    assert result['action']=='hold'
    assert result['memory_forecast_status']=='NO_CLEAR_CANDIDATE_ZERO_REQUESTED'


def test_translation_reserve_keeps_nominal_turn_clearance():
    p=np.zeros((6,8,5));p[1,:,0]=.02
    scores=dict(action='forward',candidates=[dict(action=a,utility_m=10 if a=='forward' else 0) for a in ACTIONS])
    assert select_clear_prediction(scores,p,{(49,0)},np.zeros(3),np.eye(3))['action']=='forward'
    result=select_clear_prediction(scores,p,{(49,0)},np.zeros(3),np.eye(3),translation_reserve_m=.03)
    assert result['action']=='hold'
    assert result['memory_forecast_candidates'][1]['required_path_clearance_m']==.48
    assert result['memory_forecast_candidates'][4]['required_path_clearance_m']==.45
    assert result['memory_forecast_candidates'][4]['nominal_predicted_path_clear']


def test_recovery_can_leave_existing_reserve_deficit_but_cannot_deepen_it():
    p=np.zeros((6,8,5));p[1,3:,0]=[-.01,-.02,-.03,-.04,-.04]
    scores=dict(action='forward',candidates=[dict(action=a,utility_m=10 if a=='forward' else 0) for a in ACTIONS])
    kwargs=dict(translation_reserve_m=.03,reserve_recovery=True)
    assert select_clear_prediction(scores,p,{(47,0)},np.zeros(3),np.eye(3),translation_reserve_m=.03)['action']=='hold'
    result=select_clear_prediction(scores,p,{(47,0)},np.zeros(3),np.eye(3),**kwargs)
    assert result['action']=='forward' and result['selected_reserve_recovery']
    # Merely improving clearance without restoring the final reserve is insufficient.
    p[1,3:,0]=-.005
    assert select_clear_prediction(scores,p,{(47,0)},np.zeros(3),np.eye(3),**kwargs)['action']=='hold'
    p[1,3:,0]=[-.01,-.02,-.03,-.04,-.04]
    # A path that first gets closer is rejected even if its endpoint recovers.
    p[1,3,0]=.005
    assert select_clear_prediction(scores,p,{(47,0)},np.zeros(3),np.eye(3),**kwargs)['action']=='hold'
    # A nominal-footprint violation in the unchangeable prefix is still rejected.
    p[1,3,0]=-.01;p[:,0,0]=.025
    result=select_clear_prediction(scores,p,{(47,0)},np.zeros(3),np.eye(3),**kwargs)
    assert result['memory_forecast_status']=='NO_CLEAR_CANDIDATE_ZERO_REQUESTED'
