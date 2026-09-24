from copy import deepcopy
import math
from lewm.clearance_turn_recovery_development import recover_turn,stepwise_recovery_clear
from lewm.geometry_progress_pilot_development import ACTIONS


def selection():
    distances=dict.fromkeys(ACTIONS,.6)
    distances.update(hold=.485,left_turn=.485,right_turn=.479)
    return dict(action='right_turn',before_memory_filter_action='right_turn',
        scan_heading_error_rad=-.8,waypoint_body_xy_m=[0.,0.],
        candidates=[dict(action=a,utility_m=-2.) for a in ACTIONS],
        scan_utilities=[dict(action='hold',utility_m=0.),dict(action='left_turn',utility_m=-1.),
            dict(action='right_turn',utility_m=1.)],
        memory_forecast_candidates=[dict(action=a,minimum_predicted_path_clearance_m=distances[a],
            nominal_predicted_path_clear=True,segment_clearances_m=[distances[a]]*8,
            reserve_recovery_path_clear=False) for a in ACTIONS])


def test_clear_long_turn_finishes_without_reversing_back_to_short_turn():
    s=selection();before=deepcopy(s)
    r,state=recover_turn(s,0.,0,None)
    assert r['action']=='left_turn' and state['target_heading_rad']==-.8
    assert s==before
    for row in s['memory_forecast_candidates']:
        row.update(minimum_predicted_path_clearance_m=.6,segment_clearances_m=[.6]*8)
    # The shorter right turn is now allowed and still has highest utility.
    # Continue the chosen direction until observed alignment, including wrap.
    for heading in (.5,1.5,2.9,-2.9,-1.5):
        r,state=recover_turn(s,heading,0,state)
        assert r['action']=='left_turn' and state is not None
    r,state=recover_turn(s,-.83,0,state)
    assert state is None
    assert r['clearance_turn']['event']=='MEASURED_TARGET_HEADING_REACHED'


def test_latched_turn_waits_for_clearance_and_cancels_on_mission_change():
    s=selection();_,state=recover_turn(s,0.,0,None)
    left=next(r for r in s['memory_forecast_candidates'] if r['action']=='left_turn')
    left.update(minimum_predicted_path_clearance_m=.44,segment_clearances_m=[.44]*8)
    r,state=recover_turn(s,.2,0,state)
    assert r['action']=='hold' and not r['clearance_turn']['latched_turn_forecast_clear']
    _,state=recover_turn(s,.2,1,state)
    assert state is None


def test_measured_target_crossing_finishes_even_when_alignment_band_is_skipped():
    s=selection();_,state=recover_turn(s,0.,0,None)
    state=state|dict(previous_remaining_rad=.11)
    r,state=recover_turn(s,-.68,0,state)
    assert state is None and not r['clearance_turn']['active']


def test_gradual_recovery_requires_nominal_clearance_and_positive_progress():
    assert stepwise_recovery_clear([.471]*3+[.472,.473,.474,.474,.475],.48)
    assert not stepwise_recovery_clear([.449]*3+[.450,.455,.46,.47,.48],.48)
    assert not stepwise_recovery_clear([.471]*3+[.470,.473,.474,.474,.479],.48)
    assert not stepwise_recovery_clear([.471]*8,.48)


def test_blocked_latch_can_switch_to_a_clearance_increasing_turn():
    s=selection()
    for r in s['memory_forecast_candidates']:
        if r['action']=='left_turn':r.update(minimum_predicted_path_clearance_m=.471,
            segment_clearances_m=[.471]*3+[.472,.473,.474,.474,.475])
        elif r['action']=='right_turn':r.update(minimum_predicted_path_clearance_m=.468,
            segment_clearances_m=[.471]*3+[.471,.470,.469,.468,.468])
    state=dict(target_heading_rad=-.8,direction=-1,previous_remaining_rad=.8,mission_generation=0)
    original,unchanged=recover_turn(s,0.,0,state)
    assert original['action']=='hold' and unchanged['direction']==-1
    result,updated=recover_turn(s,0.,0,state,stepwise=True)
    assert result['action']=='left_turn' and updated['direction']==1
    assert result['selected_stepwise_recovery']
    assert updated['target_heading_rad']==state['target_heading_rad']


def test_blocked_latch_can_switch_to_full_reserve_opposite_turn():
    s=selection()
    state=dict(target_heading_rad=-.8,direction=-1,previous_remaining_rad=.8,mission_generation=0)
    result,updated=recover_turn(s,0.,0,state,stepwise=True)
    assert result['action']=='left_turn' and updated['direction']==1
    assert updated['target_heading_rad']==state['target_heading_rad']
    assert result['clearance_turn']['event']=='FULL_RESERVE_ALTERNATIVE_DIRECTION_SELECTED'


def test_blocked_arc_can_start_long_way_turn_despite_negative_short_term_utility():
    s=selection();s.pop('scan_utilities');s.pop('scan_heading_error_rad')
    s['before_memory_filter_action']='right_arc';s['waypoint_body_xy_m']=[.3,-.15]
    for c in s['candidates']:c['utility_m']={'hold':0.,'right_arc':1.}.get(c['action'],-1.)
    for c in s['memory_forecast_candidates']:
        distance=.485 if c['action'] in ('hold','left_turn') else .47
        c.update(minimum_predicted_path_clearance_m=distance,segment_clearances_m=[distance]*8,
            nominal_predicted_path_clear=distance>.48)
    original,_=recover_turn(s,0.,0,None)
    assert original['action']=='hold'
    result,state=recover_turn(s,0.,0,None,stepwise=True)
    assert result['action']=='left_turn' and state['direction']==1
    assert state['target_heading_rad']==math.atan2(-.15,.3)
    # Do not turn away from a waypoint already within the alignment band.
    s['waypoint_body_xy_m']=[.3,0.]
    result,state=recover_turn(s,0.,0,None,stepwise=True)
    assert result['action']=='hold' and state is None


def test_progress_rejoins_route_without_waiting_for_old_recovery_heading():
    s=selection();s.pop('scan_utilities');s.pop('scan_heading_error_rad')
    for row in s['memory_forecast_candidates']:row['full_reserve_path_clear']=True
    for row in s['candidates']:
        row.update(utility_m=2. if row['action']=='forward' else 0.,
            predicted_progress_during_commit_m=.04 if row['action']=='forward' else 0.,
            position_contact_utility_m=.03 if row['action']=='forward' else 0.)
    state=dict(target_heading_rad=-.8,direction=1,previous_remaining_rad=5.4,mission_generation=0)
    before=deepcopy(s)
    original,retained=recover_turn(s,.2,0,state,stepwise=True)
    assert original['action']=='left_turn' and retained is not None
    result,updated=recover_turn(s,.2,0,state,stepwise=True,release_for_progress=True)
    assert result['action']=='forward' and updated is None
    assert result['clearance_turn']['event']=='FULL_RESERVE_PROGRESS_REJOINS_ROUTE'
    assert s==before
    # Positive heading utility alone does not justify ending recovery.
    next(r for r in s['candidates'] if r['action']=='forward')['predicted_progress_during_commit_m']=-.01
    assert recover_turn(s,.2,0,state,stepwise=True,release_for_progress=True)[1] is not None
    s=deepcopy(before)
    row=next(r for r in s['memory_forecast_candidates'] if r['action']=='forward')
    row['full_reserve_path_clear']=False
    assert recover_turn(s,.2,0,state,stepwise=True,release_for_progress=True)[1] is not None
    # Panorama obligations retain their heading-directed recovery.
    s=deepcopy(before);s['scan_utilities']=selection()['scan_utilities']
    assert recover_turn(s,.2,0,state,stepwise=True,release_for_progress=True)[1] is not None
