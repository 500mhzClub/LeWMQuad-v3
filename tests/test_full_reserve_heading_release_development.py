from copy import deepcopy

from lewm.full_reserve_heading_release_development import release_heading_turn
from lewm.geometry_progress_pilot_development import ACTIONS


def fixture():
    state = dict(direction=-1, target_heading_rad=1., mission_generation=0)
    scores = {'hold':0., 'left_turn':.07, 'right_turn':-.03}
    selection = dict(action='right_turn', action_index=ACTIONS.index('right_turn'),
        before_memory_filter_action='left_turn', clearance_turn=state | dict(active=True),
        candidates=[dict(action=a, utility_m=scores.get(a, -.1),
            predicted_heading_error_at_commit_start_rad=1.,
            predicted_heading_error_at_commit_end_rad=.8 if a=='left_turn' else 1.1) for a in ACTIONS],
        memory_forecast_candidates=[dict(action=a, full_reserve_path_clear=True,
            reserve_recovery_path_clear=False, nominal_footprint_path_clear=True) for a in ACTIONS])
    return selection, state


def test_release_clear_improving_preferred_turn_without_mutating_saved_state():
    selection, state = fixture(); before = deepcopy(selection)
    result, remaining = release_heading_turn(selection, state)
    assert result['action']=='left_turn' and remaining is None
    assert result['full_reserve_heading_release']['applied']
    assert not result['clearance_turn']['active']
    assert selection==before and state['direction']==-1


def test_recovery_clearance_and_unhelpful_turn_do_not_release():
    selection, state = fixture()
    clearance = next(r for r in selection['memory_forecast_candidates'] if r['action']=='left_turn')
    clearance.update(full_reserve_path_clear=False, reserve_recovery_path_clear=True)
    assert release_heading_turn(selection, state)==(selection, state)
    clearance['full_reserve_path_clear']=True
    candidate=next(r for r in selection['candidates'] if r['action']=='left_turn')
    candidate['predicted_heading_error_at_commit_end_rad']=1.1
    assert release_heading_turn(selection, state)==(selection, state)


def test_survey_requires_better_than_hold_and_preserves_later_hold():
    selection, state = fixture()
    selection['scan_utilities']=[dict(action=a, utility_m=v)
        for a,v in (('hold',0.), ('left_turn',-.01), ('right_turn',-.03))]
    assert release_heading_turn(selection, state)==(selection, state)
    selection['scan_utilities'][1]['utility_m']=.01
    assert release_heading_turn(selection, state)[1] is None
    selection['action']='hold'
    assert release_heading_turn(selection, state)==(selection, state)
