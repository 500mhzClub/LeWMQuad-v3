from copy import deepcopy
import math

from lewm.selected_route_turn_memory_development import SelectedRouteTurnMemory
from lewm.tests.test_interrupted_route_turn_memory_development import selection


def interrupted_left():
    memory = SelectedRouteTurnMemory()
    chosen = selection(1., 'left_turn')
    chosen['before_memory_filter_action'] = 'right_turn'
    memory.select(chosen, [0., 0.], 1., 0, None)
    recovery = selection(1.2, 'right_turn') | dict(scan_utilities=[])
    assert memory.select(recovery, [0., 0.], 1.2, 0, 100) is recovery
    assert {r['direction'] for r in memory.failed} == {1}
    return memory


def test_selected_failed_direction_overrides_stale_score_preference():
    memory = interrupted_left()
    chosen = selection(1., 'left_turn')
    chosen['before_memory_filter_action'] = 'right_turn'
    before = deepcopy(chosen)
    result = memory.select(chosen, [0., 0.], 1., 0, None)
    assert result['action'] == 'right_turn' and chosen == before
    assert result['before_memory_filter_action'] == 'right_turn'
    assert result['visual_route_turn_memory']['interruption_match_action'] == 'left_turn'
    # Continue the clear chosen alternative until measured target completion.
    assert memory.select(selection(.8, 'left_turn'), [0., 0.], .8, 0, None)['action'] == 'right_turn'
    assert memory.select(selection(.05, 'hold'), [0., 0.], .05, 0, None)['action'] == 'hold'
    assert memory.active is None


def test_opposite_clearance_recovery_locality_and_generation_still_apply():
    for blocked, weak, position, generation in (
            (True, None, [0., 0.], 0), (False, 101, [0., 0.], 0),
            (False, None, [.3, 0.], 0), (False, None, [0., 0.], 1)):
        memory = interrupted_left()
        chosen = selection(1., 'left_turn')
        chosen['before_memory_filter_action'] = 'right_turn'
        for row in chosen['memory_forecast_candidates']:
            if row['action'] == 'right_turn':
                row['nominal_predicted_path_clear'] = not blocked
        before = deepcopy(chosen)
        result = memory.select(chosen, position, 1., generation, weak)
        assert result['action'] == 'left_turn' and chosen == before
        assert 'visual_route_turn_memory' not in result


def test_scan_and_both_failed_directions_do_not_invent_alternative():
    memory = interrupted_left()
    scan = selection(1., 'left_turn') | dict(scan_utilities=[])
    assert memory.select(scan, [0., 0.], 1., 0, None) is scan
    chosen = selection(1., 'left_turn')
    chosen['before_memory_filter_action'] = 'right_turn'
    memory.failed.append(memory.failed[0] | dict(direction=-1))
    assert memory.select(chosen, [0., 0.], 1., 0, None)['action'] == 'left_turn'
    assert memory.active is None
