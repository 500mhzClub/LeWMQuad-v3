from types import SimpleNamespace
import numpy as np
import pytest
from lewm.current_pair_routing_memory_development import CurrentPairRoutingSnapshot
from scripts.run_go2_current_plane_heading_reactive_noise_development import MatchedHeadingReactiveRuntime


class NoForecast:
    def __getattr__(self,name):raise AssertionError('reactive control accessed a prediction model')


@pytest.mark.parametrize('goal,expected,duration',[
    ([.05,0.],'forward',100_000_000),
    ([.04,.03],'left_turn',400_000_000),
    ([.01,0.],'hold',400_000_000)])
def test_captured_map_terminal_selection_remains_model_free(goal,expected,duration):
    runtime=MatchedHeadingReactiveRuntime.__new__(MatchedHeadingReactiveRuntime)
    runtime.model=NoForecast();runtime.terminal_position_approach=True
    runtime.mission=SimpleNamespace(arrival_radius_m=.02)
    snapshot=CurrentPairRoutingSnapshot(frame=0,measured_ns=1_500_000_000,
        floor=frozenset(),occupied=frozenset({(40,40)}),position_map=(0.,0.,0.),
        map_from_initial=tuple(map(tuple,np.eye(3))),floor_height=-.32,
        primary_current_floor_cells=0,auxiliary_current_floor_cells=0,
        fine_occupied=frozenset({(200,200)}))
    selection,correction=runtime._select_action(None,None,[],np.array(goal),None,
        snapshot,np.zeros(3),np.eye(3))
    assert correction is None and selection['candidate_future_outcomes_evaluated'] is False
    assert selection['learned_model_used'] is False and selection['action']==expected
    assert selection['command_duration_ns']==duration
    assert selection['routing_memory_scope']['condition']=='persistent'
