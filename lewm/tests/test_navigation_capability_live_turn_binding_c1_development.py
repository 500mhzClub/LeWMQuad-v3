from copy import deepcopy
import numpy as np
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemory
from lewm.navigation_capability_live_turn_binding_c1_development import LiveEligibleRouteTurnMemory
from lewm.geometry_progress_pilot_development import ACTIONS


def fixture(cls, right_clear=False,left_clear=True):
    v=cls();v.generation=1
    v.active=dict(position=np.zeros(2),target_heading_rad=2.6,direction=-1,previous_remaining_rad=3.7)
    v.failed=[dict(position=np.zeros(2),target_heading_rad=2.6,direction=d)for d in [-1,1]]
    s=dict(action='left_turn' if left_clear else 'hold',before_memory_filter_action='left_turn',waypoint_body_xy_m=[np.cos(2.6),np.sin(2.6)],
        memory_forecast_candidates=[dict(action=a,nominal_predicted_path_clear=(left_clear if a=='left_turn' else right_clear if a=='right_turn' else a=='hold'),reserve_recovery_path_clear=False) for a in ACTIONS])
    return v,s


def test_retained_failure_stops_holding_when_opposite_is_eligible():
    old,s=fixture(InterruptedRouteTurnMemory);new,_=fixture(LiveEligibleRouteTurnMemory)
    assert old.select(deepcopy(s),np.zeros(2),0.,1,None)['action']=='hold'
    r=new.select(deepcopy(s),np.zeros(2),0.,1,None)
    assert r['action']=='left_turn' and new.active is None
    assert r['memory_forecast_candidates']==s['memory_forecast_candidates']
    assert r['released_visual_route_turn_memory']['action']=='right_turn'


def test_eligible_latch_is_unchanged():
    old,s=fixture(InterruptedRouteTurnMemory,right_clear=True);new,_=fixture(LiveEligibleRouteTurnMemory,right_clear=True)
    assert old.select(deepcopy(s),np.zeros(2),0.,1,None)==new.select(deepcopy(s),np.zeros(2),0.,1,None)


def test_no_eligible_turn_still_holds():
    new,s=fixture(LiveEligibleRouteTurnMemory,left_clear=False)
    assert new.select(s,np.zeros(2),0.,1,None)['action']=='hold'


def test_visual_recovery_and_generation_reset_remain_parent_behaviour():
    for weak,generation in [(100,1),(None,2)]:
        old,s=fixture(InterruptedRouteTurnMemory);new,_=fixture(LiveEligibleRouteTurnMemory)
        assert old.select(deepcopy(s),np.zeros(2),0.,generation,weak)==new.select(deepcopy(s),np.zeros(2),0.,generation,weak)


def test_actual_deployed_stacks_install_original_interrupted_memory():
    from scripts.run_go2_dense_horizon_navigation_development import DenseNavigationRuntime, DenseReactiveNavigationRuntime
    from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemoryMixin
    from lewm.selected_route_turn_memory_development import SelectedRouteTurnMemoryMixin
    for runtime in (DenseNavigationRuntime,DenseReactiveNavigationRuntime):
        assert InterruptedRouteTurnMemoryMixin in runtime.__mro__
        assert SelectedRouteTurnMemoryMixin not in runtime.__mro__
    # Exercise the exact replacement constructor without worker threads/physics.
    from lewm.navigation_capability_live_turn_binding_c1_development import LiveTurnMemoryRuntimeMixin
    from lewm.navigation_capability_exhausted_view_development import ExhaustedViewRuntimeMixin
    from unittest.mock import patch
    obj=object.__new__(LiveTurnMemoryRuntimeMixin)
    def parent(self,*a,**kw):self.route_turn_memory=InterruptedRouteTurnMemory()
    with patch.object(ExhaustedViewRuntimeMixin,'__init__',parent):
        LiveTurnMemoryRuntimeMixin.__init__(obj)
    assert type(obj.route_turn_memory) is LiveEligibleRouteTurnMemory
