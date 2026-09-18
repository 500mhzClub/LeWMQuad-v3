"""Original registration body and state; isolated floor-index substitution."""
import pytest
from lewm import eligible_floor_registration_development as candidate
from lewm import floor_pose_registration_development as plane
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration


def test_private_bindings_preserve_all_original_function_bodies():
    old=MeasuredFloorTransportRegistration.observe;new=candidate.EligibleFloorRegistration.observe
    assert old.__code__ is new.__code__
    assert candidate.measured_candidates.__code__ is plane.measured_candidates.__code__
    assert old.__globals__['measured_candidates'] is plane.measured_candidates
    assert new.__globals__['measured_candidates'] is candidate.measured_candidates
    assert candidate.measured_candidates.__globals__['observed_floor_cell_index'] is candidate.observed_floor_cell_index
    assert plane.measured_candidates.__globals__['observed_floor_cell_index'] is not candidate.observed_floor_cell_index
    for key in old.__globals__:
        if key!='measured_candidates': assert old.__globals__[key] is new.__globals__[key]


def test_initial_state_and_failure_latch_remain_original():
    old=MeasuredFloorTransportRegistration();new=candidate.EligibleFloorRegistration()
    assert vars(old)==vars(new)
    for obj in (old,new):
        with pytest.raises((ValueError,KeyError)):obj.observe({}, {}, {}, {}, now_ns=1_500_000_000)
        assert obj.failed is True
        with pytest.raises(ValueError,match='latched'):obj.observe({}, {}, {}, {}, now_ns=1_600_000_000)
    assert vars(old)==vars(new)
