from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from lewm.packed_owned_later_floor_controller_development import PackedOwnedLaterFloorController
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex


def kwargs():
    return dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=100,condition='jepa',variant='full',persistent=True)


def test_all_eight_persistent_indices_are_empty_distinct_and_explicit():
    c=PackedOwnedLaterFloorController(None,None,**kwargs());m=c.memory
    indices=[m.index,m.auxiliary_index,*[v for p in (m.partition,m.auxiliary_partition,m.confirmed_auxiliary_partition)
        for v in (p.floor,p.other)]]
    assert len(indices)==len({id(v) for v in indices})==8
    assert all(type(v) is PackedOwnedMeasuredSampleBoundsIndex and not v.cells for v in indices)
    assert m is c.mapper.surface and c.residual is c.selector.residual


def test_actual_public_packet_path_preserves_every_decision_and_bound(monkeypatch):
    from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_frame_floor_cache_development import equal
    from lewm.tests.test_batched_sample_bounds_development import assert_equal
    monkeypatch.setattr(fixture,'visual',partial(visual,origin=1_500_000_000))
    old,new=[cls(None,None,**kwargs()) for cls in (LaterFloorResolutionRoundTripController,PackedOwnedLaterFloorController)]
    previous=None
    for i in range(2):
        p,d,a,raw,now=packets(i,previous)
        for c in (old,new):c.motion=SimpleNamespace(observe=lambda *args,**kw:deepcopy(raw))
        x,y=[c.observe(p,d,None,now_ns=now,auxiliary_depth=a) for c in (old,new)]
        assert y['terminal'] is None,y.get('failure');equal(x,y)
        for name in ('index','auxiliary_index'):assert_equal(getattr(old.memory,name),getattr(new.memory,name))
        for name in ('partition','auxiliary_partition','confirmed_auxiliary_partition'):
            for member in ('floor','other'):
                aindex=getattr(getattr(old.memory,name),member);bindex=getattr(getattr(new.memory,name),member)
                assert_equal(aindex,bindex)
                assert all(v.base is None and v.flags.owndata for v in bindex.bounds.values())
        previous=raw
