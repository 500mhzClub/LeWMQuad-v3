import numpy as np
from lewm.visual_support_recovery_development import LocalSupportedView
from lewm.navigation_capability_exhausted_view_development import ExhaustibleSupportedView


def step(view, ns, counts=(20,20), angle=0., generation=0):
    c,s=np.cos(angle),np.sin(angle)
    return view.advance(counts,np.zeros(3),np.array([[c,-s,0],[s,c,0],[0,0,1.]]),ns,generation)


def test_old_deadlock_and_new_reference_consumption():
    old=LocalSupportedView(maximum_view_age_ns=None)
    new=ExhaustibleSupportedView(maximum_view_age_ns=None)
    for view in (old,new):step(view,0,(100,100))
    for ns in range(100_000_000,1_100_000_000,100_000_000):
        assert step(old,ns) is not None
        assert step(new,ns) is not None
    assert step(old,1_100_000_000) is not None
    assert step(new,1_100_000_000) is None
    assert len(new.retirements)==1
    assert step(new,1_200_000_000) is None  # Cannot reuse the consumed strong view.
    step(new,1_300_000_000,(100,100))
    assert step(new,1_400_000_000,angle=.3) is not None


def test_nonattained_target_stays_active():
    v=ExhaustibleSupportedView(maximum_view_age_ns=None);step(v,0,(100,100))
    for ns in range(100_000_000,2_100_000_000,100_000_000):
        assert step(v,ns,angle=.2) is not None
    assert not v.retirements


def test_alignment_interruption_or_missing_pose_restarts_dwell():
    for interruption in ('heading','gap'):
        v=ExhaustibleSupportedView(maximum_view_age_ns=None);step(v,0,(100,100))
        for ns in range(100_000_000,1_000_000_000,100_000_000):step(v,ns)
        if interruption=='heading':step(v,1_000_000_000,angle=.2)
        for ns in range(1_100_000_000,2_100_000_000,100_000_000):assert step(v,ns) is not None
        assert step(v,2_100_000_000) is None


def test_normal_support_release_and_generation_reset_unchanged():
    v=ExhaustibleSupportedView(maximum_view_age_ns=None);step(v,0,(100,100))
    assert step(v,100_000_000,angle=.3) is not None
    assert step(v,200_000_000,(50,50)) is None
    assert not v.retirements
    assert step(v,300_000_000,angle=.3) is not None
    assert step(v,400_000_000,generation=1) is None
    assert not v.retirements
