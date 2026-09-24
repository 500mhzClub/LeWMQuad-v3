import numpy as np
import pytest

from lewm.foot_load_sensor_development import FEET,RawFootCounts,IdealFootForceSample,LocalLoadHistory
from lewm.simulated_foot_force_development import sample_ideal_foot_forces


def packet(a=(1,),b=(99,),fa=((0,0,10),),fb=((0,0,-10),),valid=(True,)):
    return dict(geom_a=np.asarray(a,int),geom_b=np.asarray(b,int),force_a=np.asarray(fa,float).reshape(-1,3),
        force_b=np.asarray(fb,float).reshape(-1,3),valid_mask=np.asarray(valid,bool))


def acquire(p,**kwargs):
    args=dict(foot_geom_ids=(1,2,3,4),rotation_world_from_foot=np.tile(np.eye(3),(4,1,1)),
        acquisition_identity='synthetic',measured_ns=0,available_ns=0)
    return sample_ideal_foot_forces(p,**(args|kwargs))


def observation(t=0,force=10.,valid=True,saturated=False):
    return IdealFootForceSample('synthetic',t,t,np.tile([0,0,force],(4,1)),np.full(4,valid),np.full(4,saturated))


def test_all_incident_sides_including_foot_self_contact():
    p=packet((1,99,1),(99,2,2),((0,0,10),(0,0,-20),(3,0,0)),((0,0,-10),(0,0,20),(-3,0,0)),(True,)*3)
    s=acquire(p)
    np.testing.assert_array_equal(s.force_foot_n,[[3,0,10],[-3,0,20],[0,0,0],[0,0,0]])
    assert s.valid.all()


def test_other_geometry_relabeling_cannot_change_sensor():
    ground=acquire(packet(b=(99,))); wall=acquire(packet(b=(400,)))
    np.testing.assert_array_equal(ground.force_foot_n,wall.force_foot_n)
    with pytest.raises(ValueError): acquire(packet()|{'ground_geom_ids':[99]})


def test_sensor_axes_and_global_rotation_invariance():
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    p=packet(fa=((2,3,4),),fb=((-2,-3,-4),)); expected=acquire(p)
    p['force_a']=p['force_a']@R.T; p['force_b']=p['force_b']@R.T
    actual=acquire(p,rotation_world_from_foot=np.tile(R,(4,1,1)))
    np.testing.assert_allclose(expected.force_foot_n,actual.force_foot_n)
    # A wrong mounting axis changes components, not resultant magnitude.
    wrong=acquire(p); assert not np.array_equal(actual.force_foot_n,wrong.force_foot_n)
    np.testing.assert_allclose(np.linalg.norm(actual.force_foot_n,axis=1),np.linalg.norm(wrong.force_foot_n,axis=1))


def test_empty_is_zero_but_missing_is_unknown():
    zero=acquire(packet((),(),(),(),())); missing=acquire(None)
    assert zero.valid.all() and not zero.force_foot_n.any()
    assert not missing.valid.any() and np.isnan(missing.force_foot_n).all()


def test_invalid_padding_ignored_not_valid_bad_data():
    p=packet((-1,),(-1,),((np.nan,)*3,),((np.nan,)*3,),(False,))
    assert not acquire(p).force_foot_n.any()
    p['valid_mask'][:]=True
    with pytest.raises(ValueError): acquire(p)


def test_opposing_contacts_can_cancel_without_no_contact_claim():
    s=acquire(packet((1,1),(99,100),((0,0,10),(0,0,-10)),((0,0,-10),(0,0,10)),(True,True)))
    assert not s.force_foot_n.any()
    row=LocalLoadHistory(acquisition_identity='synthetic').observe(s,now_ns=0,conditional_force_error_n=0)
    assert row['feet'][0]['status']=='BELOW_LOAD_THRESHOLD'
    assert not row['feet'][0]['ground_support_established']


@pytest.mark.parametrize('ids',[(1,1,3,4),(True,2,3,4),(-1,2,3,4),(1,2,3)])
def test_bad_sensor_identities(ids):
    with pytest.raises(ValueError): acquire(packet(),foot_geom_ids=ids)


@pytest.mark.parametrize('R',[np.eye(3),np.tile(np.diag([1,1,-1]),(4,1,1)),np.full((4,3,3),np.nan)])
def test_bad_rotations(R):
    with pytest.raises(ValueError): acquire(packet(),rotation_world_from_foot=R)


def test_hardware_counts_preserved_never_converted_or_reordered():
    values=np.array([10,-3,32767,-32768],np.int16)
    raw=RawFootCounts('device-unknown',('vendor0','vendor1','vendor2','vendor3'),0,3,values,np.ones(4,bool),np.array([0,0,1,1],bool))
    values[0]=999; d=raw.diagnostic()
    assert d['values']==[10,-3,32767,-32768] and d['calibrated_force_n'] is None
    assert not d['canonical_foot_order_verified']
    with pytest.raises(ValueError): LocalLoadHistory(acquisition_identity='synthetic').observe(raw,now_ns=3)


@pytest.mark.parametrize('values',[np.array([1.,2,3,4]),np.array([1,2,3,32768]),np.array([1,2,3])])
def test_bad_hardware_counts(values):
    with pytest.raises(ValueError): RawFootCounts('x',FEET,0,0,values,np.ones(4,bool),np.zeros(4,bool))


def test_dwell_requires_continuous_fresh_loaded_samples():
    model=LocalLoadHistory(acquisition_identity='synthetic')
    for t in range(0,22_000_000,2_000_000):
        r=model.observe(observation(t),now_ns=t,conditional_force_error_n=0)
        assert all(x['dwell_observed']==(t>=20_000_000) for x in r['feet'])
    r=model.observe(observation(40_000_000),now_ns=40_000_000,conditional_force_error_n=0)
    assert not any(x['dwell_observed'] for x in r['feet'])


@pytest.mark.parametrize('case,status',[('missing','MISSING'),('saturated','SATURATED'),('stale','STALE'),('error','UNKNOWN_FORCE_ERROR')])
def test_unavailable_is_not_unloaded_or_supported(case,status):
    sample=observation(valid=case!='missing',saturated=case=='saturated')
    r=LocalLoadHistory(acquisition_identity='synthetic').observe(sample,now_ns=20_000_000 if case=='stale' else 0,
        conditional_force_error_n=None if case=='error' else 0)
    assert all(x['status']==status and x['resultant_lower_n'] is None and not x['ground_support_established'] for x in r['feet'])


def test_offset_uncertainty_changes_threshold_decision():
    model=LocalLoadHistory(acquisition_identity='synthetic')
    r=model.observe(observation(force=6),now_ns=0,conditional_force_error_n=2)
    assert all(x['status']=='AMBIGUOUS_LOAD' for x in r['feet'])


def test_causality_identity_and_bad_error_rejected():
    m=LocalLoadHistory(acquisition_identity='synthetic')
    with pytest.raises(ValueError): m.observe(observation(),now_ns=0,conditional_force_error_n=-1)
    assert m.last is None
    m.observe(observation(),now_ns=0)
    with pytest.raises(ValueError): m.observe(observation(),now_ns=0)
    with pytest.raises(ValueError): m.observe(observation(5),now_ns=4)
    delayed=IdealFootForceSample('synthetic',5,10,np.zeros((4,3)),np.ones(4,bool),np.zeros(4,bool))
    with pytest.raises(ValueError): m.observe(delayed,now_ns=9)
    with pytest.raises(ValueError): LocalLoadHistory(acquisition_identity='other').observe(observation(),now_ns=0)


def test_force_alone_cannot_identify_slip_or_support_heights():
    # Same force histories can occur on four uneven supports or while slipping.
    # No topology/velocity/height is accepted or inferred by this model.
    r=LocalLoadHistory(acquisition_identity='synthetic').observe(observation(),now_ns=0,conditional_force_error_n=0)
    assert all(not x['slip_excluded'] and not x['ground_support_established'] for x in r['feet'])
    assert not any(r[k] for k in ('continuous_floor_established','future_footfall_validated','body_sweep_validated','navigation_qualified'))
