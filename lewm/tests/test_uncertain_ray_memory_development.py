from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.uncertain_ray_memory_development import (
    depth_evidence,query_envelopes,transport_radius,FusedRayEvidenceMemory)
from lewm.tests.test_observed_turn_region_development import frame
from lewm.tests.test_observed_traversal_controller_development import Stream


def evidence():
    return depth_evidence(np.full((480,640),2.,np.float32),np.ones((480,640),bool),[0,0,1])


def observe(memory,stream,tick,*,position=0.,previous=0.,floor=False,wall=2.,weak=False):
    p,d,r,now=frame(stream,tick,position=position,previous=previous,floor=floor,wall=wall)
    if r['motion'] is not None:
        r['motion']['observable_projection_previous_body_m']=[position-previous,0.,0.]
        r['motion']['weak_directions_previous_body']=[]
        if weak:
            r['motion'].update(rank=2,status='PARTIALLY_OBSERVED_TRANSLATION',
                translation_previous_body_m=None,weak_directions_previous_body=[[0,1,0]])
            r['position_initial_body_m']=None
    before=deepcopy(r); row=memory.observe(p,d,r,now_ns=now)
    assert r==before
    return p,d,r,now,row


def test_lateral_envelope_finds_thin_obstacle_not_on_nominal_ray():
    depth=np.full((480,640),2.,np.float32); depth[240,330]=1.
    e=depth_evidence(depth,np.ones(depth.shape,bool),[0,0,1])
    points=np.array([[1.326,0,.043]])
    zero=query_envelopes(e,points,[0.],np.array([False]))
    uncertain=query_envelopes(e,points,[.04],np.array([False]))
    assert zero['free'][0]
    assert uncertain['contradictory_or_near_surface'][0] and not uncertain['free'][0]
    assert uncertain['examined_pixels'][0]>zero['examined_pixels'][0]


def test_missing_lateral_pixel_prevents_whole_envelope_clearance():
    depth=np.full((480,640),2.,np.float32); valid=np.ones(depth.shape,bool)
    valid[240,330]=False; depth[240,330]=0.; e=depth_evidence(depth,valid,[0,0,1])
    row=query_envelopes(e,[[1.326,0,.043]],[.04],np.array([False]))
    assert not row['free'][0] and not row['contradictory_or_near_surface'][0]


def test_zero_radius_keeps_four_pixel_sampling_and_unknown_rear_volume():
    row=query_envelopes(evidence(),[[1.326,0,.043],[-1,0,0]],[0.,0.],np.zeros(2,bool))
    assert row['free'].tolist()==[True,False]
    assert row['examined_pixels'].tolist()==[4,0]


@pytest.mark.parametrize('point',[[1.326,0,.043],[1.326,-.8,.043],[.526,0,.043]])
def test_growing_uncertainty_cannot_gain_free_evidence_or_erase_conflicts(point):
    depth=np.full((480,640),2.,np.float32); depth[200:280,310:340]=1.
    e=depth_evidence(depth,np.ones(depth.shape,bool),[0,0,1])
    previous_free=True; previous_conflict=False
    for radius in (0.,.005,.02,.08,.2,.8,1.5,3.):
        row=query_envelopes(e,[point],[radius],np.array([False]))
        assert previous_free or not row['free'][0]
        assert not previous_conflict or row['contradictory_or_near_surface'][0]
        previous_free=bool(row['free'][0]); previous_conflict=bool(row['contradictory_or_near_surface'][0])


def test_floor_support_is_separate_role_and_decreases_with_radius():
    p,d,r,now=frame(Stream(),0,floor=True)
    e=depth_evidence(d['depth_m'],d['valid'],[0,0,1]); previous=True
    for radius in (0.,.005,.02,.04,.07,.12):
        row=query_envelopes(e,[[1.5,0,-.317],[1.5,0,-.317]],[radius,radius],np.array([True,False]))
        assert not row['free'].any() and not row['observed_ground_support'][1]
        assert previous or not row['observed_ground_support'][0]
        if radius==0: assert row['observed_ground_support'][0]
        if radius>=.07: assert not row['observed_ground_support'][0]
        previous=bool(row['observed_ground_support'][0])


def test_current_pose_cancels_and_historical_lever_arm_increases_radius():
    current={'position':np.array([1.,0,0]),'measured_ns':10,'position_scale_m':.02,'orientation_scale_rad':.01}
    old={**current,'position':np.zeros(3),'measured_ns':0,'position_scale_m':.01}
    points=np.array([[1.,0,0],[2.,0,0]])
    assert np.array_equal(transport_radius(points,current,current),[0,0])
    radius=transport_radius(points,current,old)
    assert radius[1]>radius[0]>.03


def test_fusion_memory_keeps_depth_missing_component_and_predicts_separately():
    stream=Stream(); memory=FusedRayEvidenceMemory()
    for tick in range(4):
        p,d,r,now,row=observe(memory,stream,tick,position=.02*tick,previous=.02*max(0,tick-1),weak=tick>1)
    assert r['motion']['rank']==2 and r['motion']['translation_previous_body_m'] is None
    assert r['position_initial_body_m'] is None
    assert row['motion_kind']=='INERTIALLY_PREDICTED_WEAK_COMPONENT' and row['depth_rank']==2
    np.testing.assert_allclose(memory.position,[.06,0,0],atol=1e-12)
    assert not row['navigation_qualified']


def test_budget_exhaustion_latches_and_disables_current_and_historical_queries():
    stream=Stream(); memory=FusedRayEvidenceMemory(); failed=None
    for tick in range(30):
        try: observe(memory,stream,tick,position=.01*tick,previous=.01*max(0,tick-1),weak=tick>1)
        except SensorContractError as error:
            assert 'budget exhausted' in str(error.__cause__); failed=tick; break
    assert failed is not None and memory.failed
    with pytest.raises(SensorContractError): memory.query([[1,0,0]],np.zeros(1,bool),now_ns=memory.last_ns)
    with pytest.raises(SensorContractError,match='latched'): observe(memory,stream,failed+1)


def test_historical_free_space_can_be_transported_but_latest_conflict_overrides_it():
    stream=Stream(); memory=FusedRayEvidenceMemory()
    for tick in range(11): observe(memory,stream,tick,position=.1*tick,previous=.1*max(0,tick-1))
    result=memory.query([[-.4,0,0]],np.array([False]),now_ns=memory.last_ns)
    assert result['free'][0] and result['maximum_transport_radius_m'][0]>0
    p,d,r,now,_=observe(memory,stream,11,position=1.1,previous=1.,wall=2.)
    result=memory.query([[1.226,0,0]],np.array([False]),now_ns=now)
    assert result['contradictory_or_near_surface'][0] and not result['all_samples_supported']


def test_latest_stationary_view_not_evicted_or_given_historical_penalty():
    memory=FusedRayEvidenceMemory(); stream=Stream()
    observe(memory,stream,0)
    p,d,r,now,_=observe(memory,stream,1,wall=1.)
    assert len(memory.frames)==1
    result=memory.query([[1.326,0,0]],np.zeros(1,bool),now_ns=now)
    assert result['contradictory_or_near_surface'][0] and not result['all_samples_supported']


@pytest.mark.parametrize('fault',['depth_rewrite','clock','extra_field','identity'])
def test_binding_faults_disable_memory(fault):
    m=FusedRayEvidenceMemory(); stream=Stream(); observe(m,stream,0)
    p,d,r,now=frame(stream,1)
    r['motion'].update(observable_projection_previous_body_m=[0,0,0],weak_directions_previous_body=[])
    if fault=='depth_rewrite': d['depth_m'][0,0]=1.5
    elif fault=='clock': now+=1
    elif fault=='extra_field': r['privileged_world_pose']=[0,0,0]
    elif fault=='identity': r['local_surfaces']['identity']=(0,0,1)
    with pytest.raises(SensorContractError): m.observe(p,d,r,now_ns=now)
    assert m.failed


@pytest.mark.parametrize('radius',[-.01,np.nan,np.inf])
def test_invalid_radius_rejected(radius):
    with pytest.raises(SensorContractError): query_envelopes(evidence(),[[1,0,0]],[radius],np.array([False]))


def test_randomized_projection_growth_preserves_free_and_conflict_monotonicity():
    rng=np.random.default_rng(3107)
    d=rng.uniform(.2,5.,size=(480,640)).astype(np.float32)
    valid=rng.random(d.shape)>.05; d[~valid]=0.
    e=depth_evidence(d,valid,[0,0,1])
    points=rng.uniform([-1,-2,-1],[4,2,1],size=(100,3)); roles=np.zeros(100,bool)
    previous=None
    for radius in (0.,.001,.01,.05,.2,1.,3.):
        current=query_envelopes(e,points,np.full(100,radius),roles)
        if previous is not None:
            assert not (current['free']&~previous['free']).any()
            assert not (previous['contradictory_or_near_surface']&~current['contradictory_or_near_surface']).any()
        previous=current


def test_empty_query_is_not_an_all_clear_certificate():
    m=FusedRayEvidenceMemory(); _,_,_,now,_=observe(m,Stream(),0)
    result=m.query(np.empty((0,3)),np.zeros(0,bool),now_ns=now)
    assert not result['all_samples_supported'] and len(result['free'])==0
