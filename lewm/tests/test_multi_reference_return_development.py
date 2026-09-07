"""Real packet fixtures plus synthetic selection/mission cases, not new physics."""
from copy import deepcopy
import hashlib
import math
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose,MultiReferenceVisualLedMotion,Reference
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose
from lewm.metric_return_intent_development import ReturnIntent
from lewm.intent_room_return_development import IntentRoomReturn
from lewm.sensor_anchored_goal_development import AnchoredGoal
from lewm.coupled_pulse_rollout_development import compose
from lewm.tests.test_rgbd_correspondence_motion_development import texture,packets
from lewm.tests.test_continuous_pulse_execution_development import visual
from scripts.fixed_nominal_pulse_table_development import load_fixed_table


def test_primary_accepted_stream_is_exactly_the_predecessor():
    a=RigidRGBDKeyframePose('gyro');b=MultiReferenceRGBDPose()
    for p,d,f,t in packets([texture()]*4):
        x=a.observe(p,d,f,now_ns=t);y=b.observe(p,d,f,now_ns=t)
        assert x==y
    assert b.last_selection['status']=='PRIMARY_ACCEPTED'


def test_retained_actual_reference_can_replace_rejected_primary_without_reset(monkeypatch):
    monkeypatch.setattr('lewm.multi_reference_rgbd_pose_development.support_near_limit',lambda x:True)
    m=MultiReferenceRGBDPose();items=list(packets([texture()]*4))
    for p,d,f,t in items[:2]:m.observe(p,d,f,now_ns=t)
    original=m._candidate
    def candidate(ref,current,G):
        if m.frame==2 and ref.frame==1:raise SensorContractError('synthetic primary rejection')
        return original(ref,current,G)
    monkeypatch.setattr(m,'_candidate',candidate)
    p,d,f,t=items[2];r=m.observe(p,d,f,now_ns=t)
    assert r['reference_frame']==0 and r['promotion_reason']=='qualified_recent_reference'
    assert m.last_selection['status']=='RECENT_REFERENCE_ACCEPTED' and not m.failed
    assert m.nodes[-1]['parent_frame']==0 and m.anchor_frame==2
    p,d,f,t=items[3];r=m.observe(p,d,f,now_ns=t)
    assert r['reference_frame']==2 and m.gyro.samples_integrated==150


def test_feature_bank_bounded_and_pose_copies_not_aliases():
    m=MultiReferenceRGBDPose();p=np.zeros(3)
    for i in range(20):m.frame=i;m._remember(object(),np.eye(3),np.eye(3),p,i)
    p[0]=100
    assert [r.frame for r in m.references]==list(range(12,20))
    assert all(r.position[0]==0 for r in m.references)


@pytest.mark.parametrize('conflict',[True,False])
def test_alternative_conflict_rejects_and_sensor_quality_selects(monkeypatch,conflict):
    m=MultiReferenceRGBDPose()
    for i in range(3):m.frame=i;m._remember(object(),np.eye(3),np.eye(3),np.zeros(3),i)
    def candidate(ref,current,G):
        if ref.frame==2:raise SensorContractError('primary rejected')
        q=dict(reference_grid_cells=6+ref.frame,current_grid_cells=6+ref.frame,inlier_fraction=.9,
               inliers=50,residual_rms_m=.003)
        return dict(reference=ref,p=np.array([ref.frame*(.03 if conflict else .001),0,0]),R=np.eye(3),registration=q)
    monkeypatch.setattr(m,'_candidate',candidate)
    if conflict:
        with pytest.raises(SensorContractError,match='conflict'):m._choose(object(),np.eye(3))
        assert m.last_selection['status']=='CONFLICTING_ALTERNATIVES'
    else:
        chosen,fallback=m._choose(object(),np.eye(3))
        assert fallback and chosen['reference'].frame==1


@pytest.mark.parametrize('fault',['blank','clock','episode','privileged'])
def test_failure_latches_with_no_reset_or_invented_pose(fault):
    m=MultiReferenceVisualLedMotion();items=list(packets([texture()]*3))
    p,d,f,t=items[0];m.observe(p,d,f,now_ns=t)
    p,d,f,t=deepcopy(items[1])
    if fault=='blank':p['image']['rgb'][:]=128;d['rgb_sha256']=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    if fault=='clock':t+=1
    if fault=='episode':p['sensor_state']['identity']=(1,0,0)
    if fault=='privileged':p['native_pose']=[0,0,0]
    r=m.observe(p,d,f,now_ns=t);count=m.model.frame
    p,d,f,t=items[2];after=m.observe(p,d,f,now_ns=t)
    assert r['status']==after['status']=='VISUAL_TERMINAL_FAILURE' and after['current_pose'] is None
    assert m.model.frame==count and len(m.model.references)==1


def test_home_heading_only_on_unclipped_home_goal():
    e,t=visual(0,x=.401,yaw=-math.pi);pose=e['current_pose']
    intent=ReturnIntent.create('RETURN_HOME',pose,[0,0]);d,w,final=intent.subgoal(pose)
    g=AnchoredGoal.from_observation(e,d,w,identity=(0,0,0),now_ns=t)
    assert not final and g.target_xy==pytest.approx([.001,0],abs=1e-12)
    assert abs(abs(g.target_yaw_rad)-math.pi)<1e-12
    e,t=visual(1,x=.061,yaw=-math.pi);d,w,final=intent.subgoal(e['current_pose'])
    g=AnchoredGoal.from_observation(e,d,w,identity=(0,0,0),now_ns=t)
    assert final and g.target_xy==pytest.approx([0,0],abs=1e-12) and g.target_yaw_rad==pytest.approx(0,abs=1e-12)


def test_corner_approach_heading_survives_clipped_arrival_overshoot():
    e,t=visual(0,x=.405,yaw=math.pi);intent=ReturnIntent.create('RETURN_CORNER',e['current_pose'],[0,0])
    assert intent.subgoal(e['current_pose'])[2] is False
    e,t=visual(1,x=-.02,yaw=math.pi);d,w,final=intent.subgoal(e['current_pose'])
    g=AnchoredGoal.from_observation(e,d,w,identity=(0,0,0),now_ns=t)
    assert final and g.target_xy==pytest.approx([0,0],abs=1e-12)
    assert abs(abs(g.target_yaw_rad)-math.pi)<1e-12  # no fresh180degree reversal for2cm residual


def test_near_destination_does_not_invent_approach_from_tiny_residual():
    e,_=visual(0,x=-.001,yaw=.7)
    intent=ReturnIntent.create('RETURN_CORNER',e['current_pose'],[0,0])
    assert intent.travel_heading==pytest.approx(.7)
    with pytest.raises(SensorContractError):ReturnIntent.create('RETURN_HOME',e['current_pose'],[float('nan'),0])


@pytest.mark.parametrize('sign',[-1,1])
def test_intent_mission_completes_model_matched_plant_without_pose_reset(sign):
    table=load_fixed_table();model=IntentRoomReturn(sign,table);state=np.zeros(3);increment=np.zeros(3)
    for tick in range(3601):
        e,t=visual(tick,*state);ex=model.runtime.executor.observe(e,now_ns=t)
        r=model.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
        if r['terminal']:break
        local=ex['local_decision']
        if local and 'new_pulse' in local['diagnostic']:
            effect=table.effects[local['diagnostic']['new_pulse']['action_index']]
            increment=(compose(state,effect.delta_xy_yaw)-state)/effect.ticks
        if any(r['requested_command']):state+=increment
    assert r['terminal']=='ROOM_RETURN_CANDIDATE',r
    assert len(model.completed)==7 and np.linalg.norm(state[:2])<=.06
    assert model.runtime.executor.start_ns==1_600_000_000 and not model.snapshot()['home_verified']


def equal_nested(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_nested(a[k],b[k])
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b


def test_cached_reader_matches_frozen_readers_and_does_not_alias_returned_arrays():
    from lewm.cached_rgbd_replay_development import CachedRGBDReplay
    from lewm.rgbd_dataset_development import load_rgbd_observation
    from scripts.fast_gyro_scan_session_development import load_fast_packet
    from scripts.run_go2_coupled_room_return_v1 import OUTPUT
    directory=OUTPUT/'nominal_left';reader=CachedRGBDReplay(directory)
    for i in (0,1654,1677,1687):
        p,d,f,now=reader.packet(i);a,b=load_rgbd_observation(directory,i);c=load_fast_packet(directory,i)
        equal_nested(p,a);equal_nested(d,b);equal_nested(f,c)
        assert now==p['sensor_state']['decision_ns']
        f['values'][:]=123;p['sensor_state']['sensed']['gyro']['values'][:]=123
        p,d,f,now=reader.packet(i);equal_nested(p,a);equal_nested(f,c)
    with pytest.raises(SensorContractError):reader.packet(True)
    with pytest.raises(SensorContractError):reader.packet(1688)


def test_cached_reader_rejects_protected_path_before_io():
    from lewm.cached_rgbd_replay_development import CachedRGBDReplay
    with pytest.raises(SensorContractError):CachedRGBDReplay('/nonexistent/sealed_test.json')
    with pytest.raises(SensorContractError):CachedRGBDReplay('/nonexistent/sealed_private/trial')
