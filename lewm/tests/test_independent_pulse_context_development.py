"""Synthetic constructor, causal selector, prefix, stop and target tests."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.independent_pulse_context_development import (
    EDGES,PITCH,TRIALS,SUPPORTS,WARMUP_TICKS,WARMUP_COMMAND,geometry,specification,pack,schedule,decision)
from lewm.simulated_body_observation_development import BodyObservationBuffer,IdealBodySensor,JOINT_NAMES
from lewm.causal_depth_observation_development import from_native_depth
from lewm.pulse_timed_observation_pairing_development import pulse_window
from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from scripts.pulse_context_sensor_audit_development import native_depth_comparison_mask
from scripts.audit_go2_independent_pulse_context_pilot_v1 import (
    PREFIX_SAMPLES,PREFIX_FRAMES,prefix_witness,compare_prefixes,fingerprint,audit_commands,audit_stops,target_row)


def policy(tick):
    now=1_500_000_000+tick*100_000_000;buffer=BodyObservationBuffer((0,0,0));sensor=IdealBodySensor()
    for stamp in range(20_000_000,now+1,20_000_000):
        value=sensor.sample(measured_ns=stamp,quaternion_xyzw=[0,0,0,1],velocity_world=[0,0,0],
            angular_velocity_world=[0,0,0],joint_position=np.zeros(12),joint_velocity=np.zeros(12),joint_names=JOINT_NAMES)
        buffer.append_sensors(value,stamp)
        if stamp%100_000_000==0:buffer.append_applied_command([0,0,0],stamp)
    return buffer.packet(np.zeros((480,640,3),np.uint8),now)


def test_connected_nontrivial_geometry_and_exact_passages():
    cells={p for edge in EDGES for p in edge};seen={(0,0)}
    while True:
        expanded=seen|{b for a,b in EDGES if a in seen}|{a for a,b in EDGES if b in seen}
        if expanded==seen:break
        seen=expanded
    assert seen==cells and len(EDGES)>=len(cells)
    degrees={p:sum(p in e for e in EDGES) for p in cells}
    assert min(degrees.values())==1 and max(degrees.values())>=3
    walls=geometry()['wall_boxes'];assert len({w['wall_id'] for w in walls})==len(walls)
    passages={frozenset(e) for e in EDGES}
    for x,y in cells:
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            mid=((x+dx/2)*PITCH,(y+dy/2)*PITCH)
            actual=[w for w in walls if np.allclose(w['centre_xyz'][:2],mid,rtol=0,atol=1e-10)]
            assert len(actual)==(0 if frozenset(((x,y),(x+dx,y+dy))) in passages else 1)
    assert any(w['centre_xyz'][:2]==[.6,0.] for w in walls)


def test_all_actions_keep_one_train_layout_and_support_condition():
    specs=[specification(t) for t in TRIALS]
    assert len(specs)==12 and len({s['layout_id'] for s in specs})==1
    assert all(s['data_role']==s['evaluation_layout']['role']=='train' for s in specs)
    for support in SUPPORTS:
        siblings=[s for s in specs if s['condition']==support]
        assert {s['action_index'] for s in siblings}==set(range(6))
        for key in ('procedural_seed','appearance_seed','friction_mu','geometry','evaluation_layout'):
            assert all(s[key]==siblings[0][key] for s in siblings)
    for s in specs:
        p=pack(s);assert len(p.static_objects)==len(geometry()['wall_boxes'])
        assert p.robot.spawn_xyz_m==(.08,0.,.375) and p.physics_randomization.floor_friction_mu==s['friction_mu']
    broken=deepcopy(specs[0]);broken['friction_mu']=.8
    with pytest.raises(ValueError):pack(broken)
    with pytest.raises(ValueError):specification('nominal_action_6')


@pytest.mark.parametrize('a',range(6))
def test_six_fixed_schedules_and_tracking_free_selector(a):
    spec=specification('nominal_action_'+str(a));rows=schedule(a);count=spec['pulse_ticks']
    assert len(rows)==8+count+20
    assert all(r['requested_command']==list(WARMUP_COMMAND) for r in rows[:8])
    assert all(r['requested_command']==spec['command'] for r in rows[8:8+count])
    assert all(r['requested_command']==[0.,0.,0.] for r in rows[8+count:])
    selected=decision(a,8,policy(8))
    assert selected['requested_command']==spec['command'] and not selected['tracker_required']
    assert not selected['native_state_used'] and not selected['terminal']
    assert decision(a,len(rows),policy(len(rows)))['terminal']
    with pytest.raises(ValueError):decision(a,len(rows)+1,policy(len(rows)+1))


@pytest.mark.parametrize('fault',['clock','privilege','future','identity','tick_bool','action_bool'])
def test_selector_rejects_invalid_policy_and_indices(fault):
    p=policy(8);tick=8;a=0
    if fault=='clock':tick=9
    elif fault=='privilege':p['native_pose']=[0]*7
    elif fault=='future':p['sensor_state']['sensed']['gyro']['available_ns'][-1]+=1
    elif fault=='identity':p['sensor_state']['identity']=(0,1,0)
    elif fault=='tick_bool':tick=True
    elif fault=='action_bool':a=True
    with pytest.raises(ValueError):decision(a,tick,p)


def test_near_native_depth_comparison_does_not_change_public_sensor_validity():
    expected=np.array([.05,.055,.1,.2,1.,4.98,np.nan])
    assert native_depth_comparison_mask(expected,np.ones(7,bool)).tolist()==[False,False,True,True,True,False,False]
    with pytest.raises(ValueError):native_depth_comparison_mask(expected,np.ones(7))
    p=policy(8);raw=np.full((480,640),.1,np.float32)
    d=from_native_depth(raw,p,measured_ns=2_300_000_000,available_ns=2_300_000_000,now_ns=2_300_000_000)
    assert not d['valid'].any()


def prefix_fixture(n=1150,frames=9):
    raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=np.zeros((n,7)),requested_command=np.zeros((n,3)))
    contacts=dict(frame_offsets=np.arange(n+1),frame_timestamp_s=raw['timestamp_s'],force_a=np.zeros((n,3)))
    reader=SimpleNamespace(frames=[dict(decision_ns=1_500_000_000+i*100_000_000) for i in range(frames)],
        packet=lambda i:({'rgb':np.full((2,2,3),i,np.uint8)}, {'valid':np.ones((2,2),bool)}, {},1_500_000_000+i*100_000_000))
    return raw,contacts,reader


def test_prefix_is_all1150_native_samples_and_nine_actual_packets():
    assert PREFIX_SAMPLES==1150 and PREFIX_FRAMES==9
    raw,c,r=prefix_fixture();a=prefix_witness(raw,c,r)
    assert compare_prefixes(a,prefix_witness(raw,c,r))['matched']
    for name,index in (('base_pose_world',0),('requested_command',1149)):
        changed=deepcopy(raw);changed[name][index,0]=.001
        report=compare_prefixes(a,prefix_witness(changed,c,r))
        assert report['unequal_fields']==['native/'+name] and not report['matched']
    changed=deepcopy(c);changed['force_a'][-1,0]=1
    assert compare_prefixes(a,prefix_witness(raw,changed,r))['unequal_fields']==['contact/force_a']
    raw,c,r=prefix_fixture();r.packet=lambda i:({'rgb':np.full((2,2,3),i+1,np.uint8)},{'valid':np.ones((2,2),bool)}, {},1_500_000_000+i*100_000_000)
    assert len(compare_prefixes(a,prefix_witness(raw,c,r))['unequal_fields'])==9


@pytest.mark.parametrize('n,frames',[(1149,9),(1150,8),(1,0)])
def test_missing_prefix_is_not_a_matched_counterfactual(n,frames):
    a=prefix_witness(*prefix_fixture());b=prefix_witness(*prefix_fixture(n,frames))
    assert b['status']=='MISSING_DEPARTURE_PREFIX' and compare_prefixes(a,b)['status']=='UNAVAILABLE_PREFIX'


def test_fingerprint_preserves_type_shape_and_full_history():
    assert fingerprint(np.zeros((2,2),np.float32))!=fingerprint(np.zeros(4,np.float32))
    assert fingerprint(np.zeros(4,np.float32))!=fingerprint(np.zeros(4,np.float64))
    assert fingerprint([1,2])!=fingerprint((1,2))
    with pytest.raises(ValueError):fingerprint(np.array([{}],object))


def trace_fixture(a=0):
    planned=schedule(a);n=750+50*len(planned)
    raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=np.tile([0.,0.,.35,0.,0.,0.,1.],(n,1)),
        base_twist_world=np.zeros((n,6)),requested_command=np.zeros((n,3),np.float32),
        applied_command=np.zeros((n,3),np.float32),phase=np.zeros(n,np.uint8),physics_contact=np.zeros(n,np.uint8))
    tape=[];rows=[]
    for i,row in enumerate(planned):
        lo=749+50*i;hi=lo+50
        raw['requested_command'][lo+1:hi+1]=row['requested_command']
        raw['applied_command'][lo+1:hi+1]=raw['applied_command'][lo]+np.clip(
            np.asarray(row['requested_command'],np.float32)-raw['applied_command'][lo],[-.25,0,-.35],[.25,0,.35])
        raw['phase'][lo+1:hi+1]=row['phase']
        tape.append(row|dict(tick=i,pre_sample_index=lo,post_sample_index=hi,completed=True))
        rows.append(dict(tick=i,decision=row|dict(terminal=False)))
    rows.append(dict(tick=len(planned),decision=dict(terminal=True)))
    result=dict(command_ticks=len(tape),decisions=len(rows),completed_ticks=len(tape),departure_present=True,
        physical_stop=None,acquisition_stop=None,setup_admitted=True,schedule_terminal='FIXED_CONTEXT_PULSE_COMPLETE')
    return raw,tape,rows,result


@pytest.mark.parametrize('a',range(6))
def test_auditor_reconstructs_all_command_cells_and_exact_target_times(a):
    raw,tape,rows,result=trace_fixture(a);audit_commands(raw,tape,rows,result,a)
    frames=[dict(decision_ns=1_500_000_000+i*100_000_000,image_ns=1_500_000_000+i*100_000_000) for i in range(len(rows))]
    s=specification('nominal_action_'+str(a))
    w=dict(condition=s['trial'],departure_tick=8,decision_ns=2_300_000_000,action_index=a,command=s['command'],pulse_ticks=s['pulse_ticks'])|pulse_window(
        frames,tape,departure_tick=8,departure_ns=2_300_000_000,command=tuple(s['command']),pulse_ticks=s['pulse_ticks'])
    labels=target_row(w,RecordedPulseNativeTargets(raw).labels(w));d=PulseTimedDataset([w],[labels],{s['trial']:dict(role='train',layout_id=s['layout_id'])})
    assert len(d)==1 and sum(t['motion_valid'] for t in labels['targets'])==5
    assert labels['targets'][4]['offset_ns']==(2_200_000_000 if a%2==0 else 2_500_000_000)
    bad=deepcopy(raw);bad['requested_command'][1200,0]+=.01
    with pytest.raises(AssertionError):audit_commands(bad,tape,rows,result,a)


@pytest.mark.parametrize('reason',['DISALLOWED_CONTACT','BODY_STABILITY_LIMIT'])
def test_native_stop_must_be_terminal_not_merely_present(reason):
    raw,_,_,_=trace_fixture();raw={k:v[:31].copy() for k,v in raw.items()}
    if reason=='DISALLOWED_CONTACT':raw['physics_contact'][-1]=1
    else:raw['base_pose_world'][-1,2]=.14
    result=dict(setup_admitted=False,physical_stop=reason)
    report=audit_stops(raw,{}, {},[],None,[],result)
    assert report==dict(sample_index=30,reason=reason)
    if reason=='DISALLOWED_CONTACT':raw['physics_contact'][29]=1
    else:raw['base_pose_world'][29,2]=.14
    with pytest.raises(AssertionError):audit_stops(raw,{}, {},[],None,[],result)


def test_native_partial_stop_does_not_require_fabricated_setup_or_commands():
    raw,_,_,_=trace_fixture();raw={k:v[:31].copy() for k,v in raw.items()}
    result=dict(command_ticks=0,decisions=0,completed_ticks=0,departure_present=False,
        physical_stop='BODY_STABILITY_LIMIT',acquisition_stop=None,setup_admitted=False,schedule_terminal=None)
    audit_commands(raw,[],[],result,0)


def test_new_session_preserves_native_recorder_chain():
    from scripts.independent_pulse_context_session_development import PulseContextSession
    from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    mro=PulseContextSession.__mro__
    assert mro.index(AttributedSession)<mro.index(PulseContextPhysicalInit)


@pytest.mark.parametrize('contact',[False,True])
def test_interrupted_candidate_preserves_positive_contact_but_censors_missing_motion(contact):
    raw,tape,rows,result=trace_fixture();end=1171
    raw={k:v[:end].copy() for k,v in raw.items()};raw['physics_contact'][-1]=int(contact)
    tape=tape[:9];tape[-1]['post_sample_index']=end-1;tape[-1]['completed']=False
    frames=[dict(decision_ns=1_500_000_000+i*100_000_000,image_ns=1_500_000_000+i*100_000_000) for i in range(9)]
    s=specification('nominal_action_0')
    w=dict(condition=s['trial'],departure_tick=8,decision_ns=2_300_000_000,action_index=0,command=s['command'],pulse_ticks=2)|pulse_window(
        frames,tape,departure_tick=8,departure_ns=2_300_000_000,command=tuple(s['command']),pulse_ticks=2)
    labels=target_row(w,RecordedPulseNativeTargets(raw).labels(w))
    assert all(not t['motion_valid'] and not t['image_target_valid'] for t in labels['targets'])
    assert sum(t['contact_valid'] for t in labels['targets'])==(5 if contact else 0)
    assert sum(t['contact']==1. for t in labels['targets'])==(5 if contact else 0)
    PulseTimedDataset([w],[labels],{s['trial']:dict(role='train',layout_id=s['layout_id'])})


def test_partial_sensor_audit_reconstructs_real_available_samples_without_images(monkeypatch,tmp_path):
    import scripts.pulse_context_sensor_audit_development as audit
    from lewm.simulated_fast_gyro_development import IdealFastGyro
    raw,_,_,_=trace_fixture();raw={k:v[:31].copy() for k,v in raw.items()}
    raw['joint_position']=np.zeros((31,12));raw['joint_velocity']=np.zeros((31,12))
    bs=[];fs=[];body=IdealBodySensor();fast=IdealFastGyro()
    for i in range(31):
        stamp=(i+1)*2_000_000
        value,valid=fast.sample(measured_ns=stamp,quaternion_xyzw=raw['base_pose_world'][i,3:],angular_velocity_world=np.zeros(3))
        fs.append(dict(values=value,valid=valid,measured_ns=stamp,available_ns=stamp))
        if stamp%20_000_000==0:
            values=body.sample(measured_ns=stamp,quaternion_xyzw=raw['base_pose_world'][i,3:],velocity_world=np.zeros(3),
                angular_velocity_world=np.zeros(3),joint_position=np.zeros(12),joint_velocity=np.zeros(12),joint_names=JOINT_NAMES)
            bs.append(dict(measured_ns=stamp,**{k+'_'+field:pair[j] for k,pair in values.items() for j,field in enumerate(('values','valid'))}))
    stack=lambda rows:{k:np.stack([r[k] for r in rows]) for k in rows[0]}
    contacts=dict(frame_offsets=np.zeros(32,np.int64),frame_timestamp_s=raw['timestamp_s'],
        **{k:np.empty((0,3)) if k in ('force_a','force_b','position') else np.empty(0,int) for k in audit.CONTACT_FIELDS})
    topology=dict(ground_link_ids=[1],environment_object_ids={'1':'ground'},native_environment_count=1,selected_environment_index=0)
    events=[dict(sample_index=i,timestamp_s=float(t),phase=0,disallowed_contacts=[]) for i,t in enumerate(raw['timestamp_s'])]
    js={'contact_topology.json':topology,'floor_roles.json':dict(physical_ground_link_ids=[1],visual_only_link_ids=[2],visual_collision_geom_count=0),
        'contact_events.json':events,'camera_audit.json':[],'depth_camera_audit.json':[],
        'floor_visual_collision_identity.json':{},'terminal_environment_identity.json':{},
        'policy_observations.json':dict(frames=[]),'depth_observations.json':dict(frames=[])}
    arrays={'physics_trace.npz':raw,'native_contacts.npz':contacts,'ideal_sensor_samples.npz':stack(bs),'fast_gyro_samples.npz':stack(fs),
        'policy_histories.npz':{},'fast_gyro_histories.npz':{}}
    monkeypatch.setattr(audit,'read_json',lambda d,n:deepcopy(js[n]));monkeypatch.setattr(audit,'read_npz',lambda d,n:deepcopy(arrays[n]))
    monkeypatch.setattr(audit,'classify',lambda p,t:[])
    r=audit.audit_sensors(tmp_path,specification('nominal_action_0'),dict(physics_samples=31,rgbd_frames=0,setup_checked=False,setup_admitted=False))[-1]
    assert r['sensor_contact_reconstruction_exact'] and r['partial_setup'] and not r['complete_settling']
    assert r['paired_frames_reconstructed']==0 and not r['native_wall_inventory_checked']
    arrays['fast_gyro_samples.npz']['values'][-1,0]+=.01
    with pytest.raises(AssertionError):audit.audit_sensors(tmp_path,specification('nominal_action_0'),dict(physics_samples=31,rgbd_frames=0,setup_checked=False,setup_admitted=False))


@pytest.mark.parametrize('infrastructure_failure',[False,True])
def test_collection_continues_physical_failures_but_never_retries_infrastructure(monkeypatch,tmp_path,infrastructure_failure):
    import scripts.run_go2_independent_pulse_context_pilot_v1 as runner
    import json
    output=tmp_path/'attempt';calls=[]
    monkeypatch.setattr(runner,'OUTPUT',output)
    monkeypatch.setattr(runner,'preflight',lambda:dict(source_sha256={runner.PROTOCOL:'0'*64}))
    monkeypatch.setattr(runner,'create_output',lambda p:p.mkdir())
    monkeypatch.setattr(runner,'verify',lambda p:None);monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(runner,'artifacts',lambda c,r:[])
    def collect(c,definition):
        calls.append(c)
        if infrastructure_failure and len(calls)==2:raise RuntimeError('synthetic infrastructure fault')
        return dict(physical_stop='DISALLOWED_CONTACT',departure_present=False)
    monkeypatch.setattr(runner,'collect',collect)
    if infrastructure_failure:
        with pytest.raises(RuntimeError):runner.main()
        assert calls==list(TRIALS[:2]) and not (output/'result.json').exists()
        assert json.loads((output/'failure.json').read_text())['completed_conditions']==[TRIALS[0]]
    else:
        runner.main();assert calls==list(TRIALS)
        assert len(json.loads((output/'result.json').read_text())['conditions'])==12
    with pytest.raises(ValueError,match='exclusive'):runner.main()


def test_context_setup_encloses_every_primitive_without_room_sized_empty_space():
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.setup_snapshot_evaluation_development import check_setup_snapshot
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    from scripts.pulse_context_setup_development import context_priors
    from scripts.fresh_maze_session_development import priors as room_priors
    geom=ArticulatedCollisionGeometry(URDF);q=np.repeat([0.,.8,-1.5],4)
    velocity,region=context_priors('0'*64,geom,q);shapes=geom.supports(q,np.eye(3))['shapes']
    contained=region.query([s['lower'] for s in shapes],[s['upper'] for s in shapes],[.04]*len(shapes),
        identity=(0,0,0),now_ns=1_500_000_000,observed_conflict=[False]*len(shapes))
    assert len(shapes)==27 and contained['conditional_setup_non_floor_clearance'].all()
    assert not contained['navigation_qualified'] and not contained['observed_free_space']
    walls=[dict(native_name=w['wall_id'],native_position=w['centre_xyz'],native_quaternion_wxyz=[1.,0.,0.,0.],
        native_box_size=w['size_xyz']+[0.]*4,fixed=True,collision_enabled=True,native_collision_boxes=1) for w in geometry()['wall_boxes']]
    kw=dict(identity=(0,0,0),measured_ns=1_500_000_000,position_world_m=[.08,0.,.35],rotation_world_from_initial_body=np.eye(3),
        velocity_world_m_s=[0.,0.,0.],native_static_boxes=walls,expected_nonfloor_names=[w['native_name'] for w in walls],geometry=geom,joint_position=q)
    assert not check_setup_snapshot(*room_priors('0'*64),**kw)['native_nonfloor_region_clear']
    assert check_setup_snapshot(velocity,region,**kw)['velocity_and_nonfloor_setup_checks_pass']
    extended=np.tile([0.,.8,-1.5],4)
    assert not check_setup_snapshot(*context_priors('0'*64,geom,extended),**(kw|dict(joint_position=extended)))['velocity_and_nonfloor_setup_checks_pass']
    wall=next(w for w in walls if w['native_position'][:2]==[.6,0.]);wall['native_position'][0]=.1
    assert not check_setup_snapshot(velocity,region,**kw)['velocity_and_nonfloor_setup_checks_pass']
    assert context_priors('0'*64,geom,q)==(velocity,region)
