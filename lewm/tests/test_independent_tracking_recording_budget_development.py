"""Real sampling/snapshot methods with synthetic native getters; no Genesis."""
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
from PIL import Image

from lewm import independent_tracking_recording_budget_development as mod
from scripts import independent_tracking_artifacts_development as storage
from scripts.independent_tracking_snapshot_development import persist_snapshot
from scripts.independent_tracking_session_development import IndependentTrackingSession
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from lewm.simulated_body_observation_development import IdealBodySensor,BodyObservationBuffer
from lewm.simulated_fast_gyro_development import IdealFastGyro
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.tests.test_independent_tracking_collection_development import snapshot_session,store
from lewm.tests.independent_tracking_native_contact_fixtures import runtime
from scripts.independent_tracking_native_contact_guard_development import NativeContactGuard


def synthetic_native_session(output):
    """Only native getters and RGB rasterization are fake; recorders are real."""
    s=IndependentTrackingSession.__new__(IndependentTrackingSession)
    s.__dict__.update(vars(snapshot_session()))
    solver,gs,_=runtime();s.contact_integrity=NativeContactGuard(solver,gs)
    s.output=output;s.phase=0;s.edge_index=0;s.guard=None;s.guard_rows=[]
    s.sensor=IdealBodySensor();s.observations=BodyObservationBuffer((0,0,0))
    s.fast_sensor=IdealFastGyro();s.fast_buffer=FastGyroBuffer((0,0,0))
    from lewm.simulated_body_observation_development import JOINT_NAMES
    s.joint_names=JOINT_NAMES
    s._contact_topology=dict(robot={1,2},support={1},ground={10})
    s.link_names={1:'foot',2:'body',10:'ground',20:'wall'};s.object_ids={10:'ground',20:'wall'}
    state=NS(count=0,stop=False)
    def contacts(**kwargs):
        assert kwargs==dict(exclude_self_contact=False)
        # Foot/ground, self-contact, and a masked nonrobot contact. All raw
        # rows must survive, even though two do not enter external attribution.
        n=state.count%4
        la=np.array([[1,1,10]],np.int32)[:,:n].copy()
        lb=np.array([[10,2,20]],np.int32)[:,:n].copy()
        if state.stop:la=np.array([[2]],np.int32);lb=np.array([[10]],np.int32);n=1
        return dict(geom_a=la.copy(),geom_b=lb.copy(),link_a=la,link_b=lb,
            force_a=np.ones((1,n,3),np.float32),force_b=-np.ones((1,n,3),np.float32),
            position=np.zeros((1,n,3),np.float32),valid_mask=np.array([[True,True,False]])[:,:n].copy())
    robot=NS(get_pos=lambda:np.array([[0,0,.375]],np.float32),
        get_quat=lambda:np.array([[1,0,0,0]],np.float32),
        get_vel=lambda:np.zeros((1,3),np.float32),get_ang=lambda:np.zeros((1,3),np.float32),
        get_dofs_position=lambda _:np.zeros((1,12),np.float32),
        get_dofs_velocity=lambda _:np.zeros((1,12),np.float32),get_contacts=contacts)
    s.ctx=NS(build=NS(robot=robot),runner=NS(_as_np=np.asarray,_leg_dof_idx=np.arange(12)))
    def capture(directory,name):
        Image.fromarray(np.zeros((480,640,3),np.uint8)).save(directory/(name+'.png'))
        s.latest_native_depth=np.full((480,640),2.,np.float32)
        return dict(timestamp_s=float(s.samples[-1]['timestamp_s']),world_from_optical=np.eye(4).tolist(),
                    rigid_mount_no_obstacle_adjustment=True,rgb_sha256='a'*64)
    s.capture_fixed_rgb=capture
    def sample():
        state.count+=1
        return s._sample([.0,.0,.0],np.zeros(3,np.float32),state.count*.002)
    return s,state,sample


@pytest.mark.parametrize('stop',[False,True])
def test_actual_recorders_produce_the_calculated_numeric_layouts(store,stop):
    s,state,sample=synthetic_native_session(store.directory)
    for _ in range(750):sample()
    s.sensor_packets()
    for _ in range(2):
        for _ in range(50):sample()
        s.sensor_packets()
    if stop:
        state.stop=True
        with pytest.raises(PhysicalStop,match='DISALLOWED_CONTACT'):sample()
    assert len(s.samples)==len(s.packets)==len(s.fast_rows)==850+int(stop)
    assert len(s.sensor_rows)==85 and len(s.model_manifest)==3
    count=sum(p['link_a'].shape[1] for p in s.packets)
    persist_snapshot(s,store)
    for name in mod.numeric_envelope():
        with np.load(store.directory/name,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        report=mod.validate_numeric_archive(name,arrays,physics_samples=len(s.samples),frames=3,contact_rows=count)
        assert report['numeric_layout_verified'] and not report['measurement_validity_verified']
        layout=mod.numeric_layouts(len(s.samples),3,count)[name]
        assert sum(v.nbytes for v in arrays.values())==sum(
            int(np.prod(r['shape']))*np.dtype(r['dtype']).itemsize for r in layout.values())
        assert (store.directory/name).stat().st_size<=mod.numeric_envelope()[name]['serialization_ceiling_bytes']
    with np.load(store.directory/'native_contacts.npz') as a:
        assert a['frame_offsets'][-1]==count and len(a['valid_mask'])==count
        assert (~a['valid_mask']).any()  # No discarded invalid rows.
        assert ((a['link_a']==1)&(a['link_b']==2)).any()  # No discarded self contacts.
    assert bool(s.contact_events[-1]['disallowed_contacts'])==stop
    # All methods above consumed fake getters, not physical measurements.
    assert not report['native_recording_performed'] and not report['navigation_qualified']


def test_full_envelope_is_calculated_without_allocating_full_tapes():
    r=mod.numeric_envelope()
    assert {k:v['raw_array_bytes'] for k,v in r.items()}=={
        'physics_trace.npz':8660150,'native_contacts.npz':908653108,
        'ideal_sensor_samples.npz':635230,'policy_histories.npz':3110303,
        'fast_gyro_samples.npz':982550,'fast_gyro_histories.npz':971499}
    assert r['native_contacts.npz']['serialization_ceiling_bytes']==1000239314
    assert sum(v['serialization_ceiling_bytes'] for v in r.values())==1019246279
    contract=storage.episode_resource_contract()
    assert contract['static_recording_serialization_ceiling_bytes']==(
        1019246279+sum(n.endswith('.json') for n in storage.STATIC)*storage.JSON_BYTES)
    assert contract['static_recording_serialization_ceiling_bytes']<storage.DEFERRED_BYTES
    assert contract['internal_serialized_file_ceilings_enforced']
    assert not contract['memory_bound_proved'] and not contract['native_recording_bound_proved']


@pytest.mark.parametrize('counts',[(True,0,0),(-1,0,0),(22851,0,0),(1,444,0),(0,1,0),(1,0,751)])
def test_unbounded_or_inconsistent_counts_are_rejected(counts):
    with pytest.raises(ValueError):mod.numeric_layouts(*counts)


def test_zero_samples_do_not_require_invented_empty_measurements(store):
    s=synthetic_native_session(store.directory)[0];persist_snapshot(s,store)
    for name in mod.numeric_envelope():
        with np.load(store.directory/name,allow_pickle=False) as a:assert a.files==[]
        assert mod.validate_numeric_archive(name,{},physics_samples=0,frames=0,contact_rows=0)['numeric_layout_verified']


@pytest.mark.parametrize('fault',['member','dtype','shape'])
def test_shape_checker_does_not_silently_cast_fill_or_repair(fault):
    specs=mod.numeric_layouts(10,0,0)['physics_trace.npz']
    arrays={k:np.zeros(r['shape'],dtype=r['dtype']) for k,r in specs.items()}
    if fault=='member':arrays.pop('edge_index')
    elif fault=='dtype':arrays['base_pose_world']=arrays['base_pose_world'].astype(np.float32)
    else:arrays['base_pose_world']=arrays['base_pose_world'][:9]
    with pytest.raises(ValueError):mod.validate_numeric_archive('physics_trace.npz',arrays,
        physics_samples=10,frames=0,contact_rows=0)


def test_numeric_limit_precedes_compression_and_file_creation(store,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('oversized array cannot reach compression')
    monkeypatch.setattr(storage.np,'savez_compressed',forbidden)
    large=np.broadcast_to(np.zeros((),np.float64),(2_000_000,))
    with pytest.raises(storage.StorageStop,match='NUMERIC_RECORDING'):
        store.npz('physics_trace.npz',dict(unexpected=large))
    assert not (store.directory/'physics_trace.npz').exists()


def test_bounded_json_matches_original_encoding_and_exact_limit(store,monkeypatch):
    value=dict(a=[None,np.float64(2.),'line\nquoted"汉'],b=np.arange(4))
    payload=storage.encode(value);monkeypatch.setattr(storage,'JSON_BYTES',len(payload))
    assert storage.bounded_json(value)==payload
    store.json('result.json',value)
    assert (store.directory/'result.json').read_bytes()==payload
    monkeypatch.setattr(storage,'JSON_BYTES',len(payload)-1)
    with pytest.raises(storage.StorageStop,match='JSON_RECORDING'):store.json('failure.json',value)
    assert not (store.directory/'failure.json').exists()


def test_oversized_json_does_not_consume_entire_encoder_or_bypass_raw_writer(store,monkeypatch):
    monkeypatch.setattr(storage,'JSON_BYTES',32)
    original=storage.json.JSONEncoder.iterencode;consumed=[]
    def counted(self,value,*a,**k):
        for chunk in original(self,value,*a,**k):consumed.append(chunk);yield chunk
    monkeypatch.setattr(storage.json.JSONEncoder,'iterencode',counted)
    with pytest.raises(storage.StorageStop):store.json('result.json',list(range(1000)))
    assert len(consumed)<20 and not (store.directory/'result.json').exists()
    with pytest.raises(storage.StorageStop):store.write_bytes('result.json',b'x'*33)
    assert not (store.directory/'result.json').exists()
    # A separate small failure record remains writable; no retry of old output.
    store.json('failure.json',dict(failed=True))
    assert json.loads((store.directory/'failure.json').read_text())==dict(failed=True)
