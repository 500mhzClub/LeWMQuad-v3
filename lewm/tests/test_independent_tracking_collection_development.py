"""Actual bounded writes and synthetic episode lifecycles, no native launch."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lewm.independent_tracking_challenge_development import TRIALS, MAX_TICKS, specification
from scripts.independent_tracking_artifacts_development import EpisodeStore, StorageStop, frame_names
from lewm.tests.independent_tracking_native_contact_fixtures import MockCollectionGuard


@pytest.fixture
def store(tmp_path, monkeypatch):
    import scripts.navigation_artifact_root_development as guard
    base = tmp_path / 'artifacts'; base.mkdir()
    monkeypatch.setattr(guard, 'BASE', base)
    root = base / 'go2_tracking_synthetic_attempt_001'; root.mkdir()
    return EpisodeStore(root, TRIALS[0])


def test_exclusive_store_canonical_roster_and_no_protected_or_unlisted_paths(store):
    for name in ('../outside', '/tmp/outside', 'sealed_test.json', 'sealed/a', 'sealed_x/a', 'unlisted.json'):
        with pytest.raises(ValueError): store.json(name, {})
    store.json('result.json', {'one': np.int64(1)})
    receipt = store.verify()
    assert receipt['artifact_sizes']['result.json'] == (store.directory/'result.json').stat().st_size
    with pytest.raises(ValueError): store.json('result.json', {})
    with pytest.raises(ValueError): EpisodeStore(store.directory.parent, store.directory.name)


def test_store_rejects_symlink_target_before_read_or_write(store, tmp_path):
    target = tmp_path / 'outside'; target.write_bytes(b'private')
    (store.directory/'result.json').symlink_to(target)
    with pytest.raises(ValueError): store.json('result.json', {})
    assert target.read_bytes() == b'private'


def test_npz_roundtrip_real_serialization_and_object_rejection(store):
    arrays = dict(a=np.arange(20, dtype=np.int64), b=np.ones((3, 4), np.float32))
    store.npz('physics_trace.npz', arrays)
    with np.load(store.directory/'physics_trace.npz', allow_pickle=False) as z:
        assert set(z.files) == set(arrays)
        for k in arrays: np.testing.assert_array_equal(z[k], arrays[k])
    with pytest.raises(ValueError): store.npz('native_contacts.npz', {'a': np.array([object()])})
    assert not (store.directory/'native_contacts.npz').exists()
    with pytest.raises(ValueError): store.json('result.json', {'nan': float('nan')})


def test_budget_and_free_space_stop_before_opening_output(store, monkeypatch):
    import scripts.independent_tracking_artifacts_development as mod
    monkeypatch.setattr(mod, 'EPISODE_BYTES', 10)
    with pytest.raises(StorageStop): store.write_bytes('result.json', b'01234567890')
    assert not (store.directory/'result.json').exists()
    monkeypatch.setattr(mod, 'EPISODE_BYTES', 3*1024**3)
    monkeypatch.setattr(mod.shutil, 'disk_usage', lambda _: NS(free=mod.RESERVE_BYTES))
    with pytest.raises(StorageStop): store.write_bytes('result.json', b'x')
    assert not (store.directory/'result.json').exists()


def test_all_full_tape_reservations_fit_without_assuming_compression_or_memory(monkeypatch):
    from scripts import independent_tracking_artifacts_development as mod
    r=mod.episode_resource_contract()
    assert r['maximum_frames']==443
    assert r['all_reservations_bytes']==(
        443*6*1024**2 + 128*1024**2 + 8*1024**2 + 2*1024**3)
    assert r['all_reservations_bytes']<r['episode_bytes']==5*1024**3
    assert not r['native_recording_bound_proved'] and not r['memory_bound_proved']
    assert not r['external_os_quota_enforced']
    # Historical 3GiB / 512MiB allocation did not cover even its own
    # per-operation reservations for a complete 443-frame tape.
    monkeypatch.setattr(mod,'EPISODE_BYTES',3*1024**3)
    monkeypatch.setattr(mod,'DEFERRED_BYTES',512*1024**2)
    with pytest.raises(ValueError,match='reservations'):mod.episode_resource_contract()


def test_external_success_is_fsynced_bound_and_cannot_be_replaced(store):
    with store.external(('rgb_0000.png',), 100):
        (store.directory/'rgb_0000.png').write_bytes(b'fake image')
    assert store.verify()['artifact_bytes'] == 10
    with pytest.raises(ValueError):
        with store.external(('rgb_0000.png',), 100): pass
    (store.directory/'rgb_0000.png').write_bytes(b'corruption')
    with pytest.raises(ValueError): store.verify()


def test_internal_flush_failure_binds_present_bytes_and_forbids_overwrite(store,monkeypatch):
    import scripts.independent_tracking_artifacts_development as mod
    fsync=mod.os.fsync
    def fail(_): raise OSError('synthetic fsync failure')
    monkeypatch.setattr(mod.os,'fsync',fail)
    with pytest.raises(OSError): store.write_bytes('result.json',b'partly durable evidence')
    monkeypatch.setattr(mod.os,'fsync',fsync)
    receipt=store.verify()
    assert receipt['failed_internal'] and 'result.json' in receipt['artifact_sha256']
    assert receipt['artifact_bytes']==len(b'partly durable evidence')
    with pytest.raises(ValueError): store.write_bytes('result.json',b'replacement')


def test_external_sync_failure_still_accounts_for_all_present_outputs(store,monkeypatch):
    import scripts.independent_tracking_artifacts_development as mod
    fsync=mod.os.fsync
    def fail(_): raise OSError('synthetic external sync failure')
    monkeypatch.setattr(mod.os,'fsync',fail)
    names=('rgb_0000.png','native_depth_0000.npz')
    with pytest.raises(OSError):
        with store.external(names,10):
            for name in names: (store.directory/name).write_bytes(b'123')
    monkeypatch.setattr(mod.os,'fsync',fsync)
    receipt=store.verify()
    assert receipt['failed_external'] and receipt['artifact_bytes']==6
    assert set(receipt['artifact_sha256'])==set(names)


@pytest.mark.parametrize('fault', ['partial', 'missing', 'oversize'])
def test_failed_external_writer_retains_present_files_and_prevents_new_capture(store, fault):
    names = ('rgb_0000.png', 'native_depth_0000.npz')
    with pytest.raises((RuntimeError, ValueError, StorageStop)):
        with store.external(names, 10):
            if fault != 'missing': (store.directory/names[0]).write_bytes(b'12345678901' if fault == 'oversize' else b'123')
            if fault == 'partial': raise RuntimeError('capture failed')
    assert store.failed_external
    assert (names[0] in store.bindings) == (fault != 'missing')
    assert store.used == (11 if fault == 'oversize' else 3 if fault == 'partial' else 0)
    with pytest.raises(ValueError):
        with store.external(('rgb_0001.png',), 10): pass
    store.json('failure.json', {'retained': True})


def test_frame_roster_exact_bounds():
    assert len(frame_names(442)) == 4
    for i in (-1, 443, True):
        with pytest.raises(ValueError): frame_names(i)


def snapshot_session():
    return NS(samples=[], packets=[], packet_times=[], contact_events=[], link_names={1:'foot'}, object_ids={2:'floor'},
        _contact_topology=dict(robot={1}, support={1}, ground={2}), sensor_rows=[], packet_rows=[],
        fast_rows=[], fast_packets=[], model_manifest=[], image_audit=[], depth_manifest=[], depth_audit=[])


def test_empty_snapshot_does_not_invent_samples_or_sensor_histories(store):
    from scripts.independent_tracking_snapshot_development import persist_snapshot
    persist_snapshot(snapshot_session(), store)
    for name in ('physics_trace', 'native_contacts', 'ideal_sensor_samples', 'policy_histories', 'fast_gyro_samples', 'fast_gyro_histories'):
        with np.load(store.directory/(name+'.npz'), allow_pickle=False) as z: assert z.files == []
    p=json.loads((store.directory/'policy_observations.json').read_text())
    assert not p['frames'] and p['schema']=='causal_rgb_body_routes_development.v1'
    store.verify()


def test_snapshot_preserves_partial_sensor_lengths_and_dtypes(store):
    from scripts.independent_tracking_snapshot_development import persist_snapshot, stack
    s=snapshot_session(); s.samples=[dict(timestamp_s=np.float64(.002), phase=np.uint8(0))]
    s.fast_rows=[dict(measured_ns=np.int64(2_000_000), values=np.zeros(3,np.float64))]
    # Partial body and visual records remain absent; do not fill to fast rate.
    persist_snapshot(s,store)
    with np.load(store.directory/'physics_trace.npz') as z:
        assert z['phase'].dtype==np.uint8 and z['timestamp_s'].tolist()==[.002]
    with np.load(store.directory/'ideal_sensor_samples.npz') as z: assert z.files==[]
    with pytest.raises(ValueError): stack([dict(a=1),dict(b=2)])


@pytest.mark.parametrize('populated',[False,True])
def test_new_snapshot_semantics_equal_frozen_recorder_outputs(store,tmp_path,populated):
    from scripts.independent_tracking_snapshot_development import persist_snapshot
    from scripts.independent_tracking_session_development import IndependentTrackingSession
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession,CONTACT_FIELDS
    from scripts.rgbd_session_development import RGBDSession
    s=IndependentTrackingSession.__new__(IndependentTrackingSession)
    s.__dict__.update(vars(snapshot_session()))
    if populated:
        s.samples=[dict(timestamp_s=np.float64(.002),phase=np.uint8(0),joint_position=np.arange(12,dtype=np.float64))]
        s.sensor_rows=[dict(measured_ns=np.int64(20_000_000),joints_values=np.arange(12,dtype=np.float64))]
        s.packet_rows=[dict(image_ns=np.int64(1_500_000_000),decision_ns=np.int64(1_500_000_000))]
        s.fast_rows=[dict(measured_ns=np.int64(2_000_000),values=np.zeros(3),valid=np.ones(3,bool))]
        s.fast_packets=[dict(measured_ns=np.arange(10,dtype=np.int64),values=np.zeros((10,3)))]
        s.model_manifest=[dict(rgb_file='rgb_0000.png',image_ns=1_500_000_000,decision_ns=1_500_000_000)]
        s.image_audit=[dict(physical_sample_index=749,timestamp_s=1.5)]
        s.depth_manifest=[dict(measured_ns=1_500_000_000,depth_file='depth_0000.npz')]
        s.depth_audit=[dict(native_depth_sha256='a'*64)]
        s.packets=[{k:np.zeros((1,2,3) if k in ('force_a','force_b','position') else (1,2),
            dtype=bool if k=='valid_mask' else float if k in ('force_a','force_b','position') else np.int64) for k in CONTACT_FIELDS}]
        s.packet_times=[.002]
    old=tmp_path/'explicit_old_snapshot';old.mkdir()
    AttributedSession.persist(s,old)
    RGBDSession.persist_observations(s,old)
    persist_snapshot(s,store)
    for name in store.bindings:
        if not populated and name=='native_contacts.npz':
            assert not (old/name).exists()  # Explicit empty archive is the declared sole difference.
            continue
        if name.endswith('.json'):
            assert json.loads((old/name).read_text())==json.loads((store.directory/name).read_text())
        else:
            with np.load(old/name,allow_pickle=False) as a,np.load(store.directory/name,allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:
                    assert a[key].dtype==b[key].dtype
                    np.testing.assert_array_equal(a[key],b[key])


@pytest.fixture
def native_mock(monkeypatch):
    import scripts.independent_tracking_collection_development as mod
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    state=NS(fault=None, events=[], commands=[], session=None)
    class Session:
        def __init__(self,spec,output):
            state.session=self; self.output=output; self.__dict__.update(vars(snapshot_session()))
            self.contact_integrity=MockCollectionGuard()
            self.spec=spec; self.guard_rows=[]; self.phase=0
            (output/'visual_meshes').mkdir()
            for name in ('ground_visual.ply','wall_union_visual.ply'):
                if state.fault=='mesh' and name=='wall_union_visual.ply': continue
                (output/'visual_meshes'/name).write_bytes(b'mesh')
            self.ctx=NS(build=NS(robot=object(),scene=NS(destroy=lambda:state.events.append('destroy')),
                collision_floor=NS(links=[NS(idx=2)],geoms=[NS(idx=3)]),
                visual_surfaces=[NS(links=[NS(idx=4)],geoms=[])]),
                runner=NS(_leg_dof_idx=np.arange(12)),policy=NS(env_cfg={}))
        def install_contact_identity(self): state.events.append('contacts')
        def settle_recorded(self):
            self.samples=[dict(timestamp_s=(i+1)*.002,requested_command=np.zeros(3,np.float64),
                applied_command=np.zeros(3,np.float64),phase=np.uint8(0))
                for i in range(1 if state.fault=='settle' else 750)]
            if state.fault=='settle': raise PhysicalStop('SETTLING_STOP')
        def sensor_packets(self):
            tick=(len(self.samples)-750)//50; i=len(self.model_manifest)
            for name in frame_names(i):
                (self.output/name).write_bytes(b'captured')
                if state.fault=='capture' and i==1: raise ValueError('partial capture failure')
            if i==0: (self.output/'floor_visual_collision_identity.json').write_text('{}')
            now=1_500_000_000+tick*100_000_000
            self.model_manifest.append(dict(rgb_file=f'rgb_{i:04d}.png',image_ns=now,decision_ns=now))
            self.depth_manifest.append({}); self.fast_packets.append(dict(a=np.zeros(1)))
            # Full-schedule lifecycle test injects a selector below to avoid
            # repeatedly building442 long packet histories. Real selector tests
            # remain in the separate50-test challenge module.
            return {'tick':tick}, {}, {}, now
        def command_tick(self,c):
            state.commands.append(c)
            count=13 if state.fault=='command' else 50
            start=len(self.samples)
            self.samples.extend(dict(timestamp_s=(i+1)*.002,requested_command=np.asarray(c,np.float64),
                applied_command=np.asarray(c,np.float32).astype(np.float64),phase=np.uint8(self.phase))
                for i in range(start,start+count))
            if state.fault=='command': raise PhysicalStop('NATIVE_CONTACT_STOP')
    monkeypatch.setattr(mod,'IndependentTrackingSession',Session)
    monkeypatch.setattr(mod,'initialize_genesis',lambda **kw:state.events.append('initialize'))
    monkeypatch.setattr(mod,'shutdown_genesis',lambda:state.events.append('shutdown'))
    monkeypatch.setattr(mod,'configure_gains',lambda *a:dict(effective={'kp':1}))
    monkeypatch.setattr(mod,'read_gains',lambda *a:{'kp':2 if state.fault=='gains' else 1})
    monkeypatch.setattr(mod,'native_friction',lambda *a,**k:dict(solver_friction=.15))
    monkeypatch.setattr(mod,'capture_native_robot_geometry',lambda _:[])
    monkeypatch.setattr(mod,'appearance_environment_identity',lambda _: {})
    def setup(s,sha):
        for name in ('static_objects.json','startup_native_robot_geometry.json','setup_checks.json'):
            (s.output/name).write_text('{}')
        if state.fault=='setup': raise PhysicalStop('SETUP_REJECTED')
    monkeypatch.setattr(mod,'admit_context_setup',setup)
    from lewm.independent_tracking_challenge_development import schedule
    rows=schedule('left')
    def selector(direction,tick,p):
        assert p=={'tick':tick}
        row=rows[tick] if tick<MAX_TICKS else dict(phase=10,role='terminal',requested_command=[0.,0.,0.])
        return deepcopy(row)|dict(tick=tick,decision_ns=1_500_000_000+tick*100_000_000,terminal=tick==MAX_TICKS,
                                  tracker_required=False,native_state_used=False,navigation_qualified=False)
    monkeypatch.setattr(mod,'decision',selector)
    return state


def test_full_fixed_tape_persists_all_commands_and_never_runs_an_observer(store,native_mock):
    from scripts.independent_tracking_collection_development import collect_episode
    r,receipt=collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert r['status']=='TRACKING_TAPE_REQUIRES_RAW_AUDIT' and r['schedule_complete']
    assert r['completed_ticks']==r['command_ticks']==442 and r['rgbd_frames']==r['decisions']==443
    assert r['physics_samples']==22850 and not r['observer_executed'] and not r['native_evaluation_executed']
    assert not r['sensor_reconstruction_verified'] and not r['navigation_qualified']
    assert len(native_mock.commands)==442 and native_mock.events[-2:]==['destroy','shutdown']
    assert not r['secondary_failures'] and 'failure.json' not in receipt['artifact_sha256']
    rows=json.loads((store.directory/'tracking_decisions.json').read_text())
    assert all(not row['observer_computation_included'] for row in rows)
    from scripts.independent_tracking_command_audit_development import audit_commands
    with np.load(store.directory/'physics_trace.npz') as z: raw={k:z[k] for k in z.files}
    tape=json.loads((store.directory/'command_tape.json').read_text())
    audit=audit_commands(raw,tape,rows,r,'left')
    assert audit['completed_intervals']==442 and not audit['navigation_qualified']


@pytest.mark.parametrize('fault', ['settle','setup','command','capture','gains'])
def test_lifecycle_stops_keep_partial_evidence_and_cannot_report_success(store,native_mock,fault):
    from scripts.independent_tracking_collection_development import collect_episode
    native_mock.fault=fault
    r,receipt=collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert not r['navigation_qualified'] and native_mock.events[-2:]==['destroy','shutdown']
    if fault in ('capture','gains'):
        assert r['status']=='TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE'
        assert 'failure.json' in receipt['artifact_sha256']
    else:
        assert r['physical_stop'] and not r['schedule_complete'] and not r['infrastructure_failure']
    if fault=='settle': assert r['physics_samples']==1 and r['rgbd_frames']==0 and not native_mock.commands
    if fault=='setup': assert r['rgbd_frames']==1 and not r['setup_admitted'] and not native_mock.commands
    if fault=='command':
        assert r['command_ticks']==1 and r['completed_ticks']==0 and r['physics_samples']==763
        tape=json.loads((store.directory/'command_tape.json').read_text())
        assert tape[0]['post_sample_index']==762 and not tape[0]['completed']
    if fault=='capture':
        assert len(native_mock.commands)==1 and r['rgbd_frames']==1
        assert 'rgb_0001.png' in receipt['artifact_sha256'] and 'depth_0001.npz' not in receipt['artifact_sha256']
    from scripts.independent_tracking_command_audit_development import audit_commands
    with np.load(store.directory/'physics_trace.npz') as z: raw={k:z[k] for k in z.files}
    rows=json.loads((store.directory/'tracking_decisions.json').read_text())
    tape=json.loads((store.directory/'command_tape.json').read_text())
    if fault in ('capture','gains'):
        with pytest.raises(ValueError): audit_commands(raw,tape,rows,r,'left')
    else: assert audit_commands(raw,tape,rows,r,'left')['command_accounting_verified']


@pytest.mark.parametrize('fault',['request','applied','phase','clock','stop','index','timing','completed'])
def test_independent_command_audit_rejects_mutated_saved_prefix(store,native_mock,fault):
    from scripts.independent_tracking_collection_development import collect_episode
    from scripts.independent_tracking_command_audit_development import audit_commands
    native_mock.fault='command'
    result,_=collect_episode(store,specification(TRIALS[0]),'a'*64)
    with np.load(store.directory/'physics_trace.npz') as z: raw={k:z[k] for k in z.files}
    rows=json.loads((store.directory/'tracking_decisions.json').read_text())
    tape=json.loads((store.directory/'command_tape.json').read_text())
    if fault=='request': raw['requested_command'][750,0]=.1
    elif fault=='applied': raw['applied_command'][750,0]=.1
    elif fault=='phase': raw['phase'][750]=9
    elif fault=='clock': raw['timestamp_s'][750]+=.001
    elif fault=='stop': result['physical_stop']=None
    elif fault=='index': rows[0]['observation_index']=1
    elif fault=='timing': rows[0]['observer_computation_included']=True
    elif fault=='completed': tape[0]['completed']=True;result['completed_ticks']=1
    with pytest.raises((ValueError,AssertionError)): audit_commands(raw,tape,rows,result,'left')


def test_wrong_trial_or_protocol_is_rejected_before_allocation(store,native_mock):
    from scripts.independent_tracking_collection_development import collect_episode
    with pytest.raises(ValueError): collect_episode(store,specification(TRIALS[1]),'a'*64)
    with pytest.raises(ValueError): collect_episode(store,specification(TRIALS[0]),'not-a-hash')
    assert not native_mock.events and not store.bindings


def test_snapshot_failure_is_recorded_and_does_not_prevent_cleanup_or_tape_save(store,native_mock,monkeypatch):
    import scripts.independent_tracking_collection_development as mod
    native_mock.fault='command'
    def fail(s,st):
        st.npz('physics_trace.npz',dict(timestamp_s=np.array([.002])))
        raise RuntimeError('sensor persistence failed')
    monkeypatch.setattr(mod,'persist_snapshot',fail)
    r,receipt=mod.collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert r['status']=='TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE'
    assert r['secondary_failures'][0]['stage']=='persist_snapshot'
    assert 'physics_trace.npz' in receipt['artifact_sha256'] and 'command_tape.json' in receipt['artifact_sha256']
    assert native_mock.events[-2:]==['destroy','shutdown']


def test_post_constructor_mesh_audit_failure_still_destroys_allocated_scene(store,native_mock):
    from scripts.independent_tracking_collection_development import collect_episode
    native_mock.fault='mesh'
    r,receipt=collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert r['status']=='TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE'
    assert native_mock.events[-2:]==['destroy','shutdown'] and not native_mock.commands
    assert 'visual_meshes/ground_visual.ply' in receipt['artifact_sha256']
    assert 'visual_meshes/wall_union_visual.ply' not in receipt['artifact_sha256']


def test_storage_stop_before_dispatch_retains_acquired_frame_and_zero_commands(store,native_mock,monkeypatch):
    from scripts.independent_tracking_collection_development import collect_episode
    from scripts.independent_tracking_artifacts_development import FRAME_BYTES
    original=store.check
    def check(allocation,**kw):
        if allocation==FRAME_BYTES and 'setup_checks.json' in store.bindings:
            raise StorageStop('STORAGE_RESERVE_STOP')
        return original(allocation,**kw)
    monkeypatch.setattr(store,'check',check)
    r,_=collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert r['acquisition_stop']=='STORAGE_RESERVE_STOP' and not r['infrastructure_failure']
    assert r['rgbd_frames']==1 and r['decisions']==0 and not native_mock.commands
    assert native_mock.events[-2:]==['destroy','shutdown']


@pytest.mark.parametrize('already_destroyed',[False,True])
def test_recorder_initialization_failure_destroys_scene_once(monkeypatch,tmp_path,already_destroyed):
    from scripts.independent_tracking_session_development import IndependentTrackingSession
    from scripts.independent_pulse_context_session_development import PulseContextSession
    events=[]
    def fail(self,*a):
        self.ctx=NS(build=NS(scene=NS(destroy=lambda:events.append('destroy'))))
        if already_destroyed:
            self.ctx.build.scene.destroy();self._independent_scene_destroyed=True
        raise RuntimeError('late recorder construction failure')
    monkeypatch.setattr(PulseContextSession,'__init__',fail)
    with pytest.raises(RuntimeError): IndependentTrackingSession(specification(TRIALS[0]),tmp_path)
    assert events==['destroy']
