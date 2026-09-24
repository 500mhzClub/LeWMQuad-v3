"""Synthetic complete populations and boundary audits, without a native scene."""
import gzip
import json
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from scripts import extended_return_budget_maze_pipeline_development as new
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeBaseSession
from lewm.extended_return_budget_mission_development import ExtendedReturnBudgetMeasuredMission
from lewm.tests.test_causal_auxiliary_rgb_observation_development import inputs
from lewm.tests.test_extended_return_budget_memory_development import retime
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from lewm.causal_auxiliary_rgb_observation_development import validate_rgb
import hashlib


def test_complete_stream_is_lossless_exclusive_and_rejects_extra_row(tmp_path):
    with new.writer(tmp_path) as append:
        for tick in range(8014): append(dict(tick=tick, decision={'terminal':None}))
        with pytest.raises(ValueError): append(dict(tick=8014))
    assert [row['tick'] for row in new.read_rows(tmp_path)] == list(range(8014))
    with pytest.raises(ValueError): list(new.original.read_rows(tmp_path))
    with pytest.raises(FileExistsError):
        with new.writer(tmp_path): pass
    with gzip.open(tmp_path/new.original.stream.NAME, 'ab') as f: f.write(b'{"tick":8014}\n')
    with pytest.raises(ValueError): list(new.read_rows(tmp_path))


@pytest.mark.parametrize('frame', [-1, 4014, 8013, 8014])
def test_renderer_session_applies_new_bound_before_primary_and_latches(monkeypatch, frame):
    calls = []
    class PrimaryReached(Exception): pass
    def primary(session):
        calls.append('primary'); raise PrimaryReached()
    monkeypatch.setattr(NovelMazeBaseSession, 'sensor_packets', primary)
    session = object.__new__(new.ExtendedReturnBudgetRendererSession)
    session.samples = range(750+50*frame)
    session.renderer_witnesses = dict(primary=[], paired=[], failures=[])
    if 0 <= frame < 8014:
        with pytest.raises(PrimaryReached): session.sensor_packets()
        assert calls == ['primary']
    else:
        with pytest.raises(ValueError, match='bounded prospective'): session.sensor_packets()
        assert not calls
    assert len(session.renderer_witnesses['failures']) == 1
    before = list(calls)
    with pytest.raises(ValueError, match='terminal'): session.sensor_packets()
    assert calls == before


def test_complete_replay_constructor_and_final_public_auxiliary_packet(tmp_path):
    r = new.original.replay; count = 8014
    manifest = dict(schema='causal_rgb_body_routes_development.v1', camera_calibration_id=r.CAMERA_CALIBRATION,
        sensor_assumption='ideal_simulated_body_origin_50hz_zero_latency', sensor_schemas=r.schema_metadata(),
        history_file='policy_histories.npz', frames=[dict(rgb_file=f'rgb_{i:04d}.png',image_ns=i,decision_ns=i) for i in range(count)])
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    fields = {'image_ns','decision_ns'} | {f'{s.name}_{f}' for s in r.SCHEMAS
        for f in ('values','valid','measured_ns','available_ns')}
    # These arrays test constructor populations only, not packet reconstruction.
    np.savez_compressed(tmp_path/'policy_histories.npz', **{k:np.zeros(count) for k in fields})
    np.savez_compressed(tmp_path/'fast_gyro_histories.npz',
        **{k:np.zeros(count) for k in ('values','valid','measured_ns','available_ns')})
    depth_manifest = dict(schema=r.SCHEMA, calibration=r.calibration_metadata(), frames=[dict(
        depth_file=f'depth_{i:04d}.npz',schema=None,calibration_id=None,identity=None,measured_ns=i,
        available_ns=i,decision_ns=i,rgb_sha256=None,representation=None,hardware_calibrated=False) for i in range(count)])
    (tmp_path/'depth_observations.json').write_text(json.dumps(depth_manifest))
    reader = new.ExtendedReturnBudgetRGBDReplay(tmp_path)
    assert len(reader.frames) == count
    with pytest.raises(ValueError): new.original.ExtendedBudgetRGBDReplay(tmp_path)
    with pytest.raises(ValueError): new.validate_frame_population([None]*8015)
    policy, auxiliary, image, old_ns, native, rgb = inputs()
    now = 1_500_000_000+8013*100_000_000; policy = retime(policy, now-old_ns)
    np.savez_compressed(tmp_path/'auxiliary_depth_8013.npz',native_optical_depth_m=native,
        depth_m=auxiliary['depth_m'],valid=auxiliary['valid'])
    Image.fromarray(rgb).save(tmp_path/'auxiliary_rgb_8013.png')
    acquisition = dict(frame=8013,measured_ns=now,calibration_id=CALIBRATION_ID,
        native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),rgb_sha256=image['rgb_sha256'])
    actual_image, actual_depth = new.rgb_packet(tmp_path,8013,policy,acquisition,now_ns=now)
    validate_rgb(actual_image,actual_depth,policy,now_ns=now)
    assert actual_image['measured_ns'] == actual_depth['measured_ns'] == now
    np.testing.assert_array_equal(actual_image['rgb'],rgb)
    for index in (8014, True):
        with pytest.raises(ValueError): new.rgb_packet(tmp_path,index,policy,acquisition,now_ns=now)


def test_all_renderer_endpoints_are_bound_and_final_acquisition_tampering_is_rejected(monkeypatch):
    witness = new.original.witness
    camera = NS(uid='synthetic', transform=np.eye(4))
    identity = dict(camera_uid='synthetic', framebuffer_matches_camera_depth_target=True,
        raster_error_bound_proven=False, source_implementation_equivalence_proven=False)
    for key, value in dict(renderer_identity_readback=lambda c:deepcopy(identity),
            sampling_readback=lambda c:deepcopy(witness.SAMPLING),
            precision_readback=lambda c:{}).items():
        monkeypatch.setitem(new.capture_witness.__globals__, key, value)
    session = NS(ctx=NS(build=NS(camera=camera), runner=NS(_sim_time_ns=0)), samples=[])
    document = dict(primary=[], paired=[], failures=[]); captures = []
    for frame in range(8014):
        session.samples = range(750+50*frame)
        session.ctx.runner._sim_time_ns = 1_500_000_000+100_000_000*frame
        for phase, key in zip(witness.PHASES, ('primary', 'paired'), strict=True):
            hashes = {k:'a'*64 for k in (witness.PRIMARY_HASHES if key == 'primary' else witness.PAIRED_HASHES)}
            document[key].append(new.capture_witness(session, frame=frame, phase=phase, pixel_hashes=hashes))
        captures.append(dict(frame=frame, measured_ns=session.ctx.runner._sim_time_ns,
            physical_sample_index=len(session.samples)-1,
            primary_world_from_optical=np.diag([1., -1., -1., 1.]).tolist(),
            **document['paired'][-1]['pixel_hashes']))
    audit = new.audit_witnesses(document, captures)
    assert audit['frames'] == 8014 and audit['capture_endpoints'] == 16028
    assert audit['all_witnesses_match_raw_acquisitions'] and not audit['navigation_qualified']
    with pytest.raises(ValueError): new.original.audit_witnesses(document, captures)
    with pytest.raises(ValueError): new.capture_witness(session, frame=8014,
        phase=witness.PHASES[0], pixel_hashes={k:'a'*64 for k in witness.PRIMARY_HASHES})
    captures[-1]['auxiliary_rgb_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='pixel identities'): new.audit_witnesses(document, captures)
    captures[-1]['auxiliary_rgb_sha256'] = 'a'*64
    captures[-1]['physical_sample_index'] -= 1
    with pytest.raises(ValueError, match='physical_sample_index'): new.audit_witnesses(document, captures)
    captures[-1]['physical_sample_index'] += 1
    document['paired'][-1]['camera_transform'][0][3] = .01
    with pytest.raises(ValueError, match='endpoint drift'): new.audit_witnesses(document, captures)


def command_population():
    n=401400;raw=dict(timestamp_s=np.arange(1,n+1)*.002,phase=np.zeros(n,dtype=int))
    for key in ('requested_command','applied_command','post_slew_applied_command'):
        raw[key]=np.zeros((n,3),np.float64)
    rows=[];tape=[]
    for tick in range(8014):
        terminal='MISSION_TICK_BUDGET_EXHAUSTED' if tick>=8003 else None
        rows.append(dict(tick=tick,decision=dict(requested_command=[0.,0.,0.],terminal=terminal)))
        if tick==8013:continue
        phase,role=((3,'terminal_zero_drain') if terminal else
            (1,'causal_history_warmup') if tick<3 else (2,'online_learned_round_trip_command'))
        a=749+50*tick;b=a+50;raw['phase'][a+1:b+1]=phase
        tape.append(dict(tick=tick,requested_command=[0.,0.,0.],phase=phase,role=role,
            pre_sample_index=a,post_sample_index=b,completed=True))
    result=dict(command_ticks=8013,decisions=8014,completed_ticks=8013,terminal_zero_ticks=10,
        physical_stop=None,acquisition_stop=None,schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED')
    return raw,tape,rows,result


def test_full_command_endpoint_and_drain_audit_detects_final_sample_corruption():
    raw,tape,rows,result=command_population()
    new.audit_commands(raw,tape,rows,result)
    with pytest.raises(AssertionError):new.original.audit_commands(raw,tape,rows,result)
    raw['post_slew_applied_command'][-1,0]=.01
    with pytest.raises(AssertionError):new.audit_commands(raw,tape,rows,result)


def test_extended_evaluator_accepts_full_negative_population_and_checks_late_dwell():
    n=401400
    raw=dict(base_pose_world=np.zeros((n,7)),timestamp_s=np.arange(1,n+1)*.002,
        base_twist_world=np.zeros((n,6)),requested_command=np.zeros((n,3)),physics_contact=np.zeros(n,bool))
    raw['base_pose_world'][:,6]=1.
    collection=dict(schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED',terminal_zero_ticks=10,
        physical_stop=None,acquisition_stop=None)
    mission=dict(arrivals=[],terminal=collection['schedule_terminal'])
    outcome=new.evaluate(raw,mission,collection,layout_index=2)
    assert not outcome['native_round_trip_candidate_pass'] and not outcome['verified_round_trip']
    with pytest.raises(ValueError):new.original.original_audit.evaluate(raw,mission,collection,layout_index=2)
    target=new.original.original_audit.public_mission(2)['goal_initial_body_xy_m']
    frame=7000;end=749+50*frame
    raw['base_pose_world'][end-500:end+1,:2]=target
    mission['arrivals']=[dict(phase='OUTBOUND',frame=frame,measured_ns=1_500_000_000+frame*100_000_000,
        target_initial_body_xy_m=target,quiet_intervals=10,native_verified=False)]
    outcome=new.evaluate(raw,mission,collection,layout_index=2)
    assert outcome['arrival_windows'][0]['native_one_second_arrival_and_quiet_pass']
    # The synthetic teleport is not a valid traversal or a round trip.
    assert not outcome['native_round_trip_candidate_pass']
    raw['base_twist_world'][end-250,0]=.051
    outcome=new.evaluate(raw,mission,collection,layout_index=2)
    assert not outcome['arrival_windows'][0]['native_one_second_arrival_and_quiet_pass']
    too_long={k:np.concatenate((v,v[-1:]),axis=0) for k,v in raw.items()}
    too_long['timestamp_s'][-1]=.002*(n+1)
    with pytest.raises(ValueError):new.evaluate(too_long,mission,collection,layout_index=2)


def test_original_collection_loop_runs_complete_extended_population_and_persists(tmp_path):
    calls=[]
    class Session:
        def __init__(self,spec,directory):
            self.samples=[];self.model_manifest=[];self.auxiliary_audit=[];self.guard_rows=[];self.phase=None
            self.ctx=NS(build=NS(robot=object(),collision_floor=NS(links=[NS(idx=0)],geoms=[NS(idx=0)]),
                visual_surfaces=[],scene=NS(destroy=lambda:calls.append('destroy'))),
                runner=NS(_leg_dof_idx=np.array([0])),policy=NS(env_cfg={}))
        def install_contact_identity(self):pass
        def settle_recorded(self):self.samples.extend([None]*750)
        def capture_current(self):
            frame=(len(self.samples)-750)//50
            if len(self.model_manifest)==frame:self.model_manifest.append({});self.auxiliary_audit.append({})
        def sensor_packets(self):
            self.capture_current()
            return None,None,None,None,None,1_500_000_000+100_000_000*(len(self.model_manifest)-1)
        def command_tick(self,request):
            assert request==[0.,0.,0.];self.samples.extend([None]*50)
        def persist(self,directory):calls.append('persist')
        def persist_observations(self,directory):calls.append('persist_observations')
    class Controller:
        def __init__(self,model,geometry,*,public_mission,navigation_ticks,**kwargs):
            assert navigation_ticks==8000
            self.mission=ExtendedReturnBudgetMeasuredMission(public_mission,navigation_ticks=navigation_ticks)
            self.frame=0
        def observe(self,p,d,f,*,now_ns,**kwargs):
            mission=self.mission.advance([1.,0.,.3],frame=self.frame,now_ns=now_ns,previous_requested_command=[0.,0.,0.])
            self.frame+=1
            return dict(mission_receipt=mission,terminal=mission['terminal'],requested_command=[0.,0.,0.])
    collect=new.bind(new.collect,BASE=tmp_path,validate_root=lambda p:None,
        shutil=NS(disk_usage=lambda p:NS(free=100*1024**3)),
        initialize_genesis=lambda **kw:calls.append('initialize'),shutdown_genesis=lambda:calls.append('shutdown'),
        RendererWitnessDualCameraMazeSession=Session,ResidualAnchoredContinuationController=Controller,
        configure_gains=lambda *a:{'effective':{}},read_gains=lambda *a:{},native_friction=lambda *a:{},
        admit_context_setup=lambda *a:None,capture_native_robot_geometry=lambda *a:{},appearance_environment_identity=lambda *a:{})
    result=collect(2,'synthetic',output=tmp_path,model=None,geometry=None,episode_name='synthetic',condition='direct',variant='no_rgb')
    directory=tmp_path/'synthetic'
    assert result['navigation_ticks']==8000 and result['command_ticks']==result['completed_ticks']==8013
    assert result['decisions']==result['rgbd_frames']==result['auxiliary_frames']==8014
    assert result['physics_samples']==401400 and result['terminal_zero_ticks']==10
    assert result['schedule_terminal']=='MISSION_TICK_BUDGET_EXHAUSTED'
    assert result['storage_allowance_bytes']==28*1024**3
    rows=list(new.read_rows(directory))
    assert rows[4003]['decision']['terminal'] is None and rows[8003]['decision']['terminal']==result['schedule_terminal']
    assert rows[-1]['pre_sample_index']==401399
    tape=json.loads((directory/'command_tape.json').read_text())
    assert len(tape)==8013 and all(t['phase']==3 and t['completed'] for t in tape[-10:])
    assert calls==['initialize','persist','persist_observations','destroy','shutdown']
    assert 'auxiliary_rgb_8013.png' in new.artifacts(2,result)


def test_bindings_keep_original_loop_audit_and_evaluator_code():
    for bound,original in ((new.collect,new.original.episode.collect),
            (new.audit,new.original.original_audit.audit),
            (new.evaluate,new.original.original_audit.evaluate),
            (new.audit_commands,new.original.commands.audit_commands)):
        assert bound.__code__ is original.__code__ and bound.__closure__ is original.__closure__
        assert bound.__defaults__ is original.__defaults__ and bound.__kwdefaults__==original.__kwdefaults__
    assert new.collect.__globals__['ResidualAnchoredContinuationController'] is new.ExtendedReturnBudgetChainedController
    assert new.audit.__globals__['ResidualAnchoredContinuationController'] is new.ExtendedReturnBudgetChainedController
    assert new.audit.__globals__['evaluate'] is new.evaluate
    assert new.audit.__globals__['IntentReturnRGBDReplay'] is new.ExtendedReturnBudgetRGBDReplay
    assert new.audit_sensors.__globals__['IntentReturnRGBDReplay'] is new.ExtendedReturnBudgetRGBDReplay
    assert new.definition()['max_physics_samples']==401400
    assert not new.definition()['single_read_auxiliary_acquisition_adopted']
