"""Synthetic boundary evidence; no physical or learned navigation outcome."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import extended_budget_anchored_maze_development as new
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeBaseSession
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from lewm.causal_sensor_state import SensorContractError


def test_full_decision_population_is_lossless_exclusive_and_bounded(tmp_path):
    with new.writer(tmp_path) as append:
        for tick in range(4014): append(dict(tick=tick, decision={'terminal': None}))
        with pytest.raises(ValueError): append(dict(tick=4014))
    assert [r['tick'] for r in new.read_rows(tmp_path)] == list(range(4014))
    with pytest.raises(ValueError): list(new.stream.read_rows(tmp_path))
    with pytest.raises(FileExistsError):
        with new.writer(tmp_path): pass
    with gzip.open(tmp_path/new.stream.NAME, 'ab') as target:
        target.write(b'{"tick":4014}\n')
    with pytest.raises(ValueError): list(new.read_rows(tmp_path))


@pytest.mark.parametrize('frames', [0, 4015, None, ()])
def test_replay_population_remains_bounded(frames):
    value = [None]*frames if type(frames) is int else frames
    with pytest.raises(SensorContractError): new.validate_frame_population(value)


def test_real_replay_constructor_accepts_complete_extended_manifest(tmp_path):
    r = new.replay; count = 4014
    manifest = dict(schema='causal_rgb_body_routes_development.v1',
        camera_calibration_id=r.CAMERA_CALIBRATION,
        sensor_assumption='ideal_simulated_body_origin_50hz_zero_latency',
        sensor_schemas=r.schema_metadata(), history_file='policy_histories.npz',
        frames=[dict(rgb_file=f'rgb_{i:04d}.png', image_ns=i, decision_ns=i) for i in range(count)])
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    fields = {'image_ns', 'decision_ns'} | {f'{s.name}_{f}' for s in r.SCHEMAS
        for f in ('values', 'valid', 'measured_ns', 'available_ns')}
    # Constructor-only fixture. These arrays are not valid sensor packets and
    # no packet() call or real sensor reconstruction claim is made.
    np.savez_compressed(tmp_path/'policy_histories.npz', **{k:np.zeros(count) for k in fields})
    np.savez_compressed(tmp_path/'fast_gyro_histories.npz',
        **{k:np.zeros(count) for k in ('values', 'valid', 'measured_ns', 'available_ns')})
    depth = dict(schema=r.SCHEMA, calibration=r.calibration_metadata(), frames=[dict(
        depth_file=f'depth_{i:04d}.npz', schema=None, calibration_id=None, identity=None,
        measured_ns=i, available_ns=i, decision_ns=i, rgb_sha256=None,
        representation=None, hardware_calibrated=False) for i in range(count)])
    (tmp_path/'depth_observations.json').write_text(json.dumps(depth))
    reader = new.ExtendedBudgetRGBDReplay(tmp_path)
    assert len(reader.frames) == 4014
    assert type(reader).packet is r.IntentReturnRGBDReplay.packet
    with pytest.raises(SensorContractError): r.IntentReturnRGBDReplay(tmp_path)
    manifest['frames'][-1]['rgb_file'] = '../rgb_4013.png'
    (tmp_path/'policy_observations.json').write_text(json.dumps(manifest))
    with pytest.raises(SensorContractError): new.ExtendedBudgetRGBDReplay(tmp_path)


@pytest.mark.parametrize('index', [3014, 3611, 4013])
def test_auxiliary_readers_reach_unchanged_identity_checks_beyond_old_cap(tmp_path, index):
    for old, extended in [(new.auxiliary_depth.packet, new.depth_packet),
                          (new.auxiliary_rgb.packet, new.rgb_packet)]:
        with pytest.raises(ValueError, match='bounded actual'): old(tmp_path, index, {}, {}, now_ns=0)
        with pytest.raises(ValueError, match='exact public'): extended(tmp_path, index, {}, {}, now_ns=0)
        with pytest.raises(ValueError, match='bounded actual'): extended(tmp_path, 4014, {}, {}, now_ns=0)


@pytest.mark.parametrize('index', [3014, 3611, 4013, 4014])
def test_full_session_mro_enters_extended_paired_guard_before_primary_capture(monkeypatch, index):
    calls = []
    class ReachedPrimary(Exception): pass
    def primary(self):
        calls.append('primary')
        raise ReachedPrimary()
    monkeypatch.setattr(NovelMazeBaseSession, 'sensor_packets', primary)
    session = object.__new__(new.ExtendedBudgetRendererSession)
    session.samples = range(750+50*index)
    session.renderer_witnesses = dict(primary=[], paired=[], failures=[])
    if index < 4014:
        with pytest.raises(ReachedPrimary): session.sensor_packets()
        assert calls == ['primary']
    else:
        with pytest.raises(ValueError, match='bounded prospective'): session.sensor_packets()
        assert calls == []
    assert len(session.renderer_witnesses['failures']) == 1
    with pytest.raises(ValueError, match='terminal'): session.sensor_packets()


def test_extended_renderer_captures_and_audits_last_frame_and_rejects_drift(monkeypatch):
    camera = NS(uid='synthetic', transform=np.eye(4))
    identity = dict(camera_uid='synthetic', framebuffer_matches_camera_depth_target=True,
        raster_error_bound_proven=False, source_implementation_equivalence_proven=False)
    for key, value in dict(renderer_identity_readback=lambda c:deepcopy(identity),
            sampling_readback=lambda c:deepcopy(new.witness.SAMPLING),
            precision_readback=lambda c:{}).items():
        monkeypatch.setitem(new.capture_witness.__globals__, key, value)
    session = NS(ctx=NS(build=NS(camera=camera), runner=NS(_sim_time_ns=0)), samples=[])
    doc = dict(primary=[], paired=[], failures=[]); captures = []
    for frame in range(4014):
        session.samples = range(750+50*frame)
        session.ctx.runner._sim_time_ns = 1_500_000_000+100_000_000*frame
        for phase, key in zip(new.witness.PHASES, ('primary', 'paired'), strict=True):
            hashes = {k:'a'*64 for k in (new.witness.PRIMARY_HASHES if key == 'primary' else new.witness.PAIRED_HASHES)}
            doc[key].append(new.capture_witness(session, frame=frame, phase=phase, pixel_hashes=hashes))
        captures.append(dict(frame=frame, measured_ns=session.ctx.runner._sim_time_ns,
            physical_sample_index=len(session.samples)-1,
            primary_world_from_optical=np.diag([1., -1., -1., 1.]).tolist(),
            **doc['paired'][-1]['pixel_hashes']))
    assert new.audit_witnesses(doc, captures)['frames'] == 4014
    with pytest.raises(ValueError): new.witness.audit_witnesses(doc, captures)
    with pytest.raises(ValueError): new.capture_witness(session, frame=4014,
        phase=new.witness.PHASES[0], pixel_hashes={k:'a'*64 for k in new.witness.PRIMARY_HASHES})
    captures[-1]['auxiliary_rgb_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='pixel identities'): new.audit_witnesses(doc, captures)


def test_full_extended_command_audit_preserves_drain_and_detects_last_sample_tamper():
    n = 201400; cutoff = 4003
    raw = dict(timestamp_s=np.arange(1, n+1)*.002, phase=np.zeros(n, dtype=int))
    for key in ('requested_command', 'applied_command', 'post_slew_applied_command'):
        raw[key] = np.zeros((n, 3), dtype=np.float64)
    tape = []; rows = []
    for tick in range(4014):
        terminal = 'MISSION_TICK_BUDGET_EXHAUSTED' if tick >= cutoff else None
        rows.append(dict(tick=tick, decision=dict(requested_command=[0., 0., 0.], terminal=terminal)))
        if tick == 4013: continue
        phase, role = ((3, 'terminal_zero_drain') if terminal else
            (1, 'causal_history_warmup') if tick < 3 else (2, 'online_learned_round_trip_command'))
        a = 749+50*tick; b = a+50
        raw['phase'][a+1:b+1] = phase
        tape.append(dict(tick=tick, requested_command=[0., 0., 0.], phase=phase, role=role,
            pre_sample_index=a, post_sample_index=b, completed=True))
    result = dict(command_ticks=4013, decisions=4014, completed_ticks=4013,
        terminal_zero_ticks=10, physical_stop=None, acquisition_stop=None,
        schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED')
    new.audit_commands(raw, tape, rows, result)
    with pytest.raises(AssertionError): new.commands.audit_commands(raw, tape, rows, result)
    raw['post_slew_applied_command'][-1, 0] = .01
    with pytest.raises(AssertionError): new.audit_commands(raw, tape, rows, result)


def test_mission_keeps_return_state_and_global_deadline_after_old_cutoff():
    mission = ObservedRoundTripMission(dict(goal_initial_body_xy_m=[3.9, 2.6],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True), navigation_ticks=4000)
    for tick in range(4004):
        position = [3.9, 2.6] if tick >= 2925 else [0., 0.]
        receipt = mission.advance(position, frame=tick, now_ns=1_500_000_000+tick*100_000_000,
            previous_requested_command=[0., 0., 0.])
        if tick == 2935:
            assert receipt['phase_transition'] == 'OUTBOUND_TO_RETURN'
        if tick in (3003, 4002):
            assert receipt['phase'] == 'RETURN' and receipt['terminal'] is None
            assert not receipt['observed_map_reset_required']
    assert receipt['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert len(receipt['arrivals']) == 1 and not receipt['verified_round_trip']


def test_original_collector_loop_completes_extended_budget_and_persists_drain(tmp_path):
    """Original collector code; synthetic session/controller, no native scene."""
    calls = []; instances = []
    class Session:
        def __init__(self, spec, directory):
            instances.append(self)
            self.samples = []; self.model_manifest = []; self.auxiliary_audit = []
            self.guard_rows = []; self.phase = None
            self.ctx = NS(build=NS(robot=object(), collision_floor=NS(links=[NS(idx=0)], geoms=[NS(idx=0)]),
                visual_surfaces=[], scene=NS(destroy=lambda:calls.append('destroy'))),
                runner=NS(_leg_dof_idx=np.array([0])), policy=NS(env_cfg={}))
        def install_contact_identity(self): pass
        def settle_recorded(self): self.samples.extend([None]*750)
        def capture_current(self):
            tick = (len(self.samples)-750)//50
            if len(self.model_manifest) == tick:
                self.model_manifest.append({}); self.auxiliary_audit.append({})
        def sensor_packets(self):
            self.capture_current()
            return None, None, None, None, None, 1_500_000_000+100_000_000*(len(self.model_manifest)-1)
        def command_tick(self, request):
            assert request == [0., 0., 0.]
            self.samples.extend([None]*50)
        def persist(self, directory): calls.append('persist')
        def persist_observations(self, directory): calls.append('persist_observations')
    class Controller:
        def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
            assert navigation_ticks == 4000
            self.mission = ObservedRoundTripMission(public_mission, navigation_ticks=navigation_ticks)
            self.tick = 0
        def observe(self, p, d, f, *, now_ns, **kwargs):
            mission = self.mission.advance([0., 0.], frame=self.tick, now_ns=now_ns,
                previous_requested_command=[0., 0., 0.])
            self.tick += 1
            return dict(mission_receipt=mission, terminal=mission['terminal'], requested_command=[0., 0., 0.])
    collect = new._bind(new.collect, BASE=tmp_path, validate_root=lambda p:None,
        shutil=NS(disk_usage=lambda p:NS(free=100*1024**3)),
        initialize_genesis=lambda **kw:calls.append('initialize'), shutdown_genesis=lambda:calls.append('shutdown'),
        RendererWitnessDualCameraMazeSession=Session, ResidualAnchoredContinuationController=Controller,
        configure_gains=lambda *a:{'effective':{}}, read_gains=lambda *a:{}, native_friction=lambda *a:{},
        admit_context_setup=lambda *a:None, capture_native_robot_geometry=lambda *a:{},
        appearance_environment_identity=lambda *a:{})
    result = collect(2, 'synthetic', output=tmp_path, model=None, geometry=None,
        episode_name='synthetic', condition='no_rgb', variant='direct')
    directory = tmp_path/'synthetic'
    assert result['navigation_ticks'] == 4000 and result['command_ticks'] == result['completed_ticks'] == 4013
    assert result['rgbd_frames'] == result['decisions'] == result['auxiliary_frames'] == 4014
    assert result['physics_samples'] == 201400 and result['terminal_zero_ticks'] == 10
    assert result['schedule_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert result['storage_allowance_bytes'] == 14*1024**3
    assert result['mission_receipt']['verified_round_trip'] is False
    rows = list(new.read_rows(directory))
    assert rows[3003]['decision']['terminal'] is None
    assert rows[4003]['decision']['terminal'] == rows[-1]['decision']['terminal'] == result['schedule_terminal']
    assert rows[-1]['pre_sample_index'] == 201399
    tape = json.loads((directory/'command_tape.json').read_text())
    assert len(tape) == 4013 and all(t['phase'] == 3 and t['completed'] for t in tape[-10:])
    assert calls == ['initialize', 'persist', 'persist_observations', 'destroy', 'shutdown']
    assert json.loads((directory/'result.json').read_text()) == result


def test_code_closures_and_all_other_dependencies_are_identical():
    pairs = [
        (new.collect, new.episode.collect, {'NAVIGATION_TICKS', 'MAX_OBSERVATIONS',
            'COLLECTION_ALLOWANCE_BYTES', 'writer', 'RendererWitnessDualCameraMazeSession'}),
        (new.audit, new.original_audit.audit, {'NAVIGATION_TICKS', 'MAX_COMMAND_TICKS',
            'IntentReturnRGBDReplay', 'audit_sensors', 'audit_commands', 'read_rows', 'packet', 'renderer_audit'}),
        (new.audit_sensors, new.sensors.audit_sensors, {'IntentReturnRGBDReplay'}),
        (new.ExtendedBudgetPairedSession.sensor_packets, new.NovelMazeRoundTripSession.sensor_packets,
            {'MAX_OBSERVATIONS', 'packet'}),
        (new.ExtendedBudgetDualSession.sensor_packets, new.DualCameraNovelMazeSession.sensor_packets, {'packet'}),
        (new.ExtendedBudgetRendererSession.sensor_packets, new.RendererWitnessDualCameraMazeSession.sensor_packets,
            {'capture_witness'}),
        (new.ExtendedBudgetRendererSession.capture_fixed_rgb, new.RendererWitnessDualCameraMazeSession.capture_fixed_rgb,
            {'capture_witness'}),
    ]
    for bound, original, changes in pairs:
        assert bound.__code__ is original.__code__ and bound.__closure__ is original.__closure__
        assert bound.__defaults__ is original.__defaults__ and bound.__kwdefaults__ == original.__kwdefaults__
        assert bound.__globals__.keys() == original.__globals__.keys()
        assert {k for k in original.__globals__ if bound.__globals__[k] is not original.__globals__[k]} == changes
    assert new.episode.NAVIGATION_TICKS == new.original_audit.NAVIGATION_TICKS == 3000
    assert new.stream.MAX_OBSERVATIONS == new.witness.MAX_OBSERVATIONS == 3014
    assert new.replay.MAX_FRAMES == 3611
    assert new.collect.__globals__['RESERVE_BYTES'] == 40*1024**3
    assert new.collect.__globals__['PERSISTENCE_HEADROOM_BYTES'] == 1024**3
    assert new.collect.__globals__['ResidualAnchoredContinuationController'] is new.audit.__globals__['ResidualAnchoredContinuationController']
    assert new.audit.__globals__['evaluate'] is new.original_audit.evaluate
