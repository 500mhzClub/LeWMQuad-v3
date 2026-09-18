"""Synthetic collection/audit evidence, not native navigation qualification."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from scripts import measured_plane_independent_multiarm_pipeline_development as new
from scripts.novel_maze_round_trip_physical_session_development import (
    NovelMazeBaseSession, NovelMazeRoundTripPhysicalInit)
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit


@pytest.mark.parametrize('index', [3014, 3611, 4013, 4014])
def test_independent_session_reaches_extended_guard_before_primary_capture(monkeypatch, index):
    calls = []
    class ReachedPrimary(Exception): pass
    def primary(self):
        calls.append('primary')
        raise ReachedPrimary()
    monkeypatch.setattr(NovelMazeBaseSession, 'sensor_packets', primary)
    session = object.__new__(new.MeasuredPlaneIndependentSession)
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
    mro = new.MeasuredPlaneIndependentSession.__mro__
    assert mro.index(IndependentRoundTripPhysicalInit) < mro.index(NovelMazeRoundTripPhysicalInit)


def test_original_loops_and_private_independent_scene_dependencies_are_retained():
    for bound, original in ((new.collect, new.episode.collect), (new.audit, new.raw_audit.audit)):
        assert bound.__code__ is original.__code__
        assert bound.__closure__ is original.__closure__
        assert bound.__defaults__ is original.__defaults__
        assert bound.__kwdefaults__ == original.__kwdefaults__
        for key in ('specification', 'public_mission'):
            assert bound.__globals__[key] is original.__globals__[key]
    for key in ('evaluate', 'audit_setup', 'audit_stops', 'audit_auxiliary',
            'audit_rasters_and_footprints', 'public_acquisition'):
        assert new.audit.__globals__[key] is new.raw_audit.audit.__globals__[key]
    assert new.collect.__globals__['create'] is new.audit.__globals__['create'] is new.create
    assert new.episode.NAVIGATION_TICKS == new.raw_audit.NAVIGATION_TICKS == 3000
    assert new.raw_audit.audit_reactive_commands.__globals__['NAVIGATION_TICKS'] == 3000
    assert new.audit_reactive_commands.__globals__['NAVIGATION_TICKS'] == 4000
    assert new.collect.__globals__['RESERVE_BYTES'] == 40*1024**3
    assert new.collect.__globals__['PERSISTENCE_HEADROOM_BYTES'] == 1024**3


def synthetic_pipeline(tmp_path, case):
    """Real persistence, mission budget, raw command and complete decision loops.

    Physical acquisition, sensor reconstruction and model calls are synthetic.
    Their production implementations remain bound in new.audit, tested above;
    this fixture does not claim to validate renderer or learned inference.
    """
    events = []; sessions = []; controllers = []
    class Controller:
        def __init__(self):
            self.mission = ObservedRoundTripMission(new.episode.public_mission(case.layout_index),
                navigation_ticks=4000)
        def observe(self, policy, *args, now_ns, **kwargs):
            tick = policy['tick']
            mission = self.mission.advance([0., 0.], frame=tick, now_ns=now_ns,
                previous_requested_command=[0., 0., 0.])
            return dict(requested_command=[0., 0., 0.], terminal=mission['terminal'],
                mission_receipt=mission, new_selection=None, evidence=None)
    def create(selected, geometry, *, correction_admission):
        assert selected == case
        controller = Controller(); controllers.append(controller)
        return controller, None
    class Session:
        def __init__(self, spec, directory):
            self.spec = spec; self.directory = directory; self.phase = 0
            self.samples = []; self.model_manifest = []; self.auxiliary_audit = []; self.guard_rows = []
            self.phases = []; sessions.append(self)
            self.ctx = NS(build=NS(robot=object(),
                collision_floor=NS(links=[NS(idx=1)], geoms=[NS(idx=1)]),
                visual_surfaces=[], scene=NS(destroy=lambda:events.append('destroy'))),
                runner=NS(_leg_dof_idx=np.arange(12)), policy=NS(env_cfg={}))
        def install_contact_identity(self): events.append('contact_identity')
        def settle_recorded(self):
            self.samples = [None]*750; self.phases = [0]*750
        def capture_current(self):
            tick = (len(self.samples)-750)//50
            if len(self.model_manifest) == tick:
                self.model_manifest.append(dict(physical_sample_index=len(self.samples)-1))
                self.auxiliary_audit.append({})
        def sensor_packets(self):
            self.capture_current(); tick = len(self.model_manifest)-1
            return dict(tick=tick), {}, {}, {}, {}, 1_500_000_000+tick*100_000_000
        def command_tick(self, request):
            assert request == [0., 0., 0.]
            self.samples.extend([None]*50); self.phases.extend([self.phase]*50)
        def persist(self, directory): events.append('persist')
        def persist_observations(self, directory):
            (directory/'auxiliary_camera_audit.json').write_text(json.dumps(self.auxiliary_audit))
            events.append('persist_observations')
        def raw(self):
            n = len(self.samples); pose = np.zeros((n, 7))
            pose[:, :2] = self.spec['geometry']['spawn_se2_world'][:2]; pose[:, 6] = 1.
            return dict(timestamp_s=np.arange(1, n+1)*.002, base_pose_world=pose,
                requested_command=np.zeros((n, 3), np.float64),
                applied_command=np.zeros((n, 3), np.float64),
                post_slew_applied_command=np.zeros((n, 3), np.float64), phase=np.asarray(self.phases))
    def setup(session, definition): (session.directory/'setup_checks.json').write_text('{}')
    collect = new.extended._bind(new.collect, BASE=tmp_path, validate_root=lambda root:root,
        create=create, verify_model=lambda *args:None,
        IndependentRoundTripSession=Session,
        shutil=NS(disk_usage=lambda root:NS(free=600*1024**3)),
        initialize_genesis=lambda **kwargs:events.append('initialize'),
        shutdown_genesis=lambda:events.append('shutdown'),
        configure_gains=lambda *args:dict(effective={'fixed':True}), read_gains=lambda *args:{'fixed':True},
        capture_native_robot_geometry=lambda *args:{}, appearance_environment_identity=lambda *args:{},
        native_friction=lambda build, mu:dict(solver_friction=mu,
            solver_ratio=np.ones((1, 28)).tolist(), physics_steps=len(sessions[-1].samples)),
        admit_context_setup=setup)
    def sensors(directory, spec, result):
        assert spec == new.episode.specification(case.layout_index)
        return sessions[-1].raw(), {}, {}, {}, sessions[-1].model_manifest, None, {}, {'depth_checks':[]}
    def evaluate(raw, mission, result, *, layout_index):
        assert layout_index == case.layout_index and mission['arrivals'] == []
        return dict(native_round_trip_candidate_pass=False)
    audit = new.extended._bind(new.audit, validate_root=lambda root:root,
        artifact_path=lambda root, name:root/name, create=create, verify_model=lambda *args:None,
        audit_sensors=sensors,
        audit_auxiliary=lambda *args:[dict(auxiliary_visibility_pass=True)]*4014,
        IntentReturnRGBDReplay=lambda directory:NS(packet=lambda tick:
            (dict(tick=tick), {}, {}, 1_500_000_000+tick*100_000_000)),
        public_acquisition=lambda row:row, packet=lambda *args, **kwargs:({}, {}),
        audit_setup=lambda *args:{}, audit_stops=lambda *args:None,
        audit_rasters_and_footprints=lambda *args:[dict(stable_interior_metric_pass=True,
            near_occlusion_failure=False, original_strict_score={'passes_sampled_physical_visibility':True})]*4014,
        renderer_audit=lambda directory:{'synthetic_only':True}, evaluate=evaluate)
    return collect, audit, events, sessions, controllers


@pytest.mark.parametrize('arm_name', [arm.name for arm in new.study.ARMS])
def test_all_four_arms_complete_full_budget_and_raw_replay(tmp_path, arm_name):
    case = next(case for case in new.study.CASES if case.arm_name == arm_name and case.layout_index == 7)
    collect, audit, events, sessions, controllers = synthetic_pipeline(tmp_path, case)
    result = collect(case, 'synthetic', output=tmp_path, geometry=None)
    report = audit(case, result, 'synthetic', input_root=tmp_path, robot_geometry=None)
    assert result['decisions'] == result['rgbd_frames'] == result['auxiliary_frames'] == 4014
    assert result['completed_ticks'] == result['command_ticks'] == 4013
    assert result['physics_samples'] == 201400 and result['terminal_zero_ticks'] == 10
    assert result['schedule_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert result['storage_allowance_bytes'] == 14*1024**3
    assert report['raw_command_audit_pass'] and report['raw_controller_command_replay_pass']
    assert not report['verified_round_trip'] and not report['real_time_qualified']
    assert report['assignment'] == result['assignment'] == new.study.treatment(case)['assignment']
    assert len(controllers) == 2 and controllers[0] is not controllers[1]
    directory = tmp_path/case.name
    rows = list(new.extended.read_rows(directory))
    assert rows[3003]['decision']['terminal'] is None
    assert rows[4003]['decision']['terminal'] == rows[-1]['decision']['terminal'] == result['schedule_terminal']
    tape = json.loads((directory/'command_tape.json').read_text())
    assert tape[3]['role'] == tape[4002]['role'] == new.study.command_role(case)
    assert all(row['role'] == 'terminal_zero_drain' for row in tape[-10:])
    assert events == ['initialize', 'contact_identity', 'persist', 'persist_observations', 'destroy', 'shutdown']
    names = new.artifacts(case, result)
    assert len(names) == len(set(names))
    assert 'auxiliary_rgb_4013.png' in names and 'auxiliary_depth_4013.npz' in names
    assert 'auxiliary_rgb_4014.png' not in names and new.episode.DECISIONS in names
    wrong = result | dict(navigation_ticks=3000)
    with pytest.raises(ValueError): new.artifacts(case, wrong)


@pytest.mark.parametrize('fault', ['decision', 'command_role', 'native_command', 'timing'])
def test_late_history_corruption_rejected(tmp_path, fault):
    case = new.study.CASES[0]
    collect, audit, _, sessions, _ = synthetic_pipeline(tmp_path, case)
    result = collect(case, 'synthetic', output=tmp_path, geometry=None)
    directory = tmp_path/case.name
    if fault == 'decision':
        rows = list(new.extended.read_rows(directory))
        rows[4002]['decision']['requested_command'][0] = .1
        # Replace only this test's synthetic temporary decision stream.
        (directory/new.episode.DECISIONS).unlink()
        with new.extended.writer(directory) as append:
            for row in rows: append(row)
    elif fault == 'command_role':
        path = directory/'command_tape.json'; rows = json.loads(path.read_text())
        rows[4002]['role'] = 'online_reactive_round_trip_command'
        path.write_text(json.dumps(rows))
    elif fault == 'native_command':
        original = sessions[-1].raw
        def corrupted():
            raw = original(); raw['post_slew_applied_command'][-1, 0] = .01
            return raw
        sessions[-1].raw = corrupted
    else:
        path = directory/'decision_stream_timing.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['iteration_with_receipt_wall_ms'] = -1
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(AssertionError): audit(case, result, 'synthetic', input_root=tmp_path, robot_geometry=None)


@pytest.mark.parametrize('field', ['status', 'assignment', 'layout_index', 'navigation_ticks',
    'measured_plane_constrained_estimator'])
def test_wrong_treatment_rejected_before_raw_access(field):
    case = new.study.CASES[0]
    result = dict(status=new.study.COLLECTION_STATUS, navigation_ticks=4000, **new.study.treatment(case))
    result[field] = 'substitute'
    def forbidden(*args, **kwargs): pytest.fail('bad treatment reached raw access')
    audit = new.extended._bind(new.audit, validate_root=forbidden)
    with pytest.raises(ValueError): audit(case, result, 'synthetic', input_root=None, robot_geometry=None)
