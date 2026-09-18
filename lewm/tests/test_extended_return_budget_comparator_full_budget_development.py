"""Full synthetic 8,000-step collection, complete decision replay and command audit.

Physics, camera reconstruction and model inference are synthetic. The original
collection, persistence, raw decision and independent command-audit loops run.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scripts import extended_return_budget_comparator_pipeline_development as pipeline
from scripts import extended_return_budget_resource_audit_development as resource_audit
from lewm.extended_return_budget_mission_development import ExtendedReturnBudgetMeasuredMission


def setup(tmp_path, monkeypatch, mode):
    events = []; sessions = []; controllers = []; clock = [0.]
    monkeypatch.setattr(pipeline.resources, 'validate_root', lambda root: root)
    def snapshot(root):
        clock[0] += .001
        return dict(monotonic_s=clock[0], rss_bytes=1024**3,
            memory_available_bytes=70*1024**3, artifact_free_bytes=100*1024**3)
    monkeypatch.setattr(pipeline.resources, 'snapshot', snapshot)
    class Controller:
        def __init__(self, *args, public_mission, navigation_ticks, **kwargs):
            assert navigation_ticks == 8000
            self.mission = ExtendedReturnBudgetMeasuredMission(public_mission, navigation_ticks=navigation_ticks)
            self.frame = 0; controllers.append(self)
        def observe(self, *args, now_ns, **kwargs):
            mission = self.mission.advance([1., 0., .3], frame=self.frame, now_ns=now_ns,
                previous_requested_command=[0., 0., 0.])
            self.frame += 1
            return dict(requested_command=[0., 0., 0.], terminal=mission['terminal'],
                mission_receipt=mission, new_selection=None, evidence=None)
    monkeypatch.setitem(pipeline.CONTROLLERS, mode, Controller)
    class Session:
        def __init__(self, spec, directory):
            self.spec = spec; self.samples = []; self.phases = []; self.phase = 0
            self.model_manifest = []; self.auxiliary_audit = []; self.guard_rows = []
            self.ctx = NS(build=NS(robot=object(), collision_floor=NS(links=[NS(idx=0)], geoms=[NS(idx=0)]),
                visual_surfaces=[], scene=NS(destroy=lambda: events.append('destroy'))),
                runner=NS(_leg_dof_idx=np.arange(12)), policy=NS(env_cfg={}))
            sessions.append(self)
        def install_contact_identity(self): events.append('contact_identity')
        def settle_recorded(self): self.samples = [None]*750; self.phases = [0]*750
        def capture_current(self):
            frame = (len(self.samples)-750)//50
            if len(self.model_manifest) == frame:
                self.model_manifest.append({'physical_sample_index': len(self.samples)-1})
                self.auxiliary_audit.append({})
        def sensor_packets(self):
            self.capture_current(); frame = len(self.model_manifest)-1
            return None, None, None, None, None, 1_500_000_000+frame*100_000_000
        def command_tick(self, request):
            assert request == [0., 0., 0.]
            self.samples.extend([None]*50); self.phases.extend([self.phase]*50)
        def persist(self, directory): events.append('persist')
        def persist_observations(self, directory):
            (directory/'auxiliary_camera_audit.json').write_text(json.dumps(self.auxiliary_audit))
            events.append('persist_observations')
        def raw(self):
            n = len(self.samples); pose = np.zeros((n, 7)); pose[:, 6] = 1.
            pose[:, :2] = self.spec['geometry']['spawn_se2_world'][:2]
            return dict(timestamp_s=np.arange(1, n+1)*.002, base_pose_world=pose,
                requested_command=np.zeros((n, 3)), applied_command=np.zeros((n, 3)),
                post_slew_applied_command=np.zeros((n, 3)), phase=np.asarray(self.phases))
    monkeypatch.setattr(pipeline.guarded.pipeline, 'ExtendedReturnBudgetRendererSession', Session)
    collect, audit = pipeline.functions(mode)
    def sensors(directory, spec, result):
        assert spec == sessions[0].spec
        return sessions[0].raw(), {}, {}, {}, sessions[0].model_manifest, None, {}, {'depth_checks': []}
    monkeypatch.setattr(pipeline.extended, 'audit_sensors', sensors)
    collect = pipeline.extended.bind(collect, BASE=tmp_path, validate_root=lambda root: root,
        shutil=NS(disk_usage=lambda root: NS(free=100*1024**3)),
        initialize_genesis=lambda **kwargs: events.append('initialize'),
        shutdown_genesis=lambda: events.append('shutdown'),
        configure_gains=lambda *args: {'effective': {}}, read_gains=lambda *args: {},
        native_friction=lambda build, mu: dict(solver_friction=mu,
            solver_ratio=np.ones((1, 28)).tolist(), physics_steps=len(sessions[0].samples)),
        admit_context_setup=lambda *args: None, capture_native_robot_geometry=lambda *args: {},
        appearance_environment_identity=lambda *args: {})
    audit = pipeline.extended.bind(audit,
        audit_auxiliary=lambda *args: [{'auxiliary_visibility_pass': True}]*8014,
        IntentReturnRGBDReplay=lambda directory: NS(packet=lambda tick:
            (None, None, None, 1_500_000_000+tick*100_000_000)),
        packet=lambda *args, **kwargs: (None, None), public_acquisition=lambda row: row,
        audit_setup=lambda *args: {}, audit_stops=lambda *args: None,
        audit_rasters_and_footprints=lambda *args: [dict(stable_interior_metric_pass=True,
            near_occlusion_failure=False, original_strict_score={'passes_sampled_physical_visibility': True})]*8014,
        renderer_audit=lambda directory: {'synthetic': True},
        evaluate=lambda *args, **kwargs: {'native_round_trip_candidate_pass': False})
    monkeypatch.setattr(pipeline, 'functions', lambda selected: (collect, audit))
    kwargs = dict(episode_name='synthetic')
    if mode != 'reactive': kwargs.update(model=torch.nn.Linear(1, 1).eval(), condition='direct', variant='no_rgb')
    return kwargs, events, sessions, controllers


@pytest.mark.parametrize('mode', pipeline.MODES)
def test_complete_population_collection_replay_and_resource_cycles(tmp_path, monkeypatch, mode):
    kwargs, events, _, controllers = setup(tmp_path, monkeypatch, mode)
    result = pipeline.collect(2, 'synthetic', mode=mode, output=tmp_path, geometry=None, **kwargs)
    audit = pipeline.audit(2, result, 'synthetic', mode=mode, input_root=tmp_path, robot_geometry=None, **kwargs)
    assert result['navigation_ticks'] == 8000 and result['decisions'] == 8014
    assert result['rgbd_frames'] == result['auxiliary_frames'] == 8014
    assert result['command_ticks'] == result['completed_ticks'] == 8013
    assert result['physics_samples'] == 401400 and result['terminal_zero_ticks'] == 10
    assert result['schedule_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert audit['raw_command_audit_pass'] and not audit['verified_round_trip'] and not audit['real_time_qualified']
    assert audit['raw_controller_command_replay_pass'] if mode in ('nominal', 'reactive') else audit['raw_model_command_replay_pass']
    if mode == 'nominal': assert not audit['raw_model_command_replay_pass'] and audit['actual_learned_model_forward_calls'] == 0
    assert len(controllers) == 2 and controllers[0] is not controllers[1]
    assert [controller.frame for controller in controllers] == [8014, 8014]
    directory = tmp_path/'synthetic'
    assert json.loads((directory/'result.json').read_text()) == result
    rows = list(pipeline.extended.read_rows(directory))
    assert rows[4003]['decision']['terminal'] is None
    assert rows[8003]['decision']['terminal'] == result['schedule_terminal']
    assert rows[-1]['pre_sample_index'] == 401399
    tape = json.loads((directory/'command_tape.json').read_text())
    role = 'online_reactive_round_trip_command' if mode == 'reactive' else 'online_learned_round_trip_command'
    assert tape[3]['role'] == tape[8002]['role'] == role
    assert all(row['role'] == 'terminal_zero_drain' for row in tape[-10:])
    assert events == ['initialize', 'contact_identity', 'persist', 'persist_observations', 'destroy', 'shutdown']
    receipt = resource_audit.check(tmp_path, 'synthetic', result)
    assert receipt['collection']['samples'] == 32058 and receipt['audit']['samples'] == 16032
    names = pipeline.artifacts(2, result, mode=mode)
    assert len(names) == len(set(names)) and 'auxiliary_rgb_8013.png' in names
    assert 'auxiliary_rgb_8014.png' not in names


@pytest.mark.parametrize('fault', ['late_decision', 'late_role', 'final_command'])
def test_late_reactive_corruption_fails_complete_original_raw_audit(tmp_path, monkeypatch, fault):
    kwargs, _, sessions, _ = setup(tmp_path, monkeypatch, 'reactive')
    result = pipeline.collect(2, 'synthetic', mode='reactive', output=tmp_path, geometry=None, **kwargs)
    directory = tmp_path/'synthetic'
    if fault == 'late_decision':
        rows = list(pipeline.extended.read_rows(directory)); rows[8002]['decision']['requested_command'][0] = .1
        # Replace only this test's synthetic temporary stream.
        (directory/pipeline.extended.original.stream.NAME).unlink()
        with pipeline.extended.writer(directory) as append:
            for row in rows: append(row)
    elif fault == 'late_role':
        path = directory/'command_tape.json'; rows = json.loads(path.read_text())
        rows[8002]['role'] = 'online_learned_round_trip_command'; path.write_text(json.dumps(rows))
    else:
        original = sessions[0].raw
        def raw():
            result = original(); result['post_slew_applied_command'][-1, 0] = .01; return result
        sessions[0].raw = raw
    with pytest.raises(AssertionError):
        pipeline.audit(2, result, 'synthetic', mode='reactive', input_root=tmp_path, robot_geometry=None, **kwargs)
    saved = json.loads((tmp_path/pipeline.resources.names('synthetic', 'audit')[1]).read_text())
    assert not saved['phase_completed'] and saved['error']
