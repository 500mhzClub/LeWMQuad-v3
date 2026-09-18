"""Synthetic orchestration plus unchanged raw loops; no native qualification."""
import ast
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from lewm.independent_round_trip_comparison_study_development import CASES, require_case
from lewm import independent_round_trip_multiarm_contract_development as contract
from scripts import independent_round_trip_multiarm_episode_development as episode
from scripts import independent_round_trip_multiarm_audit_development as audit
from scripts.maze_decision_stream_development import read_rows, writer


def test_acquisition_and_full_raw_replay_loops_keep_original_operations():
    from scripts import independent_residual_round_trip_episode_development as old_episode
    from scripts import independent_residual_round_trip_audit_development as old_audit
    def loop(function, target):
        tree = ast.parse(inspect.getsource(function))
        return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
            and ast.unparse(node.iter) == target)
    new = loop(episode.collect, 'range(MAX_OBSERVATIONS)')
    old = loop(old_episode.collect, 'range(MAX_OBSERVATIONS)')
    class Role(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'command_role':
                return ast.Constant('online_learned_round_trip_command')
            return self.generic_visit(node)
    assert ast.dump(Role().visit(new)) == ast.dump(old)
    assert ast.dump(loop(audit.audit, 'enumerate(read_rows(directory))')) == ast.dump(
        loop(old_audit.audit, 'enumerate(read_rows(directory))'))
    for name in ('audit_sensors', 'audit_auxiliary', 'audit_rasters_and_footprints',
            'renderer_audit', 'evaluate', 'audit_setup', 'audit_stops'):
        assert getattr(audit, name) is getattr(old_audit, name)
    from scripts.reactive_nominal_maze_command_audit_development import audit_commands as reactive
    from scripts.novel_maze_round_trip_command_audit_development import audit_commands as learned
    assert audit.audit_reactive_commands is reactive and audit.audit_learned_commands is learned


@pytest.mark.parametrize('field', ['case', 'layout_index', 'assignment',
    'high_level_world_model_configured', 'residual_anchored_continuation_enabled',
    'planning_map_variant', 'persistent_contact_history_retained', 'status'])
def test_treatment_substitution_rejected_before_sensor_or_controller_access(monkeypatch, field):
    case = CASES[0]
    result = dict(status=contract.COLLECTION_STATUS, **contract.treatment(case))
    result[field] = 'substitute'
    def forbidden(*args, **kwargs): pytest.fail('bad treatment reached raw access')
    monkeypatch.setattr(audit, 'validate_root', forbidden)
    with pytest.raises(ValueError): audit.audit(case, result, 'definition', input_root=None, robot_geometry=None)


@pytest.mark.parametrize('fault', ['missing', 'state', 'gradients', 'reactive_model'])
def test_live_model_integrity_gate(fault, monkeypatch):
    case = CASES[0]; arm = require_case(case)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(contract, 'state_digest', lambda state: arm.model_state_sha256)
    if fault == 'missing': model = None
    elif fault == 'state': monkeypatch.setattr(contract, 'state_digest', lambda state: 'changed')
    elif fault == 'gradients': model.weight.grad = torch.ones_like(model.weight)
    else: case = next(c for c in CASES if c.arm_name == 'reactive')
    with pytest.raises(ValueError): contract.verify_model(case, model)


def harness(monkeypatch, tmp_path, case):
    """Fake physical and perception boundaries, real tape/decision/timing loops."""
    events = []; sessions = []; controllers = []
    arm = require_case(case)
    mission = dict(arrivals=[], terminal='SYNTHETIC_FAILURE')

    class Controller:
        def observe(self, policy, *args, **kwargs):
            tick = policy['tick']
            return dict(requested_command=[.2, 0., 0.] if tick == 3 else [0., 0., 0.],
                terminal=None if tick < 4 else 'SYNTHETIC_FAILURE',
                mission_receipt=deepcopy(mission), new_selection=None, evidence=None)

    def create(selected, geometry, *, correction_admission):
        assert selected == case
        controller = Controller(); controllers.append(controller)
        model = None if arm.model_name is None else torch.nn.Linear(1, 1)
        return controller, model

    class Session:
        def __init__(self, spec, directory):
            self.spec = spec; self.directory = directory; self.phase = 0
            self.samples = []; self.model_manifest = []; self.auxiliary_audit = []; self.guard_rows = []
            build = NS(robot=object(), collision_floor=NS(links=[NS(idx=1)], geoms=[NS(idx=1)]),
                visual_surfaces=[NS(links=[NS(idx=2)], geoms=[])], scene=NS(destroy=lambda: events.append('destroy')))
            self.ctx = NS(build=build, runner=NS(_leg_dof_idx=np.arange(12)), policy=NS(env_cfg={}))
            self.requested = []; self.applied = []; self.phases = []; sessions.append(self)
        def install_contact_identity(self): events.append('contact_identity')
        def settle_recorded(self):
            self.samples = [None]*750; self.requested = [[0., 0., 0.]]*750
            self.applied = [[0., 0., 0.]]*750; self.phases = [0]*750
        def capture_current(self):
            self.model_manifest.append(dict(physical_sample_index=len(self.samples)-1))
            self.auxiliary_audit.append({})
        def sensor_packets(self):
            tick = len(self.model_manifest)-1
            return dict(tick=tick), {}, {}, {}, {}, 1_500_000_000+tick*100_000_000
        def command_tick(self, request):
            self.samples.extend([None]*50); self.requested.extend([request]*50)
            prior = np.asarray(self.applied[-1], np.float32); delta = np.array([.25, 0., .35], np.float32)
            applied = np.clip(np.asarray(request, np.float32), prior-delta, prior+delta).astype(np.float64).tolist()
            self.applied.extend([applied]*50); self.phases.extend([self.phase]*50); self.capture_current()
        def persist(self, directory): events.append('persist')
        def persist_observations(self, directory):
            (directory/'auxiliary_camera_audit.json').write_text(json.dumps(self.auxiliary_audit))
            events.append('persist_observations')
        def raw(self):
            pose = np.zeros((len(self.samples), 7)); pose[:, :2] = self.spec['geometry']['spawn_se2_world'][:2]
            pose[:, 6] = 1.
            return dict(timestamp_s=np.arange(1, len(self.samples)+1)*.002,
                base_pose_world=pose, requested_command=np.asarray(self.requested, np.float64),
                applied_command=np.asarray(self.applied, np.float64), post_slew_applied_command=np.asarray(self.applied, np.float64),
                phase=np.asarray(self.phases))

    monkeypatch.setattr(episode, 'validate_root', lambda root: root)
    monkeypatch.setattr(audit, 'validate_root', lambda root: root)
    monkeypatch.setattr(audit, 'artifact_path', lambda root, name: root/name)
    monkeypatch.setattr(episode, 'create', create); monkeypatch.setattr(audit, 'create', create)
    monkeypatch.setattr(contract, 'state_digest', lambda state: arm.model_state_sha256)
    monkeypatch.setattr(episode, 'initialize_genesis', lambda **kwargs: events.append('initialize'))
    monkeypatch.setattr(episode, 'shutdown_genesis', lambda: events.append('shutdown'))
    monkeypatch.setattr(episode, 'IndependentRoundTripSession', Session)
    monkeypatch.setattr(episode.shutil, 'disk_usage', lambda root: NS(free=500*1024**3))
    monkeypatch.setattr(episode, 'configure_gains', lambda *args: dict(effective={'fixed': True}))
    monkeypatch.setattr(episode, 'read_gains', lambda *args: {'fixed': True})
    monkeypatch.setattr(episode, 'capture_native_robot_geometry', lambda *args: {})
    monkeypatch.setattr(episode, 'appearance_environment_identity', lambda *args: {})
    monkeypatch.setattr(episode, 'native_friction', lambda build, mu: dict(solver_friction=mu,
        solver_ratio=np.ones((1, 28)).tolist(), physics_steps=len(sessions[-1].samples)))
    def admit(session, definition): (session.directory/'setup_checks.json').write_text('{}')
    monkeypatch.setattr(episode, 'admit_context_setup', admit)
    def sensors(directory, spec, result):
        session = sessions[-1]
        return session.raw(), {}, {}, {}, session.model_manifest, None, {}, {'depth_checks': []}
    monkeypatch.setattr(audit, 'audit_sensors', sensors)
    monkeypatch.setattr(audit, 'audit_auxiliary', lambda *args: [dict(auxiliary_visibility_pass=True)]*15)
    monkeypatch.setattr(audit, 'IntentReturnRGBDReplay', lambda directory: NS(packet=lambda tick:
        (dict(tick=tick), {}, {}, 1_500_000_000+tick*100_000_000)))
    monkeypatch.setattr(audit, 'public_acquisition', lambda row: row)
    monkeypatch.setattr(audit, 'packet', lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(audit, 'audit_setup', lambda *args: {})
    monkeypatch.setattr(audit, 'audit_stops', lambda *args: None)
    monkeypatch.setattr(audit, 'audit_rasters_and_footprints', lambda *args: [dict(
        stable_interior_metric_pass=True, near_occlusion_failure=False,
        original_strict_score={'passes_sampled_physical_visibility': True})]*15)
    monkeypatch.setattr(audit, 'renderer_audit', lambda directory: {'synthetic_only': True})
    def evaluate(raw, observed, result, *, layout_index):
        assert layout_index == case.layout_index
        return dict(native_round_trip_candidate_pass=False)
    monkeypatch.setattr(audit, 'evaluate', evaluate)
    return events, sessions, controllers


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
def test_all_assigned_cases_collect_and_replay_negative_synthetic_episode(monkeypatch, tmp_path, case):
    events, sessions, controllers = harness(monkeypatch, tmp_path, case)
    result = episode.collect(case, 'definition', output=tmp_path, geometry=object())
    report = audit.audit(case, result, 'definition', input_root=tmp_path, robot_geometry=object())
    assert result['decisions'] == 15 and result['completed_ticks'] == 14 and result['physics_samples'] == 1450
    assert result['terminal_zero_ticks'] == 10 and result['schedule_terminal'] == 'SYNTHETIC_FAILURE'
    assert report['raw_controller_command_replay_pass'] and report['raw_command_audit_pass']
    assert not report['verified_round_trip'] and not report['navigation_qualified']
    assert report['assignment'] == result['assignment'] == contract.treatment(case)['assignment']
    assert report['high_level_world_model_used'] == (case.arm_name != 'reactive')
    assert len(controllers) == 2 and controllers[0] is not controllers[1]
    assert events == ['initialize', 'contact_identity', 'persist', 'persist_observations', 'destroy', 'shutdown']
    tape = json.loads((tmp_path/case.name/'command_tape.json').read_text())
    assert tape[3]['role'] == contract.command_role(case)


@pytest.mark.parametrize('fault', ['decision', 'command_role', 'native_command', 'timing'])
def test_corrupted_replay_evidence_is_rejected(monkeypatch, tmp_path, fault):
    case = CASES[0]; _, sessions, _ = harness(monkeypatch, tmp_path, case)
    result = episode.collect(case, 'definition', output=tmp_path, geometry=object())
    directory = tmp_path/case.name
    if fault == 'decision':
        rows = list(read_rows(directory)); rows[3]['decision']['requested_command'][0] = .1
        # Only this test's newly created synthetic file is replaced.
        (directory/episode.DECISIONS).unlink()
        with writer(directory) as append:
            for row in rows: append(row)
    elif fault == 'command_role':
        path = directory/'command_tape.json'; tape = json.loads(path.read_text())
        tape[3]['role'] = 'online_reactive_round_trip_command'; path.write_text(json.dumps(tape))
    elif fault == 'native_command': sessions[-1].requested[950] = [.1, 0., 0.]
    else:
        path = directory/'decision_stream_timing.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]; rows[3]['iteration_with_receipt_wall_ms'] = -1
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(AssertionError): audit.audit(case, result, 'definition', input_root=tmp_path, robot_geometry=object())


def test_model_failure_precedes_any_new_scene_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(episode, 'validate_root', lambda root: root)
    def fail(*args, **kwargs): raise ValueError('invalid assigned model')
    monkeypatch.setattr(episode, 'create', fail)
    monkeypatch.setattr(episode, 'initialize_genesis', lambda **kwargs: pytest.fail('scene before model admission'))
    with pytest.raises(ValueError, match='invalid assigned model'):
        episode.collect(CASES[0], 'definition', output=tmp_path, geometry=object())
    assert not (tmp_path/CASES[0].name).exists()
