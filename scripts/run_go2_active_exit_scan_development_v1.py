#!/usr/bin/env python3
"""Sixteen fixed fresh actual-RGB/body junction scans; no model or gate fitting."""
import ast
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.active_gyro_scan_development import ActiveGyroScan
from lewm.active_exit_scan_scene_development import scan_scenes
from lewm.active_exit_scan_metrics_development import reduce_scan
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.causal_sensor_state import SensorContractError
from lewm.actuator_gain_development import configure_gains, read_gains
from scripts.run_go2_gyro_turn_assay_development_v1 import live_packet
from scripts.run_go2_multijunction_route_development_v1 import RouteSession, PhysicalStop
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_active_exit_scan_development_v1_attempt_001'
NEW_SOURCES = (
    'lewm/active_gyro_scan_development.py', 'lewm/active_exit_scan_scene_development.py',
    'lewm/active_exit_scan_metrics_development.py', 'lewm/rgb_exit_candidates_development.py',
    'scripts/run_go2_active_exit_scan_development_v1.py', 'scripts/audit_go2_active_exit_scan_development_v1.py',
    'lewm/tests/test_active_gyro_scan_development.py', 'lewm/tests/test_rgb_exit_candidates_development.py',
    'lewm/tests/test_active_exit_scan_evidence_development.py',
    'docs/go2_active_exit_scan_development_v1_2026-09-05.md')
LEAVES = ('physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
          'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
          'actuator_identity.json', 'terminal_actuator_gains.json', 'scan_decisions.json', 'command_tape.json', 'process.log')


def source_closure(inherited):
    """Read explicit ordinary Python imports discovered with tracked .ignore."""
    available = set(subprocess.run(['rg', '--files', '-g', '*.py', 'lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'],
                                  cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines())
    pending = [p for p in NEW_SOURCES if p.endswith('.py')]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        path = Path(name)
        if (name not in available or path.is_absolute() or '..' in path.parts
                or any(s in ('sealed', 'sealed_test.json') or s.startswith('sealed_') for s in path.parts)
                or (ROOT / path).resolve() != ROOT / path):
            raise ValueError('explicit ignore-aware source required')
        visited.add(name)
        parts = path.parts[:-1]
        for end in range(1, len(parts) + 1):
            init = str(Path(*parts[:end]) / '__init__.py')
            if init in available:
                pending.append(init)
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = '.'.join([*parts[:len(parts) - node.level + 1], *([module] if module else [])])
                modules = [module, *[module + '.' + n.name for n in node.names]]
            else:
                continue
            for module in modules:
                if module.split('.')[0] in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'):
                    for candidate in (module.replace('.', '/') + '.py', module.replace('.', '/') + '/__init__.py'):
                        if candidate in available:
                            pending.append(candidate)
    bindings = {name: digest(ROOT / name) for name in visited | set(NEW_SOURCES)}
    if any(name in inherited and inherited[name] != sha for name, sha in bindings.items()):
        raise ValueError('frozen predecessor source changed')
    return inherited | bindings


def sensor_decision(controller, ground, packet, *, tick, observation_id):
    """Only deployment-form sensor packets enter this runtime boundary."""
    now = packet['sensor_state']['decision_ns']
    state = ground.begin(packet, now_ns=now) if tick == 0 else ground.step(packet, now_ns=now)
    decision = controller.begin(packet, now_ns=now) if tick == 0 else controller.step(packet, now_ns=now)
    observation = observe_exit_candidates(packet, state, now_ns=now, observation_id=observation_id)
    observation = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in observation.items() if k != 'candidates'}
    return decision, state, observation


def collect(spec, output):
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = None
    stop_reason = terminal = sensor_fault = None
    decisions, tape = [], []
    try:
        session = RouteSession(spec, output)
        session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(output / 'actuator_identity.json', gains)
        try:
            session.settle_recorded()
        except PhysicalStop as error:
            stop_reason = str(error)
        start = len(session.samples) - 1
        packet_index = session.capture_current()
        binding = prefix_binding(session, packet_index)
        if stop_reason is None:
            controller, ground = ActiveGyroScan(), CausalGravityFeedbackGround('transported_feedback')
            try:
                for tick in range(301):
                    index, packet = live_packet(session, output)
                    try:
                        decision, state, observation = sensor_decision(controller, ground, packet, tick=tick,
                                                                     observation_id=f'frame-{index}')
                    except SensorContractError as error:
                        sensor_fault = {'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples) - 1,
                                        'reason': str(error)}
                        terminal = 'FAILED_SENSOR'
                        break
                    done = decision['status'].startswith(('COMPLETE', 'FAILED_'))
                    decisions.append({'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples) - 1,
                                      'decision_ns': packet['sensor_state']['decision_ns'], 'executed': not done,
                                      'selected_view': tick == 0 or decision['new_completed_view'] is not None,
                                      'controller': decision, 'ground_state': state, 'observation': observation})
                    if done:
                        terminal = decision['status']
                        break
                    if tick == 300:
                        raise ValueError('scan exceeded fixed thirty-second budget')
                    session.phase = 1
                    tape.append({'phase': 1, 'pre_sample_index': len(session.samples) - 1,
                                 'requested_command': decision['requested_command']})
                    session.command_tick(decision['requested_command'])
                for _ in range(5):
                    session.phase = 2
                    tape.append({'phase': 2, 'pre_sample_index': len(session.samples) - 1, 'requested_command': [0., 0., 0.]})
                    session.command_tick([0., 0., 0.])
            except PhysicalStop as error:
                stop_reason = str(error)
        session.capture_current()
        raw = session.persist(output)
        session.persist_observations(output)
        terminal_gains = read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']:
            raise ValueError('actuator gain drift')
        write_json(output / 'terminal_actuator_gains.json', terminal_gains)
        write_json(output / 'scan_decisions.json', decisions)
        write_json(output / 'command_tape.json', tape)
        response = reduce_scan(spec, raw, start, decisions, terminal, stop_reason, sensor_fault)
        return {'scene_id': spec['scene_id'], 'case_index': spec['case_index'], 'motif': spec['motif'],
                'width_m': spec['width_m'], 'initial_heading_rad': spec['initial_heading_rad'],
                'prefix_terminal_sample_index': start, 'prefix_binding': binding,
                'branch_start_observation_index': packet_index, 'response': response,
                'physics_samples': len(session.samples), 'sensor_samples': len(session.sensor_rows),
                'rgb_packets': len(session.packet_rows)}
    except Exception:
        if session is not None and not (output / 'physics_trace.npz').exists():
            session.persist(output)
            session.persist_observations(output)
        raise
    finally:
        try:
            if session is not None:
                session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh scan root required')
    if shutil.disk_usage(OUTPUT.parent).free < 10 * 1024 ** 3:
        raise ValueError('less than ten GiB free')
    predecessor = ROOT / '.generated/go2_gravity_feedback_ground_development_v1_attempt_001'
    inputs = {str((predecessor / 'launch.json').relative_to(ROOT)): 'f3b26882fbfebdd2730b106b094cf0d40e6df388f19121ea59f63cf9311e46c0',
              str((predecessor / 'result.json').relative_to(ROOT)): '9c7534cc56f76ea36cd92737ed1dcf93463b9fcf7067e6816748aa6bb1d64bc2'}
    verify_bindings(inputs)
    prior = json.loads((predecessor / 'launch.json').read_text())
    inputs.update(prior['input_sha256'])
    sources = source_closure(prior['source_sha256'])
    gait_launch = json.loads((ROOT / '.generated/go2_gyro_turn_assay_development_v1_attempt_001/launch.json').read_text())
    gait = gait_launch['gait_sha256']
    verify_bindings(sources | inputs | gait)
    specs = scan_scenes()
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'schema': 'active_exit_scan_development.v1', 'trial_specs': specs,
               'source_sha256': sources, 'input_sha256': inputs, 'gait_sha256': gait,
               'scope': 'sixteen fresh local active scans; no learned controller, traversal, maze or hardware qualification'})
    rows = []
    try:
        for spec in specs:
            directory = OUTPUT / spec['scene_id']
            directory.mkdir()
            print(json.dumps({'event': 'scan_started', 'scene_id': spec['scene_id'], 'completed': len(rows), 'total': 16}), flush=True)
            with (directory / 'process.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = collect(spec, directory)
            leaves = [*LEAVES, *(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))]
            row['artifact_sha256'] = {name: digest(directory / name) for name in leaves}
            write_json(directory / 'result.json', row)
            rows.append(row)
            response = row['response']
            print(json.dumps({'event': 'scan_finished', 'completed': len(rows), 'total': 16,
                              'physical_task_success': response['physical_task_success'], 'stop_reason': response['stop_reason'],
                              'controller_terminal': response['controller_terminal'],
                              'opening_sides': response['selected_views']['open_sides_covered'],
                              'expected_open_sides': response['selected_views']['open_sides_expected']}), flush=True)
        verify_bindings(sources | inputs | gait)
        write_json(OUTPUT / 'result.json', {'status': 'COMPLETE', 'completed_trials': 16, 'planned_trials': 16,
                   'trials': rows, 'launch_sha256': digest(OUTPUT / 'launch.json')})
        print(json.dumps({'status': 'COMPLETE', 'trials': 16}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'INFRASTRUCTURE_FAILURE', 'error': repr(error),
                   'completed_trials': len(rows), 'planned_trials': 16, 'trials': rows,
                   'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
