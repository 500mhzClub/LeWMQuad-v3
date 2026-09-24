#!/usr/bin/env python3
"""Initial observed-heading alignment followed by the unchanged continuous task."""
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
from lewm.initially_aligned_scene_development import trials
from lewm.observed_continuation_development import METHODS
from lewm.initially_aligned_continuation_development import InitiallyAlignedContinuation as ObservedContinuation
from lewm.initially_aligned_metrics_development import reduce_aligned_continuation as reduce_continuation
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.online_temporal_choice_development import OnlineTemporalChoice, STUDY, LAUNCH_SHA, AUDIT_SHA
from scripts.fast_gyro_scan_session_development import FastGyroSession
from lewm.causal_sensor_state import SensorContractError
from lewm.actuator_gain_development import configure_gains, read_gains
from scripts.run_go2_gyro_turn_assay_development_v1 import live_packet
from scripts.run_go2_multijunction_route_development_v1 import PhysicalStop
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_initially_aligned_continuation_development_v1_attempt_001'
NEW_SOURCES = (
    'lewm/initially_aligned_continuation_development.py',
    'lewm/initially_aligned_scene_development.py',
    'lewm/initially_aligned_metrics_development.py',
    'lewm/tests/test_initially_aligned_continuation_development.py',
    'scripts/run_go2_initially_aligned_continuation_development_v1.py',
    'scripts/audit_go2_initially_aligned_continuation_development_v1.py',
    'docs/go2_initially_aligned_continuation_development_v1_2026-09-05.md')
LEAVES = (
    'physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
    'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
    'actuator_identity.json', 'terminal_actuator_gains.json', 'continuation_decisions.json',
    'command_tape.json', 'process.log', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz',
    'provisional_ledgers.json')


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


def collect(spec, output, template=None):
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    geometry = ArticulatedCollisionGeometry(URDF)
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = controller = None
    stop_reason = terminal = sensor_fault = None
    decisions, tape = [], []
    try:
        session = FastGyroSession(spec, output)
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
            controller = ObservedContinuation(spec['method'], geometry, template)
            try:
                for tick in range(801):
                    index, packet = live_packet(session, output)
                    now = packet['sensor_state']['decision_ns']
                    try:
                        decision = controller.observe(packet, session.fast_buffer.packet(now_ns=now), now_ns=now)
                    except SensorContractError as error:
                        sensor_fault = {'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples)-1,
                                        'reason': str(error)}
                        terminal = 'FAILED_SENSOR'
                        break
                    done = decision['terminal']
                    decisions.append({'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples)-1,
                                      'decision_ns': now, 'executed': not done, 'controller': decision})
                    if done:
                        terminal = decision['status']
                        break
                    if tick == 800:
                        raise ValueError('traversal exceeded fixed total budget')
                    session.phase = 1
                    tape.append({'phase': 1, 'pre_sample_index': len(session.samples)-1,
                                 'requested_command': decision['requested_command']})
                    session.command_tick(decision['requested_command'])
                for _ in range(5):
                    session.phase = 2
                    tape.append({'phase': 2, 'pre_sample_index': len(session.samples) - 1, 'requested_command': [0., 0., 0.]})
                    session.command_tick([0., 0., 0.])
            except PhysicalStop as error:
                stop_reason = str(error)
        if stop_reason is not None and controller is not None:
            controller.finish_physical_stop()
        session.capture_current()
        raw = session.persist(output)
        session.persist_observations(output)
        terminal_gains = read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']:
            raise ValueError('actuator gain drift')
        write_json(output / 'terminal_actuator_gains.json', terminal_gains)
        write_json(output / 'continuation_decisions.json', decisions)
        write_json(output / 'command_tape.json', tape)
        write_json(output / 'provisional_ledgers.json', controller.ledgers() if controller is not None else [])
        response = reduce_continuation(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, geometry)
        return {'scene_id': spec['scene_id'], 'case_index': spec['case_index'], 'method': spec['method'],
                'destination_motif': spec['destination_motif'], 'initial_offset_m': spec['initial_offset_m'],
                'initial_heading_rad': spec['initial_heading_rad'],
                'prefix_terminal_sample_index': start, 'prefix_binding': binding,
                'branch_start_observation_index': packet_index, 'response': response,
                'physics_samples': len(session.samples), 'sensor_samples': len(session.sensor_rows),
                'rgb_packets': len(session.packet_rows), 'fast_gyro_samples': len(session.fast_rows)}
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


def preflight():
    predecessor = ROOT / '.generated/go2_observed_continuation_development_v1_attempt_001'
    inputs = {str((predecessor / leaf).relative_to(ROOT)): sha for leaf, sha in (
        ('launch.json', 'cda6d07ddc8238f2a3b540924fab153829a031e24e4a7db8f046a0f4b7f162d7'),
        ('result.json', '003147bede55befe019362335faefe5173ae6e98c4b1bbb429fb22ffa44ff6bf'),
        ('raw_artifact_audit_population_v2.json', 'ad4f7d0e517041566b36b88a94b39a5d0527a93fc65790fade1726cc01ddfd3d'),
        ('audit_population_v2_binding.json', '72b06a87062929232872d1b76c30f420ecfd382e182037b8650d96746bd38aa9'))}
    inputs.update({str((STUDY/'launch.json').relative_to(ROOT)): LAUNCH_SHA,
                   str((STUDY/'raw_artifact_audit.json').relative_to(ROOT)): AUDIT_SHA})
    verify_bindings(inputs)
    prior = json.loads((predecessor/'launch.json').read_text())
    temporal = json.loads((STUDY/'launch.json').read_text())
    temporal_audit = json.loads((STUDY/'raw_artifact_audit.json').read_text())
    inputs.update(prior['input_sha256'])
    inputs[str((STUDY/'result.json').relative_to(ROOT))] = temporal_audit['study_result_sha256']
    inputs[str(URDF)] = '4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4'
    correction = json.loads((predecessor/'audit_population_v2_binding.json').read_text())
    verify_bindings(correction['source_sha256'] | correction['input_sha256'])
    inputs.update(correction['input_sha256'])
    inherited = dict(prior['source_sha256'])
    for name, sha in correction['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('conflicting correction source')
        inherited[name] = sha
    for name, sha in temporal['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('conflicting predecessor source')
        inherited[name] = sha
    templates = {method: OnlineTemporalChoice.from_completed_study(method)
                 for method in METHODS if method != 'fixed_forward'}
    for template in templates.values():
        for binding in template.bindings:
            path = STUDY / f"{binding['seed']}-{binding['condition']}" / 'final.pt'
            inputs[str(path.relative_to(ROOT))] = binding['checkpoint_sha256']
    sources = source_closure(inherited)
    gait = prior['gait_sha256']
    verify_bindings(sources | inputs | gait)
    return sources, inputs, gait, templates


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh traversal root required')
    if shutil.disk_usage(OUTPUT.parent).free < 10*1024**3:
        raise ValueError('less than ten GiB free')
    sources, inputs, gait, templates = preflight()
    specs = trials()
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', {'schema': 'initially_aligned_continuation_development.v1', 'trial_specs': specs,
        'source_sha256': sources, 'input_sha256': inputs, 'gait_sha256': gait,
        'scope': 'sixteen initial-alignment two-leg development trials; no independent-maze or hardware qualification'})
    rows, prefixes = [], {}
    try:
        for spec in specs:
            directory = OUTPUT/spec['scene_id']
            directory.mkdir()
            print(json.dumps({'event': 'continuation_started', 'scene_id': spec['scene_id'],
                              'completed': len(rows), 'total': len(specs)}), flush=True)
            with (directory/'process.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = collect(spec, directory, templates.get(spec['method']))
            leaves = [*LEAVES, *(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))]
            row['artifact_sha256'] = {name: digest(directory/name) for name in leaves}
            write_json(directory/'result.json', row)
            reference = prefixes.setdefault(spec['case_index'], row['prefix_binding'])
            for key in ('physics_arrays', 'history_arrays', 'physics_samples', 'timestamp_ns'):
                if reference[key] != row['prefix_binding'][key]:
                    raise ValueError('paired settling prefix changed')
            rows.append(row)
            response = row['response']
            print(json.dumps({'event': 'continuation_finished', 'completed': len(rows), 'total': len(specs),
                'method': spec['method'], 'controller_terminal': response['controller_terminal'],
                'stop_reason': response['stop_reason'], 'two_leg_success': response['two_leg_integration_success'],
                'first_leg_success': response['checks']['first_leg_integration'],
                'second_leg_success': response['checks']['second_leg_integration']}), flush=True)
        verify_bindings(sources | inputs | gait)
        write_json(OUTPUT/'result.json', {'status': 'COMPLETE', 'completed_trials': len(rows),
            'planned_trials': len(specs), 'trials': rows, 'launch_sha256': digest(OUTPUT/'launch.json')})
        print(json.dumps({'status': 'COMPLETE', 'trials': len(rows)}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json', {'status': 'INFRASTRUCTURE_FAILURE', 'error': repr(error),
            'completed_trials': len(rows), 'planned_trials': len(specs), 'trials': rows,
            'launch_sha256': digest(OUTPUT/'launch.json')})
        raise


if __name__ == '__main__':
    main()


