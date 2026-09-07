#!/usr/bin/env python3
"""Four fixed continuous exploration/discovery/return development runs."""
import ast
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.whole_task_navigation_development import WholeTaskNavigation, MAX_SECONDS
from lewm.whole_task_scene_development import trial_specs
from lewm.whole_task_metrics_development import marker_centres_occluded, reduce_whole_task
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.fast_gyro_scan_session_development import FastGyroSession
from scripts.run_go2_multijunction_route_development_v1 import PhysicalStop
from scripts.run_go2_gyro_turn_assay_development_v1 import live_packet
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.run_go2_marker_beacon_development_v1 import static_identity, preflight as marker_preflight, OUTPUT as MARKER_STUDY
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT/'.generated/go2_whole_task_navigation_development_v1_attempt_001'
NEW_SOURCES = ('lewm/whole_task_navigation_development.py', 'lewm/whole_task_scene_development.py',
    'lewm/whole_task_metrics_development.py', 'lewm/whole_task_rgb_dataset_development.py',
    'lewm/tests/test_whole_task_navigation_development.py', 'lewm/tests/test_whole_task_metrics_development.py',
    'lewm/tests/test_whole_task_reader_development.py', 'lewm/tests/test_whole_task_evidence_development.py',
    'lewm/memory/episodic_route_hypotheses_development.py', 'lewm/tests/test_episodic_route_hypotheses_development.py',
    'scripts/run_go2_whole_task_navigation_development_v1.py', 'scripts/audit_go2_whole_task_navigation_development_v1.py',
    'docs/go2_whole_task_navigation_development_v1_2026-09-05.md')
LEAVES = ('physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
          'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
          'fast_gyro_samples.npz', 'fast_gyro_histories.npz', 'actuator_identity.json', 'terminal_actuator_gains.json',
          'static_objects.json', 'task_decisions.json', 'command_tape.json', 'task_ledgers.json', 'task_memory.json', 'process.log')


def source_closure(inherited):
    available = set(subprocess.run(['rg', '--files', '-g', '*.py', 'lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'],
                                  cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines())
    pending = [p for p in NEW_SOURCES if p.endswith('.py')]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited or name in inherited:
            continue
        if name not in available or (ROOT/name).resolve() != ROOT/name:
            raise ValueError('ignore-aware ordinary source required')
        visited.add(name)
        parts = Path(name).parts[:-1]
        for end in range(1, len(parts)+1):
            init = str(Path(*parts[:end])/'__init__.py')
            if init in available: pending.append(init)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node, ast.Import): modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level: module = '.'.join([*parts[:len(parts)-node.level+1], *([module] if module else [])])
                modules = [module, *[module+'.'+n.name for n in node.names]]
            else: continue
            for module in modules:
                prefix = module.split('.')[0]
                if prefix not in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'): continue
                stems = [module.replace('.', '/')]
                if prefix in ('lewm_genesis', 'lewm_worlds'): stems.append(prefix+'/'+stems[0])
                for stem in stems:
                    for candidate in (stem+'.py', stem+'/__init__.py'):
                        if candidate in available: pending.append(candidate)
    sources = {p: digest(ROOT/p) for p in visited | set(NEW_SOURCES)}
    if any(p in inherited and inherited[p] != h for p, h in sources.items()):
        raise ValueError('frozen source changed')
    return inherited | sources


def preflight():
    inherited, inputs, gait = marker_preflight()
    identities = {str((MARKER_STUDY/name).relative_to(ROOT)): sha for name, sha in (
        ('launch.json', '9514555d90fc360d07e0c7efe29afe8aa9047ddbab57295019e4a8dae321faa7'),
        ('result.json', '4300c237d9ff9ba3677ea58bc904faa56f42089e03a01e9ca3dd29e84520c019'),
        ('raw_artifact_audit.json', '1a99c1ce9ede3712849d26b554c1a5804954781dcab94cec53976ff020ccc2b9'),
        ('audit_v2_binding.json', '2da9963929ae2bb7b8171afe12b14007f6afa975ec35ed6738cc0e7f02b7ea9e'),
        ('raw_artifact_audit_v2.json', 'bda9e24e12fda6e60e8305cc8b564c976343d6999642a15bb9d2687d6a1b1ebf'))}
    verify_bindings(identities)
    correction = json.loads((MARKER_STUDY/'audit_v2_binding.json').read_text())
    audit = json.loads((MARKER_STUDY/'raw_artifact_audit_v2.json').read_text())
    if audit['status'] != 'PASS' or audit['audited_trials'] != 6:
        raise ValueError('completed marker interface required')
    inherited |= correction['source_sha256']; inputs |= identities
    verify_bindings(inherited | inputs | gait)
    sources = source_closure(inherited)
    verify_bindings(sources | inputs | gait)
    return sources, inputs, gait


def collect(spec, output):
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    session = controller = None
    stop = fault = terminal = None
    decisions, tape = [], []
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = FastGyroSession(spec, output); session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(output/'actuator_identity.json', gains)
        write_json(output/'static_objects.json', static_identity(session))
        session.settle_recorded()
        start = len(session.samples)-1
        packet_index = session.capture_current()
        binding = prefix_binding(session, packet_index)
        controller = WholeTaskNavigation(spec['method'], ArticulatedCollisionGeometry(URDF), memory_arm=spec['memory_arm'])
        try:
            for tick in range(MAX_SECONDS*10+1):
                index, packet = live_packet(session, output)
                now = packet['sensor_state']['decision_ns']
                try:
                    decision = controller.observe(packet, session.fast_buffer.packet(now_ns=now), now_ns=now)
                except SensorContractError as error:
                    fault = {'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples)-1, 'reason': str(error)}
                    terminal = 'FAILED_SENSOR'; break
                done = decision['terminal']
                decisions.append({'tick': tick, 'observation_index': index, 'pre_sample_index': len(session.samples)-1,
                                  'decision_ns': now, 'executed': not done, 'controller': decision})
                if done:
                    terminal = decision['status']; break
                if tick == MAX_SECONDS*10: raise ValueError('whole-task global budget exceeded')
                session.phase = 1
                tape.append({'phase': 1, 'pre_sample_index': len(session.samples)-1, 'requested_command': decision['requested_command']})
                session.command_tick(decision['requested_command'])
            for _ in range(5):
                session.phase = 2
                tape.append({'phase': 2, 'pre_sample_index': len(session.samples)-1, 'requested_command': [0., 0., 0.]})
                session.command_tick([0., 0., 0.])
        except PhysicalStop as error:
            stop = str(error)
        if stop is not None:
            controller.finish_physical_stop(now_ns=int(round(session.samples[-1]['timestamp_s']*1e9)))
        session.capture_current()
        raw = session.persist(output); session.persist_observations(output)
        terminal_gains = read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']: raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json', terminal_gains)
        write_json(output/'task_decisions.json', decisions); write_json(output/'command_tape.json', tape)
        write_json(output/'task_ledgers.json', controller.ledgers()); write_json(output/'task_memory.json', controller.memory_snapshot())
        # Evaluation-only camera/scene geometry never enters the controller.
        origin = np.asarray(session.image_audit[0]['world_from_optical'])[:3, 3]
        occluded = marker_centres_occluded(spec, origin)
        response = reduce_whole_task(raw, start, decisions, terminal=terminal or controller.status,
                                    stop_reason=stop, sensor_fault=fault, initial_marker_occluded=occluded)
        return {**{k: spec[k] for k in ('scene_id', 'case_index', 'method', 'memory_arm', 'layout_name')},
                'prefix_terminal_sample_index': start, 'branch_start_observation_index': packet_index, 'prefix_binding': binding,
                'physics_samples': len(session.samples), 'sensor_samples': len(session.sensor_rows),
                'fast_gyro_samples': len(session.fast_rows), 'rgb_packets': len(session.packet_rows), 'response': response}
    except Exception:
        if session is not None:
            if not (output/'physics_trace.npz').exists():
                session.persist(output); session.persist_observations(output)
            if not (output/'task_decisions.json').exists(): write_json(output/'task_decisions.json', decisions)
            if not (output/'command_tape.json').exists(): write_json(output/'command_tape.json', tape)
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed absent whole-task root required, no retry')
    if shutil.disk_usage(OUTPUT.parent).free < 10*1024**3: raise ValueError('less than ten GiB free')
    sources, inputs, gait = preflight(); specs = trial_specs(); OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', {'schema': 'whole_task_navigation_development.v1', 'trial_specs': specs,
        'source_sha256': sources, 'input_sha256': inputs, 'gait_sha256': gait,
        'scope': 'four fresh whole-task development trials, no held-out/JEPA/hardware qualification'})
    rows, prefixes = [], {}
    try:
        for spec in specs:
            directory = OUTPUT/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event': 'task_started', 'scene_id': spec['scene_id'], 'completed': len(rows), 'total': 4}), flush=True)
            with (directory/'process.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = collect(spec, directory)
            leaves = [*LEAVES, *(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))]
            row['artifact_sha256'] = {p: digest(directory/p) for p in leaves}
            write_json(directory/'result.json', row)
            reference = prefixes.setdefault(spec['case_index'], row['prefix_binding'])
            if any(reference[k] != row['prefix_binding'][k] for k in ('physics_arrays', 'history_arrays', 'physics_samples', 'timestamp_ns')):
                raise ValueError('paired settling prefix changed')
            rows.append(row)
            print(json.dumps({'event': 'task_finished', 'completed': len(rows), 'total': 4, 'scene_id': spec['scene_id'],
                              'response': row['response']}), flush=True)
        verify_bindings(sources | inputs | gait)
        write_json(OUTPUT/'result.json', {'status': 'COMPLETE', 'trials': rows, 'completed_trials': 4, 'planned_trials': 4,
                                        'launch_sha256': digest(OUTPUT/'launch.json')})
        print(json.dumps({'status': 'COMPLETE', 'trials': 4}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error), 'trials': rows,
            'completed_trials': len(rows), 'planned_trials': 4, 'launch_sha256': digest(OUTPUT/'launch.json')})
        raise


if __name__ == '__main__':
    main()
