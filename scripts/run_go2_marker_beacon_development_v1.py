#!/usr/bin/env python3
"""Fixed six-case physical RGB marker acquisition; never follows an oracle route."""
import ast
import contextlib
from dataclasses import asdict
import importlib.metadata
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
from lewm.marker_beacon_scene_development import trials
from lewm.rgb_marker_beacon_development import MarkerDiscovery
from scripts.run_go2_multijunction_route_development_v1 import RouteSession, PhysicalStop
from scripts.run_go2_gyro_turn_assay_development_v1 import live_packet
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT/'.generated/go2_marker_beacon_development_v1_attempt_001'
PRIOR = ROOT/'.generated/go2_task_acquisition_continuation_development_v1_attempt_001'
NEW_SOURCES = ('lewm/rgb_marker_beacon_development.py', 'lewm/marker_beacon_scene_development.py',
               'lewm/tests/test_rgb_marker_beacon_development.py',
               'scripts/run_go2_marker_beacon_development_v1.py',
               'scripts/audit_go2_marker_beacon_development_v1.py',
               'docs/go2_marker_beacon_development_v1_2026-09-05.md')
LEAVES = ('physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
          'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
          'actuator_identity.json', 'terminal_actuator_gains.json', 'static_objects.json',
          'marker_decisions.json', 'process.log')


def preflight():
    identities = {str((PRIOR/name).relative_to(ROOT)): sha for name, sha in (
        ('launch.json', 'ed564ec9729d1f724f871438e15a3fdfddafc654afa5092771ef34ab8319eb06'),
        ('result.json', '5ac8a8109ab0c4a3442955891f4a0365ff5a342175fd6bd33b0dbee808c0e135'),
        ('raw_artifact_audit.json', '5b0ab7f848c0432eeb762f0694c65b58ee59323f158585bc3d4199f49e9c37ca'))}
    verify_bindings(identities)
    prior = json.loads((PRIOR/'launch.json').read_text())
    report = json.loads((PRIOR/'result.json').read_text())
    audit = json.loads((PRIOR/'raw_artifact_audit.json').read_text())
    if report['status'] != 'COMPLETE' or audit['status'] != 'PASS' or audit['audited_trials'] != 28:
        raise ValueError('completed audited predecessor required')
    sources = prior['source_sha256']
    # New imports must either be explicitly new or already in the bound closure.
    # Discovery honors .ignore. No source export or recursive materialization.
    available = set(subprocess.run(['rg', '--files', '-g', '*.py', 'lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'],
                                  cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines())
    for name in NEW_SOURCES:
        if name.endswith('.py'):
            if name not in available or (ROOT/name).resolve() != ROOT/name:
                raise ValueError('ignore-aware ordinary new source required')
            for node in ast.walk(ast.parse((ROOT/name).read_text())):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) else [])
                for module in modules:
                    if not module or module.split('.')[0] not in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'):
                        continue
                    parts = module.split('.')
                    stems = [module.replace('.', '/')]
                    if parts[0] in ('lewm_genesis', 'lewm_worlds'):
                        stems.append(parts[0]+'/'+module.replace('.', '/'))
                    resolved = [p for stem in stems for p in (stem+'.py', stem+'/__init__.py') if p in available]
                    if not resolved or any(p not in sources and p not in NEW_SOURCES for p in resolved):
                        raise ValueError(f'unbound new local import: {module}')
    sources = sources | {p: digest(ROOT/p) for p in NEW_SOURCES}
    inputs, gait = prior['input_sha256'] | identities, prior['gait_sha256']
    verify_bindings(sources | inputs | gait)
    return sources, inputs, gait


def static_identity(session):
    import genesis as gs
    entities = {str(e.name): e for e in session.ctx.build.scene.entities}
    rows = []
    for obj in session.ctx.pack.static_objects:
        entity = entities[obj.object_id]
        if entity.n_geoms != 1 or entity.geoms[0].type != gs.GEOM_TYPE.BOX:
            raise ValueError('exact native static box collision required')
        rows.append({'pack_object': asdict(obj), 'native_name': str(entity.name),
                     'native_collision_boxes': int(entity.n_geoms),
                     'native_box_size': np.asarray(entity.geoms[0].data).tolist(),
                     'native_position': entity.get_pos().detach().cpu().numpy().reshape(-1, 3)[0].tolist(),
                     'native_quaternion_wxyz': entity.get_quat().detach().cpu().numpy().reshape(-1, 4)[0].tolist(),
                     'fixed': bool(entity.morph.fixed), 'collision_enabled': bool(entity.morph.collision),
                     'surface_rgb': np.asarray(entity.surface.diffuse_texture.color)[:3].tolist()})
    return rows


def collect(spec, output):
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    session, stop = None, None
    decisions = []
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = RouteSession(spec, output)
        session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                session.ctx.policy.env_cfg, 'checkpoint')
        write_json(output/'actuator_identity.json', gains)
        write_json(output/'static_objects.json', static_identity(session))
        observer = MarkerDiscovery()
        try:
            session.settle_recorded()
            for tick in range(5):
                index, packet = live_packet(session, output)
                now = packet['sensor_state']['decision_ns']
                decisions.append({'observation_index': index, 'pre_sample_index': len(session.samples)-1,
                                  'result': observer.observe(packet, now_ns=now)})
                if tick < 4:
                    session.phase = 1
                    session.command_tick([0., 0., 0.])
        except PhysicalStop as error:
            stop = str(error)
        session.capture_current()
        raw = session.persist(output)
        session.persist_observations(output)
        terminal = read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal != gains['effective']:
            raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json', terminal)
        write_json(output/'marker_decisions.json', decisions)
        detected = [bool(row['result']['detections']) for row in decisions]
        discoveries = [row['observation_index'] for row in decisions if row['result']['newly_discovered']]
        return {'scene_id': spec['scene_id'], 'case_index': spec['case_index'], 'marker_case': spec['marker_case'],
                'stop_reason': stop, 'physics_samples': len(session.samples), 'sensor_samples': len(session.sensor_rows),
                'rgb_packets': len(session.packet_rows), 'detected_frames': detected, 'discovery_indices': discoveries,
                'distinct_marker_count': decisions[-1]['result']['distinct_marker_count'] if decisions else 0,
                'native_contact': bool(raw['physics_contact'].any()),
                'completed_probe': stop is None and len(session.samples) == 950 and len(decisions) == 5}
    except Exception:
        if session is not None:
            if not (output/'physics_trace.npz').exists():
                session.persist(output)
                session.persist_observations(output)
            if not (output/'marker_decisions.json').exists():
                write_json(output/'marker_decisions.json', decisions)
        raise
    finally:
        try:
            if session is not None:
                session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed absent output root required; no retry')
    if shutil.disk_usage(OUTPUT.parent).free < 10*1024**3:
        raise ValueError('less than ten GiB free')
    sources, inputs, gait = preflight()
    specs = trials()
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', {'schema': 'marker_beacon_development.v1', 'trial_specs': specs,
        'source_sha256': sources, 'input_sha256': inputs, 'gait_sha256': gait,
        'packages': {p: importlib.metadata.version(p) for p in ('genesis-world', 'torch', 'numpy', 'scipy', 'Pillow')},
        'scope': 'fixed six-case stationary physical marker acquisition, no navigation or hardware qualification'})
    rows = []
    try:
        for spec in specs:
            directory = OUTPUT/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event': 'started', 'scene_id': spec['scene_id'], 'completed': len(rows), 'total': 6}), flush=True)
            with (directory/'process.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = collect(spec, directory)
            leaves = [*LEAVES, *(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))]
            row['artifact_sha256'] = {p: digest(directory/p) for p in leaves}
            write_json(directory/'result.json', row); rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'artifact_sha256'}), flush=True)
            if not row['completed_probe']:
                raise ValueError('incomplete physical probe; preserve output and stop')
        verify_bindings(sources | inputs | gait)
        write_json(OUTPUT/'result.json', {'status': 'COMPLETE', 'trials': rows, 'completed_trials': len(rows),
                                        'planned_trials': 6, 'launch_sha256': digest(OUTPUT/'launch.json')})
        print(json.dumps({'status': 'COMPLETE', 'trials': len(rows)}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error), 'trials': rows,
            'completed_trials': len(rows), 'planned_trials': 6, 'launch_sha256': digest(OUTPUT/'launch.json')})
        raise


if __name__ == '__main__':
    main()
