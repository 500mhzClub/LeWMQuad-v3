#!/usr/bin/env python3
"""Two fixed actual-render depth interface cases, then continuous geometry work."""
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
from lewm.marker_beacon_scene_development import probe_spec
from lewm.rgb_marker_beacon_development import MarkerDiscovery
from lewm.depth_geometry_evaluation_development import evaluate_depth as evaluate_physical_reference
from lewm.visual_surface_depth_evaluation_development import evaluate_depth, check_floor_identity
from scripts.single_sample_rgbd_session_development import SingleSampleRGBDSession as RouteSession
from scripts.run_go2_multijunction_route_development_v1 import PhysicalStop
from scripts.run_go2_gyro_turn_assay_development_v1 import live_packet
from scripts.run_go2_marker_beacon_development_v1 import static_identity, LEAVES as RGB_LEAVES
from scripts.run_go2_rgbd_observation_development_v1 import preflight as prior_preflight, OUTPUT as PRIOR
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT/'.generated/go2_single_sample_rgbd_observation_development_v1_attempt_001'
NEW_SOURCES = ('scripts/single_sample_rgbd_session_development.py',
    'lewm/visual_surface_depth_evaluation_development.py',
    'scripts/run_go2_single_sample_rgbd_observation_development_v1.py',
    'scripts/audit_go2_single_sample_rgbd_observation_development_v1.py',
    'lewm/tests/test_single_sample_rgbd_evidence_development.py',
    'docs/go2_single_sample_rgbd_observation_development_v1_2026-09-05.md',
    'docs/go2_rgbd_observation_development_v1_result_2026-09-05.md')
LEAVES = (*RGB_LEAVES, 'fast_gyro_samples.npz', 'fast_gyro_histories.npz',
          'depth_observations.json', 'depth_camera_audit.json', 'floor_visual_collision_identity.json')
NATIVE_ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis')
NATIVE_SHA256 = {
    'vis/camera.py': 'b28a54aa51593eb0dda8e3b4f8cf828078b3eed256d259eeb54de44f48030c99',
    'vis/rasterizer.py': 'bdedad9b135c43a232916b7a1c978a63a2a1b4553ee6b39d0bffbb9d08ce2327',
    'ext/pyrender/camera.py': '681ba95116285f031e5193f73bce2c0cae2364c2a1334d98e3e2e786fbb84bbe',
    'ext/pyrender/renderer.py': '1dc7c47b17a82c8aad7c2801bbfad7632dcaeeab8ecc8a382ebe06dee21a0b84',
    'ext/pyrender/jit_render.py': '6fd8c293afb9d009507337d8b7caf2dd42bd7af0568a0bfad8008e8e1de22e58',
    'ext/pyrender/offscreen.py': '54d6ad33b0ca7d9fe0219f57e5990a276524dd0c1ff072cd323b86ecbac07a56',
    'utils/mesh.py': '084105dc71a9ae390c6dd794e3ca2ccc1abd5acfffc78eec78a3e26a9fabc0eb',
    'engine/entities/rigid_entity/rigid_entity.py': '05c16043c05acb129f2e54723bc2d619686a1e00cfe2f6419b0e68521c940649',
    'engine/entities/rigid_entity/rigid_geom.py': '501d5ce71ab2249d5971e501f003fac9ecbcb96db5ad893e92ba9ab850655d8a',
    'engine/mesh.py': 'a768a0188b894f8e26af664aa014aacc262fdad9ccb98180f46c967a0464c94a'}

def verify_native():
    result = {}
    for name, sha in NATIVE_SHA256.items():
        path = NATIVE_ROOT/name
        if path.resolve() != path or digest(path) != sha:
            raise ValueError('reviewed installed depth renderer changed')
        result[str(path)] = sha
    return result

def trials():
    rows = []
    for index in (0, 2):
        spec = probe_spec(index)
        spec['geometry'] = {k: spec['geometry'][k] for k in ('spawn_se2_world', 'wall_boxes')}
        spec.update(scene_id='go2-single-sample-rgbd-development-v1-'+spec['marker_case'],
                    family='RGBD_INTERFACE_DEVELOPMENT', procedural_seed=2026100400)
        rows.append(spec)
    return rows

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
    inherited, inputs, gait, _native = prior_preflight()
    identities = {str((PRIOR/name).relative_to(ROOT)): sha for name, sha in (
        ('launch.json', '50d842c6bae1537e6b79874e50235c7eea84740b7c03cb0d361c14898e2459e1'),
        ('result.json', '8063448c31d013a0e0cfc56aa4f87c1e3dfb1256016a355f5a80105c93a5441a'),
        ('raw_artifact_audit.json', 'e52424ec2b58291dfd43b97af5c2f896daf2beae3d447cb9261b98000bb701b4'))}
    verify_bindings(identities)
    launch = json.loads((PRIOR/'launch.json').read_text())
    audit = json.loads((PRIOR/'raw_artifact_audit.json').read_text())
    if inherited != launch['source_sha256'] or audit['status'] != 'PASS' or audit['audited_trials'] != 2 or audit['all_depth_checks_pass'] is not False:
        raise ValueError('completed source-bound whole-task evidence required')
    sources = source_closure(inherited); inputs |= identities
    verify_bindings(sources | inputs | gait)
    return sources, inputs, gait, verify_native()

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
        raise ValueError('fixed absent RGBD interface root required; no retry')
    if shutil.disk_usage(OUTPUT.parent).free < 10*1024**3:
        raise ValueError('less than ten GiB free')
    sources, inputs, gait, native = preflight(); specs = trials(); OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', {'schema': 'single_sample_rgbd_observation_development.v1', 'trial_specs': specs,
        'source_sha256': sources, 'input_sha256': inputs, 'gait_sha256': gait, 'native_renderer_sha256': native,
        'scope': 'two stationary actual-render interface cases; no navigation or hardware qualification'})
    rows = []
    try:
        for spec in specs:
            directory = OUTPUT/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event': 'started', 'scene_id': spec['scene_id']}), flush=True)
            with (directory/'process.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                row = collect(spec, directory)
            cameras = json.loads((directory/'camera_audit.json').read_text())
            row['floor_identity'] = check_floor_identity(json.loads((directory/'floor_visual_collision_identity.json').read_text()))
            row['physical_reference_checks'] = []
            row['depth_checks'] = []
            for i, camera in enumerate(cameras):
                with np.load(directory/f'native_depth_{i:04d}.npz', allow_pickle=False) as archive:
                    row['physical_reference_checks'].append(evaluate_physical_reference(archive['optical_depth_m'], spec['geometry']['wall_boxes'],
                        camera['world_from_optical'], marker_case=spec['marker_case']))
                    row['depth_checks'].append(evaluate_depth(archive['optical_depth_m'], spec['geometry']['wall_boxes'],
                        camera['world_from_optical'], marker_case=spec['marker_case']))
            leaves = [*LEAVES, *[name for i in range(row['rgb_packets']) for name in
                      (f'rgb_{i:04d}.png', f'depth_{i:04d}.npz', f'native_depth_{i:04d}.npz')]]
            row['artifact_sha256'] = {p: digest(directory/p) for p in leaves}
            write_json(directory/'result.json', row); rows.append(row)
            print(json.dumps({'event': 'completed', 'scene_id': row['scene_id'],
                'depth_checks_pass': all(r['passes_declared_depth_check'] for r in row['depth_checks'])}), flush=True)
            if not row['completed_probe']: raise ValueError('incomplete physical probe; preserve failure')
        verify_bindings(sources | inputs | gait)
        if native != verify_native(): raise ValueError('renderer identity drift')
        write_json(OUTPUT/'result.json', {'status': 'COMPLETE', 'trials': rows, 'completed_trials': 2,
                                        'planned_trials': 2, 'launch_sha256': digest(OUTPUT/'launch.json')})
        print(json.dumps({'status': 'COMPLETE', 'trials': 2}), flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error), 'trials': rows,
            'completed_trials': len(rows), 'planned_trials': 2, 'launch_sha256': digest(OUTPUT/'launch.json')})
        raise

if __name__ == '__main__':
    main()
