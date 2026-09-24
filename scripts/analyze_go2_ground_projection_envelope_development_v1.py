#!/usr/bin/env python3
"""New ray-range question on eight preserved routes; no refit or physics retry."""
import ast
import json
import math
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.causal_ground_plane_development import CausalGroundPlane
from lewm.causal_sensor_state import SensorContractError
from lewm.ground_projection_envelope_development import observe_ground_envelope, project_ground_rays, ORIGIN_BODY
from lewm.floor_visibility_reference_development import visible_floor
from lewm.multijunction_routes_development import route_spec, MOTIFS, WIDTHS
from lewm.physical_execution_development import rotation_xyzw
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_ground_projection_envelope_development_v1_attempt_001'
ROUTES = ROOT / '.generated/go2_multijunction_route_development_v1_attempt_001'
GROUND = ROOT / '.generated/go2_ground_plane_development_v1_attempt_001'
FLOOR = ROOT / '.generated/go2_rgb_floor_evidence_development_v1_attempt_001'
FAMILIES = tuple((h, a) for h in (.01, .03, .05) for a in (.025, .05, .10))
NEW_SOURCES = ('lewm/ground_projection_envelope_development.py',
               'lewm/tests/test_ground_projection_envelope_development.py',
               str(Path(__file__).relative_to(ROOT)),
               'lewm/tests/test_ground_projection_envelope_raw_audit.py',
               'docs/go2_ground_projection_envelope_development_v1_2026-09-05.md')


def expanded_source_bindings(inherited):
    """Bind helper/test imports too, including source-only legacy dependencies."""
    available = set(subprocess.run(
        ['rg', '--files', '-g', '*.py', 'lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'],
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
                if module.split('.')[0] not in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'):
                    continue
                for candidate in (module.replace('.', '/') + '.py', module.replace('.', '/') + '/__init__.py'):
                    if candidate in available:
                        pending.append(candidate)
    return inherited | {name: digest(ROOT / name) for name in visited | set(NEW_SOURCES)}


def range_accounting(envelope, truth, actual_height, actual_up, state):
    """Every ray retained in counts; point errors use actual visible floor only."""
    valid = truth['valid']
    visible = valid & truth['visible_floor']
    positive = valid & envelope['bottom_connected_floor_pixels']
    accepted = positive & envelope['interval_valid']
    selected = accepted & visible
    reference = truth['ground_optical_depth_m']
    lower, upper = envelope['lower_optical_depth_m'], envelope['upper_optical_depth_m']
    covered = (reference >= lower - 1e-10) & (reference <= upper + 1e-10)
    normal = np.asarray(state['up_current_body'])
    angle = math.atan2(float(np.linalg.norm(np.cross(normal, actual_up))), float(normal @ actual_up))
    hypothesis_contains_true_plane = (abs(state['body_origin_height_m'] - actual_height) <= envelope['height_radius_m'] + 1e-12
                                      and angle <= envelope['angle_radius_rad'] + 1e-12)
    if hypothesis_contains_true_plane and np.any(selected & ~covered):
        raise ValueError('conditional interval enclosure violated despite true plane inside declared set')
    norm = np.linalg.norm(envelope['rays_body'], axis=-1)
    nominal = visible & envelope['nominal_valid']
    error = np.abs(envelope['nominal_optical_depth_m'][nominal] - reference[nominal]) * norm[nominal]
    width = (upper[selected] - lower[selected]) * norm[selected]
    result = {'pixels': int(valid.size), 'reference_ambiguous': int((~valid).sum()),
              'visible_floor': int(visible.sum()), 'positive_connected_floor': int(positive.sum()),
              'positive_false_surface': int((positive & ~visible).sum()),
              'accepted_visible_floor': int(selected.sum()), 'accepted_false_surface': int((accepted & ~visible).sum()),
              'abstained_positive_visible_floor': int((positive & visible & ~envelope['interval_valid']).sum()),
              'covered_visible_floor': int((selected & covered).sum()),
              'outside_interval_visible_floor': int((selected & ~covered).sum()),
              'true_plane_inside_hypothesis': hypothesis_contains_true_plane,
              'nominal_visible_points': int(error.size), 'nominal_error_sum_m': float(error.sum()),
              'nominal_error_max_m': float(error.max()) if error.size else None,
              'accepted_width_sum_m': float(width.sum()), 'accepted_width_max_m': float(width.max()) if width.size else None,
              'nominal_error_exceeds': {str(t): int((error > t).sum()) for t in (.05, .1, .25, .5)},
              'accepted_width_at_most': {str(t): int((width <= t).sum()) for t in (.1, .25, .5, 1., 2., 5.)},
              'depth_bins': {}}
    for low, high in ((0., .5), (.5, 1.), (1., 2.), (2., 4.), (4., math.inf)):
        candidate = visible & (reference >= low) & (reference < high)
        points = candidate & nominal
        e = np.abs(envelope['nominal_optical_depth_m'][points] - reference[points]) * norm[points]
        result['depth_bins'][f'{low}:{high}'] = {
            'visible_floor': int(candidate.sum()), 'nominal_points': int(points.sum()),
            'error_sum_m': float(e.sum()), 'error_max_m': float(e.max()) if e.size else None,
            'accepted': int((candidate & accepted).sum()), 'covered': int((candidate & accepted & covered).sum())}
    return result


def aggregate(samples):
    result = {'frames': len(samples)}
    sums = ('pixels', 'reference_ambiguous', 'visible_floor', 'positive_connected_floor', 'positive_false_surface',
            'accepted_visible_floor', 'accepted_false_surface', 'abstained_positive_visible_floor',
            'covered_visible_floor', 'outside_interval_visible_floor', 'nominal_visible_points',
            'nominal_error_sum_m', 'accepted_width_sum_m')
    result.update({key: sum(s[key] for s in samples) for key in sums})
    for key in ('nominal_error_max_m', 'accepted_width_max_m'):
        values = [s[key] for s in samples if s[key] is not None]
        result[key] = max(values) if values else None
    result['true_plane_inside_hypothesis_frames'] = sum(s['true_plane_inside_hypothesis'] for s in samples)
    result['nominal_error_mean_m'] = result['nominal_error_sum_m'] / result['nominal_visible_points'] if result['nominal_visible_points'] else None
    result['accepted_width_mean_m'] = result['accepted_width_sum_m'] / result['accepted_visible_floor'] if result['accepted_visible_floor'] else None
    for field in ('nominal_error_exceeds', 'accepted_width_at_most'):
        keys = samples[0][field] if samples else ()
        result[field] = {key: sum(s[field][key] for s in samples) for key in keys}
    result['depth_bins'] = {}
    for key in samples[0]['depth_bins'] if samples else ():
        rows = [s['depth_bins'][key] for s in samples]
        value = {k: sum(r[k] for r in rows) for k in ('visible_floor', 'nominal_points', 'error_sum_m', 'accepted', 'covered')}
        errors = [r['error_max_m'] for r in rows if r['error_max_m'] is not None]
        value['error_max_m'] = max(errors) if errors else None
        value['error_mean_m'] = value['error_sum_m'] / value['nominal_points'] if value['nominal_points'] else None
        result['depth_bins'][key] = value
    return result


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh ray-envelope diagnostic required')
    inputs = {str((GROUND / 'launch.json').relative_to(ROOT)): 'aa47f0d078474668279a63b6a62ec93b5da27c788c5272962d6457de2c57ecc1',
              str((GROUND / 'result.json').relative_to(ROOT)): 'b798b0d9afd7764f37cbcdcafb14400e870138b8e13144d34c80227e0f1a1fa7',
              str((FLOOR / 'launch.json').relative_to(ROOT)): '8e1001d73d8ec141280fc79e7786963c63c8c01bf15010624d61f92fee46974c',
              str((FLOOR / 'result.json').relative_to(ROOT)): 'bf822b9eecbf711a634330dc4fa27ce33910e39f8860e8691d9988f6a2cc1a74'}
    verify_bindings(inputs)
    sources = {}
    for root in (GROUND, FLOOR):
        launch = json.loads((root / 'launch.json').read_text())
        for target, values in ((sources, launch['source_sha256']), (inputs, launch['input_sha256'])):
            for name, sha in values.items():
                if name in target and target[name] != sha:
                    raise ValueError('conflicting predecessor identity')
                target[name] = sha
    sources = expanded_source_bindings(sources)
    verify_bindings(sources | inputs)
    report = json.loads((ROUTES / 'result.json').read_text())
    if report['status'] != 'COMPLETE' or report['completed_trials'] != 8:
        raise ValueError('all eight original route outcomes required')
    specs = {spec['scene_id']: spec for spec in (route_spec(m, w) for m in MOTIFS for w in WIDTHS)}
    if len(report['trials']) != 8 or {m['scene_id'] for m in report['trials']} != set(specs):
        raise ValueError('exact eight source route identities required')
    if json.loads((ROUTES / 'launch.json').read_text())['trial_specs'] != list(specs.values()):
        raise ValueError('source-generated geometry differs from actual launched route specs')
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs,
               'height_radii_m': [.01, .03, .05], 'angle_radii_rad': [.025, .05, .10],
               'frames_expected': 1498, 'stride': 8,
               'scope': 'posthoc-motivated fixed sensitivity grid, no radius selection/calibration/clearance claim'})
    results = []
    try:
        for member in report['trials']:
            directory = ROUTES / member['scene_id']
            spec = specs[member['scene_id']]
            if spec['scene_id'] != member['scene_id']:
                raise ValueError('exact source route geometry required')
            names = ['physics_trace.npz', 'camera_audit.json', 'policy_histories.npz', 'policy_observations.json']
            names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
            verify_bindings({str((directory / name).relative_to(ROOT)): member['artifact_sha256'][name] for name in names})
            with np.load(directory / 'physics_trace.npz', allow_pickle=False) as archive:
                poses = archive['base_pose_world']
            cameras = json.loads((directory / 'camera_audit.json').read_text())
            ground = CausalGroundPlane()
            rows, fault = [], None
            for index, camera in enumerate(cameras):
                packet = load_route_observation(directory, index)
                now = packet['image']['measured_ns']
                try:
                    state = ground.begin(packet, now_ns=now) if index == 0 else ground.step(packet, now_ns=now)
                except SensorContractError as error:
                    fault = {'observation_index': index, 'reason': str(error)}
                    break
                pose = poses[camera['physical_sample_index']]
                rotation = rotation_xyzw(pose[3:])
                transform = np.asarray(camera['world_from_optical'])
                optical_to_body = np.array([[0., 0, 1], [-1., 0, 0], [0., -1., 0]])
                if (not np.allclose(rotation @ optical_to_body, transform[:3, :3], rtol=0, atol=1e-8)
                        or not np.allclose(pose[:3] + rotation @ ORIGIN_BODY, transform[:3, 3], rtol=0, atol=1e-8)):
                    raise ValueError('independent camera/body calibration disagreement')
                truth = visible_floor(transform, spec['geometry']['wall_boxes'])
                envelope = observe_ground_envelope(packet, state, now_ns=now, height_radius=.01, angle_radius=.025)
                if not np.array_equal(envelope['rows'], truth['rows']) or not np.array_equal(envelope['columns'], truth['columns']):
                    raise ValueError('native pixel-grid mismatch')
                families = {}
                for height, angle in FAMILIES:
                    hypothesis = project_ground_rays(state['up_current_body'], state['body_origin_height_m'], envelope['rays_body'],
                                                      height_radius=height, angle_radius=angle)
                    families[f'{height}:{angle}'] = range_accounting(envelope | hypothesis, truth, float(pose[2]), rotation[2], state)
                rows.append({'observation_index': index, 'decision_ns': now, 'families': families})
            results.append({'scene_id': member['scene_id'], 'source_task_success': member['status'] == 'SUCCESS',
                            'frames_expected': len(cameras), 'frames_processed': len(rows), 'fault': fault,
                            'families': {f'{h}:{a}': aggregate([r['families'][f'{h}:{a}'] for r in rows]) for h, a in FAMILIES},
                            'frames': rows})
            print(json.dumps({'event': 'projection_route_checked', 'completed': len(results), 'planned': 8,
                              'frames': len(rows), 'fault': fault}), flush=True)
        if sum(r['frames_expected'] for r in results) != 1498:
            raise ValueError('fixed source frame population changed')
        verify_bindings(sources | inputs)
        write_json(OUTPUT / 'result.json', {'status': 'COMPLETE', 'trials': results,
                   'frames_processed': sum(r['frames_processed'] for r in results),
                   'sensor_unavailable_routes': sum(r['fault'] is not None for r in results),
                   'launch_sha256': digest(OUTPUT / 'launch.json'), 'uncertainty_calibrated': False,
                   'metric_clearance_qualified': False, 'scope': 'descriptive conditional image-ray projection, not footprint safety or maze navigation'})
        print(json.dumps({'status': 'COMPLETE', 'routes': len(results), 'frames': sum(r['frames_processed'] for r in results)}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'FAIL', 'error': repr(error), 'trials': results,
                   'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
