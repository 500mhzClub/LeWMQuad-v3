#!/usr/bin/env python3
"""All recorded scan postures and native contacts; no physics or controller rerun."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_collision_grouping_development import resolve_native_groups
from lewm.causal_ground_plane_development import foot_sphere_centres_body
from lewm.physical_execution_development import rotation_xyzw
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_articulated_scan_geometry_development_v2_attempt_001'
POPULATIONS = (
    ('active', '.generated/go2_active_exit_scan_development_v1_attempt_001', 16,
     '2c484b07ee94d1366c6c744aa725a6b5da503ca1558f84b9339b1b60bd85a70d',
     'c77c12ac1a690fe984840b342e0dbf5b6b57bfd84320efbbc120f0dba65af46e',
     '6f4cd54ed3bfc2189ff468f39fae2503817b5c20750c8eb03e6e1d2549e33462'),
    ('fast', '.generated/go2_fast_gyro_scan_development_v1_attempt_001', 8,
     'f0efc9cef7a4d06f3207394fc4851f997ee379ddb18f1fa0a133b35f25c1f3ec',
     'd298f8ad1dc4c4098f7ae45a6e2025b361f356503b29dc0e45252c215eace3de',
     '9001382cfb134436b2563cdbffedecf82349ac6e462076e616e844808bcf8289'))
NEW_SOURCES = ('lewm/articulated_collision_geometry_development.py',
               'lewm/tests/test_articulated_collision_geometry_development.py',
               'scripts/analyze_go2_articulated_scan_geometry_development_v1.py',
               'lewm/tests/test_articulated_scan_geometry_accounting_development.py',
               'docs/go2_articulated_scan_geometry_development_v1_2026-09-05.md',
               'lewm/native_collision_grouping_development.py', 'lewm/tests/test_native_collision_grouping_development.py',
               'scripts/analyze_go2_articulated_scan_geometry_development_v2.py',
               'lewm/tests/test_articulated_scan_geometry_v2_accounting_development.py',
               'docs/go2_articulated_scan_geometry_development_v2_2026-09-05.md')


def contact_geometry(spec, pose, joint_position, contact, model, native_groups):
    """EVALUATION ONLY: native wall-contact plane versus instantaneous support.

    An infinite-plane overlap is not a finite-wall intersection test and is not
    an independent simulation of native collision forces.
    """
    walls = {w['wall_id']: w for w in spec['geometry']['wall_boxes']}
    if contact['environment_object_id'] not in walls:
        raise ValueError('wall-specific diagnostic requires identified wall contact')
    wall = walls[contact['environment_object_id']]
    axis = int(np.argmin(wall['size_xyz'][:2]))
    yaw = wall['yaw_rad']
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.]) if axis == 0 else np.array([-np.sin(yaw), np.cos(yaw), 0.])
    base, center = np.asarray(pose[:3]), np.asarray(wall['centre_xyz'])
    if (center - base) @ normal < 0:
        normal = -normal
    plane = float(center @ normal - wall['size_xyz'][axis] / 2)
    rotation = rotation_xyzw(pose[3:])
    support = model.supports(joint_position, (rotation.T @ normal)[None])
    shapes = support['shapes']
    group = [r for r in shapes if native_groups[r['shape_id']] == contact['robot_link_name']]
    if not group:
        raise ValueError('native contact link has no resolved nominal rigid collision group')
    limiting = max(shapes, key=lambda r: r['upper'][0])
    group_limiting = max(group, key=lambda r: r['upper'][0])
    torso = next(r for r in shapes if r['shape_id'] == 'base:0')
    offset = plane - float(base @ normal)
    return {'native_robot_link': contact['robot_link_name'], 'wall_id': contact['environment_object_id'],
            'wall_normal_world': normal.tolist(), 'wall_inner_plane_coordinate_m': plane,
            'native_contact_plane_residual_m': float(np.asarray(contact['position_world_m']) @ normal - plane),
            'torso_only_plane_gap_m': offset - torso['upper'][0],
            'whole_robot_plane_gap_m': offset - support['upper'][0],
            'native_rigid_group_plane_gap_m': offset - group_limiting['upper'][0],
            'whole_robot_limiting_shape': limiting['shape_id'], 'native_group_limiting_shape': group_limiting['shape_id'],
            'native_group_shape_ids': [r['shape_id'] for r in group],
            'future_swept_volume_qualified': False, 'runtime_environment_input': False}


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh articulated diagnostic required')
    sources, inputs, work = {}, {}, []
    failed_root = ROOT / '.generated/go2_articulated_scan_geometry_development_v1_attempt_001'
    failure_bindings = {str((failed_root / 'launch.json').relative_to(ROOT)): 'a104c36315d341d4d92d37f50dd9e497d0337d5bced068cb249e3f6a83a595b0',
                        str((failed_root / 'result.json').relative_to(ROOT)): '30f1e766bdc5dcbe76a814857ce91b6904617251c63c740e27f9b97dd8c72c58'}
    verify_bindings(failure_bindings)
    failed = json.loads((failed_root / 'result.json').read_text())
    failed_launch = json.loads((failed_root / 'launch.json').read_text())
    verify_bindings(failed_launch['source_sha256'] | failed_launch['input_sha256'])
    if failed['status'] != 'FAIL' or failed['trials']:
        raise ValueError('preserved empty topology-resolution failure required')
    inputs.update(failure_bindings)
    for group, name, count, launch_sha, result_sha, audit_sha in POPULATIONS:
        path = ROOT / name
        bindings = {str((path / leaf).relative_to(ROOT)): sha for leaf, sha in
                    (('launch.json', launch_sha), ('result.json', result_sha), ('raw_artifact_audit.json', audit_sha))}
        verify_bindings(bindings)
        inputs.update(bindings)
        launch, result, audit = [json.loads((path / leaf).read_text()) for leaf in ('launch.json', 'result.json', 'raw_artifact_audit.json')]
        if (result['status'] != 'COMPLETE' or result['completed_trials'] != count or len(result['trials']) != count
                or audit['status'] != 'PASS' or audit['audited_trials'] != count):
            raise ValueError('complete raw-audited scan population required')
        for destination, mapping in ((sources, launch['source_sha256']), (inputs, launch['input_sha256']),
                                     (inputs, launch.get('gait_sha256', {}))):
            for key, sha in mapping.items():
                if key in destination and destination[key] != sha:
                    raise ValueError('conflicting inherited identity')
                destination[key] = sha
        specs = {s['scene_id']: s for s in launch['trial_specs']}
        for member in result['trials']:
            directory = path / member['scene_id']
            names = ['physics_trace.npz', 'contact_events.json', 'contact_topology.json', 'scan_decisions.json',
                     'policy_histories.npz', 'policy_observations.json']
            names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
            inputs.update({str((directory / leaf).relative_to(ROOT)): member['artifact_sha256'][leaf] for leaf in names})
            work.append((group, directory, specs[member['scene_id']], member))
    sources.update({p: digest(ROOT / p) for p in NEW_SOURCES})
    verify_bindings(sources | inputs)
    model = ArticulatedCollisionGeometry(URDF)
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs,
               'planned_trials': 24, 'expected_control_frames': 4504,
               'scope': 'reused actual joint postures and evaluation-only contact planes; no clearance or future sweep qualification'})
    rows = []
    try:
        for group, directory, spec, member in work:
            with np.load(directory / 'physics_trace.npz', allow_pickle=False) as archive:
                poses, joints = archive['base_pose_world'], archive['joint_position']
            decisions = json.loads((directory / 'scan_decisions.json').read_text())
            topology = json.loads((directory / 'contact_topology.json').read_text())
            native_names = {topology['link_names'][str(i)] for i in topology['robot_link_ids']}
            native_groups = resolve_native_groups(URDF, model.supports(np.zeros(12), np.eye(3))['shapes'], native_names)
            frames, maximum_fk = [], 0.
            for decision in decisions:
                packet = load_route_observation(directory, decision['observation_index'])
                state = model.observe(packet, now_ns=decision['decision_ns'])
                if any(native_groups[s['shape_id']] not in native_names for s in state['shapes']):
                    raise ValueError('nominal rigid group absent from native topology')
                q = packet['sensor_state']['sensed']['joints']['values'][-1, :12]
                feet = np.array([next(s['center_body_m'] for s in state['shapes'] if s['shape_id'] == leg + '_foot:0')
                                 for leg in ('FL', 'FR', 'RL', 'RR')])
                error = float(np.abs(feet - foot_sphere_centres_body(q)).max())
                maximum_fk = max(maximum_fk, error)
                if error > 1e-12:
                    raise ValueError('independent closed-form foot kinematics disagreement')
                frames.append({'observation_index': decision['observation_index'], 'decision_ns': decision['decision_ns'],
                               'lower_body_xyz_m': state['lower'], 'upper_body_xyz_m': state['upper'],
                               'torso_lower_body_xyz_m': next(s['lower'] for s in state['shapes'] if s['shape_id'] == 'base:0'),
                               'torso_upper_body_xyz_m': next(s['upper'] for s in state['shapes'] if s['shape_id'] == 'base:0')})
            events = json.loads((directory / 'contact_events.json').read_text())
            contacts = []
            for event in events:
                for contact in event['disallowed_contacts']:
                    i = event['sample_index']
                    contacts.append({'sample_index': i, 'timestamp_s': event['timestamp_s'],
                                     **contact_geometry(spec, poses[i], joints[i], contact, model, native_groups)})
            rows.append({'group': group, 'scene_id': member['scene_id'], 'frames': frames,
                         'maximum_independent_foot_fk_error_m': maximum_fk, 'contacts': contacts,
                         'native_shape_groups': native_groups,
                         'source_contact': not member['response']['checks']['no_contact']})
            print(json.dumps({'event': 'articulated_scan_analyzed', 'completed': len(rows), 'total': 24,
                              'frames': len(frames), 'native_contact_rows': len(contacts)}), flush=True)
        if sum(len(r['frames']) for r in rows) != 4504 or sum(r['source_contact'] for r in rows) != 6:
            raise ValueError('fixed scan frame/contact-trial population changed')
        verify_bindings(sources | inputs)
        write_json(OUTPUT / 'result.json', {'status': 'COMPLETE', 'trials': rows, 'control_frames': 4504,
                   'contact_trials': 6, 'launch_sha256': digest(OUTPUT / 'launch.json'),
                   'scope': 'instantaneous sensor-derived body extent and conditional contact diagnosis; not a safe-turn classifier'})
        print(json.dumps({'status': 'COMPLETE', 'trials': 24, 'frames': 4504}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
                   'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
