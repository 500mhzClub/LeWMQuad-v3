#!/usr/bin/env python3
"""Audit explicit development artifacts; never discovers or opens held-out data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.local_execution_controller_development import (
    ARMS, LocalController, continuation_geometry, evaluate_edge, motion_window_ok, trial_spec,
)
from lewm.physical_execution_development import KINDS, WIDTHS, rotation_xyzw
from scripts.audit_go2_contact_attributed_execution_development_v1 import check, recompute_contact_flags
from scripts import run_physical_graph_edge_handoff_qualification_v1 as BASE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def crossing_for(arrays, geometry):
    active = arrays['phase'] != 0
    if not active.any():
        return None
    edge = geometry['selected_directed_edge']
    try:
        result = BASE.canonical_port_crossing(arrays['base_pose_world'][active],
            np.zeros(np.count_nonzero(active), dtype=np.uint8), edge['opening_segment_world'],
            edge['opening_normal_world'], [], sustained_samples=100)
        result['edge_id'] = edge['edge_id']
        return result
    except BASE.ExperimentError as exc:
        check(str(exc) in ('teacher trace never crosses the canonical directed port',
            'teacher enters a competing physical port first',
            'teacher did not remain beyond the port for 100 physics samples'), str(exc))
        return None


def audit_decisions(arrays, decisions, spec, edges):
    controllers = {i: LocalController(spec['arm']) for i in range(len(edges))}
    next_ticks = {i: 0 for i in controllers}
    expected_pre = int(np.count_nonzero(arrays['phase'] == 0)) - 1
    for decision in decisions:
        i = decision['edge_index']
        check(i in controllers, 'decision for unreported edge')
        geometry = edges[i]['geometry']
        index = decision['pre_sample_index']
        check(index == expected_pre and index >= 0, 'decision skipped/reused physical samples')
        check(decision['timestamp_s'] == arrays['timestamp_s'][index], 'decision time is not pre-action boundary')
        pose, twist = arrays['base_pose_world'][index], arrays['base_twist_world'][index]
        yaw = BASE._pose_yaw_xyzw(pose)
        route = geometry['teacher_route_polyline_world']
        initial = math.atan2(route[1][1]-route[0][1], route[1][0]-route[0][0])
        normal = geometry['selected_directed_edge']['opening_normal_world']
        desired = math.atan2(normal[1], normal[0])
        lookahead, _, _ = BASE._lookahead_from_port([pose[0], pose[1], yaw], route, .18)
        heading = math.atan2(lookahead[1]-pose[1], lookahead[0]-pose[0])
        observed = (np.arange(len(arrays['phase'])) <= index) & (arrays['edge_index'] == i)
        prefix = {key: value[observed] for key, value in arrays.items()}
        arrival = observed & (arrays['phase'] == 2)
        values = dict(tick=next_ticks[i], alignment_error=BASE._wrap_angle(initial-yaw),
            pursuit_error=BASE._wrap_angle(heading-yaw), arrival_error=BASE._wrap_angle(desired-yaw),
            body_forward_velocity=float(rotation_xyzw(pose[3:])[:,0] @ twist[:3]),
            angular_velocity=float(twist[5]), crossed=crossing_for(prefix, geometry) is not None,
            stable_arrival=motion_window_ok(arrays['base_pose_world'][arrival],
                arrays['base_twist_world'][arrival], geometry, spec['width_m']))
        check(values == decision['inputs'], 'decision input not reproducible from observed prefix')
        requested = controllers[i].decide(**values)
        check(requested == decision['requested_command'] and controllers[i].stage == decision['stage'],
              'controller action/stage mismatch')
        next_ticks[i] += 1
        if requested is None:
            check(index == edges[i]['terminal_global_sample_index'], 'terminal decision not at edge endpoint')
            continue
        count = min(50, len(arrays['phase'])-index-1)
        check(count > 0, 'requested action without execution')
        interval = slice(index+1, index+1+count)
        check(np.all(arrays['edge_index'][interval] == i), 'command spans edge reset')
        check(np.all(arrays['phase'][interval] == (2 if decision['stage'] == 'ARRIVE' else 1)), 'command phase mismatch')
        check(np.array_equal(arrays['requested_command'][interval], np.tile(requested, (count,1))), 'request tape mismatch')
        # First command of next edge is checked against the actual previous edge command.
        previous = arrays['applied_command'][index]
        clipped = np.clip(np.asarray(requested, dtype=np.float32), [-.3,0,-.5], [.3,0,.5]).astype(np.float32)
        applied = previous + np.clip(clipped-previous, [-.25,0,-.35], [.25,0,.35])
        check(np.allclose(arrays['applied_command'][interval], applied, rtol=0, atol=1e-7), 'slew-integrated command mismatch')
        expected_pre = index + count
    if decisions:
        check(expected_pre == len(arrays['phase'])-1, 'unexplained trailing execution')
    for i, controller in controllers.items():
        check(controller.terminal_reason == edges[i]['controller_terminal_reason'], 'controller terminal reason mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.absolute()
    check(not any(p == 'sealed' or p == 'sealed_test.json' or p.startswith('sealed_') for p in output.parts), 'protected path')
    audit_path = output/'raw_artifact_audit.json'
    check(not audit_path.exists(), 'audit already exists')
    launch = json.loads((output/'launch.json').read_text())
    report = json.loads((output/'result.json').read_text())
    specs = [trial_spec(kind,width,arm) for kind in KINDS for width in WIDTHS for arm in ARMS]
    check(launch['trial_specs'] == specs, 'fixed population mismatch')
    check(report['status'] == 'COMPLETE' and report['completed_trials'] == report['planned_trials'] == 32, 'incomplete study')
    check(digest(output/'launch.json') == report['launch_sha256'], 'launch binding mismatch')
    for name, expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        path = Path(name)
        check(not path.is_absolute() and '..' not in path.parts and not any(
            p == 'sealed' or p == 'sealed_test.json' or p.startswith('sealed_') for p in path.parts), 'invalid source path')
        check(digest(ROOT/path) == expected, f'source/gait binding changed: {name}')
    rows, prefixes = [], {}
    for spec, supplied in zip(specs, report['trials'], strict=True):
        directory = output/spec['scene_id']
        check(supplied['scene_id'] == spec['scene_id'] and supplied['arm'] == spec['arm'], 'trial identity')
        check(json.loads((directory/'result.json').read_text()) == supplied, 'trial/report mismatch')
        required = {'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','decisions.json','process.log','final_rgb.png'}
        allowed = required | {'initial_rgb.png','edge0_rgb.png','edge1_rgb.png'}
        check(required <= set(supplied['artifact_sha256']) <= allowed, 'artifact set mismatch')
        for leaf, expected in supplied['artifact_sha256'].items():
            check(digest(directory/leaf) == expected, f'artifact binding changed: {leaf}')
        arrays = read_npz(directory/'physics_trace.npz')
        times = arrays['timestamp_s']
        check(len(times) == supplied['physics_samples'] and len(times) > 0, 'sample count')
        check(np.allclose(times, .002*np.arange(1,len(times)+1), rtol=0, atol=1e-10), 'global physics clock reset or gap')
        check(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite raw state')
        flags, first = recompute_contact_flags(read_npz(directory/'native_contacts.npz'),
            json.loads((directory/'contact_topology.json').read_text()), times)
        check(np.array_equal(flags, arrays['physics_contact'].astype(bool)), 'native force reference disagrees')
        if first is not None:
            check(first['sample_index'] == len(times)-1 and supplied['edges'][-1]['stop_reason'] == 'DISALLOWED_CONTACT', 'contact stop delay')
            check(supplied['first_disallowed_contact']['sample_index'] == first['sample_index'], 'reported contact index')
        commands = arrays['applied_command']
        check(np.all(np.abs(commands) <= np.array([.3,0,.5])+1e-7), 'command limits')
        check(np.all(np.abs(np.diff(commands,axis=0)) <= np.array([.25,0,.35])+1e-7), 'command slew')
        check(np.all(np.diff(arrays['edge_index'].astype(int)) >= 0), 'edge order')
        settle = arrays['phase'] == 0
        check(np.count_nonzero(settle) == 750 and np.all(settle[:750]) and not settle[750:].any(), 'settling/reset contract')
        prefix = {key: value[:750] for key,value in arrays.items()}
        if spec['arm'] == 'baseline':
            prefixes[spec['case_index']] = prefix
        else:
            check(all(np.array_equal(value, prefixes[spec['case_index']][key]) for key,value in prefix.items()), 'paired initial state differs')
        check(len(supplied['edges']) == (1 if supplied['edges'][0]['stop_reason'] else 2), 'continuation skipped after proxy failure')
        for i, edge in enumerate(supplied['edges']):
            geometry = spec['geometry'] if i == 0 else continuation_geometry(spec['geometry'])
            check(edge['geometry'] == geometry, 'edge geometry changed')
            mask = arrays['edge_index'] == i
            sub = {key:value[mask] for key,value in arrays.items()}
            check(set(sub['phase']) <= {0,1,2} and np.all(np.diff(sub['phase'].astype(int)) >= 0), 'edge phase order')
            check(edge['terminal_global_sample_index'] == int(np.flatnonzero(mask)[-1]), 'edge terminal boundary')
            reduced = evaluate_edge(spec | {'geometry':geometry}, sub, stop_reason=edge['stop_reason'], crossing=crossing_for(sub,geometry))
            check(all(edge[key] == value for key,value in reduced.items()), 'raw endpoint reduction differs')
        decisions = json.loads((directory/'decisions.json').read_text())
        audit_decisions(arrays, decisions, spec, supplied['edges'])
        from PIL import Image
        for name, metadata in supplied['images'].items():
            check(name in ('initial','edge0','edge1','final'), 'unknown image')
            pixels = np.asarray(Image.open(directory/f'{name}_rgb.png'))
            check(pixels.shape == (480,640,3) and pixels.dtype == np.uint8, 'RGB contract')
            check(hashlib.sha256(pixels.tobytes()).hexdigest() == metadata['rgb_sha256'], 'image pixel binding')
            transform = np.asarray(metadata['world_from_optical'])
            check(transform.shape == (4,4) and np.allclose(transform[3],[0,0,0,1]), 'optical homogeneous frame')
            check(np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-10, rtol=0)
                and abs(np.linalg.det(transform[:3,:3])-1) < 1e-10, 'improper optical frame')
            check(np.any(np.isclose(times, metadata['timestamp_s'], rtol=0, atol=1e-12)), 'image outside trace')
        crossings = len(supplied['edges']) == 2 and all(e['checks']['sustained_correct_crossing'] and e['checks']['no_disallowed_contact'] for e in supplied['edges'])
        success = crossings and supplied['edges'][-1]['status'] == 'SUCCESS'
        check(supplied['two_contact_free_crossings'] == crossings and (supplied['status'] == 'SUCCESS') == success, 'task reduction mismatch')
        rows.append({'scene_id':spec['scene_id'],'arm':spec['arm'],'task_success':success,
            'first_arrival_success':supplied['edges'][0]['status']=='SUCCESS', 'two_crossings':crossings,
            'first_native_disallowed_contact':first,'decisions_audited':len(decisions)})
    for arm, summary in report['by_arm'].items():
        selected = [r for r in report['trials'] if r['arm'] == arm]
        check(summary == {'trials':8,'task_successes':sum(r['status']=='SUCCESS' for r in selected),
            'two_crossings':sum(r['two_contact_free_crossings'] for r in selected),
            'two_usable_arrivals':sum(r['two_usable_arrivals'] for r in selected)}, 'arm totals mismatch')
    audit = {'status':'PASS','audited_trials':32,'exact_paired_settling_prefixes':True,'trials':rows,
        'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'scope':'raw development evidence audit; not a physical replication or novel-maze evaluation'}
    with audit_path.open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in audit.items() if key!='trials'},indent=2))


if __name__ == '__main__':
    main()
