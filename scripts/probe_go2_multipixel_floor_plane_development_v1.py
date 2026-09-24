"""Fixed float32 finite-amplitude plane diagnostic; no estimator/control changes."""
from dataclasses import asdict
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.correlated_floor_evidence_development import _sample
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.multipixel_floor_plane_development import PatchRules, fit_multipixel_plane, query_multipixel_floor
from lewm.paired_rgbd_physical_plane_development import minimum_gaps
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import OUTPUT as PREVIOUS, INPUT, plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_multipixel_floor_plane_development_v1_attempt_001'
PROTOCOL = 'docs/go2_multipixel_floor_plane_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_multipixel_floor_plane_development_v1.py',
         'lewm/tests/test_multipixel_floor_plane_development.py')
IDENTITIES = {
    'launch.json': '5361a89c73762d789cc431c3b4a35c24c440916bb1af17bf9720c22dde1d808a',
    'result.json': '12cd0cba81d4600f5777bbc811500cc80172f8e475a31343f35c50dbf35f6fe6',
    'plane_quantization_diagnostic.json': 'fb9e54331a4f3e548c0cb50f9c94d7840f94e692ed05dfd92b9bcbd77f03ef01'}
AMPLITUDES = (.005, .01, 1.)


def preflight():
    bound = {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bound)
    old = read_json(PREVIOUS, 'launch.json'); verify(old)
    result = read_json(PREVIOUS, 'result.json')
    bound |= {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    diagnosis = read_json(PREVIOUS, 'plane_quantization_diagnostic.json')
    bound |= diagnosis['source_sha256']
    launch = old | dict(source_sha256=discover_sources(SEEDS, old['source_sha256']),
        input_sha256=old['input_sha256'] | bound, diagnostic_protocol=PROTOCOL,
        fixed_patch_rules=asdict(PatchRules()), frame_indices=[0, 218],
        finite_amplitudes=list(AMPLITUDES), noise_seed_base=2026090620,
        scope='finite represented-depth plane-only diagnostic; no calibration, new physics or action permission')
    verify(launch)
    return launch


def members(depth, index):
    d = depth['depth_m']; valid = depth['valid']
    if d.dtype != np.float32: raise ValueError('actual float32 sensor representation required')
    yield 'nominal', d.copy()
    for name, loading in (('gain', (.001*d).astype(float)), ('offset', .001*valid)):
        for amp in AMPLITUDES:
            for sign in (1, -1):
                yield f'{name}_{amp:g}_{sign:+d}', (d.astype(float)+sign*amp*loading).astype(np.float32)
    pattern = np.random.default_rng(2026090620+index).uniform(-1., 1., d.shape)*valid
    for name, size in (('ulp', np.spacing(d)), ('100um', .0001)):
        for sign in (1, -1):
            yield f'independent_{name}_{sign:+d}', (d.astype(float)+sign*size*pattern).astype(np.float32)


def compare(rows):
    by_name = {r['member']: r for r in rows}
    result = {}
    for estimator in ('multipixel', 'three_point'):
        nominal = by_name['nominal'].get(estimator+'_gap_m')
        source_rows = {}
        for source in ('gain', 'offset'):
            factors = {}; changes = {}
            for amp in AMPLITUDES:
                plus, minus = [by_name[f'{source}_{amp:g}_{s:+d}'].get(estimator+'_gap_m') for s in (1, -1)]
                if plus is None or minus is None:
                    factors[f'{amp:g}'] = None; changes[f'{amp:g}'] = None
                    continue
                factors[f'{amp:g}'] = (np.asarray(plus)-minus)/(2*amp)
                changes[f'{amp:g}'] = (np.max(np.abs(np.asarray([plus, minus])-nominal), axis=0)
                                         if nominal is not None else None)
            small, large = factors['0.005'], factors['0.01']
            source_rows[source] = dict(central_factors_m=factors, maximum_signed_gap_changes_m=changes,
                factor_step_difference_m=np.abs(small-large) if small is not None and large is not None else None)
        result[estimator] = source_rows
    return plain(result)


def run_frame(index, up, R, p, geometry, joints):
    _, depth = load_rgbd_observation(INPUT/'mission', index)
    rows = []; start = time.perf_counter()
    ids = [s['shape_id'] for s in geometry.supports(joints, np.eye(3))['shapes']]
    nominal_seed = None
    for label, ranges in members(depth, index):
        row = dict(member=label)
        if np.any((ranges[depth['valid']] < .2) | (ranges[depth['valid']] > 5.)):
            rows.append(row | dict(status='REPRESENTED_RANGE_VALIDITY_CROSSING')); continue
        frame = PreparedFloorFrame(ranges, depth['valid'], up)
        plane = fit_multipixel_plane(frame)
        if label == 'nominal': nominal_seed = plane.seed_cell_rc
        row |= dict(status=plane.status, plane=asdict(plane), seed_changed=plane.seed_cell_rc != nominal_seed)
        seed = MeasuredPlaneHypothesis.from_frame(frame).cell_for(frame)
        if seed is not None:
            points, _ = _sample(ranges, depth['valid'], np.asarray(seed)[None])
            a = points[0, 0]; n = np.cross(points[0, 1]-a, points[0, 2]-a); n /= np.linalg.norm(n)
            if n@up < 0: n = -n
            row['three_point_gap_m'] = minimum_gaps(geometry, joints, a, n, R, p)
        if plane.status == 'MEASURED_PATCH_CONSISTENT':
            a, n = plane.for_frame(frame)
            row['multipixel_gap_m'] = minimum_gaps(geometry, joints, a, n, R, p)
            row['physical_query'] = query_multipixel_floor(frame, plane, geometry, joints,
                rotation_observation_from_body=R, translation_observation_from_body=p,
                normal_error=.002, up_error=.001, plane_offset_error=.001,
                point_error_by_shape=dict.fromkeys(ids, 0.), floor_backend='cached')
        rows.append(plain(row))
    assert len(rows) == 17
    return dict(frame_index=index, shape_ids=ids, members=rows, comparisons=compare(rows),
        elapsed_wall_s=time.perf_counter()-start, relative_pose_held_fixed=True,
        sensor_error_population_validated=False, independent_experimental_trials=0,
        navigation_action_permitted=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh exclusive diagnostic only')
    launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    completed = []
    try:
        initial, _ = load_rgbd_observation(INPUT/'mission', 0)
        terminal, _ = load_rgbd_observation(INPUT/'mission', 218)
        mean = initial['sensor_state']['sensed']['specific_force']['values'].mean(axis=0)
        up = mean/np.linalg.norm(mean)
        end = read_json(INPUT/'mission', 'task_decisions.json')[-1]['controller']
        R = np.asarray(end['global_orientation']['rotation_initial_body_from_current_body'])
        p = np.asarray(end['sensor_fusion']['position_initial_body_m'])
        q = terminal['sensor_state']['sensed']['joints']['values'][-1, :12]
        geometry = ArticulatedCollisionGeometry(URDF)
        for index, direction, rotation, translation in ((0, up, R, p), (218, R.T@up, np.eye(3), np.zeros(3))):
            result = run_frame(index, direction, rotation, translation, geometry, q)
            name = f'frame_{index:03d}.json'; write_json(OUTPUT/name, plain(result)); completed.append(name)
            print(index, {s: sum(r['status']==s for r in result['members']) for s in sorted({r['status'] for r in result['members']})}, flush=True)
        verify(launch)
        result = dict(status='FINITE_FLOAT32_MULTIPIXEL_PLANE_DIAGNOSTIC_COMPLETE',
            artifact_sha256={n: digest(OUTPUT/n) for n in completed},
            independent_experimental_trials=0, physics_executed=False, calibrated_error_bounds=False,
            navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(result['status'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MULTIPIXEL_DIAGNOSTIC_FAILURE', reason=str(error),
            completed_artifacts={n: digest(OUTPUT/n) for n in completed}))
        raise


if __name__ == '__main__': main()
