"""One frozen conditional-surface comparison, preserving old point-error scales."""
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_multipixel_floor_plane_development_v1 import OUTPUT as PREVIOUS, INPUT, members
from scripts.probe_go2_rgbd_physical_configuration_evidence_development_v1 import OUTPUT as CONFIGURATION
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_bounded_depth_surface_development_v1_attempt_001'
PROTOCOL = 'docs/go2_bounded_depth_surface_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_bounded_depth_surface_development_v1.py',
         'lewm/tests/test_bounded_depth_surface_development.py')
IDENTITIES = {'launch.json': 'c2099caa658c4843e8d5d8f63100dc62eccc5668bfe869eeac646a6f079bcfa6',
    'result.json': '87ba058d9e989ee22d23f98e9c910b84dc4b54a8f6e3ffc6515d46c640a3e236',
    'reference_audit.json': '7754f017d47785ebfac1214cde0510e0fa48c426acd5654c8b18770fc2c78012'}
HYPOTHESES = dict(range_error_m=.0001, surface_tube_m=.001, up_error=.001)
SELECTED = ('nominal', 'independent_100um_+1', 'independent_100um_-1')


def preflight():
    bound = {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bound)
    old = read_json(PREVIOUS, 'launch.json'); verify(old)
    result = read_json(PREVIOUS, 'result.json'); audit = read_json(PREVIOUS, 'reference_audit.json')
    verify_bindings(audit['source_sha256'] | audit['input_sha256'])
    bound |= {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    configuration = str((CONFIGURATION/'configuration_00.json').relative_to(ROOT))
    if configuration not in old['input_sha256']: raise ValueError('original point-error witness not bound')
    sources = discover_sources(SEEDS, old['source_sha256'])
    inputs = old['input_sha256'] | bound | audit['source_sha256']
    launch = old | dict(source_sha256=sources, input_sha256=inputs,
        diagnostic_protocol=PROTOCOL, bounded_surface_hypotheses=HYPOTHESES,
        selected_members=list(SELECTED), scope='conditional bounded measured surface; no calibration, new physics or permission')
    verify(launch); return launch


def original_errors(configuration, ns, depth_sha):
    errors = {}
    for primitive in configuration['primitives']:
        witnesses = [w for w in primitive['ground_witnesses'] if w['measured_ns'] == ns]
        if len(witnesses) != 1 or witnesses[0]['depth_sha256'] != depth_sha:
            raise ValueError('unique matching original per-view physical witness required')
        gap = witnesses[0]['gap']; sid = primitive['shape_id']
        if gap['shape_id'] != sid: raise ValueError('exact original primitive identity required')
        errors[sid] = gap['point_error_m']
    return errors


def summarize(query):
    covered = query['floor_coverage']; rows = query['gap_bounds']['primitives']
    return dict(covered=sum(covered.values()),
        covered_separated=sum(covered[r['shape_id']] and r['minimum_gap_lower_m'] > 0 for r in rows),
        covered_ambiguous=[r['shape_id'] for r in rows if covered[r['shape_id']]
                            and r['minimum_gap_lower_m'] <= 0 <= r['minimum_gap_upper_m']],
        covered_penetrated=[r['shape_id'] for r in rows if covered[r['shape_id']] and r['minimum_gap_upper_m'] < 0])


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh exclusive diagnostic only')
    launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    completed = []
    try:
        configuration = read_json(CONFIGURATION, 'configuration_00.json')['configuration']
        first, _ = load_rgbd_observation(INPUT/'mission', 0)
        terminal, _ = load_rgbd_observation(INPUT/'mission', 218)
        mean = first['sensor_state']['sensed']['specific_force']['values'].mean(axis=0); up = mean/np.linalg.norm(mean)
        end = read_json(INPUT/'mission', 'task_decisions.json')[-1]['controller']
        R = np.asarray(end['global_orientation']['rotation_initial_body_from_current_body'])
        p = np.asarray(end['sensor_fusion']['position_initial_body_m'])
        q = terminal['sensor_state']['sensed']['joints']['values'][-1, :12]
        geometry = ArticulatedCollisionGeometry(URDF)
        rows = []; start = time.perf_counter()
        for index, direction, rotation, translation in ((0, up, R, p), (218, R.T@up, np.eye(3), np.zeros(3))):
            policy, depth = load_rgbd_observation(INPUT/'mission', index)
            old = read_json(PREVIOUS, f'frame_{index:03d}.json')
            by_name = {r['member']: r for r in old['members']}
            errors = original_errors(configuration, policy['sensor_state']['decision_ns'],
                                      by_name['nominal']['plane']['depth_sha256'])
            for member, ranges in members(depth, index):
                if member not in SELECTED: continue
                surface = BoundedDepthSurface(ranges, depth['valid'], direction, **HYPOTHESES)
                if surface.frame.depth_sha256 != by_name[member]['plane']['depth_sha256']:
                    raise ValueError('recorded represented input changed')
                row = dict(frame_index=index, member=member, previous_status=by_name[member]['status'],
                    status=surface.status, seed_cell_rc=surface.seed, diagnostics=surface.diagnostics,
                    source_depth_sha256=surface.frame.depth_sha256, queries={})
                if surface.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE':
                    row |= dict(reference_anchor_m=surface.anchor, reference_normal=surface.normal)
                    for name, point_errors in (('surface_isolation', dict.fromkeys(errors, 0.)),
                                                ('original_point_errors', errors)):
                        kwargs = dict(rotation_observation_from_body=rotation,
                            translation_observation_from_body=translation, point_error_by_shape=point_errors)
                        query = surface.query(geometry, q, **kwargs)
                        exact(plain(query), plain(surface.query(geometry, q, **kwargs, backend='reference')))
                        row['queries'][name] = dict(result=plain(query), summary=summarize(query))
                row = plain(row); rows.append(row)
                print(index, member, row['status'], {k: v['summary'] for k, v in row['queries'].items()}, flush=True)
        assert len(rows) == 6
        write_json(OUTPUT/'surface_queries.json', rows); completed.append('surface_queries.json')
        verify(launch)
        result = dict(status='BOUNDED_DEPTH_SURFACE_DIAGNOSTIC_COMPLETE',
            artifact_sha256={n: digest(OUTPUT/n) for n in completed},
            elapsed_wall_s=time.perf_counter()-start, members=len(rows),
            accepted_surfaces=sum(r['status'] == 'BOUNDED_MEASURED_SURFACE_AVAILABLE' for r in rows),
            independent_experimental_trials=0, physics_executed=False, calibrated_error_bounds=False,
            navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(result['status'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_BOUNDED_SURFACE_DIAGNOSTIC_FAILURE', reason=str(error),
            completed_artifacts={n: digest(OUTPUT/n) for n in completed}))
        raise


if __name__ == '__main__': main()
