"""Read-only recorded Go2 plane-memory/starting-view diagnostic; no physics."""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneObstacleMemory
from lewm.primitive_obstacle_memory_development import PrimitiveObstacleMemory
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_primitive_beam_kernel_development import exact
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


DIRECTORY = ROOT / '.generated/go2_bounded_floor_robot_interface_development_v1_attempt_001'
IDENTITIES = {
    'launch.json': 'fad3666196fa4e3b6eb04f0307e5f6a8066282ffb6215841095cf3a32f4b35a2',
    'result.json': '3e9c84ddd1760789b489ae5aff194d121337f1d8cc1594ff6c1e48968cfefbd6',
    'native_box_reader_correction.json': 'a0b88df8d706e6e037536c519d8508d89c05c630f8a7bd7da589c1c8909594f0',
}
SOURCES = (
    'scripts/probe_go2_measured_plane_obstacle_memory_development.py',
    'lewm/measured_plane_obstacle_memory_development.py',
    'lewm/tests/test_measured_plane_obstacle_memory_development.py',
    'scripts/probe_go2_primitive_beam_kernel_development.py',
    'lewm/primitive_obstacle_memory_development.py',
    'lewm/primitive_floor_observation_development.py',
    'lewm/primitive_floor_relation_development.py',
    'lewm/primitive_beam_kernel_development.py',
    'lewm/floor_footprint_bounds_development.py',
    'lewm/coupled_floor_enclosure_development.py',
    'lewm/correlated_floor_evidence_development.py',
    'lewm/uncertain_ray_memory_development.py',
)


def main():
    bindings = {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((DIRECTORY / 'launch.json').read_text())
    result = json.loads((DIRECTORY / 'result.json').read_text())
    reader = json.loads((DIRECTORY / 'native_box_reader_correction.json').read_text())
    assert reader['result']['interface_check_pass'] and result['rgbd_frames'] == 26
    bindings |= launch['source_sha256'] | launch['input_sha256'] | reader['reader_source_sha256']
    bindings |= {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    development = {p: digest(ROOT / p) for p in SOURCES}
    verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'development_source_sha256': development, 'source_scope': 'explicit diagnostic inputs, not a recursive source-closure claim'}), flush=True)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    assert len(records) == 26
    geometry = ArticulatedCollisionGeometry(URDF)
    kwargs = dict(normal_error=.002, up_error=.001, plane_offset_error=.001,
                  range_error_m=.001, beam_backend='compiled')
    model, predecessor = MeasuredPlaneObstacleMemory(geometry, **kwargs), PrimitiveObstacleMemory(geometry, **kwargs)
    rows = []
    for tick in range(26):
        p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
        start = time.perf_counter(); observation = model.observe(p, d, records[tick]['observer'], now_ns=now)
        observe_ms = 1000 * (time.perf_counter() - start)
        predecessor.observe(p, d, records[tick]['observer'], now_ns=now)
        start = time.perf_counter(); actual = model.query_current_primitives(now_ns=now)
        query_ms = 1000 * (time.perf_counter() - start)
        reference = model.query_current_primitives(now_ns=now, beam_backend='reference')
        exact(actual, reference)
        old = predecessor.query_current_primitives(now_ns=now)
        foot_centres = np.array([s['center_body_m'] for s in geometry.supports(model._joints, np.eye(3))['shapes']
                                 if s['shape_id'] in FOOT_SHAPES])
        optical_depths = []
        frames = list(model._rays.frames)
        if frames[-1]['measured_ns'] != now: frames.append(model._rays.latest_frame)
        for frame in frames:
            R = frame['rotation'].T @ model._rays.rotation
            t = (model._rays.position - frame['position']) @ frame['rotation']
            optical = (foot_centres @ R.T + t - np.asarray(BODY_FROM_OPTICAL)[:3, 3]) @ np.asarray(BODY_FROM_OPTICAL)[:3, :3]
            optical_depths.extend(optical[:, 2].tolist())
        ids = np.array(actual['shape_ids'])
        row = dict(tick=tick, measured_ns=now, hypothesis_cell_rc=observation['hypothesis_cell_rc'],
            views=len(actual['views']), seeded_views=sum(v['seed_observed'] for v in actual['views']),
            predecessor_seeded_views=sum(v['seed_observed'] for v in old['views']),
            floor_covered_shape_view_pairs=sum(v['floor_covered_primitives'] for v in actual['views']),
            conditional_clearance=ids[actual['conditional_clearance']].tolist(),
            foot_contact_candidates=ids[actual['foot_contact_candidate']].tolist(),
            non_floor_conflicts=ids[actual['non_floor_conflict']].tolist(),
            predecessor_non_floor_conflicts=ids[old['non_floor_conflict']].tolist(),
            plane_exempt_shape_view_pairs=sum(v['plane_exempt_primitives'] for v in actual['views']),
            current_foot_centre_optical_depth_range_across_retained_views_m=[min(optical_depths), max(optical_depths)],
            observation_subset_ms=observe_ms, compiled_first_query_ms=query_ms,
            reference_compiled_all_fields_exact=True,
            contact_permitted=actual['contact_permitted'], navigation_qualified=actual['navigation_qualified'])
        rows.append(row); print(json.dumps(row), flush=True)
    verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'status': 'RECORDED_PLANE_MEMORY_DIAGNOSTIC_COMPLETE', 'frames': len(rows),
        'all_bindings_verified_before_and_after': True,
        'frames_with_any_floor_coverage': sum(r['floor_covered_shape_view_pairs'] > 0 for r in rows),
        'frames_with_any_conditional_clearance': sum(bool(r['conditional_clearance']) for r in rows),
        'frames_with_any_contact_candidate': sum(bool(r['foot_contact_candidates']) for r in rows),
        'maximum_nominal_foot_centre_optical_depth_m': max(r['current_foot_centre_optical_depth_range_across_retained_views_m'][1] for r in rows),
        'median_observation_subset_ms': float(np.median([r['observation_subset_ms'] for r in rows])),
        'median_compiled_first_query_ms': float(np.median([r['compiled_first_query_ms'] for r in rows])),
        'maximum_compiled_first_query_ms': max(r['compiled_first_query_ms'] for r in rows),
        'scope': 'conditional recorded-packet algorithm diagnostic, not a mission, calibration, contact permission or full-loop timing qualification'}), flush=True)


if __name__ == '__main__': main()
