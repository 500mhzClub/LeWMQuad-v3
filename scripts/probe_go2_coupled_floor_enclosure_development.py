"""Read-only finite-region diagnostic on four previously unresolved samples.

Uses the saved nominal observation records and unchanged transport-radius
proxy. The plane/up allowances are illustrative hypotheses, not calibrated
sensor bounds. No controller change, physical rerun, or result rescoring.
"""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.coupled_floor_enclosure_development import query_observed_coupled_floor
from lewm.observed_turn_region_development import nominal_turn_volume
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory, transport_radius
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    rays = FusedRayEvidenceMemory()
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick)
        now = p['sensor_state']['decision_ns']
        rays.observe(p, d, records[tick]['observer'], now_ns=now)
    volume = nominal_turn_volume(ArticulatedCollisionGeometry(URDF),
                                 p['sensor_state']['sensed']['joints']['values'][:, :12],
                                 rays.rotation.T @ rays.up_initial)
    sample_ids = np.array([217, 218, 226, 227])
    q = volume['points_body_m'][sample_ids]
    roles = volume['ground_support_allowed'][sample_ids]
    assert roles.all()
    old = rays.query(q, roles, now_ns=now)
    assert old['unknown_or_blocked'].all()
    frames = list(rays.frames)
    if frames[-1]['measured_ns'] != now: frames.append(rays.latest_frame)
    reference = q @ rays.rotation.T + rays.position
    allowances = {'normal_error': .002, 'up_error': .001, 'plane_offset_error': .001}
    rows = []
    start = time.perf_counter()
    for frame in frames:
        body = (reference - frame['position']) @ frame['rotation']
        radius = transport_radius(q, rays.latest_frame, frame)
        evidence = frame['evidence']
        result = query_observed_coupled_floor(evidence['depth'], evidence['valid'],
                                             body - radius[:, None], body + radius[:, None],
                                             evidence['up'], roles, **allowances)
        rows.append(result)
        if frame is frames[0]:
            print(json.dumps({'first_view_ns': frame['measured_ns'], 'sample_ids': sample_ids.tolist(),
                              'body_points_at_tick180_m': q.tolist(), 'unchanged_radii_m': radius.tolist(),
                              'assumed_error_allowances': allowances,
                              'result': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}}), flush=True)
    keys = ('seed_observed', 'height_within_existing_band', 'all_projected_cells_observed_ground',
            'measured_planes_within_supplied_family', 'conditional_mesh_coverage')
    counts = {key: np.sum([row[key] for row in rows], axis=0).tolist() for key in keys}
    verify_bindings(bindings)
    print(json.dumps({'status': 'COUPLED_FLOOR_DIAGNOSTIC_COMPLETE', 'frames': 181, 'views': len(frames),
                      'sample_ids': sample_ids.tolist(), 'passing_view_counts': counts,
                      'four_point_all_view_diagnostic_seconds': time.perf_counter() - start,
                      'all_predecessor_bindings_verified_before_and_after': True,
                      'supplied_bounds_calibrated': False, 'ground_support_approved': False,
                      'whole_task_results_changed': False}), flush=True)


if __name__ == '__main__': main()
