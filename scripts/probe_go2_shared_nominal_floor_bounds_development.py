"""Read-only shared nominal/ray/floor integration and supplied-bound coverage.

Bounds reuse the existing UNCALIBRATED pose proxy and fixed observed up. This
does not validate uncertainty, bound surface/up error, or approve ground support.
"""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.correlated_moment_sensitivity_development import RAW_SHAPES
from lewm.floor_footprint_bounds_development import observed_floor_cell_index, query_floor_bounds
from lewm.observed_turn_region_development import nominal_turn_volume
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.shared_nominal_floor_development import SharedNominalFloorEvidence
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory, transport_radius
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    source_ids = ['roll_gyro_0.001', 'force_y_0.02_from_tick80', 'depth_scale_0.002']
    shared = SharedNominalFloorEvidence(source_ids, difference_step=.01)
    rays = FusedRayEvidenceMemory(); geometry = ArticulatedCollisionGeometry(URDF)
    indices = {}; first_ns = None; timing = []
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick); f = load_fast_packet(DIRECTORY, tick)
        now = p['sensor_state']['decision_ns']
        if first_ns is None: first_ns = now
        loading = {name: np.zeros((*shape, 3)) for name, shape in RAW_SHAPES.items()}
        loading['gyro'][:, 0, 0] = .001; loading['fast_gyro'][:, 0, 0] = .001
        times = p['sensor_state']['sensed']['specific_force']['measured_ns']
        loading['specific_force'][times >= first_ns + 8_000_000_000, 1, 1] = .02
        loading['depth_m'][..., 2] = .002 * d['depth_m']
        start = time.perf_counter()
        result = shared.observe(p, d, f, loading)
        observer_end = time.perf_counter()
        # Saved JSON identities are lists; the live sensor identity is a tuple.
        # Compare the exact serialized record, with no numerical tolerance.
        assert json.loads(json.dumps(result['nominal_depth_state'])) == records[tick]['observer']
        rays.observe(p, d, result['nominal_depth_state'], now_ns=now)
        assert rays.fusion == result['nominal_fusion']
        memory_end = time.perf_counter()
        index = observed_floor_cell_index(d['depth_m'], d['valid'], rays.rotation.T @ rays.up_initial)
        index_end = time.perf_counter()
        keep = {frame['measured_ns'] for frame in rays.frames} | {now}
        indices = {t: value for t, value in indices.items() if t in keep}; indices[now] = index
        timing.append([observer_end - start, memory_end - observer_end, index_end - memory_end])
        if tick not in (80, 140, 180): continue
        start = time.perf_counter()
        volume = nominal_turn_volume(geometry, p['sensor_state']['sensed']['joints']['values'][:, :12],
                                     rays.rotation.T @ rays.up_initial)
        points, roles = volume['points_body_m'], volume['ground_support_allowed']
        old = rays.query(points, roles, now_ns=now)
        geometry_and_ray_seconds = time.perf_counter() - start
        selected = np.flatnonzero(roles); q = points[selected]
        covered = np.zeros(len(q), bool); cell_count = np.zeros(len(q), np.int64)
        reference = q @ rays.rotation.T + rays.position
        frames = list(rays.frames)
        if frames[-1]['measured_ns'] != now: frames.append(rays.latest_frame)
        start = time.perf_counter()
        for frame in frames:
            body = (reference - frame['position']) @ frame['rotation']
            radius = transport_radius(q, rays.latest_frame, frame)
            row = query_floor_bounds(indices[frame['measured_ns']], body - radius[:, None], body + radius[:, None],
                                     np.full(len(q), -.06), np.full(len(q), .06), np.ones(len(q), bool))
            covered |= row['all_projected_cells_observed_ground']; cell_count += row['covered_cells']
        coverage_seconds = time.perf_counter() - start
        blocked = old['unknown_or_blocked'][selected]
        print(json.dumps({'tick': tick, 'nominal_depth_record_exact': True, 'fusion_equal_between_consumers': True,
                          'views': len(frames), 'old_unknown_ground': int(blocked.sum()),
                          'old_blocked_with_nominal_fixed_up_rectangle_coverage': int((covered & blocked).sum()),
                          'indexed_cell_queries': int(cell_count.sum()),
                          'observer_memory_index_ms': [1000 * v for v in timing[-1]],
                          'geometry_and_original_ray_ms': 1000 * geometry_and_ray_seconds,
                          'supplied_bound_coverage_ms': 1000 * coverage_seconds,
                          'measured_pipeline_subset_ms': 1000 * (sum(timing[-1]) + geometry_and_ray_seconds + coverage_seconds),
                          'bounds_calibrated': False, 'up_and_surface_error_covered': False,
                          'ground_support_approved': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'SHARED_NOMINAL_AND_FLOOR_BOUNDS_DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'all_nominal_records_exact': True, 'all_consumer_fusion_records_equal': True,
                      'median_observer_memory_index_ms': (1000 * np.median(timing, axis=0)).tolist(),
                      'scope': 'shared computation and fixed-up coverage of supplied proxy boxes; not calibrated uncertainty or navigation'}), flush=True)


if __name__ == '__main__': main()
