"""Read-only paired floor diagnostics on the recorded north development arrival.

Three declared error hypotheses, two difference steps, unchanged actual packets
and original ray-memory decisions. Not calibration, clearance approval, physical
navigation, result rescoring or a held-out evaluation.
"""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.correlated_floor_evidence_development import PairedFloorEvidence
from lewm.correlated_moment_sensitivity_development import RAW_SHAPES
from lewm.observed_turn_region_development import nominal_turn_volume
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values): return {'count': 0}
    return {'count': len(values), 'minimum': float(np.min(values)),
            'median': float(np.median(values)), 'maximum': float(np.max(values))}


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    names = ['shared_roll_gyro_bias_0.001_rad_s', 'force_y_bias_0.02_m_s2_from_tick80',
             'shared_depth_range_scale_0.002']
    models = [PairedFloorEvidence(names, difference_step=s) for s in (.01, .005)]
    memory = FusedRayEvidenceMemory(); geometry = ArticulatedCollisionGeometry(URDF)
    first_ns = None; labels = []; times = []
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick); f = load_fast_packet(DIRECTORY, tick)
        now = p['sensor_state']['decision_ns']
        if first_ns is None: first_ns = now
        original = records[tick]['observer']
        memory_row = memory.observe(p, d, original, now_ns=now)
        loadings = {name: np.zeros((*shape, 3)) for name, shape in RAW_SHAPES.items()}
        loadings['gyro'][:, 0, 0] = .001; loadings['fast_gyro'][:, 0, 0] = .001
        force_times = p['sensor_state']['sensed']['specific_force']['measured_ns']
        loadings['specific_force'][force_times >= first_ns + 8_000_000_000, 1, 1] = .02
        loadings['depth_m'][..., 2] = .002 * d['depth_m']
        start = time.perf_counter()
        outputs = [model.observe(p, d, f, loadings) for model in models]
        times.append(time.perf_counter() - start)
        assert all(output['nominal_fusion'] == memory.fusion for output in outputs)
        if memory_row['new_view_added']:
            label = f'tick{tick}'; labels.append(label)
            for model in models: model.retain(label)
        if tick not in (80, 140, 180): continue
        query_labels = list(labels)
        if query_labels[-1] != f'tick{tick}':
            query_labels.append(f'tick{tick}')
            for model in models: model.retain(query_labels[-1])
        up = memory.rotation.T @ memory.up_initial
        q = p['sensor_state']['sensed']['joints']['values'][:, :12]
        volume = nominal_turn_volume(geometry, q, up)
        points = volume['points_body_m']; roles = volume['ground_support_allowed']
        old = memory.query(points, roles, now_ns=now)
        blocked = old['unknown_or_blocked'] & roles
        rows = []
        start = time.perf_counter()
        for model in models:
            found = np.zeros(len(points), bool); paired = found.copy()
            factors = np.full((len(points), 3), np.nan)
            independent = np.full(len(points), np.nan)
            heights = np.full(len(points), np.nan)
            selected_view = np.full(len(points), -1)
            for view_index, label in enumerate(query_labels):
                result = model.query(label, points, roles, now_ns=now)
                found |= result['nominal_observed_footprint']
                available = result['paired_footprints_observed']
                paired |= available
                # Most recent locally valid view, not a minimum-variance search.
                factors[available] = result['joint_height_factor_m'][available]
                independent[available] = result['incorrect_independent_height_variance_m2'][available]
                heights[available] = result['nominal_height_m'][available]
                selected_view[available] = view_index
            rows.append((found, paired, factors, independent, heights, selected_view))
        joint_query_ms = 1000 * (time.perf_counter() - start)
        comparable = blocked & rows[0][1] & rows[1][1]
        changed_view = comparable & (rows[0][5] != rows[1][5])
        comparable &= ~changed_view
        factor_difference = np.abs(rows[0][2][comparable] - rows[1][2][comparable])
        first = rows[0]
        print(json.dumps({'tick': tick, 'views': len(query_labels), 'ground_role_samples': int(roles.sum()),
                          'original_unknown_ground_samples': int(blocked.sum()),
                          'blocked_with_nominal_observed_footprint': int((blocked & first[0]).sum()),
                          'blocked_with_all_small_perturbation_footprints': int((blocked & first[1]).sum()),
                          'height_of_blocked_samples_m': describe(first[4][blocked]),
                          'declared_joint_height_std_m': describe(np.linalg.norm(first[2][blocked], axis=1)),
                          'incorrect_independent_height_std_m': describe(np.sqrt(first[3][blocked])),
                          'difference_step_factor_disagreement': describe(factor_difference.ravel()),
                          'step_pairs_selecting_different_views_excluded': int(changed_view.sum()),
                          'two_paired_floor_queries_ms': joint_query_ms,
                          'ground_support_approved': False, 'calibrated': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'PAIRED_FLOOR_DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'all_nominal_fusion_records_exact': True,
                      'two_paired_observer_median_ms': 1000 * float(np.median(times)),
                      'two_paired_observer_maximum_ms': 1000 * max(times),
                      'source_hypotheses': names,
                      'scope': 'reused development evidence and three assumed error sources; no calibrated envelope or navigation approval'}), flush=True)


if __name__ == '__main__':
    main()
