"""Read-only actual-primitive floor semantics, not a navigation rescoring.

The local plane is seeded at one previously checked footprint. Whole-primitive
floor projections must then be observed independently. Non-floor clearance and
contact remain unresolved. Existing transport proxies/results are preserved.
"""
from itertools import product
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.correlated_floor_evidence_development import locate_floor_patches
from lewm.coupled_floor_enclosure_development import query_observed_coupled_floor
from lewm.observed_turn_region_development import nominal_turn_volume
from lewm.primitive_floor_relation_development import assess_primitive_floor_relation
from lewm.primitive_floor_observation_development import observe_primitive_floor
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory, transport_radius
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    rays = FusedRayEvidenceMemory(); model = ArticulatedCollisionGeometry(URDF)
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
        rays.observe(p, d, records[tick]['observer'], now_ns=now)
    joints = p['sensor_state']['sensed']['joints']['values'][:, :12]
    volume = nominal_turn_volume(model, joints, rays.rotation.T @ rays.up_initial)
    sample_ids = [217, 218, 226, 227]; q = volume['points_body_m'][sample_ids]
    old = rays.query(q, np.ones(4, bool), now_ns=now)
    assert old['unknown_or_blocked'].all()
    frame = rays.frames[0]; evidence = frame['evidence']
    body = (q @ rays.rotation.T + rays.position - frame['position']) @ frame['rotation']
    radii = transport_radius(q, rays.latest_frame, frame)
    floor = query_observed_coupled_floor(evidence['depth'], evidence['valid'],
                                         body - radii[:, None], body + radii[:, None], evidence['up'],
                                         np.ones(4, bool), normal_error=.002, up_error=.001,
                                         plane_offset_error=.001)
    assert floor['all_projected_cells_observed_ground'].all()
    assert floor['measured_planes_within_supplied_family'].all()
    assert not floor['height_within_existing_band'].any()
    assert (floor['height_lower_m'] > 0).all()
    # Seed one local observed plane. The observer below independently checks
    # all projected cells of each complete primitive before claiming coverage.
    seed = locate_floor_patches(evidence['depth'], evidence['valid'], body[:1], evidence['up'])
    assert seed['observed_footprint'][0]
    rotation = frame['rotation'].T @ rays.rotation
    translation = (rays.position - frame['position']) @ frame['rotation']
    corners = np.asarray(list(product((0, 1), repeat=3)), dtype=bool)
    all_rows = []; all_coverage = []
    for posture in joints:
        boxes = model.supports(posture, np.eye(3))['shapes']
        point_errors = {}
        for shape in boxes:
            points = np.where(corners, shape['upper'], shape['lower'])
            # Maximum of the unchanged radius proxy over a primitive AABB;
            # never shrink it using the four favourable footprint queries.
            point_errors[shape['shape_id']] = float(transport_radius(points, rays.latest_frame, frame).max())
        observed = observe_primitive_floor(model, posture, evidence['depth'], evidence['valid'], evidence['up'],
                                            seed['cells_rc'][0], rotation_observation_from_body=rotation,
                                            translation_observation_from_body=translation, normal_error=.002,
                                            up_error=.001, plane_offset_error=.001, point_error_by_shape=point_errors)
        bounds = observed['gap_bounds']
        ids = [row['shape_id'] for row in bounds['primitives']]
        assessment = assess_primitive_floor_relation(bounds, floor_coverage=observed['floor_coverage'],
                                                     non_floor_clearance=dict.fromkeys(ids, False))
        assert not assessment['all_primitives_conditionally_clear']
        all_rows.append(bounds['primitives'])
        all_coverage.append(observed['floor_coverage'])
    summary = []
    for i in range(27):
        rows = [r[i] for r in all_rows]
        summary.append({'shape_id': rows[0]['shape_id'], 'foot_geometry': rows[0]['contact_candidate_geometry'],
                        'nominal_minimum_gap_range_m': [min(r['nominal_minimum_gap_m'] for r in rows),
                                                        max(r['nominal_minimum_gap_m'] for r in rows)],
                        'minimum_gap_lower_over_history_m': min(r['minimum_gap_lower_m'] for r in rows),
                        'minimum_gap_upper_over_history_m': max(r['minimum_gap_upper_m'] for r in rows),
                        'postures_whole_floor_footprint_observed': sum(r[rows[0]['shape_id']] for r in all_coverage),
                        'postures_physically_separated_under_supplied_plane': sum(r['minimum_gap_lower_m'] > 0 for r in rows),
                        'postures_with_possible_padded_intersection': sum(r['padded_minimum_gap_lower_m'] <= 0 for r in rows),
                        'postures_with_certain_physical_penetration': sum(r['minimum_gap_upper_m'] < 0 for r in rows)})
    verify_bindings(bindings)
    print(json.dumps({'status': 'PRIMITIVE_FLOOR_RELATION_DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'observed_postures': len(joints), 'physical_primitives_per_posture': 27,
                      'four_padded_sample_lower_heights_m': floor['height_lower_m'].tolist(),
                      'four_padded_sample_old_height_gate_pass': floor['height_within_existing_band'].tolist(),
                      'primitives': summary, 'all_predecessor_bindings_verified_before_and_after': True,
                      'whole_primitive_floor_coverage_checked': True, 'non_floor_clearance_established': False,
                      'contact_permitted': False,
                      'navigation_qualified': False, 'whole_task_results_changed': False}), flush=True)


if __name__ == '__main__': main()
