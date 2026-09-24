"""Synthetic counterexamples to treating boundary accounting as certification.

These construct evaluator geometry and synthetic arrays only. They never read
or modify native recordings, public sensor packets or a navigation policy.
"""
import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth, evaluate_visibility
from lewm.raster_footprint_visibility_development import evaluate_footprint

PIXEL = (244, 324)


def fixture(*, near=False):
    transform = np.eye(4)
    transform[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
    transform[:3, 3] = [0., 0., .6]
    background = dict(wall_id='background', centre_xyz=[2., 0., 2.],
        size_xyz=[.08, 8., 4.], yaw_rad=0.)
    row, column = PIXEL
    optical = np.array([(column+.5-320)/FOCAL, (row+.5-240)/FOCAL, 1.])
    front = .004 if near else .96
    thickness = .0004 if near else .08
    center = transform[:3, 3]+front*(transform[:3, :3]@optical)
    center[0] += thickness/2
    obstacle = dict(wall_id='thin_near_occluder' if near else 'thin_post',
        centre_xyz=center.tolist(), size_xyz=[thickness, .00002 if near else .001,
            .00002 if near else 1.4], yaw_rad=0.)
    # The synthetic wrong sensor sees only the background/floor. This is not
    # a native render, a repaired recording or a generated policy packet.
    absent = expected_optical_depth([background], transform, stride=1)['expected_depth_m'].astype(np.float32)
    return transform, [background, obstacle], absent


def score_case(*, near=False, fabricated=False):
    transform, boxes, depth = fixture(near=near)
    if fabricated:
        if near: raise ValueError('separate counterexamples required')
        depth[PIXEL] = np.float32(1.5)
    strict = evaluate_visibility(depth, boxes, transform, render_near_m=.005)
    footprint = evaluate_footprint(depth, boxes, transform, render_near_m=.005)
    ref = expected_optical_depth(boxes, transform)
    thin = ref['object_index'] == ref['object_names'].index(boxes[1]['wall_id'])
    measured = depth[np.ix_(ref['rows'], ref['columns'])]
    y = int(np.flatnonzero(ref['rows'] == PIXEL[0])[0])
    x = int(np.flatnonzero(ref['columns'] == PIXEL[1])[0])
    error = np.abs(measured-ref['expected_depth_m'])
    if not thin[y, x] or not np.any(thin): raise ValueError('sampled foreground counterexample required')
    return dict(case='near_plane_opaque_occluder' if near else ('fabricated_thin_boundary_depth' if fabricated else 'missed_thin_post'),
        pixel_row_column=list(PIXEL), foreground_box=boxes[1],
        tracked_expected_depth_m=float(ref['expected_depth_m'][y, x]),
        tracked_synthetic_depth_m=float(measured[y, x]),
        tracked_expected_surface_interior=bool(ref['surface_interior'][y, x]),
        sampled_foreground_rays=int(thin.sum()),
        foreground_rays_excluded_by_original_interior_rule=int((thin & ~ref['surface_interior']).sum()),
        bad_foreground_rays=int((thin & (error > .001)).sum()),
        original_strict_score=strict, footprint_diagnostic=footprint,
        synthetic_only=True, native_rendered=False, public_sensor_packet_created=False,
        boundary_pixels_certified=False, navigation_qualified=False)


def diagnose():
    cases = [score_case(), score_case(fabricated=True), score_case(near=True)]
    for row in cases[:2]:
        assert row['bad_foreground_rays'] > 0
        assert row['sampled_foreground_rays'] == row['foreground_rays_excluded_by_original_interior_rule']
        assert row['original_strict_score']['passes_sampled_physical_visibility']
        assert row['footprint_diagnostic']['stable_interior_metric_pass']
        assert not row['footprint_diagnostic']['near_occlusion_failure']
    assert not cases[2]['original_strict_score']['passes_sampled_physical_visibility']
    assert cases[2]['original_strict_score']['clipped_opaque_rays'] > 0
    assert cases[2]['original_strict_score']['false_public_valid_near_rays'] > 0
    assert cases[2]['footprint_diagnostic']['near_occlusion_failure']
    return dict(cases=cases, boundary_accounting_alone_sufficient_for_visibility=False,
        original_surface_interior_filter_can_hide_thin_obstacles=True,
        original_near_occlusion_checks_preserve_this_counterexample=True,
        prospective_sensor_or_policy_fix_implemented=False,
        old_native_outcomes_relabelled=False, navigation_qualified=False, goal_achieved=False)
