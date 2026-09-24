"""Actual paired RGB-D geometry and failure accounting for the bounded probe."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import probe_go2_measured_plane_return_anchor_refinement_v1 as probe
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


@pytest.fixture(scope='module')
def geometry():
    probe.previous.cv2.setNumThreads(1)
    probe.previous.cv2.ocl.setUseOpenCL(False)
    item = next(sequence(1))
    model = probe.plane.MeasuredPlaneDualCameraPose()
    initial = run(model, item)
    features, states, planes = {}, {}, {}
    for frame in range(3099, 3104):
        now = 1_500_000_000 + frame*100_000_000
        features[frame] = {}
        for camera, rgb, original in (
                ('primary', item[0]['image']['rgb'], item[1]),
                ('auxiliary', item[3]['auxiliary_rgb']['rgb'], item[3]['auxiliary_depth'])):
            depth = deepcopy(original)
            depth.update(measured_ns=now, available_ns=now)
            features[frame][camera] = probe.previous.CornerSupportFeatureFrame(rgb, depth)
        receipt = deepcopy(initial['measured_plane_evidence'])
        receipt.update(frame=frame, measured_ns=now, up_reference_visual_frame=frame-1)
        receipt['depth_sha256'] = {camera: probe.plane.depth_hash(feature.depth)
            for camera, feature in features[frame].items()}
        states[frame] = dict(pose=dict(rotation_initial_body_from_current_body=np.eye(3).tolist(),
            position_initial_body_m=[0., 0., 0.]), plane=receipt)
        planes[frame] = probe.reconstruct_plane(features, receipt, frame)
    gyros = {frame: np.eye(3) for frame in features}
    rows = [probe.previous.fit_pair(features, gyros, ref, cur, camera, 'chained')
        for ref, cur, camera in probe.PAIRS]
    assert all(row['qualified'] for row in rows)
    return features, gyros, states, planes, rows


@pytest.mark.parametrize('index', range(3))
def test_actual_static_image_pairs_refine_with_all_original_inliers(geometry, index):
    features, gyros, states, planes, rows = geometry
    before = deepcopy(rows[index])
    result = probe.refine_pair(features, gyros, states, planes, rows[index])
    assert result['qualified'] and result['failure'] is None
    np.testing.assert_allclose(result['position_initial_body_m'], 0., atol=1e-12)
    np.testing.assert_allclose(result['rotation_initial_body_from_current_body'], np.eye(3), atol=1e-12)
    refinement = result['registration']['measured_plane_refinement']
    assert refinement['original_inliers_preserved']
    assert refinement['original_image_gate_values_unchanged']
    assert result['registration']['inliers'] == rows[index]['evidence']['inliers']
    assert before == rows[index]


@pytest.mark.parametrize('fault', ['clock', 'hash', 'up', 'plane', 'reference_clock'])
def test_raw_plane_receipt_changes_rejected(geometry, fault):
    features, _, states, _, _ = geometry
    receipt = deepcopy(states[3103]['plane'])
    if fault == 'clock': receipt['measured_ns'] += 1
    elif fault == 'hash': receipt['depth_sha256']['auxiliary'] = '0'*64
    elif fault == 'up': receipt['current_up_uses_public_gyro'] = True
    elif fault == 'plane': receipt['joint_plane']['normal_body'][0] += .01
    else: receipt['up_reference_visual_frame'] -= 1
    with pytest.raises(ValueError): probe.reconstruct_plane(features, receipt, 3103)


def test_global_displacement_failure_is_preserved_before_refinement(geometry):
    features, gyros, states, planes, rows = geometry
    states = deepcopy(states)
    states[3102]['pose']['position_initial_body_m'] = [10., 0., 0.]
    result = probe.refine_pair(features, gyros, states, planes, rows[0])
    assert not result['qualified'] and result['original_endpoint_reproduced']
    assert result['failure_stage'] == 'original_global_motion_envelopes'
    assert result['failure_type'] == 'SensorContractError'


def test_incompatible_measured_planes_preserve_real_refinement_rejection(geometry):
    features, gyros, states, planes, rows = geometry
    planes = deepcopy(planes)
    # A different valid measured height requires a translation incompatible
    # with the unchanged static image inliers.
    x, y = np.meshgrid(np.linspace(-1., 1., 20), np.linspace(-1., 1., 20))
    cloud = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.52)))
    planes[3103] = probe.plane.fit_joint_plane(cloud, cloud, np.array([0., 0., 1.]))
    result = probe.refine_pair(features, gyros, states, planes, rows[0])
    assert not result['qualified'] and result['failure']
    assert result['failure_stage'] == 'measured_plane_refinement'


@pytest.mark.parametrize('fault', ['evidence', 'mask'])
def test_changed_original_fit_is_execution_failure(geometry, fault):
    features, gyros, states, planes, rows = geometry
    row = deepcopy(rows[0])
    if fault == 'evidence': row['evidence']['inliers'] += 1
    else: row['inlier_mask_sha256'] = '0'*64
    with pytest.raises(ValueError, match='reproduce exactly'):
        probe.refine_pair(features, gyros, states, planes, row)


@pytest.mark.parametrize('fault', [None, 'missing', 'extra', 'method', 'order', 'population'])
def test_exact_previously_qualified_population(geometry, fault):
    rows = deepcopy(geometry[-1]) + [dict(qualified=False) for _ in range(61)]
    if fault == 'missing': rows[0]['qualified'] = False
    elif fault == 'extra': rows[3] = deepcopy(rows[0])
    elif fault == 'method': rows[0]['method'] = 'descriptor'
    elif fault == 'order': rows[0], rows[1] = rows[1], rows[0]
    elif fault == 'population': rows.pop()
    report = dict(status='MEASURED_PLANE_RETURN_ANCHOR_PAIR_PROBE_COMPLETE', pairs=rows)
    if fault:
        with pytest.raises(ValueError): probe.selected_pairs(report)
    else: assert probe.selected_pairs(report) == rows[:3]
