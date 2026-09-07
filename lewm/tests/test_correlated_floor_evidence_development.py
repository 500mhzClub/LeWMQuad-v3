"""Ground interpolation and correlated-error diagnostics, not motion approval."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_depth_observation_development import FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import patch_relation, locate_floor_patches, PairedFloorEvidence
from lewm.correlated_moment_sensitivity_development import RAW_SHAPES
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.uncertain_ray_memory_development import depth_evidence, query_envelopes
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth


PATCH = np.array([[1., -.05, -.3], [1.1, -.05, -.3], [1.1, .05, -.3], [1., .05, -.3]])


def test_signed_height_and_finite_triangle_footprint_not_infinite_plane():
    points = np.array([[1.05, 0, -.28], [1.05, 0, -.32], [1.15, 0, -.28]])
    result = patch_relation(points, np.broadcast_to(PATCH, (3, 4, 3)), np.ones((3, 4), bool), [0, 0, 1])
    np.testing.assert_allclose(result['height_m'], [.02, -.02, .02], atol=1e-14)
    assert result['observed_footprint'].tolist() == [True, True, False]
    assert not result['ground_support_approved']


@pytest.mark.parametrize('fault', ['missing', 'wall', 'step', 'degenerate', 'above_ground_role'])
def test_unobserved_or_non_ground_patch_cannot_supply_support(fault):
    patch = PATCH.copy(); valid = np.ones((1, 4), bool)
    if fault == 'missing': valid[0, 2] = False
    elif fault == 'wall': patch = patch @ rotation_increment([0, np.pi / 2, 0]).T
    elif fault == 'step': patch[2, 2] += .05
    elif fault == 'degenerate': patch[:] = patch[0]
    elif fault == 'above_ground_role': patch[:, 2] = -.1
    result = patch_relation([[1.05, 0, -.28]], patch[None], valid, [0, 0, 1])
    assert not result['observed_footprint'].any()
    assert not result['ground_patch_eligible'].any()


def test_shared_rigid_reference_error_cancels_in_floor_relation():
    rotation = rotation_increment([.03, -.02, .3]); shift = np.array([.02, -.03, .01])
    q = np.array([[1.05, 0, -.28]])
    base = patch_relation(q, PATCH[None], np.ones((1, 4), bool), [0, 0, 1])
    changed = patch_relation(q @ rotation.T + shift, (PATCH @ rotation.T + shift)[None],
                             np.ones((1, 4), bool), rotation @ [0, 0, 1])
    np.testing.assert_allclose(changed['height_m'], base['height_m'], atol=1e-14)
    assert changed['observed_footprint'].all()


def test_patch_locator_follows_foot_projection_not_the_nominal_camera_ray():
    _, p, _, _ = next(stream(1)); depth = room_depth(p)
    result = locate_floor_patches(depth['depth_m'], depth['valid'], [[1.5, .013, -.27]], [0, 0, 1])
    assert result['observed_footprint'].tolist() == [True]
    assert result['height_m'][0] == pytest.approx(.03, abs=1e-7)
    expected_row = int(np.floor(FOCAL * .343 / (1.5 - .326) + 239.5))
    assert result['cells_rc'][0, 0] == expected_row
    assert not result['ground_support_approved']


def test_hole_under_foot_not_filled_by_valid_floor_on_nominal_ray():
    _, p, _, _ = next(stream(1)); depth = room_depth(p)
    q = [[1.5, .013, -.33]]
    before = locate_floor_patches(depth['depth_m'], depth['valid'], q, [0, 0, 1])
    row, col = before['cells_rc'][0]
    depth['valid'][row:row+2, col:col+2] = False
    depth['depth_m'][row:row+2, col:col+2] = 0.
    result = locate_floor_patches(depth['depth_m'], depth['valid'], q, [0, 0, 1])
    assert not result['observed_footprint'].any()
    # The old ray/height predicate still sees nearby flat ground, despite
    # lacking a measured patch under this support-role query point.
    old = query_envelopes(depth_evidence(depth['depth_m'], depth['valid'], [0, 0, 1]),
                          q, [0.], np.array([True]))
    assert old['observed_ground_support'].all()


@pytest.mark.parametrize('point', [[1.5, 0, -.2], [1.5, 0, -.4], [-1, 0, -.3], [1.5, 3, -.3]])
def test_locator_does_not_expand_height_band_or_camera_coverage(point):
    _, p, _, _ = next(stream(1)); depth = room_depth(p)
    result = locate_floor_patches(depth['depth_m'], depth['valid'], [point], [0, 0, 1])
    assert not result['observed_footprint'].any()


def test_actual_raw_depth_scale_source_has_floor_height_uncertainty_at_same_pose():
    model = PairedFloorEvidence(['range_scale_0.002'], difference_step=.01)
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    loading['depth_m'][..., 0] = .002 * depth['depth_m']
    model.observe(p, depth, f, loading); model.retain('current')
    result = model.query('current', [[1.5, .013, -.27]], np.array([True]), now_ns=depth['measured_ns'])
    assert result['paired_footprints_observed'].all()
    assert result['joint_height_factor_m'][0, 0] == pytest.approx(.002 * .343, abs=4e-6)
    assert abs(result['pose_only_height_factor_m'][0, 0]) < 1e-12
    assert not result['ground_support_approved'] and not result['whole_error_envelope_covered']
    assert not result['sensor_model_calibrated']


def test_pose_and_floor_shared_vertical_source_cancels_not_independent_variances():
    # Explicit algebraic scene/source fixture, not a native sensor-noise model.
    model = PairedFloorEvidence(['shared_height_error'], difference_step=.01)
    _, p, _, _ = next(stream(1)); depth = room_depth(p)
    step = model.observer.step; scale = .001
    poses = [{'position': np.zeros(3), 'rotation': np.eye(3), 'up': np.array([0., 0., 1.])} for _ in range(3)]
    loadings = np.zeros((480, 640, 1))
    v = (np.arange(480) + .5 - 240) / FOCAL
    floor = (depth['depth_m'] * v[:, None] > .343 - 1e-6) & (v[:, None] > 0)
    loadings[..., 0] = np.where(floor, -scale / np.maximum(v[:, None], 1e-12), 0.)
    old = {'measured_ns': 1, 'poses': deepcopy(poses), 'depth': depth['depth_m'],
           'valid': depth['valid'], 'depth_loadings': loadings}
    model.retained['old'] = old
    poses[1]['position'][2] = step * scale; poses[2]['position'][2] = -step * scale
    model.current = {**old, 'measured_ns': 2, 'poses': poses}
    result = model.query('old', [[1.5, .013, -.27]], np.array([True]), now_ns=2)
    assert result['paired_footprints_observed'].all() and result['split_comparison_supported'].all()
    assert abs(result['joint_height_factor_m'][0, 0]) < 2e-6
    assert result['incorrect_independent_height_variance_m2'][0] > 1.9e-6


@pytest.mark.parametrize('fault', ['stale', 'missing_view', 'wrong_role_dtype', 'nonfinite', 'sensor_fault'])
def test_paired_floor_query_contracts(fault):
    model = PairedFloorEvidence(['zero'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    model.observe(p, depth, f, loading); model.retain('current')
    now = depth['measured_ns']; label = 'current'; q = [[1.5, .013, -.27]]; roles = np.array([True])
    if fault == 'stale': now += 1
    elif fault == 'missing_view': label = 'absent'
    elif fault == 'wrong_role_dtype': roles = np.array([1])
    elif fault == 'nonfinite': q = [[np.nan, 0, 0]]
    elif fault == 'sensor_fault':
        with pytest.raises(SensorContractError): model.observe(p, depth, f, loading)
        assert model.failed
    with pytest.raises(SensorContractError): model.query(label, q, roles, now_ns=now)


def test_non_ground_roles_and_empty_queries_do_not_approve_support():
    model = PairedFloorEvidence(['zero'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    model.observe(p, depth, f, loading); model.retain('current')
    for points, roles in (([[1.5, .013, -.27]], np.array([False])), (np.empty((0, 3)), np.zeros(0, bool))):
        result = model.query('current', points, roles, now_ns=depth['measured_ns'])
        assert not result['paired_footprints_observed'].any()
        assert not result['ground_support_approved']


def test_a_small_pair_leaving_the_observed_patch_invalidates_local_relation():
    model = PairedFloorEvidence(['zero'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    model.observe(p, depth, f, loading); model.retain('old')
    model.current = deepcopy(model.current); model.current['measured_ns'] += 100_000_000
    model.current['poses'][1]['position'][0] += .05
    result = model.query('old', [[1.5, .013, -.27]], np.array([True]), now_ns=model.current['measured_ns'])
    assert result['nominal_observed_footprint'].all()
    assert not result['paired_footprints_observed'].any()
    assert np.isnan(result['joint_height_variance_m2']).all()


def test_unsupported_independent_comparison_is_not_reported_as_a_variance(monkeypatch):
    model = PairedFloorEvidence(['shared_x_error'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    model.observe(p, depth, f, loading); model.retain('old')
    model.current = deepcopy(model.current); model.current['measured_ns'] += 100_000_000
    model.current['poses'][1]['position'][0] += .05
    model.current['poses'][2]['position'][0] -= .05
    original = model._patch
    def shifted(frame, index, cells):
        patch, valid = original(frame, index, cells)
        if index: patch[:, :, 0] += .05 if index == 1 else -.05
        return patch, valid
    monkeypatch.setattr(model, '_patch', shifted)
    result = model.query('old', [[1.5, .013, -.27]], np.array([True]), now_ns=model.current['measured_ns'])
    assert result['paired_footprints_observed'].all()
    assert not result['split_comparison_supported'].any()
    assert np.isnan(result['incorrect_independent_height_variance_m2']).all()
