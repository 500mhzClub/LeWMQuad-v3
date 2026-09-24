"""Analytic and causal-history negatives for conditional joint error factors."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_moment_sensitivity_development import (
    CorrelatedMomentSensitivity, RawRgbdMomentSensitivity, SHAPES, RAW_SHAPES)
from lewm.causal_depth_observation_development import FOCAL, from_native_depth
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.tests.test_depth_inertial_fusion_development import state
from lewm.tests.test_fast_gyro_development import buffers, advance
from lewm.tests.test_relative_gyro_turn_development import packet


def zero_loadings(count):
    return {name: np.zeros((*shape, count)) for name, shape in SHAPES.items()}


def stream(count, rate=(0., 0., 0.)):
    slow, fast = buffers(rate)
    orientation = FastRelativeOrientation()
    for tick in range(count):
        if tick == 0:
            p, f, now = packet(slow, 80), fast.packet(now_ns=1_600_000_000), 1_600_000_000
            rotation = orientation.begin(p, f, now_ns=now)
        else:
            p, f, now = advance(slow, fast, tick, rate)
            rotation = orientation.step(p, f, now_ns=now)
        yield tick, p, f, np.asarray(rotation['rotation_initial_body_from_current_body'])


def bias_loadings(policy, fast, *, force_y=.02, gyro_z=0., onset=1_600_000_001):
    result = zero_loadings(1)
    for name in ('specific_force', 'gyro'):
        times = policy['sensor_state']['sensed'][name]['measured_ns']
        result[name][times >= onset, 1 if name == 'specific_force' else 2, 0] = (
            force_y if name == 'specific_force' else gyro_z)
    result['fast_gyro'][fast['measured_ns'] >= onset, 2, 0] = gyro_z
    return result


def test_zero_sources_preserve_actual_nominal_outputs_and_inputs():
    model = CorrelatedMomentSensitivity(['declared_zero_source'])
    reference = MomentWeakSubspaceIntegrator()
    for tick, p, f, r in stream(5, (.02, -.01, .04)):
        s = state(p, None if not tick else [.01, 0., 0.], [[0, 1, 0]] if tick > 1 else (), r)
        before = deepcopy(s), p['sensor_state']['sensed']['specific_force']['values'].copy(), f['values'].copy()
        result = model.observe(p, s, f, zero_loadings(1))
        assert result['nominal_fusion'] == reference.observe(p, s)
        assert s == before[0]
        np.testing.assert_array_equal(p['sensor_state']['sensed']['specific_force']['values'], before[1])
        np.testing.assert_array_equal(f['values'], before[2])
        np.testing.assert_array_equal(result['pose_error_factor'], np.zeros((6, 1)))
        assert result['depth_rank_and_correspondences_conditioned_on']
        assert not result['source_error_model_calibrated'] and not result['navigation_qualified']


def test_bias_preserves_velocity_correlation_across_weak_intervals_and_recovery():
    model = CorrelatedMomentSensitivity(['persistent_force_y'])
    for tick, p, f, r in stream(13):
        weak = [[0, 1, 0]] if 1 < tick < 12 else ()
        result = model.observe(p, state(p, [.01, 0, 0] if tick else None, weak, r), f, bias_loadings(p, f))
        if tick == 1:
            model.retain('before_weak')
        if 1 < tick < 12:
            t = (tick - 1) * .1
            expected = .02 * (.05 * t + .5 * t * t)
            assert result['pose_error_factor'][1, 0] == pytest.approx(expected, abs=1e-12)
        if tick == 11:
            before_recovery = result['pose_error_factor'][1, 0]
    assert result['pose_error_factor'][1, 0] == pytest.approx(before_recovery, abs=1e-12)
    # Full-depth recovery resets constrained velocity, not past position error.
    assert result['nominal_fusion']['depth_rank'] == 3
    assert result['pose_error_factor'][1, 0] > .01


def test_initial_gravity_estimation_shares_the_force_bias_source():
    model = CorrelatedMomentSensitivity(['force_y_bias_present_at_anchor'])
    for tick, p, f, r in stream(8):
        s = state(p, [.01, 0, 0] if tick else None, [[0, 1, 0]] if tick > 1 else (), r)
        result = model.observe(p, s, f, bias_loadings(p, f, onset=0))
    # To first order the sideways bias was absorbed into initial gravity.
    # This cancellation is not an ability to distinguish gravity and bias.
    assert abs(result['pose_error_factor'][1, 0]) < 1e-10


def test_shared_gyro_bias_relative_uncertainty_depends_on_elapsed_interval():
    model = CorrelatedMomentSensitivity(['gyro_z_bias'])
    for tick, p, f, r in stream(11):
        s = state(p, [0, 0, 0] if tick else None, (), r)
        result = model.observe(p, s, f, bias_loadings(p, f, force_y=0, gyro_z=.001, onset=0))
        if tick == 5:
            model.retain('half_second')
    assert result['pose_error_factor'][5, 0] == pytest.approx(.001, rel=1e-9)
    relative = model.relative_moments('half_second', [[1, 0, 0]])
    assert relative['point_covariance_m2'][0, 1, 1] == pytest.approx((.0005)**2, rel=1e-9)
    assert relative['joint_pose_covariance'][5, 11] == pytest.approx(.001 * .0005, rel=1e-9)
    model.retain('now')
    np.testing.assert_allclose(model.relative_moments('now', [[1, .4, .2]])['point_covariance_m2'], 0., atol=1e-20)


@pytest.mark.parametrize('mode', ['shared_increment', 'shared_frame_difference', 'independent_increment'])
def test_depth_frame_correlations_are_not_replaced_by_independent_endpoint_errors(mode):
    count = 5 if mode == 'independent_increment' else 1
    model = CorrelatedMomentSensitivity([f'depth_source_{i}' for i in range(count)])
    for tick, p, f, r in stream(6):
        loading = zero_loadings(count)
        if tick:
            if mode == 'shared_increment': loading['depth_projection'][0, 0] = .001
            elif mode == 'shared_frame_difference':
                # A single frame's offset enters two adjacent registrations
                # with opposite signs and exactly cancels after the second.
                loading['depth_projection'][0, 0] = .001 * ((tick == 1) - (tick == 2))
            else: loading['depth_projection'][0, tick - 1] = .001
        result = model.observe(p, state(p, [.01, 0, 0] if tick else None, (), r), f, loading)
    expected = {'shared_increment': 25e-6, 'shared_frame_difference': 0., 'independent_increment': 5e-6}[mode]
    assert result['conditional_pose_covariance'][0, 0] == pytest.approx(expected, abs=1e-15)


def test_weak_basis_uncertainty_rotates_both_observation_and_nullspace():
    model = CorrelatedMomentSensitivity(['normal_direction_error'])
    for tick, p, f, r in stream(3):
        loading = zero_loadings(1)
        if tick == 2: loading['depth_basis_rotation'][2, 0] = .01
        s = state(p, [.01, 0, 0] if tick else None, [[0, 1, 0]] if tick == 2 else (), r)
        result = model.observe(p, s, f, loading)
    assert np.isfinite(result['pose_error_factor']).all()
    assert result['nominal_fusion']['depth_rank'] == 2
    assert not result['nominal_fusion']['uncertainty_model_validated']


@pytest.mark.parametrize('fault', ['missing', 'shape', 'nan', 'rewrite', 'gyro_disagreement',
                                  'weak_projection', 'attitude', 'clock', 'raw_rewrite'])
def test_bad_error_or_sensor_history_latches_and_disables_retained_queries(fault):
    model = CorrelatedMomentSensitivity(['source'])
    items = stream(2)
    _, p, f, r = next(items)
    model.observe(p, state(p, rotation=r), f, zero_loadings(1))
    model.retain('anchor')
    _, p, f, r = next(items)
    loading = zero_loadings(1)
    s = state(p, [.01, 0, 0], [[0, 1, 0]], r)
    if fault == 'missing': del loading['gyro']
    elif fault == 'shape': loading['depth_projection'] = np.zeros(3)
    elif fault == 'nan': loading['specific_force'][-1, 0, 0] = np.nan
    elif fault == 'rewrite': loading['specific_force'][0, 0, 0] = .001
    elif fault == 'gyro_disagreement': loading['fast_gyro'][-1, 0, 0] = .001
    elif fault == 'weak_projection': loading['depth_projection'][1, 0] = .001
    elif fault == 'attitude': s['relative_orientation']['rotation_initial_body_from_current_body'][0][0] += .01
    elif fault == 'clock': f['decision_ns'] += 1
    elif fault == 'raw_rewrite': p['sensor_state']['sensed']['specific_force']['values'][0, 0] += .01
    with pytest.raises(SensorContractError): model.observe(p, s, f, loading)
    assert model.failed
    with pytest.raises(SensorContractError): model.relative_moments('anchor', [[1, 0, 0]])
    with pytest.raises(SensorContractError): model.observe(p, s, f, zero_loadings(1))


@pytest.mark.parametrize('names, step', [([], .001), (['x', 'x'], .001), ([None], .001),
                                       (['x'], 0), (['x'], np.nan), (['x'], .1)])
def test_explicit_finite_unique_source_basis_required(names, step):
    with pytest.raises(SensorContractError): CorrelatedMomentSensitivity(names, difference_step=step)


def test_missing_anchor_registration_loadings_rejected():
    model = CorrelatedMomentSensitivity(['source'])
    _, p, f, r = next(stream(1))
    loading = zero_loadings(1); loading['depth_projection'][0, 0] = .001
    with pytest.raises(SensorContractError): model.observe(p, state(p, rotation=r), f, loading)


def room_depth(policy):
    x = (np.arange(640) + .5 - 320) / FOCAL
    y = (np.arange(480) + .5 - 240) / FOCAL
    native = np.minimum(2., .68 / np.maximum(abs(x), 1e-12))[None, :] * np.ones((480, 1))
    native = np.minimum(native, np.where(y[:, None] > 0, .343 / np.maximum(y[:, None], 1e-12), 200.))
    now = policy['sensor_state']['decision_ns']
    return from_native_depth(native.astype(np.float32), policy, measured_ns=now, available_ns=now, now_ns=now)


def test_raw_gyro_error_changes_registration_even_when_rank_stays_three():
    raw = RawRgbdMomentSensitivity(['shared_yaw_bias'])
    conditional = CorrelatedMomentSensitivity(['shared_yaw_bias'])
    from lewm.depth_inertial_moment_fusion_development import MomentDepthInertialState
    reference = MomentDepthInertialState()
    for tick, p, f, r in stream(4):
        depth = room_depth(p); before = depth['depth_m'].copy()
        loading = bias_loadings(p, f, force_y=0, gyro_z=.001, onset=0)
        raw_loading = {k: loading[k] for k in ('specific_force', 'gyro', 'fast_gyro')}
        raw_loading['depth_m'] = np.zeros((480, 640, 1))
        nominal = reference.observe(p, depth, f, now_ns=p['sensor_state']['decision_ns'])
        a = raw.observe(p, depth, f, raw_loading)
        b = conditional.observe(p, nominal['depth_state'], f, loading)
        assert a['nominal_fusion'] == b['nominal_fusion'] == nominal['fusion']
        np.testing.assert_array_equal(depth['depth_m'], before)
        if tick: assert a['nominal_fusion']['depth_rank'] == 3
    assert np.linalg.norm(a['pose_error_factor'][:3] - b['pose_error_factor'][:3]) > .0001
    assert a['raw_registration_recomputed'] and a['sensor_quantization_preserved']
    assert not a['depth_rank_and_correspondences_conditioned_on']
    assert not a['linearization_validated'] and not a['navigation_qualified']
    raw.retain('now')
    assert not raw.relative_moments('now', [[1, 0, 0]])['depth_rank_and_correspondences_conditioned_on']


def test_raw_depth_range_error_runs_through_actual_quantized_registration():
    model = RawRgbdMomentSensitivity(['depth_range_error'])
    for tick, p, f, r in stream(2):
        depth = room_depth(p)
        loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
        if tick: loading['depth_m'][depth['valid'], 0] = .01
        result = model.observe(p, depth, f, loading)
    assert np.linalg.norm(result['pose_error_factor'][:3]) > .001
    assert result['nominal_fusion']['depth_rank'] == 3
    assert not result['source_error_model_calibrated']


def test_missing_raw_depth_cannot_be_assigned_measurement_noise():
    model = RawRgbdMomentSensitivity(['depth_range_error'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    depth['valid'][0, 0] = False; depth['depth_m'][0, 0] = 0.
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    loading['depth_m'][0, 0, 0] = .01
    with pytest.raises(SensorContractError): model.observe(p, depth, f, loading)
    assert model.failed


def test_raw_rank_branch_change_invalidates_local_covariance(monkeypatch):
    model = RawRgbdMomentSensitivity(['depth_range_error'])
    _, p, f, _ = next(stream(1)); depth = room_depth(p)
    loading = {name: np.zeros((*shape, 1)) for name, shape in RAW_SHAPES.items()}
    original = model.models[1].observe
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result['fusion']['depth_rank'] = 2
        return result
    monkeypatch.setattr(model.models[1], 'observe', changed)
    with pytest.raises(SensorContractError) as error: model.observe(p, depth, f, loading)
    assert 'changed registration rank' in str(error.value.__cause__)
    assert model.failed and model.current is None
