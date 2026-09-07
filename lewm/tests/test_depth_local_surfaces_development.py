from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_depth_observation_development import FOCAL, from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_local_surfaces_development import observe_local_surfaces, LocalSurfaceHistory
from lewm.tests.test_observed_traversal_controller_development import Stream


def packet(kind, *, stream=None, tick=0):
    policy, _, now = (stream or Stream()).frame(tick)
    u = np.arange(640)+.5
    y_per_x = -(u-320)/FOCAL
    native = np.full((480, 640), 200., np.float32)
    if kind == 'front': native[:] = 2.
    elif kind in ('corridor', 'dead_end'):
        sides = .68/np.maximum(np.abs(y_per_x), 1e-12)
        if kind == 'dead_end': sides = np.minimum(sides, 2.)
        native[:] = sides
    elif kind == 'step':
        native[:, :320] = 1.; native[:, 320:] = 3.
    elif kind != 'unknown': raise ValueError(kind)
    depth = from_native_depth(native, policy, measured_ns=now, available_ns=now, now_ns=now)
    return policy, depth, now


def test_actual_metric_front_wall_and_no_clearance_inference():
    p, d, now = packet('front'); before = deepcopy(d)
    r = observe_local_surfaces(d, p, now_ns=now)
    assert len(r['surface_segments']) == 1
    wall = r['surface_segments'][0]
    np.testing.assert_allclose(wall['normal_body_xy'], [1., 0.], atol=1e-12)
    assert wall['offset_body_m'] == pytest.approx(2.326, abs=1e-6)
    assert wall['support_columns'] == 320
    assert wall['extent_kind'] == 'observed_support_only_not_physical_wall_endpoints'
    for k in ('body_clearance_qualified', 'turn_clearance_qualified', 'arrival_verified', 'free_volume_inferred'):
        assert r[k] is False
    np.testing.assert_array_equal(d['depth_m'], before['depth_m'])


def test_corridor_cannot_observe_forward_translation_and_has_unknown_central_rays():
    p, d, now = packet('corridor'); r = observe_local_surfaces(d, p, now_ns=now)
    assert len(r['surface_segments']) == 2
    assert r['unknown_column_runs']
    assert r['depth_discontinuities'] == []  # Never bridge through out-of-range rays.
    c = r['translation_constraint_geometry']
    assert c['conditional_rank_xy'] == 1 and not c['translation_estimated']
    np.testing.assert_allclose(np.abs(c['weak_directions_body_xy'][0]), [1., 0.], atol=1e-6)
    for segment in r['surface_segments']:
        assert segment['offset_body_m'] == pytest.approx(.68, abs=1e-6)


def test_dead_end_has_two_constraint_directions_not_a_motion_estimate():
    p, d, now = packet('dead_end'); r = observe_local_surfaces(d, p, now_ns=now)
    assert len(r['surface_segments']) >= 3
    c = r['translation_constraint_geometry']
    assert c['conditional_rank_xy'] == 2 and c['weak_directions_body_xy'] == []
    assert not c['translation_estimated'] and not c['calibrated_uncertainty']
    assert max(s['maximum_fit_residual_m'] for s in r['surface_segments']) <= .015


def test_occlusion_discontinuity_is_not_invented_portal_width_or_free_space():
    p, d, now = packet('step'); r = observe_local_surfaces(d, p, now_ns=now)
    assert len(r['surface_segments']) == 2 and len(r['depth_discontinuities']) == 1
    boundary = r['depth_discontinuities'][0]
    assert boundary['columns'] == [319, 321] and boundary['nearer_column'] == 319
    assert boundary['portal_width_m'] is None and boundary['traversable'] is None


def test_unknown_input_has_no_constraints_and_never_becomes_free():
    p, d, now = packet('unknown'); r = observe_local_surfaces(d, p, now_ns=now)
    assert not r['surface_segments'] and not any(r['valid_columns'])
    assert r['unknown_column_runs'] == [[1, 639]]
    assert r['translation_constraint_geometry']['conditional_rank_xy'] == 0
    assert not r['free_volume_inferred']


def test_hole_splits_observed_support_and_mixed_vertical_surfaces_are_rejected():
    p, d, now = packet('front')
    d['depth_m'][:, 300:340] = 0.; d['valid'][:, 300:340] = False
    r = observe_local_surfaces(d, p, now_ns=now)
    assert len(r['surface_segments']) == 2 and r['unknown_column_runs'] == [[301, 339]]
    p, d, now = packet('front')
    d['depth_m'][240:, 300:340] = 1.9
    r = observe_local_surfaces(d, p, now_ns=now)
    assert r['unknown_column_runs'] == [[301, 339]]


def test_thin_obstacle_returns_are_retained_even_when_too_small_for_a_line_fit():
    p, d, now = packet('front')
    d['depth_m'][:, 316:324] = 1.
    r = observe_local_surfaces(d, p, now_ns=now)
    assert r['unmodelled_valid_columns'] == [317, 319, 321, 323]
    for col in r['unmodelled_valid_columns']:
        i = r['sampled_columns'].index(col)
        assert r['valid_columns'][i]
        assert r['sampled_points_body_xy_m'][i][0] == pytest.approx(1.326)
    assert not r['free_volume_inferred']


def test_causal_observer_history_and_fault_latch():
    stream = Stream(); observer = LocalSurfaceHistory()
    for tick in range(6):
        p, d, now = packet('front', stream=stream, tick=tick)
        result = observer.observe(d, p, now_ns=now)
    assert len(observer.snapshot()) == 4
    result['surface_segments'].clear()
    snap = observer.snapshot(); snap[-1]['surface_segments'].clear()
    assert observer.snapshot()[-1]['surface_segments']
    with pytest.raises(SensorContractError): observer.observe(d, p, now_ns=now)
    p, d, now = packet('front', stream=stream, tick=6)
    with pytest.raises(SensorContractError): observer.observe(d, p, now_ns=now)


@pytest.mark.parametrize('fault', ['privilege', 'future', 'rgb'])
def test_observer_rejects_privileged_or_unbound_input(fault):
    p, d, now = packet('front')
    if fault == 'privilege': d['wall_boxes'] = []
    if fault == 'future': d['available_ns'] = now+1
    if fault == 'rgb': p['image']['rgb'][0, 0] ^= 1
    with pytest.raises(SensorContractError): observe_local_surfaces(d, p, now_ns=now)
