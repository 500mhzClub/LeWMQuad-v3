from copy import deepcopy
import hashlib

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.finite_rgbd_error_members_development import ErrorMember, FiniteRGBDMember, fixed_members, perturb_packets
from lewm.rgbd_shadow_motion_development import ShadowObserver
from lewm.tests.test_raw_complementary_rgbd_sensitivity_development import room
from lewm.tests.test_rgbd_correspondence_motion_development import packets
from lewm.tests.test_rgbd_inertial_fusion_development import prior


def test_nominal_matches_original_owner_exactly():
    model = FiniteRGBDMember(ErrorMember('nominal'), prior()); original = ShadowObserver(prior())
    for p, d, f, now in room(3):
        assert model.observe(p, d, f) == original.observe(p, d, f, now_ns=now)


def test_fixed_population_unique_complete_and_finite():
    cases = fixed_members()
    assert len(cases) == len({c.name for c in cases}) == 16
    assert cases[0] == ErrorMember('nominal') and sum(c.blank_rgb for c in cases) == 1
    with pytest.raises(SensorContractError): ErrorMember('bad', depth_gain=np.nan)


def test_failure_is_terminal_and_does_not_stop_sibling():
    a = FiniteRGBDMember(ErrorMember('a'), prior()); b = FiniteRGBDMember(ErrorMember('b'), prior())
    data = list(room(3)); p, d, f, _ = data[0]
    bad = deepcopy(d); bad['depth_m'][300, 320] = 0.; bad['valid'][300, 320] = True
    first = a.observe(p, bad, f)
    assert first['status'] == 'TERMINAL_SHADOW_FAILURE' and a.last_packet is None
    b.observe(p, d, f)
    count = a.observer.model.state.depth.orientation.samples_integrated
    for p, d, f, _ in data[1:]:
        failed = a.observe(p, d, f); alive = b.observe(p, d, f)
        assert failed['status'] == 'NOT_REINVOKED_AFTER_SHADOW_FAILURE' and failed['failure'] == first['failure']
        assert alive['status'] == 'SHADOW_OBSERVATION_COMPLETE'
        assert a.observer.model.state.depth.orientation.samples_integrated == count


def test_bias_is_identical_in_overlapping_histories_and_both_gyro_rates():
    case = ErrorMember('bias', gyro_z_bias_rad_s=.001, post_anchor_force_y_bias_m_s2=.01)
    seen = {}; force_seen = {}; anchor = prior().anchor_ns
    for p, d, f, _ in room(4):
        q, _, fast = perturb_packets(case, p, d, f, anchor_ns=anchor)
        for channel in (q['sensor_state']['sensed']['gyro'], fast):
            for t, value, valid in zip(channel['measured_ns'], channel['values'], channel['valid'], strict=True):
                if t < 0 or not valid.all(): continue
                if int(t) in seen: np.testing.assert_allclose(value, seen[int(t)], atol=1e-12, rtol=0)
                seen[int(t)] = value.copy()
        force = q['sensor_state']['sensed']['specific_force']; old = p['sensor_state']['sensed']['specific_force']
        for i, t in enumerate(force['measured_ns']):
            expected = .01 if t>anchor and force['valid'][i, 1] else 0.
            assert force['values'][i, 1]-old['values'][i, 1] == pytest.approx(expected)
            if t>=0:
                if int(t) in force_seen: np.testing.assert_array_equal(force['values'][i], force_seen[int(t)])
                force_seen[int(t)] = force['values'][i].copy()


def test_prior_is_changed_once_and_affects_weak_motion():
    cases = [FiniteRGBDMember(ErrorMember('nominal'), prior()),
             FiniteRGBDMember(ErrorMember('prior', initial_velocity_y_error_m_s=.01), prior())]
    for p, d, f, now in packets([np.zeros((480, 640, 3), np.uint8)]*4):
        rows = [c.observe(p, d, f) for c in cases]
    a, b = [np.asarray(r['state']['fusion']['position_initial_body_m']) for r in rows]
    assert b[1]-a[1] == pytest.approx(.003, abs=1e-12)
    assert cases[1].observer.model.state.integrator.prior.mean_initial_body_m_s[1] == .01


def test_rgb_rejection_preserves_binding_and_does_not_change_depth_or_commands():
    case = ErrorMember('blank', blank_rgb=True); model = FiniteRGBDMember(case, prior())
    for p, d, f, _ in room(2):
        old = p['sensor_state']['control']['applied_command']['values'].copy()
        q, z, fast = perturb_packets(case, p, d, f, anchor_ns=prior().anchor_ns)
        assert (q['image']['rgb'] == 128).all()
        assert z['rgb_sha256'] == hashlib.sha256(q['image']['rgb'].tobytes()).hexdigest()
        np.testing.assert_array_equal(z['depth_m'], d['depth_m'])
        np.testing.assert_array_equal(q['sensor_state']['control']['applied_command']['values'], old)
        row = model.observe(p, d, f)
    assert row['state']['point_state']['motion']['status'] == 'INSUFFICIENT_POINT_SUPPORT'


def test_noise_is_timestamp_deterministic_preserves_missing_and_inputs():
    p, d, f, _ = next(room()); d['depth_m'][300, 310] = 0.; d['valid'][300, 310] = False
    before = d['depth_m'].copy(); case = ErrorMember('noise', independent_depth_noise_m=.0001)
    _, a, _ = perturb_packets(case, p, d, f, anchor_ns=prior().anchor_ns)
    _, b, _ = perturb_packets(case, p, d, f, anchor_ns=prior().anchor_ns)
    np.testing.assert_array_equal(a['depth_m'], b['depth_m']); np.testing.assert_array_equal(d['depth_m'], before)
    assert a['depth_m'].dtype == np.float32 and not a['depth_m'][~d['valid']].any()
    assert not np.array_equal(a['depth_m'], before)


def test_range_crossing_is_not_clipped_into_valid_measurement():
    case = ErrorMember('cross', depth_offset_m=5.)
    model = FiniteRGBDMember(case, prior()); p, d, f, _ = next(room())
    _, changed, _ = perturb_packets(case, p, d, f, anchor_ns=prior().anchor_ns)
    assert (changed['depth_m'][changed['valid']] > 5).any()
    assert model.observe(p, d, f)['status'] == 'TERMINAL_SHADOW_FAILURE'
