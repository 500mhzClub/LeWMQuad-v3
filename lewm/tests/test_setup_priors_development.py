from copy import deepcopy
from dataclasses import replace, FrozenInstanceError

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator
from lewm.setup_velocity_prior_development import SetupVelocityPrior, SetupVelocityIntegrator, PriorVelocitySensitivity
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.tests.test_depth_inertial_fusion_development import Stream, state, frame
from lewm.relative_gyro_turn_development import rotation_increment


def prior(mean=(0., 0., 0.), radius=.02):
    return SetupVelocityPrior((0, 0, 0), 1_600_000_000, tuple(float(x) for x in mean), radius, '0' * 64)


@pytest.mark.parametrize('fault', ['identity', 'epoch', 'mean', 'radius', 'zero', 'digest', 'mutable'])
def test_prior_rejects_invalid_or_mutable_values(fault):
    changes = dict(identity=(True, 0, 0)) if fault == 'identity' else {
        'epoch': dict(anchor_ns=True), 'mean': dict(mean_initial_body_m_s=(0., np.nan, 0.)),
        'radius': dict(radius_m_s=np.inf), 'zero': dict(radius_m_s=0.),
        'digest': dict(setup_evidence_sha256='unverified'), 'mutable': dict(mean_initial_body_m_s=[0., 0., 0.])}.get(fault)
    with pytest.raises(SensorContractError): replace(prior(), **changes)


def test_prior_is_immutable_and_not_optional():
    p = prior()
    with pytest.raises(FrozenInstanceError): p.radius_m_s = .01
    with pytest.raises(SensorContractError): SetupVelocityIntegrator(None)


def test_weak_start_uses_explicit_prior_without_changing_depth_evidence():
    stream = Stream(); model = SetupVelocityIntegrator(prior((0., .03, 0.)))
    original = MomentWeakSubspaceIntegrator()
    for tick in range(3):
        p = frame(stream, tick); s = state(p) if tick == 0 else state(p, [0., 0., 0.], [[0, 1, 0]])
        before = deepcopy(s); row = model.observe(p, s)
        assert s == before
        if tick == 0: original.observe(p, s)
        if tick == 1:
            with pytest.raises(SensorContractError): original.observe(p, s)
        np.testing.assert_allclose(row['position_initial_body_m'], [0., tick * .003, 0.], atol=1e-12)
        assert row['depth_rank'] == (2 if tick else None)
        assert row['initial_velocity_source'] == 'SUPPLIED_SETUP_PRIOR_NOT_SENSOR'
        assert not row['setup_independently_verified'] and not row['uncertainty_model_validated']
        assert row['position_error_scale_m'] >= row['inherited_position_error_scale_m']
        np.testing.assert_allclose(row['initial_velocity_prior_transport']['position_radius_m'], tick * .002, atol=2e-12)


def test_factor_transport_matches_distinct_prior_mean_replays_with_rotating_subspaces():
    rng = np.random.default_rng(92061); stream = Stream(); models = []
    means = [np.zeros(3), *rng.normal(0., .01, (5, 3))]
    for mean in means: models.append(SetupVelocityIntegrator(prior(mean)))
    for tick in range(9):
        p = frame(stream, tick); rotation = rotation_increment([.01 * tick, -.02 * tick, .1 * tick])
        weak = () if tick == 7 else ([[1., 0., 0.]] if tick % 2 else [[0., 1., 0.], [0., 0., 1.]])
        s = state(p) if tick == 0 else state(p, [0., 0., 0.], weak, rotation)
        rows = [m.observe(p, s) for m in models]
        x = np.asarray(rows[0]['initial_velocity_prior_transport']['position_factor_s'])
        v = np.asarray(rows[0]['initial_velocity_prior_transport']['velocity_factor'])
        for mean, row in zip(means[1:], rows[1:], strict=True):
            np.testing.assert_allclose(np.array(row['position_initial_body_m']) - rows[0]['position_initial_body_m'], x @ mean, atol=1e-12)
            np.testing.assert_allclose(np.array(row['velocity_initial_body_m_s']) - rows[0]['velocity_initial_body_m_s'], v @ mean, atol=1e-12)
        if tick >= 7: assert not v.any() and np.linalg.norm(x) > 0


def test_ball_transport_bounds_random_initial_velocity_errors_and_preserves_position_after_recovery():
    model = PriorVelocitySensitivity(); rng = np.random.default_rng(61029)
    directions = rng.normal(size=(1000, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
    errors = .02 * directions; x = np.zeros_like(errors); v = errors.copy()
    for i in range(12):
        w = rng.normal(size=3); w /= np.linalg.norm(w); p = np.outer(w, w)
        if i == 10: p[:] = 0.
        model.advance(p, .1); v = v @ p.T; x += .1 * v
        row = model.snapshot(.02)
        assert np.linalg.norm(x, axis=1).max() <= row['position_radius_m']
        assert np.linalg.norm(v, axis=1).max() <= row['velocity_radius_m_s']
    assert row['velocity_radius_m_s'] == 0. and row['position_radius_m'] > 0.


def test_larger_prior_cannot_reduce_combined_budget_scale():
    stream = Stream(); models = [SetupVelocityIntegrator(prior(radius=r)) for r in (.001, .02, .2)]
    for tick in range(11):
        p = frame(stream, tick); s = state(p) if tick == 0 else state(p, [0., 0., 0.], [[0, 1, 0]])
        rows = [m.observe(p, s) for m in models]
        scales = [r['position_error_scale_m'] for r in rows]
        assert np.all(np.diff(scales) >= 0)
    assert not rows[-1]['usable_under_declared_proxy_budget']


@pytest.mark.parametrize('fault', ['epoch', 'identity', 'clock', 'rank'])
def test_prior_fusion_faults_latch(fault):
    stream = Stream(); model = SetupVelocityIntegrator(prior())
    p = frame(stream, 0)
    if fault == 'epoch': model = SetupVelocityIntegrator(replace(prior(), anchor_ns=1_500_000_000))
    if fault == 'identity': model = SetupVelocityIntegrator(replace(prior(), identity=(0, 0, 1)))
    if fault in ('clock', 'rank'):
        model.observe(p, state(p)); p = frame(stream, 2 if fault == 'clock' else 1)
        s = state(p, [0., 0., 0.], [[0, 1, 0]])
        if fault == 'rank': s['motion']['rank'] = True
    else: s = state(p)
    with pytest.raises(SensorContractError): model.observe(p, s)
    assert model.failed
    with pytest.raises(SensorContractError, match='latched'): model.observe(p, s)


def region():
    return SetupRegionPrior((0, 0, 0), 100, 200, (-1., -1., -1.), (1., 1., 1.), '0' * 64)


@pytest.mark.parametrize('stamp,active', [(99, False), (100, True), (200, True), (201, False)])
def test_setup_region_expires_and_never_becomes_measured_support(stamp, active):
    r = region().query([[-.2, -.2, -.2]], [[.2, .2, .2]], [.01], identity=(0, 0, 0), now_ns=stamp, observed_conflict=[False])
    assert bool(r['conditional_setup_non_floor_clearance'][0]) == active
    for key in ('observed_free_space', 'support_established', 'contact_permitted', 'navigation_qualified', 'setup_independently_verified'):
        assert not r[key]


def test_region_uncertainty_and_observed_negative_cannot_expand_setup_clearance():
    r = region()
    kwargs = dict(identity=(0, 0, 0), now_ns=150)
    for conflict, radius, allowed in ((False, .01, True), (True, .01, False), (False, .3, False)):
        row = r.query([[.7, 0., 0.]], [[.8, .1, .1]], [radius], observed_conflict=[conflict], **kwargs)
        assert bool(row['conditional_setup_non_floor_clearance'][0]) == allowed
    edge = r.query([[1., 0., 0.]], [[1., 0., 0.]], [0.], observed_conflict=[False], **kwargs)
    assert not edge['conditional_setup_non_floor_clearance'].any()


@pytest.mark.parametrize('fault', ['identity', 'boolean_identity', 'radius', 'conflict', 'box', 'nonfinite'])
def test_region_rejects_wrong_frames_or_malformed_query(fault):
    kwargs = dict(identity=(0, 0, 0), now_ns=150, observed_conflict=[False])
    low, high, radius = [[0., 0., 0.]], [[.1, .1, .1]], [.01]
    if fault == 'identity': kwargs['identity'] = (0, 0, 1)
    if fault == 'boolean_identity': kwargs['identity'] = (False, 0, 0)
    if fault == 'radius': radius = [-1.]
    if fault == 'conflict': kwargs['observed_conflict'] = [1]
    if fault == 'box': high = [[-1., 0., 0.]]
    if fault == 'nonfinite': low = [[np.nan, 0., 0.]]
    with pytest.raises(SensorContractError): region().query(low, high, radius, **kwargs)


@pytest.mark.parametrize('fault', ['expired', 'mutable', 'reversed', 'nonfinite'])
def test_region_declaration_rejected(fault):
    changes = {'expired': dict(valid_until_ns=99), 'mutable': dict(lower_initial_body_m=[-1., -1., -1.]),
               'reversed': dict(upper_initial_body_m=(-2., 1., 1.)),
               'nonfinite': dict(upper_initial_body_m=(np.inf, 1., 1.))}[fault]
    with pytest.raises(SensorContractError): replace(region(), **changes)


def test_plane_memory_consumes_combined_prior_scale_and_does_not_add_starting_floor():
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.setup_velocity_prior_development import SetupVelocityPlaneMemory
    from lewm.tests.test_primitive_obstacle_memory_development import observed_stream
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    model = SetupVelocityPlaneMemory(ArticulatedCollisionGeometry(URDF), prior=prior(),
        normal_error=.002, up_error=.001, plane_offset_error=.001, range_error_m=.001)
    for i, (p, d, r, now) in enumerate(observed_stream(2)):
        if i:
            m = r['motion']; projected = np.asarray(m['observable_projection_previous_body_m']).copy(); projected[1] = 0.
            m.update(rank=2, status='PARTIALLY_OBSERVED_TRANSLATION', translation_previous_body_m=None,
                     weak_directions_previous_body=[[0., 1., 0.]], observable_projection_previous_body_m=projected.tolist())
        model.observe(p, d, r, now_ns=now)
    fusion = model._rays.fusion
    assert fusion['position_error_scale_m'] > fusion['inherited_position_error_scale_m']
    assert model._rays.latest_frame['position_scale_m'] == fusion['position_error_scale_m']
    row = model.query_current_primitives(now_ns=now)
    assert not row['conditional_clearance'].any() and not row['foot_contact_candidate'].any()
    assert not row['contact_permitted'] and not row['navigation_qualified']
