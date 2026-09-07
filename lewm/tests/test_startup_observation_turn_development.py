from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.startup_observation_turn_development import StartupObservationTurn, all_posture_body_radius
from lewm.tests.test_primitive_obstacle_memory_development import observed_stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def components():
    geometry = ArticulatedCollisionGeometry(URDF)
    velocity = SetupVelocityPrior((0, 0, 0), 1_600_000_000, (0., 0., 0.), .02, '0' * 64)
    region = SetupRegionPrior((0, 0, 0), 1_600_000_000, 3_600_000_000,
                             (-1., -1., -1.), (1., 1., 1.), '0' * 64)
    admission = dict(schema='startup_setup_admission_development.v1', identity=(0, 0, 0),
        anchor_ns=1_600_000_000, definition_sha256='0' * 64, checks_sha256='1' * 64,
        velocity_and_nonfloor_checks_pass=True, initial_native_support_witness_present=True)
    return geometry, dict(velocity_prior=velocity, region_prior=region, admission=admission)


def test_all_posture_radius_bounds_primitive_supports_for_random_unrestricted_joints():
    geometry, _ = components(); bound = all_posture_body_radius(geometry); rng = np.random.default_rng(61062)
    assert len(bound['per_shape_radius_m']) == 27 and bound['all_joint_angles']
    directions = rng.normal(size=(200, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
    for q in rng.uniform(-np.pi, np.pi, (20, 12)):
        shapes = geometry.supports(q, directions)['shapes']
        for s in shapes:
            radius = bound['per_shape_radius_m'][s['shape_id']]
            assert np.max(s['upper']) <= radius + 1e-12
            assert np.min(s['lower']) >= -radius - 1e-12
    assert .74 < bound['radius_m'] < .75
    assert not bound['dynamic_base_motion_included']


@pytest.mark.parametrize('fault', ['missing', 'failed', 'support_missing', 'epoch', 'identity', 'boolean_identity', 'hash', 'privileged'])
def test_no_action_without_exact_narrow_setup_admission(fault):
    g, k = components(); a = k['admission']
    if fault == 'missing': a.pop('checks_sha256')
    if fault == 'failed': a['velocity_and_nonfloor_checks_pass'] = False
    if fault == 'support_missing': a['initial_native_support_witness_present'] = False
    if fault == 'epoch': a['anchor_ns'] += 1
    if fault == 'identity': a['identity'] = (0, 0, 1)
    if fault == 'boolean_identity': a['identity'] = (False, 0, 0)
    if fault == 'hash': a['definition_sha256'] = '2' * 64
    if fault == 'privileged': a['world_pose'] = [0, 0, 0]
    with pytest.raises(SensorContractError): StartupObservationTurn(g, **k)


def test_turn_then_full_rank_brake_and_quiet_terminal_with_real_consumer():
    g, k = components(); model = StartupObservationTurn(g, **k); rows = []
    k['admission']['checks_sha256'] = '9' * 64
    for p, d, r, now in observed_stream(5): rows.append(model.observe(p, d, r, now_ns=now))
    assert rows[0]['requested_command'] == [0., 0., .35]
    assert rows[1]['status'] == 'BRAKING_FOR_OBSERVATION'
    assert all(row['requested_command'] == [0., 0., 0.] for row in rows[1:])
    assert rows[-1]['status'] == 'COMPLETE_OBSERVATION_TURN' and rows[-1]['quiet_rank3_frames'] >= 3
    assert rows[0]['setup_checks_sha256'] == '1' * 64
    assert rows[0]['command_plus_stop_extent_m'] > .9
    assert all(not row['contact_permitted'] and not row['navigation_qualified'] for row in rows)
    with pytest.raises(SensorContractError): model.observe(p, d, r, now_ns=now)


def test_rank_two_never_becomes_observed_velocity_and_budget_stop_is_terminal():
    g, k = components(); model = StartupObservationTurn(g, **k)
    for i, (p, d, r, now) in enumerate(observed_stream(20)):
        if i:
            m = r['motion']; projection = np.array(m['observable_projection_previous_body_m']); projection[1] = 0.
            m.update(rank=2, status='PARTIALLY_OBSERVED_TRANSLATION', translation_previous_body_m=None,
                     weak_directions_previous_body=[[0., 1., 0.]], observable_projection_previous_body_m=projection.tolist())
        row = model.observe(p, d, r, now_ns=now)
        if row['terminal']: break
        assert row['first_rank3_ns'] is None
    assert row['status'] == 'FAILED_SENSOR_OR_FUSION' and i == 12
    assert any('budget exhausted' in message for message in row['failure_chain'])
    assert row['requested_command'] == [0., 0., 0.]


@pytest.mark.parametrize('fault', ['small_region', 'expired', 'depth_clock', 'observed_conflict', 'excess_speed'])
def test_conditional_envelope_and_sensor_vetoes_return_zero(fault, monkeypatch):
    g, k = components()
    if fault == 'small_region': k['region_prior'] = replace(k['region_prior'], upper_initial_body_m=(.9, .9, .9))
    if fault == 'expired': k['region_prior'] = replace(k['region_prior'], valid_until_ns=2_050_000_000)
    if fault == 'excess_speed': k['velocity_prior'] = replace(k['velocity_prior'], mean_initial_body_m_s=(.3, 0., 0.))
    model = StartupObservationTurn(g, **k)
    if fault == 'observed_conflict':
        original = model.memory.query_current_primitives
        def conflict(**kwargs):
            row = original(**kwargs); row['non_floor_conflict'][0] = True; return row
        monkeypatch.setattr(model.memory, 'query_current_primitives', conflict)
    packets = list(observed_stream(2))
    if fault == 'expired':
        p, d, r, now = packets[0]; model.observe(p, d, r, now_ns=now)
    p, d, r, now = packets[1 if fault == 'expired' else 0]
    if fault == 'depth_clock': d['measured_ns'] -= 1
    row = model.observe(p, d, r, now_ns=now)
    assert row['terminal'] and row['status'].startswith('FAILED_')
    assert row['requested_command'] == [0., 0., 0.]
