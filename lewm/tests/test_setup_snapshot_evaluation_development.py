from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import linprog

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.setup_snapshot_evaluation_development import (
    box_separation_lower_bound, check_setup_snapshot, initial_ground_support_witness)
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_box_sat_matches_independent_linear_feasibility_for_rotated_boxes():
    rng = np.random.default_rng(61061)
    seen = set()
    for _ in range(40):
        a, b = rng.uniform(-1., 1., (2, 3))
        ra, rb = [rotation_increment(rng.normal(size=3)) for _ in range(2)]
        ha, hb = rng.uniform(.1, .6, (2, 3))
        # A point lies in both boxes iff these six bounded local coordinates
        # satisfy ra*u-rb*v=b-a. No SAT axes are used by this reference.
        lp = linprog(np.zeros(6), A_eq=np.column_stack((ra, -rb)), b_eq=b-a,
            bounds=list(zip(-np.r_[ha, hb], np.r_[ha, hb], strict=True)), method='highs')
        assert lp.status in (0, 2)
        gap = box_separation_lower_bound(a, ra, ha, b, rb, hb)
        assert (gap > 0) == (lp.status == 2)
        seen.add(lp.status)
    assert seen == {0, 2}


@pytest.mark.parametrize('offset,separated', [(1.9, False), (2., False), (2.00001, True)])
def test_touching_boxes_are_not_certified_clear(offset, separated):
    gap = box_separation_lower_bound([0, 0, 0], np.eye(3), [1, 1, 1], [offset, 0, 0], np.eye(3), [1, 1, 1])
    assert (gap > 0) == separated


def test_sat_invariant_under_common_rotation_translation_and_swap():
    a, b = np.array([0., .2, -.1]), np.array([2., .3, .4])
    ra, rb = rotation_increment([.2, .1, -.5]), rotation_increment([-.1, .8, .3])
    ha, hb = [.2, .4, .3], [.5, .1, .3]
    r, t = rotation_increment([.3, -.8, 1.]), np.array([.5, -.7, 1.])
    values = [box_separation_lower_bound(a, ra, ha, b, rb, hb),
              box_separation_lower_bound(b, rb, hb, a, ra, ha),
              box_separation_lower_bound(r @ a + t, r @ ra, ha, r @ b + t, r @ rb, hb)]
    np.testing.assert_allclose(values, values[0], atol=2e-12, rtol=0)


def inputs():
    velocity = SetupVelocityPrior((0, 0, 0), 100, (0., 0., 0.), .02, '0' * 64)
    region = SetupRegionPrior((0, 0, 0), 100, 200, (-1., -1., -.5), (1., 1., .5), '0' * 64)
    box = dict(native_name='wall', native_position=[3., 0., .5], native_quaternion_wxyz=[1., 0., 0., 0.],
               native_box_size=[.1, 4., 1., 0., 0., 0., 0.], fixed=True, collision_enabled=True, native_collision_boxes=1)
    kwargs = dict(identity=(0, 0, 0), measured_ns=100, position_world_m=[0., 0., .35],
        rotation_world_from_initial_body=np.eye(3), velocity_world_m_s=[0., .001, 0.],
        native_static_boxes=[box], expected_nonfloor_names=['wall'],
        geometry=ArticulatedCollisionGeometry(URDF), joint_position=np.repeat([0., .8, -1.5], 4))
    return velocity, region, kwargs


def test_valid_setup_is_not_ground_support_or_navigation_permission():
    v, r, k = inputs(); row = check_setup_snapshot(v, r, **k)
    assert row['velocity_and_nonfloor_setup_checks_pass']
    assert row['initial_27_primitives_with_4cm_padding_inside_region']
    assert not row['support_established'] and not row['navigation_qualified'] and not row['runtime_sensor_measurement']


@pytest.mark.parametrize('fault', ['missing_wall', 'extra_wall', 'duplicate', 'moving', 'disabled', 'multiple_geoms',
                                  'padding', 'size', 'orientation', 'clock', 'episode', 'digest'])
def test_setup_rejects_incomplete_or_malformed_evaluator_evidence(fault):
    from dataclasses import replace
    v, r, k = inputs(); box = k['native_static_boxes'][0]
    if fault == 'missing_wall': k['native_static_boxes'] = []
    if fault == 'extra_wall': k['expected_nonfloor_names'] += ['other']
    if fault == 'duplicate': k['native_static_boxes'] += [deepcopy(box)]
    if fault == 'moving': box['fixed'] = False
    if fault == 'disabled': box['collision_enabled'] = False
    if fault == 'multiple_geoms': box['native_collision_boxes'] = 2
    if fault == 'padding': box['native_box_size'][3] = 1.
    if fault == 'size': box['native_box_size'][0] = -1.
    if fault == 'orientation': box['native_quaternion_wxyz'] = [1., 1., 0., 0.]
    if fault == 'clock': k['measured_ns'] = 101
    if fault == 'episode': k['identity'] = (0, 0, 1)
    if fault == 'digest': r = replace(r, setup_evidence_sha256='1' * 64)
    with pytest.raises(ValueError): check_setup_snapshot(v, r, **k)


@pytest.mark.parametrize('fault', ['speed', 'interior_wall', 'touching_wall', 'body_outside'])
def test_setup_reports_scientific_failure_without_resizing_or_relaxation(fault):
    from dataclasses import replace
    v, r, k = inputs()
    if fault == 'speed': k['velocity_world_m_s'] = [.021, 0., 0.]
    if fault == 'interior_wall': k['native_static_boxes'][0]['native_position'][0] = 0.
    if fault == 'touching_wall': k['native_static_boxes'][0]['native_position'][0] = 1.05
    if fault == 'body_outside': r = replace(r, lower_initial_body_m=(-.1, -.1, -.1), upper_initial_body_m=(.1, .1, .1))
    assert not check_setup_snapshot(v, r, **k)['velocity_and_nonfloor_setup_checks_pass']


def support_inputs():
    _, _, k = inputs(); groups = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
    contacts = [dict(force_status='measured', force_on_robot_world_n=[0., 0., 30.], robot_link_name=g,
                     environment_link_id=0, disallowed=False) for g in groups]
    return contacts, dict(expected_support_groups=groups, ground_link_ids=[0], geometry=k['geometry'],
        joint_position=k['joint_position'], position_world_m=k['position_world_m'],
        rotation_world_from_body=k['rotation_world_from_initial_body'])


def test_native_support_witness_does_not_identify_foot_geoms_or_allow_future_contact():
    c, k = support_inputs(); r = initial_ground_support_witness(c, **k)
    assert r['initial_native_support_witness_present']
    assert not r['exact_foot_collision_geom_identity_verified'] and not r['contact_permitted']
    assert not r['contact_model_validated'] and not r['navigation_qualified']


@pytest.mark.parametrize('fault', ['missing_group', 'zero_force', 'down_force', 'body_contact', 'other_object', 'nonfoot_below'])
def test_native_support_witness_retains_negative_cases(fault):
    c, k = support_inputs()
    if fault == 'missing_group': c.pop()
    if fault == 'zero_force': c[0]['force_on_robot_world_n'] = [0., 0., 0.]
    if fault == 'down_force': c[0]['force_on_robot_world_n'][2] = -1.
    if fault == 'body_contact': c[0]['robot_link_name'] = 'base'; c[0]['disallowed'] = True
    if fault == 'other_object': c[0]['environment_link_id'] = 1
    if fault == 'nonfoot_below': k['position_world_m'][2] = .1
    assert not initial_ground_support_witness(c, **k)['initial_native_support_witness_present']


def test_missing_native_force_is_not_inferred_support():
    c, k = support_inputs(); c[0]['force_status'] = 'unavailable'
    with pytest.raises(ValueError): initial_ground_support_witness(c, **k)


@pytest.mark.parametrize('fault', ['force_overflow', 'boolean_environment', 'nonboolean_flag'])
def test_support_witness_rejects_unrepresentable_force_or_ambiguous_identity(fault):
    c, k = support_inputs()
    if fault == 'force_overflow': c[0]['force_on_robot_world_n'] = [1e308, 1e308, 1e308]
    if fault == 'boolean_environment': c[0]['environment_link_id'] = False
    if fault == 'nonboolean_flag': c[0]['disallowed'] = 0
    with np.errstate(over='ignore', invalid='ignore'), pytest.raises(ValueError):
        initial_ground_support_witness(c, **k)
