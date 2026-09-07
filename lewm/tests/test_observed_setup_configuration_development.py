from dataclasses import replace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.observed_setup_configuration_development import inverse_pose_box, query_configuration, query_current_configuration
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_continuous_startup_handoff_development import frames
from lewm.tests.test_startup_observation_turn_development import components
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same


def ready(extent=1., missing_central_pixel=False):
    geometry, kwargs = components()
    kwargs['region_prior'] = replace(kwargs['region_prior'], lower_initial_body_m=(-extent,)*3, upper_initial_body_m=(extent,)*3)
    owner = ContinuousStartupHandoff(geometry, **kwargs)
    for p,d,f,now in frames(8):
        if missing_central_pixel:
            d['depth_m'][267,319] = 0.; d['valid'][267,319] = False
        assert not owner.observe(p,d,f,now_ns=now)['terminal']
    return owner, now


def query(owner, now, position=(.95, 0., 0.), **kwargs):
    return query_configuration(owner, position, np.eye(3), np.repeat([0., .8, -1.5], 4), .005,
                               now_ns=now, through_ns=now+100_000_000, **kwargs)


def by_name(result, name): return next(r for r in result['primitives'] if r['shape_id'] == name)


def test_initial_plus_observed_residual_clearance_is_bound_to_actual_views():
    owner, now = ready(); result = query(owner, now)
    base = by_name(result, 'base:0')
    assert base['supplied_clearance_used'] and base['conditional_nonfloor_clearance']
    assert base['partition_before_observed_veto']['requires_sensor_evidence_boxes']
    assert all(base['residual_clearance_sources'])
    assert all(v['full_queries_checked'] == 27 and len(v['depth_sha256']) == 64 for v in result['observation_bindings'])
    assert not result['navigation_action_permitted'] and not result['configuration_is_execution_prediction']
    assert not result['ground_support_permission'] and not result['continuous_swept_volume_established']


def test_current_configuration_can_use_setup_without_inventing_observed_coverage():
    owner, now = ready(); result = query_current_configuration(owner, now_ns=now)
    assert result['all_primitives_conditionally_nonfloor_clear']
    assert all(p['supplied_clearance_used'] for p in result['primitives'])
    assert all(not p['whole_box_clearance_sources'] for p in result['primitives'])


def test_observed_wall_vetoes_even_an_incorrectly_supplied_large_starting_region():
    # Deliberately false supplied condition: the synthetic room wall lies
    # inside this cube. Whole-query observed negatives must override it.
    owner, now = ready(3.); result = query(owner, now, (2.2, 0., 0.))
    base = by_name(result, 'base:0')
    assert base['partition_before_observed_veto']['entire_query_conditionally_setup_nonfloor_clear']
    assert base['obstacle_veto_sources'] and not base['supplied_clearance_used']
    assert not base['conditional_nonfloor_clearance']


def test_occluded_residual_is_unknown_not_clearance_from_absence_of_collision():
    owner, now = ready(); result = query(owner, now, (3., 0., 0.))
    base = by_name(result, 'base:0')
    assert not base['obstacle_veto_sources']
    assert not base['conditional_nonfloor_clearance'] and not any(base['residual_clearance_sources'])


def test_out_of_view_residual_cannot_be_approved_by_initial_region():
    owner, now = ready(); result = query(owner, now, (-1.3, 0., 0.))
    assert not result['all_primitives_conditionally_nonfloor_clear']
    assert not by_name(result, 'base:0')['conditional_nonfloor_clearance']


def test_expiring_prior_cannot_cover_a_query_even_with_healthy_memory():
    owner, now = ready()
    result = query_configuration(owner, [0.,0.,0.], np.eye(3), owner._memory._joints, .005,
                                 now_ns=now, through_ns=owner._region.valid_until_ns+1)
    assert not result['all_primitives_conditionally_nonfloor_clear']
    assert not any(r['supplied_clearance_used'] for r in result['primitives'])


def test_ground_relation_candidates_are_not_contact_permission_and_submerged_feet_fail():
    owner, now = ready(); q = np.repeat([0., .8, -1.5], 4)
    shapes = owner._memory._geometry.supports(q, np.eye(3))['shapes']
    height = -.3-min(s['lower'][2] for s in shapes if s['shape_id'] in FOOT_SHAPES)
    on_plane = query(owner, now, (1.3, 0., float(height)))
    below = query(owner, now, (1.3, 0., float(height-.10)))
    feet = [r for r in on_plane['primitives'] if r['shape_id'] in FOOT_SHAPES]
    assert any(r['observed_foot_contact_candidate'] for r in feet)
    assert not on_plane['ground_support_permission']
    bad = [r for r in below['primitives'] if r['shape_id'] in FOOT_SHAPES]
    assert any(r['floor_penetration_sources'] for r in bad)
    assert not any(r['conditional_nonfloor_clearance'] for r in bad if r['floor_penetration_sources'])


def test_compiled_and_reference_evidence_are_identical():
    owner, now = ready()
    same(query(owner, now, backend='compiled'), query(owner, now, backend='reference'))


@pytest.mark.parametrize('fault', ['stale', 'faulted_memory', 'depth_binding', 'bad_rotation', 'nonfinite', 'bad_error', 'bad_horizon'])
def test_stale_or_unbound_configuration_queries_fail_closed(fault):
    owner, now = ready(); rotation = np.eye(3); position = [.95, 0., 0.]; error = .005; through = now
    if fault == 'stale': now -= 100_000_000
    if fault == 'faulted_memory': owner._memory.failed = True
    if fault == 'depth_binding':
        stamp = owner._memory._rays.frames[0]['measured_ns']
        owner._memory._hypotheses[stamp] = replace(owner._memory._hypotheses[stamp], depth_sha256='a'*64)
    if fault == 'bad_rotation': rotation[0,0] = 2.
    if fault == 'nonfinite': position[0] = float('nan')
    if fault == 'bad_error': error = -.1
    if fault == 'bad_horizon': through = now-1
    with pytest.raises(SensorContractError):
        query_configuration(owner, position, rotation, np.zeros(12), error, now_ns=now, through_ns=through)


def test_inverse_pose_enclosure_contains_joint_translation_rotation_perturbations():
    rng = np.random.default_rng(609072)
    stored = dict(position=np.array([.1, -.2, .05]), rotation=rotation_increment([.1, -.2, .3]),
                  position_scale_m=.03, orientation_scale_rad=.05)
    box = [[.8, -.1, -.3], [1.2, .2, .1]]
    low, high, radius = inverse_pose_box(box, stored)
    assert radius > stored['position_scale_m']
    for _ in range(200):
        delta = rng.normal(size=3); delta *= .03*rng.random()/np.linalg.norm(delta)
        angle = rng.normal(size=3); angle *= .05*rng.random()/np.linalg.norm(angle)
        points = rng.uniform(box[0], box[1], (40,3))
        actual = (points-stored['position']-delta)@(stored['rotation']@rotation_increment(angle))
        assert np.all(actual >= low) and np.all(actual <= high)


def test_current_frame_queries_preserve_common_pose_cancellation_without_shrinking_setup_error(monkeypatch):
    import lewm.observed_setup_configuration_development as module
    owner, now = ready(); original = module.non_floor_box_evidence; calls = []
    def capture(frame, low, high, **kwargs):
        calls.append((frame.depth_sha256, np.array(low), np.array(high)))
        return original(frame, low, high, **kwargs)
    monkeypatch.setattr(module, 'non_floor_box_evidence', capture)
    # Raise only the present position proxy for this source-factor sensitivity
    # test. It must affect global setup use, not current-camera geometry.
    owner._memory._rays.fusion['position_error_scale_m'] = .07
    owner._memory._rays.latest_frame['position_scale_m'] = .07
    result = query_current_configuration(owner, now_ns=now)
    assert result['supplied_configuration']['initial_partition_point_error_m'] >= .07
    current_sha = owner._memory._prepared[now].depth_sha256
    latest = [c for c in calls if c[0] == current_sha]
    assert latest
    # At least one exact current full-query box encloses the base with only
    # geometry padding, not twice the global 70-mm translation proxy.
    base = owner._memory._geometry.supports(owner._memory._joints, np.eye(3))['shapes'][0]
    expected_low, expected_high = np.asarray(base['lower'])-.04, np.asarray(base['upper'])+.04
    assert any(any(np.allclose(lo, expected_low, atol=1e-12, rtol=0)
                   and np.allclose(hi, expected_high, atol=1e-12, rtol=0)
                   for lo,hi in zip(c[1], c[2], strict=True)) for c in latest)


def test_body_relative_supplied_configuration_has_same_reference_and_compiled_outcome():
    owner, now = ready()
    a = query(owner, now, reference='current_body', backend='compiled')
    b = query(owner, now, reference='current_body', backend='reference')
    same(a,b)
    assert a['supplied_configuration']['reference'] == 'current_body'
    assert a['supplied_configuration']['initial_partition_point_error_m'] > .005


def test_missing_depth_in_every_retained_view_keeps_required_residual_unknown():
    owner, now = ready(missing_central_pixel=True)
    result = query(owner, now); base = by_name(result,'base:0')
    assert base['partition_before_observed_veto']['setup_covered_boxes']
    assert not base['obstacle_veto_sources']
    assert not base['conditional_nonfloor_clearance']
    assert not all(base['residual_clearance_sources'])
