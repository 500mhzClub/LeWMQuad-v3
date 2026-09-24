"""Synthetic sensor-owned geometry and custody checks; no recorded data."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import FOCAL, from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory
from lewm.rgbd_physical_configuration_evidence_development import RGBDPhysicalConfigurationEvidence, frame_identity
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth
from lewm.tests.test_rgbd_inertial_fusion_development import prior, hypotheses
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same

ERRORS = dict(normal_error=.002, up_error=.001, plane_offset_error=.001, range_error_m=.001)
Q = np.repeat([0., .8, -1.5], 4)


def ready(kind='room', count=3):
    owner = RGBDInertialRayMemory(prior=prior(), hypotheses=hypotheses())
    for _, p, f, _ in stream(count):
        now = p['sensor_state']['decision_ns']; depth = room_depth(p)
        if kind == 'missing':
            depth['depth_m'][300:460, 260:380] = 0.; depth['valid'][300:460, 260:380] = False
        elif kind == 'vertical':
            depth = from_native_depth(np.full((480, 640), 2., np.float32), p,
                measured_ns=now, available_ns=now, now_ns=now)
        elif kind == 'elevated':
            # A distinct upward plane at body z=-.2, not a world-ground label.
            y = (np.arange(480)-239.5)/FOCAL
            raw = np.minimum(depth['depth_m'], np.where(y[:, None] > 0, .243/np.maximum(y[:, None], 1e-12), 200.))
            depth = from_native_depth(raw.astype(np.float32), p, measured_ns=now, available_ns=now, now_ns=now)
        elif kind == 'low_obstacle':
            # A local horizontal return 3 cm above the room floor; the rest
            # of the observed floor remains a distinct measured plane family.
            rows = np.arange(335, 371)
            depth['depth_m'][335:371, 280:360] = (.313*FOCAL/(rows-239.5))[:, None]
        owner.observe(p, depth, f, now_ns=now)
    model = RGBDPhysicalConfigurationEvidence(owner, ArticulatedCollisionGeometry(URDF), **ERRORS)
    model.refresh(now_ns=now)
    return model, now


def query(model, now, offset=(1.3, 0., 0.), backend='compiled', error=0.):
    return model.query(offset, np.eye(3), Q, error, now_ns=now, backend=backend)


def row(result, name):
    return next(r for r in result['primitives'] if r['shape_id'] == name)


@pytest.mark.parametrize('offset', [(0., 0., 0.), (1.3, 0., 0.), (2.2, 0., 0.), (1.3, 0., -.1)])
def test_complete_output_reference_parity(offset):
    model, now = ready()
    same(query(model, now, offset), query(model, now, offset, backend='reference'))


def test_preparation_and_queries_never_reintegrate_or_reselect(monkeypatch):
    model, now = ready(); owner = model.owner
    before = deepcopy(owner.rays.fusion), owner.state.depth.orientation.samples_integrated
    frames = [frame_identity(f) for f in owner.rays.frames]
    def forbidden(*args, **kwargs): raise AssertionError('no observation/integration/plane reselection allowed')
    monkeypatch.setattr(owner, 'observe', forbidden)
    monkeypatch.setattr(type(next(iter(model.hypotheses.values()))), 'from_frame', forbidden)
    a = query(model, now)
    model.refresh(now_ns=now)
    b = model.query((1.3, 0., 0.), rotation_increment([0., 0., .3]), Q, .002, now_ns=now)
    assert owner.rays.fusion == before[0] and owner.state.depth.orientation.samples_integrated == before[1]
    assert frames == [frame_identity(f) for f in owner.rays.frames]
    for result in (a, b):
        assert len(result['primitives']) == 27
        assert not any(result[k] for k in ('sensor_owner_reintegrated', 'initial_setup_region_used',
            'navigation_action_permitted', 'ground_support_permission', 'future_gait_qualified',
            'continuous_swept_volume_established', 'configuration_is_execution_prediction'))


def test_missing_view_cannot_borrow_initial_setup_region():
    model, now = ready()
    for offset in ((0., 0., 0.), (-1., 0., 0.), (3., 0., 0.)):
        result = query(model, now, offset)
        assert not row(result, 'base:0')['conditional_nonfloor_clearance']
        assert not result['initial_setup_region_used']
    model, now = ready('missing')
    assert not row(query(model, now), 'base:0')['conditional_nonfloor_clearance']


def test_observed_wall_is_a_nonfloor_veto():
    model, now = ready(); result = query(model, now, (2.2, 0., 0.))
    base = row(result, 'base:0')
    assert base['nonfloor_conflict_sources'] and not base['conditional_nonfloor_clearance']


def test_low_horizontal_obstacle_is_not_erased_by_upward_normal():
    model, now = ready('low_obstacle', count=1)
    result = query(model, now, (1.4, 0., -.02))
    assert any(r['nonfloor_conflict_sources'] for r in result['primitives'])
    assert not result['all_primitives_conditionally_nonfloor_clear']


def test_owner_advance_requires_refresh_without_resetting_existing_bindings():
    data = list(stream(3)); owner = RGBDInertialRayMemory(prior=prior(), hypotheses=hypotheses())
    _, p, f, _ = data[0]; now = p['sensor_state']['decision_ns']
    owner.observe(p, room_depth(p), f, now_ns=now)
    model = RGBDPhysicalConfigurationEvidence(owner, ArticulatedCollisionGeometry(URDF), **ERRORS)
    model.refresh(now_ns=now); first = dict(model.bindings)
    for _, p, f, _ in data[1:]:
        now = p['sensor_state']['decision_ns']; owner.observe(p, room_depth(p), f, now_ns=now)
        model.refresh(now_ns=now)
        assert all(model.bindings[t] == b for t, b in first.items())
    assert owner.state.depth.orientation.samples_integrated == 100
    assert not query(model, now)['sensor_owner_reintegrated']


def test_elevated_plane_and_penetration_not_relabelled_free_space():
    model, now = ready('elevated')
    result = query(model, now, (1.3, 0., -.2))
    bad = [r for r in result['primitives'] if r['ground']['observed_penetration_sources']]
    assert bad and any(r['conditional_nonfloor_clearance'] for r in bad)
    assert all(not r['conditional_observed_separation'] and not r['contact_candidate_with_nonfloor_clearance'] for r in bad)
    assert not result['navigation_action_permitted']


def test_exact_foot_contact_candidates_do_not_exempt_calves_or_grant_permission():
    model, now = ready()
    shapes = model.geometry.supports(Q, np.eye(3))['shapes']
    height = -.3-min(s['lower'][2] for s in shapes if s['shape_id'] in FOOT_SHAPES)
    result = query(model, now, (1.3, 0., float(height)))
    candidates = [r for r in result['primitives'] if r['ground']['observed_foot_contact_candidate']]
    assert candidates and all(r['shape_id'] in FOOT_SHAPES for r in candidates)
    assert all(not r['contact_permitted'] for r in result['primitives'])
    below = query(model, now, (1.3, 0., float(height-.1)))
    assert any(r['ground']['observed_penetration_sources'] for r in below['primitives'])


def test_vertical_scene_has_no_floor_support_or_plane_exemption():
    model, now = ready('vertical', count=1)
    result = query(model, now, (2.2, 0., 0.))
    assert all(h.cell_rc is None for h in model.hypotheses.values())
    assert all(r['ground']['status'] == 'UNKNOWN_FLOOR_COVERAGE' for r in result['primitives'])
    assert row(result, 'base:0')['nonfloor_conflict_sources']


@pytest.mark.parametrize('fault', ['stale', 'owner_failed', 'raw_binding', 'pose_binding', 'plane_binding',
    'pending', 'reflection', 'nan', 'negative_error', 'nonfinite_error', 'bad_backend'])
def test_faults_latch_only_adapter_and_never_recover(fault):
    model, now = ready(); offset = (1.3, 0., 0.); R = np.eye(3); error = 0.; backend = 'compiled'
    if fault == 'stale': now -= 1
    elif fault == 'owner_failed': model.owner.failed = True
    elif fault == 'raw_binding':
        f = model.owner.rays.frames[0]; f['evidence'] = dict(f['evidence']) | {'depth': np.zeros((480, 640), np.float32)}
    elif fault == 'pose_binding': model.owner.rays.frames[0]['position'][0] += .01
    elif fault == 'plane_binding':
        t = next(iter(model.hypotheses)); model.hypotheses[t] = replace(model.hypotheses[t], depth_sha256='b'*64)
    elif fault == 'pending': model.owner._reader.pending = ({}, {}, {})
    elif fault == 'reflection': R[0, 0] = -1
    elif fault == 'nan': offset = (np.nan, 0., 0.)
    elif fault == 'negative_error': error = -.1
    elif fault == 'nonfinite_error': error = np.inf
    elif fault == 'bad_backend': backend = 'fast'
    with pytest.raises(SensorContractError): model.query(offset, R, Q, error, now_ns=now, backend=backend)
    assert model.failed
    with pytest.raises(SensorContractError): model.refresh(now_ns=model.owner.last_ns)
    assert model.owner.failed == (fault == 'owner_failed')


def test_query_before_prepare_is_rejected():
    prepared, now = ready()
    model = RGBDPhysicalConfigurationEvidence(prepared.owner, prepared.geometry, **ERRORS)
    with pytest.raises(SensorContractError): query(model, now)


def test_virtual_vertex_error_is_not_confused_with_physical_padding():
    model, now = ready(); result = query(model, now, error=.01)
    for primitive in result['primitives']:
        witness = next(w for w in primitive['ground_witnesses'] if w['measured_ns'] == now)
        if witness['gap'] is None: continue
        assert witness['gap']['point_error_m'] == .01
        assert witness['gap']['padding_m'] == .04
        width = (np.asarray(witness['plane_vertex_residual_upper_m'])-witness['plane_vertex_residual_lower_m'])/2
        assert np.all(width >= .001+1.002*np.sqrt(3)*.01)
