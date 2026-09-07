from copy import deepcopy
import json

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_ground_plane_development import foot_sphere_centres_body
from lewm.causal_sensor_state import SensorContractError
from lewm.primitive_floor_relation_development import (
    FOOT_SHAPES, primitive_floor_gap_bounds, assess_primitive_floor_relation)
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.fixture(scope='module')
def geometry():
    return ArticulatedCollisionGeometry(URDF)


def calculate(geometry, *, anchor=(0., 0., -.35), normal=(0., 0., 1.), q=None,
              en=0., eb=0., ep=0., padding=.04):
    q = np.repeat([0., .8, -1.5], 4) if q is None else q
    ids = [r['shape_id'] for r in geometry.supports(q, np.eye(3))['shapes']]
    return primitive_floor_gap_bounds(geometry, q, anchor, normal, normal_error=en,
                                     plane_offset_error=eb, point_error_by_shape=dict.fromkeys(ids, ep),
                                     padding_m=padding)


def test_foot_minimum_matches_independent_kinematics_minus_radius(geometry):
    q = np.repeat([.03, .8, -1.5], 4)
    n = np.array([.03, -.02, 1.]); n /= np.linalg.norm(n)
    anchor = np.array([.2, .1, -.35])
    result = calculate(geometry, q=q, anchor=anchor, normal=n)
    feet = foot_sphere_centres_body(q)
    lookup = {r['shape_id']: r for r in result['primitives']}
    for i, leg in enumerate(('FL', 'FR', 'RL', 'RR')):
        expected = n @ (feet[i] - anchor) - .022
        row = lookup[leg + '_foot:0']
        assert row['nominal_minimum_gap_m'] == pytest.approx(expected, abs=1e-12)
        assert row['minimum_gap_lower_m'] <= expected <= row['minimum_gap_upper_m']
    assert {r['shape_id'] for r in result['primitives'] if r['contact_candidate_geometry']} == FOOT_SHAPES
    assert not result['contact_permitted'] and not result['floor_coverage_established']


def test_primitive_minimum_is_not_primitive_top_or_padded_sample(geometry):
    result = calculate(geometry)
    base = next(r for r in result['primitives'] if r['shape_id'] == 'base:0')
    assert base['minimum_gap_upper_m'] - base['minimum_gap_lower_m'] < 1e-10
    padded = calculate(geometry, padding=.2)
    for before, after in zip(result['primitives'], padded['primitives'], strict=True):
        assert before['minimum_gap_lower_m'] == after['minimum_gap_lower_m']
        assert before['minimum_gap_upper_m'] == after['minimum_gap_upper_m']
        assert after['padded_minimum_gap_lower_m'] == pytest.approx(before['padded_minimum_gap_lower_m'] - .16)


def test_public_bounds_and_assessments_are_json_serializable(geometry):
    bounds = calculate(geometry, en=.002, eb=.001, ep=.02)
    ids = [r['shape_id'] for r in bounds['primitives']]
    result = assess_primitive_floor_relation(bounds, floor_coverage=dict.fromkeys(ids, False),
                                             non_floor_clearance=dict.fromkeys(ids, False))
    assert json.loads(json.dumps(bounds)) == bounds
    assert json.loads(json.dumps(result)) == result


def test_minimum_gap_enclosure_contains_perturbed_exact_supports(geometry):
    rng = np.random.default_rng(6418); q = rng.uniform(-.4, .4, 12)
    a = np.array([.1, -.03, -.3]); n = np.array([.02, -.01, 1.]); n /= np.linalg.norm(n)
    en, eb, ep = .015, .002, .004
    result = calculate(geometry, q=q, anchor=a, normal=n, en=en, eb=eb, ep=ep)
    lower = np.array([r['minimum_gap_lower_m'] for r in result['primitives']])
    upper = np.array([r['minimum_gap_upper_m'] for r in result['primitives']])
    for _ in range(200):
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        nn = n + .5 * en * direction; nn /= np.linalg.norm(nn)
        assert np.linalg.norm(nn - n) <= en
        shift = rng.normal(size=3); shift *= ep / np.linalg.norm(shift)
        offset = rng.choice([-eb, eb])
        exact = np.array([r['lower'][0] for r in geometry.supports(q, nn[None])['shapes']])
        exact += nn @ (shift - a) - offset
        assert (exact >= lower).all() and (exact <= upper).all()


def test_normal_fixed_translation_offset_extrema_are_attained(geometry):
    result = calculate(geometry, ep=.01, eb=.002)
    for row in result['primitives']:
        assert row['physical_error_allowance_m'] == pytest.approx(.012)
        assert row['minimum_gap_lower_m'] == pytest.approx(row['nominal_minimum_gap_m'] - .012, abs=2e-12)
        assert row['minimum_gap_upper_m'] == pytest.approx(row['nominal_minimum_gap_m'] + .012, abs=2e-12)


def sample(sid='base:0', *, lo=.01, hi=.02, link=None, kind='box'):
    return {'shape_id': sid, 'link': sid.split(':')[0] if link is None else link, 'kind': kind,
            'minimum_gap_lower_m': lo, 'minimum_gap_upper_m': hi,
            'contact_candidate_geometry': True}  # Deliberately untrusted flag.


def assess(rows, floor=True, other=True):
    return assess_primitive_floor_relation({'primitives': rows},
        floor_coverage={r['shape_id']: floor for r in rows}, non_floor_clearance={r['shape_id']: other for r in rows})


def test_more_height_above_floor_cannot_turn_separation_into_contact():
    for lo, hi in ((.001, .003), (.017, .064), (.5, .7)):
        result = assess([sample(lo=lo, hi=hi)])
        assert result['all_primitives_conditionally_clear']
        assert not result['primitives'][0]['possible_foot_contact']
        assert not result['navigation_qualified'] and not result['evidence_provenance_validated']


@pytest.mark.parametrize('sid,kind', [('base:0', 'box'), ('FL_calf:0', 'cylinder'), ('FL_foot:0', 'sphere')])
def test_certain_penetration_never_permitted_even_for_foot(sid, kind):
    result = assess([sample(sid, lo=-.02, hi=-.001, kind=kind)])
    row = result['primitives'][0]
    assert row['penetration_under_every_supplied_model']
    assert not row['conditional_clearance'] and not row['contact_permitted']
    assert row['status'] == 'PENETRATION_UNDER_EVERY_SUPPLIED_MODEL'


def test_foot_contact_candidate_is_not_clearance_or_contact_permission():
    result = assess([sample('FL_foot:0', lo=-.001, hi=.001, kind='sphere')])
    row = result['primitives'][0]
    assert row['possible_foot_contact'] and row['status'] == 'FOOT_CONTACT_CANDIDATE_ONLY'
    assert not row['conditional_clearance'] and not row['contact_permitted']


@pytest.mark.parametrize('sid,kind,link', [('FL_calf:0', 'cylinder', None), ('envelope:0', 'sphere', None),
    ('FL_foot:0', 'box', None), ('FL_foot:0', 'sphere', 'base')])
def test_generic_ground_flag_calf_or_fake_foot_cannot_confer_contact_role(sid, kind, link):
    result = assess([sample(sid, lo=-.001, hi=.001, kind=kind, link=link)])
    assert result['primitives'][0]['status'] == 'NON_CONTACT_INTERSECTION_POSSIBLE'
    assert not result['primitives'][0]['possible_foot_contact']


def test_mixed_link_group_cannot_inherit_foot_exception():
    result = assess([sample('FL_foot:0', lo=.001, hi=.003, kind='sphere'),
                     sample('FL_calf:0', lo=-.002, hi=.002, kind='cylinder')])
    assert result['primitives'][0]['conditional_clearance']
    assert not result['primitives'][1]['conditional_clearance']
    assert not result['all_primitives_conditionally_clear']


@pytest.mark.parametrize('floor,other', [(False, True), (True, False), (False, False)])
def test_missing_floor_or_wall_step_unknown_nonfloor_cannot_be_cleared(floor, other):
    result = assess([sample(lo=.1, hi=.2)], floor, other)
    assert result['primitives'][0]['physical_floor_separated_under_supplied_model']
    assert not result['primitives'][0]['conditional_clearance']
    assert not result['all_primitives_conditionally_clear']


@pytest.mark.parametrize('fault', ['missing_identity', 'numeric_bool', 'extra_identity', 'reverse', 'nan', 'duplicate'])
def test_invalid_evidence_and_gap_bounds_fail_closed(fault):
    rows = [sample()]; floor = {'base:0': True}; other = floor.copy()
    if fault == 'missing_identity': floor.clear()
    if fault == 'numeric_bool': floor['base:0'] = 1
    if fault == 'extra_identity': other['FL_foot:0'] = True
    if fault == 'reverse': rows[0]['minimum_gap_upper_m'] = -.1
    if fault == 'nan': rows[0]['minimum_gap_lower_m'] = np.nan
    if fault == 'duplicate': rows.append(deepcopy(rows[0]))
    with pytest.raises(SensorContractError):
        assess_primitive_floor_relation({'primitives': rows}, floor_coverage=floor, non_floor_clearance=other)


@pytest.mark.parametrize('fault', ['normal', 'negative', 'large_normal', 'nan', 'missing_point_error', 'overflow'])
def test_invalid_primitive_error_inputs_fail_closed(geometry, fault):
    q = np.zeros(12); n = np.array([0., 0., 1.]); a = np.array([0., 0., -.3]); en = 0.
    point_errors = {r['shape_id']: .001 for r in geometry.supports(q, np.eye(3))['shapes']}
    if fault == 'normal': n *= 2
    if fault == 'negative': en = -.001
    if fault == 'large_normal': en = 1.
    if fault == 'nan': a[0] = np.nan
    if fault == 'missing_point_error': point_errors.pop('base:0')
    if fault == 'overflow': a[0] = 1e308
    with pytest.raises(SensorContractError):
        primitive_floor_gap_bounds(geometry, q, a, n, normal_error=en, plane_offset_error=0.,
                                   point_error_by_shape=point_errors)
