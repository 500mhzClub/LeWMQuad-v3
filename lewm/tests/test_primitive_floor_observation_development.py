import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import locate_floor_patches
from lewm.primitive_floor_observation_development import observe_primitive_floor
from lewm.primitive_floor_relation_development import assess_primitive_floor_relation
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.fixture(scope='module')
def setup():
    model = ArticulatedCollisionGeometry(URDF)
    q = np.repeat([0., .8, -1.5], 4)
    errors = {r['shape_id']: .001 for r in model.supports(q, np.eye(3))['shapes']}
    d, valid = scene()
    seed = locate_floor_patches(d, valid, [[1.5, 0., -.3]], [0., 0., 1.])['cells_rc'][0]
    return model, q, errors, seed


def observe(setup, d, valid, *, t=(1.5, 0., 0.), seed=None, R=None):
    model, q, errors, original_seed = setup
    return observe_primitive_floor(model, q, d, valid, [0., 0., 1.], original_seed if seed is None else seed,
                                    rotation_observation_from_body=np.eye(3) if R is None else R,
                                    translation_observation_from_body=t, normal_error=.002, up_error=.001,
                                    plane_offset_error=.001, point_error_by_shape=errors)


def test_entire_primitive_footprints_observed_but_not_other_obstacles_or_contact(setup):
    d, valid = scene(); result = observe(setup, d, valid)
    assert len(result['floor_coverage']) == 27 and all(result['floor_coverage'].values())
    assert result['covered_cells'].min() > 1
    base = next(r for r in result['gap_bounds']['primitives'] if r['shape_id'] == 'base:0')
    assert base['minimum_gap_lower_m'] > .06  # Not rejected for being far ABOVE floor.
    assessment = assess_primitive_floor_relation(result['gap_bounds'], floor_coverage=result['floor_coverage'],
                                                  non_floor_clearance=dict.fromkeys(result['floor_coverage'], False))
    assert not assessment['all_primitives_conditionally_clear']
    assert not result['non_floor_clearance_established'] and not result['contact_permitted']


@pytest.mark.parametrize('fault', ['missing', 'step'])
def test_observed_plane_seed_cannot_bridge_missing_or_non_ground_footprint(setup, fault):
    d, valid = scene(); before = observe(setup, d, valid)
    names = list(before['floor_coverage']); i = names.index('FL_foot:0')
    x, y = before['lower_cells_xy'][i] + [1, 1]
    assert (y, x) != tuple(setup[3])
    if fault == 'missing': d[y, x] = 0.; valid[y, x] = False
    else: d[y, x] += .03
    after = observe(setup, d, valid)
    assert not after['floor_coverage']['FL_foot:0'] and after['invalid_cells'][i] > 0


def test_raising_floor_into_foot_is_penetration_not_contact_permission(setup):
    d, valid = scene(); result = observe(setup, d, valid, t=(1.5, 0., -.1))
    assessment = assess_primitive_floor_relation(result['gap_bounds'], floor_coverage=result['floor_coverage'],
                                                  non_floor_clearance=dict.fromkeys(result['floor_coverage'], True))
    feet = [r for r in assessment['primitives'] if r['shape_id'].endswith('_foot:0')]
    assert all(r['penetration_under_every_supplied_model'] for r in feet)
    assert all(not r['conditional_clearance'] and not r['contact_permitted'] for r in feet)
    assert not assessment['all_primitives_conditionally_clear']


def test_plane_cannot_supply_unobserved_behind_camera_coverage(setup):
    d, valid = scene(); result = observe(setup, d, valid, t=(-1., 0., 0.))
    assert not any(result['floor_coverage'].values())


@pytest.mark.parametrize('fault', ['missing_seed', 'wall_seed', 'float_seed', 'out_of_image', 'bad_rotation'])
def test_seed_and_sensor_transform_fail_closed(setup, fault):
    d, valid = scene(); seed = setup[3].copy(); R = np.eye(3)
    if fault == 'missing_seed': d[tuple(seed)] = 0.; valid[tuple(seed)] = False
    if fault == 'wall_seed': seed = np.array([100, 320])
    if fault == 'float_seed': seed = seed.astype(float)
    if fault == 'out_of_image': seed = np.array([479, 0])
    if fault == 'bad_rotation': R[0, 0] = -1
    with pytest.raises(SensorContractError): observe(setup, d, valid, seed=seed, R=R)
