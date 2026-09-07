"""Independent frame/sign checks for the new physical gap sensitivity query."""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.paired_rgbd_physical_plane_development import minimum_gaps
from lewm.primitive_floor_relation_development import primitive_floor_gap_bounds
from lewm.relative_gyro_turn_development import rotation_increment
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_physical_gaps_are_invariant_to_observation_reference_change():
    geometry = ArticulatedCollisionGeometry(URDF); q = np.repeat([0., .8, -1.5], 4)
    a = np.array([1.3, .1, -.3]); n = np.array([.03, -.02, 1.]); n /= np.linalg.norm(n)
    R = rotation_increment([.1, -.02, .4]); p = np.array([1., .05, .02])
    Q = rotation_increment([-.2, .04, -.7]); t = np.array([.3, -.5, .1])
    reference = minimum_gaps(geometry, q, a, n, R, p)
    changed = minimum_gaps(geometry, q, Q@a+t, Q@n, Q@R, Q@p+t)
    np.testing.assert_allclose(changed, reference, atol=1e-12, rtol=0)


def test_translation_sign_and_exact_unpadded_support_match_existing_bounds():
    geometry = ArticulatedCollisionGeometry(URDF); q = np.repeat([0., .8, -1.5], 4)
    a = np.array([1., .1, -.3]); n = np.array([0., 0., 1.]); R = np.eye(3); p = np.array([1., 0., .01])
    reference = minimum_gaps(geometry, q, a, n, R, p)
    lifted = minimum_gaps(geometry, q, a, n, R, p+[0., 0., .02])
    np.testing.assert_allclose(lifted-reference, .02, atol=1e-12, rtol=0)
    ids = [s['shape_id'] for s in geometry.supports(q, np.eye(3))['shapes']]
    bounds = primitive_floor_gap_bounds(geometry, q, a-p, n, normal_error=0., plane_offset_error=0.,
                                       point_error_by_shape={sid: 0. for sid in ids})
    np.testing.assert_allclose(reference, [r['nominal_minimum_gap_m'] for r in bounds['primitives']], atol=1e-12, rtol=0)
    assert all(r['padding_m'] == .04 for r in bounds['primitives'])
