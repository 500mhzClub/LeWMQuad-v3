import math

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry, primitive_support_radius
from lewm.causal_ground_plane_development import foot_sphere_centres_body
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from lewm.tests.test_relative_gyro_turn_development import initialized, packet


@pytest.mark.parametrize('kind,size,expected', [('box', [2, 4, 6], [1, 2, 3]),
    ('sphere', [2], [2, 2, 2]), ('cylinder', [2, 6], [2, 2, 3])])
def test_axis_support_exact(kind, size, expected):
    assert primitive_support_radius(kind, size, np.eye(3)) == pytest.approx(expected)
    assert primitive_support_radius(kind, size, -2*np.eye(3)) == pytest.approx(2*np.array(expected))


def test_cylinder_oblique_support_against_explicit_surface_points():
    angle = np.linspace(0, 2*math.pi, 20001)
    points = np.concatenate([np.stack([.013*np.cos(angle), .013*np.sin(angle), np.full_like(angle, z)], axis=1) for z in (-.06, .06)])
    normal = np.array([[.3, -.4, .5]])
    predicted = primitive_support_radius('cylinder', [.013, .12], normal)[0]
    assert float(np.max(points @ normal[0])) == pytest.approx(predicted, abs=1e-9)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_complete_robot_and_independent_closed_form_foot_kinematics(seed):
    model = ArticulatedCollisionGeometry(URDF)
    q = np.random.default_rng(seed).uniform(-.5, .5, 12)
    result = model.supports(q, np.eye(3))
    assert len(result['shapes']) == 27
    lookup = {r['shape_id']: r for r in result['shapes']}
    feet = np.array([lookup[leg + '_foot:0']['center_body_m'] for leg in ('FL', 'FR', 'RL', 'RR')])
    assert np.allclose(feet, foot_sphere_centres_body(q), rtol=0, atol=1e-12)
    for leg in ('FL', 'FR', 'RL', 'RR'):
        assert lookup[leg+'_calflower:0']['urdf_rigid_group'] == leg+'_calf'
        assert lookup[leg+'_calflower1:0']['urdf_rigid_group'] == leg+'_calf'
        assert lookup[leg+'_foot:0']['urdf_rigid_group'] == leg+'_foot'
    assert not result['future_swept_volume_qualified'] and not result['environment_clearance_qualified']


def test_runtime_boundary_requires_fresh_measured_joints_not_oracle_pose():
    model = ArticulatedCollisionGeometry(URDF)
    p = packet(initialized(), 80)
    result = model.observe(p, now_ns=1_600_000_000)
    assert result['decision_ns'] == 1_600_000_000 and len(result['shapes']) == 27
    with pytest.raises(ValueError): model.observe(p, now_ns=1_600_000_001)
    p['world_pose'] = [0, 0, 0]
    with pytest.raises(ValueError): model.observe(p, now_ns=1_600_000_000)


@pytest.mark.parametrize('kind,size,normals', [('mesh', [1], [[1,0,0]]), ('sphere', [-1], [[1,0,0]]),
    ('box', [1,2], [[1,0,0]]), ('cylinder', [1,2], [[0,0,0]]), ('box', [1,2,3], [[1,math.nan,0]])])
def test_invalid_support_not_silently_certified(kind, size, normals):
    with pytest.raises(ValueError): primitive_support_radius(kind, size, normals)
