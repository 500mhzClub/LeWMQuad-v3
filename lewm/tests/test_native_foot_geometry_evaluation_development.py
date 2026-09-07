from copy import deepcopy

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.native_collision_grouping_development import resolve_native_groups
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def fixture():
    geometry = ArticulatedCollisionGeometry(URDF); q = np.repeat([0., .8, -1.5], 4)
    shapes = geometry.supports(q, np.eye(3))['shapes']; pose = np.array([0., 0., .33, 0., 0., 0., 1.])
    groups = resolve_native_groups(URDF, shapes, ['base'] + [leg + suffix for leg in ('FL', 'FR', 'RL', 'RR') for suffix in ('_hip', '_thigh', '_calf')])
    rows = []
    for i, (s, primitive) in enumerate(zip(shapes, geometry._shapes, strict=True)):
        data = np.zeros(7)
        if s['kind'] == 'sphere': data[0] = primitive['dimensions'][0]
        rows.append(dict(geom_id=100+i, link_id=200+i, link_name=groups[s['shape_id']],
            geom_type={'sphere': 'SPHERE', 'box': 'BOX', 'cylinder': 'MESH'}[s['kind']], data=data.tolist(),
            position_world_m=(pose[:3] + s['center_body_m']).tolist()))
    return rows, geometry, q, pose


def test_native_foot_identity_uses_data_and_pose_not_geometry_order():
    rows, g, q, pose = fixture(); a = match_native_foot_geometries(rows, g, q, pose)
    rows.reverse(); b = match_native_foot_geometries(rows, g, q, pose)
    assert a == b and len(a['native_foot_geom_to_shape']) == 4
    assert not a['contact_permitted'] and not a['contact_model_validated']


@pytest.mark.parametrize('fault', ['missing', 'duplicate_id', 'radius', 'reserved', 'centre', 'group', 'type'])
def test_bad_native_foot_binding_is_rejected(fault):
    rows, g, q, pose = fixture(); foot = next(r for r in rows if r['link_name'] == 'FL_calf' and r['geom_type'] == 'SPHERE')
    if fault == 'missing': rows.pop()
    if fault == 'duplicate_id': rows[1]['geom_id'] = rows[0]['geom_id']
    if fault == 'radius': foot['data'][0] = .023
    if fault == 'reserved': foot['data'][1] = .001
    if fault == 'centre': foot['position_world_m'][0] += .001
    if fault == 'group': foot['link_name'] = 'FR_calf'
    if fault == 'type': foot['geom_type'] = 'MESH'
    with pytest.raises(ValueError): match_native_foot_geometries(rows, g, q, pose)


@pytest.mark.parametrize('reverse', [False, True])
def test_nonfoot_contact_on_same_support_group_is_not_exempt(reverse):
    packet = dict(geom_a=np.array([11, 12, 12, 12]), geom_b=np.array([0, 0, 0, 0]),
        valid_mask=np.array([True, True, True, False]),
        force_a=np.array([[0., 0., 20.], [0., 0., 3.], [0., 0., 0.], [0., 0., 5.]]),
        force_b=np.array([[0., 0., -20.], [0., 0., -3.], [0., 0., 0.], [0., 0., -5.]]))
    if reverse:
        packet['geom_a'], packet['geom_b'] = packet['geom_b'], packet['geom_a']
        packet['force_a'], packet['force_b'] = packet['force_b'], packet['force_a']
    assert nonfoot_ground_contact_indices(packet, robot_geom_ids=[11, 12], foot_geom_ids=[11], ground_geom_ids=[0]) == [1]


@pytest.mark.parametrize('fault', [None, 'batch', 'nonfinite'])
def test_native_capture_reads_one_environment_and_preserves_material_identity(fault):
    from types import SimpleNamespace
    from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
    geom = SimpleNamespace(idx=12, link=SimpleNamespace(idx=8, name='FL_calf'), type=SimpleNamespace(name='SPHERE'),
        data=np.r_[.022, np.zeros(6)], friction=.6, sol_params=np.array([.02, 1., .9]),
        get_pos=lambda: np.array([[.1, .2, .3]]), get_quat=lambda: np.array([[1., 0., 0., 0.]]))
    if fault == 'batch': geom.get_pos = lambda: np.zeros((2, 3))
    if fault == 'nonfinite': geom.sol_params[0] = np.nan
    robot = SimpleNamespace(geoms=[geom])
    if fault:
        with pytest.raises(ValueError): capture_native_robot_geometry(robot)
    else:
        row = capture_native_robot_geometry(robot)[0]
        assert row['geom_id'] == 12 and row['link_name'] == 'FL_calf'
        assert row['data'] == geom.data.tolist() and row['solver_parameters'] == geom.sol_params.tolist()
        assert row['friction'] == .6 and row['position_world_m'] == [.1, .2, .3]
