import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_articulated_scan_geometry_development_v1 import contact_geometry, URDF


def test_contact_plane_separates_torso_from_articulated_rigid_group():
    model = ArticulatedCollisionGeometry(URDF)
    spec = {'geometry': {'wall_boxes': [{'wall_id': 'south', 'centre_xyz': [0, -.49, .3],
                                       'size_xyz': [1, .08, .6], 'yaw_rad': 0.}]}}
    contact = {'environment_object_id': 'south', 'robot_link_name': 'RR_calf', 'position_world_m': [0, -.45, .2]}
    q = np.array([0, 0, 0, -.5, 0, 0, 0, 0, 0, 0, 0, 0.])
    result = contact_geometry(spec, [0, 0, .3, 0, 0, 0, 1], q, contact, model)
    assert result['native_contact_plane_residual_m'] == pytest.approx(0)
    assert result['torso_only_plane_gap_m'] == pytest.approx(.45 - .0935/2)
    assert result['whole_robot_plane_gap_m'] < result['torso_only_plane_gap_m']
    assert set(result['native_group_shape_ids']) == {'RR_calf:0', 'RR_calflower:0', 'RR_calflower1:0'}
    assert not result['future_swept_volume_qualified'] and not result['runtime_environment_input']


def test_nonwall_contact_not_silently_treated_as_clearance_evidence():
    with pytest.raises(ValueError):
        contact_geometry({'geometry': {'wall_boxes': []}}, [0,0,.3,0,0,0,1], np.zeros(12),
                         {'environment_object_id': 'ground_plane'}, ArticulatedCollisionGeometry(URDF))
