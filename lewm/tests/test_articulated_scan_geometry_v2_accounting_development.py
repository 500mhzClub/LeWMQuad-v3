import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_collision_grouping_development import resolve_native_groups
from scripts.analyze_go2_articulated_scan_geometry_development_v2 import contact_geometry, URDF
from lewm.tests.test_native_collision_grouping_development import NATIVE


def test_reported_calf_group_includes_foot_without_changing_support_geometry():
    model = ArticulatedCollisionGeometry(URDF)
    shapes = model.supports(np.zeros(12), np.eye(3))['shapes']
    groups = resolve_native_groups(URDF, shapes, NATIVE)
    spec = {'geometry': {'wall_boxes': [{'wall_id': 'south', 'centre_xyz': [0, -.49, .3],
                                       'size_xyz': [1, .08, .6], 'yaw_rad': 0.}]}}
    contact = {'environment_object_id': 'south', 'robot_link_name': 'RR_calf', 'position_world_m': [0, -.45, .2]}
    result = contact_geometry(spec, [0,0,.3,0,0,0,1], np.zeros(12), contact, model, groups)
    assert set(result['native_group_shape_ids']) == {'RR_calf:0','RR_calflower:0','RR_calflower1:0','RR_foot:0'}
    assert not result['runtime_environment_input'] and not result['future_swept_volume_qualified']
