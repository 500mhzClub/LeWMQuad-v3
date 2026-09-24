import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_collision_grouping_development import resolve_native_groups
from scripts.analyze_go2_ground_plane_development_v1 import URDF

NATIVE = {'base'} | {leg+'_'+part for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf')}


def test_actual_collapsed_roster_includes_foot_shapes_in_calf_group():
    shapes = ArticulatedCollisionGeometry(URDF).supports(np.zeros(12), np.eye(3))['shapes']
    result = resolve_native_groups(URDF, shapes, NATIVE)
    assert len(result) == 27
    assert result['Head_upper:0'] == result['Head_lower:0'] == 'base'
    for leg in ('FL','FR','RL','RR'):
        assert all(result[leg+'_'+p+':0'] == leg+'_calf' for p in ('calf','calflower','calflower1','foot'))


def test_explicitly_retained_foot_roster_is_respected():
    shapes = ArticulatedCollisionGeometry(URDF).supports(np.zeros(12), np.eye(3))['shapes']
    result = resolve_native_groups(URDF, shapes, NATIVE | {'FL_foot'})
    assert result['FL_foot:0'] == 'FL_foot' and result['FR_foot:0'] == 'FR_calf'


@pytest.mark.parametrize('roster', [NATIVE - {'FL_calf'}, NATIVE | {'box_baselink'}, NATIVE - {'base'}])
def test_missing_movable_or_nonrobot_roster_rejected(roster):
    shapes = ArticulatedCollisionGeometry(URDF).supports(np.zeros(12), np.eye(3))['shapes']
    with pytest.raises(ValueError): resolve_native_groups(URDF, shapes, roster)
